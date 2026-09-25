"""
``DrakeMultirotor`` solves the dynamics using the Multibody physics engine 'Drake'.

The aircraft is modeled as a multibody system (plant), such that supporting tilting
nacelles in arbitrary configurations is possible. Control commands, forces and moments (wrench)
are then supplied to the model on each iteration given the current state of the plant.

Rotors are also modeled as separate, spinning rigid bodies connected to their parent (nacelle or airframe)
via a revolute joint. This allows for gimballed configurations as well (future work).

Conventions:

* Airframe body frame = RotorPy body frame (FLU), world z-up.
* ``x``/``v`` are the position/velocity of the instantaneous total COM, ``q`` the airframe attitude ``[x, y, z, w]``,
  ``w`` the airframe body rates.
* Spin joint rate is ``-sigma * rotor_speed`` (``sigma = rotor_directions``): the sign for which the motor reaction on
  the airframe is ``+sigma * k_m * Omega*|Omega| * axis``.
* ``cmd_tilt_torques`` (N*m about each hinge axis) pass through a first-order ``torque_lag`` (s, default 0); the
  applied torque is carried in the state as ``tilt_torques``.
* NaN commands: a NaN tilt torque is a free hinge, a NaN rotor speed a free rotor (aero drag torque still acts).
"""

from __future__ import annotations

import numpy as np
from pydrake.common.value import AbstractValue
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.math import SpatialForce, SpatialVelocity
from pydrake.multibody.plant import ExternallyAppliedSpatialForce, MultibodyPlant
from pydrake.multibody.tree import FixedOffsetFrame, RevoluteJoint, RotationalInertia, SpatialInertia
from pydrake.systems.analysis import ApplySimulatorConfig, Simulator, SimulatorConfig
from pydrake.systems.framework import DiagramBuilder, EventStatus, LeafSystem
from scipy.spatial.transform import Rotation

from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.multirotor_params import complete_state, prepare_structured, rotor_aero_loads
from rotorpy.vehicles.types import InertiaTensor, Mass, Position

# State keys owned by the Drake solver (continuous state). 
DYNAMIC_KEYS = ("x", "v", "q", "w", "rotor_speeds", "tilt_angles", "tilt_rates", "tilt_torques")


def _central_spatial_inertia(mass: Mass, com: Position, inertia: InertiaTensor) -> SpatialInertia:
    """Drake ``SpatialInertia`` from a mass, COM offset and *central* inertia."""

    if mass == 0.0:
        return SpatialInertia.Zero() 
    rotational = RotationalInertia(
        inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2])
    return SpatialInertia.MakeFromCentralInertia(mass, np.asarray(com, dtype=float), rotational)


def _rotor_frame(spec):
    """Rotation from a rotor's own frame R to its parent frame P (``R_PR``), which 
    can be the nacelle of the airframe given our model.

    R has its origin at the hub and z along the spin (thrust) axis; its x/y
    are arbitrary (assumption of always axisymmetric rotor). 
     
    Positions ``hub_offset`` and ``thrust_axis`` are given in P, the rotor ``inertia`` in R, e.g.
    ``diag(I_t, I_t, I_p)`` with I_p the spin inertia.
    """

    axis = np.asarray(spec.get("thrust_axis", [0.0, 0.0, 1.0]), dtype=float)
    return _rotation_with_z(axis / np.linalg.norm(axis))


def _with_parent_frame_inertias(quad_params, rotor_specs):
    """Copy of ``quad_params`` with rotor inertias along the parent's axes.

    The shared parser keeps Multirotor contract (rotor inertia in P) and
    derives the mass properties and axisymmetry check from it:
    I_P = R_PR I_R R_PR^T.
    """

    if "rotors" not in quad_params:
        return quad_params
    rotors = []
    for spec in rotor_specs:
        if "inertia" in spec:
            r_pr = _rotor_frame(spec)
            spec = {**spec, "inertia": r_pr @ np.asarray(spec["inertia"], dtype=float) @ r_pr.T}
        rotors.append(spec)
    return {**quad_params, "rotors": rotors}


def _rotation_with_z(axis):
    """Deterministic right-handed frame whose z-axis is the unit ``axis``.
    Helper also used for rotor positioning when assembling the plant.
    """

    reference = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(axis, reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    x_axis = reference - axis * np.dot(reference, axis)
    x_axis /= np.linalg.norm(x_axis)
    return np.column_stack((x_axis, np.cross(axis, x_axis), axis))


class DrakeMultirotor(Multirotor):
    """Standalone torque-driven RotorPy vehicle integrated by Drake."""

    def __init__(
        self,
        quad_params,
        initial_state=None,
        control_abstraction="cmd_motor_speeds",
        aero=True,
        gravity=9.81,
        integrator="runge_kutta3", # todo: also on the types improvement: let it be an enum of allowed integrators on Drake's code
        accuracy=1e-8,
        max_step=None,
        enable_ground=False,
    ):
        if enable_ground:
            raise ValueError("DrakeMultirotor has no ground model")
        if control_abstraction not in {"cmd_motor_speeds", "cmd_motor_thrusts"}:
            raise ValueError("DrakeMultirotor supports cmd_motor_speeds and cmd_motor_thrusts")
        if "airframe" not in quad_params:
            raise ValueError("DrakeMultirotor requires the structured multirotor parameter form")
        rotor_specs = quad_params.get("rotors", [])
        rotor_specs = list(rotor_specs.values()) if hasattr(rotor_specs, "values") else list(rotor_specs)

        # Rotor inertia is given in the rotor's own frame (see _rotor_frame).
        self._rotor_inertias = [np.asarray(spec.get("inertia", np.zeros((3, 3))), dtype=float) for spec in rotor_specs]
        base_params, model = prepare_structured(_with_parent_frame_inertias(quad_params, rotor_specs), initial_state)
        self._model = model
        self.num_assemblies = model["assembly_masses"].size
        initial_state = complete_state(initial_state, model["rotor_count"], self.num_assemblies,
                                       model["initial_angles"], model["initial_rates"], quad_params)
        
        # Multirotor builds scalar rotor-drag terms from k_d/k_z/k_h; the
        # per-rotor arrays are restored below, as MultirotorExtended does.
        super().__init__({**base_params, "k_d": 0.0, "k_z": 0.0, "k_h": 0.0}, initial_state,
                         control_abstraction, aero=aero)

        # todo: it's also hardcoded, hard to keep params track of. Potential fix when introducing pydantic.
        for name in ("k_eta", "k_m", "k_d", "k_z", "k_h", "k_flap", "tau_m", "rotor_speed_min", "rotor_speed_max"):
            setattr(self, name, np.asarray(base_params[name], dtype=float).copy())

        self.g = gravity
        self.weight = np.array([0.0, 0.0, -self.mass * self.g])

        #todo: why not? what breaks? we should.
        if np.any(model["rate_limit"] == 0.0):
            raise ValueError("DrakeMultirotor does not support frozen (rate_limit == 0) tilt actuators")

        self.rotor_directions = model["rotor_directions"].copy()
        self.tilt_rate_limit = model["rate_limit"].copy()
        specs = quad_params.get("assemblies", [])
        specs = list(specs.values()) if hasattr(specs, "values") else list(specs)
        self.torque_lag = np.array([float(spec.get("torque_lag", 0.0)) for spec in specs])
        if not np.all(np.isfinite(self.torque_lag)) or np.any(self.torque_lag < 0.0):
            raise ValueError("torque_lag must be finite and nonnegative")
        self.initial_state = dict(self.initial_state)
        self.initial_state.setdefault("tilt_torques", np.zeros(self.num_assemblies))
        self.total_mass = model["total_mass"]

        self._jp = model["rotor_polar_inertia"].copy()
        if np.any(np.linalg.norm(model["rotor_inertias"], axis=(1, 2)) == 0.0):
            raise ValueError("DrakeMultirotor needs a full rotor inertia on every rotor (spin DOF)")
        if np.any(self._hinge_inertias() <= 0.0):
            raise ValueError("every tilt hinge needs a positive locked inertia (assembly + rotors)")

        self._build(gravity)
        self._configure_simulator(integrator, accuracy, max_step)

    # ------------------------------------------------------------------ model
    def _hinge_inertias(self):
        """Locked inertia of each assembly + its rotors about its hinge at delta=0."""

        m = self._model
        result = np.zeros(self.num_assemblies)
        for a in range(self.num_assemblies):
            axis, r0 = m["hinge_axes"][a], m["zero_orientations"][a]
            parts = [(m["assembly_masses"][a], m["assembly_com_offsets"][a], m["assembly_inertias"][a])]
            parts += [
                (m["rotor_masses"][i], m["rotor_hub_offsets"][i], m["rotor_inertias"][i])
                for i in np.flatnonzero(m["rotor_assembly"] == a)
            ]
            for mass, offset, inertia in parts:
                r = r0 @ offset
                about_pivot = r0 @ inertia @ r0.T + mass * (r @ r * np.eye(3) - np.outer(r, r))
                result[a] += axis @ about_pivot @ axis
        return result

    def _build(self, gravity):
        m = self._model
        builder = DiagramBuilder()
        plant = builder.AddSystem(MultibodyPlant(time_step=0.0))
        plant.mutable_gravity_field().set_gravity_vector([0.0, 0.0, -gravity])

        # a massless coordinate frame rigidly attached to body, rotated and translated form the body's origin
        # by `rotation` and `translation`.
        def offset_frame(name, body, rotation, translation):
            return plant.AddFrame(FixedOffsetFrame(
                name, body.body_frame(), RigidTransform(RotationMatrix(rotation), translation)))

        self.airframe = plant.AddRigidBody("airframe", _central_spatial_inertia(
            m["airframe_mass"], m["airframe_com"], m["airframe_inertia"]))
        self.assembly_bodies, self.tilt_joints = [], []

        # Assemble all nacelles
        for a in range(self.num_assemblies):
            # Create the tilting nacelle body with it's inertia props.
            body = plant.AddRigidBody(f"assembly_{a}", _central_spatial_inertia(
                m["assembly_masses"][a], m["assembly_com_offsets"][a], m["assembly_inertias"][a]))

            # Create the "hinge frame", attached to the parent (airframe) body, at the prescribed pivot location.
            r0 = m["zero_orientations"][a]
            parent = offset_frame(f"hinge_{a}", self.airframe, r0, m["pivots"][a])

            # Connect the 2 of them, allowing for a torque applied about the hinge axis (whatever servo  used)
            joint = plant.AddJoint(RevoluteJoint(
                f"tilt_{a}", parent, body.body_frame(), r0.T @ m["hinge_axes"][a]))
            plant.AddJointActuator(f"tilt_motor_{a}", joint)

            self.assembly_bodies.append(body)
            self.tilt_joints.append(joint)

        # Assemble all rotors
        self.rotor_bodies, self.spin_joints = [], []
        for i in range(self.num_rotors):
            a = m["rotor_assembly"][i]
            # if not specified, the rotor is attached directly to the airframe (like a prop).
            parent_body = self.airframe if a < 0 else self.assembly_bodies[a]
            # Rotor body frame R (see _rotor_frame): origin at the hub, which
            # is the rotor COM, z along the thrust axis; rz = R_PR places it.
            rz = _rotation_with_z(m["rotor_axes_parent"][i])

            # Create the rotor body itself.
            body = plant.AddRigidBody(f"rotor_{i}", _central_spatial_inertia(
                m["rotor_masses"][i], np.zeros(3), self._rotor_inertias[i]))

            # Create the "hub frame", attached to the parent body at the rotor hub location.
            parent = offset_frame(f"hub_{i}", parent_body, rz, m["rotor_hub_offsets"][i])

            # Connect the rotor body to the hub frame with a revolute joint allowing it to spin about its thrust axis.
            joint = plant.AddJoint(RevoluteJoint(f"spin_{i}", parent, body.body_frame(), [0.0, 0.0, 1.0]))
            plant.AddJointActuator(f"rotor_motor_{i}", joint)


            self.rotor_bodies.append(body)
            self.spin_joints.append(joint)
        plant.Finalize()

        self.plant = plant
        effectors = builder.AddSystem(Effectors(self))
        builder.Connect(plant.get_state_output_port(), effectors.get_input_port(0))
        builder.Connect(effectors.get_output_port(0), plant.get_actuation_input_port())
        builder.Connect(effectors.get_output_port(1), plant.get_applied_spatial_force_input_port())
        builder.ExportInput(effectors.get_input_port(1), "commands")
        self.effectors = effectors
        self.diagram = builder.Build()

    def _configure_simulator(self, integrator, accuracy, max_step):
        config = SimulatorConfig(integration_scheme=integrator, accuracy=accuracy)
        if max_step is not None:
            config.max_step_size = max_step
        self.simulator = Simulator(self.diagram)
        ApplySimulatorConfig(config, self.simulator)
        self.simulator.set_monitor(self._tilt_rate_monitor)
        self.context = self.simulator.get_mutable_context()
        self.plant_context = self.plant.GetMyMutableContextFromRoot(self.context)

        # statedot's own scratch context, so it never touches the simulator's.
        self._scratch_context = self.diagram.CreateDefaultContext()
        self._scratch_plant_context = self.plant.GetMyMutableContextFromRoot(self._scratch_context)
        self._last_out = None  # dynamic keys of the state step() last returned

    def _set_state(self, state, root=None):
        """Write a RotorPy state dict into the plant context (total-COM x/v)."""

        root = self.context if root is None else root
        plant, ctx = self.plant, self.plant.GetMyMutableContextFromRoot(root)
        q = np.asarray(state["q"], dtype=float)
        rotation = RotationMatrix(Rotation.from_quat(q / np.linalg.norm(q)).as_matrix())
        omega_w = rotation.matrix() @ np.asarray(state["w"], dtype=float)

        # setting initial tilt and rotor speeds
        for joint, angle, rate in zip(self.tilt_joints, state["tilt_angles"], state["tilt_rates"]):
            joint.set_angle(ctx, angle)
            joint.set_angular_rate(ctx, rate)
        for joint, sigma, speed in zip(self.spin_joints, self.rotor_directions, state["rotor_speeds"]):
            joint.set_angle(ctx, 0.0)  # axisymmetric rotors: azimuth is not a state
            joint.set_angular_rate(ctx, -sigma * speed)

        # Setting the airframe so the total COM lands on x and moves with v.
        # Drake computes the total COM (CalcCenterOfMassPositionInWorld) but has no setter for it: the free joint holds
        # the airframe origin. With the joints set above and the airframe at the world origin with zero linear velocity,
        # the computed COM and its velocity are the internal offset and internal velocity (w x r, tilt rates), so we
        # subtract them from the desired COM position and velocity to get the airframe origin's position and velocity.
        plant.SetFreeBodyPose(ctx, self.airframe, RigidTransform(rotation, np.zeros(3)))
        plant.SetFreeBodySpatialVelocity(ctx, self.airframe, SpatialVelocity(omega_w, np.zeros(3)))
        com = plant.CalcCenterOfMassPositionInWorld(ctx)
        com_velocity = plant.CalcCenterOfMassTranslationalVelocityInWorld(ctx)
        plant.SetFreeBodyPose(ctx, self.airframe, RigidTransform(rotation, np.asarray(state["x"], dtype=float) - com))
        plant.SetFreeBodySpatialVelocity(ctx, self.airframe, SpatialVelocity(
            omega_w, np.asarray(state["v"], dtype=float) - com_velocity))

        # Set initial torques of the tilt hinges
        if self.num_assemblies:
            torques = np.asarray(state.get("tilt_torques", np.zeros(self.num_assemblies)), dtype=float)
            self.effectors.GetMyMutableContextFromRoot(root).SetContinuousState(torques)

    def _get_state(self, wind):
        plant, ctx = self.plant, self.plant_context
        pose = plant.EvalBodyPoseInWorld(ctx, self.airframe)
        rotation = pose.rotation().matrix()
        omega_w = plant.EvalBodySpatialVelocityInWorld(ctx, self.airframe).rotational()
        return {
            "x": np.asarray(plant.CalcCenterOfMassPositionInWorld(ctx)).copy(),
            "v": np.asarray(plant.CalcCenterOfMassTranslationalVelocityInWorld(ctx)).copy(),
            "q": Rotation.from_matrix(rotation).as_quat(),
            "w": rotation.T @ omega_w,
            "wind": np.asarray(wind, dtype=float).copy(),
            "rotor_speeds": np.array([-s * j.get_angular_rate(ctx) for s, j in zip(self.rotor_directions, self.spin_joints)]),
            "tilt_angles": np.array([j.get_angle(ctx) for j in self.tilt_joints]),
            "tilt_rates": np.array([j.get_angular_rate(ctx) for j in self.tilt_joints]),
            "tilt_torques": self.effectors.tilt_torques(self.effectors.GetMyContextFromRoot(self.context)),
        }

    def _commands(self, state, control):
        if self.control_abstraction == "cmd_motor_speeds":
            speeds = np.asarray(control["cmd_motor_speeds"], dtype=float)
        elif self.control_abstraction == "cmd_motor_thrusts":
            thrusts = np.asarray(control["cmd_motor_thrusts"], dtype=float) / self.k_eta
            speeds = np.sign(thrusts) * np.sqrt(np.abs(thrusts))
        else:
            raise ValueError("DrakeMultirotor supports cmd_motor_speeds and cmd_motor_thrusts")
        speeds = np.clip(speeds, self.rotor_speed_min, self.rotor_speed_max)  # NaN passes through
        tilts = np.asarray(control.get("cmd_tilt_torques", np.zeros(self.num_assemblies)), dtype=float)
        if speeds.shape != (self.num_rotors,) or tilts.shape != (self.num_assemblies,):
            raise ValueError("commands must have one value per rotor / per assembly")
        tilts = np.nan_to_num(tilts, nan=0.0)  # NaN = zero torque = free hinge
        wind = np.asarray(state["wind"], dtype=float)
        return np.concatenate((tilts, speeds, wind))

    def _load(self, state, control, root=None):
        root = self.context if root is None else root
        # A plain RotorPy state (no tilt_* keys, e.g. from a vanilla vehicle)
        # gets the model's initial tilt
        state = complete_state(state, self.num_rotors, self.num_assemblies,
                               self._model["initial_angles"], self._model["initial_rates"], {})
        self._set_state(state, root)
        self.diagram.get_input_port(0).FixValue(root, self._commands(state, control))


    def _continues(self, state):
        """True when `state` is exactly what the previous step() returned."""

        last = self._last_out
        return last is not None and all(
            key in state and np.array_equal(np.asarray(state[key]), last[key]) for key in DYNAMIC_KEYS)

    def step(self, state, control, t_step):
        # Two paths: if the caller hands back the state we last returned
        # (only commands/wind may differ), keep advancing the same context with simulator.AdvanceTo(.)
        # If not, we re-seed the context of the plant in Drake.
        if self._continues(state):
            self.diagram.get_input_port(0).FixValue(self.context, self._commands(state, control))
            self._violation = None
            target = self.context.get_time() + t_step
        else:
            self.context.SetTime(0.0)
            self._load(state, control)
            self._violation = None
            self.simulator.Initialize()
            target = t_step
        try:
            self.simulator.AdvanceTo(target)

        except RuntimeError as error:
            self._last_out = None
            if self._violation is None:
                raise
            raise RuntimeError(self._violation) from error
        out = self._get_state(state["wind"])
        self._last_out = {key: np.array(out[key], copy=True) for key in DYNAMIC_KEYS}
        return out

    def _tilt_rate_monitor(self, root_context):
        """Fail-loud tilt-rate guard, checked after every integrator step."""

        ctx = self.plant.GetMyContextFromRoot(root_context)
        self._violation = None
        rates = np.array([j.get_angular_rate(ctx) for j in self.tilt_joints])
        over = np.flatnonzero(np.abs(rates) > self.tilt_rate_limit)
        if over.size == 0:
            return EventStatus.Succeeded()
        a = int(over[0])
        self._violation = (
            f"tilt hinge {a} rate {rates[a]:.6g} rad/s exceeds "
            f"rate_limit {self.tilt_rate_limit[a]:.6g} rad/s at t={root_context.get_time():.6g} s"
        )
        return EventStatus.Failed(self.diagram, self._violation)

    def statedot(self, state, control, t_step):
        del t_step
        # When `state` is what step() last returned, the live context already
        # holds it: fix the new commands and evaluate there.
        # Any other state is loaded into the scratch context.
        if self._continues(state):
            self.diagram.get_input_port(0).FixValue(self.context, self._commands(state, control))
            plant, ctx = self.plant, self.plant_context
        else:
            self._load(state, control, self._scratch_context)
            plant, ctx = self.plant, self._scratch_plant_context
        vdot = plant.get_generalized_acceleration_output_port().Eval(ctx)
        start = self.airframe.floating_velocities_start_in_v()
        rotation = plant.EvalBodyPoseInWorld(ctx, self.airframe).rotation().matrix()
        return {
            "vdot": np.asarray(plant.CalcCenterOfMassTranslationalAccelerationInWorld(ctx)).copy(),
            "wdot": rotation.T @ vdot[start : start + 3],  # d/dt(R^T w_W) = R^T alpha_W
        }


# Use LeafSystem to apply any sort of aerodynamic, propulsive, etc. external force into the plant
# as the simulation advances.
class Effectors(LeafSystem):
    """Tilt torque lag, rotor motors and aerodynamics as continuous feedback.

    Inputs: plant state; commands = [tilt torque cmds (K), rotor speed
    cmds (N), wind (3)].  Continuous state: applied tilt torques (K).
    Outputs: joint actuation (tilt then spin, in actuator order) and the
    aerodynamic ExternallyAppliedSpatialForce list.
    """

    def __init__(self, vehicle):
        super().__init__()
        self._v = vehicle
        plant = vehicle.plant
        k, n, m = vehicle.num_assemblies, vehicle.num_rotors, vehicle._model
        self._plant_context = plant.CreateDefaultContext()
        self.DeclareVectorInputPort("state", plant.num_multibody_states())
        self.DeclareVectorInputPort("commands", k + n + 3)
        self.DeclareVectorOutputPort("actuation", plant.num_actuated_dofs(), self._calc_actuation)
        self.DeclareAbstractOutputPort(
            "spatial_forces",
            lambda: AbstractValue.Make([ExternallyAppliedSpatialForce()]),
            self._calc_forces,
        )
        self._coeffs = (vehicle.k_eta, vehicle.k_m, vehicle.k_d, vehicle.k_z, vehicle.k_h, vehicle.k_flap,
                        vehicle.rotor_directions, vehicle.drag_matrix, vehicle.aero)
        self._k_m = vehicle.k_m
        self._spin_rows =np.array([joint.velocity_start() for joint in vehicle.spin_joints], dtype=int)
        self._actuation_matrix = plant.MakeActuationMatrix()
        self._lagged = vehicle.torque_lag > 0.0
        self._lag = np.where(self._lagged, vehicle.torque_lag, 1.0)
        if k:
            self.DeclareContinuousState(k)
        self._com_offset = np.asarray(m["airframe_com"], dtype=float)

    def tilt_torques(self, context):
        """Applied hinge torques: lag state, or the command when torque_lag == 0."""

        k = self._v.num_assemblies
        if k == 0:
            return np.zeros(0)
        command = self.get_input_port(1).Eval(context)[:k]
        return np.where(self._lagged, context.get_continuous_state_vector().CopyToVector(), command)

    def DoCalcTimeDerivatives(self, context, derivatives):
        k = self._v.num_assemblies
        if k == 0:
            return
        command = self.get_input_port(1).Eval(context)[:k]
        applied = context.get_continuous_state_vector().CopyToVector()
        derivatives.get_mutable_vector().SetFromVector(np.where(self._lagged, (command - applied) / self._lag, 0.0))

    def _sync(self, context):
        self._v.plant.SetPositionsAndVelocities(self._plant_context, self.get_input_port(0).Eval(context))
        return self._plant_context

    def _aero(self, context):
        """Aero loads (body components) plus the airframe rotation."""

        v, plant = self._v, self._v.plant
        ctx = self._sync(context)
        wind = self.get_input_port(1).Eval(context)[-3:]
        pose = plant.EvalBodyPoseInWorld(ctx, v.airframe)
        r_wb = pose.rotation().matrix()
        axes, hub_air = [], []

        for body in v.rotor_bodies:
            axes.append(r_wb.T @ plant.EvalBodyPoseInWorld(ctx, body).rotation().matrix()[:, 2])
            hub_air.append(r_wb.T @ (plant.EvalBodySpatialVelocityInWorld(ctx, body).translational() - wind))

        v_air = plant.EvalBodySpatialVelocityInWorld(ctx, v.airframe).Shift(
            r_wb @ self._com_offset).translational()
        
        speeds = self._rotor_speeds(ctx)
        loads = rotor_aero_loads(speeds, np.array(axes), np.array(hub_air), r_wb.T @ (v_air - wind), *self._coeffs)
        return loads, r_wb, speeds

    def _rotor_speeds(self, ctx):
        return np.array([-s * j.get_angular_rate(ctx) for s, j in zip(self._v.rotor_directions, self._v.spin_joints)])

    def _calc_actuation(self, context, output):
        """Tilt torques, then the spin torques that prescribe each rotor's speed.

        The rotor speed is prescribed, as in the reference models: Omega (relative to the rotor's parent) follows
        the motor lag exactly, and the spin torque is the constraint torque that makes it so.  Here we are modelling 
        the motor as a rotation joint where a torque must be provided. This is more realistic, but breaks the interface
        of prescribing rotor speeds directly. 

        That's why we need to infer the exact torque that would be required at each spin joint to achieve the commanded
        rotor speed. Acceleration is affine in the actuation, vdot(u0 + du) = vdot(u0) + M^-1 B du, so the spin torque
        correction solves  S M^-1 B_s du = thetaddot_lag - S vdot(u0)  (S picks the spin rows).
        """

        v = self._v
        ctx = self._sync(context)
        cmd = self.get_input_port(1).Eval(context)
        k, n = v.num_assemblies, v.num_rotors
        speed_cmd = cmd[k : k + n]
        tau_tilt = self.tilt_torques(context)
        speed, sigma = self._rotor_speeds(ctx), v.rotor_directions
        target = -sigma * (speed_cmd - speed) / v.tau_m

        # Predictor: the isolated-rotor torque (lag inertia torque + aero drag couple).  Exact up to the parent's
        # axial acceleration.
        tau_spin = np.where(np.isnan(speed_cmd), 0.0, -sigma * (v._jp * (speed_cmd - speed) / v.tau_m
                                                                 + self._k_m * speed * np.abs(speed)))
        actuation = np.concatenate((tau_tilt, tau_spin))
        accel = self._acceleration(ctx, actuation, self._spatial_forces(context))
        prescribed = np.flatnonzero(~np.isnan(speed_cmd))
        if prescribed.size:
            rows = self._spin_rows[prescribed]
            mobility = np.linalg.solve(v.plant.CalcMassMatrix(ctx), self._actuation_matrix[:, k + prescribed])
            actuation[k + prescribed] += np.linalg.solve(mobility[rows], target[prescribed] - accel[rows])
        output.SetFromVector(actuation)

    def _acceleration(self, ctx, actuation, forces):
        """Generalized acceleration of the (synced) plant context under the given inputs."""

        plant = self._v.plant
        plant.get_actuation_input_port().FixValue(ctx, actuation)
        plant.get_applied_spatial_force_input_port().FixValue(ctx, AbstractValue.Make(forces))
        return plant.get_generalized_acceleration_output_port().Eval(ctx)

    def _spatial_forces(self, context):
        v = self._v
        loads, r_wb, _ = self._aero(context)
        forces = []
        for i, body in enumerate(v.rotor_bodies):
            moment = loads["reaction_moments"][i] + loads["flap_moments"][i]
            forces.append(self._applied(body, np.zeros(3), r_wb @ moment, r_wb @ loads["forces"][i]))
        forces.append(self._applied(v.airframe, self._com_offset, np.zeros(3), r_wb @ loads["drag"]))
        return forces

    def _calc_forces(self, context, output):
        output.set_value(self._spatial_forces(context))

    def _applied(self, body, point_body, torque_w, force_w):
        applied = ExternallyAppliedSpatialForce()
        applied.body_index = body.index()
        applied.p_BoBq_B = point_body
        applied.F_Bq_W = SpatialForce(torque_w, force_w)
        return applied
