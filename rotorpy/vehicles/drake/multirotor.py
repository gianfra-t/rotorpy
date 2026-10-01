"""``DrakeMultirotor`` solves the dynamics using the Multibody physics engine 'Drake'."""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.math import SpatialVelocity
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import FixedOffsetFrame, RevoluteJoint, RotationalInertia, SpatialInertia
from pydrake.systems.analysis import ApplySimulatorConfig, Simulator, SimulatorConfig
from pydrake.systems.framework import DiagramBuilder, EventStatus
from scipy.spatial.transform import Rotation

from rotorpy.vehicles.drake.effectors import Effectors
from rotorpy.vehicles.models.actuator import limits
from rotorpy.vehicles.models.aero import AirframeState
from rotorpy.vehicles.models.spec import VehicleSpec, complete_state, hover_state
from rotorpy.vehicles.types import (
    AngularVelocity, InertiaTensor, Mass, PerAssembly, PerRotor, PerSurface, Position, Quaternion, Velocity,
)


class DrakeControl(TypedDict, total=False):
    """The control dict ``step`` and ``statedot`` accept, shared with the controller that produces it.

    One of ``cmd_motor_speeds`` (rad/s) / ``cmd_motor_thrusts`` (N) / ``cmd_motor_torques`` (N*m), per
    ``control_abstraction``; a NaN entry is a free rotor.  The rest are optional and default to NaN: ``cmd_tilt_torques`` (N*m, NaN = free hinge),
    ``cmd_tilt_rates`` (rad/s, NaN = not prescribed), ``cmd_surface_deflections`` (rad, NaN = hold, in
    ``spec.airframe.surfaces`` order, the same order the state's ``surface_deflections`` comes back in).
    """

    cmd_motor_speeds: PerRotor
    cmd_motor_thrusts: PerRotor
    cmd_motor_torques: PerRotor
    cmd_tilt_torques: PerAssembly
    cmd_tilt_rates: PerAssembly
    cmd_surface_deflections: PerSurface


class DrakeState(TypedDict):
    """The state dict ``step`` returns (and takes back): RotorPy's keys plus the tilt and surface actuators."""

    x: Position
    v: Velocity
    q: Quaternion
    w: AngularVelocity
    wind: Velocity
    rotor_speeds: PerRotor
    tilt_angles: PerAssembly
    tilt_rates: PerAssembly
    tilt_torques: PerAssembly
    surface_deflections: PerSurface


# State keys owned by the Drake solver (continuous state).
DYNAMIC_KEYS = tuple(key for key in DrakeState.__annotations__ if key != "wind")
CONTROL_ABSTRACTIONS = ("cmd_motor_speeds", "cmd_motor_thrusts", "cmd_motor_torques")


def _central_spatial_inertia(mass: Mass, com: Position, inertia: InertiaTensor) -> SpatialInertia:
    """Drake ``SpatialInertia`` from a mass, COM offset and *central* inertia."""

    if mass == 0.0:
        return SpatialInertia.Zero() 
    rotational = RotationalInertia(
        inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2])
    return SpatialInertia.MakeFromCentralInertia(mass, np.asarray(com, dtype=float), rotational)


def _rotation_with_z(axis):
    """Rotation ``R_PR`` from a rotor's own frame R to its parent frame P: a deterministic right-handed frame whose
    z-axis is the unit thrust ``axis``.  R has its origin at the hub; its x/y are arbitrary (the rotor is
    axisymmetric), so the rotor inertia in R is ``diag(I_t, I_t, J_p)``."""

    reference = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(axis, reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    x_axis = reference - axis * np.dot(reference, axis)
    x_axis /= np.linalg.norm(x_axis)
    return np.column_stack((x_axis, np.cross(axis, x_axis), axis))


class DrakeMultirotor:
    """Standalone torque-driven RotorPy vehicle integrated by Drake.

    The part models are pure functions of their inputs; Drake may call them repeatedly at Runge-Kutta stages.
    """

    def __init__(
        self,
        spec: VehicleSpec,
        initial_state=None,
        control_abstraction="cmd_motor_speeds",
        gravity=9.81,
        integrator="runge_kutta3", # todo: also on the types improvement: let it be an enum of allowed integrators on Drake's code
        accuracy=1e-8,
        max_step=None,
    ):
        self.spec = spec
        self.num_rotors = spec.num_rotors
        self.num_assemblies = spec.num_assemblies
        self.num_surfaces = len(spec.airframe.surfaces)
        self.mass = self.total_mass = spec.total_mass
        self.g = gravity
        self.weight = np.array([0.0, 0.0, -self.mass * self.g])
        self.control_abstraction = control_abstraction

        self.rotor_directions = spec.rotor_values("direction")
        self._jp = spec.rotor_values("spin_inertia")
        self.tilt_rate_limit = np.array([assembly.rate_limit for assembly in spec.assemblies])
        # The actuators, per command (plan §R.8): rotor speed, hinge torque and rate, surface deflection.
        self.speed_actuators = [rotor.speed for rotor in spec.rotors]
        self.torque_actuators = [assembly.torque for assembly in spec.assemblies]
        self.rate_actuators = [assembly.rate for assembly in spec.assemblies]
        self.deflection_actuators = [surface.deflection for surface in spec.airframe.surfaces]

        self.initial_state = complete_state(hover_state(spec) if initial_state is None else initial_state,
                                            self.num_rotors, self.num_assemblies)
        self.initial_state.setdefault("tilt_torques", np.zeros(self.num_assemblies))
        self.initial_state.setdefault("surface_deflections", np.zeros(self.num_surfaces))

        self._build(gravity)
        self._configure_simulator(integrator, accuracy, max_step)

    # ------ Multibody Dynamic Model --------
    def _build(self, gravity):
        spec = self.spec
        builder = DiagramBuilder()
        plant = builder.AddSystem(MultibodyPlant(time_step=0.0))
        plant.mutable_gravity_field().set_gravity_vector([0.0, 0.0, -gravity])

        # a massless coordinate frame rigidly attached to body, rotated and translated form the body's origin
        # by `rotation` and `translation`.
        def offset_frame(name, body, rotation, translation):
            return plant.AddFrame(FixedOffsetFrame(
                name, body.body_frame(), RigidTransform(RotationMatrix(rotation), translation)))

        self.airframe = plant.AddRigidBody("airframe", _central_spatial_inertia(
            spec.airframe.mass, spec.airframe.com_offset, spec.airframe.inertia))
        self.assembly_bodies, self.tilt_joints = [], []

        # Assemble all nacelles
        for a, assembly in enumerate(spec.assemblies):
            # Create the tilting nacelle body with it's inertia props.
            body = plant.AddRigidBody(f"assembly_{a}", _central_spatial_inertia(
                assembly.mass, assembly.com_offset, assembly.inertia))

            # Create the "hinge frame", attached to the parent (airframe) body, at the prescribed pivot location.
            r0 = assembly.zero_orientation
            parent = offset_frame(f"hinge_{a}", self.airframe, r0, assembly.pivot)

            # Connect the 2 of them, allowing for a torque applied about the hinge axis (whatever servo  used)
            joint = plant.AddJoint(RevoluteJoint(
                f"tilt_{a}", parent, body.body_frame(), r0.T @ assembly.hinge_axis))
            plant.AddJointActuator(f"tilt_motor_{a}", joint)

            self.assembly_bodies.append(body)
            self.tilt_joints.append(joint)

        # Assemble all rotors
        self.rotor_bodies, self.spin_joints = [], []
        for i, rotor in enumerate(spec.rotors):
            # if not specified, the rotor is attached directly to the airframe (like a prop).
            parent_body = self.airframe if rotor.assembly is None else self.assembly_bodies[rotor.assembly]
            # Rotor body frame R (see _rotation_with_z): origin at the hub, which
            # is the rotor COM, z along the thrust axis; rz = R_PR places it.
            rz = _rotation_with_z(rotor.thrust_axis)

            # Create the rotor body itself.
            own_inertia = np.diag([rotor.transverse_inertia, rotor.transverse_inertia, rotor.spin_inertia])
            body = plant.AddRigidBody(f"rotor_{i}", _central_spatial_inertia(rotor.mass, np.zeros(3), own_inertia))

            # Create the "hub frame", attached to the parent body at the rotor hub location.
            parent = offset_frame(f"hub_{i}", parent_body, rz, rotor.hub_offset)

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

        # Set initial lag states: tilt torques, then surface deflections
        if self.num_assemblies or self.num_surfaces:
            torques = np.asarray(state.get("tilt_torques", np.zeros(self.num_assemblies)), dtype=float)
            deflections = np.asarray(
                state.get("surface_deflections", np.zeros(self.num_surfaces)), dtype=float)
            self.effectors.GetMyMutableContextFromRoot(root).SetContinuousState(
                np.concatenate((torques, deflections)))

    def _get_state(self, wind) -> DrakeState:
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
            "tilt_torques": self.effectors.applied_tilt_torques(self.effectors.GetMyContextFromRoot(self.context)),
            "surface_deflections": self.effectors.surface_deflections(
                self.effectors.GetMyContextFromRoot(self.context)),
        }

    def airframe_state(self) -> AirframeState:
        """What the airframe's model is handed at the current state (for logging)."""

        return self.effectors.airframe_state(self.context)

    def _commands(self, state, control: DrakeControl):
        """The command port: each command clipped to its actuator's limits (NaN passes)."""

        k, s = self.num_assemblies, self.num_surfaces
        if self.control_abstraction == "cmd_motor_torques":
            rotors = np.asarray(control["cmd_motor_torques"], dtype=float)
        else:
            if self.control_abstraction == "cmd_motor_speeds":
                rotors = np.asarray(control["cmd_motor_speeds"], dtype=float)
            else:
                rotors = np.array([rotor.model.speed_for_thrust(thrust)
                                   for rotor, thrust in zip(self.spec.rotors, control["cmd_motor_thrusts"])])
            rotors = np.clip(rotors, *limits(self.speed_actuators))
        # NaN torque = zero torque = free hinge (or a rate-prescribed one).
        tilts = np.clip(np.nan_to_num(np.asarray(control.get("cmd_tilt_torques", np.zeros(k)), dtype=float)),
                        *limits(self.torque_actuators))
        rates = np.clip(np.asarray(control.get("cmd_tilt_rates", np.full(k, np.nan)), dtype=float),
                        *limits(self.rate_actuators))
        deflections = np.clip(np.asarray(control.get("cmd_surface_deflections", np.full(s, np.nan)), dtype=float),
                              *limits(self.deflection_actuators))
        return np.concatenate((tilts, rates, rotors, deflections, np.asarray(state["wind"], dtype=float)))

    def _load(self, state, control, root=None):
        root = self.context if root is None else root
        # A plain RotorPy state (no tilt_* keys, e.g. from a vanilla vehicle) gets zero tilt.
        state = complete_state(state, self.num_rotors, self.num_assemblies)
        self._set_state(state, root)
        self.diagram.get_input_port(0).FixValue(root, self._commands(state, control))


    def _continues(self, state):
        """True when `state` is exactly what the previous step() returned."""

        last = self._last_out
        return last is not None and all(
            key in state and np.array_equal(np.asarray(state[key]), last[key]) for key in DYNAMIC_KEYS)

    def step(self, state: DrakeState, control: DrakeControl, t_step) -> DrakeState:
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

    def statedot(self, state: DrakeState, control: DrakeControl, t_step):
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
