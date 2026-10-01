"""``Effectors``: the Drake ``LeafSystem`` that feeds a ``DrakeMultirotor`` plant its joint actuation and external loads."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydrake.common.value import AbstractValue
from pydrake.multibody.math import SpatialForce
from pydrake.multibody.plant import ExternallyAppliedSpatialForce
from pydrake.systems.framework import LeafSystem, ValueProducer

from rotorpy.vehicles.models.actuator import follow
from rotorpy.vehicles.models.aero import AirframeState
from rotorpy.vehicles.models.rotor_model import RotorInflow

if TYPE_CHECKING:
    from rotorpy.vehicles.drake.multirotor import DrakeMultirotor


# Use LeafSystem to apply any sort of aerodynamic, propulsive, etc. external force into the plant Drake's model
# as the simulation advances.
class Effectors(LeafSystem):
    """Tilt torque lag, surface deflection lag, rotor motors and each body's model.

    Inputs: plant state; commands = [tilt torque cmds (K), tilt rate cmds
    (K, NaN = not prescribed), rotor speed or motor torque cmds (N, per
    ``control_abstraction``), surface deflection cmds
    (S, rad, NaN = hold), wind (3)].  Continuous state: lagged tilt torques
    (K) followed by lagged surface deflections (S).
    Outputs: joint actuation (tilt then spin, in actuator order) and the
    ExternallyAppliedSpatialForce list.

    It holds no model of its own.  It evaluates each rotor's ``RotorSpec.model`` on its inflow and applies the loads
    at the hub, then the airframe's ``Airframe.aero`` on the :class:`AirframeState` (which carries those rotor loads)
    and applies the wrench at the airframe origin.  The lag states follow the actuators' models
    (``assembly.torque``, ``surface.deflection``).
    """

    def __init__(self, vehicle: DrakeMultirotor):
        super().__init__()
        self.vehicle = vehicle
        plant = vehicle.plant
        k, n, s = vehicle.num_assemblies, vehicle.num_rotors, vehicle.num_surfaces
        self._plant_context = plant.CreateDefaultContext()

        # Connect Drake's system with our defined force, moments functions.
        self.DeclareVectorInputPort("state", plant.num_multibody_states())
        self.DeclareVectorInputPort("commands", 2 * k + n + s + 3)
        self.DeclareVectorOutputPort("actuation", plant.num_actuated_dofs(), self._calc_actuation)
        self.DeclareAbstractOutputPort(
            "spatial_forces",
            lambda: AbstractValue.Make([ExternallyAppliedSpatialForce()]),
            self._calc_forces,
        )
        # The models run once per context, shared by both outputs: the prescribed-speed solve needs the same loads the
        # plant then applies.  Default prerequisites (all sources): recomputed whenever the state, the commands or the
        # wind change.
        self._stage = self.DeclareCacheEntry(
            description="airframe state and applied spatial forces",
            value_producer=ValueProducer(allocate=AbstractValue.Make(None).Clone, calc=self._calc_stage),
        )

        self._tilt_rows = np.array([joint.velocity_start() for joint in vehicle.tilt_joints], dtype=int)
        self._spin_rows = np.array([joint.velocity_start() for joint in vehicle.spin_joints], dtype=int)
        self._actuation_matrix = plant.MakeActuationMatrix()
        self._lagged = np.array([actuator.model is not None for actuator in vehicle.torque_actuators], dtype=bool)
        if k + s:
            self.DeclareContinuousState(k + s)

    def tilt_torques(self, context):
        """Applied hinge torques: the lag state, or the command for a hinge torque without a model."""

        k = self.vehicle.num_assemblies
        if k == 0:
            return np.zeros(0)
        command = self.get_input_port(1).Eval(context)[:k]
        applied = context.get_continuous_state_vector().CopyToVector()[:k]
        return np.where(self._lagged, applied, command)

    def surface_deflections(self, context):
        """Actual surface deflections (rad), the lag states after the tilt torques."""

        s = self.vehicle.num_surfaces
        if s == 0:
            return np.zeros(0)
        k = self.vehicle.num_assemblies
        return context.get_continuous_state_vector().CopyToVector()[k : k + s]

    def applied_tilt_torques(self, context):
        """Hinge torques actually applied: the solved ones on rate-prescribed hinges, tilt_torques() elsewhere."""

        return np.asarray(self.get_output_port(0).Eval(context))[: self.vehicle.num_assemblies].copy()

    def DoCalcTimeDerivatives(self, context, derivatives):
        k, n, s = self.vehicle.num_assemblies, self.vehicle.num_rotors, self.vehicle.num_surfaces
        if k + s == 0:
            return
        cmd = self.get_input_port(1).Eval(context)
        applied = context.get_continuous_state_vector().CopyToVector()
        rate = np.zeros(k + s)
        if k:
            command, prescribed = cmd[:k], ~np.isnan(cmd[k : 2 * k])
            if prescribed.any():
                # The lag state of a rate-prescribed hinge follows the solved torque, so a switch back to a torque
                # command starts from the torque the hinge was carrying.
                command = np.where(prescribed, self.applied_tilt_torques(context), command)
            rate[:k] = follow(self.vehicle.torque_actuators, command, applied[:k])
        if s:
            # A NaN command holds the current deflection; commands arrive already clipped.
            rate[k:] = follow(self.vehicle.deflection_actuators, cmd[2 * k + n : 2 * k + n + s], applied[k:])
        derivatives.get_mutable_vector().SetFromVector(rate)

    def _sync(self, context):
        self.vehicle.plant.SetPositionsAndVelocities(self._plant_context, self.get_input_port(0).Eval(context))
        return self._plant_context

    def airframe_state(self, root_context) -> AirframeState:
        """What the airframe's model was handed at a root context (for logging)."""

        return self._stage.Eval(self.GetMyContextFromRoot(root_context))[0]

    def _calc_actuation(self, context, output):
        """Tilt torques, then the spin torques: commanded, or solved to prescribe each rotor's speed.

        A hinge with a tilt-rate command is prescribed the same way: its rate follows the command through its
        ``rate`` actuator and its torque is solved for.  Tilt and spin prescriptions are coupled through the
        mass matrix, so they are solved together.

        The rotor speed is prescribed, as in the reference models: Omega (relative to the rotor's parent) follows
        its ``speed`` actuator exactly, and the spin torque is the constraint torque that makes it so.  Here we are modelling 
        the motor as a rotation joint where a torque must be provided. This is more realistic, but breaks the interface
        of prescribing rotor speeds directly.

        With ``control_abstraction="cmd_motor_torques"`` the spin torque is the command itself and nothing is solved
        for the rotors.  Otherwise, that's why we need to infer the exact torque that would be required at each spin
        joint to achieve the commanded rotor speed. Acceleration is affine in the actuation, vdot(u0 + du) = vdot(u0) + M^-1 B du, so the spin torque
        correction solves  S M^-1 B_s du = thetaddot_lag - S vdot(u0)  (S picks the prescribed rows).
        """

        ctx = self._sync(context)
        cmd = self.get_input_port(1).Eval(context)
        k, n = self.vehicle.num_assemblies, self.vehicle.num_rotors
        rate_cmd, rotor_cmd = cmd[k : 2 * k], cmd[2 * k : 2 * k + n]
        tau_tilt = np.where(np.isnan(rate_cmd), self.tilt_torques(context), 0.0)
        sigma = self.vehicle.rotor_directions
        speed = -sigma * np.array([joint.get_angular_rate(ctx) for joint in self.vehicle.spin_joints])
        if self.vehicle.control_abstraction == "cmd_motor_torques":
            # The command is the motor torque itself, positive driving the rotor's normal spin: no spin row is
            # prescribed, and Omega follows from the torque balance with the rotor model's shaft torque.
            speed_cmd = np.full(n, np.nan)
            tau_spin = -sigma * np.nan_to_num(rotor_cmd, nan=0.0)
        else:
            # Predictor: the isolated-rotor torque (lag inertia torque + the rotor model's shaft torque).  Exact up
            # to the parent's axial acceleration.
            speed_cmd = rotor_cmd
            shaft_torque = np.array([loads.torque for loads in self._stage.Eval(context)[0].rotor_loads])
            speed_rate = follow(self.vehicle.speed_actuators, speed_cmd, speed)
            tau_spin = np.where(np.isnan(speed_cmd), 0.0, -sigma * (self.vehicle._jp * speed_rate + shaft_torque))
        actuation = np.concatenate((tau_tilt, tau_spin))
        prescribed = np.flatnonzero(~np.isnan(np.concatenate((rate_cmd, speed_cmd))))
        if prescribed.size:
            rate = np.array([joint.get_angular_rate(ctx) for joint in self.vehicle.tilt_joints])
            # Target joint accelerations, in actuator order (tilt, then spin); NaN where not prescribed.
            target = np.concatenate((follow(self.vehicle.rate_actuators, rate_cmd, rate),
                                     -sigma * follow(self.vehicle.speed_actuators, speed_cmd, speed)))
            accel = self._acceleration(ctx, actuation, self._stage.Eval(context)[1])
            rows = np.concatenate((self._tilt_rows, self._spin_rows))[prescribed]
            mobility = np.linalg.solve(self.vehicle.plant.CalcMassMatrix(ctx), self._actuation_matrix[:, prescribed])
            actuation[prescribed] += np.linalg.solve(mobility[rows], target[prescribed] - accel[rows])
        output.SetFromVector(actuation)

    def _acceleration(self, ctx, actuation, forces):
        """Generalized acceleration of the (synced) plant context under the given inputs."""

        plant = self.vehicle.plant
        plant.get_actuation_input_port().FixValue(ctx, actuation)
        plant.get_applied_spatial_force_input_port().FixValue(ctx, AbstractValue.Make(forces))
        return plant.get_generalized_acceleration_output_port().Eval(ctx)

    def _calc_stage(self, context, value):
        """The ``_stage`` cache entry: each rotor's model, then the airframe's, and the loads they put on the plant:
        ``(AirframeState, [ExternallyAppliedSpatialForce])``."""

        vehicle, plant = self.vehicle, self.vehicle.plant
        ctx = self._sync(context)
        wind = self.get_input_port(1).Eval(context)[-3:]
        pose = plant.EvalBodyPoseInWorld(ctx, vehicle.airframe)
        velocity = plant.EvalBodySpatialVelocityInWorld(ctx, vehicle.airframe)
        r_wb = pose.rotation().matrix()
        inflows, loads, forces = [], [], []
        for rotor, body, joint in zip(vehicle.spec.rotors, vehicle.rotor_bodies, vehicle.spin_joints):
            axis = r_wb.T @ plant.EvalBodyPoseInWorld(ctx, body).rotation().matrix()[:, 2]
            hub_airspeed = r_wb.T @ (plant.EvalBodySpatialVelocityInWorld(ctx, body).translational() - wind)
            inflow = RotorInflow.from_airspeed(-rotor.direction * joint.get_angular_rate(ctx), rotor.direction, axis,
                                               hub_airspeed)
            rotor_loads = rotor.model(inflow)
            inflows.append(inflow)
            loads.append(rotor_loads)
            forces.append(self._applied(body, r_wb @ rotor_loads.moment, r_wb @ rotor_loads.force))
        state = AirframeState(
            position=np.array(pose.translation()),
            w=r_wb.T @ velocity.rotational(),
            airspeed=r_wb.T @ (velocity.translational() - wind),
            tilt_angles=np.array([joint.get_angle(ctx) for joint in vehicle.tilt_joints]),
            surface_deflections=self.surface_deflections(context),
            rotors=tuple(inflows),
            rotor_loads=tuple(loads),
        )
        if vehicle.spec.airframe.aero is not None:
            wrench = vehicle.spec.airframe.aero(state)
            forces.append(self._applied(vehicle.airframe, r_wb @ wrench.moment, r_wb @ wrench.force))
        value.set_value((state, forces))

    def _calc_forces(self, context, output):
        output.set_value(self._stage.Eval(context)[1])

    def _applied(self, body, torque_w, force_w):
        """A world-frame wrench on ``body``, at its origin (a rotor's hub, the airframe origin)."""

        applied = ExternallyAppliedSpatialForce()
        applied.body_index = body.index()
        applied.F_Bq_W = SpatialForce(torque_w, force_w)
        return applied
