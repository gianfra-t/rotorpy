"""``PoweredAeroModel``: an airframe's aerodynamics with its rotors running."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

from rotorpy.vehicles.models.aero import AerodynamicWrench, AirframeState
from rotorpy.vehicles.models.powered_aero.fixed_wing import FixedWingAero
from rotorpy.vehicles.models.powered_aero.interaction import InteractionAero
from rotorpy.vehicles.models.powered_aero.rotor import RotorWake, wake_state_vectors
from rotorpy.vehicles.models.powered_aero.types import PoweredAeroData

if TYPE_CHECKING:
    from rotorpy.vehicles.models.spec import ControlSurface, RotorSpec


class PoweredAeroModel:
    """The fixed-wing wrench of ``surfaces`` (the airframe's, in order) plus the interaction
    increments of ``rotors`` (the vehicle's, in order).  Each pair's source rotor is the rotor of that name."""

    def __init__(self, data: PoweredAeroData, rotors: Sequence[RotorSpec], surfaces: Sequence[ControlSurface]):
        self.data = data
        self.fixed_wing = FixedWingAero(data, surfaces)
        self.rotor_index = {rotor.name: index for index, rotor in enumerate(rotors)}
        self.interaction = InteractionAero(data, self.rotor_index)
        self._areas = np.array([rotor.area for rotor in rotors])
        # Each rotor's tilt is its assembly's; a rotor on the airframe (-1) does not tilt.
        self._assembly = np.array([-1 if rotor.assembly is None else rotor.assembly for rotor in rotors])

    def rotor_wake(self, state: AirframeState) -> RotorWake:
        """Each rotor's wake at the disk, from its own thrust."""

        axes = np.array([inflow.axis for inflow in state.rotors])
        hub_airspeeds = np.array([inflow.hub_airspeed for inflow in state.rotors])
        v_ax = np.array([inflow.v_axial for inflow in state.rotors])
        v_ip = np.array([np.linalg.norm(inflow.v_inplane) for inflow in state.rotors])
        thrust = np.array([loads.thrust for loads in state.rotor_loads])
        
        #todo also develop this part , sources
        v0, chi, wake_direction = wake_state_vectors(
            thrust, v_ax, v_ip, self.data.rho, self._areas, -hub_airspeeds, axes
        )
        return RotorWake(thrust=thrust, omega=np.array([inflow.rotor_speed for inflow in state.rotors]),
                         v_ax=v_ax, v_ip=v_ip, v0=v0, chi=chi, wake_direction=wake_direction, axes=axes,
                         tilt=np.append(state.tilt_angles, 0.0)[self._assembly])  # index -1: the appended 0

    def interaction_wrench(self, state: AirframeState) -> AerodynamicWrench:
        """The interaction increments alone."""

        return AerodynamicWrench(*self.interaction.wrench(state.airspeed, state.w, self.rotor_wake(state)))

    def __call__(self, state: AirframeState) -> AerodynamicWrench:
        return self.fixed_wing(state) + self.interaction_wrench(state)
