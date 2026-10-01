"""RotorPy's frame drag as an airframe model (``Airframe.aero``)."""

from __future__ import annotations

import numpy as np

from rotorpy.vehicles.models.aero import AerodynamicWrench, AirframeState
from rotorpy.vehicles.models.powered_aero.splines import cross3
from rotorpy.vehicles.types import Position


class FrameDrag:
    """``F = -|v| diag(c_D) v`` at the airframe COM ``com`` (body FLU, from the airframe origin), ``v`` its airspeed;
    ``c_D`` is ``(c_Dx, c_Dy, c_Dz)`` in kg/m."""

    def __init__(self, c_D, com: Position):
        self.c_D, self.com = np.asarray(c_D, dtype=float), np.asarray(com, dtype=float)

    def __call__(self, state: AirframeState) -> AerodynamicWrench:
        velocity = state.airspeed + cross3(state.w, self.com)
        force = -np.linalg.norm(velocity) * self.c_D * velocity
        return AerodynamicWrench(force, cross3(self.com, force))
