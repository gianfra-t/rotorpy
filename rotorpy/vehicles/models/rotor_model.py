"""The rotor-model contract: one rotor's loads from its own speed and inflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from rotorpy.vehicles.types import Velocity


@dataclass(frozen=True)
class RotorInflow:
    """What the vehicle hands a rotor model.

    * ``rotor_speed``: rad/s, signed; a negative speed spins the rotor backwards;
    * ``direction``: +1 or -1, the spin sense: the drag-torque couple on the parent is ``+direction * Q * axis``;
    * ``axis``: unit thrust axis;
    * ``v_axial``: hub airspeed (hub velocity minus wind) along the axis, m/s, positive when the hub moves along its
      thrust;
    * ``v_inplane``: the rest of the hub airspeed, in the disk plane, m/s.
    """

    rotor_speed: float
    direction: float
    axis: np.ndarray
    v_axial: float
    v_inplane: np.ndarray

    @classmethod
    def from_airspeed(cls, rotor_speed, direction, axis, hub_airspeed) -> RotorInflow:
        """Split the hub airspeed into its axial and in-plane parts."""

        v_axial = float(np.dot(hub_airspeed, axis))
        return cls(float(rotor_speed), float(direction), axis, v_axial, hub_airspeed - v_axial * axis)

    @property
    def hub_airspeed(self) -> Velocity:
        return self.v_axial * self.axis + self.v_inplane


@dataclass(frozen=True)
class RotorLoads:
    """What a rotor model returns.

    * ``thrust``: force along the thrust axis, N;
    * ``torque``: shaft torque ``Q``, N*m, positive when it resists the rotor's normal spin;
    * ``force``: the whole force on the rotor, at its hub (thrust plus any hub force), N;
    * ``moment``: the whole couple on the rotor (``direction * Q * axis`` plus any hub moment), N*m.

    ``force`` and ``moment`` are what the vehicle applies; ``thrust`` and ``torque`` are the scalars it reports and the
    airframe's model may read (``AirframeState.rotor_loads``).
    """

    thrust: float
    torque: float
    force: np.ndarray
    moment: np.ndarray


class RotorModel(Protocol):
    """The loads of one rotor in ``inflow``."""

    def __call__(self, inflow: RotorInflow) -> RotorLoads: ...
