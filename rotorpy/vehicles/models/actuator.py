"""The actuator contract: what a part lets a controller command, and how the commanded value becomes the actual one."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

import numpy as np


class ActuatorModel(Protocol):
    """``d value / dt`` of an actuator given its (clipped) command and its actual value."""

    def __call__(self, command: float, value: float) -> float: ...


@dataclass(frozen=True)
class FirstOrderLag:
    """``d value / dt = (command - value) / tau``, ``tau`` in s."""

    tau: float

    def __call__(self, command, value):
        return (command - value) / self.tau


@dataclass(frozen=True)
class Actuator:
    """A controllable variable: the command is clipped to ``limits`` and the actual value follows it through
    ``model``.  No model: the actual value is the command."""

    limits: Tuple[float, float] = (-np.inf, np.inf)
    model: Optional[ActuatorModel] = None


def limits(actuators):
    """``(low, high)`` arrays over ``actuators``, for ``np.clip``."""

    bounds = np.array([actuator.limits for actuator in actuators], dtype=float).reshape(-1, 2)
    return bounds[:, 0], bounds[:, 1]


def follow(actuators, commands, values):
    """``d value / dt`` of each actuator; 0 where the command is NaN (hold) or the actuator has no model."""

    return np.array([0.0 if actuator.model is None or np.isnan(command) else actuator.model(command, value)
                     for actuator, command, value in zip(actuators, commands, values)], dtype=float)
