"""The airframe-aero contract: the airframe's wrench from its flight state.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Generic, Protocol, Tuple, TypeVar

import numpy as np

from rotorpy.vehicles.models.powered_aero.splines import smooth_norm
from rotorpy.vehicles.models.rotor_model import RotorInflow, RotorLoads
from rotorpy.vehicles.types import AngularVelocity, PerAssembly, PerSurface, Position, Velocity


@dataclass(frozen=True)
class AirframeState:
    """What the vehicle hands the airframe's model, fresh at every integrator stage.

    * ``position``: the airframe origin, world (z up), m;
    * ``w``: airframe body rates, rad/s;
    * ``airspeed``: the airframe origin's velocity minus the wind, m/s;
    * ``tilt_angles``: each assembly's hinge angle, rad;
    * ``surface_deflections``: actual deflections, rad, in ``Airframe.surfaces`` order;
    * ``rotors``, ``rotor_loads``: each rotor's inflow and its model's loads, in ``VehicleSpec.rotors`` order.
    """

    position: Position
    w: AngularVelocity
    airspeed: Velocity
    tilt_angles: PerAssembly
    surface_deflections: PerSurface
    rotors: Tuple[RotorInflow, ...] = ()
    rotor_loads: Tuple[RotorLoads, ...] = ()


@dataclass(frozen=True)
class AerodynamicWrench:
    """Force (N) and moment about the airframe origin (N*m) on the airframe, body FLU.  Wrenches add with ``+``."""

    force: np.ndarray
    moment: np.ndarray

    def __add__(self, other: AerodynamicWrench) -> AerodynamicWrench:
        return AerodynamicWrench(self.force + other.force, self.moment + other.moment)


class AirframeAeroModel(Protocol):
    """The airframe's wrench in ``state``."""

    def __call__(self, state: AirframeState) -> AerodynamicWrench: ...


T = TypeVar("T")


@dataclass(frozen=True)
class AeroCoefficients(Generic[T]):
    """The six whole-aircraft coefficients, FRD: lift ``CL``, drag ``CD`` and side force ``CY`` in wind axes, roll
    ``Cl``, pitch ``Cm`` and yaw ``Cn`` in body axes.  ``AeroCoefficients[float]`` is one flow condition;
    ``AeroCoefficients[np.ndarray]`` is one table per coefficient.  Increments add with ``+``."""

    CL: T
    CD: T
    CY: T
    Cl: T
    Cm: T
    Cn: T

    def __add__(self, other: AeroCoefficients[T]) -> AeroCoefficients[T]:
        return AeroCoefficients(*(getattr(self, name) + getattr(other, name) for name in COEFFICIENT_NAMES))


@dataclass(frozen=True)
class RateDerivatives:
    """Body-rate damping derivatives, FRD, per nondimensional rate ``p b / 2V``, ``q cbar / 2V``, ``r b / 2V``."""

    CYp: float
    CYr: float
    CLq: float
    Clp: float
    Clr: float
    Cmq: float
    Cnp: float
    Cnr: float


COEFFICIENT_NAMES = tuple(field.name for field in fields(AeroCoefficients))


@dataclass(frozen=True)
class AeroReference:
    """What the coefficients are normalised on and taken about: area ``S`` (m^2), span ``b`` and mean chord
    ``cbar`` (m), the moment reference point ``mrp_xyz`` (body FLU, from the airframe origin)."""

    area: float
    span: float
    cbar: float
    mrp_xyz: Position


FRD_FROM_FLU = np.diag([1.0, -1.0, -1.0])
SPEED_EPSILON_MPS = 1e-12


def body_from_wind(alpha, beta):
    """``C_BW`` mapping ``[-CD, CY, -CL]`` in wind axes to body FRD axes."""

    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    return np.array(
        [
            [ca * cb, -ca * sb, -sa],
            [sb, cb, 0.0],
            [sa * cb, -sa * sb, ca],
        ]
    )


def flow_angles(velocity_flu):
    """``(speed, alpha, beta)`` of the freestream with airspeed ``velocity_flu`` (body FLU; rad, FRD convention)."""

    velocity_flu = np.asarray(velocity_flu, dtype=float)
    speed = smooth_norm(velocity_flu, SPEED_EPSILON_MPS)
    u, v, w = FRD_FROM_FLU @ velocity_flu
    alpha = np.arctan2(w, u)
    beta = np.arctan2(v, np.sqrt(u * u + w * w + SPEED_EPSILON_MPS**2))
    return speed, alpha, beta
