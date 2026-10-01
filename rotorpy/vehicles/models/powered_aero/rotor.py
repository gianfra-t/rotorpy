"""The powered-aero rotor: the rotor polynomial, its fit, and the wake."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Dict, Tuple

import numpy as np

from rotorpy.vehicles.models.rotor_model import RotorInflow, RotorLoads
from rotorpy.vehicles.types import PerRotor, PerRotorVector
from rotorpy.vehicles.models.powered_aero.splines import cross3, softplus, smooth_max

if TYPE_CHECKING:
    from rotorpy.vehicles.models.powered_aero.types import PoweredAeroData


def _coefficient(rotorpy=None, fitted=True):
    """A polynomial coefficient, 0 unless given.  ``rotorpy`` names the same term in RotorPy's ``k_*`` params;
    """

    return field(default=0.0, metadata={"rotorpy": rotorpy, "fitted": fitted})


@dataclass(frozen=True, kw_only=True)
class PolynomialRotorModel:
    """One rotor's dimensional polynomial, one coefficient per term (0 unless given):

    #todo explain this formulation and fit tests/error measurements.
    """

    a0: float = _coefficient("k_eta")
    a1: float = _coefficient()
    a2: float = _coefficient()
    a3: float = _coefficient("k_h")
    b0: float = _coefficient("k_m")
    b1: float = _coefficient()
    b2: float = _coefficient()
    b3: float = _coefficient()
    h0: float = _coefficient("k_d")
    h1: float = _coefficient()
    z0: float = _coefficient("k_z", fitted=False)
    m0: float = _coefficient("k_flap")
    m1: float = _coefficient()
    s0: float = _coefficient()
    s1: float = _coefficient()
    p0: float = _coefficient()
    p1: float = _coefficient()

    @classmethod
    def rotorpy(cls, *, k_eta, k_m, k_d=0.0, k_z=0.0, k_h=0.0, k_flap=0.0) -> PolynomialRotorModel:
        """RotorPy's native rotor (``Multirotor``): ``T = k_eta W|W| + k_h |Vip|^2``, ``Q = k_m W|W|``,
        hub force ``-W (k_d Vip + k_z Vax n)``, hub moment ``-k_flap W (Vip x n)``.  Good at low speed; it has no
        axial-inflow thrust loss.
        """

        values = {"k_eta": k_eta, "k_m": k_m, "k_d": k_d, "k_z": k_z, "k_h": k_h, "k_flap": k_flap}
        return cls(**{name: values[alias] for name, alias in ROTORPY_NAMES.items()})

    @classmethod
    def from_data(cls, data: PoweredAeroData, name: str) -> PolynomialRotorModel:
        """The powered-aero data file's fit of rotor ``name``"""

        fit = data.rotor(name).model
        return cls(**{coefficient.name: getattr(fit, coefficient.name) for coefficient in fields(cls)})

    def static(self) -> PolynomialRotorModel:
        """Only the static thrust and torque terms ``a0``, ``b0``: RotorPy's rotor with ``aero=False``."""

        return PolynomialRotorModel(a0=self.a0, b0=self.b0)

    def speed_for_thrust(self, thrust):
        """The speed that makes ``thrust`` with no inflow: ``T = a0 W|W|`` inverted (``cmd_motor_thrusts``)."""

        ratio = thrust / self.a0
        return float(np.sign(ratio) * np.sqrt(np.abs(ratio)))

    def __call__(self, inflow: RotorInflow) -> RotorLoads:
        w, v_ax, v_ip, axis, d = inflow.rotor_speed, inflow.v_axial, inflow.v_inplane, inflow.axis, inflow.direction
        # Omega*|Omega| (== Omega**2 for Omega >= 0): a reversed rotor brakes.
        signed_square = w * abs(w)
        v_ip2 = float(v_ip @ v_ip)
        v_ip_x_n = cross3(v_ip, axis)
        thrust = self.a0 * signed_square + self.a1 * w * v_ax + self.a2 * v_ax**2 + self.a3 * v_ip2
        torque = self.b0 * signed_square + self.b1 * w * v_ax + self.b2 * v_ax**2 + self.b3 * v_ip2
        hub_force = (-(self.h0 * w + self.h1 * v_ax) * v_ip - d * (self.s0 * w + self.s1 * v_ax) * v_ip_x_n
                     - self.z0 * w * v_ax * axis)
        hub_moment = -(self.m0 * w + self.m1 * v_ax) * v_ip_x_n + d * (self.p0 * w + self.p1 * v_ax) * v_ip
        return RotorLoads(thrust=thrust, torque=torque, force=thrust * axis + hub_force,
                          moment=d * torque * axis + hub_moment)


# The fitted coefficients (the data file's); the RotorPy ``k_*`` name of each term RotorPy has.
FITTED_ROTOR_COEFFICIENTS = tuple(coefficient.name for coefficient in fields(PolynomialRotorModel)
                                  if coefficient.metadata["fitted"])
ROTORPY_NAMES = {coefficient.name: coefficient.metadata["rotorpy"] for coefficient in fields(PolynomialRotorModel)
                 if coefficient.metadata["rotorpy"]}


def _bounds(value, name):
    low, high = (float(bound) for bound in value)
    if not (np.isfinite(low) and np.isfinite(high) and low <= high):
        raise ValueError(f"{name} must be finite [low, high] with low <= high")
    return low, high


@dataclass(frozen=True, kw_only=True)
class RotorEnvelope:
    """The box a rotor fit was fitted over: speed ``omega`` (rad/s), axial ``v_ax`` and in-plane ``v_ip`` airspeed
    (m/s), each ``(low, high)``.  Outside it the fit extrapolates; the simulation still runs."""

    omega: Tuple[float, float]
    v_ax: Tuple[float, float]
    v_ip: Tuple[float, float]

    def __post_init__(self):
        for name in ("omega", "v_ax", "v_ip"):
            object.__setattr__(self, name, _bounds(getattr(self, name), f"envelope.{name}"))

    def excess(self, omega, v_ax, v_ip) -> Dict[str, float]:
        """Signed distance outside each violated bound (negative below, positive above); empty inside."""

        result = {}
        for name, value in (("omega", omega), ("v_ax", v_ax), ("v_ip", v_ip)):
            low, high = getattr(self, name)
            if not low <= value <= high:
                result[name] = float(value - low if value < low else value - high)
        return result


@dataclass(frozen=True, kw_only=True)
class RotorFit:
    """One fitted rotor: its polynomial ``model``, the ``envelope`` it was fitted over, the fit's ``fit_rms``
    residuals per load and its ``provenance``.  ``name`` is the rotor it was fitted for."""

    name: str
    model: PolynomialRotorModel
    envelope: RotorEnvelope
    fit_rms: Dict[str, float]
    provenance: Dict[str, object]

    def __post_init__(self):
        if not self.name:
            raise ValueError("rotor fit needs a name")
        if not (self.model.a0 > 0.0 and self.model.b0 > 0.0):
            raise ValueError(f"rotor fit {self.name}: a0 and b0 must be positive")
        if not self.provenance:
            raise ValueError(f"rotor fit {self.name}: provenance must say where the numbers came from")


# 0.1 g of thrust: far below the model's accuracy, and large enough
# that dv0/dT stays finite when a tumbling rotor crosses T = 0
# (a 1e-9 N corner forces Drake's variable step to zero).
THRUST_EPSILON_N = 1e-3
NORM_EPSILON_MPS = 1e-9
PICARD_STEPS = 12


@dataclasses.dataclass(frozen=True)
class RotorWake:
    """Each rotor's wake at the disk, from the vehicle's own rotor thrust.

    * ``thrust`` (N,): N, along the axis; ``omega`` (N,): rad/s;
    * ``v_ax`` (N,): hub airspeed along the axis; ``v_ip`` (N,): in-plane speed, m/s;
    * ``v0`` (N,): Glauert mean induced velocity, m/s; ``chi`` (N,): wake skew, rad;
    * ``wake_direction`` (N, 3): unit wake centreline, body FLU; ``axes`` (N, 3): thrust axes;
    * ``tilt`` (N,): tilt of each rotor's assembly, rad.
    """

    thrust: PerRotor
    omega: PerRotor
    v_ax: PerRotor
    v_ip: PerRotor
    v0: PerRotor
    chi: PerRotor
    wake_direction: PerRotorVector
    axes: PerRotorVector
    tilt: PerRotor

    @property
    def chi_deg(self) -> PerRotor:
        return np.degrees(self.chi)

    @property
    def tilt_deg(self) -> PerRotor:
        return np.degrees(self.tilt)


def induced_velocity(thrust, v_ax, v_ip, rho, area):
    """Glauert mean induced velocity, 12 fixed damped-Picard steps.

    ``T = 2 rho A v0 sqrt(V_ip^2 + (V_ax + v0)^2)``.  The thrust is passed
    through ``softplus`` so ``v0 -> 0`` smoothly as ``T -> 0``, and the result
    is smooth-maxed against zero (a heavily descending rotor is outside the
    calibrated envelope; the simulator must still run).
    """

    thrust = np.asarray(thrust, dtype=float)
    v_ax = np.asarray(v_ax, dtype=float)
    v_ip = np.asarray(v_ip, dtype=float)
    driven = softplus(thrust, THRUST_EPSILON_N)
    denom = 2.0 * rho * area
    v0 = np.sqrt(driven / denom)
    for _ in range(PICARD_STEPS):
        speed = np.sqrt(v_ip * v_ip + (v_ax + v0) ** 2 + NORM_EPSILON_MPS**2)
        target = driven / (denom * speed)
        v0 = 0.5 * (v0 + target)
    return smooth_max(v0, 0.0, NORM_EPSILON_MPS)


def wake_state_vectors(thrust, v_ax, v_ip, rho, area, flow, axis):
    """Vector wake state for arrays: ``(v0, chi, d_hat)`` with ``d_hat`` the
    smooth unit wake centreline ``(flow - v0 axis) / |flow - v0 axis|``."""

    flow = np.asarray(flow, dtype=float)
    axis = np.asarray(axis, dtype=float)
    v0 = induced_velocity(thrust, v_ax, v_ip, rho, area)
    chi = np.arctan2(np.asarray(v_ip, dtype=float), np.asarray(v_ax, dtype=float) + v0)
    numerator = flow - v0[:, None] * axis
    norm = np.sqrt(np.sum(numerator * numerator, axis=-1, keepdims=True) + NORM_EPSILON_MPS**2)
    direction = numerator / norm
    return v0, chi, direction
