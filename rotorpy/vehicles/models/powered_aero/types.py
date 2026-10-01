"""The powered-aero model's data, typed: what an offline pipeline fits and the online model reads."""

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import cached_property
from typing import Dict, Optional, Tuple

import numpy as np

from rotorpy.vehicles.models.aero import COEFFICIENT_NAMES, AeroCoefficients, AeroReference, RateDerivatives
from rotorpy.vehicles.models.powered_aero.rotor import RotorFit

GAIN_NAMES = ("k_w", "gF", "gm", "dalpha")
GAIN_STATUSES = ("B3-default", "B4-fitted")


def _array(value, shape, name):
    array = np.array(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite array with shape {shape}, got {array.shape}")
    return array


def _unit(value, name):
    vector = _array(value, (3,), name)
    if abs(np.linalg.norm(vector) - 1.0) > 1e-6:
        raise ValueError(f"{name} must be a unit vector")
    return vector


def _grid(value, name, periodic=False):
    grid = np.array(value, dtype=float)
    if grid.ndim != 1 or grid.size < 2 or np.any(np.diff(grid) <= 0.0):
        raise ValueError(f"{name} must be a strictly increasing 1-D grid of at least 2 points")
    if periodic and not np.allclose(np.diff(grid), grid[1] - grid[0], rtol=1e-9, atol=1e-12):
        raise ValueError(f"periodic {name} must be uniform")
    return grid


def _tables(value: AeroCoefficients, shape, name) -> AeroCoefficients:
    """An :class:`AeroCoefficients` of tables, each checked to ``shape``."""

    return AeroCoefficients(*(_array(getattr(value, key), shape, f"{name}.{key}") for key in COEFFICIENT_NAMES))


def _set(record, **values):
    for key, value in values.items():
        object.__setattr__(record, key, value)


def _provenance(record, name):
    if not record.provenance:
        raise ValueError(f"{name}.provenance must say where the numbers came from")


@dataclass(frozen=True, kw_only=True)
class CleanData:
    """The rotors-off airframe: coefficient tables over ``(alpha_grid, beta_grid)`` (deg), rate derivatives, the
    bluff-body drag coefficient ``body_drag_cd``, and the ``trusted_alpha_deg`` band where the tables are solver
    data only (outside it they blend into a crude post-stall model; None: no band declared)."""

    alpha_grid: np.ndarray
    beta_grid: np.ndarray
    coefficients: AeroCoefficients[np.ndarray]
    rate_derivatives: RateDerivatives
    provenance: Dict[str, object]
    body_drag_cd: float = 0.0
    trusted_alpha_deg: Optional[Tuple[float, float]] = None

    def __post_init__(self):
        alpha = _grid(self.alpha_grid, "clean.alpha_grid", periodic=True)
        beta = _grid(self.beta_grid, "clean.beta_grid")
        if not all(np.isfinite(getattr(self.rate_derivatives, f.name)) for f in fields(RateDerivatives)):
            raise ValueError("clean.rate_derivatives must be finite")
        _set(self, alpha_grid=alpha, beta_grid=beta,
             coefficients=_tables(self.coefficients, (alpha.size, beta.size), "clean.coefficients"),
             body_drag_cd=float(self.body_drag_cd))
        if self.trusted_alpha_deg is not None:
            low, high = (float(value) for value in self.trusted_alpha_deg)
            if not low < high:
                raise ValueError("clean.trusted_alpha_deg must be [low, high] with low < high")
            _set(self, trusted_alpha_deg=(low, high))
        _provenance(self, "clean")


@dataclass(frozen=True, kw_only=True)
class SurfaceIncrements:
    """One control surface's effect on the whole aircraft: ``increments[alpha, delta]`` on the clean alpha grid and
    ``delta_grid_deg``, zero at zero deflection.  The surface itself (its actuator) is the vehicle's
    ``ControlSurface`` of the same ``name``."""

    name: str
    delta_grid_deg: np.ndarray
    increments: AeroCoefficients[np.ndarray]
    provenance: Dict[str, object]

    def __post_init__(self):
        grid = _grid(self.delta_grid_deg, f"{self.name}.delta_grid_deg")
        zero = np.flatnonzero(np.abs(grid) < 1e-9)
        if zero.size != 1:
            raise ValueError(f"{self.name}.delta_grid_deg must contain 0 deg")
        shape = np.shape(self.increments.CL)
        increments = _tables(self.increments, (shape[0] if shape else -1, grid.size), f"{self.name}.increments")
        if any(np.max(np.abs(getattr(increments, key)[:, zero[0]])) > 1e-12 for key in COEFFICIENT_NAMES):
            raise ValueError(f"{self.name}.increments must be zero at zero deflection")
        _set(self, delta_grid_deg=grid, increments=increments)
        _provenance(self, self.name)


@dataclass(frozen=True, kw_only=True)
class ReceiverData:
    """A lifting panel that rotor wakes wash: chordwise ``c_hat`` and spanwise ``s_hat`` unit axes (body FLU), chord,
    total area, and its 360-deg section polar ``cl, cd, cm`` over ``polar_alpha_grid`` (deg)."""

    name: str
    c_hat: np.ndarray
    s_hat: np.ndarray
    chord: float
    area_total: float
    polar_alpha_grid: np.ndarray
    cl: np.ndarray
    cd: np.ndarray
    cm: np.ndarray
    provenance: Dict[str, object]

    def __post_init__(self):
        c_hat, s_hat = _unit(self.c_hat, f"{self.name}.c_hat"), _unit(self.s_hat, f"{self.name}.s_hat")
        if abs(np.dot(c_hat, s_hat)) > 1e-6:
            raise ValueError(f"{self.name}: c_hat and s_hat must be orthogonal")
        if not (self.chord > 0.0 and self.area_total > 0.0):
            raise ValueError(f"{self.name}: chord and area_total must be positive")
        grid = _grid(self.polar_alpha_grid, f"{self.name}.polar_alpha_grid", periodic=True)
        polar = {key: _array(getattr(self, key), grid.shape, f"{self.name}.{key}") for key in ("cl", "cd", "cm")}
        if np.any(polar["cd"] < -1e-9):
            raise ValueError(f"{self.name}: cd must be nonnegative")
        _set(self, c_hat=c_hat, s_hat=s_hat, polar_alpha_grid=grid, **polar)
        _provenance(self, self.name)

    @cached_property
    def n_hat(self) -> np.ndarray:
        return np.cross(self.c_hat, self.s_hat)


@dataclass(frozen=True, kw_only=True)
class PairData:
    """How rotor ``source``'s wake washes receiver ``receiver``: the immersed area and its centroid over the wake
    direction ``(theta_grid, psi_grid)`` (deg), the wake radius and decay, and the gain polynomials."""

    receiver: str
    source: str
    theta_grid: np.ndarray
    psi_grid: np.ndarray
    immersion_area: np.ndarray
    immersion_centroid: np.ndarray
    wake_radius: float
    wake_zeta: float
    gains: Dict[str, np.ndarray]
    gain_status: str
    calibration_report: Dict[str, object]

    def __post_init__(self):
        name = f"pair {self.receiver}<-{self.source}"
        theta = _grid(self.theta_grid, f"{name}.theta_grid")
        psi = _grid(self.psi_grid, f"{name}.psi_grid", periodic=True)
        if theta[0] < 0.0 or theta[-1] > 180.0 + 1e-9:
            raise ValueError(f"{name}.theta_grid must lie in [0, 180] deg")
        shape = (theta.size, psi.size)
        area = _array(self.immersion_area, shape, f"{name}.immersion_area")
        if np.any(area < 0.0):
            raise ValueError(f"{name}.immersion_area must be nonnegative")
        if not (self.wake_radius > 0.0 and self.wake_zeta >= 0.0):
            raise ValueError(f"{name}: wake radius must be positive and zeta nonnegative")
        if set(self.gains) != set(GAIN_NAMES):
            raise ValueError(f"{name}.gains must define {GAIN_NAMES}")
        if self.gain_status not in GAIN_STATUSES:
            raise ValueError(f"{name}.gain_status must be one of {GAIN_STATUSES}")
        if self.gain_status == "B4-fitted" and not self.calibration_report:
            raise ValueError(f"{name}: a B4-fitted pair must carry its calibration_report")
        _set(self, theta_grid=theta, psi_grid=psi, immersion_area=area,
             immersion_centroid=_array(self.immersion_centroid, shape + (3,), f"{name}.immersion_centroid"),
             gains={key: _array(self.gains[key], (6,), f"{name}.gains[{key}]") for key in GAIN_NAMES})


@dataclass(frozen=True, kw_only=True)
class PoweredAeroData:
    """The whole data file: reference geometry (``area``, ``span``, ``cbar``, the MRP ``mrp_xyz``), the air density
    ``rho`` it assumes, the rotor fits, the clean airframe, the receivers, the (receiver, source) pairs and the
    control-surface increments.  ``meta`` says when and from what it was built."""

    meta: Dict[str, object]
    rho: float
    area: float
    span: float
    cbar: float
    mrp_xyz: np.ndarray
    rotors: Tuple[RotorFit, ...]
    clean: CleanData
    receivers: Tuple[ReceiverData, ...]
    pairs: Tuple[PairData, ...]
    surfaces: Tuple[SurfaceIncrements, ...] = ()

    def __post_init__(self):
        if not all(value > 0.0 for value in (self.rho, self.area, self.span, self.cbar)):
            raise ValueError("rho and the reference S, b and cbar must be positive")
        _set(self, mrp_xyz=_array(self.mrp_xyz, (3,), "mrp_xyz"), rotors=tuple(self.rotors),
             receivers=tuple(self.receivers), pairs=tuple(self.pairs), surfaces=tuple(self.surfaces))
        for kind, records in (("rotor", self.rotors), ("receiver", self.receivers), ("surface", self.surfaces)):
            names = [record.name for record in records]
            if len(set(names)) != len(names):
                raise ValueError(f"duplicate {kind} names in {names}")
        receivers, sources = {r.name for r in self.receivers}, {r.name for r in self.rotors}
        if not self.pairs or {p.receiver for p in self.pairs} != receivers or {p.source for p in self.pairs} != sources:
            raise ValueError("every pair must name a known receiver and rotor, and every receiver and rotor a pair")
        for surface in self.surfaces:
            if surface.increments.CL.shape[0] != self.clean.alpha_grid.size:
                raise ValueError(f"{surface.name}.increments must lie on the clean alpha grid")
        by_name = {receiver.name: receiver for receiver in self.receivers}
        for receiver in self.receivers:  # mirrored panels: mirrored suction normals, equal size and polars
            other = by_name.get(receiver.name.replace("left", "right")) if "left" in receiver.name else None
            if other is None:
                continue
            if (not np.allclose(receiver.n_hat, other.n_hat * [1.0, -1.0, 1.0], atol=1e-6)
                    or abs(receiver.chord - other.chord) > 1e-9 or abs(receiver.area_total - other.area_total) > 1e-12
                    or not np.allclose(receiver.cl, other.cl, atol=1e-12)
                    or not np.allclose(receiver.cd, other.cd, atol=1e-12)):
                raise ValueError(f"mirrored receivers {receiver.name}/{other.name} differ")

    @property
    def reference(self) -> AeroReference:
        return AeroReference(area=self.area, span=self.span, cbar=self.cbar, mrp_xyz=self.mrp_xyz)

    def receiver(self, name) -> ReceiverData:
        return {receiver.name: receiver for receiver in self.receivers}[name]

    def surface(self, name) -> SurfaceIncrements:
        return {surface.name: surface for surface in self.surfaces}[name]

    def rotor(self, name) -> RotorFit:
        return {rotor.name: rotor for rotor in self.rotors}[name]
