"""The data file's fixed-wing tables as an airframe model: ``C + sum(delta C)``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

from rotorpy.vehicles.models.aero import (
    COEFFICIENT_NAMES, FRD_FROM_FLU, AeroCoefficients, AerodynamicWrench, AirframeState, body_from_wind, flow_angles,
)
from rotorpy.vehicles.models.powered_aero.splines import RectSpline, cross3, smooth_clamp
from rotorpy.vehicles.models.powered_aero.types import PoweredAeroData

if TYPE_CHECKING:
    from rotorpy.vehicles.models.spec import ControlSurface

BETA_CLAMP_EPS_DEG = 2.0


def _splines(x_grid, y_grid, tables: AeroCoefficients[np.ndarray]) -> AeroCoefficients[RectSpline]:
    return AeroCoefficients(*(RectSpline(x_grid, y_grid, getattr(tables, name), periodic_x=True)
                              for name in COEFFICIENT_NAMES))


def _evaluate(splines: AeroCoefficients[RectSpline], x, y) -> AeroCoefficients[float]:
    return AeroCoefficients(*(float(getattr(splines, name)(x, y)) for name in COEFFICIENT_NAMES))


class FixedWingAero:
    """The clean airframe and the increments of ``surfaces`` (the airframe's, in order; each found in the data file by its name), at
    ``data.rho``.  ``beta`` is smoothly clamped to the clean grid, each ``delta`` clipped to its surface's grid."""

    def __init__(self, data: PoweredAeroData, surfaces: Sequence[ControlSurface]):
        clean = data.clean
        self.reference, self.rates, self.rho = data.reference, clean.rate_derivatives, data.rho
        self._clean = _splines(clean.alpha_grid, clean.beta_grid, clean.coefficients)
        self._beta_range = (clean.beta_grid[0], clean.beta_grid[-1])

        # body drag comes (generally) from a different source so it's treated differently
        self._body_drag = AeroCoefficients(CL=0.0, CD=clean.body_drag_cd, CY=0.0, Cl=0.0, Cm=0.0, Cn=0.0)
        self._surfaces = []
        for surface in surfaces:
            increments = data.surface(surface.name)
            grid = increments.delta_grid_deg
            self._surfaces.append(((grid[0], grid[-1]), _splines(clean.alpha_grid, grid, increments.increments)))

    def coefficients(self, alpha, beta, deflections) -> AeroCoefficients[float]:
        """``C(alpha, beta)`` plus each surface's ``delta C(alpha, delta)`` (rad)."""

        alpha_deg = np.degrees(alpha)
        beta_deg = smooth_clamp(np.degrees(beta), *self._beta_range, BETA_CLAMP_EPS_DEG)
        c = _evaluate(self._clean, alpha_deg, beta_deg) + self._body_drag
        for (delta_range, splines), delta in zip(self._surfaces, deflections):
            c = c + _evaluate(splines, alpha_deg, float(np.clip(np.degrees(delta), *delta_range)))
        return c

    def __call__(self, state: AirframeState) -> AerodynamicWrench:
        reference, w = self.reference, state.w
        # One frame conversion at the boundary: FRD flow angles and rates at the MRP.
        speed, alpha, beta = flow_angles(state.airspeed + cross3(w, reference.mrp_xyz))
        p, q, r = FRD_FROM_FLU @ w
        c = self.coefficients(alpha, beta, state.surface_deflections)
        qbar = 0.5 * self.rho * speed * speed
        area, span, cbar = reference.area, reference.span, reference.cbar

        force_frd = qbar * area * body_from_wind(alpha, beta) @ np.array([-c.CD, c.CY, -c.CL])
        moment_frd = qbar * area * np.array([span * c.Cl, cbar * c.Cm, span * c.Cn])

        # Rate terms, written multiplied out so nothing divides by V.
        d = self.rates
        scale = 0.25 * self.rho * speed * area
        force_frd[1] += scale * span * (d.CYp * p + d.CYr * r)
        force_frd[2] -= scale * cbar * d.CLq * q
        moment_frd[0] += scale * span * span * (d.Clp * p + d.Clr * r)
        moment_frd[1] += scale * cbar * cbar * d.Cmq * q
        moment_frd[2] += scale * span * span * (d.Cnp * p + d.Cnr * r)

        # Moment transfer from the MRP to the airframe origin: M_O = M_MRP + r_MRP x F.
        force = FRD_FROM_FLU @ force_frd
        return AerodynamicWrench(force, FRD_FROM_FLU @ moment_frd + cross3(reference.mrp_xyz, force))
