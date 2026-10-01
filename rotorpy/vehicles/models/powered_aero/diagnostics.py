"""Where :class:`~rotorpy.vehicles.models.powered_aero.model.PoweredAeroModel` is being evaluated, for judging a simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from rotorpy.vehicles.models.aero import AirframeState, flow_angles
from rotorpy.vehicles.models.powered_aero.rotor import RotorWake
from rotorpy.vehicles.models.powered_aero.splines import cross3

if TYPE_CHECKING:
    from rotorpy.vehicles.models.powered_aero.model import PoweredAeroModel

# Freestream dynamic pressure above which post-stall clean data is flagged:
# 50 Pa is ~9 m/s at sea level, where the airframe starts to carry a visible
# share of the weight.  An analyst's threshold, not a physical constant.
DEFAULT_Q_ALERT_PA = 50.0
# Pairs whose immersed area is below this are left out of the log.
PAIR_AREA_LOG_FLOOR_M2 = 1e-6


@dataclass(frozen=True)
class EnvelopeReport:
    """Membership of the fitted rotor envelopes (a flag, not a switch).

    ``outside`` holds ``(rotor name, signed excess)`` per violated bound; ``margin`` is the largest excess.
    """

    outside: tuple
    margin: float


def envelope_report(model: PoweredAeroModel, wake: RotorWake) -> EnvelopeReport:
    """Which fitted rotors are outside their fit's ``(omega, v_ax, v_ip)`` envelope, and by how much."""

    outside = []
    for fit in model.data.rotors:
        index = model.rotor_index[fit.name]
        excess = fit.envelope.excess(wake.omega[index], wake.v_ax[index], wake.v_ip[index])
        outside.extend((fit.name, value) for value in excess.values())
    excesses = [excess for _, excess in outside]
    margin = 0.0
    if excesses:
        margin = min(excesses) if min(excesses) < 0.0 else max(excesses)
    return EnvelopeReport(tuple(outside), margin)


def flight_diagnostics(model: PoweredAeroModel, state: AirframeState, q_alert_pa=DEFAULT_Q_ALERT_PA) -> dict:
    """One log record for ``state``.

    * ``fixed_wing``: freestream speed, ``alpha``, ``beta`` and dynamic pressure
      at the MRP (the rotors-off portion), and whether ``alpha`` lies in the
      data file's trusted band (solver data; outside it the tables blend
      into a crude post-stall model).
    * ``pairs``: per (receiver, source) with nonzero immersion, the
      freestream and jet angle of attack on the immersed patch, the
      freestream and jet dynamic pressure (the rotor portion), the jet
      fraction and the increment force.  The jet side uses 360 deg section
      polars, so a large jet angle (hover download) is not by itself an
      alert.
    * ``alert``: the fixed-wing freestream dynamic pressure exceeds
      ``q_alert_pa`` while its ``alpha`` is outside the trusted band: the
      airframe carries real load on post-stall data.  Hover download never
      trips it, because there the freestream dynamic pressure is ~0.
    """

    speed, alpha, beta = flow_angles(state.airspeed + cross3(state.w, model.data.mrp_xyz))
    qbar = 0.5 * model.data.rho * float(speed) ** 2
    alpha_deg, beta_deg = float(np.degrees(alpha)), float(np.degrees(beta))
    band = model.data.clean.trusted_alpha_deg
    in_band = True if band is None else bool(band[0] <= alpha_deg <= band[1])
    wake = model.rotor_wake(state)
    pairs = [
        terms for terms in model.interaction.pair_diagnostics(state.airspeed, state.w, wake)
        if terms["area"] > PAIR_AREA_LOG_FLOOR_M2
    ]
    return {
        "fixed_wing": {"speed": float(speed), "alpha_deg": alpha_deg, "beta_deg": beta_deg,
                  "qbar": qbar, "in_trusted_band": in_band, "trusted_alpha_deg": band,
                  "surface_deflections_deg": np.degrees(state.surface_deflections)},
        "rotors": {"v0": wake.v0.copy(), "chi_deg": wake.chi_deg, "thrust": wake.thrust.copy()},
        "pairs": pairs,
        "alert": bool(qbar > q_alert_pa and not in_band),
        "q_alert_pa": q_alert_pa,
        "envelope": envelope_report(model, wake),
    }
