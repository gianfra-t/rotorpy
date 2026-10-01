"""Component interaction increments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import numpy as np

from rotorpy.vehicles.models.powered_aero.rotor import RotorWake
from rotorpy.vehicles.models.powered_aero.splines import (
    PeriodicSpline1D,
    PeriodicSplineBatch,
    RectSpline,
    RectSplineBatch,
    cross3,
    smooth_clamp,
)

if TYPE_CHECKING:
    from rotorpy.vehicles.models.powered_aero.types import PoweredAeroData

FLOW_EPSILON_MPS = 1e-12

# Smooth clamps: the interior is untouched, so the B3
# defaults (k_w=1.6, g=1, dalpha=0) pass through exactly, and a B4 fit cannot
# leave the sane band.
GAIN_CLAMP_EPS = 1e-3
K_W_BAND = (1.0, 2.0)
G_BAND = (0.2, 3.0)
DALPHA_BAND_DEG = (-15.0, 15.0)
GAIN_NAMES = ("k_w", "gF", "gm", "dalpha")
_GAIN_LOW = np.array([K_W_BAND[0], G_BAND[0], G_BAND[0], DALPHA_BAND_DEG[0]])[:, None]
_GAIN_HIGH = np.array([K_W_BAND[1], G_BAND[1], G_BAND[1], DALPHA_BAND_DEG[1]])[:, None]


def wake_angles(direction):
    """``(theta_deg, psi_deg)`` of a wake centreline in body FLU.

    ``theta`` is measured from ``-z_B`` (hover: 0) and ``psi`` around it:
    ``d = (-sin(theta) cos(psi), -sin(theta) sin(psi), -cos(theta))``.
    """

    direction = np.asarray(direction, dtype=float)
    theta = np.arccos(np.clip(-direction[..., 2], -1.0, 1.0))
    psi = np.arctan2(-direction[..., 1], -direction[..., 0])
    return np.degrees(theta), np.degrees(psi)


class _PanelPolar:
    def __init__(self, grid, cl, cd, cm):
        self.cl = PeriodicSpline1D(grid, cl)
        self.cd = PeriodicSpline1D(grid, cd)
        self.cm = PeriodicSpline1D(grid, cm)


class InteractionAero:
    """Pure component-interaction wrench (body FLU, about the airframe origin)."""

    def __init__(self, data: PoweredAeroData, rotor_index: Dict[str, int]):
        """``rotor_index`` places each pair's source rotor, by name, in the vehicle's rotor order."""

        self.data = data
        self._receivers = {receiver.name: receiver for receiver in data.receivers}
        self._rotor_index = rotor_index
        self._polar = {
            receiver.name: _PanelPolar(
                receiver.polar_alpha_grid, receiver.cl, receiver.cd, receiver.cm
            )
            for receiver in data.receivers
        }
        # Per-pair tables, kept for inspection; the call path uses the
        # stacked arrays below, built from these same splines.
        self._tables = []
        for pair in data.pairs:
            self._tables.append(
                {
                    "pair": pair,
                    "source": self._rotor_index[pair.source],
                    "area": RectSpline(
                        pair.psi_grid, pair.theta_grid, pair.immersion_area.T, periodic_x=True
                    ),
                    "centroid": [
                        RectSpline(
                            pair.psi_grid, pair.theta_grid,
                            pair.immersion_centroid[..., axis].T, periodic_x=True,
                        )
                        for axis in range(3)
                    ],
                    "receiver": self._receivers[pair.receiver],
                }
            )

        receiver_order = [receiver.name for receiver in data.receivers]
        receiver_row = {name: index for index, name in enumerate(receiver_order)}
        receivers = [table["receiver"] for table in self._tables]
        self._pair_names = [(table["pair"].receiver, table["pair"].source)
                            for table in self._tables]
        self._source = np.array([table["source"] for table in self._tables], dtype=int)
        self._receiver_row = np.array([receiver_row[receiver.name] for receiver in receivers],
                                      dtype=int)
        self._c_hat = np.array([receiver.c_hat for receiver in receivers], dtype=float)
        self._s_hat = np.array([receiver.s_hat for receiver in receivers], dtype=float)
        self._n_hat = cross3(self._c_hat, self._s_hat)
        self._chord = np.array([receiver.chord for receiver in receivers], dtype=float)
        # gains[g, p, :] are the six quadratic coefficients of GAIN_NAMES[g].
        self._gains = np.array(
            [[table["pair"].gains[name] for table in self._tables] for name in GAIN_NAMES],
            dtype=float,
        ).reshape(len(GAIN_NAMES), len(self._tables), 6)
        self._immersion = RectSplineBatch(
            [[table["area"]] + table["centroid"] for table in self._tables]
        )
        polars = [self._polar[name] for name in receiver_order]
        self._polars = PeriodicSplineBatch([[polar.cl, polar.cd, polar.cm] for polar in polars])

    def wrench(self, v_origin_flu, rates_flu, wake: RotorWake):
        """Sum all pair increments.  ``wake`` comes from the *same* rotor model that loads the rotor bodies."""

        terms = self._pair_arrays(v_origin_flu, rates_flu, wake)
        return terms["force"].sum(axis=0), terms["moment"].sum(axis=0)

    def pair_diagnostics(self, v_origin_flu, rates_flu, wake: RotorWake):
        """Per-pair flow state, for logging: the same numbers ``wrench`` sums."""

        terms = self._pair_arrays(v_origin_flu, rates_flu, wake)
        return [
            {
                "receiver": receiver,
                "source": source,
                "area": float(terms["area"][index]),
                "alpha_free_deg": float(terms["alpha_free_deg"][index]),
                "alpha_jet_deg": float(terms["alpha_jet_deg"][index]),
                "q_free": float(terms["q_free"][index]),
                "q_jet": float(terms["q_jet"][index]),
                "jet_fraction": float(terms["jet_fraction"][index]),
                "force": terms["force"][index].copy(),
                "moment": terms["moment"][index].copy(),
            }
            for index, (receiver, source) in enumerate(self._pair_names)
        ]

    def _gain_values(self, chi_deg, delta_t_deg):
        """Clamped gains ``(k_w, g_F, g_m, dalpha_deg)``, each of shape (P,)."""

        c = self._gains
        x = chi_deg
        y = delta_t_deg
        raw = c[..., 0] + c[..., 1] * x + c[..., 2] * y + c[..., 3] * x * x \
            + c[..., 4] * x * y + c[..., 5] * y * y
        return smooth_clamp(raw, _GAIN_LOW, _GAIN_HIGH, GAIN_CLAMP_EPS)

    def _pair_arrays(self, v_origin_flu, rates_flu, wake: RotorWake):
        """All pair terms as arrays over the pair axis (see module docstring)."""

        rho = self.data.rho
        source = self._source
        chi_deg = wake.chi_deg[source]
        delta_t_deg = wake.tilt_deg[source]
        k_w, g_f, g_m, d_alpha_deg = self._gain_values(chi_deg, delta_t_deg)

        direction = wake.wake_direction[source]
        axis = wake.axes[source]
        v0 = wake.v0[source]
        theta_deg, psi_deg = wake_angles(direction)

        immersion = self._immersion(psi_deg, theta_deg)
        area = immersion[:, 0]
        centroid = immersion[:, 1:]

        c_hat = self._c_hat
        s_hat = self._s_hat
        n_hat = self._n_hat

        jet = k_w * v0
        u_r = -(np.asarray(v_origin_flu, dtype=float)
                + cross3(np.asarray(rates_flu, dtype=float), centroid))
        u_s = u_r - jet[:, None] * axis

        alpha_r, q_r, e_d, e_l = _panel_flow(u_r, c_hat, s_hat, n_hat, rho)
        alpha_s, q_s, e_d_s, e_l_s = _panel_flow(u_s, c_hat, s_hat, n_hat, rho)

        speed_r = np.sqrt(_dot(u_r, u_r) + FLOW_EPSILON_MPS * FLOW_EPSILON_MPS)
        lam = jet / (speed_r + jet + FLOW_EPSILON_MPS)
        shift = lam * np.radians(d_alpha_deg)
        alpha_jet = alpha_s + shift

        n_pairs = source.size
        coefficients = self._polars(
            np.degrees(np.concatenate((alpha_jet, alpha_r))),
            np.concatenate((self._receiver_row, self._receiver_row)),
        )
        cl_s, cd_s, cm_s = coefficients[:n_pairs].T
        cl_r, cd_r, cm_r = coefficients[n_pairs:].T

        load_s = q_s * area
        load_r = q_r * area
        force = g_f[:, None] * (
            load_s[:, None] * (cl_s[:, None] * e_l_s + cd_s[:, None] * e_d_s)
            - load_r[:, None] * (cl_r[:, None] * e_l + cd_r[:, None] * e_d)
        )
        m_s = load_s * self._chord * cm_s
        m_r = load_r * self._chord * cm_r
        moment = cross3(centroid, force) + (g_m * (m_s - m_r))[:, None] * s_hat
        return {
            "area": area,
            "alpha_free_deg": np.degrees(alpha_r),
            "alpha_jet_deg": np.degrees(alpha_jet),
            "q_free": q_r,
            "q_jet": q_s,
            "jet_fraction": lam,
            "force": force,
            "moment": moment,
        }


def _dot(a, b):
    """Row-wise dot product of ``(P, 3)`` stacks."""

    return a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1] + a[:, 2] * b[:, 2]


def _panel_flow(u, c_hat, s_hat, n_hat, rho):
    """Angle of attack, dynamic pressure, drag and lift directions of panels.

    All vector arguments are ``(P, 3)`` stacks; the scalars come back ``(P,)``.
    """

    u_perp = u - _dot(u, s_hat)[:, None] * s_hat
    speed2 = _dot(u_perp, u_perp) + FLOW_EPSILON_MPS**2
    alpha = np.arctan2(_dot(u_perp, n_hat), _dot(u_perp, c_hat))
    q = 0.5 * rho * speed2
    e_d = u_perp / np.sqrt(speed2)[:, None]
    e_l = cross3(e_d, s_hat)
    return alpha, q, e_d, e_l
