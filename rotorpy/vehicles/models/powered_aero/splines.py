"""Smooth interpolation and blending helpers."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline, PPoly, RectBivariateSpline


def cross3(a, b):
    """Cross product of 3-vectors (or stacks ``(..., 3)``), written out.

    ``np.cross`` spends most of its time in axis bookkeeping for inputs this
    small; the callback calls it per pair and per Runge-Kutta stage.
    """

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
    bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
    return np.stack((ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx), axis=-1)


def smooth_norm(vector, eps):
    """Smooth positive norm ``sqrt(|v|^2 + eps^2)``; never zero."""

    vector = np.asarray(vector, dtype=float)
    return np.sqrt(np.sum(vector * vector) + eps * eps)


def smooth_max(a, b, eps):
    """C-infinity max with width ``eps`` (exact to O(eps^2))."""

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return 0.5 * (a + b + np.hypot(a - b, eps))


def smooth_min(a, b, eps):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return 0.5 * (a + b - np.hypot(a - b, eps))


def softplus(x, eps):
    x = np.asarray(x, dtype=float)
    return eps * np.logaddexp(0.0, x / eps)


def smooth_clamp(x, low, high, eps):
    """Smooth clamp to ``[low, high]`` that is the identity in the interior."""

    x = np.asarray(x, dtype=float)
    return x - softplus(x - high, eps) + softplus(low - x, eps)


class PeriodicSpline1D:
    """Exact periodic cubic spline on a uniform, endpoint-free grid."""

    def __init__(self, grid, values):
        grid = np.asarray(grid, dtype=float)
        values = np.asarray(values, dtype=float)
        if grid.ndim != 1 or values.shape != grid.shape:
            raise ValueError("grid and values must be 1-D and aligned")
        if grid.size < 4:
            raise ValueError("a periodic cubic spline needs at least 4 samples")
        diffs = np.diff(grid)
        if not np.allclose(diffs, diffs[0], rtol=1e-9, atol=1e-12):
            raise ValueError("a periodic spline grid must be uniform")
        self.grid = grid
        self.period = float(diffs[0]) * grid.size
        extended = np.append(grid, grid[0] + self.period)
        self._spline = CubicSpline(extended, np.append(values, values[0]), bc_type="periodic")

    def __call__(self, x):
        wrapped = np.mod(np.asarray(x, dtype=float) - self.grid[0], self.period) + self.grid[0]
        result = self._spline(wrapped)
        return result


class Spline1D:
    """Natural cubic spline on a monotone grid, clamped evaluation."""

    def __init__(self, grid, values):
        self.grid = np.asarray(grid, dtype=float)
        self._spline = CubicSpline(self.grid, np.asarray(values, dtype=float))

    def __call__(self, x):
        return self._spline(np.clip(np.asarray(x, dtype=float), self.grid[0], self.grid[-1]))


class RectSpline:
    """Tensor cubic interpolation on a rectangular grid.

    ``values[i, j]`` is the value at ``(x_grid[i], y_grid[j])``.  The x axis may
    be periodic (``alpha``, wake azimuth); the y axis is clamped at its edges
    and callers smooth-saturate the input where the table should flatten.
    The periodic wrap is implemented by mirroring four sample columns across
    each edge before fitting, which makes the local B-spline coefficients
    periodic up to the wrap tolerance used in the smoothness tests.
    """

    def __init__(self, x_grid, y_grid, values, periodic_x=False, wrap_margin=4):
        x_grid = np.asarray(x_grid, dtype=float)
        y_grid = np.asarray(y_grid, dtype=float)
        values = np.asarray(values, dtype=float)
        if values.shape != (x_grid.size, y_grid.size):
            raise ValueError("values must have shape (len(x_grid), len(y_grid))")
        if periodic_x:
            if x_grid.size < 2 * wrap_margin:
                raise ValueError("periodic grid needs at least 2 * wrap_margin samples")
            period = x_grid[-1] + (x_grid[1] - x_grid[0]) - x_grid[0]
            left = x_grid[-wrap_margin:] - period
            right = x_grid[:wrap_margin] + period
            x_ext = np.concatenate((left, x_grid, right))
            values_ext = np.concatenate(
                (values[-wrap_margin:], values, values[:wrap_margin]), axis=0
            )
            self._period = period
        else:
            x_ext = x_grid
            values_ext = values
            self._period = None
        self.x_grid = x_grid
        self.y_grid = y_grid
        # RectBivariateSpline wants z[i, j] = f(x[i], y[j]).
        self._spline = RectBivariateSpline(x_ext, y_grid, values_ext, kx=3, ky=3, s=0)

    def __call__(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if self._period is not None:
            # The wrapped x lies in [x0, x0 + period), inside the extended
            # grid; clipping to the original grid would flatten the last cell.
            x = np.mod(x - self.x_grid[0], self._period) + self.x_grid[0]
        else:
            x = np.clip(x, self.x_grid[0], self.x_grid[-1])
        y = np.clip(y, self.y_grid[0], self.y_grid[-1])
        # Recent scipy returns shape (1,) for scalar inputs; keep the input shape.
        return self._spline(x, y, grid=False).reshape(np.broadcast(x, y).shape)


def _group_by_key(keys):
    """``{key: [row indices]}`` in first-seen order."""

    groups = {}
    for index, key in enumerate(keys):
        groups.setdefault(key, []).append(index)
    return groups


class PeriodicSplineBatch:
    """Evaluate rows of :class:`PeriodicSpline1D` at stacked points.

    ``rows[r]`` is a sequence of ``K`` splines on one grid (e.g. a receiver's
    ``cl, cd, cm``).  ``batch(x, row)`` returns ``(M, K)``: row ``row[m]``
    evaluated at ``x[m]``, with the same wrap as ``PeriodicSpline1D`` and the
    same piecewise cubic coefficients, one array evaluation per distinct grid.
    """

    def __init__(self, rows):
        rows = [list(row) for row in rows]
        self.n_rows = len(rows)
        self._groups = []
        keys = [(row[0].grid.tobytes(), row[0].period) for row in rows]
        for indices in _group_by_key(keys).values():
            first = rows[indices[0]][0]
            breaks = first._spline.x
            # coefficients[power, interval, row-in-group, k]
            coefficients = np.stack(
                [np.stack([spline._spline.c for spline in rows[index]], axis=-1)
                 for index in indices],
                axis=2,
            )
            local = np.full(self.n_rows, -1, dtype=int)
            local[indices] = np.arange(len(indices))
            self._groups.append((np.array(indices), local, first.grid[0], first.period,
                                 breaks, coefficients))
        self.width = len(rows[0])

    def __call__(self, x, row):
        x = np.asarray(x, dtype=float)
        row = np.asarray(row, dtype=int)
        result = np.empty((x.size, self.width))
        for indices, local, x0, period, breaks, coefficients in self._groups:
            mask = np.isin(row, indices) if len(self._groups) > 1 else slice(None)
            wrapped = np.mod(x[mask] - x0, period) + x0
            interval = np.clip(np.searchsorted(breaks, wrapped, side="right") - 1,
                               0, breaks.size - 2)
            dx = (wrapped - breaks[interval])[:, None]
            c = coefficients[:, interval, local[row[mask]]]
            result[mask] = ((c[0] * dx + c[1]) * dx + c[2]) * dx + c[3]
        return result


def _cubic_bspline_power_basis(knots):
    """Power-form cubic B-splines per knot interval, for fast basis lookups.

    ``table[l, a, i]`` is the coefficient of ``(x - knots[l])**(3 - a)`` in
    ``B_{l - 3 + i}`` on interval ``l`` (the exact conversion of
    ``PPoly.from_spline``).
    """

    n_basis = knots.size - 4
    table = np.zeros((knots.size - 1, 4, 4))
    for basis in range(n_basis):
        unit = np.zeros(n_basis)
        unit[basis] = 1.0
        power = PPoly.from_spline((knots, unit, 3)).c
        for offset in range(4):
            interval = basis + 3 - offset
            if 0 <= interval < table.shape[0]:
                table[interval, :, offset] = power[:, interval]
    return table


def _cubic_bspline_basis(knots, table, x):
    """Nonzero cubic B-spline values at ``x`` with FITPACK's interval rule.

    Returns ``(first, basis)``: ``basis[m, i]`` is ``B_{first[m] + i}(x[m])``.
    """

    span = np.clip(np.searchsorted(knots, x, side="right") - 1, 3, knots.size - 5)
    dx = x - knots[span]
    powers = np.stack((dx * dx * dx, dx * dx, dx, np.ones_like(dx)), axis=-1)
    return span - 3, np.einsum("ma,mai->mi", powers, table[span])


class RectSplineBatch:
    """Evaluate rows of :class:`RectSpline` at one point per row.

    ``rows[p]`` is a sequence of ``K`` splines on one grid (a pair's immersed
    area and centroid components).  ``batch(x, y)`` with ``x, y`` of shape
    ``(P,)`` returns ``(P, K)``: row ``p`` at ``(x[p], y[p])``, with the same
    wrap/clamp as ``RectSpline`` and the same tensor B-spline coefficients.
    """

    def __init__(self, rows):
        rows = [list(row) for row in rows]
        self.n_rows = len(rows)
        self.width = len(rows[0])
        self._groups = []
        keys = []
        for row in rows:
            spline = row[0]
            tx, ty = spline._spline.tck[:2]
            if spline._spline.degrees != (3, 3):
                raise ValueError("RectSplineBatch needs bicubic splines")
            keys.append((tx.tobytes(), ty.tobytes(), spline._period,
                         spline.x_grid[0], spline.x_grid[-1],
                         spline.y_grid[0], spline.y_grid[-1]))
        for indices in _group_by_key(keys).values():
            first = rows[indices[0]][0]
            tx, ty = first._spline.tck[:2]
            shape = (tx.size - 4, ty.size - 4)
            # coefficients[row-in-group, ix, iy, k]
            coefficients = np.array(
                [np.stack([spline._spline.tck[2].reshape(shape) for spline in rows[index]],
                          axis=-1)
                 for index in indices]
            )
            self._groups.append((np.array(indices), first._period, first.x_grid, first.y_grid,
                                 tx, ty, _cubic_bspline_power_basis(tx),
                                 _cubic_bspline_power_basis(ty), coefficients,
                                 np.arange(len(indices))[:, None, None]))

    def __call__(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        result = np.empty((self.n_rows, self.width))
        offsets = np.arange(4)
        for (indices, period, x_grid, y_grid, tx, ty, table_x, table_y,
             coefficients, rows) in self._groups:
            xs = x[indices]
            if period is not None:
                xs = np.mod(xs - x_grid[0], period) + x_grid[0]
            else:
                xs = np.clip(xs, x_grid[0], x_grid[-1])
            ys = np.clip(y[indices], y_grid[0], y_grid[-1])
            first_x, basis_x = _cubic_bspline_basis(tx, table_x, xs)
            first_y, basis_y = _cubic_bspline_basis(ty, table_y, ys)
            local = coefficients[
                rows,
                (first_x[:, None] + offsets)[:, :, None],
                (first_y[:, None] + offsets)[:, None, :],
            ]
            result[indices] = np.einsum("pi,pj,pijk->pk", basis_x, basis_y, local)
        return result


def quadratic_2d(coefficients, x_deg, y_deg):
    """Evaluate ``sum_{j+k<=2} c[j,k] x^j y^k``.

    ``coefficients`` is the flat list ``[c00, c10, c01, c20, c11, c02]`` with
    x and y in degrees (the gain-polynomial convention of the data file).
    """

    c = np.asarray(coefficients, dtype=float)
    if c.shape != (6,):
        raise ValueError("a quadratic 2-D gain needs exactly 6 coefficients")
    x = np.asarray(x_deg, dtype=float)
    y = np.asarray(y_deg, dtype=float)
    return (
        c[0]
        + c[1] * x
        + c[2] * y
        + c[3] * x * x
        + c[4] * x * y
        + c[5] * y * y
    )
