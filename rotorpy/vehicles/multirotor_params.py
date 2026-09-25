"""Structured multirotor parameters and rotor aerodynamics.

The parser for the structured airframe/assemblies/rotors parameter form that
:class:`~rotorpy.vehicles.drake.multirotor.DrakeMultirotor` takes, its
converter from a legacy ``quad_params`` dict, and the pure per-rotor aero-load
function.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from rotorpy.vehicles.multirotor import Multirotor


_MISSING = object()


def _vector(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector with shape (3,)")
    return result.copy()


def _inertia(value, name, positive=True):
    result = np.asarray(value, dtype=float)
    if result.shape != (3, 3) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite matrix with shape (3, 3)")
    if not np.allclose(result, result.T, rtol=0.0, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(result)
    if positive and np.any(eigenvalues <= 0.0):
        raise ValueError(f"{name} must be positive definite")
    if not positive and np.any(eigenvalues < -1e-12):
        raise ValueError(f"{name} must be positive semidefinite")
    tolerance = 1e-12 * max(1.0, float(np.sum(np.abs(eigenvalues))))
    if eigenvalues[-1] > eigenvalues[0] + eigenvalues[1] + tolerance:
        raise ValueError(f"{name} principal moments violate the triangle inequalities")
    return result.copy()


def _rotation(value, name):
    if value is None:
        return np.eye(3)
    if hasattr(value, "as_matrix"):
        result = np.asarray(value.as_matrix(), dtype=float)
    else:
        array = np.asarray(value, dtype=float)
        if array.shape == (4,):
            if not np.all(np.isfinite(array)) or np.linalg.norm(array) == 0.0:
                raise ValueError(f"{name} quaternion must be finite and nonzero")
            result = Rotation.from_quat(array).as_matrix()
        else:
            result = array
    if result.shape != (3, 3) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a rotation matrix or scalar-last quaternion")
    if not np.allclose(result.T @ result, np.eye(3), rtol=0.0, atol=1e-10):
        raise ValueError(f"{name} must be orthonormal")
    if np.linalg.det(result) <= 0.0:
        raise ValueError(f"{name} must have positive determinant")
    return result.copy()


def _axis(value, name):
    result = _vector(value, name)
    magnitude = np.linalg.norm(result)
    if magnitude == 0.0:
        raise ValueError(f"{name} must be nonzero")
    return result / magnitude


def _validate_axisymmetric_inertia(inertia, axis, name):
    """Validate the axisymmetric-rotor assumption used without an azimuth state."""

    scale = float(np.linalg.norm(inertia))
    if scale == 0.0:
        return 0.0
    polar = float(axis @ inertia @ axis)
    if not np.allclose(inertia @ axis, polar * axis, rtol=1e-9, atol=1e-10 * scale):
        raise ValueError(f"{name} thrust_axis must be a principal inertia axis")
    trial = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(trial, axis)) > 0.8:
        trial = np.array([0.0, 1.0, 0.0])
    transverse_1 = trial - np.dot(trial, axis) * axis
    transverse_1 /= np.linalg.norm(transverse_1)
    transverse_2 = np.cross(axis, transverse_1)
    moment_1 = float(transverse_1 @ inertia @ transverse_1)
    moment_2 = float(transverse_2 @ inertia @ transverse_2)
    cross_moment = float(transverse_1 @ inertia @ transverse_2)
    if not np.isclose(moment_1, moment_2, rtol=1e-9, atol=1e-10 * scale) or not np.isclose(
        cross_moment, 0.0, rtol=0.0, atol=1e-10 * scale
    ):
        raise ValueError(f"{name} must be axisymmetric about thrust_axis")
    return polar


def _broadcast(value, size, name, minimum=None, strict_minimum=None):
    result = np.asarray(value, dtype=float)
    try:
        result = np.broadcast_to(result, (size,)).astype(float, copy=True)
    except ValueError as error:
        raise ValueError(f"{name} must be a scalar or have one value per rotor") from error
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values")
    if minimum is not None and np.any(result < minimum):
        raise ValueError(f"{name} must be at least {minimum}")
    if strict_minimum is not None and np.any(result <= strict_minimum):
        raise ValueError(f"{name} must be greater than {strict_minimum}")
    return result


def _required(mapping, name, where):
    if name not in mapping:
        raise ValueError(f"{where}.{name} is required")
    return mapping[name]


def _parallel_axis(offset):
    return (np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset))


def _rest_geometry(model, angles):
    """Hub positions and thrust axes about the total COM, and the locked inertia, at tilt ``angles``."""

    orientations = [
        Rotation.from_rotvec(axis * angle).as_matrix() @ zero
        for axis, angle, zero in zip(model["hinge_axes"], angles, model["zero_orientations"])
    ]
    masses = [model["airframe_mass"], *model["assembly_masses"], *model["rotor_masses"]]
    coms = [model["airframe_com"]]
    inertias = [model["airframe_inertia"]]
    for index, orientation in enumerate(orientations):
        coms.append(model["pivots"][index] + orientation @ model["assembly_com_offsets"][index])
        inertias.append(orientation @ model["assembly_inertias"][index] @ orientation.T)
    hubs, axes = [], []
    for index, parent in enumerate(model["rotor_assembly"]):
        orientation, origin = (np.eye(3), np.zeros(3)) if parent < 0 else (orientations[parent], model["pivots"][parent])
        hubs.append(origin + orientation @ model["rotor_hub_offsets"][index])
        axes.append(orientation @ model["rotor_axes_parent"][index])
        coms.append(hubs[-1])
        inertias.append(orientation @ model["rotor_inertias"][index] @ orientation.T)
    masses, coms = np.array(masses), np.array(coms)
    total_com = masses @ coms / np.sum(masses)
    locked = sum(inertia + mass * _parallel_axis(com - total_com) for mass, com, inertia in zip(masses, coms, inertias))
    return np.array(hubs) - total_com, np.array(axes), 0.5 * (locked + locked.T)


def rotor_aero_loads(
    rotor_speeds, axes, hub_airspeeds, airframe_airspeed,
    k_eta, k_m, k_d, k_z, k_h, k_flap, rotor_directions, drag_matrix, aero,
):
    """Per-rotor and airframe aerodynamic loads, all in body components.

    Pure function; ``DrakeMultirotor`` applies it each step.
    ``hub_airspeeds`` (N, 3) are each hub's inertial velocity minus wind and
    ``airframe_airspeed`` (3,) is the airframe COM's.  Returns a dict with

    * ``forces`` (N, 3): thrust + hub drag, applied at each hub;
    * ``reaction_moments`` (N, 3): ``sigma * k_m * Omega*|Omega| * axis``, the rotor
      drag-torque couple (the net airframe yaw moment at steady spin);
    * ``flap_moments`` (N, 3): blade-flapping couples;
    * ``drag`` (3,): airframe parasitic drag, applied at the airframe COM.
    """

    rotor_speeds = np.asarray(rotor_speeds, dtype=float)
    axes = np.asarray(axes, dtype=float)
    hub_airspeeds = np.asarray(hub_airspeeds, dtype=float)
    airframe_airspeed = np.asarray(airframe_airspeed, dtype=float)
    # Omega*|Omega| (== Omega**2 for Omega >= 0): a reversed rotor brakes.
    signed_square = rotor_speeds * np.abs(rotor_speeds)
    thrust = axes * (k_eta * signed_square)[:, None]
    if aero:
        drag = -Multirotor._norm(airframe_airspeed) * drag_matrix @ airframe_airspeed
        axial = np.sum(hub_airspeeds * axes, axis=1)[:, None] * axes
        inplane = hub_airspeeds - axial
        hub_drag = -rotor_speeds[:, None] * (k_d[:, None] * inplane + k_z[:, None] * axial)
        flap = -k_flap[:, None] * rotor_speeds[:, None] * np.cross(hub_airspeeds, axes)
        thrust = thrust + k_h[:, None] * np.sum(inplane**2, axis=1)[:, None] * axes
    else:
        drag = np.zeros(3)
        hub_drag = np.zeros_like(axes)
        flap = np.zeros_like(axes)
    reaction = rotor_directions[:, None] * k_m[:, None] * signed_square[:, None] * axes
    return {
        "forces": thrust + hub_drag,
        "reaction_moments": reaction,
        "flap_moments": flap,
        "drag": drag,
    }


def _rotor_fields(params, specs, n, name, default=_MISSING, minimum=None, strict_minimum=None):
    """Per-rotor ``name``: each rotor spec's own value, else the top-level scalar/array, else ``default``."""

    top = _broadcast(params[name], n, name) if name in params else None
    values = np.empty(n, dtype=float)
    for index, spec in enumerate(specs):
        if name in spec:
            values[index] = spec[name]
        elif top is not None:
            values[index] = top[index]
        elif default is not _MISSING:
            values[index] = default
        else:
            raise ValueError(f"missing required rotor parameter {name}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain finite values")
    if minimum is not None and np.any(values < minimum):
        raise ValueError(f"{name} must be at least {minimum}")
    if strict_minimum is not None and np.any(values <= strict_minimum):
        raise ValueError(f"{name} must be greater than {strict_minimum}")
    return values


def prepare_structured(params, initial_state):
    """Parse the structured parameter form into ``(base_params, model)``.

    ``model`` holds the component arrays the Drake plant is built from;
    ``base_params`` is the flat RotorPy dict (total mass, locked inertia at the
    initial tilt, hub positions about the total COM, per-rotor coefficients).

    Structure: ``airframe`` {``mass``, ``com_offset`` (default 0), ``inertia``};
    ``assemblies``, a list or a name-keyed mapping of {``mass``, ``pivot``,
    ``com_offset``, ``inertia``, ``hinge_axis``, ``zero_orientation`` (default
    identity), ``rate_limit``}; ``rotors`` {``assembly`` (``None`` for the
    airframe, else an index or name), ``hub_offset``, ``thrust_axis`` (default
    +z), ``mass``, ``inertia``}.  Positions and axes are in the parent's frame.
    Rotor coefficients (``k_eta``, ``k_m``, ...) may be given per rotor spec or
    top level.
    """

    airframe = _required(params, "airframe", "params")
    if not isinstance(airframe, Mapping):
        raise ValueError("airframe must be a mapping")
    airframe_mass = float(_required(airframe, "mass", "airframe"))
    if not np.isfinite(airframe_mass) or airframe_mass <= 0.0:
        raise ValueError("airframe.mass must be positive")
    airframe_com = _vector(airframe.get("com_offset", np.zeros(3)), "airframe.com_offset")
    airframe_inertia = _inertia(_required(airframe, "inertia", "airframe"), "airframe.inertia")

    raw_assemblies = params.get("assemblies", [])
    if isinstance(raw_assemblies, Mapping):
        assembly_name_to_index = {name: i for i, name in enumerate(raw_assemblies)}
        assembly_specs = list(raw_assemblies.values())
    else:
        assembly_name_to_index = {}
        assembly_specs = list(raw_assemblies)
    if not all(isinstance(spec, Mapping) for spec in assembly_specs):
        raise ValueError("each assembly must be a mapping")
    k = len(assembly_specs)

    assembly_masses = np.empty(k)
    pivots = np.empty((k, 3))
    assembly_com_offsets = np.empty((k, 3))
    assembly_inertias = np.empty((k, 3, 3))
    hinge_axes = np.empty((k, 3))
    zero_orientations = np.empty((k, 3, 3))
    rate_limit = np.empty(k)
    for index, spec in enumerate(assembly_specs):
        where = f"assemblies[{index}]"
        assembly_masses[index] = float(_required(spec, "mass", where))
        if not np.isfinite(assembly_masses[index]) or assembly_masses[index] < 0.0:
            raise ValueError("assembly.mass must be finite and nonnegative")
        pivots[index] = _vector(_required(spec, "pivot", where), f"{where}.pivot")
        assembly_com_offsets[index] = _vector(_required(spec, "com_offset", where), f"{where}.com_offset")
        assembly_inertias[index] = _inertia(
            _required(spec, "inertia", where), f"{where}.inertia", positive=assembly_masses[index] > 0.0
        )
        if assembly_masses[index] == 0.0 and np.linalg.norm(assembly_inertias[index]) > 0.0:
            raise ValueError("a zero-mass assembly must have zero inertia")
        hinge_axes[index] = _axis(_required(spec, "hinge_axis", where), f"{where}.hinge_axis")
        zero_orientations[index] = _rotation(spec.get("zero_orientation"), f"{where}.zero_orientation")
        rate_limit[index] = float(_required(spec, "rate_limit", where))
    if not np.all(np.isfinite(rate_limit)) or np.any(rate_limit < 0.0):
        raise ValueError("assembly rate_limit must be finite and nonnegative")

    raw_rotors = _required(params, "rotors", "params")
    rotor_specs = list(raw_rotors.values()) if isinstance(raw_rotors, Mapping) else list(raw_rotors)
    if not all(isinstance(spec, Mapping) for spec in rotor_specs):
        raise ValueError("each rotor must be a mapping")
    n = len(rotor_specs)
    if n == 0:
        raise ValueError("at least one rotor is required")
    if "num_rotors" in params and int(params["num_rotors"]) != n:
        raise ValueError("num_rotors and rotors must agree")

    rotor_assembly = np.full(n, -1, dtype=int)
    rotor_hub_offsets = np.empty((n, 3))
    rotor_axes_parent = np.empty((n, 3))
    rotor_masses = np.empty(n)
    rotor_inertias = np.empty((n, 3, 3))
    for index, spec in enumerate(rotor_specs):
        where = f"rotors[{index}]"
        parent = spec.get("assembly")
        if parent is not None:
            parent = assembly_name_to_index.get(parent, parent)
            if not isinstance(parent, (int, np.integer)) or not 0 <= parent < k:
                raise ValueError(f"invalid assembly for rotor {index}")
            rotor_assembly[index] = parent
        rotor_masses[index] = float(spec.get("mass", 0.0))
        if not np.isfinite(rotor_masses[index]) or rotor_masses[index] < 0.0:
            raise ValueError(f"{where}.mass must be finite and nonnegative")
        rotor_inertias[index] = _inertia(
            spec.get("inertia", np.zeros((3, 3))), f"{where}.inertia", positive=rotor_masses[index] > 0.0
        )
        rotor_hub_offsets[index] = _vector(_required(spec, "hub_offset", where), f"{where}.hub_offset")
        rotor_axes_parent[index] = _axis(spec.get("thrust_axis", [0.0, 0.0, 1.0]), f"{where}.thrust_axis")
        if rotor_masses[index] == 0.0 and np.linalg.norm(rotor_inertias[index]) > 0.0:
            raise ValueError("a zero-mass rotor must have zero locked inertia")

    directions = _rotor_fields(params, rotor_specs, n, "rotor_directions")
    if not np.all(np.isin(directions, (-1.0, 1.0))):
        raise ValueError("rotor_directions must contain only -1 or +1")
    k_eta = _rotor_fields(params, rotor_specs, n, "k_eta", strict_minimum=0.0)
    k_m = _rotor_fields(params, rotor_specs, n, "k_m", strict_minimum=0.0)
    k_d = _rotor_fields(params, rotor_specs, n, "k_d", default=0.0, minimum=0.0)
    k_z = _rotor_fields(params, rotor_specs, n, "k_z", default=0.0, minimum=0.0)
    k_h = _rotor_fields(params, rotor_specs, n, "k_h", default=0.0, minimum=0.0)
    k_flap = _rotor_fields(params, rotor_specs, n, "k_flap", default=0.0, minimum=0.0)
    tau_m = _rotor_fields(params, rotor_specs, n, "tau_m", strict_minimum=0.0)
    speed_min = _rotor_fields(params, rotor_specs, n, "rotor_speed_min", minimum=0.0)
    speed_max = _rotor_fields(params, rotor_specs, n, "rotor_speed_max", strict_minimum=0.0)
    if np.any(speed_min >= speed_max):
        raise ValueError("rotor_speed_min must be below rotor_speed_max")
    # The spin inertia is the full rotor inertia's moment about the thrust axis.
    rotor_polar_inertia = np.array([
        _validate_axisymmetric_inertia(rotor_inertias[index], rotor_axes_parent[index], f"rotors[{index}].inertia")
        for index in range(n)
    ])
    if "rotor_inertia" in params and not np.allclose(
        _broadcast(params["rotor_inertia"], n, "rotor_inertia"), rotor_polar_inertia, rtol=1e-9, atol=0.0
    ):
        raise ValueError("rotor_inertia must match the spin-axis moment of the full rotor inertia")

    initial_angles = np.asarray(
        (initial_state or {}).get("tilt_angles", params.get("initial_tilt_angles", np.zeros(k))), dtype=float
    )
    initial_rates = np.asarray(
        (initial_state or {}).get("tilt_rates", params.get("initial_tilt_rates", np.zeros(k))), dtype=float
    )
    if initial_angles.shape != (k,) or initial_rates.shape != (k,):
        raise ValueError("initial tilt angles and rates must have one value per assembly")
    if not np.all(np.isfinite(initial_angles)) or not np.all(np.isfinite(initial_rates)):
        raise ValueError("initial tilt angles and rates must be finite")
    if np.any(np.abs(initial_rates) > rate_limit):
        raise ValueError("initial tilt rates must be within rate limits")

    total_mass = float(airframe_mass + np.sum(assembly_masses) + np.sum(rotor_masses))
    model = {
        "airframe_mass": airframe_mass,
        "airframe_com": airframe_com,
        "airframe_inertia": airframe_inertia,
        "assembly_masses": assembly_masses,
        "pivots": pivots,
        "assembly_com_offsets": assembly_com_offsets,
        "assembly_inertias": assembly_inertias,
        "hinge_axes": hinge_axes,
        "zero_orientations": zero_orientations,
        "rate_limit": rate_limit,
        "total_mass": total_mass,
        "rotor_assembly": rotor_assembly,
        "rotor_hub_offsets": rotor_hub_offsets,
        "rotor_axes_parent": rotor_axes_parent,
        "rotor_masses": rotor_masses,
        "rotor_inertias": rotor_inertias,
        "rotor_directions": directions,
        "rotor_polar_inertia": rotor_polar_inertia,
        "rotor_count": n,
        "initial_angles": initial_angles,
        "initial_rates": initial_rates,
    }
    hubs, axes, inertia = _rest_geometry(model, initial_angles)
    base_params = dict(params)
    base_params.update(
        {
            "mass": total_mass,
            "num_rotors": n,
            "rotor_pos": {f"r{index + 1}": hubs[index] for index in range(n)},
            "rotor_directions": directions,
            "rotor_thrust_axes": axes,
            "k_eta": k_eta,
            "k_m": k_m,
            "k_d": k_d,
            "k_z": k_z,
            "k_h": k_h,
            "k_flap": k_flap,
            "tau_m": tau_m,
            "rotor_speed_min": speed_min,
            "rotor_speed_max": speed_max,
            "rotor_inertia": rotor_polar_inertia,
            "Ixx": inertia[0, 0],
            "Iyy": inertia[1, 1],
            "Izz": inertia[2, 2],
            "Ixy": inertia[0, 1],
            "Ixz": inertia[0, 2],
            "Iyz": inertia[1, 2],
        }
    )
    return base_params, model


def complete_state(state, n, k, initial_angles, initial_rates, params):
    """Fill tilt defaults into a RotorPy state dict and validate its shapes."""

    if state is None:
        rotor_speeds = params.get("initial_rotor_speeds", np.full(n, 1788.53))
        state = {
            "x": np.zeros(3),
            "v": np.zeros(3),
            "q": np.array([0.0, 0.0, 0.0, 1.0]),
            "w": np.zeros(3),
            "wind": np.zeros(3),
            "rotor_speeds": rotor_speeds,
            "tilt_angles": initial_angles,
            "tilt_rates": initial_rates,
        }
    result = dict(state)
    defaults = {
        "tilt_angles": initial_angles,
        "tilt_rates": initial_rates,
    }
    for key, default in defaults.items():
        result[key] = np.asarray(result.get(key, default), dtype=float).copy()
    expected = {
        "x": (3,),
        "v": (3,),
        "q": (4,),
        "w": (3,),
        "wind": (3,),
        "rotor_speeds": (n,),
        "tilt_angles": (k,),
        "tilt_rates": (k,),
    }
    for key, shape in expected.items():
        if key not in result:
            raise ValueError(f"initial_state is missing {key}")
        result[key] = np.asarray(result[key], dtype=float).copy()
        if result[key].shape != shape or not np.all(np.isfinite(result[key])):
            raise ValueError(f"initial_state[{key}] must be finite with shape {shape}")
    if np.linalg.norm(result["q"]) == 0.0:
        raise ValueError("initial_state[q] must be nonzero")
    return result


# Keys of a legacy (vanilla / MultirotorExtended) dict that the structured
# form replaces with its airframe and rotor specs.
_LEGACY_ONLY_KEYS = (
    "mass", "Ixx", "Iyy", "Izz", "Ixy", "Ixz", "Iyz", "inertia",
    "rotor_pos", "rotor_thrust_axes", "rotor_inertia",
)


def structured_from_legacy(quad_params, rotor_mass=0.0, rotor_inertia=None, rotor_frame=False):
    """Structured K=0 parameters for a legacy ``quad_params`` rigid body.

    Converts a vanilla ``Multirotor`` / ``MultirotorExtended`` dict
    (``mass``, ``I**``, ``rotor_pos``, optional ``rotor_thrust_axes``,
    scalar or per-rotor coefficients, optional ``rotor_inertia``) into the
    structured form (see ``prepare_structured``): an ``airframe``, no
    assemblies and one rotor spec per ``rotor_pos`` entry, in the same order.

    The legacy mass, COM (the body origin) and inertia describe the *whole*
    vehicle, so they are kept as totals: each rotor gets ``rotor_mass`` (kg,
    scalar or per rotor) and the central inertia ``diag(I_t, I_t, J_p)`` about
    its thrust axis, and the airframe gets what is left.  Its mass is
    ``mass - sum(rotor_mass)``, its COM is offset so the total COM stays at the
    origin, and its central inertia is the legacy inertia minus the rotors'
    central and parallel-axis terms and its own parallel-axis term.  The
    locked total mass, COM and inertia therefore equal the legacy ones.

    ``J_p`` is ``rotor_inertia`` (scalar or per rotor) when given, else the
    legacy ``quad_params['rotor_inertia']``, else 0.  ``I_t = J_p / 2`` (thin
    disc), which keeps the rotor inside the triangle inequalities.  A rotor
    with ``J_p > 0`` needs ``rotor_mass > 0`` (the parser rejects a massless
    rotor that has inertia).

    Rotor inertias are in the parent (airframe) frame by default, as
    ``prepare_structured`` takes them.  ``rotor_frame=True`` returns them in each
    rotor's own frame (z = spin axis), as ``DrakeMultirotor`` expects.  For an
    axisymmetric rotor the parent-frame form is
    ``I_t (1 - a a^T) + J_p a a^T`` with ``a`` the thrust axis, so both are
    independent of the choice of the rotor's transverse axes.

    Raises ``ValueError`` when the legacy inertia is not a physical rigid-body
    inertia (``crazyflie_params`` violates the triangle inequality:
    Ixx + Iyy < Izz), or when removing the rotors leaves an airframe with
    non-positive mass or a non-physical inertia.
    """

    rotor_pos = quad_params["rotor_pos"]
    if not isinstance(rotor_pos, Mapping):
        raise ValueError("legacy rotor_pos must be a mapping")
    hubs = np.array([_vector(rotor_pos[key], f"rotor_pos[{key}]") for key in rotor_pos])
    n = len(hubs)
    axes = np.asarray(quad_params.get("rotor_thrust_axes", np.tile([0.0, 0.0, 1.0], (n, 1))), dtype=float)
    if axes.shape != (n, 3) or np.any(np.linalg.norm(axes, axis=1) == 0.0):
        raise ValueError("rotor_thrust_axes must have one nonzero vector per rotor")
    axes = axes / np.linalg.norm(axes, axis=1)[:, None]

    total_inertia = np.array(
        [
            [quad_params["Ixx"], quad_params["Ixy"], quad_params["Ixz"]],
            [quad_params["Ixy"], quad_params["Iyy"], quad_params["Iyz"]],
            [quad_params["Ixz"], quad_params["Iyz"], quad_params["Izz"]],
        ],
        dtype=float,
    )
    try:
        _inertia(total_inertia, "legacy inertia")
    except ValueError as error:
        raise ValueError(f"legacy quad_params do not describe a physical rigid body: {error}") from error

    masses = _broadcast(rotor_mass, n, "rotor_mass", minimum=0.0)
    polar = _broadcast(
        quad_params.get("rotor_inertia", 0.0) if rotor_inertia is None else rotor_inertia,
        n, "rotor_inertia", minimum=0.0,
    )
    if np.any((masses == 0.0) & (polar > 0.0)):
        raise ValueError("a rotor with rotor_inertia > 0 needs rotor_mass > 0")
    transverse = 0.5 * polar

    airframe_mass = float(quad_params["mass"]) - float(np.sum(masses))
    if airframe_mass <= 0.0:
        raise ValueError("rotor masses leave no airframe mass")
    airframe_com = -(masses @ hubs) / airframe_mass
    airframe_inertia = total_inertia - airframe_mass * _parallel_axis(airframe_com)
    rotors = []
    for index in range(n):
        own = np.diag([transverse[index], transverse[index], polar[index]])
        axis = axes[index]
        parent = transverse[index] * (np.eye(3) - np.outer(axis, axis)) + polar[index] * np.outer(axis, axis)
        airframe_inertia -= parent + masses[index] * _parallel_axis(hubs[index])
        rotors.append(
            {
                "assembly": None,
                "hub_offset": hubs[index].copy(),
                "thrust_axis": axis.copy(),
                "mass": float(masses[index]),
                "inertia": own if rotor_frame else parent,
            }
        )
    airframe_inertia = 0.5 * (airframe_inertia + airframe_inertia.T)
    try:
        _inertia(airframe_inertia, "airframe inertia")
    except ValueError as error:
        raise ValueError(
            f"subtracting the rotors from the legacy inertia leaves a non-physical airframe: {error}"
        ) from error

    result = {key: value for key, value in quad_params.items() if key not in _LEGACY_ONLY_KEYS}
    result.update(
        {
            "num_rotors": n,
            "airframe": {"mass": airframe_mass, "com_offset": airframe_com, "inertia": airframe_inertia},
            "assemblies": [],
            "rotors": rotors,
        }
    )
    return result
