"""Typed vehicle description, the parts ``DrakeMultirotor`` is built from."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import numpy as np

from rotorpy.vehicles.models.actuator import Actuator, FirstOrderLag
from rotorpy.vehicles.models.aero import AirframeAeroModel
from rotorpy.vehicles.models.frame_drag import FrameDrag
from rotorpy.vehicles.models.powered_aero.rotor import PolynomialRotorModel
from rotorpy.vehicles.models.rotor_model import RotorModel
from rotorpy.vehicles.types import InertiaTensor, Mass, Position


def _arrays(part, *names):
    for name in names:
        object.__setattr__(part, name, np.asarray(getattr(part, name), dtype=float))


def parallel_axis(offset) -> InertiaTensor:
    """``|r|^2 I - r r^T``: the parallel-axis term per unit mass."""

    offset = np.asarray(offset, dtype=float)
    return np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset)


@dataclass(frozen=True, kw_only=True)
class ControlSurface:
    """A control surface: its ``deflection`` actuator (rad, trailing-edge-down positive).  Its aerodynamics are the
    airframe model's (``C + delta C``), which reads the actual deflection."""

    name: str
    deflection: Actuator


@dataclass(frozen=True, kw_only=True)
class Airframe:
    """The fixed body: ``mass`` (kg), ``com_offset`` from the airframe origin, central ``inertia``; its control
    ``surfaces`` and its aerodynamic model ``aero`` (``None``: no airframe aerodynamics).
    
    On any standard 6-DOF code, this class and it's surfaces would be enough to model the aircraft 
    """

    mass: Mass
    inertia: InertiaTensor
    com_offset: Position = field(default_factory=lambda: np.zeros(3))
    aero: Optional[AirframeAeroModel] = None
    surfaces: Sequence[ControlSurface] = ()  # stored as a tuple

    def __post_init__(self):
        _arrays(self, "inertia", "com_offset")
        object.__setattr__(self, "surfaces", tuple(self.surfaces))


@dataclass(frozen=True, kw_only=True)
class TiltAssembly:
    """A rigid body on a one-DOF hinge at ``pivot`` (body frame), turning about the unit ``hinge_axis``.

    ``com_offset`` (from the pivot) and ``inertia`` are in the zero-tilt frame, which is the body frame rotated by
    ``zero_orientation``.  A hinge rate beyond ``rate_limit`` (rad/s) stops the run.  The hinge is commanded in
    ``rate`` or ``torque``.
    """

    mass: Mass
    inertia: InertiaTensor
    pivot: Position
    hinge_axis: Position
    rate_limit: float
    com_offset: Position = field(default_factory=lambda: np.zeros(3))
    zero_orientation: np.ndarray = field(default_factory=lambda: np.eye(3))
    rate: Actuator = Actuator(model=FirstOrderLag(0.05))
    torque: Actuator = Actuator()

    def __post_init__(self):
        _arrays(self, "inertia", "pivot", "hinge_axis", "com_offset", "zero_orientation")


@dataclass(frozen=True, kw_only=True)
class RotorSpec:
    """One rotor: its place, its rigid body, its aerodynamic model and its motor.

    * ``name``: how aerodynamic data refers to it;
    * ``assembly``: index of the tilting assembly it rides on, ``None`` for the airframe;
    * ``hub_offset``, ``thrust_axis``: hub (the rotor COM) and unit thrust axis in the parent frame (see module);
    * ``direction``: +1 or -1, RotorPy's ``rotor_directions``;
    * ``mass`` (kg), ``spin_inertia`` ``J_p`` about the axis and ``transverse_inertia`` ``I_t`` across it
      (kg*m^2): an axisymmetric rotor.  ``DrakeMultirotor`` needs a massive rotor, since its spin is a joint;
    * ``radius`` (m): the disk, for models that need it (the powered-aero wake); ``None`` when unknown;
    * ``model``: the rotor's loads from its inflow (:class:`~rotorpy.vehicles.models.rotor_model.RotorModel`), e.g.
      RotorPy's rotor ``PolynomialRotorModel.rotorpy(k_eta=..., k_m=...)`` or a fitted polynomial;
    * ``speed``: the motor, rad/s: its speed limits and lag, e.g. ``Actuator((0.0, 2000.0), FirstOrderLag(0.05))``.
    """

    hub_offset: Position
    direction: float
    model: RotorModel
    speed: Actuator
    assembly: Optional[int] = None
    thrust_axis: Position = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    mass: Mass = 0.0
    spin_inertia: float = 0.0
    transverse_inertia: float = 0.0
    radius: Optional[float] = None
    name: str = ""

    def __post_init__(self):
        _arrays(self, "hub_offset", "thrust_axis")

    @property
    def area(self) -> float:
        """Disk area ``pi r^2``."""

        return np.pi * self.radius**2

    @property
    def inertia(self) -> InertiaTensor:
        """Locked central inertia in the parent frame, ``I_t (1 - a a^T) + J_p a a^T`` (zero when massless)."""

        if self.mass == 0.0:
            return np.zeros((3, 3))
        axial = np.outer(self.thrust_axis, self.thrust_axis)
        return self.transverse_inertia * (np.eye(3) - axial) + self.spin_inertia * axial


@dataclass(frozen=True, kw_only=True)
class VehicleSpec:
    """The whole vehicle."""

    airframe: Airframe
    assemblies: Sequence[TiltAssembly] = ()  # stored as tuples
    rotors: Sequence[RotorSpec] = ()

    def __post_init__(self):
        object.__setattr__(self, "assemblies", tuple(self.assemblies))
        object.__setattr__(self, "rotors", tuple(self.rotors))

    @property
    def num_rotors(self) -> int:
        return len(self.rotors)

    @property
    def num_assemblies(self) -> int:
        return len(self.assemblies)

    @property
    def total_mass(self) -> Mass:
        return (self.airframe.mass + sum(assembly.mass for assembly in self.assemblies)
                + sum(rotor.mass for rotor in self.rotors))

    def rotor_values(self, name) -> np.ndarray:
        """One numeric rotor field over the rotors, e.g. ``rotor_values("direction")``."""

        return np.array([getattr(rotor, name) for rotor in self.rotors], dtype=float)

    def with_rotors(self, **values) -> VehicleSpec:
        """A copy with rotor fields replaced, each value one for all rotors or one per rotor:
        ``spec.with_rotors(model=PolynomialRotorModel.rotorpy(k_eta=1e-5, k_m=1e-7), mass=[0.01, 0.01, 0.005, 0.005])``."""

        def column(value):
            per_rotor = isinstance(value, (list, tuple, np.ndarray)) and len(value) == self.num_rotors
            return list(value) if per_rotor else [value] * self.num_rotors

        columns = {name: column(value) for name, value in values.items()}
        return replace(self, rotors=tuple(
            replace(rotor, **{name: column[index] for name, column in columns.items()})
            for index, rotor in enumerate(self.rotors)))

    @classmethod
    def from_rotorpy(cls, quad_params, rotor_mass=0.0, rotor_inertia=None) -> VehicleSpec:
        """The spec of a RotorPy ``quad_params`` rigid body (``Multirotor`` / ``MultirotorExtended``).

        The legacy mass, COM (the body origin) and inertia describe the *whole* vehicle and stay the totals: each
        rotor gets ``rotor_mass`` (kg, scalar or per rotor) and, when massive, the thin-disc inertia
        ``I_t = J_p / 2``; the airframe gets what is left, its COM offset so the total COM stays at the origin.
        ``J_p`` is ``rotor_inertia`` when given, else ``quad_params['rotor_inertia']``, else 0.  Each rotor's model is RotorPy's rotor,
        :meth:`PolynomialRotorModel.rotorpy`, from its ``k_*``, and its motor RotorPy's first-order ``tau_m`` lag
        within ``[rotor_speed_min, rotor_speed_max]``; its radius is half ``rotor_diameter`` when given.  The
        airframe's model is RotorPy's frame drag (:class:`FrameDrag`), none when every ``c_D*`` is 0.

        Nothing is checked: ``crazyflie_params`` (``Ixx + Iyy < Izz``) gives a non-physical airframe.
        """

        keys = list(quad_params["rotor_pos"])
        n = len(keys)
        hubs = np.array([quad_params["rotor_pos"][key] for key in keys], dtype=float)
        axes = np.asarray(quad_params.get("rotor_thrust_axes", np.tile([0.0, 0.0, 1.0], (n, 1))), dtype=float)

        def per_rotor(value):
            return np.broadcast_to(np.asarray(value, dtype=float), (n,)).copy()

        total_inertia = np.array([
            [quad_params["Ixx"], quad_params["Ixy"], quad_params["Ixz"]],
            [quad_params["Ixy"], quad_params["Iyy"], quad_params["Iyz"]],
            [quad_params["Ixz"], quad_params["Iyz"], quad_params["Izz"]],
        ], dtype=float)
        masses = per_rotor(rotor_mass)
        polar = per_rotor(quad_params.get("rotor_inertia", 0.0) if rotor_inertia is None else rotor_inertia)
        transverse = np.where(masses > 0.0, 0.5 * polar, 0.0)
        tau_m, speed_min, speed_max = (per_rotor(quad_params[name])
                                       for name in ("tau_m", "rotor_speed_min", "rotor_speed_max"))
        rotors = tuple(
            RotorSpec(
                name=str(key), hub_offset=hubs[index], thrust_axis=axes[index],
                direction=float(per_rotor(quad_params["rotor_directions"])[index]),
                mass=masses[index], spin_inertia=polar[index], transverse_inertia=transverse[index],
                model=PolynomialRotorModel.rotorpy(**{name: float(per_rotor(quad_params.get(name, 0.0))[index])
                                                      for name in ("k_eta", "k_m", "k_d", "k_z", "k_h", "k_flap")}),
                speed=Actuator((float(speed_min[index]), float(speed_max[index])), FirstOrderLag(float(tau_m[index]))),
                radius=(0.5 * per_rotor(quad_params["rotor_diameter"])[index]
                        if "rotor_diameter" in quad_params else None),
            )
            for index, key in enumerate(keys)
        )

        airframe_mass = float(quad_params["mass"]) - float(np.sum(masses))
        airframe_com = -(masses @ hubs) / airframe_mass
        airframe_inertia = total_inertia - airframe_mass * parallel_axis(airframe_com)
        for rotor in rotors:
            airframe_inertia = airframe_inertia - rotor.inertia - rotor.mass * parallel_axis(rotor.hub_offset)
        # Legacy RotorPy models: their per-axis frame drag (c_Dx, c_Dy, c_Dz) is the airframe's aero model.
        c_D = np.array([quad_params.get(name, 0.0) for name in ("c_Dx", "c_Dy", "c_Dz")], dtype=float)
        airframe = Airframe(mass=airframe_mass, com_offset=airframe_com,
                            inertia=0.5 * (airframe_inertia + airframe_inertia.T),
                            aero=FrameDrag(c_D, airframe_com) if c_D.any() else None)
        return cls(airframe=airframe, rotors=rotors)


def complete_state(state, num_rotors, num_assemblies):
    """A RotorPy state dict with the tilt keys filled (zeros); a fresh copy."""

    result = {key: np.array(value, dtype=float) for key, value in state.items()}
    result.setdefault("tilt_angles", np.zeros(num_assemblies))
    result.setdefault("tilt_rates", np.zeros(num_assemblies))
    return result


def hover_state(spec: VehicleSpec, rotor_speeds=None):
    """At rest at the origin, level, rotors at ``rotor_speeds`` (default 0), tilts at 0."""

    return {
        "x": np.zeros(3), "v": np.zeros(3), "q": np.array([0.0, 0.0, 0.0, 1.0]), "w": np.zeros(3),
        "wind": np.zeros(3),
        "rotor_speeds": np.zeros(spec.num_rotors) if rotor_speeds is None else np.array(rotor_speeds, dtype=float),
        "tilt_angles": np.zeros(spec.num_assemblies), "tilt_rates": np.zeros(spec.num_assemblies),
    }
