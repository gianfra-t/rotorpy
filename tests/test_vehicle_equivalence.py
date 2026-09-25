"""MultirotorExtended and DrakeMultirotor fly like vanilla Multirotor.

The case is the hummingbird example made as hard as vanilla Multirotor can express: a full inertia tensor, every aero term on,
 asymmetric non-planar arms, per-rotor flapping, etc.

Both vehicles fly the same closed loop: SE3Control, on a circle with a vertical oscillation and a yaw sinusoid, in
a time-varying wind. 

Integrators are tight (DOP853, rtol 1e-11, atol 1e-12; Drake runge_kutta5 at accuracy 1e-10), so Extended is
held to the integrator floor.

Drake implementation cannot fly the exact vehicle: a rotor spin joint without inertia is singular, so it carries a tiny rotor
(J_p = 1e-9 Izz, 1e-6 kg). Its tolerances are set by that rotor, not by the model. Still, it manages to closely follow the vanilla Multirotor's behavior.
"""

import copy
import importlib.util
from functools import lru_cache

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.estimators.nullestimator import NullEstimator
from rotorpy.sensors.external_mocap import MotionCapture
from rotorpy.sensors.imu import Imu
from rotorpy.simulate import simulate
from rotorpy.trajectories.circular_traj import ThreeDCircularTraj
from rotorpy.vehicles.hummingbird_params import quad_params as hummingbird_params
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.multirotor_extended import MultirotorExtended
from rotorpy.vehicles.multirotor_params import structured_from_legacy
from rotorpy.vehicles.drake.multirotor import DrakeMultirotor
from rotorpy.wind.default_winds import SinusoidWind
from rotorpy.world import World

DT = 0.01
T_FINAL = 3.0
TIGHT = {"method": "DOP853", "rtol": 1e-11, "atol": 1e-12}
# state comparisson tolerances
TOLERANCE = {"x": 1e-9, "v": 1e-9, "q": 1e-10, "w": 1e-9, "rotor_speeds": 1e-8}
# command comparisson tolerance, relative to each command's peak (they span 1 for cmd_q to ~850 rad/s)
COMMAND_RTOL = 1e-10
# Drake: about 3x the measured tiny-rotor effect (see the module docstring)
DRAKE = {"integrator": "runge_kutta5", "accuracy": 1e-10}
DRAKE_TOLERANCE = {"x": 5e-8, "v": 3e-7, "q": 6e-8, "w": 3e-6, "rotor_speeds": 3e-3}
DRAKE_COMMAND_RTOL = 4e-6
DRAKE_ROTOR_MASS, DRAKE_ROTOR_JP = 1e-6, 1e-9 * hummingbird_params["Izz"]

# -----------------  the case  --------------------------------------------------------- 
PARAMS = copy.deepcopy(hummingbird_params)
PARAMS.update(
    # Products of inertia 11-14 % of the diagonal.
    Ixy=5.0e-4, Ixz=-4.0e-4, Iyz=4.0e-4,
    # Arms 0.13-0.19 m, all four different; hubs 1-3 cm above/below the COM.
    rotor_pos={
        "r1": np.array([0.13, 0.11, 0.020]),
        "r2": np.array([0.12, -0.14, -0.015]),
        "r3": np.array([-0.15, -0.12, 0.030]),
        "r4": np.array([-0.11, 0.16, -0.010]),
    },
    # Per rotor, so a rotor index mix-up cannot cancel.
    k_flap=np.array([1.0e-5, 1.2e-5, 0.8e-5, 1.1e-5]),
    tau_m=np.array([0.005, 0.007, 0.006, 0.008]),

    rotor_speed_max=655.0,
    c_Dx=0.6e-2, c_Dy=0.8e-2, c_Dz=1.2e-2,
)
TRAJECTORY = dict(center=np.array([0.0, 0.0, 1.0]), radius=np.array([1.0, 1.0, 0.3]),
                  freq=np.array([0.35, 0.35, 0.175]), yaw_bool=True)
WIND = dict(amplitudes=np.array([2.0, -1.5, 1.0]), frequencies=np.array([0.5, 0.7, 0.9]),
            phase=np.array([0.0, 0.3, 0.1]))
# Start on a level position, at hover speed, on the trajectory's start point with its velocity and yaw rate .
START = ThreeDCircularTraj(**TRAJECTORY).update(0.0)
HOVER_SPEED = np.sqrt(PARAMS["mass"] * 9.81 / (4 * PARAMS["k_eta"]))
INITIAL_STATE = {"x": START["x"], "v": START["x_dot"], "q": np.array([0.0, 0.0, 0.0, 1.0]),
                 "w": np.array([0.0, 0.0, START["yaw_dot"]]), "wind": np.zeros(3),
                 "rotor_speeds": np.full(4, HOVER_SPEED)}


def make(impl):
    if impl == "vanilla":
        return Multirotor(PARAMS, aero=True, integrator_kwargs=TIGHT)
    if impl == "extended":
        return MultirotorExtended(PARAMS, aero=True, integrator_kwargs=TIGHT)
    if impl == "drake":
        legacy = {**PARAMS, "rotor_inertia": DRAKE_ROTOR_JP}
        return DrakeMultirotor(structured_from_legacy(legacy, DRAKE_ROTOR_MASS, rotor_frame=True), aero=True, **DRAKE)
    raise ValueError(impl)


@lru_cache(maxsize=None)
def fly(impl):

    vehicle = make(impl)
    np.random.seed(0)  # the IMU and mocap are noisy; they are recorded, never fed back
    time, state, control, flat, *_ = simulate(
        World.empty((-9, 9, -9, 9, -9, 9)), copy.deepcopy(INITIAL_STATE), vehicle, SE3Control(PARAMS),
        ThreeDCircularTraj(**TRAJECTORY), SinusoidWind(**WIND), Imu(sampling_rate=1 / DT),
        MotionCapture(sampling_rate=1 / DT, with_artifacts=False), NullEstimator(),
        T_FINAL, DT, 0.25, use_mocap=False, terminate=False,
    )
    return time, state, control, flat


@pytest.mark.parametrize("impl, tolerances, command_rtol", [
    pytest.param("extended", TOLERANCE, COMMAND_RTOL, id="extended"),
    pytest.param("drake", DRAKE_TOLERANCE, DRAKE_COMMAND_RTOL, id="drake", marks=pytest.mark.skipif(
        importlib.util.find_spec("pydrake") is None, reason="DrakeMultirotor needs pydrake")),
])
def test_flies_like_vanilla(impl, tolerances, command_rtol):
    time, vanilla, commands, flat = fly("vanilla")
    other_time, other, other_commands, _ = fly(impl)

    # The flight is hard enough to mean something (~35 deg bank, the speed limit reached on ~14 % of the steps)
    # and still under control (~0.13 m tracking).
    tilt = np.degrees(np.arccos(Rotation.from_quat(vanilla["q"]).as_matrix()[:, 2, 2]))
    assert tilt.max() > 25.0
    assert np.any(commands["cmd_motor_speeds"] >= PARAMS["rotor_speed_max"])
    assert np.max(np.linalg.norm(vanilla["x"] - flat["x"], axis=1)) < 0.25

    # Same clock. The flat outputs are the trajectory evaluated on it, so they need no comparison of their own.
    np.testing.assert_array_equal(other_time, time)
    for key, tolerance in tolerances.items():
        np.testing.assert_allclose(other[key], vanilla[key], rtol=0, atol=tolerance, err_msg=key)
    for key, command in commands.items():
        bound = command_rtol * np.max(np.abs(command))
        np.testing.assert_allclose(other_commands[key], command, rtol=0, atol=bound, err_msg=key)
