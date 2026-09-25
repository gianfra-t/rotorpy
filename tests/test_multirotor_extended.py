'''
Test the MultirotorExtended vehicle class: dissimilar, canted and non-planar rotors driven by
dissimilar motors (per-rotor lag, speed limits and rotor inertia) using a set of self-contained 
tests that validates against analytical results or directional expectations.
'''

import copy

import numpy as np

from rotorpy.vehicles.hummingbird_params import quad_params
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.multirotor_extended import MultirotorExtended


airframe_2kg = {
    'mass': 2.0,        
    'Ixx':  0.035,      
    'Iyy':  0.070,      
    'Izz':  0.100,
    'Ixy':  0.0,        
    'Iyz':  0.0,        
    'Ixz':  0.0,        
    'rI': np.array([0, 0, 0]),
}

x_main = 0.15   # m ahead of the CoM
x_tail = 0.30   # m behind the CoM


def cant_axes(cant_degrees):
    """Thrust axes for [main, main, tail y-, tail y+], the tail pair canted outward by cant_degrees."""
    cant = np.deg2rad(cant_degrees)
    return np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, -np.sin(cant), np.cos(cant)],
        [0.0, np.sin(cant), np.cos(cant)],
    ])


def make_params(cant_degrees=5.0):
    return {
        **airframe_2kg,

        'num_rotors': 4,
        'rotor_pos': {
            'main_left':  np.array([ x_main,  0.30, 0.04]),
            'main_right': np.array([ x_main, -0.30, 0.04]),
            'tail_right': np.array([-x_tail, -0.10, 0.02]),
            'tail_left':  np.array([-x_tail,  0.10, 0.02]),
        },
        'rotor_directions': np.array([1, -1, 1, -1]),
        'rotor_thrust_axes': cant_axes(cant_degrees),
        'rotor_diameter': np.array([0.508, 0.508, 0.152, 0.152]), 

        'k_eta': np.array([1.860e-4, 1.860e-4, 1.841e-6, 1.841e-6]),
        'k_m':   np.array([6.683e-6, 6.683e-6, 2.030e-8, 2.030e-8]),   

        'k_d':   np.array([1.95e-3, 1.95e-3, 5.3e-5, 5.3e-5]),         
        'k_z':   np.array([3.80e-3, 3.80e-3, 1.03e-4, 1.03e-4]), 
        'k_h':   0.0,
        'k_flap': 0.0,
        'c_Dx':  0.012,     
        'c_Dy':  0.012,     
        'c_Dz':  0.025,     

        'tau_m':           np.array([0.10, 0.10, 0.025, 0.025]),    
        'rotor_speed_min': np.array([0.0, 0.0, 0.0, 0.0]),
        'rotor_speed_max': np.array([350.0, 350.0, 2600.0, 2600.0]),   
        'rotor_inertia':   np.array([9.0e-4, 9.0e-4, 1.2e-5, 1.2e-5]),
        'motor_noise_std': 0.0,
    }


def params_without_rotor_inertia(cant_degrees=5.0):
    params = make_params(cant_degrees)
    del params['rotor_inertia']
    return params


def make_state(rotor_speeds, position=(0.0, 0.0, 1.0)):
    """A level, motionless state at ``position`` with the rotors already turning at ``rotor_speeds``."""
    return {
        'x': np.array(position, dtype=float),
        'v': np.zeros(3),
        'q': np.array([0.0, 0.0, 0.0, 1.0]),
        'w': np.zeros(3),
        'wind': np.zeros(3),
        'rotor_speeds': np.asarray(rotor_speeds, dtype=float).copy(),
    }


def solve_static_hover_trim(vehicle):
    """Rotor speeds holding full static equilibrium at identity attitude.

    The allocation matrix is inverted for the thrusts that carry the weight with no body moment, then the
    resulting wrench is recomputed from the vehicle itself, so a vehicle whose allocation does not describe
    its own dynamics is rejected rather than silently trimmed.
    """
    target_wrench = np.array([vehicle.mass * vehicle.g, 0.0, 0.0, 0.0])
    try:
        thrusts = np.linalg.solve(vehicle.f_to_TM, target_wrench)
    except np.linalg.LinAlgError as error:
        raise ValueError("Hover trim requires a nonsingular allocation matrix") from error
    if np.any(thrusts < -1e-12):
        raise ValueError("Hover trim requires negative rotor thrust")

    trim_speeds = np.sqrt(np.maximum(thrusts, 0.0) / vehicle.k_eta)
    if np.any(trim_speeds < vehicle.rotor_speed_min) or np.any(trim_speeds > vehicle.rotor_speed_max):
        raise ValueError("Hover trim exceeds motor-speed limits")

    force, moment = vehicle.compute_body_wrench(np.zeros(3), trim_speeds, np.zeros(3))
    if not np.allclose(vehicle.weight + force, np.zeros(3), atol=1e-9) or not np.allclose(
        moment, np.zeros(3), atol=1e-9
    ):
        raise ValueError("Hover trim does not satisfy full static equilibrium")
    return trim_speeds




def test_aero_terms_follow_each_rotor_axis():
    """Rotor drag, translational lift and flapping use each rotor's own axis.
    
    With every rotor canted 45 degrees towards x, a body-z airspeed splits into equal axial and in-plane
    parts, so each term is non-zero and checked numerically against its closed form.
    """
    k_d, k_z, k_h, k_flap, speed = 2e-4, 5e-4, 3e-3, 1e-5, 500.0
    params = make_params(0)
    params.update({'k_d': k_d, 'k_z': k_z, 'k_h': k_h, 'k_flap': 0.0, 'c_Dx': 0.0, 'c_Dy': 0.0, 'c_Dz': 0.0})
    axis = np.array([1.0, 0.0, 1.0]) / np.sqrt(2)
    params['rotor_thrust_axes'] = np.tile(axis, (4, 1))  # every rotor tilted 45 degrees toward body x
    vehicle = MultirotorExtended(params, aero=True)
    speeds = np.full(4, speed)
    airspeed = np.array([0.0, 0.0, 2.0])

    # Body-z airspeed splits into an axial part (1, 0, 1) and an in-plane part (-1, 0, 1).
    axial, inplane = np.array([1.0, 0.0, 1.0]), np.array([-1.0, 0.0, 1.0])
    force, _ = vehicle.compute_body_wrench(np.zeros(3), speeds, airspeed)
    thrust = np.sum(vehicle.k_eta * speeds**2) + 4 * k_h * inplane @ inplane
    expected = thrust * axis - 4 * speed * (k_d * inplane + k_z * axial)
    np.testing.assert_allclose(force, expected)

    # Flapping acts about v x axis.
    vehicle.k_d = vehicle.k_z = vehicle.k_h = np.zeros(4)
    _, moment_dry = vehicle.compute_body_wrench(np.zeros(3), speeds, np.zeros(3))
    vehicle.k_flap = np.full(4, k_flap)
    _, moment = vehicle.compute_body_wrench(np.zeros(3), speeds, airspeed)
    np.testing.assert_allclose(moment - moment_dry, -4 * k_flap * speed * np.cross(airspeed, axis), atol=1e-12)


def test_rotor_inertia_moments():
    vehicle = MultirotorExtended(make_params(5), aero=False)
    reference = MultirotorExtended(params_without_rotor_inertia(5), aero=False)
    np.testing.assert_array_equal(reference.rotor_inertia, np.zeros(4))
    speeds = np.array([180.0, 190.0, 1300.0, 1350.0])
    command = np.array([200.0, 170.0, 1400.0, 1250.0])
    w = np.array([0.3, -0.2, 0.5])
    state = make_state(speeds)
    state['w'] = w
    s = vehicle._pack_state(state)

    extra_wdot = (vehicle._s_dot_fn(0, s, command) - reference._s_dot_fn(0, s, command))[10:13]
    rotor_accel = (command - speeds) / vehicle.tau_m
    # Rotor i spins about -rotor_dir[i] * axis_i, so that is the direction of its angular momentum.
    h = -np.sum((vehicle.rotor_dir * vehicle.rotor_inertia * speeds)[:, np.newaxis] * vehicle.rotor_thrust_axes, axis=0)
    h_dot = -np.sum((vehicle.rotor_dir * vehicle.rotor_inertia * rotor_accel)[:, np.newaxis] * vehicle.rotor_thrust_axes, axis=0)
    np.testing.assert_allclose(vehicle.inertia @ extra_wdot, -h_dot - np.cross(w, h), rtol=1e-10)

    # Spinning up a rotor with positive rotor_direction yaws the body the same way its drag torque does.
    single = np.zeros(4)
    single[0] = 1.0
    state = make_state(speeds)
    extra = (vehicle._s_dot_fn(0, vehicle._pack_state(state), speeds + single)
             - reference._s_dot_fn(0, vehicle._pack_state(state), speeds + single))[10:13]
    assert extra[2] > 0


def test_pure_gyroscopic_precession():
    """Commanding the speeds the rotors already hold isolates w x h_int, the confusing term."""
    vehicle = MultirotorExtended(make_params(5), aero=False)
    reference = MultirotorExtended(params_without_rotor_inertia(5), aero=False)
    speeds = np.array([180.0, 190.0, 1300.0, 1350.0])
    w = np.array([0.3, -0.2, 0.5])
    state = make_state(speeds)
    state['w'] = w
    s = vehicle._pack_state(state)

    # Commanding the current speeds makes rotor_accel zero, so h_int_dot drops out entirely.
    extra_wdot = (vehicle._s_dot_fn(0, s, speeds) - reference._s_dot_fn(0, s, speeds))[10:13]
    h = -np.sum((vehicle.rotor_dir * vehicle.rotor_inertia * speeds)[:, np.newaxis] * vehicle.rotor_thrust_axes, axis=0)
    np.testing.assert_allclose(vehicle.inertia @ extra_wdot, -np.cross(w, h), rtol=1e-10)

    # Precession does no work about the axis that caused it, whatever the geometry.
    np.testing.assert_allclose((vehicle.inertia @ extra_wdot) @ w, 0.0, atol=1e-12)

    # With no cant every rotor axis is body z, so h_int is purely along z and roll precesses into pitch.
    flat = MultirotorExtended(make_params(0), aero=False)
    flat_reference = MultirotorExtended(params_without_rotor_inertia(0), aero=False)
    roll = np.array([0.4, 0.0, 0.0])
    state = make_state(speeds)
    state['w'] = roll
    s = flat._pack_state(state)

    extra_wdot = (flat._s_dot_fn(0, s, speeds) - flat_reference._s_dot_fn(0, s, speeds))[10:13]
    # Nonzero because the rotors are dissimilar: the usual quad cancellation fails.
    h_z = -np.sum(flat.rotor_dir * flat.rotor_inertia * speeds)
    moment = flat.inertia @ extra_wdot
    np.testing.assert_allclose(moment, [0.0, 0.4 * h_z, 0.0], atol=1e-12)
    assert h_z > 0
    assert abs(moment[1]) > 1e-6
