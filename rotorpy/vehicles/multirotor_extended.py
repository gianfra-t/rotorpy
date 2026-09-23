import numpy as np
from scipy.spatial.transform import Rotation

from rotorpy.vehicles.multirotor import Multirotor, quat_dot

"""
Multirotor with dissimilar, canted and non-planar rotors driven by dissimilar motors.
"""


class MultirotorExtended(Multirotor):
    """
    Multirotor class that supports asymmetrical configurations, whose rotors may differ from each other in size,
    need not lie in one plane or point along body z. 

    Extra quad_params (all optional, defaults reproduce Multirotor exactly):
        k_eta:              scalar or one value per rotor, thrust coeff, N/(rad/s)**2
        k_m:                scalar or one value per rotor, yaw moment coeff, Nm/(rad/s)**2
        k_d, k_z, k_h:      scalar or one value per rotor, rotor drag (in-plane), induced inflow (axial) and
                            translational lift coefficients, as in Multirotor
        k_flap:             scalar or one value per rotor, hub flapping moment coeff, Nm/(rad*m/s**2)
        rotor_thrust_axes:  (num_rotors, 3) thrust direction of each rotor in the body frame.
        tau_m:              scalar or one value per rotor, motor response time, seconds
        rotor_speed_min:    scalar or one value per rotor, rad/s
        rotor_speed_max:    scalar or one value per rotor, rad/s
        rotor_inertia:      scalar or one value per rotor, rotating inertia (prop + motor bell) about its axis, kg*m^2
    """
    def __init__(self, quad_params, *args, **kwargs):
        # Multirotor builds a scalar rotor_drag_matrix from these, compute_body_wrench below does not use it since it's now
        # different implementation.
        super().__init__({**quad_params, 'k_d': 0.0, 'k_z': 0.0, 'k_h': 0.0}, *args, **kwargs)

        self.rotor_dir = np.asarray(quad_params['rotor_directions'], dtype=float)

        self.k_eta = self.per_rotor(quad_params, 'k_eta')
        self.k_m = self.per_rotor(quad_params, 'k_m')
        self.k_d = self.per_rotor(quad_params, 'k_d', positive=False, default=0.0)
        self.k_z = self.per_rotor(quad_params, 'k_z', positive=False, default=0.0)
        self.k_h = self.per_rotor(quad_params, 'k_h', positive=False, default=0.0)
        self.k_flap = self.per_rotor(quad_params, 'k_flap', positive=False, default=0.0)

        self.rotor_thrust_axes = np.array(
            quad_params.get('rotor_thrust_axes', np.tile([0.0, 0.0, 1.0], (self.num_rotors, 1))),
            dtype=float,
        )

        self.tau_m = self.per_rotor(quad_params, 'tau_m')
        self.rotor_speed_min = self.per_rotor(quad_params, 'rotor_speed_min', positive=False)
        self.rotor_speed_max = self.per_rotor(quad_params, 'rotor_speed_max')
        self.rotor_inertia = self.per_rotor(quad_params, 'rotor_inertia', positive=False, default=0.0)

        self._validate_construction()

        self.rotor_thrust_axes /= np.linalg.norm(self.rotor_thrust_axes, axis=1)[:, np.newaxis]

        # Control allocation: column i is the [F_z, M] produced by a unit thrust on rotor i.
        rotor_moments = np.cross(self.rotor_geometry, self.rotor_thrust_axes)
        rotor_moments += (self.rotor_dir * self.k_m / self.k_eta)[:, np.newaxis] * self.rotor_thrust_axes
        self.f_to_TM = np.vstack((self.rotor_thrust_axes[:, 2], rotor_moments.T))
        rank = np.linalg.matrix_rank(self.f_to_TM)
        self.has_full_wrench_authority = rank == 4
        if self.f_to_TM.shape == (4, 4) and rank == 4:
            self.TM_to_f = np.linalg.inv(self.f_to_TM)
        else:
            self.TM_to_f = np.linalg.pinv(self.f_to_TM)

        # Column i is rotor_directions[i] * J_i * axis_i, so the rotors' angular momentum is
        # h_int = -rotor_momentum_axes @ rotor_speeds.
        self.rotor_momentum_axes = self.rotor_thrust_axes.T * (self.rotor_dir * self.rotor_inertia)

    def _validate_construction(self):
        """
        Check that the gathered per-rotor configuration is mutually consistent. Raises on the first
        check found.
        """
        if len(self.rotor_pos) != self.num_rotors or self.rotor_dir.shape != (self.num_rotors,):
            raise ValueError("rotor_pos and rotor_directions must contain one entry per rotor")
        if self.rotor_thrust_axes.shape != (self.num_rotors, 3):
            raise ValueError("rotor_thrust_axes must have shape (num_rotors, 3)")
        axis_norms = np.linalg.norm(self.rotor_thrust_axes, axis=1)
        if not np.all(np.isfinite(axis_norms)) or np.any(axis_norms == 0):
            raise ValueError("rotor_thrust_axes must contain finite nonzero vectors")
        if np.any(self.rotor_speed_min >= self.rotor_speed_max):
            raise ValueError("rotor_speed_min must be nonnegative and below rotor_speed_max")

    def per_rotor(self, quad_params, name, positive=True, default=None):
        """
        Return quad_params[name] broadcast to one finite value per rotor. Nonpositive values are rejected
        when positive is set, negative ones otherwise. default is used when name is absent.
        """
        values = np.asarray(quad_params[name] if default is None else quad_params.get(name, default), dtype=float)
        try:
            values = np.broadcast_to(values, (self.num_rotors,)).copy()
        except ValueError as error:
            raise ValueError(f"{name} must be a scalar or have one value per rotor") from error
        if not np.all(np.isfinite(values)) or np.any(values <= 0 if positive else values < 0):
            raise ValueError(f"{name} must contain finite {'positive' if positive else 'nonnegative'} values")
        return values

    def compute_body_wrench(self, body_rates, rotor_speeds, body_airspeed_vector):
        """
        Same as Multirotor.compute_body_wrench, with each rotor's thrust and reaction torque along its own axis.
        """
        local_airspeeds = body_airspeed_vector[:, np.newaxis] + Multirotor.hat_map(body_rates)@(self.rotor_geometry.T)

        T = self.rotor_thrust_axes.T * (self.k_eta * rotor_speeds**2)

        if self.aero:
            D = -Multirotor._norm(body_airspeed_vector)*self.drag_matrix@body_airspeed_vector
            axes = self.rotor_thrust_axes.T
            axial_airspeeds = axes * np.sum(axes * local_airspeeds, axis=0)
            inplane_airspeeds = local_airspeeds - axial_airspeeds
            H = -rotor_speeds*(self.k_d*inplane_airspeeds + self.k_z*axial_airspeeds)
            M_flap = -self.k_flap*rotor_speeds*np.cross(local_airspeeds.T, axes.T).T
            T += axes*(self.k_h*np.sum(inplane_airspeeds**2, axis=0))
        else:
            D = np.zeros(3,)
            H = np.zeros((3,self.num_rotors))
            M_flap = np.zeros((3,self.num_rotors))

        M_force = np.sum(np.cross(self.rotor_geometry, (T + H).T), axis=0)
        M_yaw = self.rotor_thrust_axes.T * (self.rotor_dir * self.k_m * rotor_speeds**2)

        FtotB = np.sum(T + H, axis=1) + D
        MtotB = M_force + np.sum(M_yaw + M_flap, axis=1)

        return (FtotB, MtotB)

    def _s_dot_fn(self, t, s, cmd_rotor_speeds):
        """
        Same as Multirotor._s_dot_fn, with the rotor angular momentum h_int in the rotational equation.
        """
        state = Multirotor._unpack_state(s)

        rotor_speeds = state['rotor_speeds']
        inertial_velocity = state['v']
        wind_velocity = state['wind']

        R = Rotation.from_quat(state['q']).as_matrix()

        # Rotor speed derivative
        rotor_accel = (1/self.tau_m)*(cmd_rotor_speeds - rotor_speeds)

        # Position derivative.
        x_dot = state['v']

        # Orientation derivative.
        q_dot = quat_dot(state['q'], state['w'])

        # Compute airspeed vector in the body frame
        body_airspeed_vector = R.T@(inertial_velocity - wind_velocity)

        # Compute total wrench in the body frame based on the current rotor speeds and their location w.r.t. CoM
        (FtotB, MtotB) = self.compute_body_wrench(state['w'], rotor_speeds, body_airspeed_vector)

        # Rotate the force from the body frame to the inertial frame
        Ftot = R@FtotB

        # Ground reaction force: apply normal force when on ground to prevent penetration
        if self._enable_ground and self._on_ground(state):
            total_force_no_ground = self.weight + Ftot
            if total_force_no_ground[2] < 0:
                ground_normal_force = np.array([0, 0, -total_force_no_ground[2]])
                Ftot += ground_normal_force

        # Velocity derivative.
        v_dot = (self.weight + Ftot) / self.mass

        # Angular velocity derivative, with the rotors' angular momentum and its rate (spin-up).
        w = state['w']
        h_int = -self.rotor_momentum_axes @ rotor_speeds
        h_int_dot = -self.rotor_momentum_axes @ rotor_accel
        w_dot = self.inv_inertia @ (MtotB - np.cross(w, self.inertia @ w + h_int) - h_int_dot)

        # Wind is assigned by the wind_profile object, not integrated.
        wind_dot = np.zeros(3,)

        # Pack derivatives state
        s_dot = np.zeros((16+self.num_rotors,))
        s_dot[0:3]   = x_dot
        s_dot[3:6]   = v_dot
        s_dot[6:10]  = q_dot
        s_dot[10:13] = w_dot
        s_dot[13:16] = wind_dot
        s_dot[16:]   = rotor_accel

        return s_dot
