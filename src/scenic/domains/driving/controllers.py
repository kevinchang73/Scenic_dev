"""Low-level controllers useful for vehicles.

The Lateral/Longitudinal PID controllers are adapted from `CARLA`_'s PID controllers,
which are licensed under the following terms:

    Copyright (c) 2018-2020 CVC.

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>.

The `MPCCController` implements a Model Predictive Contouring Controller, a
nonlinear model-predictive controller which jointly computes throttle and
steering by tracking a reference path over a receding horizon. It requires the
optional `casadi`_ dependency, installable via ``pip install scenic[mpcc]``.

.. _CARLA: https://carla.org/
.. _casadi: https://web.casadi.org/
"""

from collections import deque
import math

import numpy as np


def _require_casadi():
    """Import and return the :mod:`casadi` module, or raise a helpful error.

    The Model Predictive Contouring Controller depends on CasADi (with its
    bundled IPOPT solver), which is an optional dependency of Scenic. This
    helper defers the import so that users who do not use MPCC-based behaviors
    are not required to install it.
    """
    try:
        import casadi  # noqa: F401
    except ImportError as e:  # pragma: no cover - exercised only without casadi
        raise ImportError(
            "The MPCC controller requires the optional 'casadi' dependency. "
            "Install it with 'pip install scenic[mpcc]' (or 'pip install casadi')."
        ) from e
    return casadi


class PIDLongitudinalController:
    """Longitudinal control using a PID to reach a target speed.

    Arguments:
        K_P: Proportional gain
        K_D: Derivative gain
        K_I: Integral gain
        dt: time step
    """

    def __init__(self, K_P=0.5, K_D=0.1, K_I=0.2, dt=0.1):
        self._k_p = K_P
        self._k_d = K_D
        self._k_i = K_I
        self._dt = dt
        self._error_buffer = deque(maxlen=10)

    def run_step(self, speed_error):
        """Estimate the throttle/brake of the vehicle based on the PID equations.

        Arguments:
            speed_error: target speed minus current speed

        Returns:
            a signal between -1 and 1, with negative values indicating braking.
        """
        error = speed_error
        self._error_buffer.append(error)

        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip(
            (self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0
        )


class PIDLateralController:
    """Lateral control using a PID to track a trajectory.

    Arguments:
        K_P: Proportional gain
        K_D: Derivative gain
        K_I: Integral gain
        dt: time step
    """

    def __init__(self, K_P=0.3, K_D=0.2, K_I=0, dt=0.1):
        self.Kp = K_P
        self.Kd = K_D
        self.Ki = K_I
        self.PTerm = 0
        self.ITerm = 0
        self.DTerm = 0
        self.dt = dt
        self.last_error = 0
        self.windup_guard = 20.0
        self.output = 0

    def run_step(self, cte):
        """Estimate the steering angle of the vehicle based on the PID equations.

        Arguments:
            cte: cross-track error (distance to right of desired trajectory)

        Returns:
            a signal between -1 and 1, with -1 meaning maximum steering to the left.
        """
        error = cte
        delta_error = error - self.last_error
        self.PTerm = self.Kp * error
        self.ITerm += error * self.dt

        if self.ITerm < -self.windup_guard:
            self.ITerm = -self.windup_guard
        elif self.ITerm > self.windup_guard:
            self.ITerm = self.windup_guard

        self.DTerm = delta_error / self.dt

        # Remember last error for next calculation
        self.last_error = error

        self.output = self.PTerm + (self.Ki * self.ITerm) + (self.Kd * self.DTerm)

        return np.clip(self.output, -1, 1)


class MPCCController:
    """Model Predictive Contouring Controller (MPCC).

    A nonlinear model-predictive controller that jointly computes throttle and
    steering by tracking a reference path (a centerline) over a receding
    horizon. Unlike the PID controllers, which use two decoupled loops fed with
    scalar errors, the MPCC uses the full vehicle state and a model of the
    vehicle dynamics (a kinematic bicycle model) to optimize both controls at
    once, trading off path-tracking accuracy against progress along the path.

    The optimization problem is formulated and solved with `CasADi`_ (using its
    bundled IPOPT solver), an optional dependency installable via
    ``pip install scenic[mpcc]``.

    The controller works in Scenic's global coordinate frame, in which a heading
    of ``0`` points along ``+y`` and increases counterclockwise, so the vehicle
    velocity is ``v * (-sin(heading), cos(heading))``. The kinematic bicycle
    model used internally is therefore

    .. math::

        \\dot{x} = -v \\sin\\psi, \\quad
        \\dot{y} = v \\cos\\psi, \\quad
        \\dot{\\psi} = \\frac{v}{L}\\tan\\delta, \\quad
        \\dot{v} = a,

    where :math:`\\psi` is the heading, :math:`v` the speed, :math:`\\delta` the
    front-wheel steering angle, :math:`a` the longitudinal acceleration and
    :math:`L` the wheelbase.

    Before use, a reference path must be supplied with `setReference`. Each
    control step, `run_step` returns a ``(throttle, steer)`` pair, where
    **throttle** is a signed value in ``[-1, 1]`` (negative meaning braking) and
    **steer** is a value in ``[-1, 1]`` (positive steering to the right,
    matching `SetSteerAction`). These are suitable for passing directly to
    `RegulatedControlAction`.

    Arguments:
        wheelbase: Distance between the front and rear axles, in meters.
        dt: Discretization time step, in seconds (typically the simulation step).
        horizon: Number of steps in the prediction horizon.
        q_c: Weight penalizing the contouring (lateral) error.
        q_l: Weight penalizing the lag (longitudinal) error.
        q_v: Weight rewarding progress along the reference path.
        q_speed: Weight penalizing deviation from the target speed.
        r_accel: Weight penalizing acceleration effort.
        r_steer: Weight penalizing steering effort.
        r_daccel: Weight penalizing changes in acceleration between steps.
        r_dsteer: Weight penalizing changes in steering between steps.
        max_accel: Maximum absolute longitudinal acceleration, in m/s^2.
        max_steer_angle: Maximum absolute steering angle, in radians.
        max_speed: Maximum allowed speed, in m/s.
        accel_scale: Acceleration (in m/s^2) mapped to a throttle command of 1.0.
        ref_spacing: Spacing between sampled reference points, in meters.
        solver_options: Optional dict of options forwarded to the CasADi/IPOPT
            solver (merged with quiet defaults).

    .. _CasADi: https://web.casadi.org/
    """

    def __init__(
        self,
        wheelbase=2.7,
        dt=0.1,
        horizon=12,
        q_c=8.0,
        q_l=8.0,
        q_v=0.6,
        q_speed=1.0,
        r_accel=0.01,
        r_steer=0.5,
        r_daccel=0.01,
        r_dsteer=2.0,
        max_accel=5.6,
        max_steer_angle=0.6,
        max_speed=30.0,
        accel_scale=5.6,
        ref_spacing=1.0,
        solver_options=None,
    ):
        _require_casadi()
        self.L = float(wheelbase)
        self.dt = float(dt)
        self.N = int(horizon)
        self.q_c = float(q_c)
        self.q_l = float(q_l)
        self.q_v = float(q_v)
        self.q_speed = float(q_speed)
        self.r_accel = float(r_accel)
        self.r_steer = float(r_steer)
        self.r_daccel = float(r_daccel)
        self.r_dsteer = float(r_dsteer)
        self.max_accel = float(max_accel)
        self.max_steer_angle = float(max_steer_angle)
        self.max_speed = float(max_speed)
        self.accel_scale = float(accel_scale)
        self.ref_spacing = float(ref_spacing)
        self.solver_options = solver_options

        # Reference path data (populated by setReference).
        self._ref_len = None
        self._x_ref = None
        self._y_ref = None
        self._phi_ref = None

        # Warm-start caches for the primal variables.
        self._warm_z = None
        self._warm_u = None

    def _samplePath(self, centerline):
        """Sample the reference path into (arclength, x, y, heading) arrays."""
        casadi = _require_casadi()

        # Accept a PolylineRegion (with .length / .pointAlongBy) or a sequence of
        # (x, y) points.
        if hasattr(centerline, "length") and hasattr(centerline, "pointAlongBy"):
            total = float(centerline.length)
            num = max(2, int(math.ceil(total / self.ref_spacing)) + 1)
            dists = np.linspace(0.0, total, num)
            pts = [centerline.pointAlongBy(float(d)) for d in dists]
            xs = np.array([float(p[0]) for p in pts])
            ys = np.array([float(p[1]) for p in pts])
        else:
            pts = np.asarray([[float(p[0]), float(p[1])] for p in centerline])
            xs = pts[:, 0]
            ys = pts[:, 1]
            seg = np.hypot(np.diff(xs), np.diff(ys))
            dists = np.concatenate([[0.0], np.cumsum(seg)])

        # Deduplicate coincident samples so the arclength grid is strictly
        # increasing (required by the CasADi interpolant).
        keep = np.concatenate([[True], np.diff(dists) > 1e-6])
        dists, xs, ys = dists[keep], xs[keep], ys[keep]
        if len(dists) < 2:
            raise ValueError("Reference path is too short for MPCC tracking.")

        # Path tangent heading at each sample, in the math-angle convention.
        dx = np.gradient(xs, dists)
        dy = np.gradient(ys, dists)
        phi = np.unwrap(np.arctan2(dy, dx))

        grid = dists.tolist()
        self._ref_len = float(dists[-1])
        self._x_ref = casadi.interpolant("x_ref", "linear", [grid], xs.tolist())
        self._y_ref = casadi.interpolant("y_ref", "linear", [grid], ys.tolist())
        self._phi_ref = casadi.interpolant("phi_ref", "linear", [grid], phi.tolist())

    def setReference(self, centerline):
        """Set (or replace) the reference path to track.

        This resamples the path and rebuilds the underlying optimization
        problem, so it should only be called when the path actually changes
        (e.g. upon entering an intersection), not every control step.

        Arguments:
            centerline: A `PolylineRegion` (or a sequence of ``(x, y)`` points)
                giving the reference path in world coordinates.
        """
        self._samplePath(centerline)
        self._buildProblem()
        self._warm_z = None
        self._warm_u = None

    def _buildProblem(self):
        casadi = _require_casadi()
        N, dt, L = self.N, self.dt, self.L

        opti = casadi.Opti()

        # Decision variables: state z = [x, y, psi, v, theta] and control
        # u = [a, delta, v_theta] over the horizon.
        Z = opti.variable(5, N + 1)
        U = opti.variable(3, N)

        # Parameters: initial state and target speed.
        z0 = opti.parameter(5, 1)
        v_target = opti.parameter(1, 1)

        def dynamics(z, u):
            x, y, psi, v, theta = z[0], z[1], z[2], z[3], z[4]
            a, delta, v_theta = u[0], u[1], u[2]
            return casadi.vertcat(
                -v * casadi.sin(psi),
                v * casadi.cos(psi),
                v / L * casadi.tan(delta),
                a,
                v_theta,
            )

        cost = 0
        for k in range(N):
            zk = Z[:, k]
            uk = U[:, k]

            # RK4 integration of the continuous dynamics.
            k1 = dynamics(zk, uk)
            k2 = dynamics(zk + dt / 2 * k1, uk)
            k3 = dynamics(zk + dt / 2 * k2, uk)
            k4 = dynamics(zk + dt * k3, uk)
            z_next = zk + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            opti.subject_to(Z[:, k + 1] == z_next)

            # Contouring and lag errors relative to the reference at progress
            # theta.
            theta = zk[4]
            xr = self._x_ref(theta)
            yr = self._y_ref(theta)
            phi = self._phi_ref(theta)
            dx = zk[0] - xr
            dy = zk[1] - yr
            e_c = casadi.sin(phi) * dx - casadi.cos(phi) * dy
            e_l = -casadi.cos(phi) * dx - casadi.sin(phi) * dy

            cost += self.q_c * e_c**2 + self.q_l * e_l**2
            cost += self.q_speed * (zk[3] - v_target) ** 2
            cost -= self.q_v * uk[2] * dt
            cost += self.r_accel * uk[0] ** 2 + self.r_steer * uk[1] ** 2
            if k > 0:
                du = U[:, k] - U[:, k - 1]
                cost += self.r_daccel * du[0] ** 2 + self.r_dsteer * du[1] ** 2

        # Terminal speed and contouring cost.
        theta_N = Z[4, N]
        xrN, yrN, phiN = (
            self._x_ref(theta_N),
            self._y_ref(theta_N),
            self._phi_ref(theta_N),
        )
        dxN, dyN = Z[0, N] - xrN, Z[1, N] - yrN
        e_cN = casadi.sin(phiN) * dxN - casadi.cos(phiN) * dyN
        cost += self.q_c * e_cN**2 + self.q_speed * (Z[3, N] - v_target) ** 2

        # Bounds.
        opti.subject_to(opti.bounded(-self.max_accel, U[0, :], self.max_accel))
        opti.subject_to(
            opti.bounded(-self.max_steer_angle, U[1, :], self.max_steer_angle)
        )
        opti.subject_to(opti.bounded(0.0, U[2, :], self.max_speed))
        opti.subject_to(opti.bounded(0.0, Z[3, :], self.max_speed))

        # Initial condition.
        opti.subject_to(Z[:, 0] == z0)

        opti.minimize(cost)

        options = {
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "ipopt.max_iter": 60,
            "print_time": 0,
        }
        if self.solver_options:
            options.update(self.solver_options)
        opti.solver("ipopt", options)

        self._opti = opti
        self._Z = Z
        self._U = U
        self._z0 = z0
        self._v_target = v_target

    def _initialProgress(self, x, y):
        """Estimate the path progress (arclength) nearest to ``(x, y)``."""
        # Coarse search over the sampled arclength grid, refined locally.
        grid = np.linspace(0.0, self._ref_len, max(2, int(self._ref_len) + 1))
        xs = np.array([float(self._x_ref(float(t))) for t in grid])
        ys = np.array([float(self._y_ref(float(t))) for t in grid])
        d2 = (xs - x) ** 2 + (ys - y) ** 2
        return float(grid[int(np.argmin(d2))])

    def run_step(self, x, y, heading, speed, target_speed):
        """Compute throttle and steering for the current vehicle state.

        Arguments:
            x: Current world x-coordinate of the vehicle.
            y: Current world y-coordinate of the vehicle.
            heading: Current heading, in radians (Scenic convention).
            speed: Current forward speed, in m/s.
            target_speed: Desired speed, in m/s.

        Returns:
            A ``(throttle, steer)`` pair. **throttle** is a signed value in
            ``[-1, 1]`` (negative meaning braking) and **steer** is in
            ``[-1, 1]`` (positive steering to the right).
        """
        if self._x_ref is None:
            raise RuntimeError(
                "MPCCController.run_step called before setReference; "
                "a reference path must be provided first."
            )
        casadi = _require_casadi()

        theta0 = self._initialProgress(x, y)
        speed = max(0.0, float(speed))
        state0 = np.array([float(x), float(y), float(heading), speed, theta0])

        self._opti.set_value(self._z0, state0)
        self._opti.set_value(self._v_target, float(target_speed))

        # Warm start from the previous solution (shifted) when available.
        if self._warm_z is not None:
            self._opti.set_initial(self._Z, self._warm_z)
            self._opti.set_initial(self._U, self._warm_u)
        else:
            z_guess = np.tile(state0.reshape(5, 1), (1, self.N + 1))
            # Advance the progress guess so the solver starts moving forward.
            z_guess[4, :] = theta0 + np.arange(self.N + 1) * speed * self.dt
            self._opti.set_initial(self._Z, z_guess)
            self._opti.set_initial(self._U, np.zeros((3, self.N)))

        try:
            sol = self._opti.solve()
            a0 = float(sol.value(self._U[0, 0]))
            delta0 = float(sol.value(self._U[1, 0]))
            self._warm_z = np.array(sol.value(self._Z))
            self._warm_u = np.array(sol.value(self._U))
        except RuntimeError:
            # Solver failed to converge: fall back to the best iterate and
            # discard the warm start so the next step restarts cleanly.
            a0 = float(self._opti.debug.value(self._U[0, 0]))
            delta0 = float(self._opti.debug.value(self._U[1, 0]))
            self._warm_z = None
            self._warm_u = None

        throttle = float(np.clip(a0 / self.accel_scale, -1.0, 1.0))
        # Positive steering angle (delta) turns left (increasing heading), while
        # a positive steer command steers right, so negate.
        steer = float(np.clip(-delta0 / self.max_steer_angle, -1.0, 1.0))
        return throttle, steer
