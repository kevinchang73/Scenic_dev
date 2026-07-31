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
import time

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


def _footprintCircleGeometry(width, length, count=3):
    """Return longitudinal offsets and radius of a covering-circle footprint."""
    width = float(width)
    length = float(length)
    count = int(count)
    if width < 0 or length < 0:
        raise ValueError("footprint width and length must be nonnegative")
    if count < 1:
        raise ValueError("footprint circle count must be positive")
    slice_length = length / count
    offsets = np.linspace(
        -length / 2 + slice_length / 2,
        length / 2 - slice_length / 2,
        count,
    )
    radius = math.hypot(width / 2, slice_length / 2)
    return offsets, radius


def _orientedCircleCenters(x, y, heading, offsets):
    """Compute footprint-circle centers using Scenic's heading convention."""
    offsets = np.asarray(offsets, dtype=float)
    return np.column_stack(
        (
            float(x) - offsets * math.sin(float(heading)),
            float(y) + offsets * math.cos(float(heading)),
        )
    )


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
        max_steer_rate: Optional maximum steering-angle rate, in radians per
            second. This should match any downstream steering slew limiter.
        max_speed: Maximum allowed speed, in m/s.
        accel_scale: Acceleration (in m/s^2) mapped to a throttle command of 1.0.
        ref_spacing: Spacing between sampled reference points, in meters.
        collision_avoidance: Whether to add predicted pairwise safety constraints.
        vehicle_width: Width of the controlled object's footprint, in meters.
        vehicle_length: Length of the controlled object's footprint, in meters.
        max_obstacles: Initial number of parameterized obstacle slots.
        collision_margin: Additional clearance between object footprints, in meters.
        footprint_circles: Number of longitudinal circles covering each footprint.
        collision_slack_weight: Cost applied to squared-distance safety slack.
        path_boundary_margin: Optional lateral margin from the reference path,
            used to keep the vehicle footprint inside a lane or drivable corridor.
        path_boundary_slack_weight: Cost applied to path-boundary slack.
        failure_feasibility_tol: Maximum constraint violation at which a finite
            IPOPT iterate may be used after non-successful solver termination.
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
        max_steer_rate=None,
        max_speed=30.0,
        accel_scale=5.6,
        ref_spacing=1.0,
        collision_avoidance=False,
        vehicle_width=2.0,
        vehicle_length=4.5,
        max_obstacles=0,
        collision_margin=0.25,
        footprint_circles=3,
        collision_slack_weight=100000.0,
        path_boundary_margin=None,
        path_boundary_slack_weight=100000.0,
        failure_feasibility_tol=1e-5,
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
        self.max_steer_rate = None if max_steer_rate is None else float(max_steer_rate)
        self.max_speed = float(max_speed)
        self.accel_scale = float(accel_scale)
        self.ref_spacing = float(ref_spacing)
        self.collision_avoidance = bool(collision_avoidance)
        self.vehicle_width = float(vehicle_width)
        self.vehicle_length = float(vehicle_length)
        self.max_obstacles = int(max_obstacles)
        self.collision_margin = float(collision_margin)
        self.footprint_circles = int(footprint_circles)
        self.collision_slack_weight = float(collision_slack_weight)
        self.path_boundary_margin = (
            None if path_boundary_margin is None else float(path_boundary_margin)
        )
        self.path_boundary_slack_weight = float(path_boundary_slack_weight)
        self.failure_feasibility_tol = float(failure_feasibility_tol)
        self.solver_options = solver_options
        if self.max_obstacles < 0:
            raise ValueError("max_obstacles must be nonnegative")
        if self.collision_margin < 0:
            raise ValueError("collision_margin must be nonnegative")
        if self.collision_slack_weight <= 0:
            raise ValueError("collision_slack_weight must be positive")
        if self.path_boundary_margin is not None and self.path_boundary_margin <= 0:
            raise ValueError("path_boundary_margin must be positive")
        if self.path_boundary_slack_weight <= 0:
            raise ValueError("path_boundary_slack_weight must be positive")
        if self.failure_feasibility_tol < 0:
            raise ValueError("failure_feasibility_tol must be nonnegative")
        if self.max_steer_rate is not None and self.max_steer_rate <= 0:
            raise ValueError("max_steer_rate must be positive")
        self._footprint_offsets, self._footprint_radius = _footprintCircleGeometry(
            self.vehicle_width, self.vehicle_length, self.footprint_circles
        )

        # Reference path data (populated by setReference).
        self._ref_len = None
        self._x_ref = None
        self._y_ref = None
        self._phi_ref = None
        self._ref_s = None
        self._ref_x_samples = None
        self._ref_y_samples = None

        # Warm-start caches for the primal variables.
        self._warm_z = None
        self._warm_u = None
        self._previous_accel = 0.0
        self._previous_steer = 0.0
        self.last_collision_slack = np.empty((0, self.N))
        self.last_collision_max_slack = 0.0
        self.last_path_boundary_slack = np.empty(0)
        self.last_path_boundary_max_slack = 0.0
        self.last_solve_succeeded = True
        self.last_solver_succeeded = True
        self.last_accepted_feasible_iterate = False
        self.last_solve_time = 0.0
        self.last_solve_iterations = None
        self.last_solve_status = None
        self.solve_history = []

    @staticmethod
    def _solverStats(solver):
        """Return solver statistics when available, including after failures."""
        try:
            return solver.stats()
        except (AttributeError, RuntimeError):
            return {}

    def _debugSolveDiagnostics(self):
        """Summarize the failed iterate without using it for actuation."""
        diagnostics = {
            "debug_objective": None,
            "debug_max_constraint_violation": None,
        }
        try:
            objective = float(self._opti.debug.value(self._opti.f))
            constraints = np.asarray(
                self._opti.debug.value(self._opti.g), dtype=float
            ).reshape(-1)
            lower = np.asarray(
                self._opti.debug.value(self._opti.lbg), dtype=float
            ).reshape(-1)
            upper = np.asarray(
                self._opti.debug.value(self._opti.ubg), dtype=float
            ).reshape(-1)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return diagnostics
        if math.isfinite(objective):
            diagnostics["debug_objective"] = objective
        if constraints.size == lower.size == upper.size:
            violations = np.zeros_like(constraints)
            finite_lower = np.isfinite(lower)
            finite_upper = np.isfinite(upper)
            violations[finite_lower] = np.maximum(
                violations[finite_lower],
                lower[finite_lower] - constraints[finite_lower],
            )
            violations[finite_upper] = np.maximum(
                violations[finite_upper],
                constraints[finite_upper] - upper[finite_upper],
            )
            if np.all(np.isfinite(violations)):
                diagnostics["debug_max_constraint_violation"] = float(
                    np.max(violations, initial=0.0)
                )
        return diagnostics

    def getSolveMetrics(self, include_history=True):
        """Return JSON-compatible timing and IPOPT iteration statistics."""
        times = [sample["solve_time_seconds"] for sample in self.solve_history]
        iterations = [
            sample["iterations"]
            for sample in self.solve_history
            if sample["iterations"] is not None
        ]
        successful_solves = sum(sample["succeeded"] for sample in self.solve_history)
        solver_successes = sum(
            sample.get("solver_succeeded", sample["succeeded"])
            for sample in self.solve_history
        )
        accepted_feasible_iterates = sum(
            sample.get("accepted_feasible_iterate", False)
            for sample in self.solve_history
        )
        metrics = {
            "solves": len(self.solve_history),
            "successful_solves": successful_solves,
            "failed_solves": len(self.solve_history) - successful_solves,
            "solver_successful_solves": solver_successes,
            "solver_failed_solves": len(self.solve_history) - solver_successes,
            "accepted_feasible_iterates": accepted_feasible_iterates,
            "total_solve_time_seconds": float(sum(times)),
            "mean_solve_time_seconds": float(np.mean(times)) if times else None,
            "median_solve_time_seconds": float(np.median(times)) if times else None,
            "max_solve_time_seconds": float(max(times)) if times else None,
            "p95_solve_time_seconds": (
                float(np.percentile(times, 95)) if times else None
            ),
            "mean_iterations": (float(np.mean(iterations)) if iterations else None),
            "max_iterations": max(iterations) if iterations else None,
        }
        if include_history:
            metrics["history"] = list(self.solve_history)
        return metrics

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
        self._ref_s = np.asarray(dists, dtype=float)
        self._ref_x_samples = np.asarray(xs, dtype=float)
        self._ref_y_samples = np.asarray(ys, dtype=float)
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

        collision_enabled = self.collision_avoidance and self.max_obstacles > 0
        if collision_enabled:
            obstacle_rows = self.max_obstacles * self.footprint_circles
            obstacle_x = opti.parameter(obstacle_rows, N + 1)
            obstacle_y = opti.parameter(obstacle_rows, N + 1)
            obstacle_radius = opti.parameter(self.max_obstacles, 1)
            obstacle_active = opti.parameter(self.max_obstacles, 1)
            collision_slack = opti.variable(self.max_obstacles, N)
        else:
            obstacle_x = None
            obstacle_y = None
            obstacle_radius = None
            obstacle_active = None
            collision_slack = None

        if self.path_boundary_margin is not None:
            path_boundary_slack = opti.variable(1, N + 1)
        else:
            path_boundary_slack = None

        # Parameters: initial state and target speed.
        z0 = opti.parameter(5, 1)
        v_target = opti.parameter(1, 1)
        u_previous = opti.parameter(2, 1)

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
            if k == 0:
                d_accel = U[0, k] - u_previous[0]
                d_steer = U[1, k] - u_previous[1]
            else:
                d_accel = U[0, k] - U[0, k - 1]
                d_steer = U[1, k] - U[1, k - 1]
            cost += self.r_daccel * d_accel**2 + self.r_dsteer * d_steer**2

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

        if path_boundary_slack is not None:
            for k in range(N + 1):
                theta = Z[4, k]
                xr = self._x_ref(theta)
                yr = self._y_ref(theta)
                phi = self._phi_ref(theta)
                dx = Z[0, k] - xr
                dy = Z[1, k] - yr
                contouring_error = casadi.sin(phi) * dx - casadi.cos(phi) * dy
                opti.subject_to(
                    contouring_error
                    <= self.path_boundary_margin + path_boundary_slack[0, k]
                )
                opti.subject_to(
                    contouring_error
                    >= -self.path_boundary_margin - path_boundary_slack[0, k]
                )
            opti.subject_to(path_boundary_slack >= 0)
            cost += self.path_boundary_slack_weight * casadi.sumsqr(path_boundary_slack)

        if collision_enabled:
            circles = self.footprint_circles
            for obstacle_index in range(self.max_obstacles):
                minimum_distance = (
                    self._footprint_radius
                    + obstacle_radius[obstacle_index]
                    + self.collision_margin
                )
                for k in range(1, N + 1):
                    state = Z[:, k]
                    for self_offset in self._footprint_offsets:
                        self_x = state[0] - self_offset * casadi.sin(state[2])
                        self_y = state[1] + self_offset * casadi.cos(state[2])
                        for obstacle_circle in range(circles):
                            row = obstacle_index * circles + obstacle_circle
                            dx = self_x - obstacle_x[row, k]
                            dy = self_y - obstacle_y[row, k]
                            opti.subject_to(
                                dx**2 + dy**2 + collision_slack[obstacle_index, k - 1]
                                >= obstacle_active[obstacle_index] * minimum_distance**2
                            )
            # CasADi interprets inequalities on a non-square matrix as a matrix
            # (semidefinite) inequality. Vectorize to request element-wise
            # nonnegativity for every obstacle/time slack variable.
            opti.subject_to(casadi.vec(collision_slack) >= 0)
            cost += self.collision_slack_weight * casadi.sumsqr(collision_slack)

        # Bounds.
        opti.subject_to(opti.bounded(-self.max_accel, U[0, :], self.max_accel))
        opti.subject_to(
            opti.bounded(-self.max_steer_angle, U[1, :], self.max_steer_angle)
        )
        if self.max_steer_rate is not None:
            # Keep the predicted steering trajectory dynamically achievable by
            # the downstream actuator. Without this constraint, aggressive
            # path tracking can plan an instantaneous steering reversal which
            # is then clipped by RegulatedControlAction, producing lag and
            # lateral overshoot even though the optimized trajectory looks
            # smooth.
            max_steer_delta = self.max_steer_rate * dt
            opti.subject_to(
                opti.bounded(
                    -max_steer_delta,
                    U[1, 0] - u_previous[1],
                    max_steer_delta,
                )
            )
            if N > 1:
                opti.subject_to(
                    opti.bounded(
                        -max_steer_delta,
                        U[1, 1:] - U[1, :-1],
                        max_steer_delta,
                    )
                )
        opti.subject_to(opti.bounded(0.0, U[2, :], self.max_speed))
        opti.subject_to(opti.bounded(0.0, Z[3, :], self.max_speed))

        # Initial condition.
        opti.subject_to(Z[:, 0] == z0)

        opti.minimize(cost)

        options = {
            "expand": True,
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
        self._u_previous = u_previous
        self._obstacle_x = obstacle_x
        self._obstacle_y = obstacle_y
        self._obstacle_radius = obstacle_radius
        self._obstacle_active = obstacle_active
        self._collision_slack = collision_slack
        self._path_boundary_slack = path_boundary_slack

    def _ensureObstacleCapacity(self, obstacle_count):
        """Grow the parameterized obstacle slots if objects were created dynamically."""
        if not self.collision_avoidance or obstacle_count <= self.max_obstacles:
            return
        self.max_obstacles = max(obstacle_count, max(1, 2 * self.max_obstacles))
        self._buildProblem()
        self._warm_z = None
        self._warm_u = None
        self.last_collision_slack = np.empty((0, self.N))
        self.last_collision_max_slack = 0.0

    @staticmethod
    def _finiteObstacleValue(obstacle, name, default=0.0):
        value = getattr(obstacle, name, default)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return float(default)
        return value if math.isfinite(value) else float(default)

    def _predictObstacleFootprints(self, obstacles):
        """Predict obstacle covering circles over the controller horizon."""
        circles = self.footprint_circles
        x_values = np.zeros((self.max_obstacles * circles, self.N + 1))
        y_values = np.zeros_like(x_values)
        radii = np.zeros((self.max_obstacles, 1))
        active = np.zeros((self.max_obstacles, 1))
        times = np.arange(self.N + 1, dtype=float) * self.dt

        for obstacle_index, obstacle in enumerate(obstacles):
            x = self._finiteObstacleValue(obstacle, "x")
            y = self._finiteObstacleValue(obstacle, "y")
            velocity_x = self._finiteObstacleValue(obstacle, "velocity_x")
            velocity_y = self._finiteObstacleValue(obstacle, "velocity_y")
            heading = self._finiteObstacleValue(obstacle, "heading")
            yaw_rate = self._finiteObstacleValue(obstacle, "yaw_rate")
            width = max(0.0, self._finiteObstacleValue(obstacle, "width"))
            length = max(0.0, self._finiteObstacleValue(obstacle, "length"))
            offsets, radius = _footprintCircleGeometry(width, length, circles)

            predicted_x = x + times * velocity_x
            predicted_y = y + times * velocity_y
            predicted_heading = heading + times * yaw_rate
            for circle_index, offset in enumerate(offsets):
                row = obstacle_index * circles + circle_index
                x_values[row, :] = predicted_x - offset * np.sin(predicted_heading)
                y_values[row, :] = predicted_y + offset * np.cos(predicted_heading)
            radii[obstacle_index, 0] = radius
            active[obstacle_index, 0] = 1.0

        return x_values, y_values, radii, active

    def _initialProgress(self, x, y):
        """Estimate the path progress (arclength) nearest to ``(x, y)``."""
        # Project continuously onto every sampled path segment. The previous
        # implementation selected the nearest point on a one-meter grid, making
        # progress jump discontinuously as the vehicle moved and causing the
        # optimizer to alternate steering solutions on otherwise straight lanes.
        x0 = self._ref_x_samples[:-1]
        y0 = self._ref_y_samples[:-1]
        dx = np.diff(self._ref_x_samples)
        dy = np.diff(self._ref_y_samples)
        lengths_squared = dx**2 + dy**2
        fractions = np.divide(
            (x - x0) * dx + (y - y0) * dy,
            lengths_squared,
            out=np.zeros_like(lengths_squared),
            where=lengths_squared > 1e-12,
        )
        fractions = np.clip(fractions, 0.0, 1.0)
        projected_x = x0 + fractions * dx
        projected_y = y0 + fractions * dy
        index = int(np.argmin((projected_x - x) ** 2 + (projected_y - y) ** 2))
        return float(
            self._ref_s[index]
            + fractions[index] * (self._ref_s[index + 1] - self._ref_s[index])
        )

    def run_step(self, x, y, heading, speed, target_speed, obstacles=None):
        """Compute throttle and steering for the current vehicle state.

        Arguments:
            x: Current world x-coordinate of the vehicle.
            y: Current world y-coordinate of the vehicle.
            heading: Current heading, in radians (Scenic convention).
            speed: Current forward speed, in m/s.
            target_speed: Desired speed, in m/s.
            obstacles: Iterable of obstacle-state snapshots. Ignored when collision
                avoidance is disabled.

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

        if obstacles is None:
            obstacles = ()
        else:
            obstacles = tuple(obstacles)
        if self.collision_avoidance:
            self._ensureObstacleCapacity(len(obstacles))

        theta0 = self._initialProgress(x, y)
        speed = max(0.0, float(speed))
        state0 = np.array([float(x), float(y), float(heading), speed, theta0])

        self._opti.set_value(self._z0, state0)
        self._opti.set_value(self._v_target, float(target_speed))
        self._opti.set_value(
            self._u_previous, [self._previous_accel, self._previous_steer]
        )
        if self._collision_slack is not None:
            obstacle_x, obstacle_y, obstacle_radius, obstacle_active = (
                self._predictObstacleFootprints(obstacles)
            )
            self._opti.set_value(self._obstacle_x, obstacle_x)
            self._opti.set_value(self._obstacle_y, obstacle_y)
            self._opti.set_value(self._obstacle_radius, obstacle_radius)
            self._opti.set_value(self._obstacle_active, obstacle_active)

        # Warm start from the previous solution (shifted) when available.
        if self._warm_z is not None:
            z_guess = np.hstack((self._warm_z[:, 1:], self._warm_z[:, -1:]))
            z_guess[:, 0] = state0
            u_guess = np.hstack((self._warm_u[:, 1:], self._warm_u[:, -1:]))
            self._opti.set_initial(self._Z, z_guess)
            self._opti.set_initial(self._U, u_guess)
        else:
            z_guess = np.tile(state0.reshape(5, 1), (1, self.N + 1))
            # Advance the progress guess so the solver starts moving forward.
            z_guess[4, :] = theta0 + np.arange(self.N + 1) * speed * self.dt
            self._opti.set_initial(self._Z, z_guess)
            self._opti.set_initial(self._U, np.zeros((3, self.N)))

        emergency_brake = False
        accepted_feasible_iterate = False
        failure_diagnostics = {
            "debug_objective": None,
            "debug_max_constraint_violation": None,
        }
        solve_started = time.perf_counter()
        try:
            sol = self._opti.solve()
        except RuntimeError:
            solve_time = time.perf_counter() - solve_started
            stats = self._solverStats(self._opti)
            failure_diagnostics = self._debugSolveDiagnostics()
            violation = failure_diagnostics["debug_max_constraint_violation"]
            try:
                debug_z = np.asarray(
                    self._opti.debug.value(self._Z), dtype=float
                ).reshape(5, self.N + 1)
                debug_u = np.asarray(
                    self._opti.debug.value(self._U), dtype=float
                ).reshape(3, self.N)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                debug_z = debug_u = None
            accepted_feasible_iterate = bool(
                violation is not None
                and violation <= self.failure_feasibility_tol
                and debug_z is not None
                and np.all(np.isfinite(debug_z))
                and np.all(np.isfinite(debug_u))
            )
            if accepted_feasible_iterate:
                # IPOPT can exhaust its stationarity iterations even though its
                # current iterate satisfies every dynamics, actuator, and
                # collision constraint. Such an iterate is safe to actuate after
                # the explicit feasibility and finiteness checks above.
                a0 = float(debug_u[0, 0])
                delta0 = float(debug_u[1, 0])
                self._warm_z = debug_z
                self._warm_u = debug_u
                self.last_solve_succeeded = True
                if self._collision_slack is not None:
                    self.last_collision_slack = np.asarray(
                        self._opti.debug.value(self._collision_slack), dtype=float
                    ).reshape(self.max_obstacles, self.N)
                    self.last_collision_max_slack = float(
                        np.max(self.last_collision_slack, initial=0.0)
                    )
                else:
                    self.last_collision_slack = np.empty((0, self.N))
                    self.last_collision_max_slack = 0.0
                if self._path_boundary_slack is not None:
                    self.last_path_boundary_slack = np.asarray(
                        self._opti.debug.value(self._path_boundary_slack), dtype=float
                    ).reshape(self.N + 1)
                    self.last_path_boundary_max_slack = float(
                        np.max(self.last_path_boundary_slack, initial=0.0)
                    )
                else:
                    self.last_path_boundary_slack = np.empty(0)
                    self.last_path_boundary_max_slack = 0.0
            else:
                # Never actuate an unverified failed iterate. Collision-aware
                # controllers fail safe with emergency braking; unconstrained
                # controllers preserve their previous command.
                emergency_brake = self.collision_avoidance
                a0 = -self.max_accel if emergency_brake else self._previous_accel
                delta0 = self._previous_steer
                self._warm_z = None
                self._warm_u = None
                self.last_solve_succeeded = False
                self.last_collision_slack = np.empty((0, self.N))
                self.last_collision_max_slack = math.inf if emergency_brake else 0.0
                self.last_path_boundary_slack = np.empty(0)
                self.last_path_boundary_max_slack = (
                    math.inf
                    if emergency_brake and self._path_boundary_slack is not None
                    else 0.0
                )
            self.last_solver_succeeded = False
            self.last_accepted_feasible_iterate = accepted_feasible_iterate
        else:
            solve_time = time.perf_counter() - solve_started
            stats = self._solverStats(sol)
            a0 = float(sol.value(self._U[0, 0]))
            delta0 = float(sol.value(self._U[1, 0]))
            self._warm_z = np.array(sol.value(self._Z))
            self._warm_u = np.array(sol.value(self._U))
            self.last_solve_succeeded = True
            self.last_solver_succeeded = True
            self.last_accepted_feasible_iterate = False
            if self._collision_slack is not None:
                self.last_collision_slack = np.asarray(
                    sol.value(self._collision_slack), dtype=float
                ).reshape(self.max_obstacles, self.N)
                self.last_collision_max_slack = float(
                    np.max(self.last_collision_slack, initial=0.0)
                )
            else:
                self.last_collision_slack = np.empty((0, self.N))
                self.last_collision_max_slack = 0.0
            if self._path_boundary_slack is not None:
                self.last_path_boundary_slack = np.asarray(
                    sol.value(self._path_boundary_slack), dtype=float
                ).reshape(self.N + 1)
                self.last_path_boundary_max_slack = float(
                    np.max(self.last_path_boundary_slack, initial=0.0)
                )
            else:
                self.last_path_boundary_slack = np.empty(0)
                self.last_path_boundary_max_slack = 0.0

        iterations = stats.get("iter_count")
        try:
            iterations = int(iterations) if iterations is not None else None
        except (TypeError, ValueError):
            iterations = None
        status = stats.get("return_status") or stats.get("unified_return_status")
        self.last_solve_time = float(solve_time)
        self.last_solve_iterations = iterations
        self.last_solve_status = str(status) if status is not None else None
        self.solve_history.append(
            {
                "solve_time_seconds": self.last_solve_time,
                "iterations": self.last_solve_iterations,
                "succeeded": self.last_solve_succeeded,
                "solver_succeeded": self.last_solver_succeeded,
                "accepted_feasible_iterate": self.last_accepted_feasible_iterate,
                "status": self.last_solve_status,
                **failure_diagnostics,
            }
        )

        self._previous_accel = a0
        self._previous_steer = delta0

        if emergency_brake:
            throttle = -1.0
        else:
            throttle = float(np.clip(a0 / self.accel_scale, -1.0, 1.0))
        # Positive steering angle (delta) turns left (increasing heading), while
        # a positive steer command steers right, so negate.
        steer = float(np.clip(-delta0 / self.max_steer_angle, -1.0, 1.0))
        return throttle, steer
