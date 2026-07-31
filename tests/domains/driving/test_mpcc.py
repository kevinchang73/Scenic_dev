import math
from types import SimpleNamespace

import numpy as np
import pytest

from scenic.domains.driving.controllers import (
    MPCCController,
    _footprintCircleGeometry,
    _orientedCircleCenters,
)
from scenic.domains.driving.simulators import (
    DrivingSimulation,
    validateMPCCCollisionPolicy,
)

casadi = pytest.importorskip("casadi")


class TestDrivingSimulation(DrivingSimulation):
    __test__ = False

    def createObjectInSimulator(self, obj):
        pass

    def getProperties(self, obj, properties):
        return {}

    def step(self):
        pass


def make_object(x=0, y=0, heading=0, width=2, length=4.5, vx=0, vy=0, yaw=0):
    return SimpleNamespace(
        position=SimpleNamespace(x=x, y=y),
        velocity=SimpleNamespace(x=vx, y=vy),
        angularVelocity=SimpleNamespace(z=yaw),
        angularSpeed=yaw,
        heading=heading,
        width=width,
        length=length,
    )


def make_simulation(ego, *objects):
    simulation = object.__new__(TestDrivingSimulation)
    simulation.scene = SimpleNamespace(egoObject=ego)
    simulation.objects = list(objects)
    return simulation


def make_obstacle(**kwargs):
    values = dict(
        x=0,
        y=8,
        velocity_x=0,
        velocity_y=0,
        heading=0,
        yaw_rate=0,
        width=2,
        length=4.5,
    )
    values.update(kwargs)
    return SimpleNamespace(**values)


def test_collision_policy_validation():
    for policy in ("none", "all", "ego_asymmetric"):
        assert validateMPCCCollisionPolicy(policy) == policy
    with pytest.raises(ValueError, match="unknown MPCC collision-avoidance policy"):
        validateMPCCCollisionPolicy("invalid")


def test_collision_policy_object_filtering():
    ego = make_object(x=0)
    non_ego = make_object(x=1)
    static = make_object(x=2)
    simulation = make_simulation(ego, ego, non_ego, static)

    assert simulation.getMPCCObstacleStates(ego, "none") == ()
    assert [state.x for state in simulation.getMPCCObstacleStates(ego, "all")] == [
        1,
        2,
    ]
    assert [
        state.x for state in simulation.getMPCCObstacleStates(ego, "ego_asymmetric")
    ] == [1, 2]
    assert [
        state.x for state in simulation.getMPCCObstacleStates(non_ego, "ego_asymmetric")
    ] == [2]


def test_asymmetric_policy_requires_ego():
    agent = make_object()
    simulation = make_simulation(None, agent)
    with pytest.raises(ValueError, match="requires an ego object"):
        simulation.getMPCCObstacleStates(agent, "ego_asymmetric")


def test_three_circle_footprint_covers_box_corners():
    offsets, radius = _footprintCircleGeometry(2, 4.5, count=3)
    assert offsets == pytest.approx((-1.5, 0, 1.5))
    assert radius == pytest.approx(math.hypot(1, 0.75))

    for heading in (0, math.pi / 2, -math.pi / 3):
        centers = _orientedCircleCenters(3, -2, heading, offsets)
        for lateral in (-1, 1):
            for longitudinal in (-2.25, 2.25):
                corner_x = (
                    3 + lateral * math.cos(heading) - longitudinal * math.sin(heading)
                )
                corner_y = (
                    -2 + lateral * math.sin(heading) + longitudinal * math.cos(heading)
                )
                distances = np.hypot(centers[:, 0] - corner_x, centers[:, 1] - corner_y)
                assert np.min(distances) <= radius + 1e-12


def test_obstacle_prediction_uses_velocity_and_yaw_rate():
    controller = MPCCController(
        dt=0.5,
        horizon=2,
        collision_avoidance=True,
        max_obstacles=1,
        vehicle_width=2,
        vehicle_length=4.5,
    )
    obstacle = make_obstacle(
        x=1,
        y=2,
        velocity_x=3,
        velocity_y=-1,
        heading=0,
        yaw_rate=math.pi / 2,
        width=2,
        length=6,
    )

    xs, ys, radii, active = controller._predictObstacleFootprints((obstacle,))
    offsets, radius = _footprintCircleGeometry(2, 6, count=3)
    assert radii[0, 0] == pytest.approx(radius)
    assert active[0, 0] == 1
    # At t=1, heading is pi/2 and the center is (4, 1).
    assert xs[:, 2] == pytest.approx(4 - offsets)
    assert ys[:, 2] == pytest.approx((1, 1, 1))


def test_collision_constraint_changes_trajectory_and_grows_slots():
    path = [(0, 0), (0, 50)]
    baseline = MPCCController(dt=0.1, horizon=8)
    baseline.setReference(path)
    baseline_throttle, baseline_steer = baseline.run_step(0, 0, 0, 5, 10)

    constrained = MPCCController(
        dt=0.1,
        horizon=8,
        collision_avoidance=True,
        max_obstacles=0,
        vehicle_width=2,
        vehicle_length=4.5,
        solver_options={"ipopt.max_iter": 150},
    )
    constrained.setReference(path)
    throttle, steer = constrained.run_step(0, 0, 0, 5, 10, obstacles=(make_obstacle(),))

    assert constrained.max_obstacles >= 1
    assert constrained.last_solve_succeeded
    assert math.isfinite(throttle)
    assert abs(steer) > abs(baseline_steer) + 0.05
    assert constrained.last_collision_max_slack < 1e-3
    assert baseline_throttle > 0


def test_disabled_collision_avoidance_keeps_original_problem_shape():
    default = MPCCController(dt=0.1, horizon=4)
    explicit = MPCCController(dt=0.1, horizon=4, collision_avoidance=False)
    for controller in (default, explicit):
        controller.setReference([(0, 0), (0, 20)])
        assert controller._collision_slack is None
        assert controller.max_obstacles == 0

    assert default.run_step(0, 0, 0, 0, 5) == pytest.approx(
        explicit.run_step(0, 0, 0, 0, 5)
    )


def test_solve_metrics_record_time_iterations_and_controller_identity():
    ego = make_object()
    simulation = make_simulation(ego, ego)
    controller = MPCCController(dt=0.1, horizon=4)
    simulation._registerMPCCController(
        controller, ego, mode="lane_following", collision_avoidance="none"
    )
    controller.setReference([(0, 0), (0, 20)])

    controller.run_step(0, 0, 0, 1, 5)
    controller.run_step(0, 0.1, 0, 1, 5)
    metrics = simulation.getMPCCSolveMetrics()[0]

    assert metrics["controller_index"] == 0
    assert metrics["object_index"] == 0
    assert metrics["mode"] == "lane_following"
    assert metrics["solves"] == 2
    assert metrics["successful_solves"] == 2
    assert metrics["failed_solves"] == 0
    assert metrics["total_solve_time_seconds"] >= 0
    assert metrics["mean_solve_time_seconds"] >= 0
    assert metrics["max_solve_time_seconds"] >= 0
    assert metrics["mean_iterations"] >= 0
    assert metrics["max_iterations"] >= 0
    assert len(metrics["history"]) == 2
    assert all(sample["iterations"] is not None for sample in metrics["history"])
    assert all(sample["status"] for sample in metrics["history"])


def test_steering_rate_limit_applies_across_prediction_horizon():
    controller = MPCCController(
        dt=0.1,
        horizon=8,
        max_steer_rate=0.5,
        collision_avoidance=True,
        max_obstacles=1,
        solver_options={"ipopt.max_iter": 150},
    )
    controller.setReference([(0, 0), (0, 50)])

    _, steer = controller.run_step(0, 0, 0, 5, 10, obstacles=(make_obstacle(x=0, y=8),))
    steering_angles = controller._warm_u[1]
    all_deltas = np.diff(np.concatenate(([0.0], steering_angles)))

    assert controller.last_solve_succeeded
    assert abs(steer) <= 0.5 * 0.1 / controller.max_steer_angle + 1e-6
    assert np.max(np.abs(all_deltas)) <= 0.5 * 0.1 + 1e-6


def test_steering_rate_limit_must_be_positive():
    with pytest.raises(ValueError, match="max_steer_rate must be positive"):
        MPCCController(max_steer_rate=0)

    with pytest.raises(ValueError, match="failure_feasibility_tol must be nonnegative"):
        MPCCController(failure_feasibility_tol=-1)

    with pytest.raises(ValueError, match="path_boundary_margin must be positive"):
        MPCCController(path_boundary_margin=0)


def test_path_boundary_limits_collision_avoidance_departure():
    controller = MPCCController(
        dt=0.1,
        horizon=8,
        collision_avoidance=True,
        max_obstacles=1,
        vehicle_width=2,
        vehicle_length=4.5,
        path_boundary_margin=0.5,
        solver_options={"ipopt.max_iter": 150},
    )
    controller.setReference([(0, 0), (0, 50)])

    controller.run_step(
        0,
        0,
        0,
        5,
        10,
        obstacles=(make_obstacle(x=0, y=8),),
    )

    assert controller.last_solve_succeeded
    contouring_errors = np.abs(controller._warm_z[0])
    assert np.max(contouring_errors) <= (
        controller.path_boundary_margin + controller.last_path_boundary_max_slack + 1e-6
    )
    assert controller.last_path_boundary_max_slack < 2e-3


def test_feasible_failed_iterate_is_accepted_without_emergency_braking():
    controller = MPCCController(
        dt=0.1,
        horizon=4,
        collision_avoidance=True,
        max_obstacles=1,
    )
    controller.setReference([(0, 0), (0, 20)])
    controller.run_step(0, 0, 0, 0, 0, obstacles=())
    assert controller.last_solver_succeeded

    original_opti = controller._opti

    class FailingOpti:
        class Debug:
            @staticmethod
            def value(variable):
                if variable is controller._Z:
                    return np.zeros((5, controller.N + 1))
                if variable is controller._U:
                    return np.zeros((3, controller.N))
                if variable is controller._collision_slack:
                    return np.zeros((controller.max_obstacles, controller.N))
                raise AssertionError("unexpected debug variable")

        debug = Debug()

        def __getattr__(self, name):
            return getattr(original_opti, name)

        def solve(self):
            raise RuntimeError("forced non-successful termination")

    controller._opti = FailingOpti()
    controller._debugSolveDiagnostics = lambda: {
        "debug_objective": 0.0,
        "debug_max_constraint_violation": 0.0,
    }
    controller.solve_history.clear()
    throttle, steer = controller.run_step(0, 0, 0, 0, 0, obstacles=())
    metrics = controller.getSolveMetrics()

    assert throttle == pytest.approx(0, abs=1e-6)
    assert steer == pytest.approx(0, abs=1e-6)
    assert controller.last_solve_succeeded
    assert not controller.last_solver_succeeded
    assert controller.last_accepted_feasible_iterate
    assert metrics["successful_solves"] == 1
    assert metrics["failed_solves"] == 0
    assert metrics["solver_failed_solves"] == 1
    assert metrics["accepted_feasible_iterates"] == 1


def test_collision_solver_failure_commands_emergency_braking():
    controller = MPCCController(
        dt=0.1,
        horizon=4,
        collision_avoidance=True,
        max_obstacles=1,
    )
    controller.setReference([(0, 0), (0, 20)])

    original_opti = controller._opti

    class FailingOpti:
        def __getattr__(self, name):
            return getattr(original_opti, name)

        def solve(self):
            raise RuntimeError("forced solver failure")

    controller._opti = FailingOpti()
    throttle, steer = controller.run_step(0, 0, 0, 2, 5, obstacles=())

    assert throttle == -1
    assert steer == 0
    assert not controller.last_solve_succeeded
    assert math.isinf(controller.last_collision_max_slack)
    assert controller.getSolveMetrics()["failed_solves"] == 1
    assert controller.solve_history[0]["succeeded"] is False
