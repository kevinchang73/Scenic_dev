"""Abstract interface to simulators supporting the driving domain."""

from dataclasses import dataclass
import math

from scenic.core.simulators import Simulation, Simulator
from scenic.domains.driving.controllers import (
    MPCCController,
    PIDLateralController,
    PIDLongitudinalController,
)

MPCC_COLLISION_POLICIES = frozenset(("none", "all", "ego_asymmetric"))


def validateMPCCCollisionPolicy(policy):
    """Validate and normalize an MPCC collision-avoidance policy name."""
    if policy not in MPCC_COLLISION_POLICIES:
        choices = ", ".join(sorted(MPCC_COLLISION_POLICIES))
        raise ValueError(
            f"unknown MPCC collision-avoidance policy {policy!r}; "
            f"expected one of: {choices}"
        )
    return policy


@dataclass(frozen=True)
class MPCCObstacleState:
    """Snapshot of a physical object's planar state for MPCC prediction."""

    x: float
    y: float
    velocity_x: float
    velocity_y: float
    heading: float
    yaw_rate: float
    width: float
    length: float


def _finiteFloat(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


class DrivingSimulator(Simulator):
    """A `Simulator` supporting the driving domain."""

    pass


class DrivingSimulation(Simulation):
    """A `Simulation` with a simulator supporting the driving domain.

    This subclass of `Simulation` provides no special behavior by itself; it
    just provides convenience methods for creating controllers to be used by
    `FollowLaneBehavior` and related behaviors, so that the parameters of these
    controllers can be customized for different simulators.
    """

    def _registerMPCCController(self, controller, agent, mode, collision_avoidance):
        """Register an MPCC controller so its solve metrics can be collected."""
        registrations = getattr(self, "_mpccControllerRegistrations", None)
        if registrations is None:
            registrations = []
            self._mpccControllerRegistrations = registrations
        object_index = next(
            (index for index, obj in enumerate(self.objects) if obj is agent), None
        )
        registrations.append(
            {
                "controller": controller,
                "object_index": object_index,
                "mode": mode,
                "collision_avoidance": collision_avoidance,
            }
        )
        return controller

    def getMPCCSolveMetrics(self, include_history=True):
        """Return solve metrics for every MPCC controller created this simulation."""
        metrics = []
        for controller_index, registration in enumerate(
            getattr(self, "_mpccControllerRegistrations", ())
        ):
            controller = registration["controller"]
            summary = {
                "controller_index": controller_index,
                "object_index": registration["object_index"],
                "mode": registration["mode"],
                "collision_avoidance": registration["collision_avoidance"],
                "horizon": controller.N,
                "max_obstacles": controller.max_obstacles,
                "footprint_circles": controller.footprint_circles,
            }
            summary.update(controller.getSolveMetrics(include_history=include_history))
            metrics.append(summary)
        return metrics

    def getLaneFollowingControllers(self, agent):
        """Get longitudinal and lateral controllers for lane following.

        The default controllers are simple PID controllers with parameters that
        work reasonably well for cars in simulators with realistic physics. See the
        classes `PIDLongitudinalController` and `PIDLateralController` for details,
        and `NewtonianSimulation` for an example of how to override this function.

        Returns:
            A pair of controllers for throttle and steering respectively.
        """
        dt = self.timestep
        if agent.isCar:
            lon_controller = PIDLongitudinalController(K_P=0.5, K_D=0.1, K_I=0.7, dt=dt)
            lat_controller = PIDLateralController(K_P=0.2, K_D=0.1, K_I=0.0, dt=dt)
        else:
            lon_controller = PIDLongitudinalController(
                K_P=0.25, K_D=0.025, K_I=0.0, dt=dt
            )
            lat_controller = PIDLateralController(K_P=0.2, K_D=0.1, K_I=0.0, dt=dt)
        return lon_controller, lat_controller

    def getTurningControllers(self, agent):
        """Get longitudinal and lateral controllers for turning."""
        dt = self.timestep
        if agent.isCar:
            lon_controller = PIDLongitudinalController(K_P=0.5, K_D=0.1, K_I=0.7, dt=dt)
            lat_controller = PIDLateralController(K_P=0.8, K_D=0.2, K_I=0.0, dt=dt)
        else:
            lon_controller = PIDLongitudinalController(
                K_P=0.25, K_D=0.025, K_I=0.0, dt=dt
            )
            lat_controller = PIDLateralController(K_P=0.4, K_D=0.1, K_I=0.0, dt=dt)
        return lon_controller, lat_controller

    def getLaneChangingControllers(self, agent):
        """Get longitudinal and lateral controllers for lane changing."""
        dt = self.timestep
        if agent.isCar:
            lon_controller = PIDLongitudinalController(K_P=0.5, K_D=0.1, K_I=0.7, dt=dt)
            lat_controller = PIDLateralController(K_P=0.08, K_D=0.3, K_I=0.0, dt=dt)
        else:
            lon_controller = PIDLongitudinalController(
                K_P=0.25, K_D=0.025, K_I=0.0, dt=dt
            )
            lat_controller = PIDLateralController(K_P=0.1, K_D=0.3, K_I=0.0, dt=dt)
        return lon_controller, lat_controller

    def getMPCCObstacleStates(self, agent, collision_avoidance="none"):
        """Snapshot objects constrained against by an agent's MPCC.

        Under ``"all"``, every other physical object participates. Under
        ``"ego_asymmetric"``, the ego constrains against every other object,
        while non-ego agents constrain against every object except themselves
        and the ego. The returned order follows :attr:`Simulation.objects` so
        CasADi obstacle slots remain deterministic.
        """
        policy = validateMPCCCollisionPolicy(collision_avoidance)
        if policy == "none":
            return ()

        ego = self.scene.egoObject
        if policy == "ego_asymmetric" and ego is None:
            raise ValueError(
                'MPCC collision policy "ego_asymmetric" requires an ego object'
            )

        states = []
        for obj in self.objects:
            if obj is agent:
                continue
            if policy == "ego_asymmetric" and agent is not ego and obj is ego:
                continue

            velocity = obj.velocity
            angular_velocity = obj.angularVelocity
            yaw_rate = _finiteFloat(getattr(angular_velocity, "z", 0.0))
            if abs(yaw_rate) <= 1e-12:
                # The Newtonian simulator stores its signed planar yaw rate in
                # angularSpeed while leaving angularVelocity at zero.
                yaw_rate = _finiteFloat(obj.angularSpeed)

            states.append(
                MPCCObstacleState(
                    x=_finiteFloat(obj.position.x),
                    y=_finiteFloat(obj.position.y),
                    velocity_x=_finiteFloat(velocity.x),
                    velocity_y=_finiteFloat(velocity.y),
                    heading=_finiteFloat(obj.heading),
                    yaw_rate=yaw_rate,
                    width=max(0.0, _finiteFloat(obj.width)),
                    length=max(0.0, _finiteFloat(obj.length)),
                )
            )
        return tuple(states)

    def _getMPCCCollisionOptions(
        self, agent, collision_avoidance="none", collision_margin=0.25
    ):
        policy = validateMPCCCollisionPolicy(collision_avoidance)
        collision_margin = _finiteFloat(collision_margin)
        if collision_margin < 0:
            raise ValueError("MPCC collision margin must be nonnegative")
        return dict(
            collision_avoidance=policy != "none",
            vehicle_width=float(agent.width),
            vehicle_length=float(agent.length),
            max_obstacles=len(self.getMPCCObstacleStates(agent, policy)),
            collision_margin=collision_margin,
        )

    def getMPCCController(
        self,
        agent,
        mode="lane_following",
        collision_avoidance="none",
        collision_margin=0.25,
    ):
        """Get a Model Predictive Contouring Controller for path tracking.

        The MPCC is a single nonlinear model-predictive controller that jointly
        computes throttle and steering; it is used by the ``*BehaviorMPCC``
        driving behaviors as an alternative to the PID controllers. This default
        implementation returns a `MPCCController` with parameters that work
        reasonably well for cars in simulators with realistic physics. See
        `MPCCController` for details, and `NewtonianSimulation` for an example of
        how to override this function for a particular simulator.

        Using it requires the optional ``casadi`` dependency (``pip install
        scenic[mpcc]``).

        Arguments:
            agent: The agent that will use the controller.
            mode: One of ``"lane_following"``, ``"turning"`` or
                ``"lane_changing"``, allowing per-maneuver tuning.
            collision_avoidance: One of ``"none"``, ``"all"``, or
                ``"ego_asymmetric"``.
            collision_margin: Additional pairwise footprint clearance in meters.

        Returns:
            An `MPCCController` configured for the given agent and mode.
        """
        dt = self.timestep
        wheelbase = 0.6 * agent.length
        collision_options = self._getMPCCCollisionOptions(
            agent, collision_avoidance, collision_margin
        )
        if mode == "turning":
            controller = MPCCController(
                wheelbase=wheelbase,
                dt=dt,
                horizon=12,
                q_c=12.0,
                r_dsteer=1.0,
                **collision_options,
            )
        elif mode == "lane_changing":
            controller = MPCCController(
                wheelbase=wheelbase,
                dt=dt,
                horizon=15,
                q_c=6.0,
                q_l=6.0,
                **collision_options,
            )
        else:
            controller = MPCCController(
                wheelbase=wheelbase, dt=dt, horizon=12, **collision_options
            )
        return self._registerMPCCController(controller, agent, mode, collision_avoidance)
