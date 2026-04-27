"""PID controllers for waypoint tracking."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2

from terrascout.sim.geometry import Point2D, Pose2D, distance, wrap_angle


@dataclass
class PID:
    """A compact PID controller with integral clamping."""

    kp: float
    ki: float
    kd: float
    integral_limit: float = 1.0

    _integral: float = 0.0
    _previous_error: float | None = None

    def reset(self) -> None:
        self._integral = 0.0
        self._previous_error = None

    def update(self, error: float, dt: float) -> float:
        self._integral = max(
            -self.integral_limit,
            min(self.integral_limit, self._integral + error * dt),
        )
        derivative = 0.0 if self._previous_error is None else (error - self._previous_error) / dt
        self._previous_error = error
        return self.kp * error + self.ki * self._integral + self.kd * derivative


@dataclass
class DriveController:
    """Twin-loop waypoint controller that emits left/right wheel speeds."""

    heading_pid: PID
    speed_pid: PID
    wheel_base_m: float = 0.72
    cruise_speed_mps: float = 1.0
    max_wheel_speed_mps: float = 1.6
    slow_radius_m: float = 1.1

    @classmethod
    def default(cls) -> "DriveController":
        return cls(
            heading_pid=PID(kp=2.95, ki=0.0, kd=0.05, integral_limit=0.5),
            speed_pid=PID(kp=1.2, ki=0.05, kd=0.02, integral_limit=1.0),
        )

    def wheel_commands(self, pose: Pose2D, waypoint: Point2D, dt: float) -> tuple[float, float]:
        """Compute differential wheel commands for the next waypoint."""

        dist = distance(pose, waypoint)
        target_heading = atan2(waypoint.y - pose.y, waypoint.x - pose.x)
        heading_error = wrap_angle(target_heading - pose.theta)
        target_speed = min(self.cruise_speed_mps, self.cruise_speed_mps * dist / self.slow_radius_m)
        if abs(heading_error) > 1.2:
            target_speed *= 0.25
        linear = self.speed_pid.update(target_speed, dt)
        angular = self.heading_pid.update(heading_error, dt)

        left = linear - 0.5 * angular * self.wheel_base_m
        right = linear + 0.5 * angular * self.wheel_base_m
        return self._clip(left), self._clip(right)

    def _clip(self, value: float) -> float:
        return max(-self.max_wheel_speed_mps, min(self.max_wheel_speed_mps, value))
