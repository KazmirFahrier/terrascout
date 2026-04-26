"""Differential-drive rover dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin

from terrascout.sim.geometry import Pose2D, wrap_angle


@dataclass
class DifferentialDriveRover:
    """A lightweight differential-drive model with bounded wheel commands."""

    pose: Pose2D
    wheel_base_m: float = 0.72
    max_wheel_speed_mps: float = 1.6
    slip_fraction: float = 0.0

    left_velocity_mps: float = 0.0
    right_velocity_mps: float = 0.0

    def command(self, left_mps: float, right_mps: float) -> None:
        """Set saturated wheel velocity commands."""

        self.left_velocity_mps = max(-self.max_wheel_speed_mps, min(self.max_wheel_speed_mps, left_mps))
        self.right_velocity_mps = max(
            -self.max_wheel_speed_mps, min(self.max_wheel_speed_mps, right_mps)
        )

    def step(self, dt: float) -> Pose2D:
        """Advance the rover by one simulation step."""

        slip_scale = max(0.0, 1.0 - self.slip_fraction)
        left = self.left_velocity_mps * slip_scale
        right = self.right_velocity_mps * slip_scale
        linear = 0.5 * (left + right)
        angular = (right - left) / self.wheel_base_m

        theta_mid = self.pose.theta + 0.5 * angular * dt
        x = self.pose.x + linear * cos(theta_mid) * dt
        y = self.pose.y + linear * sin(theta_mid) * dt
        theta = wrap_angle(self.pose.theta + angular * dt)
        self.pose = Pose2D(x=x, y=y, theta=theta)
        return self.pose

