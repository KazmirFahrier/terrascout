"""Small geometry helpers shared across the autonomy stack."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, pi, sin


@dataclass(frozen=True)
class Pose2D:
    """Planar SE(2) pose."""

    x: float
    y: float
    theta: float = 0.0


@dataclass(frozen=True)
class Point2D:
    """Planar point."""

    x: float
    y: float


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""

    return (angle + pi) % (2.0 * pi) - pi


def distance(a: Point2D | Pose2D, b: Point2D | Pose2D) -> float:
    """Euclidean distance between two planar objects with x/y fields."""

    return hypot(a.x - b.x, a.y - b.y)


def bearing(a: Pose2D, b: Point2D | Pose2D) -> float:
    """Bearing from pose ``a`` to point ``b`` in the global frame."""

    return atan2(b.y - a.y, b.x - a.x)


def step_from_heading(pose: Pose2D, speed: float, dt: float) -> Pose2D:
    """Move a pose forward along its heading."""

    return Pose2D(
        x=pose.x + speed * cos(pose.theta) * dt,
        y=pose.y + speed * sin(pose.theta) * dt,
        theta=pose.theta,
    )

