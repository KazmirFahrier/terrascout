"""Shared circular footprint geometry and continuous collision evaluation."""

from __future__ import annotations

from math import hypot

from terrascout.sim.geometry import Point2D, Pose2D

ROVER_RADIUS_M = 0.45
TREE_RADIUS_M = 0.18
WORKER_RADIUS_M = 0.35


def segment_clearance(
    start: Point2D | Pose2D,
    end: Point2D | Pose2D,
    point: Point2D | Pose2D,
) -> float:
    """Minimum centre distance along a line segment, including both endpoints."""
    dx, dy = end.x - start.x, end.y - start.y
    length_sq = dx * dx + dy * dy
    fraction = (
        0.0
        if length_sq == 0.0
        else max(0.0, min(1.0, ((point.x - start.x) * dx + (point.y - start.y) * dy) / length_sq))
    )
    return hypot(start.x + fraction * dx - point.x, start.y + fraction * dy - point.y)


def moving_clearance(
    start: Pose2D,
    end: Pose2D,
    obstacle_start: Point2D,
    obstacle_end: Point2D,
) -> float:
    """Minimum separation for simultaneous linear motion over one simulator tick."""
    return segment_clearance(
        Point2D(start.x - obstacle_start.x, start.y - obstacle_start.y),
        Point2D(end.x - obstacle_end.x, end.y - obstacle_end.y),
        Point2D(0.0, 0.0),
    )
