"""Command-level safety supervisor for moving field workers."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from terrascout.sim.geometry import Point2D, Pose2D
from terrascout.sim.world import LidarDetection


@dataclass(frozen=True)
class SafetyDecision:
    """Wheel command after safety supervision."""

    left_mps: float
    right_mps: float
    scale: float
    min_clearance_m: float
    intervened: bool
    stopped: bool


class SafetySupervisor:
    """Slow or stop the rover when perceived workers are too close."""

    def __init__(self, stop_radius_m: float = 1.25, slow_radius_m: float = 2.4) -> None:
        if stop_radius_m <= 0.0 or slow_radius_m <= stop_radius_m:
            raise ValueError("Expected 0 < stop_radius_m < slow_radius_m")
        self.stop_radius_m = stop_radius_m
        self.slow_radius_m = slow_radius_m

    def supervise(
        self,
        pose: Pose2D,
        left_mps: float,
        right_mps: float,
        worker_detections: list[LidarDetection],
        predicted_workers: list[tuple[int, float, float]],
    ) -> SafetyDecision:
        """Scale wheel commands based on perceived and predicted worker clearance."""

        worker_points = [Point2D(det.x, det.y) for det in worker_detections if det.kind == "worker"]
        worker_points.extend(Point2D(x, y) for _, x, y in predicted_workers)
        min_clearance = min(
            (hypot(point.x - pose.x, point.y - pose.y) for point in worker_points),
            default=float("inf"),
        )
        scale = self._scale_for_clearance(min_clearance)
        return SafetyDecision(
            left_mps=left_mps * scale,
            right_mps=right_mps * scale,
            scale=scale,
            min_clearance_m=min_clearance,
            intervened=scale < 0.999,
            stopped=scale == 0.0,
        )

    def _scale_for_clearance(self, clearance_m: float) -> float:
        if clearance_m <= self.stop_radius_m:
            return 0.0
        if clearance_m >= self.slow_radius_m:
            return 1.0
        return (clearance_m - self.stop_radius_m) / (self.slow_radius_m - self.stop_radius_m)
