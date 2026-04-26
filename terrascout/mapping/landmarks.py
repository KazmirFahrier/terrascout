"""Online tree-landmark mapper.

The MVP mapper keeps a small Gaussian estimate per detected tree trunk. It is
not yet a full EKF-SLAM back end, but it gives the stack a real map-estimation
surface that can be swapped for EKF-SLAM without changing callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin

import numpy as np
from numpy.typing import NDArray

from terrascout.sim.geometry import Point2D, Pose2D
from terrascout.sim.world import LocalLidarDetection


@dataclass
class LandmarkEstimate:
    """Estimated tree landmark mean/covariance."""

    mean: NDArray[np.float64]
    covariance: NDArray[np.float64]
    observations: int = 1

    @property
    def point(self) -> Point2D:
        return Point2D(float(self.mean[0]), float(self.mean[1]))


class LandmarkMapper:
    """Nearest-neighbor Gaussian landmark mapper."""

    def __init__(self, association_gate_m: float = 0.75) -> None:
        self.association_gate_m = association_gate_m
        self.landmarks: list[LandmarkEstimate] = []

    def update(self, pose: Pose2D, detections: list[LocalLidarDetection]) -> list[LandmarkEstimate]:
        """Update the landmark map from local tree detections."""

        for detection in detections:
            if detection.kind != "tree":
                continue
            measurement = np.array(
                [
                    pose.x + detection.range_m * cos(pose.theta + detection.bearing_rad),
                    pose.y + detection.range_m * sin(pose.theta + detection.bearing_rad),
                ],
                dtype=float,
            )
            idx = self._associate(measurement)
            if idx is None:
                self.landmarks.append(
                    LandmarkEstimate(
                        mean=measurement,
                        covariance=np.eye(2, dtype=float) * 0.16,
                    )
                )
            else:
                landmark = self.landmarks[idx]
                measurement_covariance = np.eye(2, dtype=float) * 0.04
                innovation_covariance = landmark.covariance + measurement_covariance
                gain = landmark.covariance @ np.linalg.inv(innovation_covariance)
                landmark.mean = landmark.mean + gain @ (measurement - landmark.mean)
                landmark.covariance = (np.eye(2, dtype=float) - gain) @ landmark.covariance
                landmark.observations += 1
        return self.landmarks

    def points(self) -> list[Point2D]:
        """Return mapped tree landmarks as points."""

        return [landmark.point for landmark in self.landmarks]

    def _associate(self, measurement: NDArray[np.float64]) -> int | None:
        if not self.landmarks:
            return None
        distances = [float(np.linalg.norm(landmark.mean - measurement)) for landmark in self.landmarks]
        idx = int(np.argmin(distances))
        return idx if distances[idx] <= self.association_gate_m else None

