"""EKF-SLAM with range/bearing tree landmarks.

The implementation is deliberately compact: the state is
`[x, y, theta, l0_x, l0_y, ...]`, landmarks are associated by nearest predicted
position, and new landmarks are appended online. It gives TerraScout a real
state/covariance SLAM back end without pulling in a heavyweight robotics
framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, sin

import numpy as np
from numpy.typing import NDArray

from terrascout.sim.geometry import Point2D, Pose2D, wrap_angle
from terrascout.sim.world import LocalLidarDetection


@dataclass(frozen=True)
class EkfSlamConfig:
    """Noise and association settings for EKF-SLAM."""

    motion_linear_sigma: float = 0.05
    motion_angular_sigma: float = 0.03
    range_sigma: float = 0.08
    bearing_sigma: float = 0.03
    association_gate_m: float = 0.8
    mahalanobis_gate: float = 9.21
    innovation_gate: float = 1.5
    max_landmarks: int = 160
    max_updates_per_scan: int = 16


class EkfSlam:
    """Small EKF-SLAM estimator for orchard tree landmarks."""

    def __init__(self, initial_pose: Pose2D, config: EkfSlamConfig | None = None) -> None:
        self.config = config or EkfSlamConfig()
        self.mean = np.array([initial_pose.x, initial_pose.y, initial_pose.theta], dtype=float)
        self.covariance = np.diag([0.20, 0.20, 0.08]).astype(float)
        self.landmark_observations: list[int] = []

    @property
    def pose(self) -> Pose2D:
        """Current robot pose estimate."""

        return Pose2D(float(self.mean[0]), float(self.mean[1]), float(self.mean[2]))

    @property
    def landmark_count(self) -> int:
        return (len(self.mean) - 3) // 2

    def landmarks(self) -> list[Point2D]:
        """Return landmark means as points."""

        return [
            Point2D(float(self.mean[3 + 2 * idx]), float(self.mean[4 + 2 * idx]))
            for idx in range(self.landmark_count)
        ]

    def predict(self, linear_mps: float, angular_rps: float, dt: float) -> None:
        """EKF motion prediction for a unicycle model."""

        self._stabilize_covariance()
        theta = float(self.mean[2])
        theta_mid = theta + 0.5 * angular_rps * dt
        self.mean[0] += linear_mps * cos(theta_mid) * dt
        self.mean[1] += linear_mps * sin(theta_mid) * dt
        self.mean[2] = wrap_angle(float(self.mean[2] + angular_rps * dt))

        n = len(self.mean)
        g = np.eye(n, dtype=float)
        g[0, 2] = -linear_mps * sin(theta_mid) * dt
        g[1, 2] = linear_mps * cos(theta_mid) * dt

        r = np.zeros((n, n), dtype=float)
        r[0, 0] = self.config.motion_linear_sigma**2
        r[1, 1] = self.config.motion_linear_sigma**2
        r[2, 2] = self.config.motion_angular_sigma**2
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            self.covariance = g @ self.covariance @ g.T + r
        self._stabilize_covariance()

    def update(self, detections: list[LocalLidarDetection]) -> None:
        """Associate and update tree detections."""

        update_count = 0
        for detection in detections:
            if detection.kind != "tree":
                continue
            landmark_index = self._associate(detection)
            if landmark_index is None:
                self._append_landmark(detection)
            else:
                self._update_landmark(landmark_index, detection)
            update_count += 1
            if update_count >= self.config.max_updates_per_scan:
                break

    def _associate(self, detection: LocalLidarDetection) -> int | None:
        if self.landmark_count == 0:
            return None
        observed = self._detection_to_global(detection)
        best_index: int | None = None
        best_distance = float("inf")
        best_mahalanobis = float("inf")
        for idx, landmark in enumerate(self.landmarks()):
            distance_m = hypot(observed[0] - landmark.x, observed[1] - landmark.y)
            if distance_m > self.config.association_gate_m:
                continue
            measurement = self._measurement_terms(idx, detection, full_jacobian=False)
            if measurement is None:
                continue
            innovation, h, covariance_indices = measurement
            mahalanobis = self._mahalanobis_distance(innovation, h, covariance_indices)
            if mahalanobis is None:
                continue
            if mahalanobis < best_mahalanobis:
                best_index = idx
                best_distance = distance_m
                best_mahalanobis = mahalanobis
            elif mahalanobis == best_mahalanobis and distance_m < best_distance:
                best_index = idx
                best_distance = distance_m

        return best_index if best_mahalanobis <= self.config.mahalanobis_gate else None

    def _append_landmark(self, detection: LocalLidarDetection) -> None:
        if self.landmark_count >= self.config.max_landmarks:
            return
        landmark_xy = self._detection_to_global(detection)
        old_n = len(self.mean)
        self.mean = np.concatenate([self.mean, landmark_xy])

        new_covariance = np.zeros((old_n + 2, old_n + 2), dtype=float)
        new_covariance[:old_n, :old_n] = self.covariance
        new_covariance[old_n:, old_n:] = np.eye(2, dtype=float) * 0.50
        self.covariance = new_covariance
        self.landmark_observations.append(1)

    def _update_landmark(self, landmark_index: int, detection: LocalLidarDetection) -> None:
        self._stabilize_covariance()
        measurement = self._measurement_terms(landmark_index, detection, full_jacobian=True)
        if measurement is None:
            return
        innovation, h, _covariance_indices = measurement
        if (
            abs(float(innovation[0])) > self.config.innovation_gate
            or abs(float(innovation[1])) > self.config.innovation_gate
        ):
            return
        measurement_noise = self._measurement_noise()
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            innovation_covariance = h @ self.covariance @ h.T + measurement_noise
        if (
            not np.all(np.isfinite(innovation_covariance))
            or np.linalg.cond(innovation_covariance) > 1e10
        ):
            return
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            kalman_gain = self.covariance @ h.T @ np.linalg.inv(innovation_covariance)
        if not np.all(np.isfinite(kalman_gain)):
            return
        kalman_gain = np.clip(kalman_gain, -5.0, 5.0)
        self.mean = self.mean + kalman_gain @ innovation
        self.mean[2] = wrap_angle(float(self.mean[2]))
        identity = np.eye(len(self.mean), dtype=float)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            residual_projection = np.clip(identity - kalman_gain @ h, -5.0, 5.0)
            self.covariance = (
                residual_projection @ self.covariance @ residual_projection.T
                + kalman_gain @ measurement_noise @ kalman_gain.T
            )
        self._stabilize_covariance()
        self.landmark_observations[landmark_index] += 1

    def _measurement_terms(
        self,
        landmark_index: int,
        detection: LocalLidarDetection,
        full_jacobian: bool,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], list[int] | None] | None:
        landmark_offset = 3 + 2 * landmark_index
        dx = float(self.mean[landmark_offset] - self.mean[0])
        dy = float(self.mean[landmark_offset + 1] - self.mean[1])
        q = max(dx * dx + dy * dy, 1e-9)
        if q < 0.01:
            return None
        predicted_range = q**0.5
        predicted_bearing = wrap_angle(atan2(dy, dx) - float(self.mean[2]))
        innovation = np.array(
            [
                detection.range_m - predicted_range,
                wrap_angle(detection.bearing_rad - predicted_bearing),
            ],
            dtype=float,
        )
        if (
            not np.all(np.isfinite(innovation))
            or abs(float(innovation[0])) > self.config.innovation_gate
            or abs(float(innovation[1])) > self.config.innovation_gate
        ):
            return None

        covariance_indices = [0, 1, 2, landmark_offset, landmark_offset + 1]
        if full_jacobian:
            h = np.zeros((2, len(self.mean)), dtype=float)
            robot_x = 0
            robot_y = 1
            robot_theta = 2
            landmark_x = landmark_offset
            landmark_y = landmark_offset + 1
        else:
            h = np.zeros((2, len(covariance_indices)), dtype=float)
            robot_x = 0
            robot_y = 1
            robot_theta = 2
            landmark_x = 3
            landmark_y = 4
        sqrt_q = q**0.5
        h[0, robot_x] = -dx / sqrt_q
        h[0, robot_y] = -dy / sqrt_q
        h[0, landmark_x] = dx / sqrt_q
        h[0, landmark_y] = dy / sqrt_q
        h[1, robot_x] = dy / q
        h[1, robot_y] = -dx / q
        h[1, robot_theta] = -1.0
        h[1, landmark_x] = -dy / q
        h[1, landmark_y] = dx / q
        return innovation, h, None if full_jacobian else covariance_indices

    def _mahalanobis_distance(
        self,
        innovation: NDArray[np.float64],
        h: NDArray[np.float64],
        covariance_indices: list[int] | None,
    ) -> float | None:
        measurement_noise = self._measurement_noise()
        covariance = (
            self.covariance
            if covariance_indices is None
            else self.covariance[np.ix_(covariance_indices, covariance_indices)]
        )
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            innovation_covariance = h @ covariance @ h.T + measurement_noise
        if (
            not np.all(np.isfinite(innovation_covariance))
            or np.linalg.cond(innovation_covariance) > 1e10
        ):
            return None
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            weighted = np.linalg.solve(innovation_covariance, innovation)
        if not np.all(np.isfinite(weighted)):
            return None
        return float(innovation.T @ weighted)

    def _measurement_noise(self) -> NDArray[np.float64]:
        return np.diag([self.config.range_sigma**2, self.config.bearing_sigma**2])

    def _detection_to_global(self, detection: LocalLidarDetection) -> NDArray[np.float64]:
        theta = float(self.mean[2] + detection.bearing_rad)
        return np.array(
            [
                self.mean[0] + detection.range_m * cos(theta),
                self.mean[1] + detection.range_m * sin(theta),
            ],
            dtype=float,
        )

    def _stabilize_covariance(self) -> None:
        self.covariance = np.nan_to_num(self.covariance, nan=0.0, posinf=100.0, neginf=-100.0)
        self.covariance = np.clip(self.covariance, -100.0, 100.0)
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        diagonal = np.diag(self.covariance).copy()
        diagonal[diagonal < 1e-8] = 1e-8
        diagonal[diagonal > 100.0] = 100.0
        np.fill_diagonal(self.covariance, diagonal)
