"""Monte-Carlo localization against orchard tree landmarks."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, ceil, hypot, pi, sqrt

import numpy as np
from numpy.typing import NDArray

from terrascout.sim.geometry import Point2D, Pose2D, wrap_angle
from terrascout.sim.world import LocalLidarDetection


@dataclass
class ParticleLocalizer:
    """A compact particle filter for global pose recovery."""

    particles: NDArray[np.float64]
    weights: NDArray[np.float64]
    rng: np.random.Generator
    min_particles: int = 150
    max_particles: int = 3000
    kld_epsilon: float = 0.05
    kld_z: float = 2.33
    bin_size: tuple[float, float, float] = (0.35, 0.35, 0.18)

    @classmethod
    def uniform(
        cls,
        count: int,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        seed: int = 7,
    ) -> "ParticleLocalizer":
        rng = np.random.default_rng(seed)
        particles = np.column_stack(
            [
                rng.uniform(x_bounds[0], x_bounds[1], count),
                rng.uniform(y_bounds[0], y_bounds[1], count),
                rng.uniform(-pi, pi, count),
            ]
        )
        return cls(
            particles=particles,
            weights=np.full(count, 1.0 / count),
            rng=rng,
            min_particles=count if count <= 500 else max(100, min(count, int(0.9 * count))),
            max_particles=count,
        )

    @classmethod
    def gaussian(
        cls,
        count: int,
        mean: Pose2D,
        std: tuple[float, float, float],
        seed: int = 7,
    ) -> "ParticleLocalizer":
        """Initialize particles from a coarse Gaussian pose prior."""

        rng = np.random.default_rng(seed)
        particles = np.column_stack(
            [
                rng.normal(mean.x, std[0], count),
                rng.normal(mean.y, std[1], count),
                rng.normal(mean.theta, std[2], count),
            ]
        )
        particles[:, 2] = np.array([wrap_angle(float(theta)) for theta in particles[:, 2]])
        return cls(
            particles=particles,
            weights=np.full(count, 1.0 / count),
            rng=rng,
            min_particles=count if count <= 500 else max(100, min(count, int(0.9 * count))),
            max_particles=count,
        )

    def predict(self, linear_mps: float, angular_rps: float, dt: float) -> None:
        """Propagate particles with a noisy differential-drive motion model."""

        n = len(self.particles)
        linear = linear_mps + self.rng.normal(0.0, 0.04, n)
        angular = angular_rps + self.rng.normal(0.0, 0.025, n)
        theta_mid = self.particles[:, 2] + 0.5 * angular * dt
        self.particles[:, 0] += linear * np.cos(theta_mid) * dt
        self.particles[:, 1] += linear * np.sin(theta_mid) * dt
        self.particles[:, 2] = np.array([wrap_angle(float(t)) for t in self.particles[:, 2] + angular * dt])

    def update(
        self,
        detections: list[LocalLidarDetection],
        landmarks: list[Point2D],
        max_detections: int = 6,
        sigma_m: float = 0.35,
        landmark_margin_m: float = 3.0,
    ) -> None:
        """Weight particles by nearest-landmark residuals from tree detections."""

        tree_observations = [det for det in detections if det.kind == "tree"][:max_detections]
        if not tree_observations or not landmarks:
            return

        landmark_xy = self._candidate_landmarks(landmarks, tree_observations, landmark_margin_m)
        log_weights = np.log(self.weights + 1e-300)
        for obs in tree_observations:
            obs_global_x = self.particles[:, 0] + obs.range_m * np.cos(self.particles[:, 2] + obs.bearing_rad)
            obs_global_y = self.particles[:, 1] + obs.range_m * np.sin(self.particles[:, 2] + obs.bearing_rad)
            residuals = np.column_stack([obs_global_x, obs_global_y])[:, None, :] - landmark_xy[None, :, :]
            nearest_sq = np.min(np.sum(residuals**2, axis=2), axis=1)
            log_weights += -0.5 * nearest_sq / (sigma_m**2)

        log_weights -= float(np.max(log_weights))
        weights = np.exp(log_weights)
        total = float(np.sum(weights))
        self.weights = weights / total if total > 0.0 else np.full(len(weights), 1.0 / len(weights))
        if self.effective_sample_size < 0.55 * len(self.particles):
            self.resample()

    def scan_match_reset(
        self,
        detections: list[LocalLidarDetection],
        landmarks: list[Point2D],
        max_detections: int = 18,
    ) -> Pose2D:
        """Coarse-to-fine lidar scan matching around the current pose prior."""

        tree_observations = [det for det in detections if det.kind == "tree"][:max_detections]
        if not tree_observations or not landmarks:
            return self.estimate()

        landmark_xy = np.array([(landmark.x, landmark.y) for landmark in landmarks], dtype=float)
        center = self.estimate()
        candidates = self._scan_match_candidates(
            center,
            tree_observations,
            landmark_xy,
            xy_step_m=0.5,
            theta_step_rad=pi / 36.0,
            xy_span_m=5.5,
            theta_span_rad=pi / 5.0,
            keep=12,
            separation_m=0.75,
            separation_theta_rad=0.12,
        )
        for xy_step, theta_step, xy_span, theta_span, keep in (
            (0.16, pi / 90.0, 0.8, pi / 18.0, 8),
            (0.05, pi / 180.0, 0.2, pi / 90.0, 1),
        ):
            refined: list[tuple[float, Pose2D]] = []
            for _, candidate in candidates:
                refined.extend(
                    self._scan_match_candidates(
                        candidate,
                        tree_observations,
                        landmark_xy,
                        xy_step_m=xy_step,
                        theta_step_rad=theta_step,
                        xy_span_m=xy_span,
                        theta_span_rad=theta_span,
                        keep=2,
                        separation_m=0.25,
                        separation_theta_rad=0.04,
                    )
                )
            candidates = self._select_scan_match_candidates(
                sorted(refined, key=lambda item: item[0]),
                keep=keep,
                separation_m=0.35,
                separation_theta_rad=0.05,
            )

        best_pose = candidates[0][1]
        self._reset_gaussian(best_pose, std=(0.18, 0.18, 0.06))
        return best_pose

    @property
    def effective_sample_size(self) -> float:
        return float(1.0 / np.sum(self.weights**2))

    def resample(self) -> None:
        """KLD-adaptive low-variance resampling with a small particle-injection tail."""

        occupied_bins = {
            self._particle_bin(particle)
            for particle, weight in zip(self.particles, self.weights)
            if weight > 0.5 / len(self.particles)
        }
        target_count = self._kld_required_sample_count(len(occupied_bins))
        n = min(self.max_particles, max(self.min_particles, target_count))
        positions = (self.rng.random() + np.arange(n)) / n
        cumulative = np.cumsum(self.weights)
        indices = np.searchsorted(cumulative, positions, side="left")
        indices = np.minimum(indices, len(self.particles) - 1)
        self.particles = self.particles[indices].copy()
        n = len(self.particles)
        self.weights = np.full(n, 1.0 / n)
        inject = max(1, int(0.03 * n))
        if inject < n:
            self.particles[-inject:, 0] += self.rng.normal(0.0, 0.25, inject)
            self.particles[-inject:, 1] += self.rng.normal(0.0, 0.25, inject)
            self.particles[-inject:, 2] += self.rng.normal(0.0, 0.08, inject)
            self.particles[-inject:, 2] = np.array(
                [wrap_angle(float(theta)) for theta in self.particles[-inject:, 2]]
            )

    def _particle_bin(self, particle: NDArray[np.float64]) -> tuple[int, int, int]:
        return (
            int(particle[0] / self.bin_size[0]),
            int(particle[1] / self.bin_size[1]),
            int(wrap_angle(float(particle[2])) / self.bin_size[2]),
        )

    def _kld_required_sample_count(self, occupied_bins: int) -> int:
        if occupied_bins <= 1:
            return self.min_particles
        bin_term = 1.0 - 2.0 / (9.0 * (occupied_bins - 1))
        confidence_term = self.kld_z * sqrt(2.0 / (9.0 * (occupied_bins - 1)))
        required = (occupied_bins - 1) / (2.0 * self.kld_epsilon) * (bin_term + confidence_term) ** 3
        return max(self.min_particles, min(self.max_particles, int(ceil(required))))

    def estimate(self) -> Pose2D:
        """Return the weighted mean pose."""

        x = float(np.average(self.particles[:, 0], weights=self.weights))
        y = float(np.average(self.particles[:, 1], weights=self.weights))
        sin_mean = float(np.average(np.sin(self.particles[:, 2]), weights=self.weights))
        cos_mean = float(np.average(np.cos(self.particles[:, 2]), weights=self.weights))
        return Pose2D(x=x, y=y, theta=atan2(sin_mean, cos_mean))

    def _candidate_landmarks(
        self,
        landmarks: list[Point2D],
        observations: list[LocalLidarDetection],
        margin_m: float,
    ) -> NDArray[np.float64]:
        """Return nearby map landmarks when the particle cloud is already localized."""

        landmark_xy = np.array([(landmark.x, landmark.y) for landmark in landmarks], dtype=float)
        if len(landmark_xy) <= 80:
            return landmark_xy

        mean = self.estimate()
        max_observed_range = max(obs.range_m for obs in observations)
        particle_spread = float(
            np.sqrt(
                np.average((self.particles[:, 0] - mean.x) ** 2, weights=self.weights)
                + np.average((self.particles[:, 1] - mean.y) ** 2, weights=self.weights)
            )
        )
        radius_m = max_observed_range + particle_spread + margin_m
        deltas = landmark_xy - np.array([[mean.x, mean.y]])
        mask = np.sum(deltas**2, axis=1) <= radius_m * radius_m
        candidates = landmark_xy[mask]
        return candidates if len(candidates) >= 4 else landmark_xy

    def _scan_match_candidates(
        self,
        center: Pose2D,
        observations: list[LocalLidarDetection],
        landmark_xy: NDArray[np.float64],
        xy_step_m: float,
        theta_step_rad: float,
        xy_span_m: float,
        theta_span_rad: float,
        keep: int,
        separation_m: float,
        separation_theta_rad: float,
    ) -> list[tuple[float, Pose2D]]:
        scored: list[tuple[float, Pose2D]] = []
        for x in np.arange(center.x - xy_span_m, center.x + xy_span_m + 1e-9, xy_step_m):
            for y in np.arange(center.y - xy_span_m, center.y + xy_span_m + 1e-9, xy_step_m):
                for theta in np.arange(
                    center.theta - theta_span_rad,
                    center.theta + theta_span_rad + 1e-9,
                    theta_step_rad,
                ):
                    pose = Pose2D(float(x), float(y), wrap_angle(float(theta)))
                    scored.append((self._scan_match_score(pose, observations, landmark_xy), pose))
        return self._select_scan_match_candidates(
            sorted(scored, key=lambda item: item[0]),
            keep=keep,
            separation_m=separation_m,
            separation_theta_rad=separation_theta_rad,
        )

    def _select_scan_match_candidates(
        self,
        scored: list[tuple[float, Pose2D]],
        keep: int,
        separation_m: float,
        separation_theta_rad: float,
    ) -> list[tuple[float, Pose2D]]:
        selected: list[tuple[float, Pose2D]] = []
        for score, pose in scored:
            if all(
                hypot(pose.x - other.x, pose.y - other.y) > separation_m
                or abs(wrap_angle(pose.theta - other.theta)) > separation_theta_rad
                for _, other in selected
            ):
                selected.append((score, pose))
                if len(selected) >= keep:
                    break
        return selected or scored[:1]

    def _scan_match_score(
        self,
        pose: Pose2D,
        observations: list[LocalLidarDetection],
        landmark_xy: NDArray[np.float64],
    ) -> float:
        total = 0.0
        matched_indices: list[int] = []
        for obs in observations:
            observed_xy = np.array(
                [
                    pose.x + obs.range_m * np.cos(pose.theta + obs.bearing_rad),
                    pose.y + obs.range_m * np.sin(pose.theta + obs.bearing_rad),
                ]
            )
            distances_sq = np.sum((landmark_xy - observed_xy) ** 2, axis=1)
            best_index = int(np.argmin(distances_sq))
            total += min(float(distances_sq[best_index]), 4.0)
            matched_indices.append(best_index)
        duplicate_matches = len(matched_indices) - len(set(matched_indices))
        return total + 0.05 * duplicate_matches

    def _reset_gaussian(self, mean: Pose2D, std: tuple[float, float, float]) -> None:
        count = len(self.particles)
        self.particles = np.column_stack(
            [
                self.rng.normal(mean.x, std[0], count),
                self.rng.normal(mean.y, std[1], count),
                self.rng.normal(mean.theta, std[2], count),
            ]
        )
        self.particles[:, 2] = np.array([wrap_angle(float(theta)) for theta in self.particles[:, 2]])
        self.weights = np.full(count, 1.0 / count)
