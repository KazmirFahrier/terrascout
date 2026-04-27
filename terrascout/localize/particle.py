"""Monte-Carlo localization against orchard tree landmarks."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, ceil, pi, sqrt

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
    ) -> None:
        """Weight particles by nearest-landmark residuals from tree detections."""

        tree_observations = [det for det in detections if det.kind == "tree"][:max_detections]
        if not tree_observations or not landmarks:
            return

        landmark_xy = np.array([(landmark.x, landmark.y) for landmark in landmarks], dtype=float)
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
