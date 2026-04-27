"""Constant-velocity Kalman tracker for lidar cluster centroids."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from terrascout.sim.world import LidarDetection


@dataclass
class KalmanTrack:
    """One 2D constant-velocity track with a linear Kalman filter."""

    state: NDArray[np.float64]
    covariance: NDArray[np.float64]
    track_id: int
    missed: int = 0
    hits: int = 1

    def predict(self, dt: float, process_noise: float = 0.12) -> None:
        f = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        q = process_noise
        q_mat = np.diag([0.25 * dt**4, 0.25 * dt**4, dt**2, dt**2]) * q
        self.state = f @ self.state
        self.covariance = f @ self.covariance @ f.T + q_mat

    def update(self, measurement_xy: NDArray[np.float64], measurement_noise: float = 0.08) -> None:
        h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        r = np.eye(2) * measurement_noise**2
        residual = measurement_xy - h @ self.state
        s = h @ self.covariance @ h.T + r
        k = self.covariance @ h.T @ np.linalg.inv(s)
        self.state = self.state + k @ residual
        self.covariance = (np.eye(4) - k @ h) @ self.covariance
        self.missed = 0
        self.hits += 1

    @property
    def xy(self) -> NDArray[np.float64]:
        return self.state[:2].copy()

    def mahalanobis_distance_sq(
        self,
        measurement_xy: NDArray[np.float64],
        measurement_noise: float = 0.08,
    ) -> float:
        """Return squared innovation distance for association gating."""

        h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        r = np.eye(2) * measurement_noise**2
        residual = measurement_xy - h @ self.state
        innovation_covariance = h @ self.covariance @ h.T + r
        return float(residual.T @ np.linalg.inv(innovation_covariance) @ residual)


@dataclass
class MultiObjectTracker:
    """Nearest-neighbor multi-object tracker for worker detections."""

    gate_m: float = 1.2
    gate_mahalanobis_sq: float = 9.21
    max_missed: int = 8
    process_noise: float = 0.12
    measurement_noise: float = 0.08
    tracks: list[KalmanTrack] = field(default_factory=list)
    _next_id: int = 1

    def update(self, detections: list[LidarDetection], dt: float) -> list[KalmanTrack]:
        """Predict, associate, update, create, and prune tracks."""

        worker_measurements = [
            np.array([det.x, det.y], dtype=float) for det in detections if det.kind == "worker"
        ]
        for track in self.tracks:
            track.predict(dt, process_noise=self.process_noise)

        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_measurements = set(range(len(worker_measurements)))

        pairs: list[tuple[float, int, int]] = []
        for ti, track in enumerate(self.tracks):
            for mi, measurement in enumerate(worker_measurements):
                dist = float(np.linalg.norm(track.xy - measurement))
                mahalanobis = track.mahalanobis_distance_sq(
                    measurement,
                    measurement_noise=self.measurement_noise,
                )
                if dist <= self.gate_m and mahalanobis <= self.gate_mahalanobis_sq:
                    pairs.append((mahalanobis, ti, mi))
        pairs.sort(key=lambda item: item[0])

        for _, ti, mi in pairs:
            if ti not in unmatched_tracks or mi not in unmatched_measurements:
                continue
            self.tracks[ti].update(worker_measurements[mi], measurement_noise=self.measurement_noise)
            unmatched_tracks.remove(ti)
            unmatched_measurements.remove(mi)

        for ti in unmatched_tracks:
            self.tracks[ti].missed += 1

        for mi in unmatched_measurements:
            measurement = worker_measurements[mi]
            self.tracks.append(
                KalmanTrack(
                    state=np.array([measurement[0], measurement[1], 0.0, 0.0], dtype=float),
                    covariance=np.diag([0.15, 0.15, 0.8, 0.8]),
                    track_id=self._next_id,
                )
            )
            self._next_id += 1

        self.tracks = [track for track in self.tracks if track.missed <= self.max_missed]
        return self.tracks

    def predicted_positions(self, horizon_s: float) -> list[tuple[int, float, float]]:
        """Return track positions extrapolated by ``horizon_s`` seconds."""

        predictions: list[tuple[int, float, float]] = []
        for track in self.tracks:
            x = track.state[0] + track.state[2] * horizon_s
            y = track.state[1] + track.state[3] * horizon_s
            predictions.append((track.track_id, float(x), float(y)))
        return predictions
