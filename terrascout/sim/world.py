"""Procedural orchard world and simple sensor simulation."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, radians, sin

import numpy as np
from numpy.typing import NDArray

from terrascout.sim.geometry import Point2D, Pose2D, distance


@dataclass(frozen=True)
class ScenarioConfig:
    """Parameters for a generated orchard scenario."""

    rows: int = 8
    trees_per_row: int = 14
    row_spacing_m: float = 3.0
    tree_spacing_m: float = 2.4
    worker_count: int = 4
    width_margin_m: float = 2.0
    lidar_range_m: float = 8.0
    random_seed: int = 7


@dataclass
class MovingAgent:
    """A moving worker or field obstacle."""

    position: Point2D
    velocity: NDArray[np.float64]
    radius_m: float = 0.35


@dataclass(frozen=True)
class LidarDetection:
    """A noisy lidar cluster centroid."""

    x: float
    y: float
    kind: str


@dataclass(frozen=True)
class LocalLidarDetection:
    """A range/bearing lidar detection in the rover frame."""

    range_m: float
    bearing_rad: float
    kind: str


class OrchardWorld:
    """Synthetic orchard with tree landmarks, row goals, and moving workers."""

    def __init__(self, config: ScenarioConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)
        self.trees = self._make_trees()
        self.row_goals = self._make_row_goals()
        self.recharge_stations = self._make_recharge_stations()
        self.workers = self._make_workers()
        self.width_m = (config.rows - 1) * config.row_spacing_m + 2.0 * config.width_margin_m
        self.height_m = (config.trees_per_row - 1) * config.tree_spacing_m + 2.0

    def _make_trees(self) -> list[Point2D]:
        trees: list[Point2D] = []
        cfg = self.config
        for row in range(cfg.rows):
            x = cfg.width_margin_m + row * cfg.row_spacing_m
            for idx in range(cfg.trees_per_row):
                y = 1.0 + idx * cfg.tree_spacing_m
                jitter_x, jitter_y = self.rng.normal(0.0, 0.06, size=2)
                trees.append(Point2D(x + float(jitter_x), y + float(jitter_y)))
        return trees

    def _make_row_goals(self) -> list[Point2D]:
        cfg = self.config
        lane_count = max(1, cfg.rows - 1)
        return [
            Point2D(
                cfg.width_margin_m + (lane + 0.5) * cfg.row_spacing_m,
                self.height_y - 1.0 if lane % 2 == 0 else 1.0,
            )
            for lane in range(lane_count)
        ]

    def _make_recharge_stations(self) -> list[Point2D]:
        return [Point2D(1.0, 0.8)]

    @property
    def height_y(self) -> float:
        return (self.config.trees_per_row - 1) * self.config.tree_spacing_m + 2.0

    def _make_workers(self) -> list[MovingAgent]:
        workers: list[MovingAgent] = []
        cfg = self.config
        max_x = cfg.width_margin_m + (cfg.rows - 1) * cfg.row_spacing_m
        max_y = self.height_y - 1.0
        for _ in range(cfg.worker_count):
            x = float(self.rng.uniform(cfg.width_margin_m, max_x))
            y = float(self.rng.uniform(2.0, max_y))
            heading = float(self.rng.uniform(-np.pi, np.pi))
            speed = float(self.rng.uniform(0.15, 0.45))
            workers.append(
                MovingAgent(
                    position=Point2D(x, y),
                    velocity=np.array([speed * cos(heading), speed * sin(heading)]),
                )
            )
        return workers

    def step_workers(self, dt: float) -> None:
        """Move workers using bounded random acceleration and wall bounce."""

        cfg = self.config
        min_x = cfg.width_margin_m * 0.5
        max_x = cfg.width_margin_m + (cfg.rows - 1) * cfg.row_spacing_m + cfg.width_margin_m * 0.5
        min_y = 0.6
        max_y = self.height_y - 0.4
        for worker in self.workers:
            worker.velocity += self.rng.normal(0.0, 0.08, size=2) * dt
            speed = float(np.linalg.norm(worker.velocity))
            if speed > 0.6:
                worker.velocity *= 0.6 / speed
            x = worker.position.x + float(worker.velocity[0] * dt)
            y = worker.position.y + float(worker.velocity[1] * dt)
            if x < min_x or x > max_x:
                worker.velocity[0] *= -1.0
                x = min(max(x, min_x), max_x)
            if y < min_y or y > max_y:
                worker.velocity[1] *= -1.0
                y = min(max(y, min_y), max_y)
            worker.position = Point2D(x, y)

    def lidar_detections(
        self,
        pose: Pose2D,
        include_trees: bool = True,
        include_workers: bool = True,
    ) -> list[LidarDetection]:
        """Return noisy global-frame lidar cluster centroids within sensor range."""

        detections: list[LidarDetection] = []
        if include_trees:
            for tree in self.trees:
                if self._visible(pose, tree):
                    detections.append(self._noisy_detection(tree, "tree", sigma=0.025))
        if include_workers:
            for worker in self.workers:
                if self._visible(pose, worker.position):
                    detections.append(self._noisy_detection(worker.position, "worker", sigma=0.045))
        return detections

    def local_lidar_detections(
        self,
        pose: Pose2D,
        include_trees: bool = True,
        include_workers: bool = True,
    ) -> list[LocalLidarDetection]:
        """Return noisy range/bearing detections in the rover frame."""

        detections: list[LocalLidarDetection] = []
        if include_trees:
            for tree in self.trees:
                if self._visible(pose, tree):
                    detections.append(self._local_detection(pose, tree, "tree"))
        if include_workers:
            for worker in self.workers:
                if self._visible(pose, worker.position):
                    detections.append(self._local_detection(pose, worker.position, "worker"))
        return detections

    def _visible(self, pose: Pose2D, point: Point2D) -> bool:
        dx = point.x - pose.x
        dy = point.y - pose.y
        if hypot(dx, dy) > self.config.lidar_range_m:
            return False
        rel = atan2(dy, dx) - pose.theta
        rel = (rel + np.pi) % (2.0 * np.pi) - np.pi
        return abs(rel) <= radians(135.0)

    def _noisy_detection(self, point: Point2D, kind: str, sigma: float) -> LidarDetection:
        noise = self.rng.normal(0.0, sigma, size=2)
        return LidarDetection(x=point.x + float(noise[0]), y=point.y + float(noise[1]), kind=kind)

    def _local_detection(self, pose: Pose2D, point: Point2D, kind: str) -> LocalLidarDetection:
        dx = point.x - pose.x
        dy = point.y - pose.y
        range_noise = float(self.rng.normal(0.0, 0.02))
        bearing_noise = float(self.rng.normal(0.0, radians(0.4)))
        return LocalLidarDetection(
            range_m=max(0.0, hypot(dx, dy) + range_noise),
            bearing_rad=atan2(dy, dx) - pose.theta + bearing_noise,
            kind=kind,
        )

    def collision_with_worker(self, pose: Pose2D, rover_radius_m: float = 0.45) -> bool:
        """Return true when the rover body intersects a worker."""

        return bool(
            any(distance(pose, worker.position) <= rover_radius_m + worker.radius_m for worker in self.workers)
        )
