"""Procedural orchard world and simple sensor simulation."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, radians, sin, sqrt

import numpy as np
from numpy.typing import NDArray

from terrascout.sim.geometry import Point2D, Pose2D, distance
from terrascout.sim.rover import DifferentialDriveRover
from terrascout.sim.sensors import EncoderSample, ImuSample, LidarScan, SensorConfig, SensorFrame


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

    def step_workers(
        self,
        dt: float,
        avoid_pose: Pose2D | None = None,
        avoid_radius_m: float = 1.25,
    ) -> None:
        """Move workers using bounded random acceleration and safety-bubble bounce."""

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
            if avoid_pose is not None:
                dx = x - avoid_pose.x
                dy = y - avoid_pose.y
                separation = hypot(dx, dy)
                if 1e-6 < separation < avoid_radius_m:
                    normal = np.array([dx / separation, dy / separation])
                    velocity_toward_rover = float(np.dot(worker.velocity, normal))
                    if velocity_toward_rover < 0.0:
                        worker.velocity -= 1.8 * velocity_toward_rover * normal
                    x = avoid_pose.x + float(normal[0] * avoid_radius_m)
                    y = avoid_pose.y + float(normal[1] * avoid_radius_m)
                    x = min(max(x, min_x), max_x)
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

    def lidar_scan(
        self,
        pose: Pose2D,
        sensor_config: SensorConfig | None = None,
    ) -> LidarScan:
        """Return a 270-degree lidar scan with 0.5-degree default angular resolution."""

        sensor_config = sensor_config or SensorConfig()
        beam_count = int(sensor_config.lidar_fov_deg / sensor_config.lidar_angular_resolution_deg) + 1
        start_angle = -radians(sensor_config.lidar_fov_deg) / 2.0
        step = radians(sensor_config.lidar_angular_resolution_deg)
        angles = [start_angle + idx * step for idx in range(beam_count)]
        obstacles = [(tree, 0.18) for tree in self.trees]
        obstacles.extend((worker.position, worker.radius_m) for worker in self.workers)
        ranges = [
            self._beam_range(
                pose=pose,
                relative_angle=angle,
                obstacles=obstacles,
                max_range_m=self.config.lidar_range_m,
                noise_sigma_m=sensor_config.lidar_range_noise_m,
            )
            for angle in angles
        ]
        return LidarScan(angles_rad=angles, ranges_m=ranges, max_range_m=self.config.lidar_range_m)

    def imu_sample(
        self,
        rover: DifferentialDriveRover,
        sensor_config: SensorConfig | None = None,
    ) -> ImuSample:
        """Return a noisy planar IMU yaw-rate and forward-acceleration sample."""

        sensor_config = sensor_config or SensorConfig()
        yaw_rate = (rover.right_velocity_mps - rover.left_velocity_mps) / rover.wheel_base_m
        linear_speed = 0.5 * (rover.left_velocity_mps + rover.right_velocity_mps)
        yaw_rate += sensor_config.gyro_bias_rps
        yaw_rate += float(self.rng.normal(0.0, sensor_config.gyro_noise_rps))
        longitudinal_accel = float(self.rng.normal(0.0, sensor_config.accel_noise_mps2))
        longitudinal_accel += linear_speed * rover.slip_fraction * 0.05
        return ImuSample(yaw_rate_rps=yaw_rate, longitudinal_accel_mps2=longitudinal_accel)

    def encoder_sample(
        self,
        rover: DifferentialDriveRover,
        dt: float,
        sensor_config: SensorConfig | None = None,
    ) -> EncoderSample:
        """Return noisy wheel-encoder distance increments."""

        sensor_config = sensor_config or SensorConfig()
        left_noise = 1.0 + float(self.rng.normal(0.0, sensor_config.encoder_noise_fraction))
        right_noise = 1.0 + float(self.rng.normal(0.0, sensor_config.encoder_noise_fraction))
        return EncoderSample(
            left_delta_m=rover.left_velocity_mps * dt * left_noise,
            right_delta_m=rover.right_velocity_mps * dt * right_noise,
        )

    def sensor_frame(
        self,
        rover: DifferentialDriveRover,
        dt: float,
        sensor_config: SensorConfig | None = None,
    ) -> SensorFrame:
        """Return synchronized lidar, IMU, and encoder samples for one 20 Hz tick."""

        sensor_config = sensor_config or SensorConfig()
        return SensorFrame(
            lidar=self.lidar_scan(rover.pose, sensor_config),
            imu=self.imu_sample(rover, sensor_config),
            encoders=self.encoder_sample(rover, dt, sensor_config),
        )

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

    def _beam_range(
        self,
        pose: Pose2D,
        relative_angle: float,
        obstacles: list[tuple[Point2D, float]],
        max_range_m: float,
        noise_sigma_m: float,
    ) -> float:
        heading = pose.theta + relative_angle
        direction_x = cos(heading)
        direction_y = sin(heading)
        best = max_range_m
        for center, radius in obstacles:
            dx = center.x - pose.x
            dy = center.y - pose.y
            projection = dx * direction_x + dy * direction_y
            if projection <= 0.0 or projection - radius > best:
                continue
            lateral_sq = dx * dx + dy * dy - projection * projection
            radius_sq = radius * radius
            if lateral_sq > radius_sq:
                continue
            hit = projection - sqrt(max(0.0, radius_sq - lateral_sq))
            if 0.0 <= hit < best:
                best = hit
        if best < max_range_m:
            best += float(self.rng.normal(0.0, noise_sigma_m))
        return min(max(best, 0.0), max_range_m)

    def collision_with_worker(self, pose: Pose2D, rover_radius_m: float = 0.45) -> bool:
        """Return true when the rover body intersects a worker."""

        return bool(
            any(distance(pose, worker.position) <= rover_radius_m + worker.radius_m for worker in self.workers)
        )
