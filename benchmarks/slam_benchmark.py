"""Run a deterministic EKF-SLAM smoke benchmark."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terrascout.mapping.ekf_slam import EkfSlam
from terrascout.sim.geometry import Pose2D
from terrascout.sim.world import OrchardWorld, ScenarioConfig


@dataclass(frozen=True)
class SlamBenchmarkRow:
    seed: int
    landmark_count: int
    mean_observations: float
    covariance_trace: float
    wall_time_ms: float


def run(output: Path = Path("artifacts/slam_benchmark.csv")) -> list[SlamBenchmarkRow]:
    rows: list[SlamBenchmarkRow] = []
    poses = [
        Pose2D(2.0, 2.0, 0.8),
        Pose2D(5.5, 7.0, 1.3),
        Pose2D(9.0, 12.0, 1.7),
        Pose2D(12.0, 18.0, 1.5),
    ]
    for seed in [2, 3, 5, 7, 11]:
        world = OrchardWorld(ScenarioConfig(rows=6, trees_per_row=10, worker_count=0, random_seed=seed))
        slam = EkfSlam(poses[0])
        started = perf_counter()
        previous = poses[0]
        for pose in poses:
            slam.predict(
                linear_mps=((pose.x - previous.x) ** 2 + (pose.y - previous.y) ** 2) ** 0.5,
                angular_rps=pose.theta - previous.theta,
                dt=1.0,
            )
            slam.update(world.local_lidar_detections(pose, include_workers=False))
            previous = pose
        mean_observations = (
            sum(slam.landmark_observations) / len(slam.landmark_observations)
            if slam.landmark_observations
            else 0.0
        )
        rows.append(
            SlamBenchmarkRow(
                seed=seed,
                landmark_count=slam.landmark_count,
                mean_observations=mean_observations,
                covariance_trace=float(slam.covariance.trace()),
                wall_time_ms=(perf_counter() - started) * 1000.0,
            )
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return rows


def main() -> None:
    rows = run()
    mean_landmarks = sum(row.landmark_count for row in rows) / len(rows)
    mean_wall = sum(row.wall_time_ms for row in rows) / len(rows)
    print(f"mean_landmarks={mean_landmarks:.1f} mean_wall_ms={mean_wall:.2f}")


if __name__ == "__main__":
    main()

