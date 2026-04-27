"""Run a deterministic EKF-SLAM smoke benchmark."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terrascout.eval.benchmarks import SlamBenchmarkRow, run_slam_benchmark


def run(output: Path = Path("artifacts/slam_benchmark.csv")) -> list[SlamBenchmarkRow]:
    return run_slam_benchmark(output)


def main() -> None:
    rows = run()
    mean_landmarks = sum(row.landmark_count for row in rows) / len(rows)
    mean_pose_error = sum(row.final_pose_error_m for row in rows) / len(rows)
    mean_landmark_error = sum(row.mean_landmark_error_m for row in rows) / len(rows)
    mean_wall = sum(row.wall_time_ms for row in rows) / len(rows)
    print(
        f"mean_landmarks={mean_landmarks:.1f} "
        f"mean_pose_error_m={mean_pose_error:.3f} "
        f"mean_landmark_error_m={mean_landmark_error:.3f} "
        f"mean_wall_ms={mean_wall:.2f}"
    )


if __name__ == "__main__":
    main()
