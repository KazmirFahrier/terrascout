from __future__ import annotations

import unittest

import numpy as np

from terrascout.eval.benchmarks import run_slam_benchmark
from terrascout.mapping.ekf_slam import EkfSlam
from terrascout.sim.geometry import Pose2D, distance
from terrascout.sim.world import OrchardWorld, ScenarioConfig


class EkfSlamTest(unittest.TestCase):
    def test_ekf_slam_accumulates_landmarks_and_reduces_uncertainty(self) -> None:
        world = OrchardWorld(ScenarioConfig(rows=4, trees_per_row=7, worker_count=0, random_seed=4))
        pose = Pose2D(5.0, 5.0, 0.8)
        slam = EkfSlam(pose)

        slam.update(world.local_lidar_detections(pose, include_workers=False))
        self.assertGreater(slam.landmark_count, 5)
        initial_trace = float(np.trace(slam.covariance))

        for _ in range(4):
            slam.update(world.local_lidar_detections(pose, include_workers=False))

        self.assertLess(float(np.trace(slam.covariance)), initial_trace)
        self.assertTrue(any(count > 1 for count in slam.landmark_observations))

    def test_ekf_slam_predicts_rover_motion(self) -> None:
        slam = EkfSlam(Pose2D(0.0, 0.0, 0.0))

        slam.predict(linear_mps=1.0, angular_rps=0.0, dt=1.0)

        self.assertLess(distance(slam.pose, Pose2D(1.0, 0.0, 0.0)), 0.05)

    def test_slam_benchmark_reports_pose_and_landmark_accuracy(self) -> None:
        rows = run_slam_benchmark(seeds=[7])

        self.assertGreaterEqual(rows[0].landmark_count, 120)
        self.assertLess(rows[0].final_pose_error_m, 0.20)
        self.assertLess(rows[0].mean_landmark_error_m, 0.30)
        self.assertLess(rows[0].p95_landmark_error_m, 0.35)


if __name__ == "__main__":
    unittest.main()
