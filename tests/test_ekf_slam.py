from __future__ import annotations

import unittest

import numpy as np

from terrascout.eval.benchmarks import run_slam_benchmark
from terrascout.mapping.ekf_slam import EkfSlam, EkfSlamConfig
from terrascout.sim.geometry import Pose2D, distance
from terrascout.sim.world import LocalLidarDetection, OrchardWorld, ScenarioConfig


class EkfSlamTest(unittest.TestCase):
    def test_landmark_update_matches_dense_joseph_form_with_gain_clipping(self) -> None:
        for correlation_scale in (1.0, 40.0):
            with self.subTest(correlation_scale=correlation_scale):
                slam = EkfSlam(Pose2D(0.0, 0.0, 0.0))
                slam.mean = np.array([0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0])
                slam.landmark_observations = [1, 1]
                factor = np.random.default_rng(31).normal(size=(7, 7)) * 0.03
                factor[5] = factor[0] * correlation_scale
                prior = factor @ factor.T + np.eye(7) * 0.001
                slam.covariance = prior.copy()
                h = np.array([
                    [-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0 / 3.0, -1.0, 0.0, 1.0 / 3.0, 0.0, 0.0],
                ])
                noise = np.diag([slam.config.range_sigma**2, slam.config.bearing_sigma**2])
                gain = prior @ h.T @ np.linalg.inv(h @ prior @ h.T + noise)
                if correlation_scale > 1:
                    self.assertGreater(np.max(np.abs(gain)), 5.0)
                gain = np.clip(gain, -5.0, 5.0)
                projection = np.clip(np.eye(7) - gain @ h, -5.0, 5.0)
                expected = projection @ prior @ projection.T + gain @ noise @ gain.T

                slam._update_landmark(
                    0, LocalLidarDetection(range_m=3.1, bearing_rad=0.04, kind="tree")
                )

                np.testing.assert_allclose(slam.covariance, expected, rtol=1e-12, atol=1e-12)
                self.assertEqual(slam.landmark_observations, [2, 1])

    def test_prediction_preserves_dense_covariance_with_landmark_correlations(self) -> None:
        rng = np.random.default_rng(23)
        for size in (3, 9):
            with self.subTest(state_size=size):
                slam = EkfSlam(Pose2D(1.0, 2.0, 0.8))
                slam.mean = np.zeros(size)
                slam.mean[:3] = [1.0, 2.0, 0.8]
                factor = rng.normal(size=(size, size))
                prior = factor @ factor.T * 0.01 + np.eye(size) * 0.1
                slam.covariance = prior.copy()
                dt, linear, angular = 0.2, -0.7, 0.3
                theta_mid = 0.8 + angular * dt / 2
                jacobian = np.eye(size)
                jacobian[0, 2] = -linear * np.sin(theta_mid) * dt
                jacobian[1, 2] = linear * np.cos(theta_mid) * dt
                noise = np.zeros((size, size))
                noise[0, 0] = noise[1, 1] = slam.config.motion_linear_sigma**2
                noise[2, 2] = slam.config.motion_angular_sigma**2
                expected = jacobian @ prior @ jacobian.T + noise

                slam.predict(linear_mps=linear, angular_rps=angular, dt=dt)

                np.testing.assert_allclose(slam.covariance, expected, rtol=1e-12, atol=1e-12)
                np.testing.assert_array_equal(slam.covariance[3:, 3:], prior[3:, 3:])

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

    def test_ekf_slam_rejects_mahalanobis_outlier(self) -> None:
        slam = EkfSlam(
            Pose2D(0.0, 0.0, 0.0),
            EkfSlamConfig(
                association_gate_m=10.0,
                mahalanobis_gate=0.01,
                innovation_gate=100.0,
                max_landmarks=1,
            ),
        )
        slam.update([LocalLidarDetection(range_m=3.0, bearing_rad=0.0, kind="tree")])
        covariance_trace = float(np.trace(slam.covariance))

        slam.update([LocalLidarDetection(range_m=3.6, bearing_rad=0.2, kind="tree")])

        self.assertEqual(slam.landmark_count, 1)
        self.assertEqual(slam.landmark_observations, [1])
        self.assertAlmostEqual(float(np.trace(slam.covariance)), covariance_trace)

    def test_slam_benchmark_reports_pose_and_landmark_accuracy(self) -> None:
        rows = run_slam_benchmark(seeds=[7])

        self.assertGreaterEqual(rows[0].landmark_count, 120)
        self.assertLess(rows[0].final_pose_error_m, 0.20)
        self.assertLess(rows[0].mean_landmark_error_m, 0.30)
        self.assertLess(rows[0].p95_landmark_error_m, 0.35)


if __name__ == "__main__":
    unittest.main()
