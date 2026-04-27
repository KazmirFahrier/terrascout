from __future__ import annotations

import unittest

import numpy as np

from terrascout.eval.benchmarks import run_tracking_benchmark
from terrascout.sim.world import LidarDetection
from terrascout.tracking.kalman import MultiObjectTracker


class TrackerTest(unittest.TestCase):
    def test_tracker_follows_constant_velocity_worker(self) -> None:
        tracker = MultiObjectTracker()
        dt = 0.1
        rng = np.random.default_rng(4)

        for step in range(30):
            true_x = 1.0 + 0.4 * step * dt
            true_y = 2.0 + 0.1 * step * dt
            noise = rng.normal(0.0, 0.02, size=2)
            tracker.update(
                [LidarDetection(float(true_x + noise[0]), float(true_y + noise[1]), "worker")],
                dt,
            )

        predictions = tracker.predicted_positions(0.5)
        self.assertEqual(len(predictions), 1)
        _, pred_x, pred_y = predictions[0]
        self.assertLess(abs(pred_x - (1.0 + 0.4 * 3.4)), 0.25)
        self.assertLess(abs(pred_y - (2.0 + 0.1 * 3.4)), 0.25)

    def test_tracking_benchmark_meets_l1_acceptance_metrics(self) -> None:
        rows = run_tracking_benchmark(seeds=[7], worker_count=10)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].track_count, 10)
        self.assertLess(rows[0].mean_prediction_error_m, 0.20)
        self.assertGreaterEqual(rows[0].association_accuracy, 0.95)


if __name__ == "__main__":
    unittest.main()
