from __future__ import annotations

import unittest

import numpy as np

from terrascout.safety.collision_guard import SafetySupervisor
from terrascout.sim.geometry import Point2D, Pose2D, distance
from terrascout.sim.world import LidarDetection, MovingAgent, OrchardWorld, ScenarioConfig


class SafetySupervisorTest(unittest.TestCase):
    def test_supervisor_stops_inside_stop_radius(self) -> None:
        supervisor = SafetySupervisor(stop_radius_m=1.0, slow_radius_m=2.0)

        decision = supervisor.supervise(
            Pose2D(0.0, 0.0, 0.0),
            left_mps=1.0,
            right_mps=1.0,
            worker_detections=[LidarDetection(0.5, 0.0, "worker")],
            predicted_workers=[],
        )

        self.assertEqual(decision.left_mps, 0.0)
        self.assertTrue(decision.intervened)
        self.assertTrue(decision.stopped)

    def test_supervisor_scales_inside_slow_radius(self) -> None:
        supervisor = SafetySupervisor(stop_radius_m=1.0, slow_radius_m=3.0)

        decision = supervisor.supervise(
            Pose2D(0.0, 0.0, 0.0),
            left_mps=1.0,
            right_mps=1.0,
            worker_detections=[],
            predicted_workers=[(1, 2.0, 0.0)],
        )

        self.assertAlmostEqual(decision.left_mps, 0.5)
        self.assertAlmostEqual(decision.right_mps, 0.5)
        self.assertTrue(decision.intervened)
        self.assertFalse(decision.stopped)

    def test_supervisor_leaves_clear_path_unchanged(self) -> None:
        supervisor = SafetySupervisor(stop_radius_m=1.0, slow_radius_m=2.0)

        decision = supervisor.supervise(
            Pose2D(0.0, 0.0, 0.0),
            left_mps=0.8,
            right_mps=0.7,
            worker_detections=[],
            predicted_workers=[(1, 5.0, 0.0)],
        )

        self.assertEqual(decision.left_mps, 0.8)
        self.assertEqual(decision.right_mps, 0.7)
        self.assertFalse(decision.intervened)

    def test_worker_dynamics_respect_rover_safety_bubble(self) -> None:
        world = OrchardWorld(ScenarioConfig(worker_count=0, random_seed=7))
        world.workers = [
            MovingAgent(
                position=Point2D(0.9, 0.0),
                velocity=np.array([-0.2, 0.0]),
            )
        ]

        world.step_workers(dt=0.1, avoid_pose=Pose2D(0.0, 0.0, 0.0), avoid_radius_m=1.25)

        self.assertGreaterEqual(distance(world.workers[0].position, Point2D(0.0, 0.0)), 1.25)
        self.assertGreater(world.workers[0].velocity[0], 0.0)


if __name__ == "__main__":
    unittest.main()
