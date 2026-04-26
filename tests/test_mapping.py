from __future__ import annotations

import unittest

from terrascout.mapping.landmarks import LandmarkMapper
from terrascout.sim.geometry import Pose2D
from terrascout.sim.world import OrchardWorld, ScenarioConfig


class LandmarkMapperTest(unittest.TestCase):
    def test_mapper_accumulates_tree_landmarks(self) -> None:
        world = OrchardWorld(ScenarioConfig(rows=3, trees_per_row=5, worker_count=0, random_seed=3))
        mapper = LandmarkMapper()

        poses = [Pose2D(2.0, 2.0, 0.8), Pose2D(5.0, 5.0, 1.3), Pose2D(8.0, 8.0, 2.2)]
        for pose in poses:
            mapper.update(pose, world.local_lidar_detections(pose, include_workers=False))

        self.assertGreaterEqual(len(mapper.landmarks), 8)
        self.assertTrue(any(landmark.observations > 1 for landmark in mapper.landmarks))


if __name__ == "__main__":
    unittest.main()

