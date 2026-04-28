from __future__ import annotations

import math
import unittest

from terrascout.mapping.trunks import detect_tree_trunks
from terrascout.mapping.landmarks import LandmarkMapper
from terrascout.sim.sensors import LidarScan, SensorConfig
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

    def test_trunk_detector_fits_circle_from_lidar_scan(self) -> None:
        center_x = 3.0
        center_y = 0.6
        radius = 0.18
        angles = [math.radians(value) for value in range(-20, 31)]
        ranges: list[float] = []
        for angle in angles:
            direction_x = math.cos(angle)
            direction_y = math.sin(angle)
            projection = center_x * direction_x + center_y * direction_y
            lateral_sq = center_x**2 + center_y**2 - projection**2
            if projection > 0.0 and lateral_sq <= radius**2:
                ranges.append(projection - math.sqrt(radius**2 - lateral_sq))
            else:
                ranges.append(8.0)
        scan = LidarScan(angles_rad=angles, ranges_m=ranges, max_range_m=8.0)

        detections = detect_tree_trunks(scan)

        self.assertTrue(detections)
        self.assertLess(abs(detections[0].range_m - math.hypot(center_x, center_y)), 0.15)
        self.assertLess(abs(detections[0].bearing_rad - math.atan2(center_y, center_x)), 0.08)

    def test_trunk_detector_finds_orchard_tree_from_scan(self) -> None:
        world = OrchardWorld(ScenarioConfig(rows=3, trees_per_row=4, worker_count=0, random_seed=1))
        pose = Pose2D(2.0, 1.0, math.pi / 2.0)
        scan = world.lidar_scan(
            pose,
            SensorConfig(lidar_range_noise_m=0.0),
        )

        detections = detect_tree_trunks(scan)

        self.assertGreaterEqual(len(detections), 1)
        self.assertTrue(all(detection.kind == "tree" for detection in detections))


if __name__ == "__main__":
    unittest.main()
