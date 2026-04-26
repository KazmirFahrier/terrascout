from __future__ import annotations

import unittest

from terrascout.localize.particle import ParticleLocalizer
from terrascout.sim.geometry import Pose2D, distance
from terrascout.sim.world import OrchardWorld, ScenarioConfig


class ParticleLocalizerTest(unittest.TestCase):
    def test_particle_filter_refines_coarse_pose_prior(self) -> None:
        world = OrchardWorld(ScenarioConfig(rows=4, trees_per_row=6, worker_count=0, random_seed=4))
        truth = Pose2D(5.0, 6.0, 0.7)
        localizer = ParticleLocalizer.gaussian(
            1500,
            mean=Pose2D(5.5, 5.7, 0.55),
            std=(0.8, 0.8, 0.35),
            seed=9,
        )

        for _ in range(3):
            localizer.update(world.local_lidar_detections(truth, include_workers=False), world.trees)

        self.assertLess(distance(localizer.estimate(), truth), 0.25)


if __name__ == "__main__":
    unittest.main()

