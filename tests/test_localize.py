from __future__ import annotations

import unittest

from terrascout.eval.benchmarks import percentile, run_localization_benchmark
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
        self.assertLessEqual(len(localizer.particles), 1500)
        self.assertGreaterEqual(len(localizer.particles), localizer.min_particles)

    def test_kld_resampling_adapts_particle_count(self) -> None:
        localizer = ParticleLocalizer.gaussian(
            900,
            mean=Pose2D(1.0, 2.0, 0.2),
            std=(0.1, 0.1, 0.04),
            seed=12,
        )
        localizer.weights[:] = 1e-6
        localizer.weights[:5] = 1.0
        localizer.weights /= localizer.weights.sum()

        localizer.resample()

        self.assertLess(len(localizer.particles), 900)
        self.assertGreaterEqual(len(localizer.particles), localizer.min_particles)
        self.assertAlmostEqual(float(localizer.weights.sum()), 1.0)

    def test_localization_benchmark_meets_current_l2_metric(self) -> None:
        rows = run_localization_benchmark(seeds=[2, 3, 5, 7, 11, 13, 17, 19, 23, 29])
        errors = [row.final_pose_error_m for row in rows]

        self.assertLess(percentile(errors, 95), 0.15)
        self.assertLessEqual(max(row.particle_count for row in rows), 3000)


if __name__ == "__main__":
    unittest.main()
