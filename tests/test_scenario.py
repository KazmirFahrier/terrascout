from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from terrascout.sim.scenario import load_scenario_config, save_scenario_config
from terrascout.sim.world import ScenarioConfig


class ScenarioLoaderTest(unittest.TestCase):
    def test_scenario_config_round_trips_json(self) -> None:
        config = ScenarioConfig(rows=5, trees_per_row=9, worker_count=2, random_seed=42)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scenario.json"
            save_scenario_config(config, path)
            loaded = load_scenario_config(path)

        self.assertEqual(loaded, config)

    def test_scenario_loader_rejects_unknown_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text('{"rows": 4, "not_a_field": 123}')

            with self.assertRaises(ValueError):
                load_scenario_config(path)


if __name__ == "__main__":
    unittest.main()

