from __future__ import annotations

import unittest

from terrascout.sim.battery import BatteryModel


class BatteryModelTest(unittest.TestCase):
    def test_battery_consumes_motion_and_idle_energy(self) -> None:
        battery = BatteryModel(capacity_wh=100.0, drive_wh_per_m=1.0, idle_watts=36.0)

        battery.consume(distance_m=10.0, dt=10.0)

        self.assertLess(battery.soc_wh, 90.0)
        self.assertGreater(battery.soc_wh, 89.0)

    def test_battery_recharges_without_exceeding_capacity(self) -> None:
        battery = BatteryModel(capacity_wh=100.0, recharge_watts=360.0, initial_soc_wh=90.0)

        battery.recharge(dt=200.0)

        self.assertEqual(battery.soc_wh, 100.0)
        self.assertEqual(battery.soc_fraction, 1.0)


if __name__ == "__main__":
    unittest.main()
