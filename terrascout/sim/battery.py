"""Simple battery state model for rover missions."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BatteryModel:
    """Track battery state-of-charge from motion and recharge events."""

    capacity_wh: float = 240.0
    drive_wh_per_m: float = 0.35
    idle_watts: float = 18.0
    recharge_watts: float = 420.0
    initial_soc_wh: float | None = None
    soc_wh: float = field(init=False)

    def __post_init__(self) -> None:
        self.soc_wh = self.capacity_wh if self.initial_soc_wh is None else self.initial_soc_wh

    @property
    def soc_fraction(self) -> float:
        return max(0.0, min(1.0, float(self.soc_wh) / self.capacity_wh))

    def consume(self, distance_m: float, dt: float) -> None:
        """Consume energy for distance traveled and baseline electronics."""

        draw = max(0.0, distance_m) * self.drive_wh_per_m + self.idle_watts * dt / 3600.0
        self.soc_wh = max(0.0, float(self.soc_wh) - draw)

    def recharge(self, dt: float) -> None:
        """Recharge while within a charging station."""

        self.soc_wh = min(self.capacity_wh, float(self.soc_wh) + self.recharge_watts * dt / 3600.0)
