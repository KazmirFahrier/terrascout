"""Run the TerraScout L0 PID control benchmark."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terrascout.eval.benchmarks import ControlBenchmarkRow, run_control_benchmark


def run(output: Path = Path("artifacts/control_benchmark.csv")) -> list[ControlBenchmarkRow]:
    return run_control_benchmark(output)


def main() -> None:
    rows = run()
    max_cross_track = max(row.max_cross_track_error_m for row in rows)
    max_settle_time = max(row.heading_settle_time_s for row in rows)
    max_overshoot = max(row.heading_overshoot_fraction for row in rows)
    print(
        f"max_cross_track_m={max_cross_track:.3f} "
        f"max_settle_s={max_settle_time:.2f} "
        f"max_overshoot={max_overshoot:.3f}"
    )


if __name__ == "__main__":
    main()
