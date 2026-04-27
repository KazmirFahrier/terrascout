"""Run the TerraScout L1 multi-worker tracking benchmark."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terrascout.eval.benchmarks import TrackingBenchmarkRow, run_tracking_benchmark


def run(output: Path = Path("artifacts/tracking_benchmark.csv")) -> list[TrackingBenchmarkRow]:
    return run_tracking_benchmark(output)


def main() -> None:
    rows = run()
    mean_prediction_error = sum(row.mean_prediction_error_m for row in rows) / len(rows)
    mean_association = sum(row.association_accuracy for row in rows) / len(rows)
    print(
        f"mean_prediction_error_m={mean_prediction_error:.3f} "
        f"mean_association_accuracy={mean_association:.3f}"
    )


if __name__ == "__main__":
    main()
