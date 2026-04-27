"""Run the TerraScout L2 particle-localization benchmark."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terrascout.eval.benchmarks import LocalizationBenchmarkRow, percentile, run_localization_benchmark


def run(output: Path = Path("artifacts/localization_benchmark.csv")) -> list[LocalizationBenchmarkRow]:
    return run_localization_benchmark(output)


def main() -> None:
    rows = run()
    errors = [row.final_pose_error_m for row in rows]
    print(
        f"mean_error_m={sum(errors) / len(errors):.3f} "
        f"p95_error_m={percentile(errors, 95):.3f} "
        f"max_particles={max(row.particle_count for row in rows)}"
    )


if __name__ == "__main__":
    main()
