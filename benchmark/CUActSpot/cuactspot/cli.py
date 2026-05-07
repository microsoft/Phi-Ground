from __future__ import annotations

import argparse

from cuactspot.config import load_run_config
from cuactspot.reporting import REPORT_MODALITIES
from cuactspot.runner import EvaluationRunner


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run batch evaluation for CUActSpot-compatible models."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a JSON run config.",
    )
    parser.add_argument(
        "--visualize",
        default=None,
        help="Optional folder to save sample_id_out.png visualizations.",
    )
    parser.add_argument(
        "--best-of-N",
        type=int,
        default=1,
        dest="best_of_n",
        help="Run each sample N times and pass if any trial is correct (default: 1).",
    )
    parser.add_argument(
        "--best-of-temperature",
        type=float,
        default=0.7,
        dest="best_of_temperature",
        help="Sampling temperature used for best-of-N trials (default: 0.7).",
    )
    args = parser.parse_args()

    config = load_run_config(args.config)
    if args.visualize:
        config.output.visualize_dir = args.visualize
    if args.best_of_n > 1:
        config.runtime.best_of_n = args.best_of_n
        config.runtime.best_of_temperature = args.best_of_temperature
    report = EvaluationRunner(config).run()
    _print_overall_metrics(report.get("overall", {}))


def _print_overall_metrics(overall: dict) -> None:
    for modality in REPORT_MODALITIES + ["overall"]:
        value = overall.get(modality, 0.0)
        print(f"{modality}: {value:.2%}")


if __name__ == "__main__":
    main()