#!/usr/bin/env python3
"""Run a complete HITL experiment.

This example runs experiments across multiple conditions with error injection
and exports results to CSV and JSON.

Usage:
    python examples/run_experiment.py

Options:
    --n-trials: Number of trials per condition (default: 20)
    --injection-rate: Error injection rate (default: 0.2)
    --output-dir: Output directory (default: ./experiment_results)
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hitloop import TelemetryLogger
from hitloop.eval.runner import ExperimentRunner, create_standard_conditions
from hitloop.scenarios.email_draft import EmailDraftScenario


def progress_callback(current: int, total: int, message: str) -> None:
    """Print progress updates."""
    pct = (current / total) * 100
    print(f"[{pct:5.1f}%] {message}")


async def run_experiment(
    n_trials: int = 20,
    injection_rate: float = 0.2,
    output_dir: str = "./experiment_results",
) -> None:
    """Run a complete experiment."""
    print("=" * 60)
    print("HITL Lab - Experiment Runner")
    print("=" * 60)
    print()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize
    db_path = output_path / "experiment.db"
    logger = TelemetryLogger(db_path)

    print(f"Database: {db_path}")
    print(f"Trials per condition: {n_trials}")
    print(f"Error injection rate: {injection_rate * 100:.0f}%")
    print()

    # Create scenario
    scenario = EmailDraftScenario(
        output_dir=output_path / "emails",
    )

    # Create conditions
    conditions = create_standard_conditions(
        scenario=scenario,
        n_trials=n_trials,
        injection_rate=injection_rate,
        base_seed=42,
    )

    print(f"Conditions: {len(conditions)}")
    for c in conditions:
        print(f"  - {c.condition_id}: {c.policy.name} policy")
    print()

    # Create runner
    runner = ExperimentRunner(logger, output_dir=output_path)
    for condition in conditions:
        runner.add_condition(condition)

    # Run all trials
    print("Running experiments...")
    print("-" * 60)

    await runner.run_all(progress_callback=progress_callback)

    print("-" * 60)
    print()

    # Export results
    csv_path, json_path = runner.export_results(
        csv_path="results.csv",
        json_path="summary.json",
    )

    print(f"Results exported:")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")
    print()

    # Print summary
    summary = runner.get_summary()
    print("Summary:")
    print("-" * 40)

    for condition_id, stats in summary.get("conditions", {}).items():
        print(f"\n{condition_id}:")
        print(f"  Success rate: {stats['success_rate'] * 100:.1f}%")
        print(f"  Approval rate: {stats['approval_rate'] * 100:.1f}%")
        if stats.get("error_catch_rate") is not None:
            print(f"  Error catch rate: {stats['error_catch_rate'] * 100:.1f}%")
        print(f"  Injected errors: {stats['injected_errors']}")
        print(f"  Errors caught: {stats['errors_caught']}")

    print()
    print("=" * 60)
    print("Experiment complete!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HITL experiment")
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of trials per condition",
    )
    parser.add_argument(
        "--injection-rate",
        type=float,
        default=0.2,
        help="Error injection rate (0.0-1.0)",
    )
    parser.add_argument(
        "--output-dir",
        default="./experiment_results",
        help="Output directory",
    )
    args = parser.parse_args()

    asyncio.run(
        run_experiment(
            n_trials=args.n_trials,
            injection_rate=args.injection_rate,
            output_dir=args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
