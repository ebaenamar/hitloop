"""Metrics calculation for HITL experiments.

This module computes research metrics from experiment traces,
aligned with the paper framework.
"""

from __future__ import annotations

import csv
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hitloop.core.logger import TelemetryLogger
from hitloop.core.models import EventType, RunMetadata


@dataclass
class RunMetrics:
    """Metrics for a single experiment run.

    Attributes:
        run_id: Unique run identifier
        scenario_id: Scenario identifier
        condition_id: Experimental condition
        policy_name: Policy used
        seed: Random seed
        task_success: Whether the task succeeded
        approval_requested: Whether approval was requested
        approval_granted: Whether approval was granted
        human_latency_ms: Time to get human decision (if applicable)
        time_to_complete_ms: Total run time
        injected_error: Whether this run had injected error
        error_caught: Whether injected error was caught (rejected)
        tool_failure: Whether tool execution failed
        tokens_used: Token count (if available)
        llm_calls: Number of LLM calls
    """

    run_id: str
    scenario_id: str = ""
    condition_id: str = ""
    policy_name: str = ""
    seed: int | None = None

    task_success: bool = False
    approval_requested: bool = False
    approval_granted: bool = True
    human_latency_ms: float = 0.0
    time_to_complete_ms: float = 0.0

    injected_error: bool = False
    error_caught: bool = False
    tool_failure: bool = False

    tokens_used: int = 0
    llm_calls: int = 0


@dataclass
class ConditionMetrics:
    """Aggregated metrics for an experimental condition.

    Attributes:
        condition_id: Condition identifier
        policy_name: Policy used
        n_runs: Number of runs
        success_rate: Task success rate
        approval_rate: Fraction of runs where approval was requested
        human_latency_mean_ms: Mean human decision latency
        human_latency_p95_ms: 95th percentile latency
        time_to_complete_mean_ms: Mean completion time
        time_to_complete_p95_ms: 95th percentile completion time
        error_catch_rate: Fraction of injected errors caught
        false_reject_rate: Fraction of correct actions rejected
        tool_failure_rate: Tool execution failure rate
        cost_proxy: Token/call count proxy for cost
    """

    condition_id: str
    policy_name: str = ""
    n_runs: int = 0

    success_rate: float = 0.0
    approval_rate: float = 0.0

    human_latency_mean_ms: float = 0.0
    human_latency_p95_ms: float = 0.0

    time_to_complete_mean_ms: float = 0.0
    time_to_complete_p95_ms: float = 0.0

    error_catch_rate: float = 0.0
    false_reject_rate: float = 0.0
    tool_failure_rate: float = 0.0

    cost_proxy: float = 0.0


class MetricsCalculator:
    """Calculate metrics from experiment traces.

    Computes per-run and aggregated metrics aligned with research framework:
    - Quality: success_rate
    - Leverage proxy: approval_rate, error_catch_rate
    - Learning signal: human decisions and rejections
    - Human burden: human_latency_ms, approval_rate
    - Appropriate reliance: error_catch_rate, false_reject_rate

    Example:
        >>> calc = MetricsCalculator(logger)
        >>> run_metrics = calc.compute_run_metrics(run_id)
        >>> condition_metrics = calc.compute_condition_metrics(condition_id)
    """

    def __init__(self, logger: TelemetryLogger) -> None:
        """Initialize metrics calculator.

        Args:
            logger: Telemetry logger with experiment data
        """
        self.logger = logger

    def compute_run_metrics(self, run_id: str) -> RunMetrics:
        """Compute metrics for a single run.

        Args:
            run_id: ID of the run to analyze

        Returns:
            RunMetrics for the run
        """
        metadata = self.logger.get_run_metadata(run_id)
        events = self.logger.get_events(run_id=run_id)

        metrics = RunMetrics(run_id=run_id)

        if metadata:
            metrics.scenario_id = metadata.scenario_id
            metrics.condition_id = metadata.condition_id
            metrics.seed = metadata.seed
            metrics.task_success = metadata.task_success or False
            metrics.time_to_complete_ms = metadata.duration_ms() or 0.0

        # Process events
        has_injected_error = False
        approval_requested = False
        approval_granted = True
        human_latency = 0.0
        tokens = 0
        llm_calls = 0

        for event in events:
            if event.event_type == EventType.INJECTED_ERROR:
                has_injected_error = True

            elif event.event_type == EventType.APPROVAL_REQUESTED:
                approval_requested = True
                metrics.policy_name = event.payload.get("policy_name", "")

            elif event.event_type == EventType.APPROVAL_DECIDED:
                human_latency = event.payload.get("latency_ms", 0.0)
                approval_granted = event.payload.get("approved", True)

            elif event.event_type == EventType.LLM_CALL:
                llm_calls += 1
                tokens_in = event.payload.get("tokens_in") or 0
                tokens_out = event.payload.get("tokens_out") or 0
                tokens += tokens_in + tokens_out

            elif event.event_type == EventType.TOOL_EXECUTION_END:
                if not event.payload.get("success", True):
                    metrics.tool_failure = True

        metrics.injected_error = has_injected_error
        metrics.approval_requested = approval_requested
        metrics.approval_granted = approval_granted
        metrics.human_latency_ms = human_latency
        metrics.tokens_used = tokens
        metrics.llm_calls = llm_calls

        # Determine if error was caught
        if has_injected_error and approval_requested and not approval_granted:
            metrics.error_caught = True

        return metrics

    def compute_condition_metrics(
        self, condition_id: str | None = None, run_ids: list[str] | None = None
    ) -> ConditionMetrics:
        """Compute aggregated metrics for a condition.

        Args:
            condition_id: Filter by condition ID
            run_ids: Explicit list of run IDs to include

        Returns:
            ConditionMetrics aggregated from runs
        """
        # Get all relevant runs
        all_runs = self.logger.get_all_runs()

        if run_ids:
            all_runs = [r for r in all_runs if r.run_id in run_ids]
        elif condition_id:
            all_runs = [r for r in all_runs if r.condition_id == condition_id]

        if not all_runs:
            return ConditionMetrics(condition_id=condition_id or "unknown")

        # Compute per-run metrics
        run_metrics_list = [self.compute_run_metrics(r.run_id) for r in all_runs]

        # Aggregate
        n_runs = len(run_metrics_list)
        policy_names = [m.policy_name for m in run_metrics_list if m.policy_name]

        # Success rate
        successes = sum(1 for m in run_metrics_list if m.task_success)
        success_rate = successes / n_runs

        # Approval rate
        approvals_requested = sum(1 for m in run_metrics_list if m.approval_requested)
        approval_rate = approvals_requested / n_runs

        # Human latency (only for runs with approval)
        latencies = [m.human_latency_ms for m in run_metrics_list if m.approval_requested and m.human_latency_ms > 0]
        human_latency_mean = statistics.mean(latencies) if latencies else 0.0
        human_latency_p95 = (
            sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else human_latency_mean
        )

        # Time to complete
        completion_times = [m.time_to_complete_ms for m in run_metrics_list if m.time_to_complete_ms > 0]
        time_mean = statistics.mean(completion_times) if completion_times else 0.0
        time_p95 = (
            sorted(completion_times)[int(len(completion_times) * 0.95)]
            if len(completion_times) > 1
            else time_mean
        )

        # Error catch rate (only for injected errors)
        injected_runs = [m for m in run_metrics_list if m.injected_error]
        caught = sum(1 for m in injected_runs if m.error_caught)
        error_catch_rate = caught / len(injected_runs) if injected_runs else 0.0

        # False reject rate (rejections on non-injected runs)
        non_injected_runs = [m for m in run_metrics_list if not m.injected_error and m.approval_requested]
        false_rejects = sum(1 for m in non_injected_runs if not m.approval_granted)
        false_reject_rate = false_rejects / len(non_injected_runs) if non_injected_runs else 0.0

        # Tool failure rate
        failures = sum(1 for m in run_metrics_list if m.tool_failure)
        tool_failure_rate = failures / n_runs

        # Cost proxy
        total_tokens = sum(m.tokens_used for m in run_metrics_list)
        total_llm_calls = sum(m.llm_calls for m in run_metrics_list)
        cost_proxy = total_tokens if total_tokens > 0 else float(total_llm_calls)

        return ConditionMetrics(
            condition_id=condition_id or "all",
            policy_name=policy_names[0] if policy_names else "",
            n_runs=n_runs,
            success_rate=success_rate,
            approval_rate=approval_rate,
            human_latency_mean_ms=human_latency_mean,
            human_latency_p95_ms=human_latency_p95,
            time_to_complete_mean_ms=time_mean,
            time_to_complete_p95_ms=time_p95,
            error_catch_rate=error_catch_rate,
            false_reject_rate=false_reject_rate,
            tool_failure_rate=tool_failure_rate,
            cost_proxy=cost_proxy,
        )

    def export_results_csv(
        self, output_path: Path | str, run_ids: list[str] | None = None
    ) -> None:
        """Export per-run metrics to CSV.

        Args:
            output_path: Path for output CSV
            run_ids: Optional list of specific runs to export
        """
        output_path = Path(output_path)

        if run_ids:
            runs = run_ids
        else:
            runs = [r.run_id for r in self.logger.get_all_runs()]

        fieldnames = [
            "run_id",
            "scenario_id",
            "condition_id",
            "policy_name",
            "seed",
            "task_success",
            "approval_requested",
            "approval_granted",
            "human_latency_ms",
            "time_to_complete_ms",
            "injected_error",
            "error_caught",
            "tool_failure",
            "tokens_used",
            "llm_calls",
        ]

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for run_id in runs:
                metrics = self.compute_run_metrics(run_id)
                writer.writerow(
                    {
                        "run_id": metrics.run_id,
                        "scenario_id": metrics.scenario_id,
                        "condition_id": metrics.condition_id,
                        "policy_name": metrics.policy_name,
                        "seed": metrics.seed,
                        "task_success": int(metrics.task_success),
                        "approval_requested": int(metrics.approval_requested),
                        "approval_granted": int(metrics.approval_granted),
                        "human_latency_ms": round(metrics.human_latency_ms, 2),
                        "time_to_complete_ms": round(metrics.time_to_complete_ms, 2),
                        "injected_error": int(metrics.injected_error),
                        "error_caught": int(metrics.error_caught),
                        "tool_failure": int(metrics.tool_failure),
                        "tokens_used": metrics.tokens_used,
                        "llm_calls": metrics.llm_calls,
                    }
                )

    def export_summary_json(
        self, output_path: Path | str, by_condition: bool = True
    ) -> None:
        """Export aggregated metrics summary to JSON.

        Args:
            output_path: Path for output JSON
            by_condition: If True, aggregate by condition_id
        """
        output_path = Path(output_path)

        if by_condition:
            # Get unique conditions
            all_runs = self.logger.get_all_runs()
            conditions = set(r.condition_id for r in all_runs if r.condition_id)

            summary = {}
            for condition_id in conditions:
                metrics = self.compute_condition_metrics(condition_id=condition_id)
                summary[condition_id] = {
                    "policy_name": metrics.policy_name,
                    "n_runs": metrics.n_runs,
                    "success_rate": round(metrics.success_rate, 4),
                    "approval_rate": round(metrics.approval_rate, 4),
                    "human_latency_mean_ms": round(metrics.human_latency_mean_ms, 2),
                    "human_latency_p95_ms": round(metrics.human_latency_p95_ms, 2),
                    "time_to_complete_mean_ms": round(metrics.time_to_complete_mean_ms, 2),
                    "time_to_complete_p95_ms": round(metrics.time_to_complete_p95_ms, 2),
                    "error_catch_rate": round(metrics.error_catch_rate, 4),
                    "false_reject_rate": round(metrics.false_reject_rate, 4),
                    "tool_failure_rate": round(metrics.tool_failure_rate, 4),
                    "cost_proxy": metrics.cost_proxy,
                }
        else:
            metrics = self.compute_condition_metrics()
            summary = {
                "overall": {
                    "n_runs": metrics.n_runs,
                    "success_rate": round(metrics.success_rate, 4),
                    "approval_rate": round(metrics.approval_rate, 4),
                    "human_latency_mean_ms": round(metrics.human_latency_mean_ms, 2),
                    "human_latency_p95_ms": round(metrics.human_latency_p95_ms, 2),
                    "time_to_complete_mean_ms": round(metrics.time_to_complete_mean_ms, 2),
                    "time_to_complete_p95_ms": round(metrics.time_to_complete_p95_ms, 2),
                    "error_catch_rate": round(metrics.error_catch_rate, 4),
                    "false_reject_rate": round(metrics.false_reject_rate, 4),
                    "tool_failure_rate": round(metrics.tool_failure_rate, 4),
                    "cost_proxy": metrics.cost_proxy,
                }
            }

        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
