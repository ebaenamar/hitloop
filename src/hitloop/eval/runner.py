"""Experiment runner for HITL Lab.

This module orchestrates running experiments across multiple conditions,
with support for error injection and automated metrics collection.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from hitloop.backends.cli_backend import AutoApproveBackend, ScriptedBackend
from hitloop.core.interfaces import ApprovalBackend, HITLPolicy
from hitloop.core.logger import TelemetryLogger
from hitloop.core.models import Action, ApprovalRequest, Decision, RunMetadata
from hitloop.eval.injectors import ErrorInjector, InjectionConfig
from hitloop.eval.metrics import MetricsCalculator
from hitloop.scenarios.base import Scenario


@dataclass
class ExperimentCondition:
    """Configuration for an experimental condition.

    Attributes:
        condition_id: Unique identifier for this condition
        policy: The HITL policy to use
        backend: The approval backend
        scenario: The scenario to run
        injection_config: Error injection configuration
        n_trials: Number of trials to run
        base_seed: Base random seed (trials use base_seed + trial_number)
    """

    condition_id: str
    policy: HITLPolicy
    backend: ApprovalBackend
    scenario: Scenario
    injection_config: InjectionConfig | None = None
    n_trials: int = 10
    base_seed: int = 42


@dataclass
class TrialResult:
    """Result of a single trial.

    Attributes:
        run_id: Unique run identifier
        condition_id: Condition this trial belongs to
        trial_number: Trial number within condition
        seed: Random seed used
        success: Whether the task succeeded
        injected_error: Whether an error was injected
        error_caught: Whether injected error was caught
        approval_requested: Whether approval was requested
        approval_granted: Whether approval was granted
        execution_error: Any error during execution
    """

    run_id: str
    condition_id: str
    trial_number: int
    seed: int
    success: bool = False
    injected_error: bool = False
    error_caught: bool = False
    approval_requested: bool = False
    approval_granted: bool = True
    execution_error: str | None = None


class ExperimentRunner:
    """Run HITL experiments across multiple conditions.

    The ExperimentRunner orchestrates experiments by:
    1. Running multiple trials per condition
    2. Injecting errors at configured rates
    3. Logging all events for analysis
    4. Computing and exporting metrics

    Example:
        >>> runner = ExperimentRunner(logger)
        >>> runner.add_condition(condition1)
        >>> runner.add_condition(condition2)
        >>> await runner.run_all()
        >>> runner.export_results("results.csv", "summary.json")
    """

    def __init__(
        self,
        logger: TelemetryLogger,
        output_dir: Path | str = "./experiment_output",
    ) -> None:
        """Initialize experiment runner.

        Args:
            logger: Telemetry logger for event storage
            output_dir: Directory for output files
        """
        self.logger = logger
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.conditions: list[ExperimentCondition] = []
        self.results: list[TrialResult] = []

    def add_condition(self, condition: ExperimentCondition) -> None:
        """Add an experimental condition.

        Args:
            condition: The condition configuration
        """
        self.conditions.append(condition)

    async def run_all(
        self,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> list[TrialResult]:
        """Run all conditions and trials.

        Args:
            progress_callback: Optional callback(current, total, message)

        Returns:
            List of all trial results
        """
        total_trials = sum(c.n_trials for c in self.conditions)
        current = 0

        for condition in self.conditions:
            for trial_num in range(condition.n_trials):
                current += 1
                if progress_callback:
                    progress_callback(
                        current,
                        total_trials,
                        f"Running {condition.condition_id} trial {trial_num + 1}/{condition.n_trials}",
                    )

                result = await self.run_trial(condition, trial_num)
                self.results.append(result)

        return self.results

    async def run_trial(
        self, condition: ExperimentCondition, trial_number: int
    ) -> TrialResult:
        """Run a single trial.

        Args:
            condition: The condition configuration
            trial_number: Trial number (0-indexed)

        Returns:
            TrialResult for this trial
        """
        seed = condition.base_seed + trial_number
        run_id = str(uuid.uuid4())

        # Create run metadata
        metadata = RunMetadata(
            run_id=run_id,
            scenario_id=condition.scenario.config.scenario_id,
            condition_id=condition.condition_id,
            seed=seed,
        )

        self.logger.log_run_start(metadata)

        result = TrialResult(
            run_id=run_id,
            condition_id=condition.condition_id,
            trial_number=trial_number,
            seed=seed,
        )

        try:
            # Reset scenario
            condition.scenario.reset()

            # Generate action
            injector = None
            if condition.injection_config:
                injector = ErrorInjector(
                    InjectionConfig(
                        injection_rate=condition.injection_config.injection_rate,
                        injection_types=condition.injection_config.injection_types,
                        seed=seed,
                    )
                )

            # Generate correct action
            action = condition.scenario.generate_action(correct=True)

            # Maybe inject error
            if injector:
                injection_result = injector.maybe_inject(action)
                if injection_result.injected:
                    action = injection_result.modified_action or action
                    result.injected_error = True

            # Log the proposed action
            self.logger.log_action_proposed(run_id, action, result.injected_error)

            # Check if approval needed
            needs_approval, policy_reason = condition.policy.should_request_approval(
                action, {"run_id": run_id}
            )

            result.approval_requested = needs_approval

            if needs_approval:
                # Request approval
                request = ApprovalRequest(
                    run_id=run_id,
                    action=action,
                    summary_context=condition.scenario.get_task_description(),
                    policy_name=condition.policy.name,
                    policy_reason=policy_reason,
                )

                self.logger.log_approval_requested(run_id, request, "experiment")
                decision = await condition.backend.request_approval(request)
                self.logger.log_approval_decided(run_id, decision, "experiment")

                result.approval_granted = decision.approved

                # Check if injected error was caught
                if result.injected_error and not decision.approved:
                    result.error_caught = True

            # Execute if approved (or no approval needed)
            if result.approval_granted:
                tools = condition.scenario.get_tools()
                tool_func = tools.get(action.tool_name)

                if tool_func:
                    self.logger.log_tool_execution_start(run_id, action)
                    try:
                        if asyncio.iscoroutinefunction(tool_func):
                            tool_result = await tool_func(**action.tool_args)
                        else:
                            tool_result = tool_func(**action.tool_args)

                        # Validate result
                        validation = condition.scenario.validate_result(tool_result)
                        result.success = validation.success

                    except Exception as e:
                        result.execution_error = str(e)
                        self.logger.log_error(run_id, e, {"phase": "execution"})
                else:
                    result.execution_error = f"Unknown tool: {action.tool_name}"
            else:
                # Rejected - if error was injected and caught, that's a success
                if result.error_caught:
                    result.success = True

        except Exception as e:
            result.execution_error = str(e)
            self.logger.log_error(run_id, e, {"phase": "trial"})

        finally:
            # Log run end
            self.logger.log_run_end(
                run_id,
                result.success,
                {
                    "injected_error": result.injected_error,
                    "error_caught": result.error_caught,
                    "approval_requested": result.approval_requested,
                    "approval_granted": result.approval_granted,
                },
            )

        return result

    def export_results(
        self,
        csv_path: str | None = None,
        json_path: str | None = None,
    ) -> tuple[Path | None, Path | None]:
        """Export experiment results.

        Args:
            csv_path: Path for CSV output (relative to output_dir)
            json_path: Path for JSON summary (relative to output_dir)

        Returns:
            Tuple of (csv_path, json_path) for created files
        """
        calculator = MetricsCalculator(self.logger)

        csv_output = None
        json_output = None

        if csv_path:
            csv_output = self.output_dir / csv_path
            calculator.export_results_csv(csv_output)

        if json_path:
            json_output = self.output_dir / json_path
            calculator.export_summary_json(json_output)

        return csv_output, json_output

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of experiment results.

        Returns:
            Dict with summary statistics
        """
        if not self.results:
            return {"status": "no results"}

        summary: dict[str, Any] = {
            "total_trials": len(self.results),
            "conditions": {},
        }

        for condition in self.conditions:
            condition_results = [
                r for r in self.results if r.condition_id == condition.condition_id
            ]

            if condition_results:
                successes = sum(1 for r in condition_results if r.success)
                injected = sum(1 for r in condition_results if r.injected_error)
                caught = sum(1 for r in condition_results if r.error_caught)
                approvals = sum(1 for r in condition_results if r.approval_requested)

                summary["conditions"][condition.condition_id] = {
                    "n_trials": len(condition_results),
                    "success_rate": successes / len(condition_results),
                    "injected_errors": injected,
                    "errors_caught": caught,
                    "error_catch_rate": caught / injected if injected > 0 else None,
                    "approval_rate": approvals / len(condition_results),
                }

        return summary


def create_standard_conditions(
    scenario: Scenario,
    n_trials: int = 20,
    injection_rate: float = 0.2,
    base_seed: int = 42,
) -> list[ExperimentCondition]:
    """Create standard experimental conditions for comparison.

    Creates conditions for each of the three policies with matching backends.

    Args:
        scenario: The scenario to run
        n_trials: Number of trials per condition
        injection_rate: Error injection rate
        base_seed: Base random seed

    Returns:
        List of experimental conditions
    """
    from hitloop.policies.always_approve import AlwaysApprovePolicy
    from hitloop.policies.audit_plus_escalate import AuditPlusEscalatePolicy
    from hitloop.policies.risk_based import RiskBasedPolicy

    injection_config = InjectionConfig(injection_rate=injection_rate, seed=base_seed)

    conditions = [
        # Condition 1: Always approve (Tier 4 baseline - no HITL)
        ExperimentCondition(
            condition_id="always_approve",
            policy=AlwaysApprovePolicy(),
            backend=AutoApproveBackend(delay_ms=10),
            scenario=scenario,
            injection_config=injection_config,
            n_trials=n_trials,
            base_seed=base_seed,
        ),
        # Condition 2: Risk-based with auto-approve backend
        ExperimentCondition(
            condition_id="risk_based_auto",
            policy=RiskBasedPolicy(
                require_approval_for_high=True,
                require_approval_for_medium=True,
            ),
            backend=AutoApproveBackend(delay_ms=50),
            scenario=scenario,
            injection_config=injection_config,
            n_trials=n_trials,
            base_seed=base_seed,
        ),
        # Condition 3: Risk-based with scripted rejections (simulates human catching errors)
        ExperimentCondition(
            condition_id="risk_based_human",
            policy=RiskBasedPolicy(
                require_approval_for_high=True,
                require_approval_for_medium=True,
            ),
            backend=ScriptedBackend(
                decisions=[True, True, False, True, True],  # 20% rejection
                delay_ms=500,  # Simulate human latency
                default_approve=True,
            ),
            scenario=scenario,
            injection_config=injection_config,
            n_trials=n_trials,
            base_seed=base_seed,
        ),
        # Condition 4: Audit + escalate with scripted backend
        ExperimentCondition(
            condition_id="audit_escalate",
            policy=AuditPlusEscalatePolicy(
                audit_sample_rate=0.3,
                escalate_on_high_risk=True,
                seed=base_seed,
            ),
            backend=ScriptedBackend(
                decisions=[True, True, True, False, True],  # 20% rejection
                delay_ms=300,
                default_approve=True,
            ),
            scenario=scenario,
            injection_config=injection_config,
            n_trials=n_trials,
            base_seed=base_seed,
        ),
    ]

    return conditions
