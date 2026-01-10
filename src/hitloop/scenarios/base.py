"""Base scenario interface for HITL Lab.

Scenarios define the context for HITL experiments: what task the agent
is trying to accomplish, what tools are available, and how to validate
success.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from hitloop.core.models import Action, RiskClass


@dataclass
class ScenarioConfig:
    """Configuration for a scenario instance.

    Attributes:
        scenario_id: Unique identifier for this scenario
        name: Human-readable scenario name
        description: What the scenario tests
        expected_tool: The tool that should be used
        expected_args: Expected argument patterns
        risk_class: Risk classification for the scenario
        side_effects: List of side effects
        seed: Random seed for reproducibility
        extra: Additional scenario-specific configuration
    """

    scenario_id: str
    name: str
    description: str = ""
    expected_tool: str = ""
    expected_args: dict[str, Any] = field(default_factory=dict)
    risk_class: RiskClass = RiskClass.LOW
    side_effects: list[str] = field(default_factory=list)
    seed: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of scenario validation.

    Attributes:
        success: Whether the task was completed successfully
        reason: Explanation of success/failure
        details: Additional validation details
    """

    success: bool
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


class Scenario(ABC):
    """Abstract base class for HITL scenarios.

    A scenario defines:
    1. The task the agent should accomplish
    2. The tools available to the agent
    3. How to validate task completion
    4. How to generate test actions for experiments

    Subclasses implement specific scenarios like email drafting
    or record updates.
    """

    def __init__(self, config: ScenarioConfig) -> None:
        """Initialize the scenario.

        Args:
            config: Scenario configuration
        """
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Return scenario name."""
        pass

    @abstractmethod
    def get_tools(self) -> dict[str, Callable[..., Any]]:
        """Return the tools available in this scenario.

        Returns:
            Dict mapping tool names to callable implementations
        """
        pass

    @abstractmethod
    def generate_action(self, correct: bool = True) -> Action:
        """Generate an action for this scenario.

        Args:
            correct: If True, generate a correct action. If False,
                    generate an incorrect action (for error injection).

        Returns:
            An Action that the agent might propose
        """
        pass

    @abstractmethod
    def validate_result(self, result: dict[str, Any]) -> ValidationResult:
        """Validate the result of executing an action.

        Args:
            result: The tool execution result

        Returns:
            ValidationResult indicating success/failure
        """
        pass

    def get_initial_state(self) -> dict[str, Any]:
        """Get the initial state for this scenario.

        Override to provide scenario-specific initial state.

        Returns:
            Initial state dictionary
        """
        return {
            "scenario_id": self.config.scenario_id,
            "messages": [],
        }

    def get_task_description(self) -> str:
        """Get a human-readable task description.

        Returns:
            Description of what the agent should do
        """
        return self.config.description or f"Complete the {self.name} scenario"

    def reset(self) -> None:
        """Reset scenario state for a new run.

        Override if the scenario maintains state between runs.
        """
        pass
