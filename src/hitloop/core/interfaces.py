"""Core interfaces for HITL Lab.

This module defines the abstract interfaces that approval backends and policies
must implement. These interfaces ensure clean separation of concerns and enable
easy extension with custom implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from hitloop.core.models import Action, ApprovalRequest, Decision


class ApprovalBackend(ABC):
    """Abstract interface for approval backends.

    An ApprovalBackend is responsible for presenting approval requests to humans
    and collecting their decisions. Implementations might use CLI prompts, web UIs,
    Slack integrations, or external services like HumanLayer.

    Example:
        >>> backend = CLIBackend()
        >>> decision = await backend.request_approval(request)
        >>> print(f"Approved: {decision.approved}")
    """

    @abstractmethod
    async def request_approval(self, request: ApprovalRequest) -> Decision:
        """Request approval for an action.

        This method should present the approval request to a human and wait
        for their decision. The implementation is responsible for:
        1. Formatting the request for display
        2. Collecting the human's decision
        3. Recording timing information
        4. Returning a Decision object

        Args:
            request: The approval request containing action and context

        Returns:
            A Decision object with the approval result

        Raises:
            ApprovalTimeoutError: If the request times out
            ApprovalError: If there's an error in the approval process
        """
        pass

    async def close(self) -> None:
        """Clean up any resources held by the backend.

        Override this method if your backend needs cleanup (e.g., closing
        connections, shutting down servers).
        """
        pass


class HITLPolicy(ABC):
    """Abstract interface for Human-in-the-Loop policies.

    A HITLPolicy determines when human approval is needed and how to update
    state based on decisions and execution results. Policies implement the
    decision logic for different tiers of human oversight.

    The policy lifecycle for each action:
    1. should_request_approval() - Determine if approval is needed
    2. post_decision_update() - Update state after human decision
    3. post_execution_update() - Update state after tool execution

    Example:
        >>> policy = RiskBasedPolicy()
        >>> needs_approval, reason = policy.should_request_approval(action, state)
        >>> if needs_approval:
        ...     decision = await backend.request_approval(...)
        ...     state = policy.post_decision_update(state, action, decision)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the policy name for logging and display."""
        pass

    @abstractmethod
    def should_request_approval(
        self, action: Action, state: dict[str, Any]
    ) -> tuple[bool, str]:
        """Determine if an action requires human approval.

        This method encapsulates the policy's decision logic. It examines the
        action and current state to determine whether human oversight is needed.

        Args:
            action: The proposed action to evaluate
            state: Current state of the graph/workflow

        Returns:
            A tuple of (needs_approval: bool, reason: str) where reason
            explains why approval is or isn't needed.
        """
        pass

    def post_decision_update(
        self, state: dict[str, Any], action: Action, decision: Decision
    ) -> dict[str, Any]:
        """Update state after an approval decision is made.

        This hook allows policies to modify state based on the human's decision.
        For example, a policy might track rejection patterns or update risk
        assessments based on human feedback.

        Args:
            state: Current state to update
            action: The action that was evaluated
            decision: The human's decision

        Returns:
            Updated state dictionary
        """
        return state

    def post_execution_update(
        self, state: dict[str, Any], action: Action, result: dict[str, Any]
    ) -> dict[str, Any]:
        """Update state after tool execution completes.

        This hook allows policies to learn from execution outcomes. For example,
        a policy might adjust risk assessments based on execution success/failure
        rates.

        Args:
            state: Current state to update
            action: The action that was executed
            result: The execution result

        Returns:
            Updated state dictionary
        """
        return state


class ApprovalError(Exception):
    """Base exception for approval-related errors."""

    pass


class ApprovalTimeoutError(ApprovalError):
    """Raised when an approval request times out."""

    pass


class ApprovalCancelledError(ApprovalError):
    """Raised when an approval request is cancelled."""

    pass
