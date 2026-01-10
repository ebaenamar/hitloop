"""Risk-based approval policy for HITL Lab.

This policy requests human approval based on action risk classification
and configurable rules for high-impact operations.
"""

from __future__ import annotations

from typing import Any

from hitloop.core.interfaces import HITLPolicy
from hitloop.core.models import Action, Decision, RiskClass


class RiskBasedPolicy(HITLPolicy):
    """Policy that requests approval based on risk classification.

    This policy implements risk-based filtering where only high-risk actions
    or actions matching certain criteria require human approval. This is the
    recommended production policy balancing safety with efficiency.

    Configuration:
        - require_approval_for_high: Require approval for HIGH risk actions
        - require_approval_for_medium: Require approval for MEDIUM risk
        - high_risk_tools: List of tool names always considered high risk
        - sensitive_arg_patterns: Argument patterns triggering approval

    Example:
        >>> policy = RiskBasedPolicy(
        ...     high_risk_tools=["send_email", "delete_record"],
        ...     require_approval_for_medium=True
        ... )
        >>> needs, reason = policy.should_request_approval(action, state)
    """

    def __init__(
        self,
        require_approval_for_high: bool = True,
        require_approval_for_medium: bool = False,
        high_risk_tools: list[str] | None = None,
        sensitive_arg_patterns: dict[str, list[str]] | None = None,
        max_amount_without_approval: float | None = None,
    ) -> None:
        """Initialize risk-based policy.

        Args:
            require_approval_for_high: Require approval for HIGH risk actions
            require_approval_for_medium: Require approval for MEDIUM risk actions
            high_risk_tools: Tool names that always require approval
            sensitive_arg_patterns: Dict mapping arg names to sensitive values
            max_amount_without_approval: Max monetary amount without approval
        """
        self.require_approval_for_high = require_approval_for_high
        self.require_approval_for_medium = require_approval_for_medium
        self.high_risk_tools = set(high_risk_tools or [])
        self.sensitive_arg_patterns = sensitive_arg_patterns or {}
        self.max_amount_without_approval = max_amount_without_approval

    @property
    def name(self) -> str:
        """Return policy name."""
        return "risk_based"

    def should_request_approval(
        self, action: Action, state: dict[str, Any]
    ) -> tuple[bool, str]:
        """Determine if approval is needed based on risk.

        Checks the following in order:
        1. Is the tool in the high-risk tools list?
        2. Does the action's risk class require approval?
        3. Do any arguments match sensitive patterns?
        4. Does any amount exceed the threshold?

        Args:
            action: The proposed action
            state: Current workflow state

        Returns:
            (needs_approval, reason) tuple
        """
        # Check if tool is explicitly high-risk
        if action.tool_name in self.high_risk_tools:
            return True, f"Tool '{action.tool_name}' is in high-risk tool list"

        # Check risk class
        if action.risk_class == RiskClass.HIGH and self.require_approval_for_high:
            return True, f"Action has HIGH risk classification"

        if action.risk_class == RiskClass.MEDIUM and self.require_approval_for_medium:
            return True, f"Action has MEDIUM risk classification"

        # Check sensitive argument patterns
        for arg_name, sensitive_values in self.sensitive_arg_patterns.items():
            if arg_name in action.tool_args:
                arg_value = str(action.tool_args[arg_name]).lower()
                for pattern in sensitive_values:
                    if pattern.lower() in arg_value:
                        return True, f"Argument '{arg_name}' contains sensitive pattern '{pattern}'"

        # Check amount threshold
        if self.max_amount_without_approval is not None:
            for amount_key in ["amount", "value", "price", "cost", "total"]:
                if amount_key in action.tool_args:
                    try:
                        amount = float(action.tool_args[amount_key])
                        if amount > self.max_amount_without_approval:
                            return True, f"Amount ${amount:.2f} exceeds threshold ${self.max_amount_without_approval:.2f}"
                    except (TypeError, ValueError):
                        pass

        return False, "Action does not meet approval criteria"

    def post_decision_update(
        self, state: dict[str, Any], action: Action, decision: Decision
    ) -> dict[str, Any]:
        """Track rejection patterns for analysis.

        Args:
            state: Current state
            action: The evaluated action
            decision: Human's decision

        Returns:
            Updated state with rejection tracking
        """
        if not decision.approved:
            rejections = state.get("_risk_policy_rejections", [])
            rejections.append(
                {
                    "action_id": action.id,
                    "tool_name": action.tool_name,
                    "risk_class": action.risk_class.value,
                    "reason": decision.reason,
                }
            )
            state["_risk_policy_rejections"] = rejections
        return state
