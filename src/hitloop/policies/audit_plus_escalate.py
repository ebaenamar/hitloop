"""Audit-plus-escalate policy for HITL Lab.

This policy implements a hybrid approach: random sampling for audit purposes
combined with escalation based on anomaly signals.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

from hitloop.core.interfaces import HITLPolicy
from hitloop.core.models import Action, Decision, RiskClass


class AuditPlusEscalatePolicy(HITLPolicy):
    """Policy combining random auditing with anomaly-based escalation.

    This policy implements a sophisticated oversight strategy:
    1. Random sampling: X% of actions are sampled for human review (audit)
    2. Anomaly escalation: Actions matching anomaly signals trigger review
    3. Risk escalation: High-risk actions always escalate

    This approach balances:
    - Coverage: Random sampling catches unexpected issues
    - Efficiency: Most routine actions proceed without delay
    - Safety: Anomalous behavior triggers immediate review

    Example:
        >>> policy = AuditPlusEscalatePolicy(
        ...     audit_sample_rate=0.1,  # 10% random audit
        ...     anomaly_signals=["unusual_recipient", "large_amount"]
        ... )
    """

    def __init__(
        self,
        audit_sample_rate: float = 0.1,
        anomaly_signals: list[str] | None = None,
        escalate_on_high_risk: bool = True,
        escalate_on_medium_risk: bool = False,
        consecutive_action_threshold: int = 5,
        seed: int | None = None,
    ) -> None:
        """Initialize audit-plus-escalate policy.

        Args:
            audit_sample_rate: Fraction of actions to sample (0.0-1.0)
            anomaly_signals: List of signals that trigger escalation
            escalate_on_high_risk: Always escalate HIGH risk actions
            escalate_on_medium_risk: Always escalate MEDIUM risk actions
            consecutive_action_threshold: Escalate after N same-type actions
            seed: Random seed for reproducibility
        """
        if not 0.0 <= audit_sample_rate <= 1.0:
            raise ValueError("audit_sample_rate must be between 0.0 and 1.0")

        self.audit_sample_rate = audit_sample_rate
        self.anomaly_signals = set(anomaly_signals or [])
        self.escalate_on_high_risk = escalate_on_high_risk
        self.escalate_on_medium_risk = escalate_on_medium_risk
        self.consecutive_action_threshold = consecutive_action_threshold

        self._rng = random.Random(seed)
        self._action_history: list[str] = []

    @property
    def name(self) -> str:
        """Return policy name."""
        return "audit_plus_escalate"

    def should_request_approval(
        self, action: Action, state: dict[str, Any]
    ) -> tuple[bool, str]:
        """Determine if approval is needed via audit or escalation.

        Args:
            action: The proposed action
            state: Current workflow state

        Returns:
            (needs_approval, reason) tuple
        """
        # Check risk-based escalation first
        if action.risk_class == RiskClass.HIGH and self.escalate_on_high_risk:
            return True, "Escalation: HIGH risk action"

        if action.risk_class == RiskClass.MEDIUM and self.escalate_on_medium_risk:
            return True, "Escalation: MEDIUM risk action"

        # Check for anomaly signals in state
        detected_anomalies = state.get("anomaly_signals", [])
        if isinstance(detected_anomalies, list):
            matching = self.anomaly_signals.intersection(detected_anomalies)
            if matching:
                return True, f"Escalation: Anomaly signals detected: {', '.join(matching)}"

        # Check for anomaly signals in action metadata
        action_signals = set(action.context_refs)
        matching = self.anomaly_signals.intersection(action_signals)
        if matching:
            return True, f"Escalation: Action flagged with: {', '.join(matching)}"

        # Check consecutive action threshold
        self._action_history.append(action.tool_name)
        if len(self._action_history) >= self.consecutive_action_threshold:
            recent = self._action_history[-self.consecutive_action_threshold :]
            if len(set(recent)) == 1:
                return True, f"Escalation: {self.consecutive_action_threshold} consecutive '{action.tool_name}' actions"

        # Random audit sampling
        # Use action ID for deterministic sampling (same action always gets same decision)
        action_hash = int(hashlib.md5(action.id.encode()).hexdigest()[:8], 16)
        sample_threshold = int(self.audit_sample_rate * 0xFFFFFFFF)

        if action_hash < sample_threshold:
            return True, f"Audit: Random sample ({self.audit_sample_rate * 100:.1f}% rate)"

        return False, "No audit or escalation triggered"

    def post_decision_update(
        self, state: dict[str, Any], action: Action, decision: Decision
    ) -> dict[str, Any]:
        """Update policy state after decision.

        Tracks:
        - Rejection patterns for anomaly detection tuning
        - Human feedback for learning signal analysis

        Args:
            state: Current state
            action: The evaluated action
            decision: Human's decision

        Returns:
            Updated state
        """
        audit_log = state.get("_audit_log", [])
        audit_log.append(
            {
                "action_id": action.id,
                "tool_name": action.tool_name,
                "approved": decision.approved,
                "reason": decision.reason,
                "tags": decision.tags,
            }
        )
        state["_audit_log"] = audit_log

        # Track rejection as potential anomaly signal
        if not decision.approved:
            rejection_count = state.get("_rejection_count", 0) + 1
            state["_rejection_count"] = rejection_count

            # Add anomaly signal if multiple rejections
            if rejection_count >= 3:
                anomalies = state.get("anomaly_signals", [])
                if "repeated_rejections" not in anomalies:
                    anomalies.append("repeated_rejections")
                    state["anomaly_signals"] = anomalies

        return state

    def post_execution_update(
        self, state: dict[str, Any], action: Action, result: dict[str, Any]
    ) -> dict[str, Any]:
        """Update policy state after execution.

        Tracks execution failures as potential anomaly signals.

        Args:
            state: Current state
            action: The executed action
            result: Execution result

        Returns:
            Updated state
        """
        if not result.get("success", True):
            failure_count = state.get("_tool_failure_count", 0) + 1
            state["_tool_failure_count"] = failure_count

            # Add anomaly signal if multiple failures
            if failure_count >= 2:
                anomalies = state.get("anomaly_signals", [])
                if "repeated_failures" not in anomalies:
                    anomalies.append("repeated_failures")
                    state["anomaly_signals"] = anomalies

        return state

    def reset(self) -> None:
        """Reset policy state for new run."""
        self._action_history.clear()
