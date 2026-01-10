"""Always-approve policy for HITL Lab.

This policy represents Tier 4 (no human oversight) - all actions are
automatically approved without human review.
"""

from __future__ import annotations

from typing import Any

from hitloop.core.interfaces import HITLPolicy
from hitloop.core.models import Action, Decision


class AlwaysApprovePolicy(HITLPolicy):
    """Policy that never requests human approval.

    This represents the baseline "no HITL" condition (Tier 4) where all actions
    proceed automatically. Useful for:
    - Baseline experiments measuring impact of human oversight
    - Low-risk scenarios where human review adds no value
    - Testing and development workflows

    Example:
        >>> policy = AlwaysApprovePolicy()
        >>> needs_approval, reason = policy.should_request_approval(action, state)
        >>> assert needs_approval is False
    """

    @property
    def name(self) -> str:
        """Return policy name."""
        return "always_approve"

    def should_request_approval(
        self, action: Action, state: dict[str, Any]
    ) -> tuple[bool, str]:
        """Always returns False - no approval needed.

        Args:
            action: The proposed action (ignored)
            state: Current state (ignored)

        Returns:
            (False, reason) indicating no approval is needed
        """
        return False, "Policy: all actions auto-approved (Tier 4 baseline)"
