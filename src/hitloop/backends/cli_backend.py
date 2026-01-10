"""CLI-based approval backend for HITL Lab.

This module provides a command-line interface for human approval decisions,
suitable for local development and testing.
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable

from hitloop.core.interfaces import ApprovalBackend, ApprovalCancelledError
from hitloop.core.models import ApprovalRequest, Decision


class CLIBackend(ApprovalBackend):
    """Command-line approval backend.

    Presents approval requests to the terminal and collects yes/no decisions
    from the user. Supports auto-approval mode for automated testing.

    Example:
        >>> backend = CLIBackend()
        >>> decision = await backend.request_approval(request)

        # Auto-approve mode for testing
        >>> backend = CLIBackend(auto_approve=True)
    """

    def __init__(
        self,
        auto_approve: bool = False,
        auto_approve_delay_ms: float = 0.0,
        timeout_seconds: float = 0.0,
        input_func: Callable[[], str] | None = None,
        output_func: Callable[[str], None] | None = None,
    ) -> None:
        """Initialize the CLI backend.

        Args:
            auto_approve: If True, automatically approve all requests
            auto_approve_delay_ms: Simulated delay for auto-approval (ms)
            timeout_seconds: Timeout for human response (0 = no timeout)
            input_func: Custom input function (for testing), defaults to input()
            output_func: Custom output function (for testing), defaults to print()
        """
        self.auto_approve = auto_approve
        self.auto_approve_delay_ms = auto_approve_delay_ms
        self.timeout_seconds = timeout_seconds
        self._input_func = input_func or (lambda: input())
        self._output_func = output_func or print

    async def request_approval(self, request: ApprovalRequest) -> Decision:
        """Request approval via CLI.

        Displays the approval request and prompts for yes/no input.
        Optionally collects a reason for the decision.

        Args:
            request: The approval request

        Returns:
            Decision with approval result and timing

        Raises:
            ApprovalCancelledError: If the user cancels (Ctrl+C)
        """
        start_time = time.time()

        if self.auto_approve:
            if self.auto_approve_delay_ms > 0:
                await asyncio.sleep(self.auto_approve_delay_ms / 1000)
            latency_ms = (time.time() - start_time) * 1000
            return Decision(
                action_id=request.action.id,
                approved=True,
                reason="Auto-approved",
                decided_by="cli:auto",
                latency_ms=latency_ms,
            )

        # Display the request
        self._output_func(request.format_for_display())

        # Get decision (with optional timeout)
        try:
            if self.timeout_seconds > 0:
                try:
                    approved = await asyncio.wait_for(
                        self._get_yes_no_input("Approve this action? (y/n): "),
                        timeout=self.timeout_seconds
                    )
                except asyncio.TimeoutError:
                    self._output_func(f"\n⏰ Timeout after {self.timeout_seconds}s - rejecting")
                    latency_ms = (time.time() - start_time) * 1000
                    return Decision(
                        action_id=request.action.id,
                        approved=False,
                        reason=f"Timeout after {self.timeout_seconds}s",
                        decided_by="cli:timeout",
                        latency_ms=latency_ms,
                    )
            else:
                approved = await self._get_yes_no_input("Approve this action? (y/n): ")
        except KeyboardInterrupt:
            raise ApprovalCancelledError("User cancelled approval")

        # Optionally get reason
        reason = ""
        try:
            self._output_func("Reason (optional, press Enter to skip): ")
            reason = await asyncio.get_event_loop().run_in_executor(
                None, self._input_func
            )
        except (KeyboardInterrupt, EOFError):
            pass

        latency_ms = (time.time() - start_time) * 1000

        return Decision(
            action_id=request.action.id,
            approved=approved,
            reason=reason.strip() if reason else ("Approved" if approved else "Rejected"),
            decided_by="cli:human",
            latency_ms=latency_ms,
        )

    async def _get_yes_no_input(self, prompt: str) -> bool:
        """Get yes/no input from user.

        Args:
            prompt: The prompt to display

        Returns:
            True for yes, False for no
        """
        while True:
            self._output_func(prompt)
            try:
                response = await asyncio.get_event_loop().run_in_executor(
                    None, self._input_func
                )
                response = response.strip().lower()
                if response in ("y", "yes", "1", "true"):
                    return True
                if response in ("n", "no", "0", "false"):
                    return False
                self._output_func("Please enter 'y' or 'n'")
            except EOFError:
                # Treat EOF as rejection
                return False


class AutoApproveBackend(ApprovalBackend):
    """Backend that auto-approves everything.

    Useful for baseline experiments (Tier 4) where no human oversight is used.
    """

    def __init__(self, delay_ms: float = 0.0) -> None:
        """Initialize auto-approve backend.

        Args:
            delay_ms: Simulated approval delay in milliseconds
        """
        self.delay_ms = delay_ms

    async def request_approval(self, request: ApprovalRequest) -> Decision:
        """Auto-approve the request.

        Args:
            request: The approval request

        Returns:
            Decision approving the action
        """
        start_time = time.time()

        if self.delay_ms > 0:
            await asyncio.sleep(self.delay_ms / 1000)

        latency_ms = (time.time() - start_time) * 1000

        return Decision(
            action_id=request.action.id,
            approved=True,
            reason="Auto-approved (no human in loop)",
            decided_by="auto",
            latency_ms=latency_ms,
        )


class AutoRejectBackend(ApprovalBackend):
    """Backend that auto-rejects everything.

    Useful for testing rejection handling and safety validation.
    """

    def __init__(self, delay_ms: float = 0.0) -> None:
        """Initialize auto-reject backend.

        Args:
            delay_ms: Simulated rejection delay in milliseconds
        """
        self.delay_ms = delay_ms

    async def request_approval(self, request: ApprovalRequest) -> Decision:
        """Auto-reject the request.

        Args:
            request: The approval request

        Returns:
            Decision rejecting the action
        """
        start_time = time.time()

        if self.delay_ms > 0:
            await asyncio.sleep(self.delay_ms / 1000)

        latency_ms = (time.time() - start_time) * 1000

        return Decision(
            action_id=request.action.id,
            approved=False,
            reason="Auto-rejected (testing mode)",
            decided_by="auto",
            latency_ms=latency_ms,
        )


class ScriptedBackend(ApprovalBackend):
    """Backend with scripted responses for testing.

    Allows specifying a sequence of approval decisions for deterministic testing.
    """

    def __init__(
        self,
        decisions: list[bool] | None = None,
        delay_ms: float = 0.0,
        default_approve: bool = True,
    ) -> None:
        """Initialize scripted backend.

        Args:
            decisions: List of approval decisions (True/False) to return in order
            delay_ms: Simulated delay in milliseconds
            default_approve: Default decision when scripted list is exhausted
        """
        self.decisions = list(decisions) if decisions else []
        self.delay_ms = delay_ms
        self.default_approve = default_approve
        self._index = 0

    async def request_approval(self, request: ApprovalRequest) -> Decision:
        """Return the next scripted decision.

        Args:
            request: The approval request

        Returns:
            Decision based on the script or default
        """
        start_time = time.time()

        if self.delay_ms > 0:
            await asyncio.sleep(self.delay_ms / 1000)

        if self._index < len(self.decisions):
            approved = self.decisions[self._index]
            self._index += 1
        else:
            approved = self.default_approve

        latency_ms = (time.time() - start_time) * 1000

        return Decision(
            action_id=request.action.id,
            approved=approved,
            reason=f"Scripted decision #{self._index}",
            decided_by="scripted",
            latency_ms=latency_ms,
        )

    def reset(self) -> None:
        """Reset the script index to replay decisions."""
        self._index = 0
