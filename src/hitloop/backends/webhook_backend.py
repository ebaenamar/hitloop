"""Webhook-based approval backend for hitloop.

This module provides a generic webhook backend that can integrate with any
third-party service (Slack, Telegram, Discord, custom apps, etc.) through
a simple HTTP callback pattern.

Architecture:
    1. Your app calls request_approval()
    2. WebhookBackend sends POST to your configured outbound_url
    3. Your service (Slack bot, Telegram bot, etc.) shows the request to human
    4. Human clicks Approve/Reject
    5. Your service POSTs back to the callback endpoint
    6. WebhookBackend resolves the pending future

This is completely agnostic - hitloop doesn't care what service you use.
You just need to:
    1. Handle the outbound webhook (show to human)
    2. POST back the decision to the callback URL
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Awaitable
from datetime import datetime, timezone

from hitloop.core.interfaces import ApprovalBackend
from hitloop.core.models import ApprovalRequest, Decision


@dataclass
class PendingApproval:
    """Tracks a pending approval request."""
    request: ApprovalRequest
    future: asyncio.Future[Decision]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    callback_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


class WebhookBackend(ApprovalBackend):
    """Generic webhook-based approval backend.
    
    This backend is designed to be completely agnostic to the third-party
    service you're integrating with. It works with any service that can:
    1. Receive HTTP POST requests (outbound webhook)
    2. Send HTTP POST requests back (callback)
    
    Example with a generic setup:
        >>> backend = WebhookBackend(
        ...     send_request=my_send_function,  # Your function to send to Slack/Telegram/etc
        ...     timeout_seconds=300,
        ... )
        >>> 
        >>> # In your webhook handler (FastAPI, Flask, etc.):
        >>> @app.post("/hitloop/callback")
        >>> async def handle_callback(data: dict):
        ...     await backend.handle_callback(
        ...         callback_id=data["callback_id"],
        ...         approved=data["approved"],
        ...         reason=data.get("reason", ""),
        ...         decided_by=data.get("user", "unknown"),
        ...     )
    
    Example with Slack:
        >>> async def send_to_slack(request: ApprovalRequest, callback_id: str, callback_url: str):
        ...     # Send Slack message with buttons
        ...     await slack_client.chat_postMessage(
        ...         channel="#approvals",
        ...         blocks=[
        ...             {"type": "section", "text": {"type": "mrkdwn", "text": f"*Approval needed*\\n{request.action.summary()}"}},
        ...             {"type": "actions", "elements": [
        ...                 {"type": "button", "text": {"type": "plain_text", "text": "Approve"}, "action_id": f"approve_{callback_id}"},
        ...                 {"type": "button", "text": {"type": "plain_text", "text": "Reject"}, "action_id": f"reject_{callback_id}"},
        ...             ]}
        ...         ]
        ...     )
        >>> 
        >>> backend = WebhookBackend(send_request=send_to_slack)
    
    Example with Telegram:
        >>> async def send_to_telegram(request: ApprovalRequest, callback_id: str, callback_url: str):
        ...     keyboard = InlineKeyboardMarkup([
        ...         [InlineKeyboardButton("✅ Approve", callback_data=f"approve:{callback_id}")],
        ...         [InlineKeyboardButton("❌ Reject", callback_data=f"reject:{callback_id}")],
        ...     ])
        ...     await bot.send_message(chat_id=ADMIN_CHAT, text=request.action.summary(), reply_markup=keyboard)
        >>> 
        >>> backend = WebhookBackend(send_request=send_to_telegram)
    """
    
    def __init__(
        self,
        send_request: Callable[[ApprovalRequest, str, str], Awaitable[None]],
        timeout_seconds: float = 300.0,
        callback_base_url: str = "http://localhost:8000/hitloop/callback",
        on_timeout: Callable[[ApprovalRequest], Awaitable[Decision]] | None = None,
    ) -> None:
        """Initialize the webhook backend.
        
        Args:
            send_request: Async function to send approval request to your service.
                          Signature: (request, callback_id, callback_url) -> None
            timeout_seconds: How long to wait for approval before timing out.
                            Set to 0 for no timeout (not recommended for production).
            callback_base_url: Base URL where your app receives callbacks.
            on_timeout: Optional custom handler for timeouts. If None, returns
                       a rejected Decision with reason "Timeout".
        """
        self.send_request = send_request
        self.timeout_seconds = timeout_seconds
        self.callback_base_url = callback_base_url.rstrip("/")
        self.on_timeout = on_timeout
        
        # Track pending approvals
        self._pending: dict[str, PendingApproval] = {}
    
    async def request_approval(self, request: ApprovalRequest) -> Decision:
        """Request approval via webhook.
        
        Sends the request to your configured service and waits for a callback.
        
        Args:
            request: The approval request
            
        Returns:
            Decision from the human (or timeout)
            
        Raises:
            ApprovalTimeoutError: If timeout_seconds > 0 and no response received
        """
        start_time = time.time()
        
        # Create pending approval tracker
        loop = asyncio.get_event_loop()
        future: asyncio.Future[Decision] = loop.create_future()
        pending = PendingApproval(request=request, future=future)
        
        self._pending[pending.callback_id] = pending
        
        # Build callback URL
        callback_url = f"{self.callback_base_url}/{pending.callback_id}"
        
        try:
            # Send to external service
            await self.send_request(request, pending.callback_id, callback_url)
            
            # Wait for callback (with timeout)
            if self.timeout_seconds > 0:
                try:
                    decision = await asyncio.wait_for(future, timeout=self.timeout_seconds)
                except asyncio.TimeoutError:
                    # Handle timeout
                    if self.on_timeout:
                        decision = await self.on_timeout(request)
                    else:
                        decision = Decision(
                            action_id=request.action.id,
                            approved=False,
                            reason=f"Timeout after {self.timeout_seconds}s - no human response",
                            decided_by="system:timeout",
                            latency_ms=(time.time() - start_time) * 1000,
                        )
            else:
                # No timeout - wait indefinitely (not recommended)
                decision = await future
            
            # Add latency if not already set
            if decision.latency_ms is None:
                decision = Decision(
                    action_id=decision.action_id,
                    approved=decision.approved,
                    reason=decision.reason,
                    decided_by=decision.decided_by,
                    tags=decision.tags,
                    latency_ms=(time.time() - start_time) * 1000,
                )
            
            return decision
            
        finally:
            # Cleanup
            self._pending.pop(pending.callback_id, None)
    
    async def handle_callback(
        self,
        callback_id: str,
        approved: bool,
        reason: str = "",
        decided_by: str = "human",
        tags: list[str] | None = None,
    ) -> bool:
        """Handle a callback from your external service.
        
        Call this from your webhook endpoint when the human responds.
        
        Args:
            callback_id: The callback_id from the original request
            approved: Whether the action was approved
            reason: Optional reason for the decision
            decided_by: Identifier for who made the decision (e.g., "slack:@john")
            tags: Optional tags for the decision
            
        Returns:
            True if the callback was handled, False if callback_id not found
            
        Example (FastAPI):
            >>> @app.post("/hitloop/callback/{callback_id}")
            >>> async def callback(callback_id: str, data: CallbackData):
            ...     success = await backend.handle_callback(
            ...         callback_id=callback_id,
            ...         approved=data.approved,
            ...         reason=data.reason,
            ...         decided_by=f"slack:{data.user_id}",
            ...     )
            ...     return {"handled": success}
        """
        pending = self._pending.get(callback_id)
        if pending is None:
            return False
        
        if pending.future.done():
            return False
        
        decision = Decision(
            action_id=pending.request.action.id,
            approved=approved,
            reason=reason or ("Approved" if approved else "Rejected"),
            decided_by=decided_by,
            tags=tags or [],
        )
        
        pending.future.set_result(decision)
        return True
    
    def get_pending_count(self) -> int:
        """Get the number of pending approval requests."""
        return len(self._pending)
    
    def get_pending_requests(self) -> list[tuple[str, ApprovalRequest]]:
        """Get all pending approval requests.
        
        Returns:
            List of (callback_id, request) tuples
        """
        return [(p.callback_id, p.request) for p in self._pending.values()]
    
    async def cancel_pending(self, callback_id: str, reason: str = "Cancelled") -> bool:
        """Cancel a pending approval request.
        
        Args:
            callback_id: The callback_id to cancel
            reason: Reason for cancellation
            
        Returns:
            True if cancelled, False if not found
        """
        return await self.handle_callback(
            callback_id=callback_id,
            approved=False,
            reason=reason,
            decided_by="system:cancelled",
        )


class SimpleHTTPWebhookBackend(WebhookBackend):
    """Webhook backend that sends HTTP POST requests.
    
    This is a convenience class for simple integrations where you just
    need to POST JSON to a URL and receive callbacks.
    
    Example:
        >>> backend = SimpleHTTPWebhookBackend(
        ...     outbound_url="https://my-service.com/approval-requests",
        ...     callback_base_url="https://my-app.com/hitloop/callback",
        ...     timeout_seconds=300,
        ... )
    """
    
    def __init__(
        self,
        outbound_url: str,
        callback_base_url: str = "http://localhost:8000/hitloop/callback",
        timeout_seconds: float = 300.0,
        headers: dict[str, str] | None = None,
        on_timeout: Callable[[ApprovalRequest], Awaitable[Decision]] | None = None,
    ) -> None:
        """Initialize simple HTTP webhook backend.
        
        Args:
            outbound_url: URL to POST approval requests to
            callback_base_url: Base URL for callbacks
            timeout_seconds: Timeout for approval
            headers: Optional headers to include in outbound requests
            on_timeout: Optional timeout handler
        """
        self.outbound_url = outbound_url
        self.headers = headers or {}
        
        super().__init__(
            send_request=self._send_http_request,
            timeout_seconds=timeout_seconds,
            callback_base_url=callback_base_url,
            on_timeout=on_timeout,
        )
    
    async def _send_http_request(
        self,
        request: ApprovalRequest,
        callback_id: str,
        callback_url: str,
    ) -> None:
        """Send HTTP POST request."""
        import aiohttp
        
        payload = {
            "callback_id": callback_id,
            "callback_url": callback_url,
            "run_id": request.run_id,
            "action": {
                "id": request.action.id,
                "tool_name": request.action.tool_name,
                "tool_args": request.action.tool_args,
                "risk_class": request.action.risk_class.value,
                "rationale": request.action.rationale,
                "summary": request.action.summary(),
            },
            "policy_name": request.policy_name,
            "policy_reason": request.policy_reason,
            "context": request.summary_context,
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.outbound_url,
                json=payload,
                headers={"Content-Type": "application/json", **self.headers},
            ) as response:
                if response.status >= 400:
                    raise RuntimeError(f"Failed to send webhook: {response.status}")
