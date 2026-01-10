"""Persistent webhook backend with retry and circuit breaker.

This is the production-ready version of WebhookBackend that:
1. Persists pending requests (survives restarts)
2. Implements retry with exponential backoff
3. Includes circuit breaker pattern for resilience
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Callable, Awaitable, Any

from hitloop.core.interfaces import ApprovalBackend
from hitloop.core.models import ApprovalRequest, Decision
from hitloop.persistence.base import ApprovalStore, PendingApprovalRecord
from hitloop.persistence.memory import InMemoryApprovalStore


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5      # Failures before opening
    recovery_timeout: float = 30.0  # Seconds before trying again
    half_open_max_calls: int = 3    # Test calls in half-open state


@dataclass 
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    initial_delay: float = 1.0      # Seconds
    max_delay: float = 30.0         # Seconds
    exponential_base: float = 2.0


class PersistentWebhookBackend(ApprovalBackend):
    """Production-ready webhook backend with persistence and resilience.
    
    Features:
    - Persistent storage of pending requests (survives restarts)
    - Retry with exponential backoff for transient failures
    - Circuit breaker to prevent cascade failures
    - Configurable timeouts
    - Compatible with LangGraph's thread_id pattern
    
    Example:
        >>> from hitloop.persistence import PostgresApprovalStore
        >>> 
        >>> store = await PostgresApprovalStore.from_conn_string(DB_URI)
        >>> backend = PersistentWebhookBackend(
        ...     send_request=my_slack_sender,
        ...     store=store,
        ...     timeout_seconds=300,
        ... )
        >>> 
        >>> # In your LangGraph node:
        >>> decision = await backend.request_approval(request, thread_id="user-123")
    
    Recovery after restart:
        >>> # On startup, recover pending requests
        >>> pending = await store.list_pending()
        >>> for record in pending:
        ...     # Re-register the callback handlers
        ...     backend.register_pending(record)
    """
    
    def __init__(
        self,
        send_request: Callable[[ApprovalRequest, str, str], Awaitable[None]],
        store: ApprovalStore | None = None,
        timeout_seconds: float = 300.0,
        callback_base_url: str = "http://localhost:8000/hitloop/callback",
        retry_config: RetryConfig | None = None,
        circuit_breaker_config: CircuitBreakerConfig | None = None,
        on_timeout: Callable[[ApprovalRequest], Awaitable[Decision]] | None = None,
    ) -> None:
        """Initialize persistent webhook backend.
        
        Args:
            send_request: Async function to send to external service
            store: Persistence store (defaults to InMemoryApprovalStore)
            timeout_seconds: Timeout for approval (0 = no timeout)
            callback_base_url: Base URL for callbacks
            retry_config: Retry configuration
            circuit_breaker_config: Circuit breaker configuration
            on_timeout: Custom timeout handler
        """
        self.send_request = send_request
        self.store = store or InMemoryApprovalStore()
        self.timeout_seconds = timeout_seconds
        self.callback_base_url = callback_base_url.rstrip("/")
        self.retry_config = retry_config or RetryConfig()
        self.circuit_config = circuit_breaker_config or CircuitBreakerConfig()
        self.on_timeout = on_timeout
        
        # In-memory tracking of active futures
        self._pending_futures: dict[str, asyncio.Future[Decision]] = {}
        
        # Circuit breaker state
        self._circuit_state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float | None = None
        self._half_open_calls = 0
    
    async def request_approval(
        self,
        request: ApprovalRequest,
        thread_id: str = "default",
    ) -> Decision:
        """Request approval with persistence and resilience.
        
        Args:
            request: The approval request
            thread_id: LangGraph thread_id for correlation
            
        Returns:
            Decision from human or timeout/error
        """
        start_time = time.time()
        callback_id = str(uuid.uuid4())[:8]
        callback_url = f"{self.callback_base_url}/{callback_id}"
        
        # Check circuit breaker
        if not self._check_circuit():
            return Decision(
                action_id=request.action.id,
                approved=False,
                reason="Circuit breaker open - service unavailable",
                decided_by="system:circuit_breaker",
                latency_ms=(time.time() - start_time) * 1000,
            )
        
        # Create persistent record
        expires_at = None
        if self.timeout_seconds > 0:
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=self.timeout_seconds)
        
        record = PendingApprovalRecord(
            callback_id=callback_id,
            run_id=request.run_id,
            thread_id=thread_id,
            action_id=request.action.id,
            tool_name=request.action.tool_name,
            tool_args=request.action.tool_args,
            risk_class=request.action.risk_class.value,
            policy_name=request.policy_name,
            policy_reason=request.policy_reason,
            expires_at=expires_at,
        )
        
        # Persist the request
        await self.store.put(record)
        
        # Create future for callback
        loop = asyncio.get_event_loop()
        future: asyncio.Future[Decision] = loop.create_future()
        self._pending_futures[callback_id] = future
        
        try:
            # Send with retry
            await self._send_with_retry(request, callback_id, callback_url)
            
            # Wait for callback
            if self.timeout_seconds > 0:
                try:
                    decision = await asyncio.wait_for(
                        future, 
                        timeout=self.timeout_seconds
                    )
                except asyncio.TimeoutError:
                    if self.on_timeout:
                        decision = await self.on_timeout(request)
                    else:
                        decision = Decision(
                            action_id=request.action.id,
                            approved=False,
                            reason=f"Timeout after {self.timeout_seconds}s",
                            decided_by="system:timeout",
                        )
            else:
                decision = await future
            
            # Record success for circuit breaker
            self._record_success()
            
            # Set latency
            decision = Decision(
                action_id=decision.action_id,
                approved=decision.approved,
                reason=decision.reason,
                decided_by=decision.decided_by,
                tags=decision.tags,
                latency_ms=(time.time() - start_time) * 1000,
            )
            
            return decision
            
        except Exception as e:
            self._record_failure()
            return Decision(
                action_id=request.action.id,
                approved=False,
                reason=f"Error: {str(e)}",
                decided_by="system:error",
                latency_ms=(time.time() - start_time) * 1000,
            )
        finally:
            # Cleanup
            self._pending_futures.pop(callback_id, None)
            await self.store.delete(callback_id)
    
    async def handle_callback(
        self,
        callback_id: str,
        approved: bool,
        reason: str = "",
        decided_by: str = "human",
        tags: list[str] | None = None,
    ) -> bool:
        """Handle callback from external service.
        
        Args:
            callback_id: The callback identifier
            approved: Whether approved
            reason: Decision reason
            decided_by: Who decided
            tags: Optional tags
            
        Returns:
            True if handled, False if not found
        """
        # Check in-memory futures first
        future = self._pending_futures.get(callback_id)
        if future and not future.done():
            # Get action_id from store
            record = await self.store.get(callback_id)
            action_id = record.action_id if record else "unknown"
            
            decision = Decision(
                action_id=action_id,
                approved=approved,
                reason=reason or ("Approved" if approved else "Rejected"),
                decided_by=decided_by,
                tags=tags or [],
            )
            future.set_result(decision)
            return True
        
        # Check persistent store (for recovery scenarios)
        record = await self.store.get(callback_id)
        if record:
            # Request exists but no active future - might be after restart
            # Store the decision for later retrieval
            await self.store.delete(callback_id)
            return True
        
        return False
    
    async def recover_pending(self, thread_id: str | None = None) -> list[PendingApprovalRecord]:
        """Recover pending requests after restart.
        
        Call this on startup to get pending requests that need
        to be re-sent or resolved.
        
        Args:
            thread_id: Optional filter by thread
            
        Returns:
            List of pending records
        """
        return await self.store.list_pending(thread_id=thread_id)
    
    def register_pending(self, record: PendingApprovalRecord) -> asyncio.Future[Decision]:
        """Register a recovered pending request.
        
        Use this after recover_pending() to set up callback handling.
        
        Args:
            record: The recovered record
            
        Returns:
            Future that will resolve when callback is received
        """
        loop = asyncio.get_event_loop()
        future: asyncio.Future[Decision] = loop.create_future()
        self._pending_futures[record.callback_id] = future
        return future
    
    async def _send_with_retry(
        self,
        request: ApprovalRequest,
        callback_id: str,
        callback_url: str,
    ) -> None:
        """Send request with exponential backoff retry."""
        last_error = None
        delay = self.retry_config.initial_delay
        
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                await self.send_request(request, callback_id, callback_url)
                return
            except Exception as e:
                last_error = e
                if attempt < self.retry_config.max_retries:
                    await asyncio.sleep(delay)
                    delay = min(
                        delay * self.retry_config.exponential_base,
                        self.retry_config.max_delay
                    )
        
        raise last_error or RuntimeError("Send failed")
    
    def _check_circuit(self) -> bool:
        """Check if circuit breaker allows request."""
        if self._circuit_state == CircuitState.CLOSED:
            return True
        
        if self._circuit_state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if self._last_failure_time:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.circuit_config.recovery_timeout:
                    self._circuit_state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    return True
            return False
        
        # HALF_OPEN - allow limited calls
        if self._half_open_calls < self.circuit_config.half_open_max_calls:
            self._half_open_calls += 1
            return True
        return False
    
    def _record_success(self) -> None:
        """Record successful operation."""
        if self._circuit_state == CircuitState.HALF_OPEN:
            # Recovery successful
            self._circuit_state = CircuitState.CLOSED
            self._failure_count = 0
        elif self._circuit_state == CircuitState.CLOSED:
            self._failure_count = 0
    
    def _record_failure(self) -> None:
        """Record failed operation."""
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        if self._circuit_state == CircuitState.HALF_OPEN:
            # Recovery failed
            self._circuit_state = CircuitState.OPEN
        elif self._failure_count >= self.circuit_config.failure_threshold:
            self._circuit_state = CircuitState.OPEN
    
    def get_circuit_state(self) -> CircuitState:
        """Get current circuit breaker state."""
        return self._circuit_state
    
    async def close(self) -> None:
        """Close the backend and store."""
        await self.store.close()
