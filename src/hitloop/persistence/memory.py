"""In-memory approval store for development and testing.

This store does NOT survive restarts - use PostgresApprovalStore or
RedisApprovalStore for production.
"""

from __future__ import annotations

from datetime import datetime, timezone

from hitloop.persistence.base import ApprovalStore, PendingApprovalRecord


class InMemoryApprovalStore(ApprovalStore):
    """In-memory approval store.
    
    Useful for:
    - Development and testing
    - Single-process deployments where persistence isn't critical
    
    NOT suitable for:
    - Production deployments that need to survive restarts
    - Multi-process deployments
    
    Example:
        >>> store = InMemoryApprovalStore()
        >>> await store.setup()
        >>> await store.put(record)
    """
    
    def __init__(self) -> None:
        """Initialize in-memory store."""
        self._records: dict[str, PendingApprovalRecord] = {}
    
    async def setup(self) -> None:
        """No-op for in-memory store."""
        pass
    
    async def put(self, record: PendingApprovalRecord) -> None:
        """Store a pending approval."""
        self._records[record.callback_id] = record
    
    async def get(self, callback_id: str) -> PendingApprovalRecord | None:
        """Retrieve a pending approval."""
        record = self._records.get(callback_id)
        if record and record.is_expired():
            del self._records[callback_id]
            return None
        return record
    
    async def delete(self, callback_id: str) -> bool:
        """Delete a pending approval."""
        if callback_id in self._records:
            del self._records[callback_id]
            return True
        return False
    
    async def list_pending(
        self,
        thread_id: str | None = None,
        limit: int = 100,
    ) -> list[PendingApprovalRecord]:
        """List pending approvals."""
        now = datetime.now(timezone.utc)
        results = []
        
        for record in self._records.values():
            if record.is_expired():
                continue
            if thread_id and record.thread_id != thread_id:
                continue
            results.append(record)
            if len(results) >= limit:
                break
        
        return results
    
    async def cleanup_expired(self) -> int:
        """Remove expired records."""
        expired = [
            cid for cid, record in self._records.items()
            if record.is_expired()
        ]
        for cid in expired:
            del self._records[cid]
        return len(expired)
