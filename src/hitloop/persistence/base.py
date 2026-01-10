"""Base interfaces for persistence layer.

Defines abstract interfaces that all storage backends must implement.
Follows LangGraph's pattern of pluggable, database-agnostic storage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class PendingApprovalRecord:
    """Record of a pending approval request.
    
    This is what gets persisted to survive restarts.
    """
    callback_id: str
    run_id: str
    thread_id: str  # LangGraph thread_id for correlation
    action_id: str
    tool_name: str
    tool_args: dict[str, Any]
    risk_class: str
    policy_name: str
    policy_reason: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if this request has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "callback_id": self.callback_id,
            "run_id": self.run_id,
            "thread_id": self.thread_id,
            "action_id": self.action_id,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "risk_class": self.risk_class,
            "policy_name": self.policy_name,
            "policy_reason": self.policy_reason,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PendingApprovalRecord:
        """Create from dictionary."""
        return cls(
            callback_id=data["callback_id"],
            run_id=data["run_id"],
            thread_id=data["thread_id"],
            action_id=data["action_id"],
            tool_name=data["tool_name"],
            tool_args=data["tool_args"],
            risk_class=data["risk_class"],
            policy_name=data["policy_name"],
            policy_reason=data["policy_reason"],
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            metadata=data.get("metadata", {}),
        )


class ApprovalStore(ABC):
    """Abstract interface for approval request persistence.
    
    Implementations store pending approval requests so they survive
    service restarts. This is critical for production deployments.
    
    Pattern follows LangGraph's BaseCheckpointSaver interface.
    
    Example:
        >>> store = PostgresApprovalStore.from_conn_string(DB_URI)
        >>> await store.setup()  # Create tables if needed
        >>> 
        >>> # Save pending request
        >>> await store.put(record)
        >>> 
        >>> # Retrieve by callback_id
        >>> record = await store.get("abc123")
        >>> 
        >>> # Delete after resolution
        >>> await store.delete("abc123")
    """
    
    @abstractmethod
    async def setup(self) -> None:
        """Initialize the store (create tables, indexes, etc.).
        
        Must be called before using the store. Idempotent.
        """
        pass
    
    @abstractmethod
    async def put(self, record: PendingApprovalRecord) -> None:
        """Store a pending approval request.
        
        Args:
            record: The pending approval to store
        """
        pass
    
    @abstractmethod
    async def get(self, callback_id: str) -> PendingApprovalRecord | None:
        """Retrieve a pending approval by callback_id.
        
        Args:
            callback_id: The callback identifier
            
        Returns:
            The record if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def delete(self, callback_id: str) -> bool:
        """Delete a pending approval.
        
        Args:
            callback_id: The callback identifier
            
        Returns:
            True if deleted, False if not found
        """
        pass
    
    @abstractmethod
    async def list_pending(
        self,
        thread_id: str | None = None,
        limit: int = 100,
    ) -> list[PendingApprovalRecord]:
        """List pending approvals.
        
        Args:
            thread_id: Optional filter by LangGraph thread_id
            limit: Maximum number of records to return
            
        Returns:
            List of pending approval records
        """
        pass
    
    @abstractmethod
    async def cleanup_expired(self) -> int:
        """Remove expired pending approvals.
        
        Returns:
            Number of records deleted
        """
        pass
    
    async def close(self) -> None:
        """Close any connections. Override if needed."""
        pass
    
    async def __aenter__(self) -> ApprovalStore:
        """Async context manager entry."""
        await self.setup()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
