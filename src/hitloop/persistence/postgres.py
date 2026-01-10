"""PostgreSQL approval store for production deployments.

Follows LangGraph's pattern for PostgreSQL persistence.
Requires: pip install asyncpg
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from hitloop.persistence.base import ApprovalStore, PendingApprovalRecord


class PostgresApprovalStore(ApprovalStore):
    """PostgreSQL-backed approval store.
    
    Production-ready persistence for pending approval requests.
    Survives restarts and supports multi-process deployments.
    
    Example:
        >>> store = PostgresApprovalStore.from_conn_string(
        ...     "postgresql://user:pass@localhost:5432/hitloop"
        ... )
        >>> async with store:
        ...     await store.put(record)
        ...     record = await store.get("callback_id")
    """
    
    def __init__(self, pool: Any) -> None:
        """Initialize with connection pool.
        
        Use from_conn_string() for easier setup.
        
        Args:
            pool: asyncpg connection pool
        """
        self._pool = pool
    
    @classmethod
    async def from_conn_string(cls, conn_string: str) -> PostgresApprovalStore:
        """Create store from connection string.
        
        Args:
            conn_string: PostgreSQL connection string
            
        Returns:
            Configured PostgresApprovalStore
            
        Example:
            >>> store = await PostgresApprovalStore.from_conn_string(
            ...     "postgresql://localhost:5432/hitloop"
            ... )
        """
        try:
            import asyncpg
        except ImportError:
            raise ImportError(
                "asyncpg is required for PostgresApprovalStore. "
                "Install with: pip install asyncpg"
            )
        
        pool = await asyncpg.create_pool(conn_string)
        return cls(pool)
    
    async def setup(self) -> None:
        """Create tables if they don't exist."""
        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS hitloop_pending_approvals (
                    callback_id VARCHAR(64) PRIMARY KEY,
                    run_id VARCHAR(64) NOT NULL,
                    thread_id VARCHAR(64) NOT NULL,
                    action_id VARCHAR(64) NOT NULL,
                    tool_name VARCHAR(128) NOT NULL,
                    tool_args JSONB NOT NULL,
                    risk_class VARCHAR(32) NOT NULL,
                    policy_name VARCHAR(64) NOT NULL,
                    policy_reason TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    expires_at TIMESTAMPTZ,
                    metadata JSONB DEFAULT '{}'::jsonb
                );
                
                CREATE INDEX IF NOT EXISTS idx_pending_thread_id 
                ON hitloop_pending_approvals(thread_id);
                
                CREATE INDEX IF NOT EXISTS idx_pending_expires_at 
                ON hitloop_pending_approvals(expires_at) 
                WHERE expires_at IS NOT NULL;
            """)
    
    async def put(self, record: PendingApprovalRecord) -> None:
        """Store a pending approval."""
        async with self._pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO hitloop_pending_approvals 
                (callback_id, run_id, thread_id, action_id, tool_name, 
                 tool_args, risk_class, policy_name, policy_reason, 
                 created_at, expires_at, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (callback_id) DO UPDATE SET
                    tool_args = EXCLUDED.tool_args,
                    expires_at = EXCLUDED.expires_at,
                    metadata = EXCLUDED.metadata
            """,
                record.callback_id,
                record.run_id,
                record.thread_id,
                record.action_id,
                record.tool_name,
                json.dumps(record.tool_args),
                record.risk_class,
                record.policy_name,
                record.policy_reason,
                record.created_at,
                record.expires_at,
                json.dumps(record.metadata),
            )
    
    async def get(self, callback_id: str) -> PendingApprovalRecord | None:
        """Retrieve a pending approval."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM hitloop_pending_approvals 
                WHERE callback_id = $1
                AND (expires_at IS NULL OR expires_at > NOW())
            """, callback_id)
            
            if row is None:
                return None
            
            return self._row_to_record(row)
    
    async def delete(self, callback_id: str) -> bool:
        """Delete a pending approval."""
        async with self._pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM hitloop_pending_approvals 
                WHERE callback_id = $1
            """, callback_id)
            return result == "DELETE 1"
    
    async def list_pending(
        self,
        thread_id: str | None = None,
        limit: int = 100,
    ) -> list[PendingApprovalRecord]:
        """List pending approvals."""
        async with self._pool.acquire() as conn:
            if thread_id:
                rows = await conn.fetch("""
                    SELECT * FROM hitloop_pending_approvals 
                    WHERE thread_id = $1
                    AND (expires_at IS NULL OR expires_at > NOW())
                    ORDER BY created_at DESC
                    LIMIT $2
                """, thread_id, limit)
            else:
                rows = await conn.fetch("""
                    SELECT * FROM hitloop_pending_approvals 
                    WHERE expires_at IS NULL OR expires_at > NOW()
                    ORDER BY created_at DESC
                    LIMIT $1
                """, limit)
            
            return [self._row_to_record(row) for row in rows]
    
    async def cleanup_expired(self) -> int:
        """Remove expired records."""
        async with self._pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM hitloop_pending_approvals 
                WHERE expires_at IS NOT NULL AND expires_at <= NOW()
            """)
            # Parse "DELETE N" to get count
            return int(result.split()[-1]) if result else 0
    
    async def close(self) -> None:
        """Close the connection pool."""
        await self._pool.close()
    
    def _row_to_record(self, row: Any) -> PendingApprovalRecord:
        """Convert database row to record."""
        return PendingApprovalRecord(
            callback_id=row["callback_id"],
            run_id=row["run_id"],
            thread_id=row["thread_id"],
            action_id=row["action_id"],
            tool_name=row["tool_name"],
            tool_args=json.loads(row["tool_args"]) if isinstance(row["tool_args"], str) else row["tool_args"],
            risk_class=row["risk_class"],
            policy_name=row["policy_name"],
            policy_reason=row["policy_reason"],
            created_at=row["created_at"].replace(tzinfo=timezone.utc) if row["created_at"].tzinfo is None else row["created_at"],
            expires_at=row["expires_at"].replace(tzinfo=timezone.utc) if row["expires_at"] and row["expires_at"].tzinfo is None else row["expires_at"],
            metadata=json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"],
        )
