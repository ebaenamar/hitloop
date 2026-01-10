"""Redis approval store for production deployments.

Follows LangGraph's pattern for Redis persistence.
Requires: pip install redis
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from typing import Any

from hitloop.persistence.base import ApprovalStore, PendingApprovalRecord


class RedisApprovalStore(ApprovalStore):
    """Redis-backed approval store.
    
    Production-ready persistence for pending approval requests.
    Fast, survives restarts, supports multi-process deployments.
    
    Advantages over PostgreSQL:
    - Faster for simple key-value operations
    - Built-in TTL support for automatic expiration
    - Better for high-throughput scenarios
    
    Example:
        >>> store = RedisApprovalStore.from_url("redis://localhost:6379/0")
        >>> async with store:
        ...     await store.put(record)
        ...     record = await store.get("callback_id")
    """
    
    KEY_PREFIX = "hitloop:pending:"
    INDEX_KEY = "hitloop:pending:index"
    THREAD_INDEX_PREFIX = "hitloop:pending:thread:"
    
    def __init__(self, client: Any) -> None:
        """Initialize with Redis client.
        
        Use from_url() for easier setup.
        
        Args:
            client: redis.asyncio.Redis client
        """
        self._client = client
    
    @classmethod
    async def from_url(cls, url: str) -> RedisApprovalStore:
        """Create store from Redis URL.
        
        Args:
            url: Redis URL (e.g., "redis://localhost:6379/0")
            
        Returns:
            Configured RedisApprovalStore
            
        Example:
            >>> store = await RedisApprovalStore.from_url(
            ...     "redis://localhost:6379/0"
            ... )
        """
        try:
            import redis.asyncio as redis
        except ImportError:
            raise ImportError(
                "redis is required for RedisApprovalStore. "
                "Install with: pip install redis"
            )
        
        client = redis.from_url(url)
        return cls(client)
    
    async def setup(self) -> None:
        """No-op for Redis - no schema to create."""
        # Ping to verify connection
        await self._client.ping()
    
    async def put(self, record: PendingApprovalRecord) -> None:
        """Store a pending approval."""
        key = f"{self.KEY_PREFIX}{record.callback_id}"
        data = json.dumps(record.to_dict())
        
        # Calculate TTL if expiration is set
        ttl = None
        if record.expires_at:
            ttl_seconds = (record.expires_at - datetime.now(timezone.utc)).total_seconds()
            ttl = max(1, int(ttl_seconds))  # At least 1 second
        
        pipe = self._client.pipeline()
        
        # Store the record
        if ttl:
            pipe.setex(key, ttl, data)
        else:
            pipe.set(key, data)
        
        # Add to global index
        pipe.sadd(self.INDEX_KEY, record.callback_id)
        
        # Add to thread index
        thread_key = f"{self.THREAD_INDEX_PREFIX}{record.thread_id}"
        pipe.sadd(thread_key, record.callback_id)
        
        await pipe.execute()
    
    async def get(self, callback_id: str) -> PendingApprovalRecord | None:
        """Retrieve a pending approval."""
        key = f"{self.KEY_PREFIX}{callback_id}"
        data = await self._client.get(key)
        
        if data is None:
            # Clean up indexes if key doesn't exist
            await self._client.srem(self.INDEX_KEY, callback_id)
            return None
        
        return PendingApprovalRecord.from_dict(json.loads(data))
    
    async def delete(self, callback_id: str) -> bool:
        """Delete a pending approval."""
        key = f"{self.KEY_PREFIX}{callback_id}"
        
        # Get record first to find thread_id
        data = await self._client.get(key)
        if data is None:
            return False
        
        record = PendingApprovalRecord.from_dict(json.loads(data))
        
        pipe = self._client.pipeline()
        pipe.delete(key)
        pipe.srem(self.INDEX_KEY, callback_id)
        pipe.srem(f"{self.THREAD_INDEX_PREFIX}{record.thread_id}", callback_id)
        results = await pipe.execute()
        
        return results[0] > 0
    
    async def list_pending(
        self,
        thread_id: str | None = None,
        limit: int = 100,
    ) -> list[PendingApprovalRecord]:
        """List pending approvals."""
        if thread_id:
            callback_ids = await self._client.smembers(
                f"{self.THREAD_INDEX_PREFIX}{thread_id}"
            )
        else:
            callback_ids = await self._client.smembers(self.INDEX_KEY)
        
        if not callback_ids:
            return []
        
        # Limit the number of IDs to fetch
        callback_ids = list(callback_ids)[:limit]
        
        # Fetch all records
        keys = [f"{self.KEY_PREFIX}{cid.decode() if isinstance(cid, bytes) else cid}" 
                for cid in callback_ids]
        values = await self._client.mget(keys)
        
        records = []
        for data in values:
            if data:
                records.append(PendingApprovalRecord.from_dict(json.loads(data)))
        
        # Sort by created_at descending
        records.sort(key=lambda r: r.created_at, reverse=True)
        
        return records
    
    async def cleanup_expired(self) -> int:
        """Remove expired records.
        
        Note: Redis TTL handles most expiration automatically.
        This cleans up index entries for expired keys.
        """
        callback_ids = await self._client.smembers(self.INDEX_KEY)
        if not callback_ids:
            return 0
        
        expired_count = 0
        for cid in callback_ids:
            cid_str = cid.decode() if isinstance(cid, bytes) else cid
            key = f"{self.KEY_PREFIX}{cid_str}"
            
            # Check if key still exists
            if not await self._client.exists(key):
                await self._client.srem(self.INDEX_KEY, cid_str)
                expired_count += 1
        
        return expired_count
    
    async def close(self) -> None:
        """Close the Redis connection."""
        await self._client.close()
