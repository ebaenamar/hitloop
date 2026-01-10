"""Persistence layer for hitloop.

This module provides storage backends for:
1. Pending approval requests (survives restarts)
2. Telemetry/traces (production-ready databases)

Follows LangGraph's pattern of pluggable storage backends.
"""

from hitloop.persistence.base import (
    ApprovalStore,
    PendingApprovalRecord,
)
from hitloop.persistence.memory import InMemoryApprovalStore
from hitloop.persistence.postgres import PostgresApprovalStore
from hitloop.persistence.redis import RedisApprovalStore

__all__ = [
    "ApprovalStore",
    "PendingApprovalRecord",
    "InMemoryApprovalStore",
    "PostgresApprovalStore",
    "RedisApprovalStore",
]
