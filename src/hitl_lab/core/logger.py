"""Telemetry logging for HITL Lab.

This module provides structured event logging with SQLite persistence,
enabling comprehensive experiment analysis and debugging.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import JSON, Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from hitl_lab.core.models import (
    Action,
    ApprovalRequest,
    Decision,
    EventType,
    RunMetadata,
    ToolResult,
    TraceEvent,
)

Base = declarative_base()


class TraceEventRecord(Base):  # type: ignore[valid-type,misc]
    """SQLAlchemy model for trace events."""

    __tablename__ = "trace_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    run_id = Column(String(64), nullable=False, index=True)
    event_type = Column(String(64), nullable=False, index=True)
    payload = Column(JSON, nullable=False)


class RunRecord(Base):  # type: ignore[valid-type,misc]
    """SQLAlchemy model for run metadata."""

    __tablename__ = "runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64), nullable=False, unique=True, index=True)
    scenario_id = Column(String(128), nullable=True)
    condition_id = Column(String(128), nullable=True)
    agent_version = Column(String(64), nullable=True)
    model = Column(String(128), nullable=True)
    seed = Column(Integer, nullable=True)
    started_at = Column(DateTime, nullable=False)
    finished_at = Column(DateTime, nullable=True)
    task_success = Column(Integer, nullable=True)  # Boolean as int
    validation_details = Column(Text, nullable=True)


class TelemetryLogger:
    """Structured event logger with SQLite persistence.

    The TelemetryLogger is the central instrumentation component of HITL Lab.
    It captures all significant events in the agent workflow, enabling:
    - Experiment replay and analysis
    - Debugging agent behavior
    - Computing metrics for research papers

    Usage:
        >>> logger = TelemetryLogger("experiments.db")
        >>> logger.log_run_start(run_metadata)
        >>> logger.log_action_proposed(run_id, action)
        >>> # ... more events ...
        >>> logger.log_run_end(run_id, success=True)

    Events are persisted immediately to SQLite for durability.
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        """Initialize the telemetry logger.

        Args:
            db_path: Path to SQLite database file, or ":memory:" for in-memory
        """
        self.db_path = str(db_path)
        self._engine: Engine | None = None
        self._session_factory: sessionmaker[Session] | None = None
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Initialize database connection and create tables."""
        if self.db_path == ":memory:":
            self._engine = create_engine("sqlite:///:memory:", echo=False)
        else:
            self._engine = create_engine(f"sqlite:///{self.db_path}", echo=False)

        Base.metadata.create_all(self._engine)
        self._session_factory = sessionmaker(bind=self._engine)

    def _get_session(self) -> Session:
        """Get a new database session."""
        if self._session_factory is None:
            raise RuntimeError("Logger not initialized")
        return self._session_factory()

    def _log_event(self, event: TraceEvent) -> None:
        """Persist a trace event to the database."""
        with self._get_session() as session:
            record = TraceEventRecord(
                timestamp=event.timestamp,
                run_id=event.run_id,
                event_type=event.event_type.value,
                payload=event.payload,
            )
            session.add(record)
            session.commit()

    def log_run_start(self, metadata: RunMetadata) -> None:
        """Log the start of a run.

        Args:
            metadata: Run metadata including scenario, condition, seed, etc.
        """
        with self._get_session() as session:
            run_record = RunRecord(
                run_id=metadata.run_id,
                scenario_id=metadata.scenario_id,
                condition_id=metadata.condition_id,
                agent_version=metadata.agent_version,
                model=metadata.model,
                seed=metadata.seed,
                started_at=metadata.started_at,
            )
            session.add(run_record)
            session.commit()

        event = TraceEvent(
            run_id=metadata.run_id,
            event_type=EventType.RUN_START,
            payload={
                "scenario_id": metadata.scenario_id,
                "condition_id": metadata.condition_id,
                "agent_version": metadata.agent_version,
                "model": metadata.model,
                "seed": metadata.seed,
            },
        )
        self._log_event(event)

    def log_run_end(
        self,
        run_id: str,
        success: bool,
        validation_details: dict[str, Any] | None = None,
    ) -> None:
        """Log the end of a run.

        Args:
            run_id: ID of the run
            success: Whether the task succeeded
            validation_details: Optional validation result details
        """
        finished_at = datetime.now(timezone.utc)

        with self._get_session() as session:
            run_record = session.query(RunRecord).filter_by(run_id=run_id).first()
            if run_record:
                run_record.finished_at = finished_at
                run_record.task_success = 1 if success else 0
                if validation_details:
                    run_record.validation_details = json.dumps(validation_details)
                session.commit()

        event = TraceEvent(
            run_id=run_id,
            event_type=EventType.RUN_END,
            payload={
                "task_success": success,
                "validation_details": validation_details or {},
            },
        )
        self._log_event(event)

    def log_llm_call(
        self,
        run_id: str,
        prompt_hash: str | None = None,
        template_id: str | None = None,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        latency_ms: float | None = None,
        model: str | None = None,
    ) -> None:
        """Log an LLM call.

        Args:
            run_id: ID of the current run
            prompt_hash: Hash of the prompt for deduplication
            template_id: ID of the prompt template used
            tokens_in: Input token count
            tokens_out: Output token count
            latency_ms: Call latency in milliseconds
            model: Model name/identifier
        """
        event = TraceEvent(
            run_id=run_id,
            event_type=EventType.LLM_CALL,
            payload={
                "prompt_hash": prompt_hash,
                "template_id": template_id,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "latency_ms": latency_ms,
                "model": model,
            },
        )
        self._log_event(event)

    def log_action_proposed(
        self, run_id: str, action: Action, injected_error: bool = False
    ) -> None:
        """Log a proposed action.

        Args:
            run_id: ID of the current run
            action: The proposed action
            injected_error: Whether this action contains an injected error
        """
        event = TraceEvent(
            run_id=run_id,
            event_type=EventType.ACTION_PROPOSED,
            payload={
                "action_id": action.id,
                "tool_name": action.tool_name,
                "args_hash": action.args_hash(),
                "risk_class": action.risk_class.value,
                "side_effects": action.side_effects,
                "injected_error": injected_error,
            },
        )
        self._log_event(event)

        if injected_error:
            error_event = TraceEvent(
                run_id=run_id,
                event_type=EventType.INJECTED_ERROR,
                payload={
                    "action_id": action.id,
                    "tool_name": action.tool_name,
                },
            )
            self._log_event(error_event)

    def log_approval_requested(
        self, run_id: str, request: ApprovalRequest, channel: str = "unknown"
    ) -> None:
        """Log an approval request.

        Args:
            run_id: ID of the current run
            request: The approval request
            channel: Approval channel (e.g., "cli", "web", "slack")
        """
        event = TraceEvent(
            run_id=run_id,
            event_type=EventType.APPROVAL_REQUESTED,
            payload={
                "action_id": request.action.id,
                "channel": channel,
                "policy_name": request.policy_name,
                "policy_reason": request.policy_reason,
                "requested_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        self._log_event(event)

    def log_approval_decided(
        self, run_id: str, decision: Decision, channel: str = "unknown"
    ) -> None:
        """Log an approval decision.

        Args:
            run_id: ID of the current run
            decision: The approval decision
            channel: Approval channel
        """
        event = TraceEvent(
            run_id=run_id,
            event_type=EventType.APPROVAL_DECIDED,
            payload={
                "action_id": decision.action_id,
                "approved": decision.approved,
                "reason": decision.reason,
                "tags": decision.tags,
                "decided_by": decision.decided_by,
                "latency_ms": decision.latency_ms,
                "channel": channel,
                "decided_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        self._log_event(event)

    def log_tool_execution_start(self, run_id: str, action: Action) -> None:
        """Log the start of tool execution.

        Args:
            run_id: ID of the current run
            action: The action being executed
        """
        event = TraceEvent(
            run_id=run_id,
            event_type=EventType.TOOL_EXECUTION_START,
            payload={
                "action_id": action.id,
                "tool_name": action.tool_name,
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        self._log_event(event)

    def log_tool_execution_end(self, run_id: str, result: ToolResult) -> None:
        """Log the end of tool execution.

        Args:
            run_id: ID of the current run
            result: The tool execution result
        """
        event = TraceEvent(
            run_id=run_id,
            event_type=EventType.TOOL_EXECUTION_END,
            payload={
                "action_id": result.action_id,
                "success": result.success,
                "error": result.error,
                "error_class": result.error_class,
                "retry_count": result.retry_count,
                "execution_time_ms": result.execution_time_ms(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        self._log_event(event)

    def log_error(
        self, run_id: str, error: Exception, context: dict[str, Any] | None = None
    ) -> None:
        """Log an error event.

        Args:
            run_id: ID of the current run
            error: The exception that occurred
            context: Optional context about where the error occurred
        """
        event = TraceEvent(
            run_id=run_id,
            event_type=EventType.ERROR,
            payload={
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context or {},
            },
        )
        self._log_event(event)

    def get_events(
        self,
        run_id: str | None = None,
        event_type: EventType | None = None,
        limit: int | None = None,
    ) -> list[TraceEvent]:
        """Query trace events.

        Args:
            run_id: Filter by run ID
            event_type: Filter by event type
            limit: Maximum number of events to return

        Returns:
            List of matching trace events
        """
        with self._get_session() as session:
            query = session.query(TraceEventRecord)

            if run_id:
                query = query.filter_by(run_id=run_id)
            if event_type:
                query = query.filter_by(event_type=event_type.value)

            query = query.order_by(TraceEventRecord.timestamp)

            if limit:
                query = query.limit(limit)

            records = query.all()

            events = []
            for record in records:
                event = TraceEvent(
                    timestamp=record.timestamp,
                    run_id=record.run_id,
                    event_type=EventType(record.event_type),
                    payload=record.payload,
                )
                events.append(event)

            return events

    def get_run_metadata(self, run_id: str) -> RunMetadata | None:
        """Get metadata for a specific run.

        Args:
            run_id: ID of the run

        Returns:
            RunMetadata if found, None otherwise
        """
        with self._get_session() as session:
            record = session.query(RunRecord).filter_by(run_id=run_id).first()
            if not record:
                return None

            validation_details = {}
            if record.validation_details:
                validation_details = json.loads(record.validation_details)

            return RunMetadata(
                run_id=record.run_id,
                scenario_id=record.scenario_id or "",
                condition_id=record.condition_id or "",
                agent_version=record.agent_version or "",
                model=record.model or "",
                seed=record.seed,
                started_at=record.started_at,
                finished_at=record.finished_at,
                task_success=bool(record.task_success) if record.task_success is not None else None,
                validation_details=validation_details,
            )

    def get_all_runs(self) -> list[RunMetadata]:
        """Get metadata for all runs.

        Returns:
            List of RunMetadata for all runs in the database
        """
        with self._get_session() as session:
            records = session.query(RunRecord).order_by(RunRecord.started_at).all()

            runs = []
            for record in records:
                validation_details = {}
                if record.validation_details:
                    validation_details = json.loads(record.validation_details)

                runs.append(
                    RunMetadata(
                        run_id=record.run_id,
                        scenario_id=record.scenario_id or "",
                        condition_id=record.condition_id or "",
                        agent_version=record.agent_version or "",
                        model=record.model or "",
                        seed=record.seed,
                        started_at=record.started_at,
                        finished_at=record.finished_at,
                        task_success=bool(record.task_success)
                        if record.task_success is not None
                        else None,
                        validation_details=validation_details,
                    )
                )

            return runs

    def close(self) -> None:
        """Close the database connection."""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            self._session_factory = None
