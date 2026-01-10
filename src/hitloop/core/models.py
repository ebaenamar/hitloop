"""Core data models for HITL Lab.

This module defines the fundamental data structures used throughout the library
for representing actions, approval requests, decisions, and trace events.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class RiskClass(str, Enum):
    """Risk classification levels for actions."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EventType(str, Enum):
    """Types of trace events that can be logged."""

    RUN_START = "run_start"
    RUN_END = "run_end"
    LLM_CALL = "llm_call"
    ACTION_PROPOSED = "action_proposed"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_DECIDED = "approval_decided"
    TOOL_EXECUTION_START = "tool_execution_start"
    TOOL_EXECUTION_END = "tool_execution_end"
    ERROR = "error"
    INJECTED_ERROR = "injected_error"


class Action(BaseModel):
    """Represents a proposed action by an AI agent.

    An Action captures everything needed to:
    1. Execute a tool call
    2. Assess its risk and impact
    3. Present it to a human for approval

    Attributes:
        id: Unique identifier for this action
        tool_name: Name of the tool to be called
        tool_args: Arguments to pass to the tool
        risk_class: Risk classification (low/medium/high)
        side_effects: List of potential side effects
        rationale: Explanation of why this action is proposed
        context_refs: References to relevant context used in decision
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str
    tool_args: dict[str, Any] = Field(default_factory=dict)
    risk_class: RiskClass = RiskClass.LOW
    side_effects: list[str] = Field(default_factory=list)
    rationale: str = ""
    context_refs: list[str] = Field(default_factory=list)

    def args_hash(self) -> str:
        """Generate a stable hash of the tool arguments."""
        args_json = json.dumps(self.tool_args, sort_keys=True, default=str)
        return hashlib.sha256(args_json.encode()).hexdigest()[:16]

    def summary(self, max_args_length: int = 100) -> str:
        """Generate a human-readable summary of the action."""
        args_str = json.dumps(self.tool_args, default=str)
        if len(args_str) > max_args_length:
            args_str = args_str[: max_args_length - 3] + "..."
        return f"[{self.risk_class.value.upper()}] {self.tool_name}({args_str})"


class ApprovalRequest(BaseModel):
    """A request for human approval of an action.

    This captures the context needed for a human to make an informed decision
    about whether to approve or reject an action.

    Attributes:
        run_id: ID of the current run/session
        action: The action requiring approval
        summary_context: Human-readable context for the decision
        policy_name: Name of the policy that triggered this request
        policy_reason: Explanation of why approval was requested
    """

    run_id: str
    action: Action
    summary_context: str = ""
    policy_name: str = ""
    policy_reason: str = ""

    def format_for_display(self) -> str:
        """Format the request for CLI display."""
        lines = [
            "=" * 60,
            "APPROVAL REQUEST",
            "=" * 60,
            f"Run ID: {self.run_id}",
            f"Action ID: {self.action.id}",
            "",
            f"Tool: {self.action.tool_name}",
            f"Risk: {self.action.risk_class.value.upper()}",
            "",
            "Arguments:",
            json.dumps(self.action.tool_args, indent=2, default=str),
            "",
        ]

        if self.action.side_effects:
            lines.extend(
                [
                    "Side Effects:",
                    *[f"  - {effect}" for effect in self.action.side_effects],
                    "",
                ]
            )

        if self.action.rationale:
            lines.extend(["Rationale:", f"  {self.action.rationale}", ""])

        if self.summary_context:
            lines.extend(["Context:", f"  {self.summary_context}", ""])

        if self.policy_reason:
            lines.extend(
                [f"Policy ({self.policy_name}):", f"  {self.policy_reason}", ""]
            )

        lines.append("=" * 60)
        return "\n".join(lines)


class Decision(BaseModel):
    """The result of an approval decision.

    Attributes:
        action_id: ID of the action this decision is for
        approved: Whether the action was approved
        reason: Explanation for the decision
        tags: Optional tags for categorization/filtering
        decided_by: Identifier of who/what made the decision
        latency_ms: Time taken to make the decision in milliseconds
    """

    action_id: str
    approved: bool
    reason: str = ""
    tags: list[str] = Field(default_factory=list)
    decided_by: str = "unknown"
    latency_ms: float = 0.0

    @field_validator("latency_ms")
    @classmethod
    def latency_must_be_non_negative(cls, v: float) -> float:
        """Ensure latency is non-negative."""
        if v < 0:
            raise ValueError("latency_ms must be non-negative")
        return v


class TraceEvent(BaseModel):
    """A single event in the execution trace.

    TraceEvents are the fundamental unit of telemetry in HITL Lab.
    They capture everything needed to reconstruct and analyze agent behavior.

    Attributes:
        timestamp: When the event occurred (UTC)
        run_id: ID of the current run/session
        event_type: Type of event
        payload: Event-specific data
    """

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    run_id: str
    event_type: EventType
    payload: dict[str, Any] = Field(default_factory=dict)

    def model_dump_json_safe(self) -> dict[str, Any]:
        """Dump the model to a JSON-serializable dict."""
        data = self.model_dump()
        data["timestamp"] = self.timestamp.isoformat()
        data["event_type"] = self.event_type.value
        return data


class ToolResult(BaseModel):
    """Result of executing a tool.

    Attributes:
        action_id: ID of the action that was executed
        success: Whether execution succeeded
        result: The result data (if successful)
        error: Error information (if failed)
        error_class: Classification of error type
        retry_count: Number of retries attempted
        started_at: When execution started
        finished_at: When execution finished
    """

    action_id: str
    success: bool
    result: Any = None
    error: str | None = None
    error_class: str | None = None
    retry_count: int = 0
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None

    def execution_time_ms(self) -> float | None:
        """Calculate execution time in milliseconds."""
        if self.finished_at is None:
            return None
        delta = self.finished_at - self.started_at
        return delta.total_seconds() * 1000


class RunMetadata(BaseModel):
    """Metadata for a complete run/session.

    Attributes:
        run_id: Unique identifier for this run
        scenario_id: ID of the scenario being run
        condition_id: ID of the experimental condition
        agent_version: Version of the agent
        model: Model name/identifier
        seed: Random seed for reproducibility
        started_at: When the run started
        finished_at: When the run finished
        task_success: Whether the task succeeded
        validation_details: Details about validation results
    """

    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    scenario_id: str = ""
    condition_id: str = ""
    agent_version: str = ""
    model: str = ""
    seed: int | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    task_success: bool | None = None
    validation_details: dict[str, Any] = Field(default_factory=dict)

    def duration_ms(self) -> float | None:
        """Calculate run duration in milliseconds."""
        if self.finished_at is None:
            return None
        delta = self.finished_at - self.started_at
        return delta.total_seconds() * 1000
