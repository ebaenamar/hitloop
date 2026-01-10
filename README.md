# hitloop

**Human-in-the-Loop control library for AI agent workflows with LangGraph integration.**

hitloop provides explicit control nodes for human oversight in AI agent workflows, with strong instrumentation for research experiments. Unlike passive monitoring, human approval is a first-class control signal and event in the execution trace.

## Core Concept

```
LLM proposes action → HITL policy decides → Human approves/rejects → Tool executes → Telemetry logs all
```

Human approval is not a UI gimmick. It is:
- A **control signal** that gates execution
- A **first-class event** in the trace
- A **research artifact** for measuring oversight effectiveness

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ebaenamar/hitloop.git
cd hitloop

# Install with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .

# For development
pip install -e ".[dev]"
```

### Run an Example

```bash
# Basic workflow (with CLI approval prompts)
python examples/basic_workflow.py

# Auto-approve mode (no prompts)
python examples/basic_workflow.py --auto

# Run a full experiment
python examples/run_experiment.py --n-trials 20
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      LangGraph Workflow                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────────┐  │
│  │   LLM    │───►│  HITL    │───►│   Tool Executor      │  │
│  │  Node    │    │  Gate    │    │                      │  │
│  └──────────┘    └────┬─────┘    └──────────────────────┘  │
│                       │                                     │
│                       ▼                                     │
│              ┌────────────────┐                             │
│              │  HITL Policy   │                             │
│              │  ┌──────────┐  │                             │
│              │  │ Approval │  │◄──► Human (CLI/Web/etc)    │
│              │  │ Backend  │  │                             │
│              │  └──────────┘  │                             │
│              └────────────────┘                             │
│                       │                                     │
│                       ▼                                     │
│              ┌────────────────┐                             │
│              │   Telemetry    │───► SQLite / Analysis      │
│              │    Logger      │                             │
│              └────────────────┘                             │
└─────────────────────────────────────────────────────────────┘
```

## API Overview

### Core Models

```python
from hitloop import Action, Decision, RiskClass

# Define an action
action = Action(
    tool_name="send_email",
    tool_args={"recipient": "alice@example.com", "subject": "Hello"},
    risk_class=RiskClass.MEDIUM,
    side_effects=["email_sent"],
    rationale="Sending follow-up email to client",
)

# Decisions from human review
decision = Decision(
    action_id=action.id,
    approved=True,
    reason="Verified recipient is correct",
    decided_by="human:operator",
    latency_ms=1500.0,
)
```

### Policies

Three built-in policies for different oversight tiers:

```python
from hitloop import AlwaysApprovePolicy, RiskBasedPolicy, AuditPlusEscalatePolicy

# Tier 4: No human oversight (baseline)
policy = AlwaysApprovePolicy()

# Risk-based: Approve high-risk actions only
policy = RiskBasedPolicy(
    require_approval_for_high=True,
    require_approval_for_medium=False,
    high_risk_tools=["send_email", "delete_record"],
)

# Audit + Escalate: Random sampling + anomaly detection
policy = AuditPlusEscalatePolicy(
    audit_sample_rate=0.1,  # 10% random audit
    escalate_on_high_risk=True,
    anomaly_signals=["unusual_recipient", "large_amount"],
)
```

### Adding a New Policy

Create a single file in `src/hitloop/policies/`:

```python
# src/hitloop/policies/my_policy.py
from hitloop.core.interfaces import HITLPolicy
from hitloop.core.models import Action, Decision

class MyCustomPolicy(HITLPolicy):
    @property
    def name(self) -> str:
        return "my_custom"

    def should_request_approval(
        self, action: Action, state: dict
    ) -> tuple[bool, str]:
        # Your logic here
        if action.tool_name in self.critical_tools:
            return True, "Critical tool requires approval"
        return False, "Auto-approved"
```

### LangGraph Integration

Add human approval to any LangGraph agent:

```python
from hitloop import RiskBasedPolicy, ApprovalRequest, Action, Decision, RiskClass
from hitloop.backends import AutoApproveBackend, CLIBackend

# 1. Configure policy - which actions need approval?
policy = RiskBasedPolicy(
    require_approval_for_high=True,
    high_risk_tools=["send_email", "delete_file", "transfer_money"],
)

# 2. Choose backend - how to get approval?
backend = CLIBackend()  # Interactive CLI prompts
# backend = AutoApproveBackend()  # For testing

# 3. In your LangGraph node, check if approval is needed:
async def hitl_gate_node(state):
    action = state["pending_action"]
    needs_approval, reason = policy.should_request_approval(action, state)
    
    if needs_approval:
        request = ApprovalRequest(run_id="my-run", action=action)
        decision = await backend.request_approval(request)
        return {"approval_decision": decision}
    else:
        # Auto-approve low-risk actions
        return {"approval_decision": Decision(action_id=action.id, approved=True)}
```

**Full working example:** See `examples/langgraph_agent.py`

```bash
# Run with simulated LLM (no API key needed)
python examples/langgraph_agent.py --simulate --auto

# With CLI approval prompts
python examples/langgraph_agent.py --simulate
```

### Third-Party Integrations (Slack, Telegram, Discord, etc.)

hitloop is **completely agnostic** to your approval channel. Use `WebhookBackend` to integrate with any service:

```python
from hitloop.backends import WebhookBackend

# Your custom function to send to Slack/Telegram/Discord/etc.
async def send_to_slack(request, callback_id, callback_url):
    await slack_client.chat_postMessage(
        channel="#approvals",
        text=f"Approve {request.action.tool_name}?",
        # Include buttons that POST to callback_url
    )

backend = WebhookBackend(
    send_request=send_to_slack,
    timeout_seconds=300,  # 5 min timeout (rejects if no response)
    callback_base_url="https://your-app.com/hitloop/callback",
)

# In your webhook handler (FastAPI/Flask):
@app.post("/hitloop/callback/{callback_id}")
async def handle_callback(callback_id: str, data: dict):
    await backend.handle_callback(
        callback_id=callback_id,
        approved=data["approved"],
        decided_by=f"slack:{data['user']}",
    )
```

**Full working example:** See `examples/webhook_server.py`

```bash
# Install server dependencies
pip install hitloop[server]

# Run the webhook server
uvicorn examples.webhook_server:app --port 8000

# Test with curl
curl -X POST http://localhost:8000/test/request-approval
# Then approve/reject via the callback URL shown in console
```

## Production Deployment

### Persistent Storage (Survives Restarts)

hitloop follows LangGraph's pattern for pluggable storage backends:

```bash
# Install with your preferred backend
pip install hitloop[postgres]  # PostgreSQL
pip install hitloop[redis]     # Redis
```

```python
from hitloop.persistence import PostgresApprovalStore, RedisApprovalStore
from hitloop.backends import PersistentWebhookBackend

# PostgreSQL (recommended for most cases)
store = await PostgresApprovalStore.from_conn_string(
    "postgresql://user:pass@localhost:5432/hitloop"
)
await store.setup()  # Creates tables

# Or Redis (faster, built-in TTL)
store = await RedisApprovalStore.from_url("redis://localhost:6379/0")

# Use with PersistentWebhookBackend
backend = PersistentWebhookBackend(
    send_request=my_slack_sender,
    store=store,
    timeout_seconds=300,
)
```

### Retry & Circuit Breaker

Built-in resilience for production:

```python
from hitloop.backends import PersistentWebhookBackend, RetryConfig, CircuitBreakerConfig

backend = PersistentWebhookBackend(
    send_request=my_sender,
    store=store,
    retry_config=RetryConfig(
        max_retries=3,
        initial_delay=1.0,
        exponential_base=2.0,
    ),
    circuit_breaker_config=CircuitBreakerConfig(
        failure_threshold=5,    # Open after 5 failures
        recovery_timeout=30.0,  # Try again after 30s
    ),
)

# Check circuit state
print(backend.get_circuit_state())  # CLOSED, OPEN, or HALF_OPEN
```

### Recovery After Restart

```python
# On startup, recover pending requests
pending = await store.list_pending(thread_id="user-123")
for record in pending:
    # Re-send or resolve as needed
    future = backend.register_pending(record)
```

## Running Experiments

```python
from hitloop import TelemetryLogger
from hitloop.eval import ExperimentRunner, ExperimentCondition
from hitloop.eval.runner import create_standard_conditions
from hitloop.scenarios import EmailDraftScenario

# Setup
logger = TelemetryLogger("experiment.db")
scenario = EmailDraftScenario()

# Create standard conditions (4 policies × scenarios)
conditions = create_standard_conditions(
    scenario=scenario,
    n_trials=20,
    injection_rate=0.2,  # 20% error injection
)

# Run
runner = ExperimentRunner(logger)
for c in conditions:
    runner.add_condition(c)

await runner.run_all()

# Export
runner.export_results("results.csv", "summary.json")
```

### Output: results.csv

| run_id | scenario_id | condition_id | policy_name | task_success | approval_requested | injected_error | error_caught |
|--------|-------------|--------------|-------------|--------------|-------------------|----------------|--------------|
| abc123 | email_draft | risk_based   | risk_based  | 1            | 1                 | 0              | 0            |
| def456 | email_draft | risk_based   | risk_based  | 0            | 1                 | 1              | 1            |

### Output: summary.json

```json
{
  "risk_based": {
    "n_runs": 20,
    "success_rate": 0.85,
    "approval_rate": 0.65,
    "error_catch_rate": 0.75,
    "false_reject_rate": 0.05,
    "human_latency_mean_ms": 1200.5
  }
}
```

## Research Alignment

hitloop metrics map directly to the research framework:

| Metric | Research Concept | Description |
|--------|------------------|-------------|
| `success_rate` | Quality | Task completion rate |
| `approval_rate` | Leverage proxy | Human involvement frequency |
| `error_catch_rate` | Appropriate reliance | Injected error detection |
| `false_reject_rate` | Appropriate reliance | Unnecessary rejections |
| `human_latency_ms` | Human burden | Time cost per decision |
| `cost_proxy` | Efficiency | Token/call overhead |

### Injected Errors for Ground Truth

The error injector provides ground truth for measuring oversight effectiveness:

```python
from hitloop.eval import ErrorInjector, InjectionConfig

injector = ErrorInjector(InjectionConfig(
    injection_rate=0.2,
    injection_types=[
        InjectionType.WRONG_RECIPIENT,
        InjectionType.WRONG_RECORD_ID,
    ]
))

# Every action has known correctness
result = injector.maybe_inject(action)
if result.injected:
    # This action is KNOWN to be wrong
    # If human approves it → false negative
    # If human rejects it → true positive (error caught)
```

## Project Structure

```
hitloop/
├── src/hitloop/
│   ├── core/
│   │   ├── models.py      # Action, Decision, TraceEvent
│   │   ├── interfaces.py  # ApprovalBackend, HITLPolicy
│   │   └── logger.py      # TelemetryLogger (SQLite)
│   ├── policies/
│   │   ├── always_approve.py
│   │   ├── risk_based.py
│   │   └── audit_plus_escalate.py
│   ├── backends/
│   │   ├── cli_backend.py
│   │   └── humanlayer_backend.py  # Optional
│   ├── langgraph/
│   │   └── nodes.py       # hitl_gate_node, execute_tool_node
│   ├── scenarios/
│   │   ├── email_draft.py
│   │   └── record_update.py
│   └── eval/
│       ├── runner.py      # ExperimentRunner
│       ├── injectors.py   # ErrorInjector
│       └── metrics.py     # MetricsCalculator
├── tests/
├── examples/
├── pyproject.toml
└── README.md
```

## Instrumentation

Every run emits structured events:

```python
# Per run
{
    "run_id": "abc123",
    "scenario_id": "email_draft",
    "condition_id": "risk_based",
    "seed": 42
}

# Per action
{
    "action_id": "xyz789",
    "tool_name": "send_email",
    "args_hash": "a1b2c3d4",
    "risk_class": "medium",
    "injected_error": false
}

# Per approval
{
    "channel": "cli",
    "latency_ms": 1250.0,
    "decision": true,
    "decided_by": "human"
}

# Per execution
{
    "success": true,
    "execution_time_ms": 45.2
}
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy src/hitloop

# Linting
ruff check src/hitloop
ruff format src/hitloop
```

## License

MIT License - see LICENSE file.
