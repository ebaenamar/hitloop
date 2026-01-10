#!/usr/bin/env python3
"""
Example: Webhook Server for hitloop Integration

This example shows how to create a webhook server that can integrate
hitloop with ANY third-party service (Slack, Telegram, Discord, etc.)

The pattern is simple:
1. hitloop sends approval requests to your outbound handler
2. You forward to your service (Slack, Telegram, etc.)
3. Human responds via your service
4. Your service calls the callback endpoint
5. hitloop continues the workflow

Usage:
    # Start the server
    uvicorn examples.webhook_server:app --reload --port 8000
    
    # In another terminal, run the test client
    python examples/webhook_client.py

This example includes:
- FastAPI server with callback endpoint
- Simulated "external service" (prints to console, waits for input)
- Full working example you can adapt for Slack/Telegram/etc.
"""

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

from hitloop import (
    Action,
    ApprovalRequest,
    Decision,
    RiskBasedPolicy,
    RiskClass,
    TelemetryLogger,
)
from hitloop.backends import WebhookBackend


# =============================================================================
# Models for the API
# =============================================================================

class ApprovalRequestPayload(BaseModel):
    """Incoming approval request from hitloop."""
    callback_id: str
    callback_url: str
    run_id: str
    action: dict[str, Any]
    policy_name: str
    policy_reason: str
    context: str = ""


class CallbackPayload(BaseModel):
    """Callback from external service (Slack, Telegram, etc.)"""
    approved: bool
    reason: str = ""
    user: str = "unknown"


# =============================================================================
# Global state (in production, use Redis or similar)
# =============================================================================

pending_requests: dict[str, ApprovalRequestPayload] = {}
backend: WebhookBackend | None = None


# =============================================================================
# Your custom send function - adapt this for Slack/Telegram/etc.
# =============================================================================

async def send_to_console(
    request: ApprovalRequest,
    callback_id: str,
    callback_url: str,
) -> None:
    """
    Example: Send approval request to console.
    
    In production, replace this with your Slack/Telegram/Discord integration:
    
    For Slack:
        await slack_client.chat_postMessage(
            channel="#approvals",
            blocks=[...],  # Include approve/reject buttons
            metadata={"callback_id": callback_id}
        )
    
    For Telegram:
        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton("✅", callback_data=f"approve:{callback_id}"),
            InlineKeyboardButton("❌", callback_data=f"reject:{callback_id}"),
        ]])
        await bot.send_message(chat_id=ADMIN_ID, text=msg, reply_markup=keyboard)
    
    For Discord:
        await channel.send(embed=embed, view=ApprovalView(callback_id))
    """
    print("\n" + "=" * 60)
    print("📬 NEW APPROVAL REQUEST")
    print("=" * 60)
    print(f"Callback ID: {callback_id}")
    print(f"Tool: {request.action.tool_name}")
    print(f"Risk: {request.action.risk_class.value}")
    print(f"Args: {json.dumps(request.action.tool_args, indent=2)}")
    print(f"Reason: {request.policy_reason}")
    print("-" * 60)
    print(f"To approve:  curl -X POST {callback_url} -H 'Content-Type: application/json' -d '{{\"approved\": true}}'")
    print(f"To reject:   curl -X POST {callback_url} -H 'Content-Type: application/json' -d '{{\"approved\": false}}'")
    print("=" * 60 + "\n")


# =============================================================================
# FastAPI App
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize backend on startup."""
    global backend
    backend = WebhookBackend(
        send_request=send_to_console,
        timeout_seconds=300,  # 5 minute timeout
        callback_base_url="http://localhost:8000/callback",
    )
    print("🚀 Webhook server started")
    print("   Callback URL: http://localhost:8000/callback/{callback_id}")
    yield
    print("👋 Webhook server stopped")


app = FastAPI(
    title="hitloop Webhook Server",
    description="Example webhook server for hitloop third-party integrations",
    lifespan=lifespan,
)


@app.post("/callback/{callback_id}")
async def handle_callback(callback_id: str, payload: CallbackPayload):
    """
    Callback endpoint - your external service calls this when human responds.
    
    For Slack: Call this from your Slack interaction handler
    For Telegram: Call this from your callback_query handler
    For Discord: Call this from your button interaction handler
    """
    if backend is None:
        raise HTTPException(status_code=500, detail="Backend not initialized")
    
    success = await backend.handle_callback(
        callback_id=callback_id,
        approved=payload.approved,
        reason=payload.reason,
        decided_by=f"webhook:{payload.user}",
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Callback ID not found or expired")
    
    status = "✅ APPROVED" if payload.approved else "❌ REJECTED"
    print(f"\n{status}: {callback_id} by {payload.user}")
    
    return {"status": "ok", "approved": payload.approved}


@app.get("/pending")
async def get_pending():
    """Get all pending approval requests."""
    if backend is None:
        return {"pending": []}
    
    pending = backend.get_pending_requests()
    return {
        "count": len(pending),
        "pending": [
            {
                "callback_id": cid,
                "tool": req.action.tool_name,
                "risk": req.action.risk_class.value,
            }
            for cid, req in pending
        ]
    }


@app.post("/test/request-approval")
async def test_request_approval(background_tasks: BackgroundTasks):
    """
    Test endpoint - simulates an agent requesting approval.
    
    In real usage, your LangGraph agent would call backend.request_approval()
    """
    if backend is None:
        raise HTTPException(status_code=500, detail="Backend not initialized")
    
    # Create a test action
    action = Action(
        tool_name="send_email",
        tool_args={
            "recipient": "ceo@company.com",
            "subject": "Urgent: Budget Approval",
            "body": "Please approve the Q4 budget of $1M",
        },
        risk_class=RiskClass.HIGH,
        rationale="User requested to send important email",
    )
    
    request = ApprovalRequest(
        run_id="test-run-001",
        action=action,
        policy_name="risk_based",
        policy_reason="High-risk tool requires approval",
    )
    
    # Request approval (this will wait for callback)
    async def wait_for_approval():
        decision = await backend.request_approval(request)
        print(f"\n🎯 Decision received: {'APPROVED' if decision.approved else 'REJECTED'}")
        print(f"   Reason: {decision.reason}")
        print(f"   Decided by: {decision.decided_by}")
        print(f"   Latency: {decision.latency_ms:.0f}ms")
    
    background_tasks.add_task(wait_for_approval)
    
    return {
        "status": "pending",
        "message": "Approval request sent. Check console for callback instructions.",
    }


# =============================================================================
# Example integrations (copy-paste templates)
# =============================================================================

SLACK_EXAMPLE = '''
# Slack Integration Example
# -------------------------

from slack_sdk.web.async_client import AsyncWebClient

slack = AsyncWebClient(token="xoxb-your-token")

async def send_to_slack(request: ApprovalRequest, callback_id: str, callback_url: str):
    """Send approval request to Slack."""
    await slack.chat_postMessage(
        channel="#approvals",
        blocks=[
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "🔔 Approval Required"}
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Tool:* `{request.action.tool_name}`"},
                    {"type": "mrkdwn", "text": f"*Risk:* {request.action.risk_class.value.upper()}"},
                ]
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"```{json.dumps(request.action.tool_args, indent=2)}```"}
            },
            {
                "type": "actions",
                "block_id": f"approval_{callback_id}",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "✅ Approve"},
                        "style": "primary",
                        "action_id": "approve",
                        "value": callback_id,
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "❌ Reject"},
                        "style": "danger",
                        "action_id": "reject",
                        "value": callback_id,
                    },
                ]
            }
        ]
    )

# In your Slack event handler:
@app.post("/slack/interactions")
async def slack_interaction(request: Request):
    payload = json.loads((await request.form())["payload"])
    action = payload["actions"][0]
    callback_id = action["value"]
    approved = action["action_id"] == "approve"
    user = payload["user"]["username"]
    
    await backend.handle_callback(callback_id, approved, decided_by=f"slack:@{user}")
'''

TELEGRAM_EXAMPLE = '''
# Telegram Integration Example
# ----------------------------

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler

async def send_to_telegram(request: ApprovalRequest, callback_id: str, callback_url: str):
    """Send approval request to Telegram."""
    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Approve", callback_data=f"approve:{callback_id}"),
            InlineKeyboardButton("❌ Reject", callback_data=f"reject:{callback_id}"),
        ]
    ])
    
    text = f"""
🔔 *Approval Required*

*Tool:* `{request.action.tool_name}`
*Risk:* {request.action.risk_class.value.upper()}

```json
{json.dumps(request.action.tool_args, indent=2)}
```

*Reason:* {request.policy_reason}
"""
    
    await bot.send_message(
        chat_id=ADMIN_CHAT_ID,
        text=text,
        parse_mode="Markdown",
        reply_markup=keyboard,
    )

# In your Telegram callback handler:
async def button_callback(update: Update, context):
    query = update.callback_query
    action, callback_id = query.data.split(":")
    approved = action == "approve"
    user = query.from_user.username
    
    await backend.handle_callback(callback_id, approved, decided_by=f"telegram:@{user}")
    await query.answer("✅ Decision recorded" if approved else "❌ Rejected")
'''


@app.get("/examples/slack")
async def get_slack_example():
    """Get Slack integration example code."""
    return {"code": SLACK_EXAMPLE}


@app.get("/examples/telegram")
async def get_telegram_example():
    """Get Telegram integration example code."""
    return {"code": TELEGRAM_EXAMPLE}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
