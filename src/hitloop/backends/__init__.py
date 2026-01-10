"""Approval backend implementations."""

from hitloop.backends.cli_backend import (
    AutoApproveBackend,
    AutoRejectBackend,
    CLIBackend,
    ScriptedBackend,
)
from hitloop.backends.webhook_backend import (
    WebhookBackend,
    SimpleHTTPWebhookBackend,
)

__all__ = [
    "AutoApproveBackend",
    "AutoRejectBackend",
    "CLIBackend",
    "ScriptedBackend",
    "WebhookBackend",
    "SimpleHTTPWebhookBackend",
]
