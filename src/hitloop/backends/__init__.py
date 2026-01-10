"""Approval backend implementations."""

from hitloop.backends.cli_backend import (
    AutoApproveBackend,
    AutoRejectBackend,
    CLIBackend,
    ScriptedBackend,
)

__all__ = [
    "AutoApproveBackend",
    "AutoRejectBackend",
    "CLIBackend",
    "ScriptedBackend",
]
