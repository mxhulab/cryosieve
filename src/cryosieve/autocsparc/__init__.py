"""
Lightweight client for CryoSPARC command_core JSON-RPC APIs.

The package intentionally depends only on Python's standard library. It is
designed for small automation scripts that need to talk to a running
CryoSPARC command_core service.
"""

from .client import CommandClient
from .dryrun import DryRunClient
from .errors import CommandError

__all__ = ["CommandClient", "DryRunClient", "CommandError"]
