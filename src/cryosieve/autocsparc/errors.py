"""
Error types used by autocsparc.
"""

from typing import Any


class CommandError(Exception):
    """
    Raised when a request to CryoSPARC command_core fails.

    Attributes:
        reason: Human-readable failure reason.
        url: URL that was requested.
        code: HTTP or JSON-RPC error code when available.
        data: Optional response data attached to the error.
    """

    reason: str
    url: str
    code: int
    data: Any

    def __init__(self, reason: str, *args: object, url: str = "", code: int = 500, data: Any = None) -> None:
        self.reason = reason
        self.url = url
        self.code = code
        self.data = data
        super().__init__(f"*** ({url}, code {code}) {reason}", *args)
