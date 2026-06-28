"""
Core-only CryoSPARC command client.

This module implements a small JSON-RPC 2.0 client for CryoSPARC's
``command_core`` service. It intentionally uses only Python's standard library:

* ``urllib`` for HTTP
* ``json`` for JSON serialization
* ``uuid`` for JSON-RPC request IDs

Typical usage:

    from autocsparc import CommandClient

    cli = CommandClient()

    print(cli.get_system_info())
    print(cli.call("list_projects"))

The client calls ``system.describe`` at initialization and creates dynamic
methods for each exposed JSON-RPC endpoint. If ``host``, ``port`` or
``license_id`` are not passed explicitly, the client reads
``CRYOSPARC_MASTER_HOSTNAME``, ``CRYOSPARC_COMMAND_CORE_PORT`` and
``CRYOSPARC_LICENSE_ID`` from the current environment, falling back to
``cryosparcm env`` when necessary. ``host`` and ``port`` must be provided either
explicitly or through the CryoSPARC environment.
"""

import json
import socket
import time
import uuid
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from warnings import warn

from .env import (
    get_env,
    get_env_int,
    get_env_value,
)
from .errors import CommandError


SERVICE_NAME = "command_core"
ENV_MASTER_HOSTNAME = "CRYOSPARC_MASTER_HOSTNAME"
ENV_COMMAND_CORE_PORT = "CRYOSPARC_COMMAND_CORE_PORT"
ENV_LICENSE_ID = "CRYOSPARC_LICENSE_ID"
ENV_COMMAND_RETRIES = "CRYOSPARC_COMMAND_RETRIES"
ENV_COMMAND_RETRY_SECONDS = "CRYOSPARC_COMMAND_RETRY_SECONDS"
DEFAULT_RETRIES = 3
DEFAULT_RETRY_INTERVAL = 30


class CommandClient:
    """
    Lightweight client for CryoSPARC's ``command_core`` JSON-RPC service.

    Args:
        host: Hostname or IP address of the CryoSPARC master. If omitted,
            ``CRYOSPARC_MASTER_HOSTNAME`` is used, falling back to
            ``cryosparcm env``.
        port: command_core port. With the default CryoSPARC base port 39000,
            command_core is usually 39002. If omitted,
            ``CRYOSPARC_COMMAND_CORE_PORT`` is used, falling back to
            ``cryosparcm env``.
        license_id: CryoSPARC license ID used in the ``License-ID`` header.
            If omitted, ``CRYOSPARC_LICENSE_ID`` is read from the environment,
            falling back to ``cryosparcm env``.
        timeout: Request timeout in seconds.
        retries: Number of attempts for URL/timeout transport failures.
        retry_interval: Seconds to sleep between retry attempts.

    Examples:
        >>> cli = CommandClient(license_id="...")
        >>> cli.get_system_info()
        >>> cli.call("get_system_info")
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        *,
        license_id: Optional[str] = None,
        timeout: int = 300,
        retries: Optional[int] = None,
        retry_interval: Optional[int] = None,
    ) -> None:
        env = get_env(
            (
                ENV_MASTER_HOSTNAME,
                ENV_COMMAND_CORE_PORT,
                ENV_LICENSE_ID,
                ENV_COMMAND_RETRIES,
                ENV_COMMAND_RETRY_SECONDS,
            )
        )

        if host is None:
            host = get_env_value(env, ENV_MASTER_HOSTNAME)
        if port is None:
            port_value = get_env_value(env, ENV_COMMAND_CORE_PORT)
            if port_value is not None:
                port = int(port_value)
        if license_id is None:
            license_id = get_env_value(env, ENV_LICENSE_ID)
        if retries is None:
            retries = get_env_int(env, ENV_COMMAND_RETRIES, DEFAULT_RETRIES)
        if retry_interval is None:
            retry_interval = get_env_int(env, ENV_COMMAND_RETRY_SECONDS, DEFAULT_RETRY_INTERVAL)

        if host is None:
            raise CommandError(
                f"Missing {ENV_MASTER_HOSTNAME}; pass host explicitly or make it available via cryosparcm env"
            )
        if port is None:
            raise CommandError(
                f"Missing {ENV_COMMAND_CORE_PORT}; pass port explicitly or make it available via cryosparcm env"
            )

        self.service = SERVICE_NAME
        self.host = host
        self.port = port
        self._url = f"http://{host}:{port}"
        self._timeout = timeout
        self._retries = retries
        self._retry_interval = retry_interval
        self._endpoints: List[str] = []

        self._headers = {"Originator": "client"}
        if license_id:
            self._headers["License-ID"] = license_id

        self.reload()

    @property
    def base_url(self) -> str:
        """Base URL for the command_core service, e.g. ``http://localhost:39002``."""

        return self._url

    @property
    def endpoints(self) -> List[str]:
        """Names of JSON-RPC endpoints discovered via ``system.describe``."""

        return list(self._endpoints)

    def call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """
        Call a JSON-RPC method exposed by command_core.

        Positional arguments are encoded as a JSON-RPC params array. Keyword
        arguments are encoded as a params object. Mixing both is rejected to
        avoid ambiguous calls.
        """

        if args and kwargs:
            raise TypeError("Use either positional args or keyword args, not both")

        params: Any = kwargs if kwargs else list(args)
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": str(uuid.uuid4()),
        }

        try:
            response = self.json_request("/api", data=payload)
        except CommandError as err:
            raise CommandError(
                f'Encountered transport error from JSON-RPC method "{method}" with params {params}',
                url=err.url or self._url,
                code=err.code,
                data=err.data,
            ) from err

        if not response:
            raise CommandError(
                f'JSON response not received from JSON-RPC method "{method}" with params {params}',
                url=self._url,
            )

        if "error" in response:
            error = response["error"]
            raise CommandError(
                f'Encountered {error.get("name", "Error")} from JSON-RPC method "{method}" with params {params}:\n'
                f"{format_server_error(error)}",
                url=self._url,
                code=error.get("code", 500),
                data=error.get("data"),
            )

        return response.get("result")

    def reload(self) -> None:
        """
        Refresh the list of JSON-RPC endpoints and attach them as methods.
        """

        system = self.call("system.describe")
        procs = system.get("procs", []) if isinstance(system, dict) else []
        self._endpoints = [proc["name"] for proc in procs if isinstance(proc, dict) and "name" in proc]

        for endpoint in self._endpoints:
            setattr(self, endpoint, self._make_rpc_method(endpoint))

    def __call__(self) -> None:
        """Alias for ``reload()`` for compatibility with CryoSPARC's client."""

        self.reload()

    def request(
        self,
        path: str = "",
        *,
        method: str = "POST",
        query: Optional[Dict[str, Any]] = None,
        data: Optional[bytes] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> bytes:
        """
        Make a raw HTTP request to command_core and return response bytes.

        This is mostly a low-level escape hatch. For JSON-RPC calls, prefer
        ``call(...)``.
        """

        request_url = self._url + path
        if query:
            request_url += "?" + urlencode(query)

        request_headers = dict(self._headers)
        if headers:
            request_headers.update(headers)

        last_reason = "<unknown>"
        last_code = 500
        last_data: Any = None

        for attempt in range(1, self._retries + 1):
            req = Request(request_url, data=data, headers=request_headers, method=method)
            try:
                with urlopen(req, timeout=self._timeout) as response:
                    return response.read()
            except HTTPError as error:
                last_code = error.code
                last_reason = (
                    f"HTTP Error {error.code} {error.reason}; "
                    f"please check cryosparcm log {self.service} for additional information."
                )
                last_data = error.read() if error.readable() else None
                if last_data and error.headers.get_content_type() == "application/json":
                    try:
                        last_data = json.loads(last_data)
                    except (TypeError, ValueError):
                        pass
                raise CommandError(last_reason, url=request_url, code=last_code, data=last_data)
            except URLError as error:
                last_reason = f"URL Error {error.reason}"
                if attempt < self._retries:
                    warn(
                        f"*** {type(self).__name__}: ({request_url}) {last_reason}, "
                        f"attempt {attempt} of {self._retries}. Retrying in {self._retry_interval} seconds",
                        stacklevel=2,
                    )
                    time.sleep(self._retry_interval)
            except (TimeoutError, socket.timeout):
                last_reason = f"Timeout Error after {self._timeout} seconds"
                if attempt < self._retries:
                    warn(
                        f"*** {type(self).__name__}: command ({request_url}) did not reply within "
                        f"timeout of {self._timeout} seconds, attempt {attempt} of {self._retries}. "
                        f"Retrying in {self._retry_interval} seconds",
                        stacklevel=2,
                    )
                    time.sleep(self._retry_interval)

        raise CommandError(last_reason, url=request_url, code=last_code, data=last_data)

    def json_request(
        self,
        path: str = "",
        *,
        query: Optional[Dict[str, Any]] = None,
        data: Any = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        """
        Send JSON request data and decode a JSON response.
        """

        request_headers = {"Content-Type": "application/json"}
        if headers:
            request_headers.update(headers)

        body = json.dumps(data).encode("utf-8")
        raw = self.request(path, method="POST", query=query, data=body, headers=request_headers)
        return json.loads(raw.decode("utf-8"))

    def _make_rpc_method(self, method: str):
        def rpc_method(*args: Any, **kwargs: Any) -> Any:
            return self.call(method, *args, **kwargs)

        rpc_method.__name__ = method.replace(".", "_")
        rpc_method.__doc__ = f'Dynamically generated wrapper for JSON-RPC method "{method}".'
        return rpc_method


def format_server_error(error: Dict[str, Any]) -> str:
    """
    Format a JSON-RPC error object returned by CryoSPARC.
    """

    message = error["message"] if "message" in error else str(error)
    data = error.get("data")
    if data:
        if isinstance(data, dict) and "traceback" in data:
            message += "\n" + data["traceback"]
        else:
            message += "\n" + str(data)
    return message
