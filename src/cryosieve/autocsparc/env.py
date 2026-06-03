"""
Utilities for discovering environment settings.

This module first reads the current process environment. If any requested
variables are missing, it falls back to ``cryosparcm env`` and parses its
shell-style output, for example::

    export "CRYOSPARC_COMMAND_CORE_PORT=39102"
"""

import os
import re
import subprocess
from typing import Dict, Iterable, Mapping, Optional


ENV_COMMAND = ("cryosparcm", "env")
ENV_EXPORT_RE = re.compile(r'^export\s+"([^=]+)=(.*)"\s*$')


def parse_env_output(output: str) -> Dict[str, str]:
    """
    Parse output produced by ``cryosparcm env``.
    """

    env: Dict[str, str] = {}
    for line in output.splitlines():
        match = ENV_EXPORT_RE.match(line.strip())
        if match:
            key, value = match.groups()
            env[key] = value
    return env


def read_cryosparcm_env() -> Dict[str, str]:
    """
    Execute ``cryosparcm env`` and return parsed variables.

    If ``cryosparcm`` is unavailable or exits with an error, an empty mapping is
    returned. This keeps environment discovery best-effort and lets the caller
    fall back to hard-coded defaults when necessary.
    """

    try:
        completed = subprocess.run(
            ENV_COMMAND,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return {}
    return parse_env_output(completed.stdout)


def get_env(required_keys: Iterable[str]) -> Dict[str, str]:
    """
    Return environment values for the requested keys.

    Values already present in ``os.environ`` take precedence. ``cryosparcm env``
    is only executed if one or more requested keys are missing.
    """

    keys = list(required_keys)
    env = {key: os.environ[key] for key in keys if key in os.environ}
    missing = [key for key in keys if key not in env]
    if missing:
        fallback = read_cryosparcm_env()
        for key in missing:
            if key in fallback:
                env[key] = fallback[key]
    return env


def get_env_value(env: Mapping[str, str], key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Return ``env[key]`` unless it is missing or empty.
    """

    value = env.get(key)
    return value if value else default


def get_env_int(env: Mapping[str, str], key: str, default: int) -> int:
    """
    Return an integer environment value, or ``default`` if missing or invalid.
    """

    value = get_env_value(env, key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default
