"""DSH fine-tune worker package.

Run as a module so the host can pass a config path instead of building a command
line:  ``python -m fnt_worker <verb> --config <path>``.

Deliberately dependency-free at import time. ``--help`` and config validation
must work on a machine that has no torch; each verb imports its ML stack inside
its own function and reports a missing dependency as a protocol error.

The protocol helpers live here (not only in ``__main__``) because submodules
such as ``calibrate`` need them: importing them from ``__main__`` would execute
the CLI entry point a second time.
"""

from __future__ import annotations

import json
import sys
from typing import Any

__version__ = "0.1.0"


def emit(**event: Any) -> None:
    """One protocol event on stdout. Keep it small: the host caps lines at 256 KB."""
    sys.stdout.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
    sys.stdout.flush()


def log(message: str, level: str = "info") -> None:
    """A human-readable line on the protocol channel (rendered in the panel)."""
    emit(kind="log", level=level, message=message)


def note(message: str) -> None:
    """A human-readable line on stderr only (goes to stderr.log)."""
    sys.stderr.write(message + "\n")
    sys.stderr.flush()


def die(code: str, message: str) -> None:
    """Report a fatal, structured error on the protocol channel and exit non-zero."""
    emit(kind="error", code=code, message=message)
    raise SystemExit(2)
