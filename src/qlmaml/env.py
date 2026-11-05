"""Load secrets (IQM_TOKEN, NEO4J_PASSWORD) from the repo's git-ignored `.env` into the environment.

The search starts at this package (not the current folder, so it works from anywhere) and walks
up until it finds a `.env`. It stops at the main checkout's root (the folder whose `.git` is a
directory), so a git worktree under `.claude/worktrees/` finds the main checkout's `.env`, and a
stray `.env` above the repo (e.g. in the home folder) is never read.
Variables already set in the shell win over the file. Values are never printed or logged.
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from pathlib import Path


def _parse(text: str) -> dict[str, str]:
    values = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].rstrip()
        values[key.strip()] = value
    return values


def load_dotenv(start: str | Path | None = None, environ: MutableMapping[str, str] = os.environ) -> Path | None:
    """Copy `KEY=value` lines of the repo's `.env` into `environ` (no override). Returns the file read, or None."""
    folder = Path(start or Path(__file__).parent).resolve()
    for candidate in (folder, *folder.parents):
        path = candidate / ".env"
        if path.is_file():
            for key, value in _parse(path.read_text(encoding="utf-8")).items():
                environ.setdefault(key, value)
            return path
        if (candidate / ".git").is_dir():
            break
    return None
