"""Immutable JSONL run-record format (P0.1).

Every gate evaluation appends exactly one line to an append-only JSONL file.
Each record pins the code revision, config hash, seeds, and the full result so a
gate decision can never be silently re-litigated. Records are content-hashed;
the harness never rewrites a line.
"""
from __future__ import annotations

import getpass
import hashlib
import json
import platform
import socket
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs_causalarc"


def _git_rev() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "no-git"
    except Exception:
        return "no-git"


@dataclass
class RunRecord:
    kind: str  # e.g. "gate0"
    config: Dict[str, Any]
    result: Dict[str, Any]
    seeds: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: time.time())
    git_rev: str = field(default_factory=_git_rev)
    host: str = field(default_factory=socket.gethostname)
    user: str = field(default_factory=lambda: _safe_user())
    python: str = field(default_factory=platform.python_version)

    def content_hash(self) -> str:
        blob = json.dumps(
            {"kind": self.kind, "config": self.config, "result": self.result,
             "seeds": self.seeds},
            sort_keys=True, default=str,
        ).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def to_json(self) -> str:
        d = asdict(self)
        d["record_hash"] = self.content_hash()
        return json.dumps(d, sort_keys=True, default=str)


def _safe_user() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"


def write_record(record: RunRecord, path: Path | None = None) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if path is None:
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(record.timestamp))
        path = RUNS_DIR / f"{stamp}_{record.kind}.jsonl"
    with path.open("a") as f:  # append-only
        f.write(record.to_json() + "\n")
    return path
