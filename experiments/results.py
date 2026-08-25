"""Phase B results store — append-only JSONL with dedup by (D, φ, ansatz, seed).

JSONL chosen over a single JSON file so partial runs can append without rewriting
the whole file (resumability per CLAUDE.md §Computational scope).

Schema per line:
    {
      "dataset_id": str,
      "encoding_id": str,
      "ansatz_id": str,
      "seed": int,
      "n_qubits": int,
      "v": {component_name: {value, ...}, ...},
      "q": {component_name: float | null, ...},
      "epsilon_train_q": float,
      "epsilon_test_q": float,
      "epsilon_train_c": float,
      "epsilon_test_c": float,
      "e_raw": float,                  # ε_test_c - ε_test_q
      "e_budget": float | null,        # placeholder until budget calibration
      "gap_obs": float,                # ε_test_q - ε_train_q
      "epsilon_emp": float,            # ε_train_q
      "wall_time_s": {...},            # per-stage timings
      "timestamp": str (UTC ISO),
      "config": {...}                  # trainer hyperparams for reproducibility
    }
"""
from __future__ import annotations

import fcntl
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional


@dataclass
class PhaseBRecord:
    dataset_id: str
    encoding_id: str
    ansatz_id: str
    seed: int
    n_qubits: int
    v: dict[str, Any]
    q: dict[str, Any]
    epsilon_train_q: float
    epsilon_test_q: float
    epsilon_train_c: float
    epsilon_test_c: float
    e_raw: float
    e_budget: Optional[float]
    gap_obs: float
    epsilon_emp: float
    wall_time_s: dict[str, float]
    timestamp: str
    config: dict[str, Any] = field(default_factory=dict)

    def key(self) -> tuple[str, str, str, int]:
        return (self.dataset_id, self.encoding_id, self.ansatz_id, self.seed)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_existing_keys(path: str | Path) -> set[tuple[str, str, str, int]]:
    """Return the set of (D, φ, ansatz, seed) keys already present in the JSONL."""
    path = Path(path)
    if not path.exists():
        return set()
    seen: set[tuple[str, str, str, int]] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                seen.add((r["dataset_id"], r["encoding_id"], r["ansatz_id"], int(r["seed"])))
            except (json.JSONDecodeError, KeyError):
                continue
    return seen


def append_record(path: str | Path, record: PhaseBRecord) -> None:
    """Append a record to the JSONL, safe for concurrent writers.

    The bulk sweep runs many worker processes that all append to the same file as
    each cell finishes. An advisory `flock` (LOCK_EX) serializes the writes so
    lines never interleave (records exceed PIPE_BUF, so O_APPEND alone is not
    atomic). Contention is negligible — a cell takes seconds-to-minutes, the write
    microseconds. The main file therefore grows live and is crash-safe per cell."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(record)) + "\n"
    with open(path, "a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def iter_records(path: str | Path) -> Iterator[PhaseBRecord]:
    """Yield PhaseBRecord objects from the JSONL."""
    path = Path(path)
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                yield PhaseBRecord(**data)
            except (json.JSONDecodeError, TypeError):
                continue
