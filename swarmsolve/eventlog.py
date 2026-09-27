"""Structured event log: one JSON object per line, plus an in-memory copy, hash-chained
for tamper-evidence.

Every run emits ``episode_start``, a ``step`` per agent action, ``round_end``, and
``episode_end``. This is the single source of truth for what happened during a run.
"""

from __future__ import annotations

import json
import hashlib
import copy
import time
from pathlib import Path
from typing import Any, Optional


class EventLog:
    def __init__(self, path: Optional[str | Path] = None, keep_in_memory: bool = True) -> None:
        self.path = Path(path) if path else None
        self.keep_in_memory = keep_in_memory
        self.records: list[dict[str, Any]] = []
        self._fh = None
        self._last_hash = "GENESIS"
        self._sequence = 0
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("w", encoding="utf-8")

    def emit(self, kind: str, **data: Any) -> None:
        record = {"seq": self._sequence, "ts": round(time.time(), 6), "kind": kind,
                  "previous_hash": self._last_hash, **copy.deepcopy(data)}
        canonical = json.dumps(record, default=_safe, sort_keys=True, separators=(",", ":"))
        record["hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        self._last_hash = record["hash"]
        self._sequence += 1
        if self.keep_in_memory:
            self.records.append(record)
        if self._fh is not None:
            self._fh.write(json.dumps(record, default=_safe) + "\n")
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    @staticmethod
    def verify(records: list[dict[str, Any]]) -> bool:
        previous = "GENESIS"
        for sequence, original in enumerate(records):
            record = dict(original)
            digest = record.pop("hash", None)
            if record.get("seq") != sequence or record.get("previous_hash") != previous:
                return False
            canonical = json.dumps(record, default=_safe, sort_keys=True, separators=(",", ":"))
            if digest != hashlib.sha256(canonical.encode("utf-8")).hexdigest():
                return False
            previous = digest
        return True

    def __enter__(self) -> "EventLog":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _safe(obj: Any) -> Any:
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    return str(obj)
