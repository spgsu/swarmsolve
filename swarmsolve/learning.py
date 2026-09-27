"""Auditable episodic memory for completed swarm runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RunMemory:
    def __init__(self, path: Path) -> None:
        self.path = path

    def recall(self, task: str, limit: int = 3) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        terms = set(task.lower().split())
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append((len(terms & set(str(item.get("task", "")).lower().split())), item))
        rows.sort(key=lambda row: row[0], reverse=True)
        return [item for score, item in rows[:limit] if score]

    def record(self, task: str, answer: str, metadata: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"task": task, "answer": answer, "metadata": metadata}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
