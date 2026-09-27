"""Shared blackboard with optional pheromone-style decay.

Two channels:

* ``facts``     — merged world knowledge, last-write-wins per key. Persistent.
* ``pheromone`` — scalar intensities keyed by an arbitrary tag. Decays each
                  round by ``decay``.

Agents write via writes shaped as::

    {"facts": {...}, "pheromone": {tag: amount, ...}}
"""

from __future__ import annotations

import copy
from typing import Any


class Blackboard:
    def __init__(self, decay: float = 0.0, prune_below: float = 1e-3) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must be in [0, 1)")
        self.decay = decay
        self.prune_below = prune_below
        self.facts: dict[str, Any] = {}
        self.pheromone: dict[str, float] = {}
        self.write_log: list[dict[str, Any]] = []  # provenance for analysis

    def write(self, writes: dict[str, Any], agent_id: str, round_: int) -> None:
        if not writes:
            return
        facts = writes.get("facts") or {}
        for key, value in facts.items():
            self.facts[key] = _merge(self.facts.get(key), value)
        for tag, amount in (writes.get("pheromone") or {}).items():
            self.pheromone[tag] = self.pheromone.get(tag, 0.0) + float(amount)
        self.write_log.append(
            {"round": round_, "agent": agent_id, "facts": list(facts), "pheromone": dict(writes.get("pheromone") or {})}
        )

    def tick(self) -> None:
        """Apply decay. Call once per round."""
        if self.decay <= 0.0:
            return
        keep = (1.0 - self.decay)
        for tag in list(self.pheromone):
            self.pheromone[tag] *= keep
            if self.pheromone[tag] < self.prune_below:
                del self.pheromone[tag]

    def view(self) -> dict[str, Any]:
        """A read-only snapshot handed to agents."""
        return {"facts": copy.deepcopy(self.facts), "pheromone": dict(self.pheromone)}


def _merge(cur: Any, new: Any) -> Any:
    """Combine an existing fact with an incoming one.

    dict + dict  -> recursive merge   (e.g. the shared adjacency map)
    list + list  -> set union         (e.g. the shared 'visited' set)
    otherwise    -> last write wins
    """
    if isinstance(cur, dict) and isinstance(new, dict):
        out = dict(cur)
        for k, v in new.items():
            out[k] = _merge(out.get(k), v)
        return out
    if isinstance(cur, list) and isinstance(new, list):
        try:
            return sorted(set(cur) | set(new))
        except TypeError:
            return new
    return new
