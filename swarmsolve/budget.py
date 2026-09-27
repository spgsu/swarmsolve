"""Per-episode budget tracking + caps.

A cap of 0 means "unlimited on that axis".
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BudgetTracker:
    max_agent_steps: int = 100_000
    max_tokens: int = 0
    max_usd: float = 0.0

    agent_steps: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    usd: float = 0.0

    def charge_step(self, n: int = 1) -> None:
        self.agent_steps += n

    def charge_tokens(self, tokens_in: int, tokens_out: int, usd: float = 0.0) -> None:
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        self.usd += usd

    def exhausted(self) -> bool:
        if self.max_agent_steps and self.agent_steps >= self.max_agent_steps:
            return True
        if self.max_tokens and (self.tokens_in + self.tokens_out) >= self.max_tokens:
            return True
        if self.max_usd and self.usd >= self.max_usd:
            return True
        return False

    def as_dict(self) -> dict[str, float]:
        return {
            "agent_steps": self.agent_steps,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tokens_total": self.tokens_in + self.tokens_out,
            "usd": round(self.usd, 6),
        }
