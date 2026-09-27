"""General-purpose task swarm: planner + adaptive worker swarm + judge on a
shared blackboard.

Give it a natural-language task or question; a swarm of tool-using agents
(local Ollama / Claude / Gemini / DeepSeek, mix freely) works it on a shared
blackboard until a judge agent scores the answer past a threshold.

    from swarmsolve import solve, SolveConfig
    from swarmsolve.llm import ModelSpec

    cfg = SolveConfig(task="...", workers=[ModelSpec.parse("deepseek/deepseek-chat")])
    result = solve(cfg)
    print(result.answer)

CLI: ``python -m swarmsolve "..."`` (see ``README.md``).
"""

from __future__ import annotations

from swarmsolve.pipeline import SolveConfig, SolveResult, solve

__all__ = ["solve", "SolveConfig", "SolveResult"]
