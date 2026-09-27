"""The judge -- scores the swarm's current answer AND advises on the next move.

Point of routing this to a stronger model (e.g. Claude) while workers run on a
weaker/local one: the judge sees the same findings the workers do, every round,
and is asked to reason harder about them -- not just grade, but name the single
best concrete next action (a path, a payload, a technique). That advice flows
back into every worker's view next round (``render_workspace``'s
``judge_feedback``).

Consulted as soon as there's something to react to (an answer OR findings) --
not only once an answer exists, so the advisor role is useful during pure
recon/tool-use rounds too, before anyone has proposed anything.

Fail-closed by contract: if the judge model errors or returns something
unparseable, :meth:`assess` returns a zero score with a CRITICAL gap, so a broken
judge can never let the pipeline declare the task done. A verification failure
(real build/test output, from ``swarmsolve.verify``) is the same contract
applied one level deeper: the judge's own opinion of the answer never
overrides what the project's own build/test command actually reported.

``assess()`` also returns ``raw_done`` -- the judge's own "done" claim,
*before* it gets masked by ``has_answer`` (and, when it fails, verification).
The masked ``done`` field answers "should the pipeline stop"; ``raw_done``
answers "does the judge itself think this is finished" independent of
whether anything was actually submitted -- the distinction the pipeline's
stall detector needs (a judge saying "complete" for round after round while
no worker ever calls ``propose_answer`` is a stall, not progress; see
``pipeline.py``'s round loop).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from swarmsolve import prompts
from swarmsolve.llm import LLMError, ModelSpec, chat, estimate_cost

if TYPE_CHECKING:
    from swarmsolve.verify import VerificationReport


@dataclass
class Judge:
    # 600 truncated a reasoning model's score+verdict+gaps+advice JSON mid-object
    # for most rounds of a real run -> "judge reply unparseable" -> score 0.0 all
    # run. The judge output is bounded (one score, a verdict line, a few gaps, one
    # advice paragraph); 1200 comfortably fits it without meaningfully changing cost.
    spec: ModelSpec
    max_tokens: int = 1200

    def assess(
        self,
        *,
        task: str,
        criteria: list[str],
        answer: str,
        workspace_summary: str,
        findings_tail: list[Any] | None = None,
        remaining_budget_usd: float | None = None,
        verification: "VerificationReport | None" = None,
    ) -> dict[str, Any]:
        findings_tail = findings_tail or []
        has_answer = bool(answer.strip())
        if not has_answer and not findings_tail:
            # round 0-ish: truly nothing to react to yet, don't spend a call
            return {"score": 0.0, "verdict": "nothing to assess yet",
                    "gaps": ["CRITICAL: the swarm has not proposed an answer"], "done": False,
                    "raw_done": False, "advice": "", "_valid": True}

        user = json.dumps(
            {
                "task": task,
                "acceptance_criteria": criteria,
                "current_answer": answer or "(none proposed yet -- advise based on findings below)",
                "workspace_files": workspace_summary,
                "recent_findings": findings_tail,
                "verification": verification.summary() if verification else None,
            },
            default=str,
        )
        system = prompts.judge_system()
        if remaining_budget_usd is not None:
            est = estimate_cost(self.spec, system, user, self.max_tokens)
            if est > remaining_budget_usd:
                return {
                    "score": 0.0,
                    "verdict": "judge skipped: insufficient budget remaining",
                    "gaps": ["SYSTEM ERROR: judge skipped -- budget cap reached"],
                    "done": False,
                    "raw_done": False,
                    "advice": "",
                    "_valid": False,
                }
        replies = []
        parsed = None
        judge_user = user
        for attempt in range(2):
            try:
                reply = chat(self.spec, system, judge_user, max_tokens=self.max_tokens, temperature=0.1)
            except LLMError as e:
                return {
                    "score": 0.0,
                    "verdict": f"judge unavailable: {e}",
                    "gaps": [f"SYSTEM ERROR: judge model error ({e})"],
                    "done": False,
                    "raw_done": False,
                    "advice": "",
                    "_valid": False,
                    "_usage": _combined_usage(replies),
                    "_calls": len(replies),
                }
            replies.append(reply)
            parsed = prompts.parse_json_reply(reply.text)
            if parsed and "score" in parsed:
                break
            if attempt == 0:
                judge_user = (
                    user + "\n\nYour previous response was not a complete valid JSON object. Retry once with "
                    "only the required compact JSON fields: score, verdict, gaps, done, advice."
                )
        if not parsed or "score" not in parsed:
            return {
                "score": 0.0,
                "verdict": "judge reply unparseable",
                # Fail closed, but do not pretend an infrastructure/protocol
                # failure is task complexity eligible for adaptive rounds.
                "gaps": ["SYSTEM ERROR: judge output was not valid JSON after retry"],
                "done": False,
                "raw_done": False,
                "advice": "",
                "_raw": reply.text[:1000],
                "_usage": _combined_usage(replies),
                "_valid": False,
                "_calls": len(replies),
            }
        try:
            score = max(0.0, min(1.0, float(parsed.get("score", 0.0))))
        except (TypeError, ValueError):
            score = 0.0
        gaps = parsed.get("gaps") or []
        if not isinstance(gaps, list):
            gaps = [str(gaps)]
        raw_done = bool(parsed.get("done", False))
        result = {
            "score": score if has_answer else 0.0,  # never "done"/scored without an actual proposal
            "verdict": str(parsed.get("verdict", ""))[:200],
            "gaps": [str(g) for g in gaps][:10],
            "done": raw_done and has_answer,
            "raw_done": raw_done,  # the judge's own claim, unmasked -- see the module docstring
            "advice": str(parsed.get("advice", ""))[:400],
            "_usage": _combined_usage(replies),
            "_valid": True,
            "_calls": len(replies),
        }
        if verification is not None and verification.has_failure:
            # deliberately NOT zeroing score here (unlike the other fail-closed branches
            # above) -- keeping the judge's own number visible is a useful diagnostic: it
            # shows whether the judge was fooled (high score despite a real failure) or
            # agreed the answer was bad on its own merits.
            result["done"] = False
            crit = f"CRITICAL: verification failed -- {verification.summary()}"
            if crit not in result["gaps"]:
                result["gaps"] = [crit] + result["gaps"][:9]
        return result


def _combined_usage(replies: list[Any]) -> dict[str, float | int]:
    usage = {"tokens_in": 0, "tokens_out": 0, "usd": 0.0}
    for reply in replies:
        item = reply.usage()
        usage["tokens_in"] += int(item.get("tokens_in", 0))
        usage["tokens_out"] += int(item.get("tokens_out", 0))
        usage["usd"] += float(item.get("usd", 0.0))
    return usage
