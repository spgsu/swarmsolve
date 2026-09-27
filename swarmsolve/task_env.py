"""OpenTaskEnv -- the "world" for an open-ended user task.

There is no enumerable action set and no ground-truth goal predicate. The
world is:

* the task text,
* a shared blackboard (``plan``, ``criteria``, append-only ``findings``, the
  current ``answer`` draft, the latest ``judge`` result), reusing
  :class:`swarmsolve.blackboard.Blackboard`,
* a :class:`~swarmsolve.tools.Workspace` directory on disk.

Progress is whatever the judge says (:meth:`set_judge` / :meth:`score`). It is
a plain class rather than any kind of ABC subclass, because the pipeline
drives it directly and a generic ``legal_actions`` / ``max_rounds`` contract
would not fit an open task.
"""

from __future__ import annotations

import threading
import re
from datetime import datetime, timezone
from typing import Any

from swarmsolve.toolbox import TOOLBOX
from swarmsolve.tools import ToolPolicy, ToolResult, Workspace
from swarmsolve.blackboard import Blackboard


class OpenTaskEnv:
    name = "open_task"

    def __init__(self, task: str, workspace: Workspace, policy: ToolPolicy, threshold: float = 0.85) -> None:
        self.task = task
        self.workspace = workspace
        self.policy = policy
        self.threshold = threshold
        self.bb = Blackboard(decay=0.0)
        self.bb.facts.update({"plan": [], "criteria": [], "findings": [], "answer": "", "work_items": []})
        self.last_judge: dict[str, Any] | None = None
        self._finding_seq = 0
        self._finding_texts: set[str] = set()
        self.generated_files_by_round: dict[int, set[str]] = {}
        self.action_fingerprints_by_revision: dict[int, set[str]] = {}
        self.mutation_revision = 0
        self._action_claim_lock = threading.Lock()
        self._work_seq = 0

    def claim_action(self, round_index: int, fingerprint: str) -> bool:
        """Reserve an inspection/test until a workspace mutation makes it useful again."""
        with self._action_claim_lock:
            claimed = self.action_fingerprints_by_revision.setdefault(self.mutation_revision, set())
            if fingerprint in claimed:
                return False
            claimed.add(fingerprint)
            return True

    def mark_mutation(self) -> None:
        with self._action_claim_lock:
            self.mutation_revision += 1
            # A blocked worker may have been waiting on workspace state rather
            # than a formal DAG edge. A new revision is the safe event that
            # permits one fresh attempt; the attempt budget still prevents loops.
            for item in self.bb.facts.get("work_items", []):
                if (item.get("status") == "blocked"
                        and int(item.get("blocked_revision", -1)) < self.mutation_revision):
                    item.update(status="ready", owner=None, lease_revision=None)

    def open_work_items(self) -> list[dict[str, Any]]:
        """Backward-compatible name for selectable, unclaimed work."""
        return [
            dict(item) for item in self.bb.facts.get("work_items", [])
            if item.get("status") in {"open", "ready"}
        ]

    def pending_work_items(self) -> list[dict[str, Any]]:
        return [
            dict(item) for item in self.bb.facts.get("work_items", [])
            if item.get("status") != "resolved"
        ]

    def ready_work_items(self) -> list[dict[str, Any]]:
        """Return open work whose declared dependencies have been resolved.

        Judge-created gaps have no dependencies and are immediately ready.  A
        structured planner may supply ``depends_on`` identifiers, allowing the
        scheduler to avoid waking an agent for integration work before its
        prerequisite artifact exists.
        """
        items = self.bb.facts.get("work_items", [])
        status = {str(item.get("id")): item.get("status") for item in items}
        ready: list[dict[str, Any]] = []
        for item in items:
            if item.get("status") not in {"open", "ready"}:
                continue
            dependencies = [str(dep) for dep in item.get("depends_on", [])]
            if all(status.get(dep) == "resolved" for dep in dependencies):
                ready.append(dict(item))
        return ready

    def claim_work_item(self, item_id: str, agent_id: str, round_index: int) -> dict[str, Any] | None:
        """Lease one agenda item to an agent and record revision-local attempts."""
        with self._action_claim_lock:
            for item in self.bb.facts.get("work_items", []):
                if item.get("id") != item_id or item.get("status") not in {"open", "ready"}:
                    continue
                if item.get("attempt_revision") != self.mutation_revision:
                    item["attempt_revision"] = self.mutation_revision
                    item["revision_attempts"] = 0
                item["revision_attempts"] = int(item.get("revision_attempts", 0)) + 1
                item["attempts"] = int(item.get("attempts", 0)) + 1
                item.update(
                    status="in_progress",
                    owner=agent_id,
                    started_at=datetime.now(timezone.utc).isoformat(),
                    lease_revision=self.mutation_revision,
                    last_round=round_index,
                    outcome=None,
                )
                return dict(item)
        return None

    def record_work_outcome(
        self,
        item_id: str,
        agent_id: str,
        work_status: str,
        *,
        evidence: list[str] | None = None,
        lessons_learned: list[str] | None = None,
        next_action: str = "",
    ) -> dict[str, Any] | None:
        """Close a lease with a durable handoff; completion awaits verification."""
        normalized = work_status if work_status in {"completed", "incomplete", "blocked"} else "incomplete"
        with self._action_claim_lock:
            for item in self.bb.facts.get("work_items", []):
                if item.get("id") != item_id or item.get("owner") != agent_id:
                    continue
                target = {
                    "completed": "needs_verification",
                    "incomplete": "ready",
                    "blocked": "blocked",
                }[normalized]
                item.update(
                    status=target,
                    outcome=normalized,
                    evidence=(list(item.get("evidence", [])) + list(evidence or []))[-12:],
                    lessons_learned=(list(item.get("lessons_learned", [])) + list(lessons_learned or []))[-12:],
                    next_action=str(next_action or "")[:500],
                    finished_at=datetime.now(timezone.utc).isoformat(),
                )
                if target != "in_progress":
                    item["lease_revision"] = None
                if target == "blocked":
                    item["blocked_revision"] = self.mutation_revision
                return dict(item)
        return None

    def expire_leases(self, round_index: int, max_age_rounds: int = 1) -> list[str]:
        """Recover work left in-progress by a crashed or interrupted worker."""
        expired: list[str] = []
        with self._action_claim_lock:
            for item in self.bb.facts.get("work_items", []):
                if item.get("status") != "in_progress":
                    continue
                if round_index - int(item.get("last_round", round_index)) < max_age_rounds:
                    continue
                item.update(
                    status="ready", owner=None, lease_revision=None, outcome="lease_expired",
                    next_action="Retry from the recorded evidence; the previous worker did not post a handoff.",
                )
                item.setdefault("evidence", []).append("worker lease expired without a handoff")
                expired.append(str(item.get("id")))
        return expired

    def apply_verification(self, report: Any) -> list[dict[str, Any]]:
        """Promote completed work only after independent project verification."""
        transitions: list[dict[str, Any]] = []
        summary = report.summary()
        with self._action_claim_lock:
            for item in self.bb.facts.get("work_items", []):
                if item.get("status") != "needs_verification":
                    continue
                before = item["status"]
                if report.has_failure:
                    item.update(status="ready", outcome="verification_failed",
                                next_action="Fix the verification failure and rerun the narrowest relevant check.")
                    item.setdefault("evidence", []).append(summary[:500])
                else:
                    item.update(status="resolved", outcome="verified", resolved_at=datetime.now(timezone.utc).isoformat())
                transitions.append({"id": item.get("id"), "from": before, "to": item.get("status")})
            # Resolving a dependency may make a blocked node runnable even if
            # verification itself did not mutate the workspace.
            status = {str(item.get("id")): item.get("status") for item in self.bb.facts.get("work_items", [])}
            for item in self.bb.facts.get("work_items", []):
                deps = [str(dep) for dep in item.get("depends_on", [])]
                if item.get("status") == "blocked" and deps and all(status.get(dep) == "resolved" for dep in deps):
                    item.update(status="ready", owner=None, lease_revision=None)
                    transitions.append({"id": item.get("id"), "from": "blocked", "to": "ready"})
        return transitions

    # -- setup -----------------------------------------------------------
    def set_plan(self, plan: list[Any], criteria: list[str]) -> None:
        """Install both the human-readable plan and its executable work graph.

        String steps remain supported for old callers.  Structured steps may
        declare ``id``, ``task``, ``depends_on``, and a local ``acceptance``
        predicate.  The latter form is what the planner prompt now requests.
        """
        display: list[str] = []
        items = self.bb.facts.setdefault("work_items", [])
        known_ids = {str(item.get("id")) for item in items}
        for index, raw in enumerate(plan or [], 1):
            if isinstance(raw, dict):
                text = str(raw.get("task") or raw.get("text") or "").strip()
                proposed_id = re.sub(r"[^A-Za-z0-9_-]", "", str(raw.get("id") or f"P{index:03d}"))
                item_id = proposed_id or f"P{index:03d}"
                # The planner contract permits dependencies on earlier nodes
                # only. Drop unknown/self/forward references rather than letting
                # one malformed model id deadlock the scheduler forever.
                dependencies = [
                    str(dep) for dep in (raw.get("depends_on") or [])
                    if str(dep).strip() and str(dep) in known_ids
                ]
                acceptance = str(raw.get("acceptance") or "").strip()
            else:
                text = str(raw).strip()
                item_id = f"P{index:03d}"
                dependencies = []
                acceptance = ""
            if not text:
                continue
            while item_id in known_ids:
                item_id = f"P{index:03d}-{len(known_ids) + 1}"
            known_ids.add(item_id)
            display.append(text)
            items.append({
                "id": item_id,
                "text": text,
                "status": "ready",
                "source": "plan",
                "depends_on": dependencies,
                "acceptance": acceptance,
                "critical": False,
                "seen": 1,
                "attempts": 0,
                "revision_attempts": 0,
                "attempt_revision": self.mutation_revision,
                "owner": None,
                "lease_revision": None,
                "outcome": None,
                "evidence": [],
                "lessons_learned": [],
                "next_action": "",
            })
        self.bb.facts["plan"] = display
        self.bb.facts["criteria"] = list(criteria or [])

    def add_finding(self, agent_id: str, text: str, round_index: int) -> None:
        text = str(text).strip()[:1200]
        if not text or text in self._finding_texts:
            return
        self._finding_texts.add(text)
        self._finding_seq += 1
        self.bb.facts["findings"].append(
            {"n": self._finding_seq, "round": round_index, "agent": agent_id, "text": text}
        )

    # -- observation ---------------------------------------------------
    def observe(self) -> dict[str, Any]:
        judge_view = None
        if self.last_judge:
            # workers only need the public fields -- don't burn their context on _usage/_raw
            judge_view = {k: v for k, v in self.last_judge.items() if not k.startswith("_")}
        return {
            "facts": self.bb.view()["facts"],
            "files": self.workspace.summary(),
            "judge": judge_view,
        }

    # -- action dispatch --------------------------------------------
    def apply(self, agent_id: str, action: str, args: dict[str, Any], round_index: int) -> dict[str, Any]:
        """Run one agent action. Returns a record for the event log."""
        rec: dict[str, Any] = {"agent": agent_id, "action": action, "ok": True}
        ws, pol = self.workspace, self.policy

        if action == "think":
            txt = str(args.get("text", "")).strip()
            self.add_finding(agent_id, f"[think] {txt}", round_index)
            rec["detail"] = txt[:200]

        elif action == "note":
            txt = str(args.get("text", "")).strip()
            if txt:
                self.add_finding(agent_id, txt, round_index)
            rec["detail"] = txt[:200]

        elif action in TOOLBOX:
            res = TOOLBOX.dispatch(action, ws, args, pol)
            tag = TOOLBOX.label_for(action, args)
            self.add_finding(agent_id, f"[{tag}] {res.short(800)}", round_index)
            rec.update(ok=res.ok, detail=res.short(300))
            if res.ok and action in {"write_file", "replace_text"} and (res.meta or {}).get("changed", True):
                self.mark_mutation()

        elif action == "propose_answer":
            txt = str(args.get("text", "")).strip()
            if txt:
                self.bb.facts["answer"] = txt
                self.add_finding(agent_id, f"[proposed answer, {len(txt)} chars]", round_index)
            else:
                rec.update(ok=False, detail="empty proposal ignored")

        else:  # pragma: no cover - parse_action already normalises
            rec.update(ok=False, detail=f"unknown action {action!r}")

        return rec

    # -- judgement / outcome --------------------------------------
    @property
    def answer(self) -> str:
        return self.bb.facts.get("answer", "")

    def set_judge(self, result: dict[str, Any]) -> None:
        self.last_judge = result
        if result.get("_valid", True):
            self._sync_work_items(result.get("gaps") or [])

    @staticmethod
    def _gap_words(text: str) -> set[str]:
        stop = {"critical", "the", "and", "that", "this", "with", "from", "into", "not", "still", "has", "have"}
        return {w for w in re.findall(r"[a-z0-9_]+", text.lower()) if len(w) > 2 and w not in stop}

    def _sync_work_items(self, gaps: list[Any]) -> None:
        """Maintain a durable, semantic agenda instead of a prose-only gap tail."""
        items = self.bb.facts.setdefault("work_items", [])
        seen_ids: set[str] = set()
        for raw in gaps:
            text = str(raw).strip()
            if not text:
                continue
            words = self._gap_words(text)
            best = None
            best_score = 0.0
            for item in items:
                existing = self._gap_words(str(item.get("text", "")))
                union = words | existing
                score = len(words & existing) / len(union) if union else 0.0
                if score > best_score:
                    best, best_score = item, score
            if best is None or best_score < 0.45:
                self._work_seq += 1
                best = {
                    "id": f"G{self._work_seq:03d}", "text": text, "status": "ready",
                    "source": "judge", "depends_on": [], "acceptance": "",
                    "critical": text.upper().startswith("CRITICAL"), "seen": 0,
                    "attempts": 0, "revision_attempts": 0,
                    "attempt_revision": self.mutation_revision,
                    "owner": None, "lease_revision": None, "outcome": None,
                    "evidence": [], "lessons_learned": [], "next_action": "",
                }
                items.append(best)
            best.update(
                text=text,
                status="ready",
                critical=text.upper().startswith("CRITICAL"),
                seen=int(best.get("seen", 0)) + 1,
            )
            seen_ids.add(best["id"])
        for item in items:
            if item.get("status") != "resolved" and item.get("id") not in seen_ids:
                item["status"] = "resolved"

    def score(self) -> float:
        if not self.last_judge:
            return 0.0
        try:
            return max(0.0, min(1.0, float(self.last_judge.get("score", 0.0))))
        except (TypeError, ValueError):
            return 0.0

    def critical_gaps(self) -> list[str]:
        gaps = (self.last_judge or {}).get("gaps") or []
        return [g for g in gaps if isinstance(g, str) and g.strip().upper().startswith("CRITICAL")]

    def goal_satisfied(self) -> bool:
        if not self.last_judge or not self.answer:
            return False
        if self.critical_gaps():
            return False
        if self.last_judge.get("done") is True and self.score() >= self.threshold:
            return True
        return self.score() >= self.threshold

    def snapshot(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "plan": self.bb.facts.get("plan", []),
            "criteria": self.bb.facts.get("criteria", []),
            "n_findings": len(self.bb.facts.get("findings", [])),
            "answer_chars": len(self.answer),
            "score": round(self.score(), 3),
            "files": self.workspace.list_files(),
        }
