"""The opinionated solve loop: adaptive blackboard scheduler + planner + judge.

    planner (round 0)  ->  plan + acceptance criteria onto the blackboard
    coordination cycles 1..R:
        activate only workers owning ready work (a small exploratory cohort on
        cycle one, normally one implementation owner thereafter)
        each active worker takes ONE action
        judge scores the current answer -> blackboard (workers see the gaps next round)
        stop when judge score >= threshold (and no CRITICAL gap), or budget/rounds run out
    finalise: if no answer was ever proposed, one synthesis pass writes one

Reuses ``swarmsolve.blackboard.Blackboard`` (via OpenTaskEnv),
``swarmsolve.eventlog.EventLog``, ``swarmsolve.budget.BudgetTracker`` and
``swarmsolve.env.load_env``. Model calls go through ``swarmsolve.llm.chat``.
"""

from __future__ import annotations

import random
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from swarmsolve.budget import BudgetTracker
from swarmsolve.env import load_env
from swarmsolve.eventlog import EventLog
from swarmsolve.learning import RunMemory
from swarmsolve import prompts, repo
from swarmsolve.judge import Judge
from swarmsolve.llm import LLMError, ModelSpec, Reply, chat, estimate_cost, reset_stub
from swarmsolve.task_env import OpenTaskEnv
from swarmsolve.tools import ToolPolicy, Workspace, read_file, write_file
from swarmsolve.verify import VerificationReport, run_verification

RUNS_ROOT = Path("runs/solve")
DEFAULT_MEMORY_PATH = RUNS_ROOT / "memory.jsonl"
# TOML manifest tools (see swarmsolve/external_tools.py). Every entrypoint
# funnels through solve(), so anchoring the default here means every entrypoint
# picks up the same manifests without each having to wire it in themselves.
DEFAULT_TOOLBOX_DIR = Path(__file__).resolve().parent / "toolbox.d"

# Belt-and-suspenders: this package's OWN repo root (2 parents up from
# swarmsolve/pipeline.py). Never let isolation target this repo, full stop --
# a solve() run whose workspace_dir resolves here could initialize, commit, or
# merge directly onto the tool's own files.
_OWN_REPO_ROOT = Path(__file__).resolve().parent.parent

# Stall detection: a local worker model can fully finish the task (verification:
# real tests passing) but never call propose_answer -- the judge keeps saying
# "done" every round while nothing changes, and the run burns its remaining
# rounds/budget doing nothing. STALL_NUDGE_ROUNDS consecutive (raw_done, no
# answer, unchanged verification) rounds -> the next round's workers get an
# explicit forced directive instead of relying on the softer judge `advice`
# text; STALL_STOP_ROUNDS -> give up on this round and go straight to the
# finalize/synthesis step rather than exhausting cfg.rounds for no further
# chance of a different outcome.
STALL_NUDGE_ROUNDS = 2
STALL_STOP_ROUNDS = 3

_STALL_DIRECTIVE = (
    "SYSTEM: verification has shown this task complete and passing for multiple "
    "rounds with no files changing and no answer submitted. You MUST call "
    "propose_answer THIS turn with the complete final answer -- do not "
    "read_file/list_files/think further."
)

_SURPLUS_WORKER_ROLES = (
    "Run the narrowest relevant import/build/test command and record exact failures.",
    "Trace integration call sites and identify one concrete wiring change not already attempted.",
    "Check configuration and environment examples for the feature; implement one missing piece.",
    "Inspect downstream schema consumers and verify the changed data contract against them.",
    "Review the current diff for correctness and fix one distinct defect using generate_file.",
    "Exercise a real smoke path and record evidence instead of rereading source already inspected.",
)

_MUTATION_ACTIONS = frozenset({"replace_text", "write_file", "generate_file"})
_MUTATION_LANGUAGE = re.compile(
    r"\b(fix|implement|add|remove|replace|rename|edit|change|wire|rewrite|create|update|patch)\b",
    re.IGNORECASE,
)

# Distinct work-item ids do NOT guarantee distinct files -- two different judge
# gaps can both be about the same file (e.g. "mean() doesn't raise ValueError"
# and "stdev() doesn't raise ValueError", both in stats.py). Activating two such
# items in the same round lets two workers each independently full-file-rewrite
# the same file from their own narrow view of just their one gap; the second
# write silently clobbers the first worker's fix. `_item_files` extracts
# filenames an item's own text mentions so the scheduler can keep same-file
# items from co-activating.
_FILE_EXTENSIONS = (
    "py", "pyi", "js", "jsx", "ts", "tsx", "json", "yaml", "yml", "toml", "cfg",
    "ini", "md", "txt", "html", "css", "scss", "c", "h", "cpp", "hpp", "go",
    "rs", "rb", "java", "sh", "ps1", "sql", "xml",
)
_FILE_MENTION_RE = re.compile(
    r"\b[\w][\w./-]*\.(?:" + "|".join(_FILE_EXTENSIONS) + r")\b", re.IGNORECASE
)


def _item_files(text: str) -> frozenset[str]:
    """Filenames mentioned in a work item's own text, lowercased for comparison."""
    return frozenset(m.group(0).lower() for m in _FILE_MENTION_RE.finditer(text))


def _worker_directive_for_item(
    env: OpenTaskEnv, work: dict[str, Any], index: int, count: int,
) -> str:
    text = str(work.get("text") or "").strip()
    item_id = str(work.get("id") or "WORK")
    seen = int(work.get("seen", 1))
    acceptance = str(work.get("acceptance") or "").strip()
    advice = str((env.last_judge or {}).get("advice") or "").strip()
    mutative = bool(_MUTATION_LANGUAGE.search(advice + " " + text))
    advisor = f" Primary advisor action: {advice}." if index == 0 and advice else ""
    control = (
        " MUTATION REQUIRED: diagnosis is over. Your action must be replace_text, write_file, "
        "or generate_file and must directly change the assigned defect. Do not inspect or retest first."
        if mutative else ""
    )
    passing = f" Local passing condition: {acceptance}." if acceptance else ""
    return (
        f"SYSTEM ASSIGNMENT (worker slot {index + 1}/{count}, item {item_id}, observed {seen} cycle(s)): "
        f"{text}.{passing}{advisor}{control}\n"
        "Own this item only. Read the live findings first; if it is already resolved, verify the result "
        "or report that fact rather than restarting it. Do not repeat an inspection or command already "
        "recorded at the current workspace revision. Prefer replace_text for a small known edit; use "
        "generate_file only for a genuinely substantial file."
    )


def _worker_directives(env: OpenTaskEnv, count: int, forced: str | None = None) -> list[str]:
    """Give workers distinct ownership instead of broadcasting one judge hint.

    Broadcasting ``judge.advice`` made homogeneous swarms stampede toward the
    same command. Assign gaps one-per-worker, then give surplus workers distinct
    verification/integration roles. Later workers must check live findings and
    avoid repeating work that an earlier worker completed in this same round.
    """
    if forced:
        return [forced] * count
    judge = env.last_judge or {}
    items: list[tuple[str, int]] = []
    advice = str(judge.get("advice") or "").strip()
    for work in env.open_work_items():
        text = str(work.get("text") or "").strip()
        if text:
            items.append((f"{work.get('id')}: {text}", int(work.get("seen", 1))))
    if not items:
        items.extend((str(x), 1) for x in env.bb.facts.get("plan", []) if str(x).strip())

    directives: list[str] = []
    for index in range(count):
        if index < len(items):
            assignment, seen = items[index]
            work = {"id": f"slot-{index + 1}", "text": assignment, "seen": seen}
        else:
            work = {
                "id": f"role-{index + 1}",
                "text": _SURPLUS_WORKER_ROLES[(index - len(items)) % len(_SURPLUS_WORKER_ROLES)],
                "seen": 1,
            }
        directive = _worker_directive_for_item(env, work, index, count)
        # Broadcast compatibility preserves the original single implementation
        # owner; surplus workers remain reviewers even when advice is mutative.
        if index > 0:
            directive = directive.replace(
                " MUTATION REQUIRED: diagnosis is over. Your action must be replace_text, write_file, "
                "or generate_file and must directly change the assigned defect. Do not inspect or retest first.",
                "",
            )
        directives.append(directive)
    return directives


def _timestamp() -> str:
    """A run-id timestamp, unique even across runs started in the same second
    (matters once it's used as a subprocess argument -- see solve()'s own
    comment on run_dir -- not just as a human-readable label)."""
    return time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns() % 1_000_000:06d}"


@dataclass
class SolveConfig:
    task: str
    workers: list[ModelSpec] = field(default_factory=list)
    judge: ModelSpec | None = None
    planner: ModelSpec | None = None  # None -> no round-0 planning pass (CLI defaults it to workers[0])
    rounds: int = 8
    adaptive_rounds: bool = True  # cfg.rounds is the initial allocation; extend while critical work remains
    max_rounds: int = 16  # hard ceiling for adaptive extension
    adaptive_workers: bool = True  # selectively wake owners instead of broadcasting to the full roster
    max_active_workers: int = 3  # exploration ceiling; mutative work is single-owner
    max_item_attempts: int = 2  # attempts at one workspace revision before changing strategy/stopping
    threshold: float = 0.85
    budget_usd: float = 1.0
    max_tokens: int = 1200
    seed: int = 0
    policy: ToolPolicy = field(default_factory=ToolPolicy)
    workspace_dir: str | Path | None = None
    run_dir: str | Path | None = None
    on_event: Callable[[str, dict], None] | None = None  # progress callback for the CLI
    memory_path: str | Path | None = DEFAULT_MEMORY_PATH  # None/False -> disabled
    judge_fallback_reason: str | None = None  # set by the caller when it substituted the judge model
    cancel: threading.Event | None = None  # set to request a stop at the next round boundary
    isolate: bool = True  # run in a git worktree/branch when possible, instead of writing to workspace_dir directly
    auto_merge: bool = True  # on success, merge the isolated branch into whatever's checked out (see repo.py)
    build_cmd: str | None = None  # override auto-detection in verify.detect_commands
    test_cmd: str | None = None   # override auto-detection in verify.detect_commands
    toolbox_dir: str | Path | None = DEFAULT_TOOLBOX_DIR  # None -> skip TOML manifest loading

    @classmethod
    def for_embedding(cls, task: str, root_dir: str | Path, **kwargs: Any) -> "SolveConfig":
        """Convenience constructor for embedding ``solve()`` into another
        application. Every path this pipeline would otherwise default to a
        cwd-relative ``runs/solve/...`` (this run's own directory, cross-run
        memory) is anchored under ``root_dir`` instead -- so the host
        application's working directory can be anything, and multiple embedded
        call sites/processes sharing one ``root_dir`` don't collide with a
        plain CLI run's own ``runs/solve/``.

        Everything else (``workers``, ``judge``, ``policy``, ...) passes
        through as ordinary keyword arguments; passing ``run_dir`` or
        ``memory_path`` explicitly overrides just that one default instead of
        the whole anchoring.
        """
        root = Path(root_dir).resolve()
        kwargs.setdefault("run_dir", root / "runs" / _timestamp())
        kwargs.setdefault("memory_path", root / "memory.jsonl")
        return cls(task=task, **kwargs)


@dataclass
class SolveResult:
    answer: str
    score: float
    rounds: int
    success: bool
    cost: dict[str, float]
    run_dir: Path
    verdict: str = ""
    gaps: list[str] = field(default_factory=list)
    audit_verified: bool = False
    judge_fallback_reason: str | None = None
    cancelled: bool = False
    stalled: bool = False  # stopped early: judge kept saying done, no answer ever came, no progress
    verification: VerificationReport | None = None  # last round's report
    branch: str | None = None  # set when the run was isolated onto its own git branch
    merged: bool = False
    merge_error: str | None = None
    checkpoint_sha: str | None = None
    checkpoint_error: str | None = None


def _notify(cfg: SolveConfig, kind: str, data: dict) -> None:
    if cfg.on_event:
        try:
            cfg.on_event(kind, data)
        except Exception:  # a noisy callback must not break a run
            pass


def _charge(budget: BudgetTracker, reply: Reply) -> None:
    budget.charge_step()
    budget.charge_tokens(reply.tokens_in, reply.tokens_out, reply.usd)


def _remaining_budget(budget: BudgetTracker) -> float | None:
    """USD headroom left, or None if the run has no $ cap (``max_usd`` == 0 = unlimited)."""
    return max(0.0, budget.max_usd - budget.usd) if budget.max_usd else None


def _would_exceed_budget(budget: BudgetTracker, spec: ModelSpec, system: str, user: str, max_tokens: int) -> bool:
    remaining = _remaining_budget(budget)
    return remaining is not None and estimate_cost(spec, system, user, max_tokens) > remaining


def _is_cancelled(cfg: SolveConfig) -> bool:
    return cfg.cancel is not None and cfg.cancel.is_set()


def _blackboard_snapshot(env: OpenTaskEnv, rnd: int) -> dict:
    """Compact projection of the shared blackboard for a live spectator.

    Keeps the payload bounded even on runs with hundreds of findings by
    truncating each field to a display-oriented cap. The audit trail already
    has the full state in ``events.jsonl`` -- this event is only for a spectator.
    """
    facts = env.bb.facts
    findings = facts.get("findings", [])
    answer = facts.get("answer", "") or ""
    # Skip environmental noise: .git/ appears in every isolated-worktree run
    # (dozens of files a viewer never wanted to see), and the usual tooling dirs
    # would drown a real project's source under sheer count. The workspace still
    # has these files -- we just don't put them in a "here's what the swarm did"
    # spectator pane.
    files = [
        f for f in env.workspace.list_files()
        if not (f.startswith((".git/", ".git\\", ".venv/", ".venv\\", "node_modules/", "node_modules\\", "__pycache__/", "__pycache__\\"))
                or "/.git/" in f or "\\.git\\" in f
                or "/__pycache__/" in f or "\\__pycache__\\" in f
                or "/node_modules/" in f or "\\node_modules\\" in f)
    ][:60]
    return {
        "round": rnd,
        "plan": list(facts.get("plan", [])),
        "criteria": list(facts.get("criteria", [])),
        "n_findings": len(findings),
        "recent_findings": findings[-12:],  # tail: what the workers are seeing NOW
        "answer": answer[:4000],  # cap: a wall-of-text draft never floods the event stream
        "answer_chars": len(answer),
        "files": files,
        "work_items": [
            {
                "id": item.get("id"), "text": str(item.get("text", ""))[:240],
                "status": item.get("status"), "owner": item.get("owner"),
                "outcome": item.get("outcome"), "next_action": str(item.get("next_action", ""))[:240],
                "evidence": [str(value)[:240] for value in item.get("evidence", [])[-2:]],
                "lessons_learned": [str(value)[:240] for value in item.get("lessons_learned", [])[-2:]],
            }
            for item in facts.get("work_items", [])[-20:]
        ],
    }


@dataclass
class _Isolation:
    workspace_root: Path
    base_sha: str | None
    worktree: repo.WorktreeInfo | None  # set only when a real repo worktree was created (mergeable)
    note: str | None = None  # surfaced as a "warn" notification when isolation degraded


def _setup_isolation(cfg: SolveConfig, run_id: str, run_dir: Path) -> _Isolation:
    """Where the run's files actually live, and whether that's a git-backed,
    isolated worktree. The worktree (when one is created) always lives at
    ``run_dir/workspace`` -- ``cfg.workspace_dir``, when given, names which
    existing repo/directory to isolate FROM, not literally where to write.
    """
    default_ws = run_dir / "workspace"
    if not cfg.isolate or not repo.git_available():
        note = "git not found -- isolation/verification-diff/auto-merge disabled" if cfg.isolate else None
        return _Isolation(Path(cfg.workspace_dir) if cfg.workspace_dir else default_ws, None, None, note)

    if cfg.workspace_dir:
        base_dir = Path(cfg.workspace_dir).resolve()
        # Check the literal target before asking Git for a root. If this source
        # tree has no .git yet, repo_root() returns None; checking only that
        # would let a fresh directory slip past the safety guard below.
        if base_dir == _OWN_REPO_ROOT.resolve():
            raise ValueError(
                f"refusing to run solve() with workspace_dir resolving to this package's own "
                f"repository/source tree ({_OWN_REPO_ROOT}) -- this can initialize, commit, or merge directly "
                f"onto the tool's files; point --workspace at a different project, or pass "
                f"isolate=False if writing here is genuinely intended."
            )
        root = repo.repo_root(base_dir)
        if root is not None and root.resolve() == _OWN_REPO_ROOT.resolve():
            raise ValueError(
                f"refusing to run solve() with workspace_dir resolving to this package's own "
                f"repository ({_OWN_REPO_ROOT}) -- this can commit/merge directly onto its real "
                f"branch; point --workspace at a different project, or pass isolate=False if "
                f"writing here is genuinely intended."
            )
        if root is not None:
            try:
                wt = repo.create_run_worktree(root, default_ws, run_id)
                return _Isolation(wt.workspace, wt.base_sha, wt)
            except RuntimeError as e:
                return _Isolation(base_dir, None, None,
                                  f"could not create an isolated worktree ({e}); writing directly to {base_dir}")
        # Existing plain directory: create a local baseline commit, then use the
        # exact same isolated branch/worktree flow as an established repository.
        # A successful run is committed in the worktree and merged back locally;
        # no remote is created or contacted.
        base_dir.mkdir(parents=True, exist_ok=True)
        base_sha = repo.init_fresh_repo(base_dir)
        if base_sha:
            try:
                wt = repo.create_run_worktree(base_dir, default_ws, run_id)
                return _Isolation(wt.workspace, wt.base_sha, wt,
                                  "initialized a local git repository and baseline commit")
            except RuntimeError as e:
                return _Isolation(base_dir, base_sha, None,
                                  f"initialized local git, but could not create an isolated worktree ({e}); "
                                  f"writing directly to {base_dir}")
        return _Isolation(base_dir, None, None,
                          "git init failed; verification will run without a diff base")

    base_sha = repo.init_fresh_repo(default_ws)
    return _Isolation(default_ws, base_sha, None,
                      None if base_sha else "git init failed; verification will run without a diff base")


def _plan(
    cfg: SolveConfig, spec: ModelSpec, budget: BudgetTracker, prior_runs: list[dict] | None = None
) -> tuple[list[Any], list[str], Reply | None]:
    system = prompts.planner_system()
    user = f'TASK:\n{cfg.task}'
    if prior_runs:
        hints = "\n".join(f"- {r.get('task', '')[:120]} -> {str(r.get('metadata', {}))[:150]}" for r in prior_runs)
        user += f"\n\nRELATED PRIOR RUNS (context only, may or may not still apply):\n{hints}"
    if _would_exceed_budget(budget, spec, system, user, min(cfg.max_tokens, 800)):
        _notify(cfg, "warn", {"msg": "planner skipped: insufficient budget remaining"})
        return [], [], None
    try:
        reply = chat(spec, system, user, max_tokens=min(cfg.max_tokens, 800), temperature=0.3)
    except LLMError as e:
        _notify(cfg, "warn", {"msg": f"planner failed ({e}); continuing without a plan"})
        return [], [], None
    parsed = prompts.parse_json_reply(reply.text) or {}
    # Keep structured planner nodes intact so OpenTaskEnv can build a dependency
    # graph.  Legacy planners returning strings remain fully supported.
    plan = [s for s in (parsed.get("plan") or []) if isinstance(s, (str, dict))][:10]
    crit = [str(s) for s in (parsed.get("acceptance_criteria") or [])][:6]
    return plan, crit, reply


def _eligible_work_items(env: OpenTaskEnv, max_attempts: int) -> list[dict[str, Any]]:
    """Ready agenda items which have not exhausted this revision's attempts."""
    items = env.ready_work_items()
    eligible = [
        item for item in items
        if item.get("attempt_revision") != env.mutation_revision
        or int(item.get("revision_attempts", 0)) < max(1, max_attempts)
    ]
    return sorted(
        eligible,
        key=lambda item: (
            not bool(item.get("critical")),
            int(item.get("revision_attempts", 0)),
            -int(item.get("seen", 1)),
            str(item.get("id", "")),
        ),
    )


def _schedule_worker_turns(
    cfg: SolveConfig,
    env: OpenTaskEnv,
    roster: list[tuple[str, ModelSpec]],
    rnd: int,
    rng: random.Random,
    forced_directive: str | None = None,
) -> tuple[list[tuple[str, ModelSpec, str, str | None]], dict[str, Any]]:
    """Select only agents that can make distinct progress in this cycle.

    Cycle one permits a bounded exploratory cohort when no plan exists, keeping
    the old useful breadth without waking an arbitrary eight-agent roster.
    Once concrete work exists, each ready item gets its own single owner -- a
    worker never shares an item with another worker in the same round. Several
    DIFFERENT mutative items may run in the same round only when they don't
    name the same file (see `_item_files`); items that collide on a file are
    serialized across rounds instead of racing full-file rewrites against
    each other.
    """
    order = list(roster)
    rng.shuffle(order)
    if not cfg.adaptive_workers:
        directives = _worker_directives(env, len(order), forced_directive)
        return [(*worker, directives[i], None) for i, worker in enumerate(order)], {
            "mode": "broadcast", "reason": "adaptive worker routing disabled",
            "active": [wid for wid, _ in order], "work_items": [],
        }

    if forced_directive:
        selected = order[:1]
        return [(*selected[0], forced_directive, None)] if selected else [], {
            "mode": "adaptive", "reason": "stalled-answer synthesis nudge",
            "active": [wid for wid, _ in selected], "work_items": [],
        }

    ready = _eligible_work_items(env, cfg.max_item_attempts)
    exhausted = env.ready_work_items() and not ready
    if exhausted:
        return [], {
            "mode": "adaptive", "reason": "ready work exhausted its revision-local attempt limit",
            "active": [], "work_items": [item.get("id") for item in env.ready_work_items()],
        }
    if env.pending_work_items() and not ready:
        pending = env.pending_work_items()
        return [], {
            "mode": "adaptive", "reason": "no selectable work; remaining items are blocked or awaiting dependencies",
            "active": [], "work_items": [item.get("id") for item in pending],
        }

    if ready:
        # Mutativeness is judged per item, from that item's OWN text only -- not the
        # single shared judge.advice string appended to every item. Appending a global
        # "create/fix ..." advice to each item's check meant one judge sentence made
        # EVERY ready item test positive, collapsing the round to one worker even when
        # several genuinely independent, distinctly-owned items (e.g. three unrelated
        # files) were ready at once. `claim_work_item` leases each item to exactly one
        # owner, preventing two workers sharing the SAME item -- but not two DIFFERENT
        # items that happen to name the same file, which really does collide (see
        # `_item_files`'s comment). Greedily pick ready items up to the worker cap,
        # skipping any that name a file an already-picked item this round also names;
        # a skipped item simply stays ready for a later round instead of co-activating
        # unsafely.
        active_cap = min(max(1, cfg.max_active_workers), len(ready), len(order))
        assignments: list[dict[str, Any]] = []
        claimed_files: set[str] = set()
        for item in ready:
            if len(assignments) >= active_cap:
                break
            files = _item_files(str(item.get("text", "")))
            if files and claimed_files & files:
                continue  # collides with a file an already-selected item this round names
            assignments.append(item)
            claimed_files |= files
        active_count = len(assignments)
        selected = order[:active_count]
        directives: list[str] = []
        for index, ((wid, _spec), item) in enumerate(zip(selected, assignments)):
            claimed = env.claim_work_item(str(item.get("id")), wid, rnd) or item
            directives.append(_worker_directive_for_item(env, claimed, index, active_count))
        any_mutative = any(_MUTATION_LANGUAGE.search(str(item.get("text", ""))) for item in assignments)
        if any_mutative:
            reason = "parallel implementation" if active_count > 1 else "single-writer implementation"
        else:
            reason = "independent ready work"
        return [(*worker, directives[i], str(assignments[i].get("id"))) for i, worker in enumerate(selected)], {
            "mode": "adaptive", "reason": reason,
            "active": [wid for wid, _ in selected],
            "work_items": [item.get("id") for item in assignments],
        }

    # No agenda yet: perform bounded discovery on the first cycle.  Later
    # cycles use one agent because repeating broad reconnaissance is precisely
    # the failure mode this scheduler is meant to prevent.
    active_count = min(len(order), max(1, cfg.max_active_workers) if rnd == 1 else 1)
    selected = order[:active_count]
    directives = _worker_directives(env, active_count)
    return [(*worker, directives[i], None) for i, worker in enumerate(selected)], {
        "mode": "adaptive", "reason": "bounded initial exploration" if rnd == 1 else "focused continuation",
        "active": [wid for wid, _ in selected], "work_items": [],
    }


def _worker_turn(
    cfg: SolveConfig, worker_id: str, spec: ModelSpec, env: OpenTaskEnv, rnd: int, budget: BudgetTracker,
    forced_directive: str | None = None,
) -> dict[str, Any]:
    obs = env.observe()
    view = prompts.render_workspace(
        task=cfg.task,
        facts=obs["facts"],
        files=obs["files"],
        judge_feedback=obs["judge"],
        round_index=rnd,
    )
    if forced_directive:
        view = view + "\n\n" + forced_directive
    wsys = prompts.worker_system(workspace_root=str(env.workspace.root))
    if _would_exceed_budget(budget, spec, wsys, view, cfg.max_tokens):
        return {"agent": worker_id, "action": "error", "ok": False,
                "detail": "skipped: insufficient budget remaining"}
    try:
        reply = chat(spec, wsys, view, max_tokens=cfg.max_tokens, temperature=0.5)
    except LLMError as e:
        _notify(cfg, "warn", {"msg": f"{worker_id} ({spec}) call failed: {e}"})
        return {"agent": worker_id, "action": "error", "ok": False, "detail": str(e)}
    _charge(budget, reply)

    name, args, note, outcome = prompts.parse_worker_reply(reply.text)
    if name == "think" and note == "unparseable -> think":
        # One corrective reprompt. In real runs the dominant failure is a large
        # write_file object cut off at the output ceiling, not a model forgetting
        # what JSON looks like. Repeating the generic instruction made DeepSeek
        # regenerate the same truncated file and doubled both latency and spend.
        looks_like_write = '"write_file"' in reply.text[:240]
        if looks_like_write:
            correction = (
                "Your previous write_file JSON was truncated because it was too large. "
                "Do NOT repeat the full file. Return one compact valid JSON object under "
                "700 tokens. Put parameters inside args. Either write a SMALL complete "
                "implementation (content under 2000 characters), or choose a different "
                "small action that advances the task. Never return partial file content."
            )
        else:
            correction = (
                "Your previous reply was not valid JSON. Return exactly one compact JSON "
                "object under 700 tokens, with all action parameters nested inside args."
            )
        reprompt = view + "\n\nSYSTEM CORRECTION: " + correction
        retry_max_tokens = min(cfg.max_tokens, 900)
        if not _would_exceed_budget(budget, spec, wsys, reprompt, retry_max_tokens):
            try:
                reply2 = chat(spec, wsys, reprompt, max_tokens=retry_max_tokens, temperature=0.1)
                _charge(budget, reply2)
                name, args, note, outcome = prompts.parse_worker_reply(reply2.text)
            except LLMError:
                pass

    if name == "think" and note == "unparseable -> think":
        return {
            "agent": worker_id,
            "action": "error",
            "ok": False,
            "detail": "worker reply remained unparseable after a compact retry (likely truncated output)",
            "note": "unparseable after retry",
            "model": str(spec),
        }

    if forced_directive and "MUTATION REQUIRED:" in forced_directive and name not in _MUTATION_ACTIONS:
        correction = (
            "Your assigned gap requires implementation, but you selected a non-mutating action. "
            "Return exactly one compact JSON action using replace_text for a targeted exact-block change, "
            "write_file for a small complete file, or generate_file for a substantial rewrite. "
            "Do not inspect, explain, or test in this turn."
        )
        reprompt = view + "\n\nSYSTEM CONTROL FAILURE: " + correction
        retry_max_tokens = min(cfg.max_tokens, 900)
        if not _would_exceed_budget(budget, spec, wsys, reprompt, retry_max_tokens):
            try:
                reply2 = chat(spec, wsys, reprompt, max_tokens=retry_max_tokens, temperature=0.1)
                _charge(budget, reply2)
                name, args, note, outcome = prompts.parse_worker_reply(reply2.text)
            except LLMError:
                pass
        if name not in _MUTATION_ACTIONS:
            return {
                "agent": worker_id, "action": "error", "ok": False,
                "detail": "implementation owner refused/failed the mandatory mutation action",
                "note": note, "model": str(spec),
            }

    if name == "generate_file":
        rec = _generate_file_turn(cfg, worker_id, spec, env, rnd, budget, args, note)
        rec["work_outcome"] = outcome
        return rec

    fingerprint = _action_fingerprint(name, args)
    if fingerprint and not env.claim_action(rnd, fingerprint):
        return {
            "agent": worker_id,
            "action": name,
            "ok": False,
            "detail": (
                "duplicate action blocked: another worker already ran the same inspection/test "
                "this round; follow your distinct assignment"
            ),
            "note": note,
            "model": str(spec),
        }

    rec = env.apply(worker_id, name, args, rnd)
    rec["note"] = note
    rec["model"] = str(spec)
    rec["work_outcome"] = outcome
    return rec


def _action_fingerprint(name: str, args: dict[str, Any]) -> str | None:
    """Identify repeated read/test actions without conflating real edits.

    PowerShell presentation suffixes do not make a command a different test.
    A smoke test in particular can appear several times in one round, differing
    only by ``Out-String -Width``.
    """
    if name == "shell":
        cmd = " ".join(str(args.get("cmd", "")).lower().split())
        cmd = re.sub(r"\s+2>&1\s*\|\s*out-string(?:\s+-width\s+\d+)?\s*$", "", cmd)
        return f"shell:{cmd}" if cmd else None
    if name == "read_file":
        path = str(args.get("path", "")).replace("\\", "/").strip().lower()
        return f"read_file:{path}" if path else None
    if name in {"fetch_url", "web_search", "list_files"}:
        stable = repr(sorted((str(k), repr(v)) for k, v in args.items()))
        return f"{name}:{stable}"
    return None


_FILE_EOF = "<<SWARMSOLVE_EOF>>"
_MAX_FILE_CHUNKS = 8


def _strip_outer_code_fence(text: str) -> str:
    text = text.strip()
    if not text.startswith("```"):
        return text
    first_newline = text.find("\n")
    if first_newline < 0:
        return text
    body = text[first_newline + 1:]
    if body.rstrip().endswith("```"):
        body = body.rstrip()[:-3]
    return body.strip("\n")


def _generate_file_turn(
    cfg: SolveConfig,
    worker_id: str,
    spec: ModelSpec,
    env: OpenTaskEnv,
    rnd: int,
    budget: BudgetTracker,
    args: dict[str, Any],
    note: str,
) -> dict[str, Any]:
    """Generate a complete file outside the worker action's JSON envelope.

    Source code is raw text and may span several model responses. Nothing is
    written until an explicit EOF marker (or a provider stop reason) proves the
    generation completed, so a token cutoff can never leave a partial file.
    """
    path = str(args.get("path", "")).strip()
    instructions = str(args.get("instructions", "")).strip()
    if not path or not instructions:
        return {"agent": worker_id, "action": "generate_file", "ok": False,
                "detail": "generate_file requires path and instructions", "note": note,
                "model": str(spec)}

    generated_this_round = env.generated_files_by_round.setdefault(rnd, set())
    if path in generated_this_round:
        return {"agent": worker_id, "action": "generate_file", "ok": False,
                "detail": f"{path} was already generated successfully this round; choose a different unresolved gap",
                "note": note, "model": str(spec)}

    existing_result = read_file(env.workspace, path, max_chars=30_000)
    existing = existing_result.output if existing_result.ok else "(new file)"
    system = (
        "You generate one complete workspace file. Return ONLY the literal file contents: "
        "no Markdown fence, commentary, JSON, or filename. On a final line append "
        f"{_FILE_EOF}. If the response limit is reached before completion, stop cleanly; "
        "the caller will request the next chunk. Preserve behavior not targeted by the request. "
        "Implement the smallest maintainable solution that satisfies the requirements; avoid "
        "speculative abstractions, duplicate helpers, and oversized framework-style modules."
    )
    base_request = (
        f"USER TASK:\n{cfg.task}\n\nFILE: {path}\n\nREQUIREMENTS:\n{instructions}\n\n"
        f"EXISTING FILE (or marker for a new file):\n{existing}"
    )
    chunks: list[str] = []
    chunk_tokens = max(2048, cfg.max_tokens)

    for chunk_index in range(_MAX_FILE_CHUNKS):
        if chunk_index == 0:
            user = base_request
        else:
            tail = "".join(chunks)[-4000:]
            user = (
                base_request + "\n\nCONTINUATION REQUEST:\nContinue exactly where the generated file "
                "stopped. Do not repeat text already emitted. Here is the trailing text already "
                f"received:\n---\n{tail}\n---\nEnd with {_FILE_EOF} only when the file is complete."
            )
        if _would_exceed_budget(budget, spec, system, user, chunk_tokens):
            return {"agent": worker_id, "action": "generate_file", "ok": False,
                    "detail": "file generation skipped: insufficient budget remaining",
                    "note": note, "model": str(spec)}
        try:
            generated = chat(spec, system, user, max_tokens=chunk_tokens,
                             temperature=0.15, want_json=False)
        except LLMError as e:
            return {"agent": worker_id, "action": "generate_file", "ok": False,
                    "detail": f"file generation failed: {e}", "note": note,
                    "model": str(spec)}
        _charge(budget, generated)
        text = generated.text
        if _FILE_EOF in text:
            chunks.append(text.split(_FILE_EOF, 1)[0])
            break
        chunks.append(text)
        # Providers report a normal stop when they intentionally completed the
        # response. Accept it even if a model forgot the requested marker; only
        # length/max-token stops require a continuation.
        reason = (generated.finish_reason or "").lower()
        if reason and reason not in {"length", "max_tokens", "max_token", "max_tokens_reached"}:
            break
    else:
        return {"agent": worker_id, "action": "generate_file", "ok": False,
                "detail": f"file generation did not complete after {_MAX_FILE_CHUNKS} chunks; nothing written",
                "note": note, "model": str(spec)}

    content = _strip_outer_code_fence("".join(chunks))
    if not content.strip():
        return {"agent": worker_id, "action": "generate_file", "ok": False,
                "detail": "file generation returned empty content; nothing written",
                "note": note, "model": str(spec)}
    if Path(path).suffix.lower() == ".py":
        try:
            compile(content, path, "exec")
        except SyntaxError as e:
            location = f"line {e.lineno}" + (f", column {e.offset}" if e.offset else "")
            return {"agent": worker_id, "action": "generate_file", "ok": False,
                    "detail": f"generated Python failed syntax validation at {location}: {e.msg}; "
                              "existing file was left unchanged",
                    "note": note, "model": str(spec)}
    result = write_file(env.workspace, path, content, env.policy)
    if result.ok:
        generated_this_round.add(path)
        if (result.meta or {}).get("changed", True):
            env.mark_mutation()
    env.add_finding(worker_id, f"[generate {path}] {result.short(800)}", rnd)
    return {"agent": worker_id, "action": "generate_file", "ok": result.ok,
            "detail": result.short(300), "note": note, "model": str(spec)}


def solve(cfg: SolveConfig) -> SolveResult:
    load_env()
    reset_stub()
    if not cfg.workers:
        raise ValueError("SolveConfig.workers is empty")

    # Register any TOML manifest tools -- idempotent, so re-runs in the same process
    # don't collide with themselves. Any parse error becomes a warn event once the
    # log is up, below.
    manifest_errors: list[str] = []
    if cfg.toolbox_dir:
        from swarmsolve.external_tools import load_manifests
        _, manifest_errors = load_manifests(cfg.toolbox_dir)
    judge_spec = cfg.judge or cfg.workers[0]
    judge_fallback_reason = cfg.judge_fallback_reason or (
        None if cfg.judge is not None else "no judge configured; falling back to the first worker"
    )
    planner_spec = cfg.planner  # None -> skip the planning round entirely

    ts = _timestamp()
    # .resolve() is not cosmetic: run_dir gets passed to `git worktree add` as the
    # worktree's destination path, run as a subprocess with cwd=<the target repo> --
    # a relative path would resolve against THAT cwd, nesting the worktree (and its
    # writes) inside the very repo it's supposed to isolate from.
    run_dir = (Path(cfg.run_dir) if cfg.run_dir else RUNS_ROOT / ts).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    isolation = _setup_isolation(cfg, run_id=ts, run_dir=run_dir)
    workspace = Workspace(isolation.workspace_root)

    budget = BudgetTracker(max_usd=cfg.budget_usd, max_agent_steps=0, max_tokens=0)
    rng = random.Random(cfg.seed)
    env = OpenTaskEnv(cfg.task, workspace, cfg.policy, threshold=cfg.threshold)
    judge = Judge(judge_spec)
    memory = RunMemory(Path(cfg.memory_path)) if cfg.memory_path else None

    worker_ids = [f"w{i}" for i in range(len(cfg.workers))]

    with EventLog(run_dir / "events.jsonl") as log:
        log.emit(
            "episode_start", env=env.name, task=cfg.task,
            workers=[str(s) for s in cfg.workers], judge=str(judge_spec), planner=str(planner_spec),
            rounds=cfg.rounds, threshold=cfg.threshold, budget_usd=cfg.budget_usd,
            adaptive_workers=cfg.adaptive_workers, max_active_workers=cfg.max_active_workers,
            max_item_attempts=cfg.max_item_attempts,
            policy=cfg.policy.describe(), workspace=str(workspace.root), seed=cfg.seed,
            judge_fallback=judge_fallback_reason,
            branch=isolation.worktree.branch if isolation.worktree else None,
        )
        _notify(cfg, "start", {"run_dir": str(run_dir), "workspace": str(workspace.root),
                               "policy": cfg.policy.describe(),
                               "branch": isolation.worktree.branch if isolation.worktree else None})
        if judge_fallback_reason:
            # judge independence matters precisely when the configured judge isn't
            # available -- this must reach the audit trail and the operator, not just
            # an easily-missed stderr line, and must never be silenced by --quiet.
            _notify(cfg, "warn", {"msg": f"judge fallback: {judge_fallback_reason} (judge={judge_spec})"})
        if isolation.note:
            _notify(cfg, "warn", {"msg": isolation.note})
        for err in manifest_errors:
            _notify(cfg, "warn", {"msg": f"toolbox: {err}"})

        prior_runs = memory.recall(cfg.task, limit=3) if memory is not None else []
        if memory is not None:
            log.emit("memory_recall", task=cfg.task, n_prior_runs=len(prior_runs))

        # -- round 0: plan --------------------------------------------------
        if planner_spec is not None:
            plan, crit, preply = _plan(cfg, planner_spec, budget, prior_runs)
            if preply is not None:
                _charge(budget, preply)
            env.set_plan(plan, crit)
            log.emit("plan", plan=plan, acceptance_criteria=crit, cost=budget.as_dict())
            _notify(cfg, "plan", {"plan": env.bb.facts.get("plan", []), "criteria": crit})
            _notify(cfg, "blackboard", _blackboard_snapshot(env, 0))

        # -- adaptive rounds ---------------------------------------------
        rounds_done = 0
        cancelled = False
        stalled = False
        stall_count = 0
        last_verify_sig: tuple | None = None
        previous_score = 0.0
        last_report: VerificationReport | None = None
        round_limit = max(1, cfg.rounds)
        hard_round_limit = max(round_limit, cfg.max_rounds) if cfg.adaptive_rounds else round_limit
        rnd = 1
        while rnd <= round_limit:
            if _is_cancelled(cfg):
                cancelled = True
                _notify(cfg, "warn", {"msg": "cancelled; stopping"})
                break
            if budget.exhausted():
                _notify(cfg, "warn", {"msg": f"budget cap ${cfg.budget_usd} reached; stopping"})
                break
            forced_directive = _STALL_DIRECTIVE if stall_count >= STALL_NUDGE_ROUNDS else None
            round_start_revision = env.mutation_revision
            round_start_answer = env.answer
            expired = env.expire_leases(rnd)
            for item_id in expired:
                transition = {"id": item_id, "status": "ready", "outcome": "lease_expired"}
                log.emit("work_item", round=rnd, **transition)
                _notify(cfg, "work_item", {"round": rnd, **transition})
            scheduled, schedule = _schedule_worker_turns(
                cfg, env, list(zip(worker_ids, cfg.workers)), rnd, rng, forced_directive
            )
            log.emit("schedule", round=rnd, **schedule)
            _notify(cfg, "schedule", {"round": rnd, **schedule})
            if not scheduled:
                stalled = True
                msg = f"convergence stalled: {schedule.get('reason', 'no selectable work')}"
                _notify(cfg, "warn", {"msg": msg})
                break
            for wid, spec, directive, item_id in scheduled:
                if _is_cancelled(cfg):
                    cancelled = True
                    break
                if budget.exhausted():
                    break
                rec = _worker_turn(cfg, wid, spec, env, rnd, budget,
                                   forced_directive=directive)
                log.emit("step", round=rnd, **rec, cost=budget.as_dict())
                _notify(cfg, "step", {"round": rnd, **rec})
                if item_id:
                    handoff = rec.get("work_outcome") or {}
                    reported = str(handoff.get("work_status") or "")
                    if not reported:
                        # Old models remain compatible: a successful mutation is
                        # treated as completed-but-unverified; everything else is
                        # an incomplete handoff rather than an invented success.
                        reported = "completed" if rec.get("ok") and rec.get("action") in _MUTATION_ACTIONS else "incomplete"
                    evidence = list(handoff.get("evidence") or [])
                    if rec.get("detail"):
                        evidence.append(str(rec["detail"])[:500])
                    updated = env.record_work_outcome(
                        item_id, wid, reported,
                        evidence=evidence,
                        lessons_learned=list(handoff.get("lessons_learned") or []),
                        next_action=str(handoff.get("next_action") or ""),
                    )
                    if updated:
                        transition = {
                            "id": item_id, "owner": wid, "status": updated.get("status"),
                            "outcome": updated.get("outcome"),
                            "next_action": updated.get("next_action", ""),
                        }
                        log.emit("work_item", round=rnd, **transition)
                        _notify(cfg, "work_item", {"round": rnd, **transition})
                # Live blackboard snapshot for anything watching: a small, cheap
                # projection of env.observe() -- files list, current answer,
                # counted findings, and the latest few finding lines. The full
                # blackboard is what the workers read next turn; this event just
                # lets a spectator see it evolve.
                _notify(cfg, "blackboard", _blackboard_snapshot(env, rnd))
            if cancelled:
                _notify(cfg, "warn", {"msg": "cancelled; stopping"})
                break

            report = run_verification(workspace, cfg.policy, base_sha=isolation.base_sha,
                                      build_cmd=cfg.build_cmd, test_cmd=cfg.test_cmd)
            last_report = report
            log.emit("verify", round=rnd, files_changed=report.files_changed,
                     files_changed_known=report.files_changed_known,
                     build_cmd=report.build_cmd, build_ok=report.build_ok,
                     test_cmd=report.test_cmd, test_ok=report.test_ok,
                     build_output=report.build_output, test_output=report.test_output)
            _notify(cfg, "verify", {"round": rnd, "summary": report.summary(), "has_failure": report.has_failure})
            verification_transitions = env.apply_verification(report)
            for transition in verification_transitions:
                log.emit("work_item", round=rnd, outcome="verification", **transition)
                _notify(cfg, "work_item", {"round": rnd, "outcome": "verification", **transition})
            if verification_transitions:
                _notify(cfg, "blackboard", _blackboard_snapshot(env, rnd))

            jr = judge.assess(
                task=cfg.task,
                criteria=env.bb.facts.get("criteria", []),
                answer=env.answer,
                workspace_summary=workspace.summary(),
                findings_tail=[f["text"] for f in env.bb.facts.get("findings", [])[-12:]],
                remaining_budget_usd=_remaining_budget(budget),
                verification=report,
            )
            if jr.get("_usage"):
                for _ in range(max(1, int(jr.get("_calls", 1)))):
                    budget.charge_step()
                budget.charge_tokens(int(jr["_usage"]["tokens_in"]), int(jr["_usage"]["tokens_out"]),
                                     float(jr["_usage"]["usd"]))
            env.set_judge(jr)
            rounds_done = rnd
            log.emit("judge", round=rnd, score=jr["score"], verdict=jr["verdict"], gaps=jr["gaps"],
                     done=jr["done"], raw_done=jr.get("raw_done", False), advice=jr.get("advice", ""),
                     raw=jr.get("_raw", ""), cost=budget.as_dict())
            log.emit("round_end", round=rnd, score=env.score(), snapshot=env.snapshot())
            _notify(cfg, "judge", {"round": rnd, "score": jr["score"], "verdict": jr["verdict"],
                                   "gaps": jr["gaps"], "advice": jr.get("advice", ""),
                                   "usd": budget.as_dict()["usd"]})

            # stall detection: the judge keeps claiming the work is done, but nothing is
            # actually progressing toward a submitted answer -- see STALL_NUDGE_ROUNDS'
            # comment above.
            verify_sig = (tuple(sorted(report.files_changed)), report.build_ok, report.test_ok)
            tangible_progress = (
                env.mutation_revision > round_start_revision
                or env.answer != round_start_answer
                or env.score() > previous_score
                or (last_verify_sig is not None and verify_sig != last_verify_sig
                    and (report.build_ok is True or report.test_ok is True))
            )
            if jr.get("raw_done") and not env.answer.strip() and verify_sig == last_verify_sig:
                stall_count += 1
            else:
                stall_count = 0
            last_verify_sig = verify_sig
            previous_score = max(previous_score, env.score())

            if env.goal_satisfied():
                break
            if stall_count >= STALL_STOP_ROUNDS:
                stalled = True
                _notify(cfg, "warn", {
                    "msg": f"stalled: judge has reported done for {stall_count} rounds with no answer "
                          f"proposed and no progress (even after a forced directive); stopping early "
                          f"rather than exhausting the remaining rounds",
                })
                break

            if (rnd >= round_limit and cfg.adaptive_rounds and rnd < hard_round_limit
                    and env.critical_gaps() and not budget.exhausted()
                    and jr.get("_valid", True) and tangible_progress):
                old_limit = round_limit
                round_limit = min(hard_round_limit, round_limit + 2)
                msg = (f"adaptive rounds extended {old_limit} → {round_limit}: "
                       f"{len(env.critical_gaps())} critical gap(s) remain and budget is available")
                log.emit("round_extension", round=rnd, old_limit=old_limit,
                         new_limit=round_limit, critical_gaps=env.critical_gaps())
                _notify(cfg, "warn", {"msg": msg})
            elif (rnd >= round_limit and cfg.adaptive_rounds and rnd < hard_round_limit
                  and env.critical_gaps() and not budget.exhausted()):
                reason = ("judge result was invalid" if not jr.get("_valid", True)
                          else "no tangible progress in the final allocated round")
                log.emit("round_extension_refused", round=rnd, old_limit=round_limit,
                         reason=reason, critical_gaps=env.critical_gaps())
                _notify(cfg, "warn", {"msg": f"adaptive extension refused: {reason}"})
            rnd += 1

        # -- finalise -----------------------------------------------------
        if not env.answer.strip() and not cancelled:
            _notify(cfg, "warn", {"msg": "no answer proposed; running one synthesis pass"})
            synthesis_system = (
                "Synthesise the best possible answer to the user's task from the findings below. "
                "Reply with ONLY the answer text, no JSON. "
                "Ground every claim in something that actually appears in the findings -- a command "
                "you can point to, a fetched response, an observed output. Do NOT state anything as "
                "fact that was not actually run/tested/fetched; if a plausible next step was never "
                "attempted, label it explicitly as 'NOT VERIFIED -- not tested' rather than asserting it."
            )
            synthesis_user = prompts.render_workspace(
                task=cfg.task, facts=env.bb.view()["facts"], files=workspace.summary(),
                judge_feedback=env.last_judge, round_index=rounds_done,
            )
            if _would_exceed_budget(budget, cfg.workers[0], synthesis_system, synthesis_user, cfg.max_tokens):
                _notify(cfg, "warn", {"msg": "synthesis skipped: insufficient budget remaining"})
            else:
                try:
                    syn = chat(cfg.workers[0], synthesis_system, synthesis_user,
                              max_tokens=cfg.max_tokens, temperature=0.2, want_json=False)
                    _charge(budget, syn)
                    env.bb.facts["answer"] = syn.text.strip()
                except LLMError as e:
                    _notify(cfg, "warn", {"msg": f"synthesis failed: {e}"})

        result = SolveResult(
            answer=env.answer,
            score=env.score(),
            rounds=rounds_done,
            success=env.goal_satisfied(),
            cost=budget.as_dict(),
            run_dir=run_dir,
            verdict=(env.last_judge or {}).get("verdict", ""),
            gaps=(env.last_judge or {}).get("gaps", []),
            judge_fallback_reason=judge_fallback_reason,
            cancelled=cancelled,
            stalled=stalled,
            verification=last_report,
            branch=isolation.worktree.branch if isolation.worktree else None,
        )

        # isolation.worktree is not None is the load-bearing half of this guard, not
        # just an optimization: _setup_isolation's "existing plain directory" branch
        # can return a truthy base_sha with worktree=None and workspace_root pointing
        # at the REAL, unisolated target directory (repo_root() returned None for a
        # dir that genuinely was a repo, init_fresh_repo() then ran `git init`+`git
        # add -A`+`git commit` directly in it, then create_run_worktree() failed).
        # The merge call below is already correctly gated on isolation.worktree is
        # not None; this checkpoint call must be too, or it would happily commit
        # straight onto whatever branch was really checked out there.
        if isolation.worktree is not None and isolation.base_sha and repo.is_git_repo(workspace.root):
            checkpoint = repo.checkpoint_worktree(
                workspace.root,
                "solve: final state" if result.success else "solve: incomplete checkpoint",
            )
            result.checkpoint_sha = checkpoint.commit_sha
            result.checkpoint_error = checkpoint.error
            log.emit("checkpoint", branch=result.branch, committed=checkpoint.committed,
                     commit_sha=checkpoint.commit_sha, error=checkpoint.error,
                     success=result.success, cancelled=cancelled)
            _notify(cfg, "warn" if checkpoint.error else "checkpoint", {
                "msg": (f"checkpoint failed: {checkpoint.error}" if checkpoint.error
                        else f"saved run state at commit {checkpoint.commit_sha}"),
            })

        if isolation.worktree is not None:
            if result.success and cfg.auto_merge and not cancelled:
                mr = repo.merge_worktree_branch(isolation.worktree.repo_root, workspace.root,
                                                isolation.worktree.branch)
                result.merged = mr.merged
                result.merge_error = mr.error
            else:
                result.merge_error = "not attempted (auto-merge disabled, run not successful, or cancelled)"
            log.emit("merge", branch=isolation.worktree.branch, merged=result.merged, error=result.merge_error)
            _notify(cfg, "warn" if not result.merged else "merge",
                   {"msg": (f"merged {isolation.worktree.branch} into "
                            f"{isolation.worktree.repo_root}" if result.merged
                            else f"not merged: {result.merge_error} (branch: {isolation.worktree.branch})")})

        if memory is not None:
            memory.record(cfg.task, result.answer, {"success": result.success, "score": result.score,
                                                     "cost": result.cost})
            log.emit("memory_record", task=cfg.task)
        log.emit("episode_end", success=result.success, score=result.score, rounds=result.rounds,
                 answer_chars=len(result.answer), cost=result.cost)

        result.audit_verified = EventLog.verify(log.records)
        if not result.audit_verified:
            raise RuntimeError("solve audit ledger verification failed")

    banner = "" if result.success else (
        f"> WARNING: UNVERIFIED / INCOMPLETE -- judge score {result.score:.2f}, did not clear "
        f"the acceptance threshold. Open gaps: {'; '.join(result.gaps) or 'none logged'}.\n"
        "> Treat every claim below as unconfirmed until independently checked.\n\n"
    )
    (run_dir / "answer.md").write_text(banner + (result.answer or "(no answer produced)\n"), encoding="utf-8")
    _notify(cfg, "done", {"success": result.success, "score": result.score, "run_dir": str(run_dir)})
    return result
