"""System prompts, the shared-workspace renderer, and tolerant reply parsers.

Three roles share one JSON contract so any backend can fill any role:

* **worker** -- each turn, read the workspace and take exactly one action.
* **planner** -- round 0 only, turn the task into a checklist + acceptance criteria.
* **judge**  -- each round end, score the current answer against the task.

Parsing is tolerant: slice the first ``{`` to its balanced end and
``json.loads`` it, tolerating code fences and trailing prose that small local
models like to add.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from swarmsolve.toolbox import TOOLBOX

# Core protocol actions -- these mutate the blackboard/answer directly rather than
# being (Workspace, args, ToolPolicy) -> ToolResult capabilities, so they live here
# rather than in TOOLBOX. Everything else a worker can call is a registered tool;
# ACTIONS and the prompt's args-help block are both generated from TOOLBOX so a new
# tool only needs registering once (see swarmsolve/toolbox.py) to show up in both.
_CORE_PRE = (("think", '{"text": "a reasoning step, no side effects"}'),
             ("note", '{"text": "a finding/fact to record on the shared blackboard"}'))
_CORE_POST = (
    ("generate_file", '{"path": "relative/path", "instructions": "what the complete file must contain"}'),
    ("propose_answer", '{"text": "the complete answer/deliverable for the user"}'),
)

ACTIONS = tuple(n for n, _ in _CORE_PRE) + tuple(TOOLBOX.names()) + tuple(n for n, _ in _CORE_POST)


def _actions_block() -> str:
    custom = {s.name for s in TOOLBOX.custom_tools()}
    builtin_names = [n for n in TOOLBOX.names() if n not in custom]
    all_names = ([n for n, _ in _CORE_PRE] + builtin_names + list(custom)
                 + [n for n, _ in _CORE_POST])
    width = max(len(n) for n in all_names) + 1 if all_names else 14
    lines = [f"  {n:<{width}}{a}" for n, a in _CORE_PRE]
    lines += [f"  {n:<{width}}{TOOLBOX.args_help(n)}" for n in builtin_names]
    if custom:
        lines.append("")
        lines.append("  SPECIALIZED TOOLS (use these when they fit the task — they exist because")
        lines.append("  a generic web_search or shell command cannot do what they do):")
        for name in sorted(custom):
            spec = TOOLBOX.get(name)
            desc = f"  — {spec.description}" if spec and spec.description else ""
            lines.append(f"  {name:<{width}}{TOOLBOX.args_help(name)}{desc}")
    lines += [f"  {n:<{width}}{a}" for n, a in _CORE_POST]
    return "\n".join(lines)


WORKER_SYSTEM_TEMPLATE = (
    "You are ONE agent in a swarm working on the user's task. You share a workspace "
    "(a plan, findings from every agent, files on disk, the current answer draft, and "
    "the judge's latest feedback) with the other agents.\n\n"
    "Each turn, look at what is already done and take ONE action that most improves the "
    "answer -- fill a gap the judge named, verify a claim, write or fix a file, or "
    "propose/revise the final answer. Do not repeat work already in the findings.\n\n"
    'Reply with ONLY a JSON object: {{"action": "<name>", "args": {{<...>}}, "note": "<=15 words", '
    '"work_status": "completed|incomplete|blocked", "evidence": ["observed result"], '
    '"lessons_learned": ["useful handoff fact"], "next_action": "specific continuation or empty"}}\n'
    "Actions and their args:\n"
    "{actions_block}\n\n"
    "The action MUST be one of the listed names. Use replace_text for a small targeted edit "
    "when you know the exact existing block. Use generate_file for new source files "
    "or genuinely substantial rewrites: describe the desired file in short instructions and the "
    "pipeline will generate its body outside this JSON protocol. Use write_file only for "
    "small content such as a short config or targeted replacement. Never embed a large "
    "source file in this action JSON. Shell is for inspection and testing, NEVER for "
    "editing source through python -c, sed, redirection, Set-Content, or similar quoting-"
    "fragile commands; use generate_file for those edits. The shell runner already captures "
    "stdout and stderr: never append `2>&1`, `Out-String`, or other output-formatting pipelines "
    "to a diagnostic command because they obscure the useful exception. Always put action parameters "
    "inside the `args` object, never alongside `action`. When the task looks done, use "
    "propose_answer.\n\n"
    "If a SYSTEM ASSIGNMENT is present, it overrides the shared judge advice; work only on "
    "your assigned item so the swarm does not duplicate one task. Report completed only when this "
    "turn finished the assigned item; completed still requires independent verification. Report "
    "incomplete with evidence, lessons_learned, and a concrete next_action when another turn is "
    "needed. Report blocked only when a named dependency or external input prevents progress. "
    "Otherwise, judge_feedback.advice comes from a stronger reasoning model watching progress and "
    "is your strongest lead."
)
# Unfiltered rendering (all tools shown), kept for anything that imports the constant directly.
WORKER_SYSTEM = WORKER_SYSTEM_TEMPLATE.format(actions_block=_actions_block())

# Orientation hint added when the pipeline knows the workspace root -- prevents
# the workers from trying to `cd` to an absolute path in the task text when that
# path refers to a project that isolation has already checked out INTO the
# workspace, under a different root.
WORKSPACE_ADDENDUM = (
    "\n\nYour workspace root is: {workspace_root}\n"
    "* The shell action's working directory is ALREADY this workspace root.\n"
    "* read_file / write_file paths are RELATIVE to this root -- use 'app.py',\n"
    "  NOT an absolute path like 'C:\\\\Users\\\\...\\\\app.py' or '/home/.../app.py'.\n"
    "* If the user's task text mentions a project at some OTHER absolute path,\n"
    "  that project has already been checked out INTO your workspace root as an\n"
    "  isolated git worktree. Do NOT try to `cd` to the original location -- it's\n"
    "  outside your workspace and the shell will fail. Explore what's actually\n"
    "  here: `list_files`, or `git ls-tree -r HEAD` (fastest for a repo). All\n"
    "  files you see there are the project you were asked to work on.\n"
    "{shell_guidance}"
)


def _shell_guidance() -> str:
    if sys.platform == "win32":
        return (
            "* The shell is PowerShell on Windows. Use PowerShell commands such as "
            "Get-ChildItem, Get-Content, Select-String, and Select-Object -First/-Last. "
            "Do NOT use Unix-only grep/head/tail, `ls -la`, or wrap commands in another "
            "`powershell -Command`. For Python snippets, use `python -c \"...\"`; the "
            "inner quotes are preserved."
        )
    return (
        "* The shell is POSIX /bin/sh. Use portable POSIX commands; do not emit "
        "PowerShell or cmd.exe syntax."
    )

PLANNER_SYSTEM = (
    "You turn the user's task into a short dependency-aware execution plan for a swarm of agents.\n"
    'Reply with ONLY a JSON object: {"plan": [{"id": "T1", "task": "concrete action", '
    '"depends_on": [], "acceptance": "locally testable passing condition"}, ...], '
    '"acceptance_criteria": ["what a complete, correct answer must satisfy", ...]}\n'
    "Keep it to 3-7 plan steps and 2-5 acceptance criteria. Dependencies must reference earlier "
    "step ids. Make every step independently assignable and its acceptance condition executable "
    "or objectively inspectable. Do not create multiple steps that write the same file."
)

JUDGE_SYSTEM = (
    "You are the JUDGE and SENIOR ADVISOR for a swarm working on the user's task. You are "
    "deliberately a stronger reasoner than the worker agents -- your job is not just to "
    "grade, but to think harder than they can and hand back the single most useful next move.\n\n"
    "Score how well the current answer (if any) satisfies the task and the acceptance "
    "criteria, using the workspace file list and recent findings as evidence. If no answer "
    "has been proposed yet, score 0 but still read the findings and give real strategic "
    "advice -- don't wait for a draft to start reasoning.\n\n"
    'Reply with ONLY a JSON object: {"score": <0.0-1.0>, "verdict": "<one line>", '
    '"gaps": ["specific missing/wrong thing", ...], "done": <true|false>, '
    '"advice": "<one concrete, specific next action for the workers -- not a vague '
    'suggestion>"}\n\n'
    "Be strict on scoring. Lower the score for: unsupported claims, missing deliverables, "
    "code that was never run/tested, criteria not met. Set done=true only when the answer is "
    "genuinely complete and correct. Prefix a gap with 'CRITICAL:' if it blocks acceptance. "
    "Make 'advice' the best idea a capable human expert would try next given exactly what's "
    "in the findings -- name the specific path/parameter/technique/command, not 'investigate "
    "further'."
)


def worker_system(workspace_root: str | None = None) -> str:
    text = WORKER_SYSTEM_TEMPLATE.format(actions_block=_actions_block())
    if workspace_root:
        text = (text + WORKSPACE_ADDENDUM
                .replace("{workspace_root}", str(workspace_root))
                .replace("{shell_guidance}", _shell_guidance()))
    return text


def planner_system() -> str:
    base = PLANNER_SYSTEM
    custom = TOOLBOX.custom_tools()
    if custom:
        tool_lines = ", ".join(
            f"{s.name} ({s.description})" if s.description else s.name
            for s in custom
        )
        base += (
            f"\n\nSPECIALIZED TOOLS AVAILABLE: {tool_lines}. "
            "When a plan step would benefit from one of these, name it explicitly in "
            "the step's task description. "
            "Workers pick actions from the prompt — they are more likely to use a "
            "specialized tool when the plan tells them to."
        )
    return base


def judge_system() -> str:
    return JUDGE_SYSTEM


def render_workspace(
    *,
    task: str,
    facts: dict[str, Any],
    files: str,
    judge_feedback: dict[str, Any] | None,
    round_index: int,
    action_hint: str = "",
) -> str:
    payload = {
        "task": task,
        "round": round_index,
        "plan": facts.get("plan", []),
        "acceptance_criteria": facts.get("criteria", []),
        # A blackboard is working state, not a transcript replay. A bounded,
        # deduplicated tail prevents token growth from becoming quadratic in
        # round count while the structured work-item agenda preserves continuity.
        "findings": facts.get("findings", [])[-16:],
        # Include owned/blocked/verifying items, not merely selectable work, so
        # every worker can see that another agent already has a lease and can
        # learn from an incomplete handoff.
        "work_items": [x for x in facts.get("work_items", []) if x.get("status") != "resolved"][:12],
        "current_answer": str(facts.get("answer", ""))[-8000:],
        "workspace_files": files,
        "judge_feedback": judge_feedback or "none yet",
        "action_hint": action_hint or "take the single most useful next action; propose_answer when done",
    }
    return json.dumps(payload, default=str)


def _slice_json(text: str) -> Any:
    # first-'{' to raw_decode's balanced end, NOT first-'{' to last-'}' -- a model that
    # emits a valid object and then keeps generating (e.g. hallucinating its own next
    # "user" turn, echoing JSON-shaped text back at itself) would otherwise span both
    # objects and fail to parse at all.
    if not text:
        raise ValueError("empty reply")
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found")
    value, _end = json.JSONDecoder().raw_decode(text, start)
    return value


def parse_worker_reply(text: str) -> tuple[str, dict[str, Any], str, dict[str, Any]]:
    """Parse an action plus its structured work-item handoff."""
    try:
        obj = _slice_json(text)
        name = str(obj.get("action", "")).strip()
        args = obj.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        # Several OpenAI-compatible models emit a perfectly valid but flattened
        # tool object despite being shown the nested contract, e.g.
        # {"action":"write_file","path":"x","content":"..."}. Accept that
        # common dialect instead of dispatching write_file with an empty path.
        outcome_keys = {"work_status", "evidence", "lessons_learned", "next_action"}
        flat_args = {k: v for k, v in obj.items() if k not in {"action", "args", "note", *outcome_keys}}
        if flat_args:
            args = {**flat_args, **args}
        note = str(obj.get("note", ""))[:120]
        status = str(obj.get("work_status") or "").strip().lower()
        if status not in {"completed", "incomplete", "blocked"}:
            status = ""

        def _string_list(value: Any) -> list[str]:
            if isinstance(value, str):
                value = [value]
            if not isinstance(value, list):
                return []
            return [str(v).strip()[:500] for v in value if str(v).strip()][:8]

        outcome = {
            "work_status": status,
            "evidence": _string_list(obj.get("evidence")),
            "lessons_learned": _string_list(obj.get("lessons_learned")),
            "next_action": str(obj.get("next_action") or "").strip()[:500],
        }
        # Check the frozen core actions AND the live TOOLBOX registry — manifest
        # tools are loaded after import time, so they won't be in the ACTIONS tuple
        # but will be in TOOLBOX by the time a worker reply is parsed.
        if name in ACTIONS or name in TOOLBOX:
            return name, args, note, outcome
        return "think", {"text": f"(model chose unknown action {name!r})"}, note or "fallback", outcome
    except (ValueError, json.JSONDecodeError, TypeError, AttributeError):
        snippet = text.strip().replace("\n", " ")[:160]
        return "think", {"text": f"(unparseable reply: {snippet})"}, "unparseable -> think", {}


def parse_action(text: str) -> tuple[str, dict[str, Any], str]:
    """Backward-compatible action-only parser used by integrations/tests."""
    name, args, note, _outcome = parse_worker_reply(text)
    return name, args, note


def parse_json_reply(text: str) -> dict[str, Any] | None:
    """For planner / judge. Returns None if nothing JSON-ish is present."""
    try:
        obj = _slice_json(text)
        return obj if isinstance(obj, dict) else None
    except (ValueError, json.JSONDecodeError, TypeError):
        return None
