"""``python -m swarmsolve "your task"`` -- run the general-purpose swarm.

Examples::

    # local only, no web, quick code task
    python -m swarmsolve "Write parser.py: nginx access log -> list[dict], with a pytest" \
        --no-web --rounds 4

    # heterogeneous swarm, Claude as judge, hard $ cap
    python -m swarmsolve --file TASK.md \
        --workers ollama/qwen3-coder-30b-uncensored-hq*2,gemini/gemini-3.6-flash \
        --judge claude/claude-haiku-4-5 --budget-usd 0.50
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from swarmsolve.env import load_env
from swarmsolve.llm import LLMError, ModelSpec
from swarmsolve.pipeline import DEFAULT_TOOLBOX_DIR, SolveConfig, SolveResult, solve
from swarmsolve.tools import ToolPolicy

DEFAULT_WORKER = "deepseek/deepseek-chat"  # cloud, key-only -- no assumption about what's pulled locally on ollama
DEFAULT_JUDGE = "claude/claude-haiku-4-5"


def _configure_stdio() -> None:
    """Make CLI output safe for model text on Windows and redirected consoles."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            # Test capture streams and already-closed streams may not support it.
            continue


def _parse_roster(text: str) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        count = 1
        if "*" in chunk:
            chunk, _, c = chunk.rpartition("*")
            count = max(1, int(c))
        specs.extend(ModelSpec.parse(chunk) for _ in range(count))
    return specs


def build_config(argv: list[str] | None = None) -> tuple[SolveConfig, argparse.Namespace]:
    ap = argparse.ArgumentParser(prog="swarmsolve", description="General-purpose agent swarm.")
    ap.add_argument("task", nargs="?", help="the task / question (or use --file)")
    ap.add_argument("--file", type=Path, help="read the task from a file")
    ap.add_argument("--workers", default=DEFAULT_WORKER,
                    help=f"comma list of backend/model, 'spec*N' for copies (default: {DEFAULT_WORKER})")
    ap.add_argument("--agents", type=int, default=3,
                    help="shorthand: N copies of a single --workers spec (ignored if --workers has a list)")
    ap.add_argument("--judge", default=DEFAULT_JUDGE, help=f"judge model (default: {DEFAULT_JUDGE})")
    ap.add_argument("--planner", default=None, help="planner model, or 'none' (default: first worker)")
    ap.add_argument("--rounds", type=int, default=8, help="initial round allocation")
    ap.add_argument("--max-rounds", type=int, default=16,
                    help="hard ceiling when adaptive rounds extend unfinished work")
    ap.add_argument("--no-adaptive-rounds", action="store_true",
                    help="treat --rounds as a fixed hard limit")
    ap.add_argument("--no-adaptive-workers", action="store_true",
                    help="wake the full worker roster every cycle (legacy broadcast mode)")
    ap.add_argument("--max-active-workers", type=int, default=3,
                    help="maximum workers used for independent exploration (writes still get one owner)")
    ap.add_argument("--max-item-attempts", type=int, default=2,
                    help="attempts per work item without a workspace change before stopping/escalating")
    ap.add_argument("--threshold", type=float, default=0.85, help="judge score to accept (0-1)")
    ap.add_argument("--budget-usd", type=float, default=1.0, help="hard $ cap for the run (0 = unlimited)")
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-shell", action="store_true")
    ap.add_argument("--no-web", action="store_true")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--no-bwrap", action="store_true", help="do not sandbox shell with bubblewrap")
    ap.add_argument("--shell-timeout", type=float, default=60.0)
    ap.add_argument("--workspace", type=Path, default=None,
                    help="an existing project dir to work in, or a fresh scratch dir "
                         "(default: runs/solve/<ts>/workspace). If it's a git repo, writes are "
                         "isolated onto a new worktree/branch -- see --no-isolate.")
    ap.add_argument("--no-memory", action="store_true", help="don't recall/record runs.jsonl cross-run memory")
    ap.add_argument("--toolbox-dir", type=Path, default=DEFAULT_TOOLBOX_DIR,
                    help=f"directory of .toml tool manifests to load, no Python required "
                         f"(default: {DEFAULT_TOOLBOX_DIR}); pass an empty/missing dir to load none")
    ap.add_argument("--no-isolate", action="store_true",
                    help="write directly into --workspace instead of an isolated git worktree/branch")
    ap.add_argument("--no-auto-merge", action="store_true",
                    help="never auto-merge the isolated branch back, even on a fully verified success")
    ap.add_argument("--build-cmd", default=None, help="override the auto-detected build command")
    ap.add_argument("--test-cmd", default=None, help="override the auto-detected test command")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="use the stub backend (no network, no cost)")
    args = ap.parse_args(argv)

    task = args.task
    if args.file:
        task = args.file.read_text(encoding="utf-8").strip()
    if not task:
        ap.error("provide a task argument or --file")

    if "," in args.workers or "*" in args.workers:
        workers = _parse_roster(args.workers)
    else:
        workers = _parse_roster(f"{args.workers}*{max(1, args.agents)}")
    judge_spec = ModelSpec.parse(args.judge)

    if args.dry_run:
        workers = [ModelSpec.parse("stub/scripted") for _ in workers]
        judge_spec = ModelSpec.parse("stub/scripted")

    planner = None
    if args.planner and args.planner.lower() != "none":
        planner = ModelSpec.parse(args.planner)
    elif args.planner and args.planner.lower() == "none":
        planner = None
    else:
        planner = workers[0]

    # judge fallback: claude backend unusable (missing key OR missing package)
    # -> use the first worker instead of erroring mid-run. Judge independence
    # matters most precisely when this triggers, so the warning is never
    # suppressed by --quiet (unlike routine progress output), and the reason is
    # threaded into SolveConfig so it also lands in the run's own audit trail below.
    load_env()
    judge_fallback_reason = None
    if judge_spec.backend == "claude":
        judge_fallback_reason = _claude_unavailable_reason()
        if judge_fallback_reason:
            print(f"! {judge_fallback_reason}; judge falls back to {workers[0]}", file=sys.stderr)
            judge_spec = workers[0]

    policy = ToolPolicy(
        allow_shell=not args.no_shell,
        allow_web=not args.no_web,
        allow_write=not args.no_write,
        bwrap=not args.no_bwrap,
        shell_timeout=args.shell_timeout,
    )
    cfg = SolveConfig(
        task=task, workers=workers, judge=judge_spec, planner=planner,
        rounds=args.rounds, adaptive_rounds=not args.no_adaptive_rounds,
        max_rounds=max(args.rounds, args.max_rounds),
        adaptive_workers=not args.no_adaptive_workers,
        max_active_workers=max(1, args.max_active_workers),
        max_item_attempts=max(1, args.max_item_attempts),
        threshold=args.threshold, budget_usd=args.budget_usd,
        max_tokens=args.max_tokens, seed=args.seed, policy=policy,
        workspace_dir=args.workspace,
        memory_path=None if args.no_memory else SolveConfig.memory_path,
        judge_fallback_reason=judge_fallback_reason,
        isolate=not args.no_isolate,
        auto_merge=not args.no_auto_merge,
        build_cmd=args.build_cmd,
        test_cmd=args.test_cmd,
        toolbox_dir=args.toolbox_dir,
    )
    return cfg, args


def _has_anthropic_key() -> bool:
    import os

    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _claude_unavailable_reason() -> str | None:
    """None if the claude backend is fully usable; else a short human-readable reason.

    Checks BOTH the API key and whether ``import anthropic`` actually works -- a
    valid key with no anthropic package installed would otherwise slip past a
    key-only pre-flight check and blow up every judge call with an ImportError,
    leaving the judge score at 0.0 for the whole run. Checking both here catches
    that once at the start of a run and falls back cleanly to a worker model
    instead of burning tokens against a broken judge.
    """
    import os

    if not os.environ.get("ANTHROPIC_API_KEY"):
        return "no ANTHROPIC_API_KEY"
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return 'anthropic package not installed (pip install -e ".[claude]")'
    return None


def _make_reporter(quiet: bool):
    def report(kind: str, data: dict) -> None:
        if quiet:
            return
        if kind == "start":
            print(f"  run dir : {data['run_dir']}")
            print(f"  posture : {data['policy']}")
            if data.get("branch"):
                print(f"  isolated: branch {data['branch']}")
        elif kind == "plan":
            if data.get("plan"):
                print("  plan    : " + " | ".join(data["plan"]))
        elif kind == "step":
            ok = "ok" if data.get("ok") else "!!"
            print(f"    r{data['round']} {data['agent']:<3} {ok} {data['action']:<14} {data.get('note', '')[:60]}")
        elif kind == "verify":
            flag = " !!" if data.get("has_failure") else ""
            print(f"    r{data['round']} verify{flag}: {data['summary']}")
        elif kind == "judge":
            gaps = data.get("gaps") or []
            g = f"  gaps: {'; '.join(gaps[:3])}" if gaps else ""
            print(f"  round {data['round']}: judge={data['score']:.2f}  \"{data['verdict'][:70]}\"  "
                  f"${data['usd']:.4f}{g}")
            if data.get("advice"):
                print(f"    advice: {data['advice'][:200]}")
        elif kind == "merge":
            print(f"  {data['msg']}")
        elif kind == "warn":
            print(f"  ! {data['msg']}", file=sys.stderr)

    return report


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    try:
        cfg, args = build_config(argv)
    except SystemExit as e:
        return int(e.code or 0)

    if not args.quiet:
        print(f"\ntask: {cfg.task[:200]}{'...' if len(cfg.task) > 200 else ''}")
        print(f"swarm: {len(cfg.workers)} workers [{', '.join(str(s) for s in cfg.workers)}]  "
              f"judge={cfg.judge}  rounds={cfg.rounds}  cap=${cfg.budget_usd}")
        if not args.no_bwrap and not shutil.which("bwrap") and not args.no_shell and not args.dry_run:
            print("  ! bubblewrap not found -- shell falls back to denylist + workspace-cwd only", file=sys.stderr)

    cfg.on_event = _make_reporter(args.quiet)

    try:
        result = solve(cfg)
    except (LLMError, ValueError) as e:
        print(f"\nsolve failed: {e}", file=sys.stderr)
        return 2

    if not args.quiet:
        verdict = "SUCCESS" if result.success else "INCOMPLETE"
        print(f"\n=== {verdict}  score={result.score:.2f}  rounds={result.rounds}  "
              f"${result.cost['usd']:.4f}  {result.cost['tokens_total']} tok ===")
        if not result.success and result.gaps:
            print("open gaps: " + "; ".join(result.gaps[:5]))
        print(f"\n{result.answer}\n")
        print(f"trace: {result.run_dir}/events.jsonl   workspace: {result.run_dir}/workspace/")
        if result.branch:
            if result.merged:
                print(f"merged: {result.branch} -> your checked-out branch")
            else:
                print(f"not merged: {result.branch} ({result.merge_error}) -- "
                      f"merge yourself with: git merge {result.branch}")
    else:
        if not result.success:
            print(f"! INCOMPLETE (score={result.score:.2f}) -- unverified, do not trust as-is", file=sys.stderr)
        print(result.answer)

    return 0 if result.success else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
