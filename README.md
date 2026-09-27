# SwarmSolve

On long or open-ended tasks, a single agent has to hold the whole task in one
context window for the entire run. This project instead splits the task
across multiple agents that coordinate through shared state, so no one
context has to carry everything at once. Idea is from the structural pattern as
ant-colony and bacterial-foraging algorithms (see [References](#references)).

Swarmsolve allows workers (any mix of local Ollama models, Claude, Gemini, DeepSeek) read and
write a shared blackboard across rounds. A judge model scores each round's
answer against the task's acceptance criteria. A planner reassigns work to
whatever is still unresolved.

No round is accepted on a model's self-report. Each one is checked against
the project's own build/test commands, run against the actual diff. Runs are
isolated to a separate git worktree, every event is hash-chained into an
audit log, and spend is capped by a budget you set before the run starts.

```
planner (round 0)  ->  plan + acceptance criteria onto the blackboard
coordination rounds 1..R:
    wake only workers who own ready work
    each active worker takes ONE action (read, edit, shell, search, ...)
    a verification pass runs your build/test commands against the real diff
    a judge scores the current answer -> gaps flow back to workers next round
    stop once the judge clears the threshold (and no CRITICAL gap remain),
    or budget/rounds run out
finalise: if no answer was ever proposed, one synthesis pass writes one
```

## Install

```bash
python -m venv .venv
.venv\Scripts\activate       # or: source .venv/bin/activate
pip install -e .
```

Optional, only if you want Claude as a worker or judge:

```bash
pip install -e ".[claude]"
```

Copy `.env.example` to `.env` and fill in keys for whichever backends you use.
`ollama/...` and `stub/...` model specs need no key.

## Use

```bash
python -m swarmsolve "Write parser.py: nginx access log -> list[dict], with a pytest" \
    --no-web --rounds 4

python -m swarmsolve --file TASK.md \
    --workers ollama/qwen3-coder-30b*2,gemini/gemini-2.5-flash \
    --judge claude/claude-haiku-4-5 --budget-usd 0.50
```

Or embed it:

```python
from swarmsolve import solve, SolveConfig
from swarmsolve.llm import ModelSpec

cfg = SolveConfig(task="...", workers=[ModelSpec.parse("deepseek/deepseek-chat")])
result = solve(cfg)
print(result.answer)
```

Run `python -m swarmsolve --help` for the full flag list (budget cap, worker
roster, round limits, workspace isolation, build/test command overrides, ...).

## What a worker can do

Every worker action is a compact JSON object dispatched through a small,
generic tool registry (`swarmsolve/toolbox.py`):

- `list_files`, `read_file`, `write_file`, `replace_text`, `generate_file` --
  inspect and edit files in the isolated workspace
- `shell` -- run an inspection/test command (source edits through shell are
  blocked; use the file tools above so edits stay atomic and syntax-checked)
- `web_search`, `fetch_url` -- DuckDuckGo search and SSRF-guarded HTTP fetch
- `think`, `note`, `propose_answer` -- reasoning and submitting the final answer

You can add more tools with zero Python by dropping a `.toml` manifest into
`swarmsolve/toolbox.d/` -- see `line_count.toml` (a CLI tool) and `my_ip.toml`
(an HTTP tool) for the format, documented in `swarmsolve/external_tools.py`.

## Safety by design

- **Isolated by default**: writes land on a fresh `solve/<run_id>` git branch
  in its own worktree, never directly on your checked-out branch. A run is
  only merged back automatically if it fully verifies (build/tests pass) and
  the judge accepts it.
- **Verification-gated**: every round runs your project's own build/test
  commands (auto-detected, or pass `--build-cmd`/`--test-cmd`) against the
  real diff. A judge's opinion never overrides what actually ran.
- **Tamper-evident audit log**: every event (plan, step, verify, judge,
  merge) is hash-chained into `events.jsonl`; a run is rejected if the chain
  doesn't verify.
- **Budget-capped**: a hard `$` ceiling with pre-flight cost estimation, so a
  run can't blow past what you set.
- **SSRF-guarded fetches**: `fetch_url`/manifest HTTP tools refuse requests to
  private/link-local addresses (resolved at request time; not hardened
  against DNS rebinding).

### What this is *not*

The `shell` tool runs LLM-authored commands for real. A regex denylist blocks
the obviously catastrophic (`rm -rf /`, fork bombs, `sudo`, `format C:`, ...),
and commands run `cwd`-scoped to the workspace with a timeout and output cap
-- but that's mitigation, not a jail. On Linux, `bwrap` (if installed) adds a
real filesystem/network sandbox; on Windows and macOS there is currently no
sandbox at all, denylist only. Git-worktree isolation protects your checked-out
branch from a bad *diff*, but does nothing to contain what a shell command does
while the run is live. Don't point this at a workspace containing anything you
wouldn't want an untrusted script to read or touch.

## Credits

**Qwen3.5** and **Mistral 7B** (run locally via Ollama) designed the original
framework and wrote the initial code. Development was later taken over by the
coding agents **Claude** (Anthropic) and **Codex** (OpenAI), with substantial
direction and review from the author throughout.

### References

The swarm-of-agents design draws on the following prior research:

- L. Rosenberg, "Artificial Swarm Intelligence, a Human-in-the-Loop Approach
  to A.I.," in *Proc. 30th AAAI Conf. on Artificial Intelligence (AAAI-16)*,
  2016, pp. 4381-4382.
- H. Ahmed and J. Glasgow, "Swarm Intelligence: Concepts, Models and
  Applications," Technical Report 2012-585, School of Computing, Queen's
  University, Kingston, Ontario, Canada, Feb. 2012.
- J. Tang, G. Liu, and Q. T. Pan, "A Review on Representative Swarm
  Intelligence Algorithms for Solving Optimization Problems: Applications and
  Trends," *IEEE/CAA J. Autom. Sinica*, vol. 8, no. 10, pp. 1627-1643, Oct.
  2021.

## License

MIT -- see [LICENSE](LICENSE).
