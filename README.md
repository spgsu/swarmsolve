# swarmsolve

A small blackboard swarm of LLM agents that plans, works, verifies, and judges
its way to an answer inside an isolated git worktree.

Give it a natural-language task; a swarm of tool-using worker agents (mix
local Ollama, Claude, Gemini, DeepSeek freely) works it on a shared blackboard
across adaptive rounds, a judge model scores each round's answer, and the
whole run is checked against your project's own build/test commands before
anything is trusted or merged.

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
  private/link-local addresses.

## License

MIT -- see [LICENSE](LICENSE).
