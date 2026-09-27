"""Unified chat across model backends.

One entry point -- :func:`chat` -- hides five backends behind a ``backend/model``
spec string:

    ollama/qwen3-coder-30b-uncensored-hq    local, free, stdlib HTTP
    claude/claude-haiku-4-5                  anthropic SDK (optional dep)
    gemini/gemini-3.6-flash                  Google REST API, stdlib HTTP
    deepseek/deepseek-chat                   OpenAI-compatible REST API, stdlib HTTP
    stub/scripted                            scripted replies (tests / --dry-run)

Every backend returns the same :class:`Reply` and raises the same
:class:`LLMError` on transport trouble, so the pipeline never has to branch on
provider. Keys come from ``.env`` via :func:`swarmsolve.env.load_env`
(``ANTHROPIC_API_KEY`` / ``GEMINI_API_KEY`` / ``DEEPSEEK_API_KEY``); existing
env vars win.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from swarmsolve.env import load_env

# $ per million tokens (input, output). Verify against current pricing before a
# big run; an unlisted model just prices at $0 rather than guessing, which
# under-reports cost so keep an eye on the provider console for spend.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-fable-5-1": (10.0, 50.0),
    "gemini-3.6-flash": (0.10, 0.40),
    "gemini-2.0-flash-lite": (0.075, 0.30),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-1.5-pro": (1.25, 5.0),
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.5-flash": (0.30, 2.50),
    "deepseek-chat": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
}

_KNOWN_BACKENDS = ("ollama", "claude", "gemini", "deepseek", "stub")


def estimate_cost(spec: ModelSpec, system: str, user: str, max_tokens: int) -> float:
    """Rough worst-case USD estimate for one call, BEFORE making it.

    A hard $ budget is otherwise only checked by charging the real provider-reported
    cost after each call completes, so a single expensive call can push spend well
    past the cap before anything notices. This estimate lets a caller skip a call
    that would clearly blow the remaining headroom. Input tokens use the same
    ~4-chars/token heuristic the stub backend uses; output is bounded by
    ``max_tokens``, the real ceiling passed to the provider -- so this is an upper
    bound, not a prediction, and free/unpriced backends (ollama, stub, an unlisted
    model) estimate at $0.
    """
    pin, pout = PRICES.get(spec.model, (0.0, 0.0))
    if pin == 0.0 and pout == 0.0:
        return 0.0
    tokens_in = (len(system) + len(user)) // 4
    return tokens_in / 1e6 * pin + max_tokens / 1e6 * pout


class LLMError(RuntimeError):
    """Any failure talking to a model backend."""


@dataclass(frozen=True)
class ModelSpec:
    backend: str
    model: str
    params: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def parse(text: str) -> "ModelSpec":
        """``"ollama/qwen3-coder-30b"`` -> ModelSpec. Backend defaults to ollama."""
        text = text.strip()
        if "/" in text:
            backend, model = text.split("/", 1)
        else:
            backend, model = "ollama", text
        backend = backend.strip().lower()
        if backend not in _KNOWN_BACKENDS:
            raise LLMError(f"unknown backend {backend!r}; known: {', '.join(_KNOWN_BACKENDS)}")
        return ModelSpec(backend=backend, model=model.strip())

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.backend}/{self.model}"


@dataclass
class Reply:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    usd: float = 0.0
    finish_reason: str | None = None

    def usage(self) -> dict[str, float]:
        return {"tokens_in": self.tokens_in, "tokens_out": self.tokens_out, "usd": self.usd}


def chat(
    spec: ModelSpec,
    system: str,
    user: str,
    *,
    max_tokens: int = 1024,
    temperature: float = 0.4,
    want_json: bool = True,
) -> Reply:
    """Single-turn chat. ``want_json`` asks the backend to constrain output to JSON."""
    load_env()
    if spec.backend == "stub":
        return _stub(spec, system, user)
    if spec.backend == "ollama":
        return _ollama(spec, system, user, max_tokens, temperature, want_json)
    if spec.backend == "claude":
        return _claude(spec, system, user, max_tokens, temperature)
    if spec.backend == "gemini":
        return _gemini(spec, system, user, max_tokens, temperature, want_json)
    if spec.backend == "deepseek":
        return _deepseek(spec, system, user, max_tokens, temperature, want_json)
    raise LLMError(f"unknown backend {spec.backend!r}")  # pragma: no cover


# -- stub ---------------------------------------------------------------------

_STUB_CURSOR = {"i": 0}


def reset_stub() -> None:
    _STUB_CURSOR["i"] = 0


def _stub(spec: ModelSpec, system: str, user: str) -> Reply:
    raw = os.environ.get("SOLVE_STUB_REPLIES", "").strip()
    scripted: list[Any] = json.loads(raw) if raw else []
    i = _STUB_CURSOR["i"]
    if i < len(scripted):
        _STUB_CURSOR["i"] = i + 1
        item = scripted[i]
        text = item if isinstance(item, str) else json.dumps(item)
    else:
        # sensible defaults so a script that runs dry still terminates
        text = (
            json.dumps({"score": 1.0, "verdict": "stub auto-pass", "gaps": [], "done": True})
            if "JUDGE" in system or "score" in system.lower()
            else json.dumps({"action": "propose_answer", "args": {"text": "stub answer"}, "note": "stub"})
        )
    return Reply(text=text, tokens_in=len(user) // 4, tokens_out=len(text) // 4, usd=0.0)


# -- ollama -----------------------------------------------------------------

def _ollama(
    spec: ModelSpec, system: str, user: str, max_tokens: int, temperature: float, want_json: bool
) -> Reply:
    host = (
        spec.params.get("host")
        or os.environ.get("OLLAMA_HOST")
        or "http://localhost:11434"
    ).rstrip("/")
    body: dict[str, Any] = {
        "model": spec.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    if want_json:
        body["format"] = "json"
    req = urllib.request.Request(
        f"{host}/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=spec.params.get("timeout", 300.0)) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:200]
        raise LLMError(f"Ollama HTTP {e.code}: {detail} (is `ollama pull {spec.model}` done?)") from e
    except urllib.error.URLError as e:
        raise LLMError(
            f"cannot reach Ollama at {host} ({e.reason}); is `ollama serve` running?"
        ) from e
    except TimeoutError as e:
        # urllib can raise a bare (socket-layer) TimeoutError instead of URLError when the
        # read times out mid-response -- e.g. a CPU-bound model still generating. Must be
        # caught separately or it escapes as an unhandled crash instead of a retryable LLMError.
        raise LLMError(f"Ollama timed out after {spec.params.get('timeout', 300.0):g}s "
                       f"({spec.model} may be slow/CPU-bound or the box is under load)") from e
    except json.JSONDecodeError as e:
        raise LLMError(f"Ollama returned a non-JSON/incomplete response: {e}") from e
    text = ((data.get("message") or {}).get("content") or "").strip()
    return Reply(
        text=text,
        tokens_in=int(data.get("prompt_eval_count", 0) or 0),
        tokens_out=int(data.get("eval_count", 0) or 0),
        usd=0.0,
        finish_reason=str(data.get("done_reason") or "") or None,
    )


# -- claude ---------------------------------------------------------------

def _claude(spec: ModelSpec, system: str, user: str, max_tokens: int, temperature: float) -> Reply:
    try:
        import anthropic
    except ImportError as e:  # pragma: no cover - optional dep
        raise LLMError('Claude backend needs anthropic: pip install "swarmsolve[claude]"') from e
    client = anthropic.Anthropic()
    kwargs = dict(
        model=spec.model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    try:
        # `temperature` was dropped from the typed create() signature in newer SDKs
        # (>=1.x) but the Messages API itself still honors it for models that still
        # support it -- pass it as a raw body field.
        resp = client.messages.create(**kwargs, extra_body={"temperature": temperature})
    except anthropic.BadRequestError as e:
        if "temperature" not in str(e).lower():
            raise LLMError(f"Claude call failed ({type(e).__name__}): {e}") from e
        # some model families have deprecated temperature control entirely and
        # 400 on any value -- retry once without it.
        try:
            resp = client.messages.create(**kwargs)
        except Exception as e2:
            raise LLMError(f"Claude call failed ({type(e2).__name__}): {e2}") from e2
    except Exception as e:  # anthropic raises a family of errors; surface one type
        raise LLMError(f"Claude call failed ({type(e).__name__}): {e}") from e
    text = "".join(
        getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text"
    ).strip()
    usage = getattr(resp, "usage", None)
    tin = int(getattr(usage, "input_tokens", 0) or 0)
    tout = int(getattr(usage, "output_tokens", 0) or 0)
    pin, pout = PRICES.get(spec.model, (0.0, 0.0))
    return Reply(text=text, tokens_in=tin, tokens_out=tout, usd=tin / 1e6 * pin + tout / 1e6 * pout,
                 finish_reason=str(getattr(resp, "stop_reason", "") or "") or None)


# -- deepseek --------------------------------------------------------------
# OpenAI-compatible REST API.

def _deepseek(
    spec: ModelSpec, system: str, user: str, max_tokens: int, temperature: float, want_json: bool
) -> Reply:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise LLMError("DeepSeek backend needs DEEPSEEK_API_KEY in the environment / .env")
    base_url = (spec.params.get("base_url") or os.environ.get("DEEPSEEK_BASE_URL")
               or "https://api.deepseek.com").rstrip("/")
    body: dict[str, Any] = {
        "model": spec.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    # deepseek-reasoner doesn't support forced JSON mode, and the API 400s on
    # json_object unless the literal word "json" appears in the prompt somewhere
    # (this package's own system prompts always say "JSON object", so this is
    # normally a no-op safety check, not a real constraint on this package's
    # own callers).
    if want_json and spec.model != "deepseek-reasoner" and "json" in (system + user).lower():
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=spec.params.get("timeout", 120.0)) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        raise LLMError(f"DeepSeek HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise LLMError(f"cannot reach DeepSeek at {base_url} ({e.reason})") from e
    except TimeoutError as e:
        raise LLMError(f"DeepSeek timed out after {spec.params.get('timeout', 120.0):g}s") from e
    except json.JSONDecodeError as e:
        raise LLMError(f"DeepSeek returned a non-JSON/incomplete response: {e}") from e

    choice = (data.get("choices") or [{}])[0]
    text = ((choice.get("message") or {}).get("content", "")) or ""
    usage = data.get("usage") or {}
    tin = int(usage.get("prompt_tokens", 0) or 0)
    tout = int(usage.get("completion_tokens", 0) or 0)
    pin, pout = PRICES.get(spec.model, (0.0, 0.0))
    return Reply(text=text.strip(), tokens_in=tin, tokens_out=tout, usd=tin / 1e6 * pin + tout / 1e6 * pout,
                 finish_reason=str(choice.get("finish_reason") or "") or None)


# -- gemini -------------------------------------------------------------

def _gemini(
    spec: ModelSpec, system: str, user: str, max_tokens: int, temperature: float, want_json: bool
) -> Reply:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise LLMError("Gemini backend needs GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment / .env")
    gen: dict[str, Any] = {"temperature": temperature, "maxOutputTokens": max_tokens}
    if want_json:
        gen["responseMimeType"] = "application/json"
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": gen,
    }
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{spec.model}:generateContent?key={key}"
    )
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=spec.params.get("timeout", 120.0)) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        raise LLMError(f"Gemini HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise LLMError(f"cannot reach Gemini API ({e.reason})") from e
    except TimeoutError as e:  # bare socket-layer timeout, not always wrapped as URLError
        raise LLMError(f"Gemini timed out after {spec.params.get('timeout', 120.0):g}s") from e
    except json.JSONDecodeError as e:
        raise LLMError(f"Gemini returned a non-JSON/incomplete response: {e}") from e

    cands = data.get("candidates") or []
    text = ""
    if cands:
        parts = ((cands[0].get("content") or {}).get("parts")) or []
        text = "".join(p.get("text", "") for p in parts).strip()
    if not text and data.get("promptFeedback", {}).get("blockReason"):
        raise LLMError(f"Gemini blocked the prompt: {data['promptFeedback']['blockReason']}")
    meta = data.get("usageMetadata") or {}
    tin = int(meta.get("promptTokenCount", 0) or 0)
    tout = int(meta.get("candidatesTokenCount", 0) or 0)
    pin, pout = PRICES.get(spec.model, (0.0, 0.0))
    finish = str((cands[0].get("finishReason") if cands else "") or "") or None
    return Reply(text=text, tokens_in=tin, tokens_out=tout, usd=tin / 1e6 * pin + tout / 1e6 * pout,
                 finish_reason=finish)
