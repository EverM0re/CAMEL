"""LLM client with retry + on-disk response caching.

Speaks either protocol, selected by ``llm.api`` in config.yaml:

  openai     -> DeepSeek's OpenAI-compatible endpoint  (https://api.deepseek.com/v1)
  anthropic  -> DeepSeek's Anthropic-compatible endpoint (.../anthropic)

Responses are cached by (model, system, user) so re-runs and sweeps do not pay
for the same call twice.
"""
import hashlib
import json
import threading
import time
from pathlib import Path

from . import config

# ---------------------------------------------------------------------------
# Global abort latch. Set on a non-recoverable API condition (out of credit,
# bad key) so that a thread pool with hundreds of queued questions stops
# immediately instead of retrying every one of them to the same dead end.
# ---------------------------------------------------------------------------
_fatal_lock = threading.Lock()
_fatal_reason = None


class FatalLLMError(RuntimeError):
    """Raised when no further LLM call in this run can succeed."""


def set_fatal(reason):
    global _fatal_reason
    with _fatal_lock:
        if _fatal_reason is None:
            _fatal_reason = reason
            print(f"\n*** ABORTING: {reason}\n"
                  f"*** Completed work is saved; re-run to resume.\n", flush=True)


def fatal_reason():
    return _fatal_reason


# Per-provider API quirks, discovered on the first rejected request and then
# reused so the whole run adapts after a single failed call.
_quirks = set()


def _quirk(name):
    return name in _quirks


def _set_quirk(name):
    with _fatal_lock:
        if name not in _quirks:
            _quirks.add(name)
            print(f"[llm] provider quirk detected: {name}", flush=True)


# Server-side conditions that clear on their own. 429 is included: a rate
# limit is a "slow down", not a "stop".
_RETRY_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 529})


def _status_of(exc):
    """HTTP status of an SDK exception, or None.

    Read from the exception object rather than its text. Provider request ids
    are long alphanumeric strings that routinely contain digit runs like
    '402' or '404', so substring-matching a status code in the message text
    misclassifies healthy transient errors as fatal.
    """
    for attr in ("status_code", "status", "http_status", "code"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return v
        if isinstance(v, str) and v.isdigit():
            return int(v)
    resp = getattr(exc, "response", None)
    v = getattr(resp, "status_code", None)
    return v if isinstance(v, int) else None


class LLMClient:
    """Thin wrapper over a chat-completions API."""

    def __init__(self, model=None, max_tokens=512, temperature=0.0,
                 cache_dir=None, api=None, base_url=None, api_key=None):
        self.model = model or config.ANSWER_MODEL
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.api = (api or config.LLM_API).strip().lower()
        # Per-client endpoint. The backbone-transfer experiment points the
        # ANSWER model at a different server while the judge must stay on its
        # own; a single process-wide endpoint would send the judge's model
        # name to the answer model's server and 404.
        self.base_url = base_url or config.LLM_BASE_URL
        self.api_key = api_key or config.LLM_API_KEY
        self.cache_dir = Path(cache_dir) if cache_dir else config.CACHE_DIR / "llm"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = None

    # -- client construction -------------------------------------------
    def _get_client(self):
        if self._client is not None:
            return self._client
        if self.api == "anthropic":
            try:
                import anthropic
            except ImportError as e:
                raise SystemExit(
                    "The 'anthropic' package is required for llm.api: anthropic\n"
                    "  pip install anthropic\n"
                    "or set llm.api: openai in config.yaml") from e
            self._client = anthropic.Anthropic(
                base_url=self.base_url, api_key=self.api_key,
                timeout=config.LLM_TIMEOUT,
            )
        else:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise SystemExit(
                    "The 'openai' package is required for llm.api: openai\n"
                    "  pip install openai") from e
            self._client = OpenAI(
                base_url=self.base_url, api_key=self.api_key,
                timeout=config.LLM_TIMEOUT,
            )
        return self._client

    # -- one request ----------------------------------------------------
    def _call(self, system, user, model, max_tokens, temperature):
        client = self._get_client()
        if self.api == "anthropic":
            resp = client.messages.create(
                model=model, max_tokens=max_tokens, temperature=temperature,
                system=system, messages=[{"role": "user", "content": user}],
            )
            return "".join(b.text for b in resp.content
                           if getattr(b, "type", "") == "text")
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
        kw = {"model": model, "messages": msgs}
        if not _quirk("no_max_tokens"):
            kw["max_completion_tokens" if _quirk("max_completion_tokens")
               else "max_tokens"] = max_tokens
        if not _quirk("no_temperature"):
            kw["temperature"] = temperature
        try:
            resp = client.chat.completions.create(**kw)
        except Exception as e:  # noqa: BLE001
            # Reasoning models on some providers reject `max_tokens` (wanting
            # `max_completion_tokens`) or any non-default temperature. Learn the
            # quirk once, then retry -- rather than failing the whole run over a
            # parameter name.
            m = str(e).lower()
            changed = False
            if "max_completion_tokens" in m and not _quirk("max_completion_tokens"):
                _set_quirk("max_completion_tokens"); changed = True
            elif "max_tokens" in m and "unsupported" in m and not _quirk("no_max_tokens"):
                _set_quirk("no_max_tokens"); changed = True
            if "temperature" in m and not _quirk("no_temperature"):
                _set_quirk("no_temperature"); changed = True
            if not changed:
                raise
            return self._call(system, user, model, max_tokens, temperature)
        choice = resp.choices[0]
        text = choice.message.content or ""
        if not text.strip():
            fr = getattr(choice, "finish_reason", "")
            if fr == "length":
                raise RuntimeError(
                    "empty response: the model used the entire token budget on "
                    "reasoning. Raise llm.min_tokens in config.yaml.")
        return text

    # -- public ---------------------------------------------------------
    def complete(self, system, user, max_tokens=None, temperature=None,
                 model=None, use_cache=True):
        model = model or self.model
        # Enforce the global floor: reasoning models need room to think before
        # they emit anything, and every historical "empty answer" bug traced
        # back to a caller passing a small budget.
        max_tokens = max(max_tokens or self.max_tokens, config.LLM_MIN_TOKENS)
        temperature = self.temperature if temperature is None else temperature

        # The endpoint is part of the key: the same model NAME served by two
        # different hosts is two different models as far as results go, and
        # the backbone experiment runs exactly that way.
        key = hashlib.sha1(
            f"{self.api}|{self.base_url}|{model}|{system}|{user}"
            .encode("utf-8")
        ).hexdigest()
        cache_file = self.cache_dir / f"{key}.json"
        if use_cache and cache_file.exists():
            try:
                return json.loads(cache_file.read_text())["text"]
            except (json.JSONDecodeError, KeyError):
                pass   # corrupt cache entry: fall through and re-request

        if _fatal_reason is not None:
            raise FatalLLMError(_fatal_reason)

        last_err = None
        for attempt in range(config.LLM_MAX_RETRIES):
            if _fatal_reason is not None:
                raise FatalLLMError(_fatal_reason)
            try:
                text = self._call(system, user, model, max_tokens, temperature)
                # atomic cache write (tmp + rename) for thread safety
                tmp = cache_file.with_suffix(f".{id(self)}.tmp")
                tmp.write_text(json.dumps({"text": text}, ensure_ascii=False))
                tmp.replace(cache_file)
                return text
            except Exception as e:  # noqa: BLE001
                last_err = e
                msg = str(e).lower()
                status = _status_of(e)

                # Transient by definition: the server is up but momentarily
                # unable to serve. These MUST be retried -- a provider under
                # load emits them routinely and they say nothing about credit.
                if status in _RETRY_STATUS or any(
                        s in msg for s in ("temporarily unavailable",
                                           "service unavailable",
                                           "overloaded", "try again later",
                                           "rate limit", "timeout",
                                           "connection reset")):
                    wait = min(2 ** attempt, 60)
                    print(f"[llm] transient error"
                          f"{f' (HTTP {status})' if status else ''}, retry "
                          f"{attempt + 1}/{config.LLM_MAX_RETRIES} in {wait}s",
                          flush=True)
                    time.sleep(wait)
                    continue

                # Deterministic failures: retrying reaches the same error.
                # Matched on the HTTP status where we have one, because a bare
                # substring like "402" also occurs inside provider request ids
                # -- that false match once killed a healthy run on a 503.
                if status in (401, 403, 404) or any(s in msg for s in (
                        "authentication", "api_key", "auth_token", "unauthorized",
                        "invalid api key", "does not exist", "invalid model")):
                    set_fatal(f"LLM request rejected: {e}")
                    raise FatalLLMError(str(e)) from e
                if status == 402 or any(s in msg for s in (
                        "insufficient balance", "exceeded your current",
                        "insufficient_quota", "billing", "payment required")):
                    # Out of credit: every remaining question fails the same
                    # way, so stop now instead of grinding through thousands.
                    set_fatal(f"LLM billing error: {e}")
                    raise FatalLLMError(str(e)) from e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"LLM call failed: {last_err}")
