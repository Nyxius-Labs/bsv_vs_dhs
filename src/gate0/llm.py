"""Claude client with an on-disk response cache and Message Batches support.

Every request is keyed by a hash of (model, system, user, schema, effort, max_tokens, thinking, nonce).
The nonce is never sent; it distinguishes deliberate repeats of an identical prompt.
Modes: "batch" (Message Batches API, 50% cost), "live" (concurrent messages.create),
"offline" (cache only; a miss raises).
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

MAX_BATCH = 10_000
POLL_SECONDS = 30
LIVE_WORKERS = 8
RETRIES = 2  # extra attempts per run (each with a fresh nonce) for failed or unparseable outputs
SDK_RETRIES = 6  # the SDK's own backoff for 429/5xx/connection errors, per request
COMMIT_EVERY = 50


@dataclass(frozen=True)
class Call:
    model: str
    system: str
    user: str
    schema: dict
    nonce: str = ""
    meta: dict | None = field(default=None, compare=False, hash=False)  # never sent or hashed

    def key(self, effort: str, max_tokens: int, thinking: str = "adaptive") -> str:
        parts = [self.model, self.system, self.user, self.schema, effort, max_tokens, self.nonce]
        if thinking != "adaptive":  # keeps keys of adaptive-thinking calls as they were in v1
            parts.append(f"thinking={thinking}")
        blob = json.dumps(parts, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()


@dataclass
class Stats:
    cache_hits: int = 0
    new_calls: int = 0
    failures: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    by_model: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)


class LLMClient:
    def __init__(self, cache_path: Path, mode: str = "batch", effort: str = "low", max_tokens: int = 2048,
                 thinking: str = "adaptive", thinking_off: dict[str, str] | None = None):
        if mode not in ("batch", "live", "offline"):
            raise ValueError(f"unknown mode {mode}")
        if thinking not in ("adaptive", "off"):
            raise ValueError(f"unknown thinking setting {thinking}")
        self.mode, self.effort, self.max_tokens = mode, effort, max_tokens
        # "off" has a model-specific API form (Haiku 5.5: disabled; Sonnet 5.5: between_tools).
        self.thinking, self.thinking_off = thinking, thinking_off or {}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if mode != "offline":
            # One spending process at a time: two would collect and delete each other's batches.
            self._lockfile = open(cache_path.with_suffix(".lock"), "w")
            try:
                fcntl.flock(self._lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"another gate0 process is using {cache_path}") from None
        self.db = sqlite3.connect(cache_path)
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS responses (
                key TEXT PRIMARY KEY, model TEXT, output TEXT, raw TEXT, error TEXT,
                input_tokens INT, output_tokens INT, created REAL);
            CREATE TABLE IF NOT EXISTS failures (
                key TEXT, model TEXT, error TEXT, raw TEXT, input_tokens INT, output_tokens INT, created REAL);
            CREATE TABLE IF NOT EXISTS inflight (
                marker INTEGER PRIMARY KEY AUTOINCREMENT, batch_id TEXT, models TEXT, created REAL);
            """
        )
        self.stats = Stats()
        self._client: anthropic.Anthropic | None = None

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(max_retries=SDK_RETRIES)
        return self._client

    # ---- public -------------------------------------------------------------------------

    def close(self) -> None:
        """Commit, close the cache and release the spending lock."""
        self.db.commit()
        self.db.close()
        if getattr(self, "_lockfile", None):
            self._lockfile.close()
            self._lockfile = None

    def run(self, calls: list[Call]) -> list[dict | None]:
        """Return parsed JSON outputs aligned with `calls`; None where every attempt failed.

        Only successes are cached. Failures (errored/expired batch results, max_tokens, refusals,
        bad JSON, API errors) are logged in `failures` and retried under a fresh nonce, and are
        retried again on a later run.
        """
        out: list[dict | None] = [None] * len(calls)
        todo = list(range(len(calls)))
        for attempt in range(RETRIES + 1):
            results = self._resolve([self._variant(calls[i], attempt) for i in todo])
            for i, r in zip(todo, results):
                if r is not None and _valid(r, calls[i].schema):
                    out[i] = r
            todo = [i for i in todo if out[i] is None]
            if not todo:
                break
        self.stats.failures += len(todo)
        return out

    def usage_by_model(self, since: float = 0.0) -> dict[str, tuple[int, int, int]]:
        """(calls, input tokens, output tokens) billed per model since a time (successes and failures)."""
        q = """SELECT model, COUNT(*), SUM(i), SUM(o) FROM (
                   SELECT model, input_tokens i, output_tokens o, created FROM responses
                   UNION ALL SELECT model, input_tokens, output_tokens, created FROM failures)
               WHERE created >= ? GROUP BY model"""
        return {m: (int(n), int(i or 0), int(o or 0)) for m, n, i, o in self.db.execute(q, (since,)) if m}

    # ---- internals ----------------------------------------------------------------------

    def _variant(self, call: Call, attempt: int) -> Call:
        if attempt == 0:
            return call
        return Call(call.model, call.system, call.user, call.schema, f"{call.nonce}|retry{attempt}")

    def _resolve(self, calls: list[Call]) -> list[dict | None]:
        keys = [c.key(self.effort, self.max_tokens, self.thinking) for c in calls]
        cached = self._lookup(keys)
        missing = {k: c for k, c in zip(keys, calls) if k not in cached}
        self.stats.cache_hits += len(calls) - len(missing)
        if missing:
            if self.mode == "offline":
                raise RuntimeError(f"{len(missing)} calls not in cache (offline mode)")
            if self.mode == "batch":
                self._run_batch(missing)
            else:
                self._run_live(missing)
            cached = self._lookup(keys)
        return [cached.get(k) for k in keys]

    def _lookup(self, keys: list[str]) -> dict[str, dict]:
        found: dict[str, dict] = {}
        for chunk in _chunks(list(set(keys)), 900):
            q = (f"SELECT key, output FROM responses WHERE output IS NOT NULL "
                 f"AND key IN ({','.join('?' * len(chunk))})")
            for k, output in self.db.execute(q, chunk):
                found[k] = json.loads(output)
        return found

    def _params(self, call: Call) -> dict:
        params = {
            "model": call.model,
            "max_tokens": self.max_tokens,
            "system": call.system,
            "messages": [{"role": "user", "content": call.user}],
            "output_config": {
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": call.schema},
            },
        }
        if self.thinking == "off":
            if call.model not in self.thinking_off:
                raise ValueError(f"no registered 'thinking off' form for {call.model}")
            params["thinking"] = {"type": self.thinking_off[call.model]}
        return params

    def _store(self, key: str, model: str, message, error: str | None = None) -> None:
        output, raw = None, None
        if message is None:
            error = error or "no message"
        elif message.stop_reason != "end_turn":
            error = f"stop_reason={message.stop_reason}"
        else:
            raw = next((b.text for b in message.content if b.type == "text"), None)
            try:
                output = json.dumps(json.loads(raw))
            except (TypeError, json.JSONDecodeError) as e:
                error = f"json: {e}"
        usage = getattr(message, "usage", None)
        it, ot = (usage.input_tokens, usage.output_tokens) if usage else (0, 0)
        self.stats.new_calls += 1
        self.stats.input_tokens += it
        self.stats.output_tokens += ot
        m = self.stats.by_model.setdefault(model, [0, 0])
        m[0] += it
        m[1] += ot
        if output is not None:
            self.db.execute("INSERT OR REPLACE INTO responses VALUES (?,?,?,?,?,?,?,?)",
                            (key, model, output, raw, None, it, ot, time.time()))
        else:
            self.stats.errors[error.split(":")[0]] = self.stats.errors.get(error.split(":")[0], 0) + 1
            self.db.execute("INSERT INTO failures VALUES (?,?,?,?,?,?,?)", (key, model, error, raw, it, ot, time.time()))

    def _run_live(self, missing: dict[str, Call]) -> None:
        def one(item):
            key, call = item
            try:
                return key, call.model, self.client.messages.create(**self._params(call)), None
            except anthropic.APIError as e:  # after SDK retries; never cached, retried next attempt
                return key, call.model, None, f"api: {type(e).__name__}: {e}"

        with ThreadPoolExecutor(LIVE_WORKERS) as pool:
            for n, (key, model, msg, err) in enumerate(pool.map(one, missing.items()), 1):
                self._store(key, model, msg, err)
                if n % COMMIT_EVERY == 0:
                    self.db.commit()
        self.db.commit()

    def _run_batch(self, missing: dict[str, Call]) -> None:
        self._resume()
        still = {k: c for k, c in missing.items() if k not in self._lookup(list(missing))}
        for chunk in _chunks(list(still.items()), MAX_BATCH):
            models = {k: c.model for k, c in chunk}
            # Record intent before creating, so a crash right after create can be recovered.
            cur = self.db.execute("INSERT INTO inflight (batch_id, models, created) VALUES (NULL,?,?)",
                                  (json.dumps(models), time.time()))
            marker = cur.lastrowid
            self.db.commit()
            requests = [
                Request(custom_id=k, params=MessageCreateParamsNonStreaming(**self._params(c)))
                for k, c in chunk
            ]
            batch = self.client.messages.batches.create(requests=requests)
            self.db.execute("UPDATE inflight SET batch_id=? WHERE marker=?", (batch.id, marker))
            self.db.commit()
            print(f"[llm] submitted batch {batch.id} with {len(requests)} requests")
            self._collect(marker, batch.id)

    def _resume(self) -> None:
        """Finish batches left by a crashed run before submitting anything new."""
        for marker, batch_id, models, created in self.db.execute("SELECT * FROM inflight").fetchall():
            if batch_id is None:
                batch_id = self._find_orphan(len(json.loads(models)), created)
                if batch_id is None:
                    self.db.execute("DELETE FROM inflight WHERE marker=?", (marker,))
                    self.db.commit()
                    continue
                print(f"[llm] recovered orphaned batch {batch_id}")
            self._collect(marker, batch_id)

    def _find_orphan(self, n_requests: int, created: float) -> str | None:
        known = {b for (b,) in self.db.execute("SELECT batch_id FROM inflight WHERE batch_id IS NOT NULL")}
        for b in self.client.messages.batches.list(limit=20):
            c = b.request_counts
            total = c.processing + c.succeeded + c.errored + c.canceled + c.expired
            if b.id not in known and total == n_requests and b.created_at.timestamp() >= created - 5:
                return b.id
        return None

    def _collect(self, marker: int, batch_id: str) -> None:
        while True:
            batch = self.client.messages.batches.retrieve(batch_id)
            if batch.processing_status == "ended":
                break
            c = batch.request_counts
            print(f"[llm] batch {batch_id}: processing={c.processing} succeeded={c.succeeded}")
            time.sleep(POLL_SECONDS)
        (models_json,) = self.db.execute("SELECT models FROM inflight WHERE marker=?", (marker,)).fetchone()
        models = json.loads(models_json)
        for result in self.client.messages.batches.results(batch_id):
            key = result.custom_id
            ok = result.result.type == "succeeded"
            self._store(key, models.get(key, ""), result.result.message if ok else None,
                        None if ok else f"batch_{result.result.type}")
        self.db.execute("DELETE FROM inflight WHERE marker=?", (marker,))
        self.db.commit()


def _valid(output: dict, schema: dict) -> bool:
    return isinstance(output, dict) and all(k in output for k in schema.get("required", []))


def _chunks(xs: list, n: int):
    for i in range(0, len(xs), n):
        yield xs[i : i + n]
