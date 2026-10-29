"""LLMClient against a fake Anthropic client: failures must never be cached, crashes must not double-bill."""

import json
import time
from datetime import datetime, timezone
from types import SimpleNamespace as NS

import anthropic
import httpx2 as httpx
import pytest

import gate0.llm as L

SCHEMA = {"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"], "additionalProperties": False}


def _msg(x=1, stop="end_turn"):
    return NS(stop_reason=stop, content=[NS(type="text", text=json.dumps({"x": x}))],
              usage=NS(input_tokens=10, output_tokens=5))


class FakeBatches:
    def __init__(self, outcome):
        self.outcome, self.store, self.created = outcome, {}, 0

    def create(self, requests):
        self.created += 1
        bid = f"b{self.created}"
        self.store[bid] = NS(id=bid, requests=[r["custom_id"] for r in requests],
                             created_at=datetime.now(timezone.utc))
        return self.store[bid]

    def _counts(self, b):
        return NS(processing=0, succeeded=len(b.requests), errored=0, canceled=0, expired=0)

    def retrieve(self, bid):
        b = self.store[bid]
        return NS(processing_status="ended", request_counts=self._counts(b))

    def results(self, bid):
        for cid in self.store[bid].requests:
            kind = self.outcome(cid)
            yield NS(custom_id=cid, result=NS(type=kind, message=_msg() if kind == "succeeded" else None))

    def list(self, limit=20):
        return [NS(id=b.id, request_counts=self._counts(b), created_at=b.created_at) for b in self.store.values()]


def _client(tmp_path, mode, fake):
    c = L.LLMClient(tmp_path / "cache.sqlite", mode)
    c._client = fake
    return c


def _calls(n):
    return [L.Call("m", "sys", f"q{i}", SCHEMA) for i in range(n)]


def test_expired_results_are_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "RETRIES", 0)
    batches = FakeBatches(lambda cid: "expired")
    c = _client(tmp_path, "batch", NS(messages=NS(batches=batches)))
    assert c.run(_calls(3)) == [None] * 3
    batches.outcome = lambda cid: "succeeded"
    assert c.run(_calls(3)) == [{"x": 1}] * 3  # retried, not replayed from cache
    assert batches.created == 2


def test_retry_recovers_within_a_run(tmp_path):
    batches = FakeBatches(None)
    batches.outcome = lambda cid: "errored" if batches.created == 1 else "succeeded"  # first batch fails
    c = _client(tmp_path, "batch", NS(messages=NS(batches=batches)))
    assert c.run(_calls(2)) == [{"x": 1}] * 2  # retry uses a fresh nonce, so a fresh key


def test_max_tokens_is_a_failure_not_an_answer(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "RETRIES", 0)
    fake = NS(messages=NS(create=lambda **kw: _msg(stop="max_tokens")))
    c = _client(tmp_path, "live", fake)
    assert c.run(_calls(1)) == [None]
    assert c.db.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 0
    assert c.db.execute("SELECT COUNT(*) FROM failures").fetchone()[0] == 1


def test_live_api_error_does_not_crash_or_lose_work(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "RETRIES", 0)
    req = httpx.Request("POST", "https://x")

    def create(**kw):
        if kw["messages"][0]["content"] == "q2":
            raise anthropic.RateLimitError("slow down", response=httpx.Response(429, request=req), body=None)
        return _msg()

    c = _client(tmp_path, "live", NS(messages=NS(create=create)))
    out = c.run(_calls(4))
    assert out.count(None) == 1 and out.count({"x": 1}) == 3
    assert c.db.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 3


def test_orphaned_batch_is_recovered_not_resubmitted(tmp_path):
    batches = FakeBatches(lambda cid: "succeeded")
    c = _client(tmp_path, "batch", NS(messages=NS(batches=batches)))
    calls = _calls(2)
    keys = [x.key(c.effort, c.max_tokens) for x in calls]
    # Simulate a crash after create() but before the batch id was recorded.
    c.db.execute("INSERT INTO inflight (batch_id, models, created) VALUES (NULL,?,?)",
                 (json.dumps({k: "m" for k in keys}), time.time() - 1))
    batches.create([{"custom_id": k} for k in keys])
    assert c.run(calls) == [{"x": 1}] * 2
    assert batches.created == 1


def test_second_spending_process_is_refused(tmp_path):
    a = L.LLMClient(tmp_path / "cache.sqlite", "batch")
    with pytest.raises(RuntimeError):
        L.LLMClient(tmp_path / "cache.sqlite", "live")
    L.LLMClient(tmp_path / "cache.sqlite", "offline")  # read-only use is fine
    del a


def test_thinking_off_is_sent_per_model_and_keyed(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "RETRIES", 0)
    sent = []
    fake = NS(messages=NS(create=lambda **kw: sent.append(kw) or _msg()))
    off = {"haiku": "disabled", "sonnet": "between_tools"}
    c = L.LLMClient(tmp_path / "cache.sqlite", "live", thinking="off", thinking_off=off)
    c._client = fake
    c.run([L.Call("haiku", "s", "q", SCHEMA), L.Call("sonnet", "s", "q", SCHEMA)])
    assert sorted(kw["thinking"]["type"] for kw in sent) == ["between_tools", "disabled"]
    call = L.Call("haiku", "s", "q", SCHEMA)
    assert call.key("low", 10, "off") != call.key("low", 10, "adaptive") == call.key("low", 10)
    with pytest.raises(ValueError):
        c._params(L.Call("unknown-model", "s", "q", SCHEMA))
