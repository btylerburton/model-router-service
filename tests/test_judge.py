"""Tests for the LLM judge: JSON parsing/validation and the decide() path
against a stubbed httpx client (no real network)."""

from __future__ import annotations

import json

import httpx
import pytest

from model_router_service.judge import Judge, _parse_decision
from model_router_service.scorer import ModelProfile


def _candidates():
    return [
        ModelProfile("haiku", "Haiku", frozenset({"coding", "fast", "cheap"}), 200000, 1),
        ModelProfile("sonnet", "Sonnet", frozenset({"coding", "reasoning"}), 200000, 3),
    ]


def test_parse_plain_json():
    d = _parse_decision('{"model":"sonnet","why":"needs reasoning"}', {"haiku", "sonnet"})
    assert d.model == "sonnet"
    assert "reasoning" in d.why


def test_parse_fenced_json():
    content = "```json\n{\"model\":\"haiku\",\"why\":\"trivial\"}\n```"
    d = _parse_decision(content, {"haiku", "sonnet"})
    assert d.model == "haiku"


def test_parse_with_prose_prefix():
    content = 'Sure! Here is my choice: {"model":"haiku","why":"simple"} hope that helps'
    d = _parse_decision(content, {"haiku", "sonnet"})
    assert d.model == "haiku"


def test_parse_rejects_hallucinated_id():
    with pytest.raises(ValueError):
        _parse_decision('{"model":"gpt-9","why":"x"}', {"haiku", "sonnet"})


def test_parse_rejects_malformed():
    with pytest.raises(Exception):
        _parse_decision("not json at all", {"haiku"})


async def test_decide_calls_upstream_and_returns_choice():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        # The judge must send the judge model + a system+user message pair.
        assert body["model"] == "judge-cheap"
        assert body["messages"][0]["role"] == "system"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"model":"sonnet","why":"hard task"}'}}]},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        judge = Judge(
            base_url="https://upstream.example/v1",
            api_key="k",
            model="judge-cheap",
            timeout_s=4,
            cache_size=8,
        )
        d = await judge.decide("redesign the architecture", _candidates(), client=client)
        assert d.model == "sonnet"
        assert d.via == "judge"


async def test_decide_is_cached():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"model":"haiku","why":"simple"}'}}]},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        judge = Judge(base_url="https://u/v1", api_key="k", model="j", timeout_s=4, cache_size=8)
        await judge.decide("fix a typo", _candidates(), client=client)
        await judge.decide("fix a typo", _candidates(), client=client)
    assert calls["n"] == 1  # second call served from cache
