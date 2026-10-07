"""Utility-call detection: harness title-gen/summarize/compaction vs real tasks."""
from model_router_service.utility import is_utility_call

OPENCODE_SYS = "You are OpenCode, a deeply pragmatic software engineer. Communicate efficiently."


def test_detects_title_generator():
    assert is_utility_call("You are a title generator. You output ONLY a thread title.")


def test_detects_brief_title_instruction():
    assert is_utility_call("Generate a brief title that would help the user find this later.")


def test_detects_conversation_summarizer():
    assert is_utility_call("Summarize this conversation into a short paragraph.")


def test_real_opencode_persona_is_not_utility():
    # The normal agent persona must NOT be treated as a utility call.
    assert not is_utility_call(OPENCODE_SYS)


def test_empty_is_not_utility():
    assert not is_utility_call("")
    assert not is_utility_call(None)  # type: ignore[arg-type]
