"""Tests for OpenAI + llama.cpp backends — extract_chunk / extract_result.

Both backends consume the OpenAI streaming schema (object attribute access, not
dicts). We build lightweight duck-typed stubs with `SimpleNamespace` rather
than constructing real `openai.types.chat.ChatCompletionChunk` objects.
"""

from types import SimpleNamespace

from ochat.backend.openai import OpenAIBackend
from ochat.backend.llama_cpp import LlamaCppBackend


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _chunk(content=None, reasoning=None, prompt_tokens=None, no_choices=False):
    """Build a stream chunk stub duck-typing an openai ChatCompletionChunk."""
    if no_choices:
        choices = []
    else:
        delta = SimpleNamespace(content=content, reasoning_content=reasoning)
        choices = [SimpleNamespace(delta=delta)]
    usage = None
    if prompt_tokens is not None:
        usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=0)
    return SimpleNamespace(choices=choices, usage=usage)


def _result(content, prompt_tokens=None, completion_tokens=None):
    message = SimpleNamespace(content=content)
    choices = [SimpleNamespace(message=message)]
    usage = None
    if prompt_tokens is not None:
        usage = SimpleNamespace(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens or 0,
        )
    return SimpleNamespace(choices=choices, usage=usage)


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------


class TestOpenAIExtractChunk:
    def test_content_only(self):
        b = OpenAIBackend()
        reasoning, content = b.extract_chunk(_chunk(content="hello"))
        assert reasoning == ""
        assert content == "hello"

    def test_reasoning_content(self):
        b = OpenAIBackend()
        reasoning, content = b.extract_chunk(_chunk(content="answer", reasoning="thinking"))
        assert reasoning == "thinking"
        assert content == "answer"

    def test_none_content_becomes_empty_string(self):
        b = OpenAIBackend()
        _, content = b.extract_chunk(_chunk(content=None))
        assert content == ""

    def test_usage_updates_context_tokens(self):
        b = OpenAIBackend()
        assert b.context_tokens == 0
        b.extract_chunk(_chunk(content="x", prompt_tokens=123))
        assert b.context_tokens == 123


class TestOpenAIExtractResult:
    def test_basic_with_usage(self):
        b = OpenAIBackend()
        content, tokens = b.extract_result(
            _result("hello", prompt_tokens=10, completion_tokens=5)
        )
        assert content == "hello"
        assert tokens == 15
        assert b.context_tokens == 10

    def test_fallback_when_usage_missing(self):
        b = OpenAIBackend()
        content, tokens = b.extract_result(_result("a" * 20))
        assert content == "a" * 20
        assert tokens == 5  # 20 // 4


# ---------------------------------------------------------------------------
# llama.cpp backend
# ---------------------------------------------------------------------------


class TestLlamaCppExtractChunk:
    def test_content_only(self):
        b = LlamaCppBackend()
        reasoning, content = b.extract_chunk(_chunk(content="hello"))
        assert reasoning == ""
        assert content == "hello"

    def test_reasoning_content(self):
        b = LlamaCppBackend()
        reasoning, content = b.extract_chunk(
            _chunk(content="answer", reasoning="thinking")
        )
        assert reasoning == "thinking"
        assert content == "answer"

    def test_last_chunk_no_choices_carries_usage(self):
        b = LlamaCppBackend()
        reasoning, content = b.extract_chunk(_chunk(no_choices=True, prompt_tokens=42))
        assert reasoning == ""
        assert content == ""
        assert b.context_tokens == 42

    def test_last_chunk_no_choices_no_usage(self):
        b = LlamaCppBackend()
        reasoning, content = b.extract_chunk(_chunk(no_choices=True))
        assert reasoning == ""
        assert content == ""
        assert b.context_tokens == 0

    def test_usage_updates_context_tokens_mid_stream(self):
        b = LlamaCppBackend()
        b.extract_chunk(_chunk(content="x", prompt_tokens=77))
        assert b.context_tokens == 77


class TestLlamaCppExtractResult:
    def test_basic_with_usage(self):
        b = LlamaCppBackend()
        content, tokens = b.extract_result(
            _result("hello", prompt_tokens=10, completion_tokens=5)
        )
        assert content == "hello"
        assert tokens == 15
        assert b.context_tokens == 10

    def test_fallback_when_usage_missing(self):
        b = LlamaCppBackend()
        content, tokens = b.extract_result(_result("a" * 40))
        assert content == "a" * 40
        assert tokens == 10  # 40 // 4
