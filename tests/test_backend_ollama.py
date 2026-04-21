"""Tests for OllamaBackend extract_chunk / extract_result (pure dict-based)."""

from ochat.backend.ollama import OllamaBackend


def _backend():
    # Construction only stores config & builds an AsyncClient — no network call.
    return OllamaBackend(host="http://localhost:11434", num_ctx=4096)


class TestExtractChunk:
    def test_basic_content(self):
        b = _backend()
        reasoning, content = b.extract_chunk({"message": {"content": "hello"}})
        assert reasoning == ""
        assert content == "hello"

    def test_thinking_field(self):
        b = _backend()
        reasoning, content = b.extract_chunk(
            {"message": {"content": "answer", "thinking": "let me think"}}
        )
        assert reasoning == "let me think"
        assert content == "answer"

    def test_empty_message(self):
        b = _backend()
        reasoning, content = b.extract_chunk({"message": {}})
        assert reasoning == ""
        assert content == ""

    def test_missing_message_key(self):
        b = _backend()
        reasoning, content = b.extract_chunk({})
        assert reasoning == ""
        assert content == ""

    def test_none_values_coerced_to_empty_string(self):
        b = _backend()
        reasoning, content = b.extract_chunk(
            {"message": {"content": None, "thinking": None}}
        )
        assert reasoning == ""
        assert content == ""

    def test_eval_count_updates_context_tokens(self):
        b = _backend()
        assert b.context_tokens == 0
        b.extract_chunk({"message": {"content": "x"}, "eval_count": 42})
        assert b.context_tokens == 42

    def test_eval_count_zero_does_not_update(self):
        """Guard: eval_count=0 is falsy and should NOT overwrite a prior value."""
        b = _backend()
        b._context_tokens = 100
        b.extract_chunk({"message": {"content": "x"}, "eval_count": 0})
        assert b.context_tokens == 100


class TestExtractResult:
    def test_basic_with_eval_count(self):
        b = _backend()
        content, tokens = b.extract_result({
            "message": {"content": "hello world"},
            "eval_count": 7,
            "prompt_eval_count": 3,
        })
        assert content == "hello world"
        assert tokens == 7
        assert b.context_tokens == 7

    def test_fallback_token_count_when_eval_count_missing(self):
        b = _backend()
        # 16 chars // 4 = 4
        content, tokens = b.extract_result({"message": {"content": "a" * 16}})
        assert content == "a" * 16
        assert tokens == 4

    def test_empty_content(self):
        b = _backend()
        content, tokens = b.extract_result({"message": {"content": ""}, "eval_count": 0})
        assert content == ""
        assert tokens == 0
