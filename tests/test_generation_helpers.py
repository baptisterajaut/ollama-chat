"""Tests for pure helpers in ochat.generation and ochat.app."""

import pytest

from ochat.generation import _clean_impersonate_response
from ochat.app import OChat


class TestCleanImpersonateResponse:
    def test_empty_string(self):
        assert _clean_impersonate_response("") == ""

    def test_already_clean(self):
        assert _clean_impersonate_response("Hello world") == "Hello world"

    def test_strips_surrounding_double_quotes(self):
        assert _clean_impersonate_response('"Hello world"') == "Hello world"

    def test_strips_leading_and_trailing_whitespace(self):
        assert _clean_impersonate_response("   Hello world   ") == "Hello world"

    def test_collapses_internal_whitespace(self):
        assert _clean_impersonate_response("Hello    world\t\tfoo") == "Hello world foo"

    def test_strips_trailing_newlines(self):
        assert _clean_impersonate_response("Hello world\n\n") == "Hello world"

    def test_strips_quotes_and_newlines_combined(self):
        assert _clean_impersonate_response('\n"Hello world"\n') == "Hello world"

    def test_does_not_strip_unbalanced_quote_left(self):
        # Only strips when both ends are quotes
        assert _clean_impersonate_response('"Hello world') == '"Hello world'

    def test_does_not_strip_unbalanced_quote_right(self):
        assert _clean_impersonate_response('Hello world"') == 'Hello world"'

    def test_strips_single_wrapper_quote_only(self):
        # Only one level of wrapper is stripped
        assert _clean_impersonate_response('""Hello""') == '"Hello"'

    def test_unicode_preserved(self):
        assert _clean_impersonate_response("Héllo wörld — café") == "Héllo wörld — café"

    def test_multiline_collapsed(self):
        assert _clean_impersonate_response("line 1\nline 2\nline 3") == "line 1 line 2 line 3"

    def test_whitespace_only(self):
        assert _clean_impersonate_response("   \n\t  ") == ""

    def test_empty_quoted_string(self):
        assert _clean_impersonate_response('""') == ""


class _MessagesStub:
    """Minimal stand-in for OChat providing only .messages."""

    def __init__(self, messages):
        self.messages = messages


class TestEstimateContextTokens:
    # _estimate_context_tokens only touches self.messages, so we can call it
    # unbound with a stub instead of constructing a full OChat (which needs a
    # Textual event loop).

    def _estimate(self, messages):
        return OChat._estimate_context_tokens(_MessagesStub(messages))

    def test_empty_list(self):
        assert self._estimate([]) == 0

    def test_single_message_short(self):
        # len("hello") // 4 == 1
        assert self._estimate([{"role": "user", "content": "hello"}]) == 1

    def test_single_message_exact_multiple(self):
        # 16 chars / 4 = 4 tokens
        assert self._estimate([{"role": "user", "content": "a" * 16}]) == 4

    def test_multiple_messages_sum(self):
        msgs = [
            {"role": "system", "content": "a" * 100},  # 25
            {"role": "user", "content": "b" * 40},     # 10
            {"role": "assistant", "content": "c" * 8},  # 2
        ]
        assert self._estimate(msgs) == 25 + 10 + 2

    def test_short_content_rounds_down_to_zero(self):
        # len("abc") // 4 == 0
        assert self._estimate([{"role": "user", "content": "abc"}]) == 0

    def test_empty_content(self):
        assert self._estimate([{"role": "user", "content": ""}]) == 0

    def test_unicode_counts_chars_not_bytes(self):
        # "héllo" = 5 Python chars → 1 token (matches current implementation)
        assert self._estimate([{"role": "user", "content": "héllo"}]) == 1
