"""Tests for ochat.config: load/save round-trip, backup regex, system prompt loading."""

from pathlib import Path

import pytest

from ochat import config as cfg


# ---------------------------------------------------------------------------
# Backup name validation regex
# ---------------------------------------------------------------------------


class TestBackupNameRegex:
    @pytest.mark.parametrize("name", [
        "mistral",
        "config-v2",
        "config.backup",
        "my_config",
        "config-default",
        "a",
        "123",
        "v1.2.3",
        "foo_bar-baz.qux",
    ])
    def test_valid_names(self, name):
        assert cfg._BACKUP_NAME_RE.match(name) is not None

    @pytest.mark.parametrize("name", [
        "../evil",
        "",
        "foo/bar",
        "foo bar",
        "foo\\bar",
        "foo:bar",
        "foo*",
        "foo?",
        "foo|bar",
        "foo\nbar",
    ])
    def test_invalid_names(self, name):
        assert cfg._BACKUP_NAME_RE.match(name) is None


# ---------------------------------------------------------------------------
# load_config / save_config round-trip
# ---------------------------------------------------------------------------


class TestLoadSaveRoundTrip:
    def test_save_then_load_preserves_all_fields(self, tmp_path):
        cf = tmp_path / "test.conf"
        cfg.save_config(
            host="http://example.com:11434",
            model="llama3.2:8b",
            num_ctx=8192,
            personality="creative",
            append_local_prompt=False,
            streaming=False,
            model_options={"temperature": 0.7, "top_k": 40, "stop": "###"},
            config_name="my-profile",
            config_file=cf,
            verify_ssl=False,
            auto_suggest=False,
            backend="llama_cpp",
        )
        loaded = cfg.load_config(cf)

        assert loaded["host"] == "http://example.com:11434"
        assert loaded["model"] == "llama3.2:8b"
        assert loaded["num_ctx"] == 8192
        assert loaded["personality"] == "creative"
        assert loaded["append_local_prompt"] is False
        assert loaded["streaming"] is False
        assert loaded["verify_ssl"] is False
        assert loaded["auto_suggest"] is False
        assert loaded["config_name"] == "my-profile"
        assert loaded["backend"] == "llama_cpp"
        assert loaded["model_options"] == {"temperature": 0.7, "top_k": 40, "stop": "###"}

    def test_load_missing_file_returns_defaults(self, tmp_path):
        cf = tmp_path / "does-not-exist.conf"
        loaded = cfg.load_config(cf)
        # All defaults from DEFAULT_CONFIG, with host filled in
        assert loaded["model"] == cfg.DEFAULT_CONFIG["model"]
        assert loaded["num_ctx"] == cfg.DEFAULT_CONFIG["num_ctx"]
        assert loaded["personality"] == cfg.DEFAULT_CONFIG["personality"]
        assert loaded["streaming"] == cfg.DEFAULT_CONFIG["streaming"]
        assert loaded["backend"] == cfg.DEFAULT_CONFIG["backend"]
        assert loaded["host"]  # filled by get_default_host()

    def test_save_minimal_then_load_defaults(self, tmp_path):
        """verify_ssl=True (default) should NOT be written; load should still return True."""
        cf = tmp_path / "minimal.conf"
        cfg.save_config(
            host="http://localhost:11434",
            model="llama3.2",
            num_ctx=4096,
            config_file=cf,
        )
        text = cf.read_text()
        # verify_ssl only written when False
        assert "verify_ssl" not in text
        loaded = cfg.load_config(cf)
        assert loaded["verify_ssl"] is True

    def test_model_options_type_coercion(self, tmp_path):
        cf = tmp_path / "types.conf"
        cfg.save_config(
            host="h", model="m", num_ctx=1,
            model_options={"a_bool": True, "an_int": 42, "a_float": 3.14, "a_str": "hello"},
            config_file=cf,
        )
        loaded = cfg.load_config(cf)
        assert loaded["model_options"]["a_bool"] is True
        assert loaded["model_options"]["an_int"] == 42
        assert loaded["model_options"]["a_float"] == 3.14
        assert loaded["model_options"]["a_str"] == "hello"

    def test_empty_model_option_values_are_skipped(self, tmp_path):
        cf = tmp_path / "empty.conf"
        # Hand-write a config with an empty value — save_config converts dict → str, but
        # a user might edit the file directly.
        cf.write_text(
            "[server]\nhost = h\n"
            "[defaults]\nmodel = m\nnum_ctx = 1\n"
            "[model_options]\ntemperature =\ntop_k = 40\n",
            encoding="utf-8",
        )
        loaded = cfg.load_config(cf)
        assert "temperature" not in loaded["model_options"]
        assert loaded["model_options"]["top_k"] == 40


# ---------------------------------------------------------------------------
# load_system_prompt
# ---------------------------------------------------------------------------


class TestLoadSystemPrompt:
    def test_missing_path_triggers_exit(self, tmp_path):
        missing = tmp_path / "does-not-exist.md"
        with pytest.raises(SystemExit) as exc_info:
            cfg.load_system_prompt(str(missing))
        assert exc_info.value.code == 1

    def test_valid_path_returns_content(self, tmp_path):
        f = tmp_path / "prompt.md"
        f.write_text("You are a helpful assistant.\n", encoding="utf-8")
        content, name = cfg.load_system_prompt(str(f))
        # .strip() is applied
        assert content == "You are a helpful assistant."
        # When -s is used, no personality name is returned
        assert name is None

    def test_personality_fallback_to_default(self, tmp_path, monkeypatch):
        # Redirect PERSONALITIES_DIR to tmp_path and write a default.md
        monkeypatch.setattr(cfg, "PERSONALITIES_DIR", tmp_path)
        monkeypatch.setattr(cfg, "BUNDLED_PERSONALITIES_DIR", tmp_path / "_empty_bundled")
        (tmp_path / "default.md").write_text("Default prompt", encoding="utf-8")

        # Ask for a non-existent personality → falls back to default
        content, name = cfg.load_system_prompt(
            path=None,
            personality="does-not-exist",
            append_local_prompt=False,
        )
        assert content == "Default prompt"
        assert name == "default"

    def test_personality_selected_when_exists(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "PERSONALITIES_DIR", tmp_path)
        monkeypatch.setattr(cfg, "BUNDLED_PERSONALITIES_DIR", tmp_path / "_empty_bundled")
        (tmp_path / "creative.md").write_text("Be creative.", encoding="utf-8")
        (tmp_path / "default.md").write_text("Default.", encoding="utf-8")

        content, name = cfg.load_system_prompt(
            path=None, personality="creative", append_local_prompt=False,
        )
        assert content == "Be creative."
        assert name == "creative"

    def test_append_local_prompt_combines_personality_and_project_file(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.setattr(cfg, "PERSONALITIES_DIR", tmp_path / "personalities")
        (tmp_path / "personalities").mkdir()
        (tmp_path / "personalities" / "default.md").write_text("Be helpful.", encoding="utf-8")
        monkeypatch.setattr(cfg, "BUNDLED_PERSONALITIES_DIR", tmp_path / "_empty_bundled")

        # load_project_prompt() uses cwd — chdir to a tmp dir with an agent.md
        proj = tmp_path / "project"
        proj.mkdir()
        (proj / "agent.md").write_text("Project context.", encoding="utf-8")
        monkeypatch.chdir(proj)

        content, name = cfg.load_system_prompt(
            path=None, personality="default", append_local_prompt=True,
        )
        assert "Be helpful." in content
        assert "Project context." in content
        assert "---" in content
        assert name == "default"
