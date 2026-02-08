# ollama-chat

Simple TUI chat client for Ollama with custom system prompt support.

## Project structure

```
ollama-chat/
├── chat.py          # Main application (Textual TUI)
├── chat.tcss        # Textual CSS styles
├── config.py        # Configuration management, personalities
├── system_instructions.json  # LLM instructions for commands (compact, impersonate)
├── ochat            # Launcher script (symlinked to ~/bin/ochat)
├── requirements.txt # Dependencies: textual, ollama
├── personalities/   # Bundled personality templates (copied on first run)
│   ├── default.md
│   ├── creative.md
│   └── storyteller.md
└── .venv/           # Python 3.11 virtual environment
```

## Configuration

Config stored in `~/.config/ollama-chat/`:
- `config.conf` - INI format settings (host, verify_ssl, model, num_ctx, personality, streaming, append_local_prompt, config_name)
- `*.conf` - Named config profiles (backups created via setup wizard or `--as-default`)
- `personalities/` - System prompt templates (`.md` files)

## API modes

The app supports two API modes:

1. **Ollama mode** (default): Uses the Ollama Python client. Supports model listing, `num_ctx`, and `model_options`.

2. **OpenAI mode** (fallback, untested): If Ollama's `/api/tags` endpoint fails, falls back to the `openai` Python client with custom `base_url`. Works with LM Studio, llama.cpp, vLLM, etc. Limitations:
   - No interactive model selection (model listing is used for connection testing only — must configure model name manually)
   - `num_ctx` and `model_options` are ignored
   - Context usage tracking is disabled (percentages and warnings don't apply)
   - Setup wizard won't work (edit config.conf manually)

Detection happens at startup in `_show_greeting()`. The `self.api_mode` attribute is set to `"ollama"` or `"openai"`. The OpenAI client is lazily initialized (only created when Ollama connection fails).

## Usage

```bash
ochat                              # Use saved config
ochat -m <model>                   # Override model
ochat -c <num_ctx>                 # Override context size
ochat -s <file>                    # Use specific system prompt file
ochat -C / --config                # Interactive setup wizard (modify current)
ochat --new                        # Create a new named config profile
ochat --use-config <name>          # Use named config for this session
ochat --use-config <name> --as-default  # Switch to named config as new default
ochat -d                               # Enable debug logging
```

## Multiple configs

Named config profiles allow switching between different setups (models, personalities, etc.):

1. **Create new config**: `--new` creates a fresh named profile from scratch
2. **Or backup existing**: Run `-C`, then backup when prompted
3. **Use temporarily**: `--use-config mistral-creative` loads for one session only
4. **Switch default**: `--use-config mistral-creative --as-default` backs up current config and switches
5. **Switch in-app**: `/config` lists profiles, `/config <n>` switches and restarts

When switching with `--as-default`:
- Current `config.conf` is backed up to `{config_name}.conf` (prompts for name if unset from CLI, defaults to "config-default" from TUI)
- Asks confirmation before overwriting existing backup (CLI only)
- The chosen config becomes the new `config.conf`

Config name is shown in greeting: `config: mistral-creative · Connected`

## Keyboard shortcuts

- **Ctrl+C** - Clear input
- **Ctrl+D** - Quit
- **Ctrl+L** - Clear chat history
- **Ctrl+O** - Toggle streaming mode (saved to config)
- **Escape** - Cancel generation / Quit

## Slash commands

- `/help`, `/h` - Show help
- `/clear`, `/c` - Clear chat history
- `/retry`, `/r` - Regenerate last response
- `/copy` - Copy last response to clipboard
- `/context`, `/ctx` - Show context info
- `/prompt` - Show system prompt
- `/sys <msg>`, `/system <msg>` - Inject a system message into conversation
- `/model`, `/m` - Show current model
- `/personality`, `/p` - List personalities
- `/p <n>` - Switch to personality by number/name
- `/config` - List config profiles
- `/config <n>` - Switch to config by number/name (restarts app)
- `/impersonate`, `/imp` - Generate user response suggestion (for RP)
- `/project` - Toggle local prompt append
- `/stats`, `/st` - Show generation statistics (TTFT, t/s, tokens)
- `/compact` - Summarize conversation to free context

## System instructions

LLM instructions for commands (compact, impersonate) are stored in `system_instructions.json` at the project root. Users can edit this file to customize the instructions. The app exits with an error if the file is missing or invalid JSON.

## Personalities

System prompt templates in `~/.config/ollama-chat/personalities/`. Bundled personalities (default, creative, storyteller) are copied on first run.

**Local prompt append** (`append_local_prompt`):
- ON: system prompt + local `agent.md`/`system.md` appended
- OFF: system prompt only, local files ignored

## Impersonate (RP feature)

`/impersonate` or `/imp` generates a suggested user response using the LLM:
- Useful for roleplay with gamemaster personality
- LLM generates what the user/player might say next
- Result is placed in input field (not sent automatically)
- User can edit or send directly

## Tech stack

- Python 3.11 (required for type syntax)
- Textual for TUI
- ollama-python for Ollama API
- openai for OpenAI-compatible API fallback

## Debugging

Logging is disabled by default. Enable with `-d` flag:
```bash
ochat -d
```

Log files are written to temp (auto-cleaned after 7 days):
- **Unix/Mac**: `/tmp/ollama-chat-YYYYMMDD-HHMMSS.log`
- **Windows**: `%TEMP%/ollama-chat-YYYYMMDD-HHMMSS.log`

Captures: app start, commands, errors, Textual exceptions.

## Gotchas

- **ollama-python global client**: Never use `ollama.chat()`, `ollama.list()` etc. (module-level functions). The library creates its default client at import time using `OLLAMA_HOST` env var — setting the env var after `import ollama` has no effect. Always use an explicit `ollama.Client(host=...)` instance (`self.client` in `OllamaChat`). Host priority: config.conf `[server] host` > `OLLAMA_HOST` env var > `http://localhost:11434`.

- **SSL verification**: `verify_ssl` in `[server]` controls SSL certificate verification. Only written to config when `false` (default is `true`/absent). When `false`, `ollama.Client` gets `verify=False` and `openai.OpenAI` gets a custom `httpx.Client(verify=False)`. The setup wizard auto-detects SSL errors on HTTPS hosts and offers to disable verification.

- **OpenAI client lazy init**: `self.openai_client` is a `@property` that lazily creates the OpenAI client on first access. Don't assign to it directly.

- **OpenAI mode**: API mode branching is handled by three helpers: `self._chat_call(messages, stream)` makes the API call, `self._extract_chunk(chunk)` extracts text from streaming chunks, and `self._extract_result(result)` extracts (content, token_count) from non-streaming results. When adding features that call the LLM, use these helpers instead of calling `self.ollama_client` or `self.openai_client` directly. The OpenAI mode uses the official `openai` Python client with a custom `base_url`.

- **Async stream iteration**: Stream chunks are fetched via `self._anext(stream)` which wraps `next()` in `asyncio.to_thread` to avoid blocking the Textual event loop. Never iterate a stream synchronously (`for chunk in stream`) in async code.

## Generation modes

Both streaming and non-streaming modes use `stream=True` under the hood:

- **Streaming**: shows text live as tokens arrive
- **Non-streaming**: buffers tokens, shows `⠋ thinking... 3.2s (42 chunks)` progress, then displays the full response prefixed with `*thought for 3.2s*`

Both modes start with a "waiting for first token" spinner and track TTFT.

## Context tracking

- Token count is approximated (~4 chars/token from all messages, 1 chunk ≈ 1 token for generated output)
- Context tracking is disabled in OpenAI mode (`_context_pct()` returns 0) since `num_ctx` is not applicable
- Context usage percentage shown in `/context` and `/stats`
- Status bar shows `ctx: N% remaining` when usage exceeds 85%
- Status bar shows `⚠ Context length probably exceeded` when over 100%
- One-shot system message warning at 80% usage suggesting `/compact`

## Notes for future development

- No conversation persistence yet (could add /save, /load commands)
- No multi-turn editing (could add message editing)
- No multiline input (Input widget limitation, would need TextArea)
