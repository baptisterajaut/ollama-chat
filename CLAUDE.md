# ollama-chat

Simple TUI chat client for Ollama with custom system prompt support.

## Project structure

```
ollama-chat/
├── chat.py          # Main application (Textual TUI)
├── chat.tcss        # Textual CSS styles
├── config.py        # Configuration management, personalities
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
- `config.conf` - INI format settings (host, model, num_ctx, personality, streaming, append_local_prompt, config_name)
- `*.conf` - Named config profiles (backups created via setup wizard or `--as-default`)
- `personalities/` - System prompt templates (`.md` files)

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
```

## Multiple configs

Named config profiles allow switching between different setups (models, personalities, etc.):

1. **Create new config**: `--new` creates a fresh named profile from scratch
2. **Or backup existing**: Run `-C`, then backup when prompted
3. **Use temporarily**: `--use-config mistral-creative` loads for one session only
4. **Switch default**: `--use-config mistral-creative --as-default` backs up current config and switches
5. **Switch in-app**: `/config` lists profiles, `/config <n>` switches and restarts

When switching with `--as-default`:
- Current `config.conf` is renamed to `{config_name}.conf` (prompts for name if unset, default: "config-default")
- Asks confirmation before overwriting existing backup
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
- `/system`, `/sys` - Show system prompt
- `/model`, `/m` - Show current model
- `/personality`, `/p` - List personalities
- `/p <n>` - Switch to personality by number/name
- `/config` - List config profiles
- `/config <n>` - Switch to config by number/name (restarts app)
- `/impersonate`, `/imp` - Generate user response suggestion (for RP)
- `/project` - Toggle local prompt append

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
- ollama-python for API

## Debugging

Logs are written to a temp file for debugging:
- **Unix/Mac**: `/tmp/ollama-chat-YYYYMMDD-HHMMSS.log`
- **Windows**: `%TEMP%/ollama-chat-YYYYMMDD-HHMMSS.log`

Captures: app start, commands, errors, Textual exceptions.

## Notes for future development

- Token count is approximated (1 chunk ≈ 1 token) in streaming mode
- No conversation persistence yet (could add /save, /load commands)
- No multi-turn editing (could add message editing)
- No multiline input (Input widget limitation, would need TextArea)
