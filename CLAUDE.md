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
- `config.conf` - INI format settings (host, model, num_ctx, personality, streaming, append_local_prompt)
- `personalities/` - System prompt templates (`.md` files)

## Usage

```bash
ochat                    # Use saved config
ochat -m <model>         # Override model
ochat -c <num_ctx>       # Override context size
ochat -s <file>          # Use specific system prompt file
ochat -C / --config      # Interactive setup wizard
```

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
- `/project` - Toggle local prompt append

## Personalities

System prompt templates in `~/.config/ollama-chat/personalities/`. Bundled personalities (default, creative, storyteller) are copied on first run.

**Local prompt append** (`append_local_prompt`):
- ON: system prompt + local `agent.md`/`system.md` appended
- OFF: system prompt only, local files ignored

## Tech stack

- Python 3.11 (required for type syntax)
- Textual for TUI
- ollama-python for API

## Notes for future development

- Token count is approximated (1 chunk ≈ 1 token) in streaming mode
- No conversation persistence yet (could add /save, /load commands)
- No multi-turn editing (could add message editing)
