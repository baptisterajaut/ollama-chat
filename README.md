# ollama-chat

*Made with Claude to talk with non-Claude.*

A simple Ollama chat TUI.

![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-public%20domain-brightgreen)
![Vibe](https://img.shields.io/badge/vibe-coded-ff69b4)

## Why?

I wanted something like [Open WebUI](https://github.com/open-webui/open-webui) but in terminal. A basic chat interface without needing a full browser eating 8GB of RAM just to exist.

Features I needed: model-level settings, customizable system prompts/personalities, and a clean interface. Couldn't find anything that did this simply, especially the personalities part.

I also wanted a TUI as pleasant to use as [Claude Code](https://claude.com/claude-code) but for local Ollama models.

This project was entirely vibe-coded with Claude. It does exactly what I need, nothing more.

## Features

- Clean TUI with streaming responses
- Switchable personalities (system prompts)
- Project-specific prompts (auto-loads `agent.md`/`system.md` from current directory)
- Slash commands (`/help`, `/retry`, `/personality`, etc.)
- Keyboard shortcuts (Ctrl+O toggle streaming, Escape to cancel, etc.)
- Persistent configuration
- SOME Advanced model options (temperature, top_p, top_k, etc.) via config file

## Non-features

- No SSL support
- No provider abstraction (Ollama hardcoded)
- No conversation memory/persistence
- No multi-model conversations
- No model templates (chat templates are handled by Ollama)
- No RAG, no agents, no tools

It's basic. That's the point.

## Installation

```bash
git clone https://github.com/youruser/ollama-chat
cd ollama-chat

# Optional: symlink to your bin
ln -s "$(pwd)/ochat" ~/bin/ochat
```

The virtual environment and dependencies are created automatically on first run.

### Windows (untested)

```cmd
git clone https://github.com/youruser/ollama-chat
cd ollama-chat
ochat.bat
```

Windows support is experimental and untested. The `/copy` command should work (uses `clip.exe`).

## Usage

```bash
ochat                # Start chatting (first run launches setup wizard)
ochat -C             # Configure (host, model, personality)
ochat -m llama3.2    # Override model
ochat --help         # Show options
```

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| `Ctrl+C` | Clear input |
| `Ctrl+D` | Quit |
| `Ctrl+L` | Clear chat |
| `Ctrl+O` | Toggle streaming |
| `Escape` | Cancel generation / Quit |
| `Tab` | Autocomplete command |

### Slash commands

| Command | Action |
|---------|--------|
| `/help` | Show help |
| `/retry` | Regenerate last response |
| `/copy` | Copy last response to clipboard |
| `/clear` | Clear chat history |
| `/personality` | List/switch personalities |
| `/project` | Toggle project prompt merge |
| `/system` | Show current system prompt |
| `/model` | Show current model |
| `/context` | Show context info |

## Configuration

Config lives in `~/.config/ollama-chat/`:
- `config.conf` - Settings (host, model, context size, etc.)
- `personalities/` - System prompt templates (`.md` files)

Bundled personalities (copied on first run):
- `default` - Helpful, concise assistant
- `creative` - Brainstorming and unconventional ideas
- `storyteller` - Narrative and creative writing

### Advanced model options

You can add model parameters directly in `config.conf` (empty = inherit from model defaults):

```ini
[model_options]
temperature =
top_p =
top_k =
min_p =
repeat_penalty =
```

These are not exposed in the setup wizard - edit the config file manually if needed.

## License

Public domain. You can't copyright AI-generated code anyway.

---

## "Vibe coding is awful, LLMs steal real code, you should've coded it yourself"

*Frankly my dear, I don't give a damn.*
