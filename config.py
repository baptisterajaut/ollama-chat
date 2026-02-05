"""Configuration management for ollama-chat."""

import configparser
import os
import subprocess
import sys
from pathlib import Path

import ollama

# Config paths
CONFIG_DIR = Path.home() / ".config" / "ollama-chat"
CONFIG_FILE = CONFIG_DIR / "config.conf"
PERSONALITIES_DIR = CONFIG_DIR / "personalities"

# Bundled personalities (shipped with the app)
BUNDLED_PERSONALITIES_DIR = Path(__file__).parent / "personalities"


def get_default_host() -> str:
    """Get default host from environment or fallback."""
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def ensure_personalities_dir() -> None:
    """Ensure personalities directory exists with bundled personalities."""
    PERSONALITIES_DIR.mkdir(parents=True, exist_ok=True)

    # Copy bundled personalities if user folder is empty
    existing = list(PERSONALITIES_DIR.glob("*.md"))
    if not existing and BUNDLED_PERSONALITIES_DIR.exists():
        for src in BUNDLED_PERSONALITIES_DIR.glob("*.md"):
            dst = PERSONALITIES_DIR / src.name
            if not dst.exists():
                dst.write_text(src.read_text())


def list_personalities() -> list[str]:
    """List available personality names (without .md extension)."""
    ensure_personalities_dir()
    return sorted([
        p.stem for p in PERSONALITIES_DIR.glob("*.md")
        if p.is_file()
    ])


def load_personality(name: str) -> str | None:
    """Load a personality by name."""
    ensure_personalities_dir()
    path = PERSONALITIES_DIR / f"{name}.md"
    if path.exists():
        return path.read_text().strip()
    return None


def load_config() -> dict:
    """Load configuration from file."""
    config = {
        "host": get_default_host(),
        "model": "llama3.2",
        "num_ctx": 4096,
        "personality": "default",
        "append_local_prompt": True,
        "streaming": True,
        "model_options": {},
    }

    if CONFIG_FILE.exists():
        parser = configparser.ConfigParser()
        parser.read(CONFIG_FILE)

        if parser.has_option("server", "host"):
            config["host"] = parser.get("server", "host")
        if parser.has_option("defaults", "model"):
            config["model"] = parser.get("defaults", "model")
        if parser.has_option("defaults", "num_ctx"):
            config["num_ctx"] = parser.getint("defaults", "num_ctx")
        if parser.has_option("defaults", "personality"):
            config["personality"] = parser.get("defaults", "personality")
        if parser.has_option("defaults", "append_local_prompt"):
            config["append_local_prompt"] = parser.getboolean("defaults", "append_local_prompt")
        if parser.has_option("defaults", "streaming"):
            config["streaming"] = parser.getboolean("defaults", "streaming")

        # Load model options (empty string = not set)
        if parser.has_section("model_options"):
            model_options = {}
            for key in ["temperature", "top_p", "top_k", "min_p", "repeat_penalty"]:
                if parser.has_option("model_options", key):
                    val = parser.get("model_options", key).strip()
                    if val:
                        model_options[key] = float(val)
            config["model_options"] = model_options

    return config


def save_config(
    host: str,
    model: str,
    num_ctx: int,
    personality: str = "default",
    append_local_prompt: bool = True,
    streaming: bool = True,
    model_options: dict | None = None,
) -> None:
    """Save configuration to file."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    parser = configparser.ConfigParser()
    parser["server"] = {"host": host}
    parser["defaults"] = {
        "model": model,
        "num_ctx": str(num_ctx),
        "personality": personality,
        "append_local_prompt": str(append_local_prompt).lower(),
        "streaming": str(streaming).lower(),
    }

    # Advanced model options (empty = inherit from model)
    if model_options is None:
        model_options = {}
    parser["model_options"] = {
        "temperature": str(model_options.get("temperature", "")),
        "top_p": str(model_options.get("top_p", "")),
        "top_k": str(model_options.get("top_k", "")),
        "min_p": str(model_options.get("min_p", "")),
        "repeat_penalty": str(model_options.get("repeat_penalty", "")),
    }

    with open(CONFIG_FILE, "w") as f:
        parser.write(f)


def save_personality_choice(personality: str) -> None:
    """Update only the personality in config, preserving other settings."""
    config = load_config()
    save_config(
        config["host"], config["model"], config["num_ctx"],
        personality, config["append_local_prompt"], config["streaming"],
        config["model_options"]
    )


def save_append_local_prompt(append_local_prompt: bool) -> None:
    """Update only the append_local_prompt setting."""
    config = load_config()
    save_config(
        config["host"], config["model"], config["num_ctx"],
        config["personality"], append_local_prompt, config["streaming"],
        config["model_options"]
    )


def save_streaming(streaming: bool) -> None:
    """Update only the streaming setting."""
    config = load_config()
    save_config(
        config["host"], config["model"], config["num_ctx"],
        config["personality"], config["append_local_prompt"], streaming,
        config["model_options"]
    )


def find_project_prompt() -> str | None:
    """Find project prompt file in current directory."""
    for name in ["agent.md", "system.md", "system.txt", "AGENT.md"]:
        p = Path(name)
        if p.exists():
            return p.read_text().strip()
    return None


def load_system_prompt(
    path: str | None,
    personality: str | None = None,
    append_local_prompt: bool = True,
) -> tuple[str | None, str | None]:
    """Load system prompt.

    Modes:
    - If -s path provided: use that file alone
    - If append_local_prompt=True: personality + project file (agent.md, etc.)
    - If append_local_prompt=False: personality alone

    Returns:
        (system_prompt_content, personality_name or None)
    """
    if path:
        p = Path(path)
        if p.exists():
            return p.read_text().strip(), None

    ensure_personalities_dir()
    personality_name = personality or "default"
    personality_content = load_personality(personality_name)
    if not personality_content:
        personality_content = load_personality("default")
        personality_name = "default"

    if append_local_prompt:
        project_prompt = find_project_prompt()
        if project_prompt:
            combined = f"{personality_content}\n\n---\n\n{project_prompt}"
            return combined, personality_name

    return personality_content, personality_name


def run_setup() -> None:
    """Run interactive setup wizard."""
    print("ollama-chat configuration\n")

    config = load_config()

    # 1. Ask for host
    default_host = config["host"]
    host = input(f"Host Ollama [{default_host}]: ").strip()
    if not host:
        host = default_host

    # 2. List and select model
    print("\nConnecting to Ollama...")
    try:
        client = ollama.Client(host=host)
        models_response = client.list()
        models = [m.model for m in models_response.models]
    except Exception as e:
        print(f"Connection error: {e}")
        sys.exit(1)

    if not models:
        print("No models found. Install one with 'ollama pull <model>'")
        sys.exit(1)

    print("\nAvailable models:")
    for i, m in enumerate(models, 1):
        print(f"  {i}. {m}")

    default_idx = 1
    if config["model"] in models:
        default_idx = models.index(config["model"]) + 1

    while True:
        choice = input(f"\nSelect model [{default_idx}]: ").strip()
        if not choice:
            choice = str(default_idx)
        try:
            idx = int(choice)
            if 1 <= idx <= len(models):
                model = models[idx - 1]
                break
            print(f"Invalid choice. Enter a number between 1 and {len(models)}")
        except ValueError:
            print("Enter a valid number")

    # 3. Ask for context size
    default_ctx = config["num_ctx"]
    while True:
        ctx_input = input(f"\nContext size [{default_ctx}]: ").strip()
        if not ctx_input:
            num_ctx = default_ctx
            break
        try:
            num_ctx = int(ctx_input)
            if num_ctx > 0:
                break
            print("Size must be positive")
        except ValueError:
            print("Enter a valid number")

    # 4. Select personality
    ensure_personalities_dir()
    personalities = list_personalities()

    print("\nAvailable personalities:")
    for i, p in enumerate(personalities, 1):
        marker = " (current)" if p == config["personality"] else ""
        print(f"  {i}. {p}{marker}")

    default_p_idx = 1
    if config["personality"] in personalities:
        default_p_idx = personalities.index(config["personality"]) + 1

    while True:
        p_choice = input(f"\nSelect personality [{default_p_idx}]: ").strip()
        if not p_choice:
            p_choice = str(default_p_idx)
        try:
            p_idx = int(p_choice)
            if 1 <= p_idx <= len(personalities):
                personality = personalities[p_idx - 1]
                break
            print(f"Invalid choice. Enter a number between 1 and {len(personalities)}")
        except ValueError:
            print("Enter a valid number")

    # 5. Edit personality?
    edit_prompt = input(f"\nEdit personality '{personality}'? [y/N]: ").strip().lower()
    if edit_prompt in ("y", "yes"):
        editor = os.environ.get("EDITOR", "nano")
        subprocess.run([editor, str(PERSONALITIES_DIR / f"{personality}.md")])

    # 6. Append local prompt?
    print(f"\nAppend local prompt (agent.md, system.md from current directory)?")
    print("  If yes: system prompt + local file merged")
    print("  If no: system prompt only")
    project_choice = input(f"Append local prompt? [{'Y/n' if config['append_local_prompt'] else 'y/N'}]: ").strip().lower()
    if project_choice in ("y", "yes"):
        append_local_prompt = True
    elif project_choice in ("n", "no"):
        append_local_prompt = False
    else:
        append_local_prompt = config["append_local_prompt"]

    # 7. Save config
    save_config(host, model, num_ctx, personality, append_local_prompt, config["streaming"], config["model_options"])
    print(f"\nConfiguration saved to {CONFIG_FILE}")
    print(f"Personalities in {PERSONALITIES_DIR}/")
