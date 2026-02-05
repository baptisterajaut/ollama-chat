#!/usr/bin/env python3
"""Simple Ollama TUI chat with custom system prompt support."""

import argparse
import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

# Add script directory to path for imports when called from elsewhere
sys.path.insert(0, str(Path(__file__).parent))

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import ScrollableContainer, Vertical
from textual.suggester import Suggester
from textual.widgets import Footer, Input, Markdown, Static
import ollama


class CommandSuggester(Suggester):
    """Suggest slash commands."""

    COMMANDS = [
        "/help", "/h",
        "/clear", "/c",
        "/retry", "/r",
        "/copy",
        "/context", "/ctx",
        "/system", "/sys",
        "/model", "/m",
        "/personality", "/p",
        "/project",
    ]

    async def get_suggestion(self, value: str) -> str | None:
        if not value.startswith("/"):
            return None
        value_lower = value.lower()
        for cmd in self.COMMANDS:
            if cmd.startswith(value_lower) and cmd != value_lower:
                return cmd
        return None

from config import (
    CONFIG_FILE,
    load_config,
    load_system_prompt,
    load_personality,
    list_personalities,
    find_project_prompt,
    save_personality_choice,
    save_append_local_prompt,
    save_streaming,
    get_default_host,
    run_setup,
)


class Message(Markdown):
    """A single chat message with role-based styling."""

    def __init__(self, content: str, role: str = "user") -> None:
        self.role = role
        content = content.strip()
        if role == "assistant":
            content = f"● {content}"
        elif role == "user":
            content = f"› {content}"
        super().__init__(content, classes=f"message {role}")


class ChatContainer(ScrollableContainer):
    """Scrollable container for chat messages."""
    pass


class OllamaChat(App):
    """Simple TUI chat for Ollama."""

    CSS_PATH = "chat.tcss"

    BINDINGS = [
        Binding("ctrl+c", "clear_input", "Clear input", show=False),
        Binding("ctrl+d", "quit", "Quit"),
        Binding("ctrl+l", "clear", "Clear"),
        Binding("ctrl+o", "toggle_streaming", "Stream"),
        Binding("escape", "cancel_or_quit", "Cancel/Quit"),
        Binding("tab", "focus_input", show=False, priority=True),
        Binding("shift+tab", "focus_input", show=False, priority=True),
    ]

    def __init__(
        self,
        model: str = "llama3.2",
        system_prompt: str | None = None,
        num_ctx: int = 4096,
        personality_name: str | None = None,
        append_local_prompt: bool = True,
        streaming: bool = True,
        model_options: dict | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.system_prompt = system_prompt
        self.num_ctx = num_ctx
        self.personality_name = personality_name
        self.append_local_prompt = append_local_prompt
        self.model_options = model_options or {}
        self.messages: list[dict] = []
        self.is_generating = False
        self.streaming = streaming
        self._generation_cancelled = False
        self.total_tokens = 0
        self.last_gen_time = 0.0
        self.last_tokens = 0
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def compose(self) -> ComposeResult:
        yield ChatContainer(id="chat")
        yield Vertical(
            Input(
                placeholder="Message... (/help for commands)",
                id="chat-input",
                suggester=CommandSuggester(use_cache=False),
            ),
            id="input-container",
        )
        yield Static(self._status_text(), id="status")
        yield Footer()

    def _status_text(self, extra: str = "") -> str:
        base = f"{self.model} | ctx:{self.num_ctx}"
        if self.last_gen_time > 0:
            tps = self.last_tokens / self.last_gen_time if self.last_gen_time else 0
            base += f" | last: {self.last_gen_time:.1f}s ({self.last_tokens}tok, {tps:.1f}t/s)"
        if self.total_tokens > 0:
            base += f" | total: {self.total_tokens}tok"
        if extra:
            base += f" | {extra}"
        return base

    async def on_mount(self) -> None:
        self.query_one("#chat-input", Input).focus()
        await self._show_greeting()

    async def _show_greeting(self) -> None:
        """Show greeting with ASCII art after validating Ollama connection."""
        chat = self.query_one("#chat", ChatContainer)
        try:
            await asyncio.to_thread(lambda: ollama.list())
            logo = """\
 ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
  ___   ___| |__   __ _| |_
 / _ \\ / __| '_ \\ / _` | __|
| (_) | (__| | | | (_| | |_
 \\___/ \\___|_| |_|\\__,_|\\__|

Connected · /help for commands"""
            msg = Static(logo, classes="greeting")
            await chat.mount(msg)
            chat.scroll_end(animate=False)
        except Exception:
            await self._show_system_message("Warning: Cannot connect to Ollama")

    async def _show_system_message(self, text: str) -> None:
        """Show a system info message in the chat."""
        chat = self.query_one("#chat", ChatContainer)
        msg = Message(f"*{text}*", "system-info")
        await chat.mount(msg)
        chat.scroll_end(animate=False)

    async def _handle_command(self, cmd: str) -> bool:
        """Handle slash commands. Returns True if command was handled."""
        cmd = cmd.strip().lower()

        if cmd in ("/help", "/h", "/?"):
            help_text = """**Commands:**
- `/retry` or `/r` - Regenerate last response
- `/copy` - Copy last response to clipboard
- `/clear` or `/c` - Clear chat history
- `/context` or `/ctx` - Show current context info
- `/system` or `/sys` - Show system prompt
- `/model` or `/m` - Show current model
- `/personality` or `/p` - List/change personality
- `/project` - Toggle local prompt append (agent.md)
- `/help` or `/h` - Show this help"""
            await self._show_system_message(help_text)
            return True

        elif cmd in ("/retry", "/r"):
            await self._handle_retry()
            return True

        elif cmd == "/copy":
            await self._handle_copy()
            return True

        elif cmd in ("/clear", "/c"):
            self.action_clear()
            await self._show_system_message("Chat cleared.")
            return True

        elif cmd in ("/context", "/ctx"):
            msg_count = len([m for m in self.messages if m["role"] != "system"])
            info = f"Messages: {msg_count} | Tokens used: ~{self.total_tokens} | Context size: {self.num_ctx}"
            await self._show_system_message(info)
            return True

        elif cmd in ("/system", "/sys"):
            if self.system_prompt:
                sp = self.system_prompt
                if len(sp) > 500:
                    sp = sp[:500] + "..."
                await self._show_system_message(f"**System prompt:**\n{sp}")
            else:
                await self._show_system_message("No system prompt set.")
            return True

        elif cmd in ("/model", "/m"):
            await self._show_system_message(f"Model: `{self.model}`")
            return True

        elif cmd.startswith("/personality") or cmd.startswith("/p ") or cmd == "/p":
            await self._handle_personality_command(cmd)
            return True

        elif cmd == "/project":
            await self._handle_project_toggle()
            return True

        return False

    async def _handle_retry(self) -> None:
        """Regenerate the last assistant response."""
        if self.is_generating:
            return

        # Find and remove last assistant message
        if not self.messages or self.messages[-1]["role"] != "assistant":
            await self._show_system_message("Nothing to regenerate")
            return

        self.messages.pop()

        # Remove last message widget from chat
        chat = self.query_one("#chat", ChatContainer)
        children = list(chat.children)
        if children:
            await children[-1].remove()

        # Regenerate
        await self._generate_response()

    async def _handle_copy(self) -> None:
        """Copy last assistant response to clipboard."""
        # Find last assistant message
        for msg in reversed(self.messages):
            if msg["role"] == "assistant":
                content = msg["content"]
                try:
                    # macOS
                    proc = subprocess.Popen(
                        ["pbcopy"],
                        stdin=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                    )
                    proc.communicate(content.encode("utf-8"))
                    if proc.returncode == 0:
                        await self._show_system_message("Copied to clipboard")
                        return
                except FileNotFoundError:
                    pass

                try:
                    # Linux with xclip
                    proc = subprocess.Popen(
                        ["xclip", "-selection", "clipboard"],
                        stdin=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                    )
                    proc.communicate(content.encode("utf-8"))
                    if proc.returncode == 0:
                        await self._show_system_message("Copied to clipboard")
                        return
                except FileNotFoundError:
                    pass

                try:
                    # Windows with clip.exe
                    proc = subprocess.Popen(
                        ["clip"],
                        stdin=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                    )
                    proc.communicate(content.encode("utf-16"))
                    if proc.returncode == 0:
                        await self._show_system_message("Copied to clipboard")
                        return
                except FileNotFoundError:
                    pass

                await self._show_system_message("Copy failed: no clipboard tool found (pbcopy/xclip/clip)")
                return

        await self._show_system_message("Nothing to copy")

    async def _handle_project_toggle(self) -> None:
        """Toggle append_local_prompt and reload system prompt."""
        self.append_local_prompt = not self.append_local_prompt
        save_append_local_prompt(self.append_local_prompt)

        if self.personality_name:
            new_prompt, _ = load_system_prompt(
                None, self.personality_name, self.append_local_prompt
            )
            self.system_prompt = new_prompt

            self.messages = [m for m in self.messages if m["role"] != "system"]
            if new_prompt:
                self.messages.insert(0, {"role": "system", "content": new_prompt})

        project_file = find_project_prompt()
        status = "ON" if self.append_local_prompt else "OFF"
        if project_file and self.append_local_prompt:
            await self._show_system_message(f"Local prompt: **{status}** (appended to system prompt)")
        elif project_file:
            await self._show_system_message(f"Local prompt: **{status}** (ignored)")
        else:
            await self._show_system_message(f"Local prompt: **{status}** (no file found)")

    async def _handle_personality_command(self, cmd: str) -> None:
        """Handle /personality command for listing and switching personalities."""
        parts = cmd.split(maxsplit=1)
        personalities = list_personalities()

        if len(parts) == 1:
            lines = ["**Available personalities:**"]
            for i, p in enumerate(personalities, 1):
                marker = " ← current" if p == self.personality_name else ""
                lines.append(f"{i}. `{p}`{marker}")
            lines.append("\n*Tapez `/p <numéro>` pour changer*")
            await self._show_system_message("\n".join(lines))
            return

        choice = parts[1].strip()
        try:
            idx = int(choice)
            if 1 <= idx <= len(personalities):
                new_personality = personalities[idx - 1]
            else:
                await self._show_system_message(f"Invalid choice (1-{len(personalities)})")
                return
        except ValueError:
            if choice in personalities:
                new_personality = choice
            else:
                await self._show_system_message(f"Personality '{choice}' not found")
                return

        content = load_personality(new_personality)
        if not content:
            await self._show_system_message(f"Error: unable to load '{new_personality}'")
            return

        self.personality_name = new_personality
        self.system_prompt = content

        self.messages = [m for m in self.messages if m["role"] != "system"]
        self.messages.insert(0, {"role": "system", "content": content})

        save_personality_choice(new_personality)

        await self._show_system_message(f"Personality changed: **{new_personality}**")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if not event.value.strip() or self.is_generating:
            return

        user_input = event.value.strip()
        event.input.value = ""

        if user_input.startswith("/"):
            handled = await self._handle_command(user_input)
            if handled:
                return

        self.messages.append({"role": "user", "content": user_input})
        chat = self.query_one("#chat", ChatContainer)
        await chat.mount(Message(user_input, "user"))
        chat.scroll_end(animate=False)

        # Let UI refresh before blocking on generation
        await asyncio.sleep(0)

        await self._generate_response()

    async def _generate_response(self) -> None:
        """Generate assistant response (streaming or not)."""
        self.is_generating = True
        self._generation_cancelled = False
        chat = self.query_one("#chat", ChatContainer)
        status = self.query_one("#status", Static)

        assistant_msg = Message("...", "assistant")
        await chat.mount(assistant_msg)
        chat.scroll_end(animate=False)
        await asyncio.sleep(0)  # Let UI refresh

        response_text = ""
        start_time = time.time()
        tokens_generated = 0
        cancelled = False

        try:
            if self.streaming:
                status.update(self._status_text("generating..."))
                options = {"num_ctx": self.num_ctx, **self.model_options}
                stream = await asyncio.to_thread(
                    lambda: ollama.chat(
                        model=self.model,
                        messages=self.messages,
                        stream=True,
                        options=options,
                    )
                )

                for chunk in stream:
                    if self._generation_cancelled:
                        cancelled = True
                        break

                    if "message" in chunk and "content" in chunk["message"]:
                        response_text += chunk["message"]["content"]
                        tokens_generated += 1
                        await assistant_msg.update(f"● {response_text}")
                        chat.scroll_end(animate=False)

                    elapsed = time.time() - start_time
                    tps = tokens_generated / elapsed if elapsed > 0 else 0
                    status.update(self._status_text(f"generating... {elapsed:.1f}s ({tokens_generated}tok, {tps:.1f}t/s)"))
            else:
                # Non-streaming: show thinking animation
                thinking_task = asyncio.create_task(self._animate_thinking(assistant_msg, status, start_time))
                try:
                    options = {"num_ctx": self.num_ctx, **self.model_options}
                    result = await asyncio.to_thread(
                        lambda: ollama.chat(
                            model=self.model,
                            messages=self.messages,
                            stream=False,
                            options=options,
                        )
                    )
                    if not self._generation_cancelled:
                        response_text = result["message"]["content"]
                        tokens_generated = result.get("eval_count", len(response_text) // 4)
                    else:
                        cancelled = True
                finally:
                    thinking_task.cancel()
                    try:
                        await thinking_task
                    except asyncio.CancelledError:
                        pass

                if not cancelled:
                    await assistant_msg.update(f"● {response_text}")
                    chat.scroll_end(animate=False)

            if cancelled:
                await assistant_msg.update("● *[cancelled]*")
                if response_text:
                    # Partial response, still save it
                    self.messages.append({"role": "assistant", "content": response_text})
            else:
                self.messages.append({"role": "assistant", "content": response_text})
                self.last_gen_time = time.time() - start_time
                self.last_tokens = tokens_generated
                self.total_tokens += tokens_generated

        except Exception as e:
            response_text = f"**Error:** {e}"
            await assistant_msg.update(f"● {response_text}")

        self.is_generating = False
        status.update(self._status_text())

    async def _animate_thinking(self, msg: Message, status: Static, start_time: float) -> None:
        """Animate thinking indicator."""
        frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        i = 0
        while True:
            elapsed = time.time() - start_time
            await msg.update(f"● {frames[i]} thinking...")
            status.update(self._status_text(f"thinking... {elapsed:.1f}s"))
            i = (i + 1) % len(frames)
            await asyncio.sleep(0.1)

    def action_clear_input(self) -> None:
        """Clear the input field."""
        self.query_one("#chat-input", Input).value = ""

    def action_focus_input(self) -> None:
        """Keep focus on input and accept suggestion if any."""
        input_widget = self.query_one("#chat-input", Input)
        if input_widget._suggestion:
            input_widget.value = input_widget._suggestion
            input_widget.cursor_position = len(input_widget.value)
        input_widget.focus()

    def action_cancel_or_quit(self) -> None:
        """Cancel generation if running, otherwise quit."""
        if self.is_generating:
            self._generation_cancelled = True
            self.notify("Cancelled", timeout=2)
        else:
            self.exit()

    def action_toggle_streaming(self) -> None:
        """Toggle streaming mode."""
        self.streaming = not self.streaming
        save_streaming(self.streaming)
        mode = "ON" if self.streaming else "OFF"
        self.notify(f"Streaming: {mode}", timeout=2)

    def action_clear(self) -> None:
        """Clear chat history."""
        chat = self.query_one("#chat", ChatContainer)
        chat.remove_children()
        self.messages = []
        self.total_tokens = 0
        self.last_gen_time = 0.0
        self.last_tokens = 0
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})
        self.query_one("#status", Static).update(self._status_text())


def main():
    # First run: launch setup wizard
    if not CONFIG_FILE.exists():
        run_setup()

    config = load_config()

    parser = argparse.ArgumentParser(description="Simple Ollama TUI chat")
    parser.add_argument(
        "-C", "--config",
        action="store_true",
        help="Run interactive configuration setup"
    )
    parser.add_argument(
        "-m", "--model",
        default=None,
        help=f"Ollama model to use (default from config: {config['model']})"
    )
    parser.add_argument(
        "-s", "--system",
        help="Path to system prompt file"
    )
    parser.add_argument(
        "--system-prompt",
        help="System prompt string (overrides -s)"
    )
    parser.add_argument(
        "-c", "--num-ctx",
        type=int,
        default=None,
        help=f"Context window size (default from config: {config['num_ctx']})"
    )

    args = parser.parse_args()

    if args.config:
        run_setup()
        return

    model = args.model if args.model else config["model"]
    num_ctx = args.num_ctx if args.num_ctx else config["num_ctx"]

    if config["host"] != get_default_host() or "OLLAMA_HOST" not in os.environ:
        os.environ["OLLAMA_HOST"] = config["host"]

    if args.system_prompt:
        system_prompt = args.system_prompt
        personality_name = None
    else:
        system_prompt, personality_name = load_system_prompt(
            args.system, config["personality"], config["append_local_prompt"]
        )

    app = OllamaChat(
        model=model,
        system_prompt=system_prompt,
        num_ctx=num_ctx,
        personality_name=personality_name,
        append_local_prompt=config["append_local_prompt"],
        streaming=config["streaming"],
        model_options=config["model_options"],
    )
    app.run()


if __name__ == "__main__":
    main()
