#!/usr/bin/env python3
"""Simple Ollama TUI chat with custom system prompt support."""

import argparse
import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import openai

# Add script directory to path for imports when called from elsewhere
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging to temp file
_log_file = Path(tempfile.gettempdir()) / f"ollama-chat-{datetime.now():%Y%m%d-%H%M%S}.log"
logging.basicConfig(
    filename=str(_log_file),
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
)
_log = logging.getLogger(__name__)

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import ScrollableContainer, Vertical
from textual.events import Click
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
        "/prompt",
        "/sys", "/system",
        "/model", "/m",
        "/personality", "/p",
        "/project",
        "/config",
        "/impersonate", "/imp",
        "/stats", "/st",
        "/compact",
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
    CONFIG_DIR,
    CONFIG_FILE,
    load_config,
    load_system_prompt,
    load_personality,
    list_personalities,
    list_configs,
    load_project_prompt,
    update_config,
    switch_config_to_default,
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
    SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

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
        config_name: str = "",
        host: str | None = None,
        verify_ssl: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.system_prompt = system_prompt
        self.num_ctx = num_ctx
        self.personality_name = personality_name
        self.append_local_prompt = append_local_prompt
        self.model_options = model_options or {}
        self.config_name = config_name
        self.host = host or "http://localhost:11434"
        self.verify_ssl = verify_ssl
        self.ollama_client = ollama.Client(host=host, verify=verify_ssl)
        if verify_ssl:
            self.openai_client = openai.OpenAI(base_url=f"{self.host.rstrip('/')}/v1", api_key="not-needed")
        else:
            import httpx
            self.openai_client = openai.OpenAI(
                base_url=f"{self.host.rstrip('/')}/v1", api_key="not-needed",
                http_client=httpx.Client(verify=False),
            )
        self.api_mode = "ollama"  # "ollama" or "openai"
        self.messages: list[dict] = []
        self.is_generating = False
        self.streaming = streaming
        self._generation_cancelled = False
        self.total_tokens = 0
        self.last_gen_time = 0.0
        self.last_tokens = 0
        self.last_ttft = 0.0
        self._context_warning_shown = False
        self.sys_instructions = self._load_system_instructions()
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    @staticmethod
    def _load_system_instructions() -> dict:
        """Load system_instructions.json. Fatal error if missing or invalid."""
        path = Path(__file__).parent / "system_instructions.json"
        try:
            with open(path) as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: {path} not found. Please restore the file.")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: {path} is not valid JSON: {e}")
            sys.exit(1)

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

    def _estimate_context_tokens(self) -> int:
        """Rough token estimate for all messages (~4 chars/token)."""
        return sum(len(m["content"]) // 4 for m in self.messages)

    def _reset_stats(self) -> None:
        """Reset generation stats and context warning."""
        self.total_tokens = 0
        self.last_gen_time = 0.0
        self.last_tokens = 0
        self.last_ttft = 0.0
        self._context_warning_shown = False

    @asynccontextmanager
    async def _generating_lock(self):
        """Context manager: set is_generating, disable input, restore on exit."""
        self.is_generating = True
        input_widget = self.query_one("#chat-input", Input)
        input_widget.disabled = True
        try:
            yield
        finally:
            input_widget.disabled = False
            self.is_generating = False
            self.query_one("#status", Static).update(self._status_text())
            input_widget.focus()

    def _context_pct(self) -> float:
        """Estimated context usage as percentage."""
        if self.num_ctx <= 0:
            return 0.0
        return self._estimate_context_tokens() / self.num_ctx * 100

    def _context_info(self) -> str:
        msg_count = len([m for m in self.messages if m["role"] != "system"])
        estimated = self._estimate_context_tokens()
        pct = self._context_pct()
        return f"Messages: {msg_count} | Tokens used: ~{estimated} ({pct:.0f}%) | Context size: {self.num_ctx}"

    def _status_text(self, extra: str = "") -> str:
        if self.api_mode == "openai":
            base = f"{self.model} (openai)"
        else:
            base = f"{self.model} | ctx:{self.num_ctx}"
        if self.last_gen_time > 0:
            tps = self.last_tokens / self.last_gen_time if self.last_gen_time else 0
            ttft_str = f", ttft:{self.last_ttft:.2f}s" if self.last_ttft > 0 else ""
            base += f" | last: {self.last_gen_time:.1f}s ({self.last_tokens}tok, {tps:.1f}t/s{ttft_str})"
        if self.total_tokens > 0:
            base += f" | total: {self.total_tokens}tok"
        # Context warnings
        pct = self._context_pct()
        if pct > 100:
            base += " | ⚠ Context length probably exceeded"
        elif pct > 85:
            base += f" | ctx: {100 - pct:.0f}% remaining"
        if extra:
            base += f" | {extra}"
        return base

    async def on_mount(self) -> None:
        _log.info(f"App mounted, model={self.model}, config={self.config_name}")
        self.query_one("#chat-input", Input).focus()
        await self._show_greeting()

    def on_click(self, event: Click) -> None:
        """Keep focus on input when clicking anywhere."""
        self.query_one("#chat-input", Input).focus()

    async def _show_greeting(self) -> None:
        """Show greeting with ASCII art after validating connection."""
        chat = self.query_one("#chat", ChatContainer)
        config_line = f"config: {self.config_name} · " if self.config_name else ""

        # Try Ollama first, then OpenAI fallback
        mode_label = ""
        try:
            await asyncio.to_thread(lambda: self.ollama_client.list())
            self.api_mode = "ollama"
            _log.info("Connected in Ollama mode")
            mode_label = "Connected"
        except Exception as e:
            _log.info(f"Ollama list failed ({e}), trying OpenAI fallback")
            try:
                await asyncio.to_thread(lambda: self.openai_client.models.list())
                self.api_mode = "openai"
                _log.info("Connected in OpenAI-compatible mode")
                mode_label = "Connected (OpenAI mode)"
            except Exception as e2:
                _log.warning(f"OpenAI fallback failed: {e2}")
                await self._show_system_message("Warning: Cannot connect to server")
                return

        logo = f"""\
 ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
  ___   ___| |__   __ _| |_
 / _ \\ / __| '_ \\ / _` | __|
| (_) | (__| | | | (_| | |_
 \\___/ \\___|_| |_|\\__,_|\\__|

{config_line}{mode_label} · /help for commands"""
        msg = Static(logo, classes="greeting")
        await chat.mount(msg)
        chat.scroll_end(animate=False)

    async def _show_system_message(self, text: str) -> None:
        """Show a system info message in the chat."""
        chat = self.query_one("#chat", ChatContainer)
        msg = Message(text, "system-info")
        await chat.mount(msg)
        chat.scroll_end(animate=False)

    async def _handle_command(self, raw_cmd: str) -> bool:
        """Handle slash commands. Returns True if command was handled."""
        _log.debug(f"Command: {raw_cmd}")
        raw_cmd = raw_cmd.strip()
        parts = raw_cmd.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in ("/help", "/h", "/?"):
            help_text = """**Commands:**
- `/retry` or `/r` - Regenerate last response
- `/copy` - Copy last response to clipboard
- `/clear` or `/c` - Clear chat history
- `/context` or `/ctx` - Show current context info
- `/prompt` - Show system prompt
- `/sys <msg>` or `/system <msg>` - Inject a system message
- `/model` or `/m` - Show current model
- `/personality` or `/p` - List/change personality
- `/config` - List/switch config profiles
- `/impersonate` or `/imp` - Generate user response (for RP)
- `/project` - Toggle local prompt append (agent.md)
- `/stats` or `/st` - Show generation statistics
- `/compact` - Summarize conversation to free context
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
            await self._show_system_message(self._context_info())
            return True

        elif cmd == "/prompt":
            if self.system_prompt:
                await self._show_system_message(f"**System prompt:**\n{self.system_prompt}")
            else:
                await self._show_system_message("No system prompt set.")
            return True

        elif cmd in ("/sys", "/system"):
            if arg:
                self.messages.append({"role": "system", "content": arg})
                await self._show_system_message(f"*[injected]* {arg}")
            return True

        elif cmd in ("/model", "/m"):
            await self._show_system_message(f"Model: `{self.model}`")
            return True

        elif cmd in ("/personality", "/p"):
            await self._handle_personality_command(arg)
            return True

        elif cmd == "/project":
            await self._handle_project_toggle()
            return True

        elif cmd in ("/config", "/cfg"):
            await self._handle_config_command(arg)
            return True

        elif cmd in ("/impersonate", "/imp"):
            await self._handle_impersonate()
            return True

        elif cmd in ("/stats", "/st"):
            await self._handle_stats()
            return True

        elif cmd == "/compact":
            await self._handle_compact()
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
        for msg in reversed(self.messages):
            if msg["role"] == "assistant":
                self.copy_to_clipboard(msg["content"])
                await self._show_system_message("Copied to clipboard")
                return
        await self._show_system_message("Nothing to copy")

    async def _handle_project_toggle(self) -> None:
        """Toggle append_local_prompt and reload system prompt."""
        self.append_local_prompt = not self.append_local_prompt
        update_config(append_local_prompt=self.append_local_prompt)

        if self.personality_name:
            new_prompt, _ = load_system_prompt(
                None, self.personality_name, self.append_local_prompt
            )
            self.system_prompt = new_prompt

            self.messages = [m for m in self.messages if m["role"] != "system"]
            if new_prompt:
                self.messages.insert(0, {"role": "system", "content": new_prompt})

        project_content = load_project_prompt()
        status = "ON" if self.append_local_prompt else "OFF"
        if project_content and self.append_local_prompt:
            await self._show_system_message(f"Local prompt: **{status}** (appended to system prompt)")
        elif project_content:
            await self._show_system_message(f"Local prompt: **{status}** (ignored)")
        else:
            await self._show_system_message(f"Local prompt: **{status}** (no file found)")

    async def _handle_config_command(self, arg: str) -> None:
        """Handle /config command for listing and switching configs."""
        configs = list_configs()

        # Add current config.conf as option
        current_name = self.config_name or "(default)"
        all_configs = [current_name] + [c for c in configs if c != self.config_name]

        if not arg:
            # List configs
            lines = ["**Available configs:**"]
            for i, c in enumerate(all_configs, 1):
                marker = " ← current" if (i == 1) else ""
                lines.append(f"{i}. `{c}`{marker}")
            lines.append("\n*Type `/config <number>` to switch (restarts app)*")
            await self._show_system_message("\n".join(lines))
            return

        choice = arg.strip()
        try:
            idx = int(choice)
            if idx == 1:
                await self._show_system_message("Already using this config")
                return
            if 2 <= idx <= len(all_configs):
                selected_config = all_configs[idx - 1]
            else:
                await self._show_system_message(f"Invalid choice (1-{len(all_configs)})")
                return
        except ValueError:
            if choice in configs:
                selected_config = choice
            else:
                await self._show_system_message(f"Config '{choice}' not found")
                return

        # Switch config and restart
        success, msg = switch_config_to_default(selected_config, interactive=False)
        if not success:
            await self._show_system_message(f"Error: {msg}")
            return

        # Restart the app with new config
        await self._restart_app()

    async def _restart_app(self) -> None:
        """Show restart message and relaunch the app."""
        input_widget = self.query_one("#chat-input", Input)
        input_widget.value = ""
        input_widget.placeholder = "Restarting..."
        input_widget.disabled = True
        await asyncio.sleep(0.1)  # Let UI refresh
        self.exit()
        os.execv(sys.executable, [sys.executable] + sys.argv)

    async def _handle_impersonate(self) -> None:
        """Generate a response as if the user was speaking, put it in input."""
        _log.debug("Impersonate started")
        if self.is_generating:
            return

        if len([m for m in self.messages if m["role"] != "system"]) == 0:
            await self._show_system_message("Need conversation context first")
            return

        async with self._generating_lock():
            input_widget = self.query_one("#chat-input", Input)
            status = self.query_one("#status", Static)

            # Build messages with impersonate instruction
            impersonate_messages = self.messages.copy()
            impersonate_messages.append({
                "role": "system",
                "content": self.sys_instructions["impersonate"],
            })

            status.update(self._status_text("impersonating..."))
            input_widget.value = "Impersonating..."

            try:
                result = await asyncio.to_thread(
                    lambda: self._chat_call(impersonate_messages, stream=False)
                )
                response, _ = self._extract_result(result)
                response = response.strip()
                # Remove quotes if the model wrapped the response
                if response.startswith('"') and response.endswith('"'):
                    response = response[1:-1]
                # Replace newlines with spaces (Input doesn't support multiline)
                response = " ".join(response.split())
                _log.debug(f"Impersonate result: {response[:100]}...")
                input_widget.value = response
                input_widget.cursor_position = len(response)
            except Exception as e:
                _log.exception("Impersonate error")
                input_widget.value = ""
                await self._show_system_message(f"Error: {e}")

    async def _handle_stats(self) -> None:
        """Show generation statistics."""
        mode = "streaming" if self.streaming else "non-streaming"
        lines = [f"**Stats** — `{self.model}` ({mode})"]

        if self.last_gen_time > 0:
            tps = self.last_tokens / self.last_gen_time if self.last_gen_time else 0
            lines.append(f"\n**Last generation:**")
            lines.append(f"- Duration: {self.last_gen_time:.2f}s")
            if self.last_ttft > 0:
                lines.append(f"- TTFT: {self.last_ttft:.2f}s")
            lines.append(f"- Tokens: {self.last_tokens}")
            lines.append(f"- Speed: {tps:.1f} t/s")
        else:
            lines.append("\nNo generation yet.")

        lines.append(f"\n{self._context_info()}")

        await self._show_system_message("\n".join(lines))

    async def _handle_compact(self) -> None:
        """Summarize conversation to free up context."""
        if self.is_generating:
            return

        conv_messages = [m for m in self.messages if m["role"] != "system"]
        if len(conv_messages) < 2:
            await self._show_system_message("Not enough conversation to compact")
            return

        async with self._generating_lock():
            chat = self.query_one("#chat", ChatContainer)
            status = self.query_one("#status", Static)

            # Show spinner message in chat
            spinner_msg = Message("...", "system-info")
            await chat.mount(spinner_msg)
            chat.scroll_end(animate=False)

            start_time = time.time()
            summary = ""
            chunks = 0

            try:
                compact_messages = self.messages.copy()
                compact_messages.append({
                    "role": "system",
                    "content": self.sys_instructions["compact"],
                })

                stream, first_chunk = await self._start_stream(
                    compact_messages, spinner_msg, status, start_time
                )

                # Phase 2: buffer response, show progress
                if first_chunk:
                    content = self._extract_chunk(first_chunk)
                    if content:
                        summary += content
                        chunks += 1

                    for chunk in stream:
                        content = self._extract_chunk(chunk)
                        if content:
                            summary += content
                            chunks += 1
                        elapsed = time.time() - start_time
                        frame = self.SPINNER_FRAMES[int(elapsed * 10) % len(self.SPINNER_FRAMES)]
                        await spinner_msg.update(f"{frame} compacting conversation... {elapsed:.1f}s ({chunks} chunks)")
                        chat.scroll_end(animate=False)
                        status.update(self._status_text(f"compacting... {elapsed:.1f}s ({chunks} chunks)"))

                summary = summary.strip()
                elapsed = time.time() - start_time

                # Rebuild messages: system prompt + summary as context
                self.messages = []
                if self.system_prompt:
                    self.messages.append({"role": "system", "content": self.system_prompt})
                prefix = self.sys_instructions["compact_prefix"]
                self.messages.append({"role": "system", "content": f"{prefix}\n\n{summary}"})

                self._reset_stats()

                # Refresh chat display
                chat.remove_children()
                await self._show_system_message(f"**Conversation compacted** *({elapsed:.1f}s)*\n\n{summary}")
                _log.info(f"Compacted conversation: {len(conv_messages)} messages -> summary ({len(summary)} chars)")

            except Exception as e:
                _log.exception("Compact error")
                await spinner_msg.remove()
                await self._show_system_message(f"Error compacting: {e}")

    async def _handle_personality_command(self, arg: str) -> None:
        """Handle /personality command for listing and switching personalities."""
        personalities = list_personalities()

        if not arg:
            lines = ["**Available personalities:**"]
            for i, p in enumerate(personalities, 1):
                marker = " ← current" if p == self.personality_name else ""
                lines.append(f"{i}. `{p}`{marker}")
            lines.append("\n*Type `/p <number>` to switch*")
            await self._show_system_message("\n".join(lines))
            return

        choice = arg.strip()
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

        update_config(personality=new_personality)

        # If conversation has started, restart app to apply cleanly
        has_conversation = any(m["role"] in ("user", "assistant") for m in self.messages)
        if has_conversation:
            await self._restart_app()

        # No conversation yet, just swap the prompt
        self.personality_name = new_personality
        self.system_prompt = content
        self.messages = [{"role": "system", "content": content}]

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

    def _chat_call(self, messages: list[dict], stream: bool):
        """Make API call, return stream iterator or result."""
        if self.api_mode == "openai":
            return self.openai_client.chat.completions.create(
                model=self.model, messages=messages, stream=stream,
            )
        options = {"num_ctx": self.num_ctx, **self.model_options}
        return self.ollama_client.chat(
            model=self.model, messages=messages, stream=stream, options=options,
        )

    def _extract_chunk(self, chunk) -> str:
        """Extract text content from a streaming chunk."""
        if self.api_mode == "openai":
            return chunk.choices[0].delta.content or ""
        return chunk.get("message", {}).get("content", "")

    def _extract_result(self, result) -> tuple[str, int]:
        """Extract (content, token_count) from a non-streaming result."""
        if self.api_mode == "openai":
            content = result.choices[0].message.content
            tokens = getattr(result.usage, "completion_tokens", None) or len(content) // 4
            return content, tokens
        content = result["message"]["content"]
        tokens = result.get("eval_count", len(content) // 4)
        return content, tokens

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
            stream, first_chunk = await self._start_stream(
                self.messages, assistant_msg, status, start_time
            )
            self.last_ttft = time.time() - start_time if not self._generation_cancelled else 0.0

            if first_chunk and not self._generation_cancelled:
                content = self._extract_chunk(first_chunk)
                if content:
                    response_text += content
                    tokens_generated += 1

                if self.streaming:
                    # Show text live as it streams
                    await assistant_msg.update(f"● {response_text}")
                    chat.scroll_end(animate=False)

                    for chunk in stream:
                        if self._generation_cancelled:
                            cancelled = True
                            break
                        content = self._extract_chunk(chunk)
                        if content:
                            response_text += content
                            tokens_generated += 1
                            await assistant_msg.update(f"● {response_text}")
                            chat.scroll_end(animate=False)
                        elapsed = time.time() - start_time
                        tps = tokens_generated / elapsed if elapsed > 0 else 0
                        status.update(self._status_text(f"generating... {elapsed:.1f}s ({tokens_generated}tok, {tps:.1f}t/s)"))
                else:
                    # Buffer response, show "thinking" with chunk count
                    for chunk in stream:
                        if self._generation_cancelled:
                            cancelled = True
                            break
                        content = self._extract_chunk(chunk)
                        if content:
                            response_text += content
                            tokens_generated += 1
                        elapsed = time.time() - start_time
                        frame = self.SPINNER_FRAMES[int(elapsed * 10) % len(self.SPINNER_FRAMES)]
                        await assistant_msg.update(f"● {frame} thinking... {elapsed:.1f}s ({tokens_generated} chunks)")
                        chat.scroll_end(animate=False)
                        status.update(self._status_text(f"thinking... {elapsed:.1f}s ({tokens_generated} chunks)"))

                    if not cancelled:
                        think_time = time.time() - start_time
                        await assistant_msg.update(f"● *thought for {think_time:.1f}s*\n\n{response_text}")
                        chat.scroll_end(animate=False)

            if cancelled:
                await assistant_msg.update("● *[cancelled]*")
                if response_text:
                    self.messages.append({"role": "assistant", "content": response_text})
            else:
                self.messages.append({"role": "assistant", "content": response_text})
                self.last_gen_time = time.time() - start_time
                self.last_tokens = tokens_generated
                self.total_tokens += tokens_generated

        except Exception as e:
            _log.exception("Generation error")
            response_text = f"**Error:** {e}"
            await assistant_msg.update(f"● {response_text}")

        self.is_generating = False
        status.update(self._status_text())

        # One-shot context warning
        if not self._context_warning_shown and self._context_pct() > 80:
            self._context_warning_shown = True
            remaining = 100 - self._context_pct()
            await self._show_system_message(
                f"⚠ Approximately {remaining:.0f}% context length remaining, consider compacting (`/compact`)"
            )

    async def _start_stream(self, messages: list[dict], msg: Message, status: Static, start_time: float):
        """Start a streaming API call with a spinner while waiting for the first token.

        Returns (stream_iterator, first_chunk) — first_chunk may be None.
        """
        spinner = asyncio.create_task(
            self._animate_spinner(msg, status, start_time, "waiting for first token")
        )
        try:
            stream = await asyncio.to_thread(
                lambda: self._chat_call(messages, stream=True)
            )
            first_chunk = await asyncio.to_thread(lambda: next(iter(stream), None))
        finally:
            spinner.cancel()
            try:
                await spinner
            except asyncio.CancelledError:
                pass
        return stream, first_chunk

    async def _animate_spinner(self, msg: Message, status: Static, start_time: float, label: str) -> None:
        """Animate spinner indicator with given label."""
        i = 0
        while True:
            elapsed = time.time() - start_time
            await msg.update(f"● {self.SPINNER_FRAMES[i]} {label}...")
            self.query_one("#chat", ChatContainer).scroll_end(animate=False)
            status.update(self._status_text(f"{label}... {elapsed:.1f}s"))
            i = (i + 1) % len(self.SPINNER_FRAMES)
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
        update_config(streaming=self.streaming)
        mode = "ON" if self.streaming else "OFF"
        self.notify(f"Streaming: {mode}", timeout=2)

    def action_clear(self) -> None:
        """Clear chat history."""
        chat = self.query_one("#chat", ChatContainer)
        chat.remove_children()
        self.messages = []
        self._reset_stats()
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})
        self.query_one("#status", Static).update(self._status_text())


def main():
    _log.info(f"Starting ollama-chat, log file: {_log_file}")

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
    parser.add_argument(
        "--use-config",
        metavar="NAME",
        help="Use a named config file (without .conf extension)"
    )
    parser.add_argument(
        "--as-default",
        action="store_true",
        help="With --use-config: make it the new default config"
    )
    parser.add_argument(
        "--new",
        action="store_true",
        help="Create a new named config profile"
    )

    args = parser.parse_args()

    if args.new:
        run_setup(create_new=True)
        return

    if args.config:
        run_setup()
        return

    # Handle --use-config
    if args.use_config:
        config_file = CONFIG_DIR / f"{args.use_config}.conf"
        if not config_file.exists():
            print(f"Error: Config '{args.use_config}' not found")
            available = list_configs()
            if available:
                print(f"Available configs: {', '.join(available)}")
            sys.exit(1)

        if args.as_default:
            success, msg = switch_config_to_default(args.use_config)
            if not success:
                print(f"Error: {msg}")
                sys.exit(1)
            # Reload config from the new default
            config = load_config()
        else:
            # Just load the specified config for this session
            config = load_config(config_file)

    model = args.model if args.model else config["model"]
    num_ctx = args.num_ctx if args.num_ctx else config["num_ctx"]

    host = config["host"]

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
        config_name=config["config_name"],
        host=host,
        verify_ssl=config["verify_ssl"],
    )
    try:
        app.run()
    except Exception:
        _log.exception("Unhandled exception in app.run()")
        raise
    finally:
        _log.info("App exited")


if __name__ == "__main__":
    main()
