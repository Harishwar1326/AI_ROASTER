from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


APP_TITLE = "AI Roast ↔ Motivation Switcher"
MAX_HISTORY_TURNS = 8
DEFAULT_MODEL = "llama-3.1-8b-instant"
DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
CHAT_EXPORT_DIR = Path(__file__).resolve().parent.parent / "exports"


def ensure_export_dir() -> Path:
    CHAT_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    return CHAT_EXPORT_DIR


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_chat_json(payload: list[dict[str, Any]]) -> Path:
    export_dir = ensure_export_dir()
    path = export_dir / f"chat_{timestamp_slug()}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def save_chat_markdown(payload: list[dict[str, Any]]) -> Path:
    export_dir = ensure_export_dir()
    path = export_dir / f"chat_{timestamp_slug()}.md"
    path.write_text(chat_to_markdown(payload), encoding="utf-8")
    return path


def chat_to_markdown(payload: list[dict[str, Any]]) -> str:
    lines: list[str] = ["# AI Roast ↔ Motivation Switcher Chat Export", ""]
    for item in payload:
        lines.append(f"## {item.get('mode', 'Mode')}")
        lines.append(f"**Input:** {item.get('user', '')}")
        lines.append("")
        lines.append(f"**Response:** {item.get('assistant', '')}")
        lines.append("")
    return "\n".join(lines)


def sanitize_input_text(text: str) -> str:
    return " ".join(text.strip().split())


def build_history_messages(history: list[dict[str, str]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for item in history[-MAX_HISTORY_TURNS:]:
        user_text = item.get("user", "")
        assistant_text = item.get("assistant", "")
        if user_text:
            messages.append({"role": "user", "content": user_text})
        if assistant_text:
            messages.append({"role": "assistant", "content": assistant_text})
    return messages
