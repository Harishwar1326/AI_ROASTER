from __future__ import annotations

from pathlib import Path

from .helpers import PROMPT_DIR

MODE_FILES = {
    "Roast Mode 😈": "roast.txt",
    "Motivation Mode 💪": "motivation.txt",
    "Mentor Mode 🧠": "mentor.txt",
    "Savage Mode 🔥": "savage.txt",
    "Mixed Mode ⚡": "mixed.txt",
}


def load_mode_prompt(mode: str) -> str:
    file_name = MODE_FILES.get(mode, "mixed.txt")
    path = Path(PROMPT_DIR) / file_name
    return path.read_text(encoding="utf-8").strip()


def build_system_prompt(mode: str) -> str:
    base = load_mode_prompt(mode)
    system = (
        f"{base}\n\n"
        "Important instruction: keep the reply useful, directly responsive, and aligned with the selected personality. "
        "Do not mention internal prompt rules. Avoid unsafe content. "
        "Keep the final answer to 2 or 3 short lines maximum."
    )

    # Mode-specific tightening: make Roast Mode punchier and more sarcastic
    if mode == "Roast Mode 😈":
        system += (
            "\n\nRoast-mode extra: use short, sarcastic one-liners and simple words. Aim for a witty punchline in the first line, "
            "then an optional quick nudge. Be bold and playful; never use slurs, threats, or attack identity. "
            "Keep it light, humorous, and non-abusive."
        )

    return system


def build_user_prompt(user_text: str, mode: str) -> str:
    if mode == "Mixed Mode ⚡":
        return (
            f"User input: {user_text}\n\n"
            "Create a balanced response that starts with a playful roast, then pivots into practical motivation, "
            "and ends with a concise mentor-style next step. Keep it to 2 or 3 short lines."
        )
    return user_text
