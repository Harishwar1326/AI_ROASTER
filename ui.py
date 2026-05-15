from __future__ import annotations

import html
import json
import os
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from utils.api_client import APIClientError, GroqAPIClient, MissingAPIKeyError, RateLimitError
from utils.helpers import APP_TITLE, build_history_messages, chat_to_markdown, sanitize_input_text

MODE_OPTIONS = [
    "Roast Mode 😈",
    "Motivation Mode 💪",
    "Mentor Mode 🧠",
    "Savage Mode 🔥",
    "Mixed Mode ⚡",
]

MODE_DESCRIPTIONS = {
    "Roast Mode 😈": "Playful, sharp, comedic pressure.",
    "Motivation Mode 💪": "Encouraging, uplifting, action-focused.",
    "Mentor Mode 🧠": "Clear, structured, and educational.",
    "Savage Mode 🔥": "Bold, intense, and high-energy.",
    "Mixed Mode ⚡": "A blend of roast, motivation, and mentorship.",
}


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #0b1020;
            --panel: rgba(16, 23, 42, 0.88);
            --panel-border: rgba(148, 163, 184, 0.18);
            --accent: #7c3aed;
            --accent-2: #22c55e;
            --text: #e2e8f0;
            --muted: #94a3b8;
            --shadow: 0 20px 60px rgba(0, 0, 0, 0.35);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(124, 58, 237, 0.22), transparent 30%),
                radial-gradient(circle at top right, rgba(34, 197, 94, 0.14), transparent 26%),
                linear-gradient(180deg, #050816 0%, #0b1020 100%);
            color: var(--text);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1150px;
        }

        .hero {
            padding: 1.25rem 1.4rem;
            border: 1px solid var(--panel-border);
            border-radius: 22px;
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.92), rgba(2, 6, 23, 0.88));
            box-shadow: var(--shadow);
            margin-bottom: 1.2rem;
        }

        .hero h1 {
            margin: 0;
            font-size: 2.4rem;
            letter-spacing: -0.03em;
        }

        .hero p {
            margin: 0.6rem 0 0;
            color: var(--muted);
            line-height: 1.6;
        }

        .card {
            padding: 1rem 1.1rem;
            border-radius: 18px;
            border: 1px solid var(--panel-border);
            background: var(--panel);
            box-shadow: var(--shadow);
        }

        .mini-card {
            padding: 0.75rem 0.9rem;
            border-radius: 14px;
            border: 1px solid rgba(148, 163, 184, 0.12);
            background: rgba(15, 23, 42, 0.72);
        }

        .muted {
            color: var(--muted);
        }

        .result-box {
            padding: 1rem 1.05rem;
            border-radius: 18px;
            border: 1px solid rgba(124, 58, 237, 0.28);
            background: linear-gradient(180deg, rgba(17, 24, 39, 0.95), rgba(12, 18, 35, 0.98));
            min-height: 180px;
        }

        .badge {
            display: inline-block;
            padding: 0.35rem 0.65rem;
            margin-right: 0.5rem;
            border-radius: 999px;
            background: rgba(124, 58, 237, 0.14);
            border: 1px solid rgba(124, 58, 237, 0.22);
            color: #c4b5fd;
            font-size: 0.82rem;
        }

        .metric-label {
            font-size: 0.78rem;
            color: var(--muted);
        }

        .history-item {
            border-left: 2px solid rgba(124, 58, 237, 0.5);
            padding-left: 0.75rem;
            margin: 0.8rem 0;
        }

        .stTextArea textarea,
        .stSelectbox div[data-baseweb="select"] > div,
        .stSlider,
        .stButton button,
        .stDownloadButton button {
            border-radius: 14px !important;
        }

        .stButton button {
            border: 1px solid rgba(124, 58, 237, 0.4);
            background: linear-gradient(135deg, #7c3aed, #22c55e);
            color: white;
            font-weight: 700;
            box-shadow: 0 10px 30px rgba(124, 58, 237, 0.25);
        }

        .stButton button:hover {
            transform: translateY(-1px);
        }

        .footer-note {
            color: var(--muted);
            font-size: 0.88rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    defaults = {
        "chat_history": [],
        "last_prompt": "",
        "last_mode": MODE_OPTIONS[0],
        "last_temperature": 0.7,
        "last_response": "",
        "last_usage": None,
        "last_error": "",
        "last_input": "",
        "last_prompt_payload": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_header() -> None:
    st.markdown(
        f"""
        <div class="hero">
            <h1>{APP_TITLE}</h1>
            <p>
                Turn one thought into five personalities. This beginner-friendly GenAI app shows how prompt engineering,
                temperature, and OpenAI-compatible API calls create different AI voices from the same input.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple[str, float, bool, bool]:
    with st.sidebar:
        st.markdown("## Controls")
        api_key = os.getenv("GROQ_API_KEY", "")
        model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        st.caption("Load your API key from .env before generating responses.")

        mode = st.selectbox("Personality mode", MODE_OPTIONS, index=0)
        st.write(f"**{MODE_DESCRIPTIONS[mode]}**")

        temperature = st.slider(
            "Creativity temperature",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Lower values are more focused. Higher values are more creative and varied.",
        )

        use_memory = st.toggle("Use conversation memory", value=True)
        stream_output = st.toggle("Stream responses", value=True)

        st.markdown("---")
        st.markdown("### Model")
        st.code(model, language="text")
        st.caption("Groq is the recommended beginner-friendly default for fast OpenAI-compatible inference.")

    return mode, temperature, use_memory, stream_output


def render_input_panel() -> str:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Your input")
    user_input = st.text_area(
        "Enter any thought, confession, struggle, or sentence.",
        placeholder="I keep procrastinating...",
        height=160,
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)
    return user_input


def render_history_panel() -> None:
    with st.expander("Conversation memory", expanded=False):
        history = st.session_state.chat_history
        if not history:
            st.write("No saved conversation yet.")
            return
        for item in reversed(history[-6:]):
            mode = html.escape(item.get('mode', ''))
            user_text = html.escape(item.get('user', ''))
            assistant_text = html.escape(item.get('assistant', ''))
            st.markdown(
                f"""
                <div class="history-item">
                    <div><strong>Mode:</strong> {mode}</div>
                    <div><strong>Input:</strong> {user_text}</div>
                    <div><strong>Reply:</strong> {assistant_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def build_prompt_preview(user_input: str, mode: str, temperature: float, use_memory: bool) -> str:
    payload: dict[str, Any] = {
        "model": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        "mode": mode,
        "temperature": temperature,
        "input": user_input,
        "memory_enabled": use_memory,
    }
    if use_memory and st.session_state.chat_history:
        payload["recent_turns"] = build_history_messages(st.session_state.chat_history)
    return json.dumps(payload, indent=2, ensure_ascii=False)


def generate_response(
    api_key: str,
    mode: str,
    temperature: float,
    user_input: str,
    use_memory: bool,
    stream_output: bool,
) -> tuple[str, dict[str, Any] | None]:
    client = GroqAPIClient(api_key=api_key, model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"))
    history = st.session_state.chat_history if use_memory else []

    if stream_output:
        placeholder = st.empty()
        running_text = ""
        usage = None
        for event in client.stream_response(user_input, mode, temperature, history):
            if event["type"] == "token":
                running_text += event["text"]
                placeholder.markdown(
                    f'<div class="result-box"><p style="white-space: pre-wrap; margin: 0; line-height: 1.7;">{running_text}</p></div>',
                    unsafe_allow_html=True,
                )
            elif event["type"] == "done":
                usage = event.get("usage")
        return running_text.strip(), usage

    response_text, usage = client.generate_response(user_input, mode, temperature, history)
    return response_text, usage


def render_result(response_text: str) -> None:
    st.markdown("### Result")
    if response_text:
        escaped_text = html.escape(response_text)
        st.markdown(
            f'<div class="result-box"><p style="white-space: pre-wrap; margin: 0; line-height: 1.7;">{escaped_text}</p></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="result-box"><p class="muted" style="margin: 0;">Your response will appear here.</p></div>',
            unsafe_allow_html=True,
        )


def render_metrics() -> None:
    usage = st.session_state.last_usage or {}
    cols = st.columns(4)
    cols[0].metric("Mode", st.session_state.last_mode)
    cols[1].metric("Temperature", f"{st.session_state.last_temperature:.2f}")
    cols[2].metric("Prompt chars", len(st.session_state.last_input or ""))
    cols[3].metric("Tokens", usage.get("total_tokens", "n/a"))


def render_footer() -> None:
    st.markdown(
        "<p class='footer-note'>Built with prompt engineering, OpenAI-compatible API calls, and session-based memory. "
        "Safe by design, portfolio-ready by default.</p>",
        unsafe_allow_html=True,
    )


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title=APP_TITLE, page_icon="😈", layout="wide")
    apply_styles()
    init_state()

    render_header()
    mode, temperature, use_memory, stream_output = render_sidebar()

    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        user_input = render_input_panel()
        action_cols = st.columns([1, 1, 1])
        submit = action_cols[0].button("Generate response", use_container_width=True)
        regenerate = action_cols[1].button("Regenerate last", use_container_width=True)
        clear = action_cols[2].button("Clear chat", use_container_width=True)

        if clear:
            st.session_state.chat_history = []
            st.session_state.last_response = ""
            st.session_state.last_input = ""
            st.session_state.last_error = ""
            st.session_state.last_usage = None
            st.rerun()

        active_input = user_input
        if regenerate and st.session_state.last_input:
            active_input = st.session_state.last_input
            mode = st.session_state.last_mode
            temperature = st.session_state.last_temperature

        if submit or regenerate:
            active_input = sanitize_input_text(active_input)
            if not active_input:
                st.session_state.last_error = "Please enter some text before generating a response."
            else:
                st.session_state.last_error = ""
                st.session_state.last_input = active_input
                st.session_state.last_mode = mode
                st.session_state.last_temperature = temperature
                st.session_state.last_prompt_payload = build_prompt_preview(active_input, mode, temperature, use_memory)

                api_key = os.getenv("GROQ_API_KEY", "")
                try:
                    response_text, usage = generate_response(api_key, mode, temperature, active_input, use_memory, stream_output)
                    st.session_state.last_response = response_text
                    st.session_state.last_usage = usage
                    if response_text:
                        st.session_state.chat_history.append(
                            {
                                "mode": mode,
                                "user": active_input,
                                "assistant": response_text,
                            }
                        )
                except MissingAPIKeyError as exc:
                    st.session_state.last_error = str(exc)
                except RateLimitError as exc:
                    st.session_state.last_error = str(exc)
                except APIClientError as exc:
                    st.session_state.last_error = str(exc)
                except Exception as exc:  # pragma: no cover - safety net for unexpected failures
                    st.session_state.last_error = f"Unexpected error: {exc}"

        if st.session_state.last_error:
            st.error(st.session_state.last_error)

        if not stream_output:
            render_result(st.session_state.last_response)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### What this app teaches")
        st.markdown(
            """
            <div class="mini-card">
            <span class="badge">Prompt engineering</span>
            <span class="badge">Temperature control</span>
            <span class="badge">Streaming tokens</span>
            <span class="badge">Conversation memory</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write(
            "The system prompt changes the personality, the user message supplies the content, and the temperature controls how creative the model can be."
        )
        st.markdown("---")
        render_metrics()
        st.markdown("---")
        render_history_panel()
        st.markdown("---")

        if st.session_state.last_response:
            export_payload = st.session_state.chat_history
            json_export = json.dumps(export_payload, indent=2, ensure_ascii=False)
            md_export = chat_to_markdown(export_payload) if export_payload else ""
            st.download_button("Download chat JSON", data=json_export, file_name="chat_history.json", mime="application/json", use_container_width=True)
            if md_export:
                st.download_button("Download chat Markdown", data=md_export, file_name="chat_history.md", mime="text/markdown", use_container_width=True)
        else:
            st.caption("Generate a response to unlock downloads and prompt diagnostics.")

        with st.expander("Prompt logging", expanded=False):
            if st.session_state.last_prompt_payload:
                st.code(st.session_state.last_prompt_payload, language="json")
            else:
                st.write("Run a generation to inspect the API payload and learn how the request is structured.")

        with st.expander("Beginner-friendly notes", expanded=False):
            st.markdown(
                """
                - **System prompt:** the high-level instruction that defines the AI personality.
                - **Temperature:** a randomness knob. Lower means safer and more focused, higher means more creative.
                - **Inference:** the model producing the next token based on all prior context.
                - **Token generation:** the response is created one small chunk at a time, which makes streaming possible.
                - **LLM workflow:** user input -> prompt builder -> API call -> model output -> UI render.
                """
            )

        st.markdown("</div>", unsafe_allow_html=True)

    render_footer()


if __name__ == "__main__":
    main()
