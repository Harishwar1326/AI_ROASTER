from __future__ import annotations

import json
from typing import Any, Generator

import requests

from .helpers import DEFAULT_BASE_URL, DEFAULT_MODEL, build_history_messages
from .prompt_builder import build_system_prompt, build_user_prompt


class APIClientError(RuntimeError):
    pass


class MissingAPIKeyError(APIClientError):
    pass


class RateLimitError(APIClientError):
    pass


class GroqAPIClient:
    def __init__(self, api_key: str | None, model: str | None = None, base_url: str | None = None) -> None:
        self.api_key = api_key.strip() if api_key else ""
        self.model = model or DEFAULT_MODEL
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise MissingAPIKeyError(
                "Missing GROQ_API_KEY. Add it to your .env file before generating a response."
            )
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _payload(
        self,
        user_text: str,
        mode: str,
        temperature: float,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        messages = [{"role": "system", "content": build_system_prompt(mode)}]
        if history:
            messages.extend(build_history_messages(history))
        messages.append({"role": "user", "content": build_user_prompt(user_text, mode)})
        return {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }

    def stream_response(
        self,
        user_text: str,
        mode: str,
        temperature: float,
        history: list[dict[str, str]] | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        url = f"{self.base_url}/chat/completions"
        payload = self._payload(user_text, mode, temperature, history)

        try:
            response = requests.post(url, headers=self._headers(), json=payload, stream=True, timeout=90)
        except requests.RequestException as exc:
            raise APIClientError(f"Network error while calling the API: {exc}") from exc

        if response.status_code == 401:
            raise APIClientError("Authentication failed. Check your GROQ_API_KEY value.")
        if response.status_code == 429:
            raise RateLimitError("Rate limit reached. Please wait a moment and try again.")
        if response.status_code >= 400:
            raise APIClientError(f"API error {response.status_code}: {response.text[:300]}")

        full_text = []
        usage = None

        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            if not raw_line.startswith("data: "):
                continue

            chunk = raw_line.removeprefix("data: ").strip()
            if chunk == "[DONE]":
                break

            try:
                data = json.loads(chunk)
            except json.JSONDecodeError:
                continue

            choice = (data.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            token = delta.get("content") or ""
            if token:
                full_text.append(token)
                yield {"type": "token", "text": token}

            if data.get("usage"):
                usage = data["usage"]

        final_text = "".join(full_text).strip()
        yield {"type": "done", "text": final_text, "usage": usage}

    def generate_response(
        self,
        user_text: str,
        mode: str,
        temperature: float,
        history: list[dict[str, str]] | None = None,
    ) -> tuple[str, dict[str, Any] | None]:
        collected = []
        usage = None
        for event in self.stream_response(user_text, mode, temperature, history):
            if event["type"] == "token":
                collected.append(event["text"])
            elif event["type"] == "done":
                usage = event.get("usage")
        return "".join(collected).strip(), usage
