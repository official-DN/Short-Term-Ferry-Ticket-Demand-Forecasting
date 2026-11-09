
"""
Google Gemini API connection for the Ferry AI Assistant.

Uses the Google Gen AI Python SDK (google-genai).
Configure GEMINI_API_KEY and GEMINI_MODEL in assist_ai/.env.
"""

from __future__ import annotations

import os
import time
from pathlib import Path


import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

# Load local environment variables
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=False)


def get_config(key, default=None):
    """Read configuration from environment or Streamlit Secrets."""

    value = os.getenv(key)

    if value:
        return value

    try:
        value = st.secrets.get(key)
        if value:
            return str(value)
    except Exception:
        pass

    return default
class GeminiConnectionError(RuntimeError):
    """Raised when Gemini cannot complete a request."""


class GeminiClient:
    """Reusable wrapper around the Gemini Developer API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_output_tokens: int = 2048,
        temperature: float = 0.4,
    ) -> None:

        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

        # Updated default model
        self.model = (
            model
            or os.getenv("GEMINI_MODEL")
            or "gemini-3.8-flash"
        ).strip()

        self.max_output_tokens = max_output_tokens
        self.temperature = temperature

        print(f"Gemini model selected: {self.model}")

        if not self.api_key:
            raise GeminiConnectionError(
                "GEMINI_API_KEY is missing. Add it to assist_ai/.env"
            )

        self.client = genai.Client(api_key=self.api_key)

    def generate(
        self,
        prompt: str,
        system_instruction: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:

        history = history or []
        conversation_parts = []

        # Include recent conversation history
        for item in history[-12:]:
            role = item.get("role", "")
            content = item.get("content", "").strip()

            if role in {"user", "assistant"} and content:
                label = "User" if role == "user" else "Assistant"
                conversation_parts.append(f"{label}: {content}")

        conversation_parts.append(f"User: {prompt}")

        contents = "\n\n".join(conversation_parts)

        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=self.temperature,
                        max_output_tokens=self.max_output_tokens,
                    ),
                )

                answer = (response.text or "").strip()

                if not answer:
                    raise GeminiConnectionError(
                        "Gemini returned an empty response."
                    )

                return answer

            except errors.APIError as exc:

                code = getattr(exc, "code", None)

                if code in (400, 401, 403, 404):
                    raise GeminiConnectionError(
                        f"Gemini API error ({code}). "
                        f"Check model availability and API access. Details: {exc}"
                    ) from exc

                if code == 429:
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                        continue

                    raise GeminiConnectionError(
                        f"Gemini quota or rate limit reached: {exc}"
                    ) from exc

                if code and code >= 500 and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue

                raise GeminiConnectionError(
                    f"Gemini API request failed: {exc}"
                ) from exc

            except GeminiConnectionError:
                raise

            except Exception as exc:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue

                raise GeminiConnectionError(
                    f"Could not connect to Gemini: {exc}"
                ) from exc

        raise GeminiConnectionError("Gemini request failed after retries.")

    def close(self) -> None:
        """Close the Gemini SDK client."""
        self.client.close()


def ask_gemini(
    prompt: str,
    system_instruction: str,
    history: list[dict[str, str]] | None = None,
) -> str:

    client = GeminiClient()

    try:
        return client.generate(
            prompt=prompt,
            system_instruction=system_instruction,
            history=history,
        )

    finally:
        client.close()
