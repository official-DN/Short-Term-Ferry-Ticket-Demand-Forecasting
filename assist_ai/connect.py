
"""
Google Gemini API connection for the Ferry AI Assistant.

Uses the Google Gen AI Python SDK (google-genai).

Local:
    Configure GEMINI_API_KEY and GEMINI_MODEL in assist_ai/.env

Streamlit Cloud:
    Configure GEMINI_API_KEY and GEMINI_MODEL in App Settings > Secrets
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types



ENV_PATH = Path(__file__).resolve().parent / ".env"

# Local .env values are loaded without overriding existing
# environment variables.
load_dotenv(dotenv_path=ENV_PATH, override=False)


# CONFIGURATION READER

def get_config(key: str, default: str | None = None) -> str | None:
    """
    Read configuration in this order:
    1. Environment variables
    2. Streamlit Cloud Secrets
    3. Default value
    """

    value = os.getenv(key)

    if value and value.strip():
        return value.strip()

    try:
        value = st.secrets.get(key)

        if value is not None and str(value).strip():
            return str(value).strip()

    except Exception:
        # Streamlit Secrets may not exist during local execution.
        pass

    return default


# CUSTOM EXCEPTION

class GeminiConnectionError(RuntimeError):
    """Raised when Gemini cannot complete a request."""


# GEMINI CLIENT

class GeminiClient:
    """Reusable wrapper around the Gemini Developer API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_output_tokens: int = 2048,
        temperature: float = 0.4,
    ) -> None:

        # Read API key from environment or Streamlit Secrets.
        self.api_key = (
            api_key
            or get_config("GEMINI_API_KEY")
        )

        # Read model from environment or Streamlit Secrets.
        self.model = (
            model
            or get_config("GEMINI_MODEL")
            or "gemini-3.8-flash"
        ).strip()

        self.max_output_tokens = max_output_tokens
        self.temperature = temperature

        # Validate API key.
        if not self.api_key:
            raise GeminiConnectionError(
                "GEMINI_API_KEY is missing. "
                "For local use, configure assist_ai/.env. "
                "For Streamlit Cloud, configure App Settings > Secrets."
            )

        # Initialize Gemini client.
        try:
            self.client = genai.Client(
                api_key=self.api_key
            )

        except Exception as exc:
            raise GeminiConnectionError(
                f"Could not initialize Gemini client: {exc}"
            ) from exc

        print(f"Gemini model selected: {self.model}")

   

    def generate(
        self,
        prompt: str,
        system_instruction: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """
        Generate an AI response using:
        - User prompt
        - System instructions
        - Recent conversation history
        """

        if not prompt or not prompt.strip():
            raise GeminiConnectionError(
                "Prompt cannot be empty."
            )

        history = history or []

        conversation_parts: list[str] = []

        # Include the latest 12 conversation messages.
        for item in history[-12:]:

            role = item.get("role", "")
            content = item.get("content", "").strip()

            if role in {"user", "assistant"} and content:

                label = (
                    "User"
                    if role == "user"
                    else "Assistant"
                )

                conversation_parts.append(
                    f"{label}: {content}"
                )

        # Add current user prompt.
        conversation_parts.append(
            f"User: {prompt.strip()}"
        )

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

                answer = (
                    response.text or ""
                ).strip()

                if not answer:
                    raise GeminiConnectionError(
                        "Gemini returned an empty response. "
                        "Please try another question."
                    )

                return answer

            # Gemini API errors.
            except errors.APIError as exc:

                code = getattr(exc, "code", None)

                # Configuration, authentication and model errors.
                if code in (400, 401, 403, 404):

                    raise GeminiConnectionError(
                        f"Gemini API error ({code}). "
                        f"Check API key, model availability "
                        f"and API access. Details: {exc}"
                    ) from exc

                # Rate limit or quota errors.
                if code == 429:

                    if attempt < 2:
                        time.sleep(2 ** attempt)
                        continue

                    raise GeminiConnectionError(
                        "Gemini rate limit or quota reached. "
                        f"Details: {exc}"
                    ) from exc

                # Temporary server errors.
                if code and code >= 500 and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue

                raise GeminiConnectionError(
                    f"Gemini API request failed. Details: {exc}"
                ) from exc

            except GeminiConnectionError:
                raise

            except Exception as exc:

                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue

                raise GeminiConnectionError(
                    "Could not connect to Gemini. "
                    f"Check internet connectivity. Details: {exc}"
                ) from exc

        raise GeminiConnectionError(
            "Gemini request failed after multiple attempts."
        )

    # CLOSE CLIENT

    def close(self) -> None:
        """Close the underlying Gemini SDK client."""

        if getattr(self, "client", None) is not None:
            self.client.close()


#  CONVENIENCE FUNCTION

def ask_gemini(
    prompt: str,
    system_instruction: str,
    history: list[dict[str, str]] | None = None,
) -> str:
    """
    Simple integration function for the Streamlit assistant.
    """

    client = GeminiClient()

    try:

        return client.generate(
            prompt=prompt,
            system_instruction=system_instruction,
            history=history,
        )

    finally:

        client.close()
