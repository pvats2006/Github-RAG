"""
LLM Client abstraction for Code Repository RAG using the Groq Cloud API.
"""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from typing import Optional

def _load_env_files():
    """Load .env from multiple standard locations if python-dotenv is available."""
    global _ENV_FILES_LOADED
    if _ENV_FILES_LOADED:
        return

    try:
        from dotenv import load_dotenv
        # 1. Current working directory
        load_dotenv()
        # 2. Workspace root relative to this file
        project_root = Path(__file__).resolve().parent.parent.parent
        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=False)
        _ENV_FILES_LOADED = True
    except ImportError:
        _ENV_FILES_LOADED = True


_ENV_FILES_LOADED = False

_load_env_files()

logger = logging.getLogger(__name__)

DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"


class GroqClient:
    """
    Groq API client for code repository reasoning and answer generation.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.1,
    ):
        self._explicit_api_key = api_key
        self._explicit_model = model
        self.temperature = temperature
        self._client = None

    @property
    def api_key(self) -> Optional[str]:
        return self._explicit_api_key or os.environ.get("GROQ_API_KEY")

    @property
    def model(self) -> str:
        return self._explicit_model or os.environ.get("GROQ_MODEL") or DEFAULT_GROQ_MODEL

    def _get_client(self):
        if self._client is not None:
            return self._client

        api_key = self.api_key
        if not api_key:
            error_msg = (
                "ERROR: GROQ_API_KEY is not configured.\n\n"
                "Set GROQ_API_KEY in your environment and restart."
            )
            raise ValueError(error_msg)

        try:
            from groq import Groq
            self._client = Groq(api_key=api_key)
            return self._client
        except ImportError as exc:
            raise RuntimeError(
                "The 'groq' package is not installed. Install it with:\n"
                "    pip install groq"
            ) from exc

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Send prompt to Groq-hosted LLM and return the generated answer text.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        client = self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            response = client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            content = choice.message.content or ""
            return content.strip()
        except Exception as exc:
            logger.error("Groq generation failed: %s", exc)
            raise RuntimeError(f"Groq generation failed: {exc}") from exc


# Aliases for backward compatibility and clean abstraction
LLMClient = GroqClient
Qwen3Client = GroqClient


def main():
    print()
    print("=" * 40)
    print("GROQ LLM CLIENT TEST")
    print("=" * 40)

    try:
        client = GroqClient()
    except ValueError as exc:
        print(f"\n{exc}\n")
        sys.exit(1)

    prompt = input("\nEnter prompt: ").strip()
    if not prompt:
        print("Prompt empty.")
        return

    print("\nGenerating response with Groq...")
    try:
        answer = client.generate(prompt)
        print("\n" + "=" * 40)
        print("GROQ RESPONSE")
        print("=" * 40)
        print(answer)
    except Exception as exc:
        print(f"\nError: {exc}")


if __name__ == "__main__":
    main()