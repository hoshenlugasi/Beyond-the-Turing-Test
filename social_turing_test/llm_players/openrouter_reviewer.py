"""
OpenRouter Suspicion Detector -- OpenRouter API backend.
Inherits all review logic from BaseReviewer.

Supports any model available on OpenRouter (e.g. openai/gpt-4o-mini,
anthropic/claude-3-haiku, google/gemini-flash-1.5, etc.).
"""

import json
import os
import time
from typing import Optional

import requests

from social_turing_test.llm_players.reviewer_base import BaseReviewer, ReviewResult, REVIEW_PROMPT_FIRST, REVIEW_PROMPT_RETRY  # noqa: F401
from social_turing_test.llm_players.openrouter_utils import model_supports_temperature, is_unsupported_param_error

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY_KEYWORD = "OPENROUTER_API_KEY"
SLEEPING_TIME_FOR_API_ERROR = 3


def _get_openrouter_api_key() -> Optional[str]:
    return os.environ.get(OPENROUTER_API_KEY_KEYWORD) or None


class OpenRouterReviewer(BaseReviewer):
    """AI Suspicion Detector using OpenRouter (supports GPT-4o, Claude, etc.)."""

    DEFAULT_MODEL = "openai/gpt-4o-mini"

    def __init__(self, model_name: str = None, threshold: int = None, logger=None,
                 experimental: bool = False):
        super().__init__(model_name or self.DEFAULT_MODEL, threshold, logger,
                         experimental=experimental)
        self.api_key = _get_openrouter_api_key()
        if not self.api_key:
            raise ValueError(
                f"OpenRouter API key not found. "
                f"Set {OPENROUTER_API_KEY_KEYWORD} env var."
            )

    def _call_api(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 600,
        }
        if model_supports_temperature(self.model_name):
            payload["temperature"] = 0.3
        last_err = None
        for attempt in range(3):
            try:
                response = requests.post(
                    OPENROUTER_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=60,
                )
                # If the provider rejects `temperature` (or similar), strip it and retry once.
                if response.status_code == 400 and "temperature" in payload and \
                        is_unsupported_param_error(response.status_code, response.text):
                    self._log("openrouter_unsupported_param",
                              f"Stripping temperature for {self.model_name}")
                    payload.pop("temperature", None)
                    response = requests.post(
                        OPENROUTER_API_URL,
                        headers=headers,
                        json=payload,
                        timeout=60,
                    )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                last_err = e
                self._log("api_error", f"Attempt {attempt + 1}: {e}")
                if attempt < 2:
                    time.sleep(SLEEPING_TIME_FOR_API_ERROR)
        raise RuntimeError(
            f"OpenRouter reviewer API failed after 3 attempts: {last_err}"
        )
