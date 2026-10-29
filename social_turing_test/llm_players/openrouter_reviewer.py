"""
OpenRouter Suspicion Detector -- OpenRouter API backend.
Inherits all review logic from BaseReviewer.

Model access and compatibility depend on the provider and account.
"""

import json
import os
from typing import Optional

from social_turing_test.llm_players.reviewer_base import BaseReviewer, ReviewResult, REVIEW_PROMPT_FIRST, REVIEW_PROMPT_RETRY  # noqa: F401
from social_turing_test.llm_players.openrouter_utils import request_completion

OPENROUTER_API_KEY_KEYWORD = "OPENROUTER_API_KEY"


def _get_openrouter_api_key() -> Optional[str]:
    return os.environ.get(OPENROUTER_API_KEY_KEYWORD) or None


class OpenRouterReviewer(BaseReviewer):
    """AI Suspicion Detector using OpenRouter (supports GPT-4o, Claude, etc.)."""

    DEFAULT_MODEL = "openai/gpt-4o-mini"

    def __init__(self, model_name: str = None, threshold: int = None, logger=None,
                 experimental: bool = False, api_timeout_sec: float = None):
        super().__init__(model_name or self.DEFAULT_MODEL, threshold, logger,
                         experimental=experimental)
        self.api_key = _get_openrouter_api_key()
        self.api_timeout_sec = 25.0 if api_timeout_sec is None else api_timeout_sec
        self._last_usage = None
        if not self.api_key:
            raise ValueError(
                f"OpenRouter API key not found. "
                f"Set {OPENROUTER_API_KEY_KEYWORD} env var."
            )

    def pop_usage(self):
        usage = self._last_usage
        self._last_usage = None
        return usage

    def _call_api(self, prompt: str) -> str:
        self._last_usage = None

        def record_usage(usage):
            self._last_usage = usage

        try:
            return request_completion(
                self.model_name, self.api_key,
                [{"role": "user", "content": prompt}],
                {"max_tokens": 600, "temperature": 0.3},
                component="reviewer", log=self._log, record_usage=record_usage,
                timeout_s=self.api_timeout_sec,
            )
        finally:
            # Retain failed-call usage too; the ranker may return without
            # reaching its successful-call cost writer.
            if self._last_usage:
                self._log("openrouter_usage", json.dumps(self._last_usage, allow_nan=False))
