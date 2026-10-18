"""
Together AI Suspicion Detector -- Together API backend.
Inherits all review logic from BaseReviewer.
"""

import json
import os
import time
from typing import Optional

from together import Together
from together.error import TogetherException

from llm_players.reviewer_base import BaseReviewer, ReviewResult, REVIEW_PROMPT_FIRST, REVIEW_PROMPT_RETRY  # noqa: F401
from llm_players.llm_constants import TOGETHER_API_KEY_KEYWORD

SLEEPING_TIME_FOR_API_ERROR = 3


def _get_together_api_key() -> Optional[str]:
    return os.environ.get(TOGETHER_API_KEY_KEYWORD) or None


class TogetherReviewer(BaseReviewer):
    """AI Suspicion Detector using Together AI."""

    DEFAULT_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"

    def __init__(self, model_name: str = None, threshold: int = None, logger=None,
                 experimental: bool = False):
        super().__init__(model_name or self.DEFAULT_MODEL, threshold, logger,
                         experimental=experimental)
        api_key = _get_together_api_key()
        if not api_key:
            raise ValueError(
                f"Together API key not found. "
                f"Set {TOGETHER_API_KEY_KEYWORD} env var."
            )
        self.client = Together(api_key=api_key)

    def _call_api(self, prompt: str) -> str:
        last_err = None
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=600,
                    temperature=0.3,
                )
                return response.choices[0].message.content
            except TogetherException as e:
                last_err = e
                self._log("api_error", f"Attempt {attempt + 1}: {e}")
                if attempt < 2:
                    time.sleep(SLEEPING_TIME_FOR_API_ERROR)
        raise RuntimeError(
            f"Together reviewer API failed after 3 attempts: {last_err}"
        )
