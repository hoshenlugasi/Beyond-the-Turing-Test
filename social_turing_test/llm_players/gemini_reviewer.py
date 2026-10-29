"""
Gemini AI Suspicion Detector -- Google GenAI backend.
Supports both Vertex AI (ADC) and Gemini API key auth.
Inherits all review logic from BaseReviewer.
"""

import os
import concurrent.futures

from social_turing_test.llm_players.reviewer_base import BaseReviewer, ReviewResult, REVIEW_PROMPT_FIRST, REVIEW_PROMPT_RETRY  # noqa: F401 (ReviewResult re-exported)

# Module-level default; the actual per-call timeout is set on the instance via
# the constructor (api_timeout_sec) and matches reviewer_api_timeout_sec from
# the player's llm_config. The constant is also available for direct import.
REVIEWER_API_TIMEOUT_SEC = 10.0
# Hard fallback timeout when caller passes None / 0 / invalid. Prevents the
# reviewer from EVER calling the SDK without an external wall-clock guard -
# critical because the Gemini SDK's internal tenacity retry can hang for
# minutes on transient errors.
_DEFAULT_API_TIMEOUT_SEC = 30.0
# Hard floor applied when api_timeout_sec is None / 0 / invalid. Prevents
# accidental no-timeout calls that hang on Gemini SDK tenacity retry storms
# and prevent the agent from reaching the voting phase.
_DEFAULT_API_TIMEOUT_SEC = 30.0


def _get_gemini_api_key():
    """Return the operator's Gemini / Google environment key, else None."""
    try:
        from social_turing_test.llm_players.llm_wrapper import get_api_key
        return get_api_key("GEMINI_API_KEY") or get_api_key("GOOGLE_API_KEY")
    except Exception:
        # Fall back to env-only if the helper can't be imported for any reason.
        return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


class GeminiReviewer(BaseReviewer):
    """AI Suspicion Detector using Gemini via google-genai SDK."""

    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(self, model_name: str = None, threshold: int = None, logger=None,
                 project_id: str = None, location: str = None,
                 api_timeout_sec: float = None, experimental: bool = False):
        super().__init__(model_name or self.DEFAULT_MODEL, threshold, logger,
                         experimental=experimental)
        # Coerce api_timeout_sec to a strictly positive float. None / 0 /
        # invalid values fall back to _DEFAULT_API_TIMEOUT_SEC. This is a
        # HARD invariant: there is no path through _call_api that can run
        # the SDK without a wall-clock timeout, because the SDK's internal
        # tenacity retries can block the main thread for ~10 minutes on
        # transient 429s.
        try:
            v = float(api_timeout_sec) if api_timeout_sec is not None else None
        except (TypeError, ValueError):
            v = None
        self.api_timeout_sec = v if (v is not None and v > 0) else _DEFAULT_API_TIMEOUT_SEC

        from google import genai  # lazy import

        api_key = _get_gemini_api_key()
        model_label = model_name or self.DEFAULT_MODEL
        if api_key:
            print(f"[GeminiReviewer auth] Using API key for model {model_label}", flush=True)
            self.client = genai.Client(api_key=api_key)
        else:
            pid = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
            loc = location or os.environ.get("GOOGLE_CLOUD_LOCATION")
            if not pid or not loc:
                raise RuntimeError(
                    "Gemini reviewer authentication missing. Set GEMINI_API_KEY or GOOGLE_API_KEY, "
                    "or explicitly supply your own Vertex AI project and location "
                    "(GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION) with Application Default Credentials."
                )
            print(f"[GeminiReviewer auth] Using Vertex AI (project={pid}, location={loc}) for model {model_label}", flush=True)
            self.client = genai.Client(vertexai=True, project=pid, location=loc)

    def _call_api(self, prompt: str) -> str:
        from google.genai import types as genai_types  # lazy import

        # Wrap the blocking SDK call in a thread with a hard wall-clock timeout
        # so a slow/hung Gemini backend cannot stall the reviewer pipeline.
        # The SDK's internal tenacity retries on transient 429s can block the
        # main thread for minutes. We never
        # call _call() directly - there is always a timeout via future.result.
        # Gemini thinking-model strategy (mirrors the generator in
        # llm_wrapper.py):
        #   Gemini 3 (3.x): use thinking_level ("low"/"high"), not
        #     thinking_budget. The lowest level limits review latency.
        #   Gemini 2.5: Flash can disable thinking entirely (budget=0), which is
        #     critical here - with thinking ON the reviewer call spends 20+ seconds
        #     "thinking" and hits the wall-clock timeout on EVERY call, returning a
        #     useless default score of 5 and adding ~20s of latency per message.
        #     Pro requires thinking; use the minimum (128).
        model_lower = self.model_name.lower()
        is_gemini_3 = "gemini-3" in model_lower
        is_pro = "pro" in model_lower and "flash" not in model_lower
        if is_gemini_3:
            thinking_cfg = genai_types.ThinkingConfig(thinking_level="low")
        else:
            thinking_cfg = genai_types.ThinkingConfig(
                thinking_budget=128 if is_pro else 0)

        def _call():
            return self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    max_output_tokens=3072,
                    temperature=0.3,
                    response_mime_type="application/json",
                    thinking_config=thinking_cfg,
                ),
            )

        # IMPORTANT: do NOT use `with ThreadPoolExecutor() as ex:` here.
        # The context-manager `__exit__` calls `shutdown(wait=True)` which
        # blocks until the running task finishes - defeating the timeout
        # whenever the SDK's internal tenacity-retry chain spins up. Use a
        # manual executor and `shutdown(wait=False)` so a hung thread is
        # abandoned to die in the background; we return control immediately.
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(_call)
            try:
                response = future.result(timeout=self.api_timeout_sec)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(
                    f"Gemini reviewer call exceeded {self.api_timeout_sec:.0f}s")
        finally:
            # wait=False: do not block on the (possibly hung) worker thread.
            executor.shutdown(wait=False)
        return response.text if hasattr(response, "text") else str(response)
