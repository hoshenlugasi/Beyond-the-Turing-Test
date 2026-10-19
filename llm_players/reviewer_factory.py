"""
Reviewer Factory -- creates a reviewer by backend name.
"""

from llm_players.reviewer_base import BaseReviewer


def create_reviewer(backend: str, model_name: str = None,
                    threshold: int = None, logger=None,
                    api_timeout_sec: float = None,
                    experimental: bool = False) -> BaseReviewer:
    """
    Instantiate a reviewer for the given backend.

    backend:         "gemini" (default), "together", or "openrouter"
    model_name:      Override the default model for that backend.
    threshold:       Suspicion score threshold 1-10 (default: 4).
    logger:          Optional logger instance.
    api_timeout_sec: Per-call API timeout (only honored by GeminiReviewer
                     today; safely ignored by other backends).
    experimental:    flag that adds an anti-shrink clause to the retry prompt.
    """
    if backend == "together":
        from llm_players.together_reviewer import TogetherReviewer
        return TogetherReviewer(model_name=model_name, threshold=threshold, logger=logger,
                                experimental=experimental)
    if backend == "openrouter":
        from llm_players.openrouter_reviewer import OpenRouterReviewer
        return OpenRouterReviewer(model_name=model_name, threshold=threshold, logger=logger,
                                  experimental=experimental)
    from llm_players.gemini_reviewer import GeminiReviewer
    return GeminiReviewer(model_name=model_name, threshold=threshold, logger=logger,
                          api_timeout_sec=api_timeout_sec, experimental=experimental)
