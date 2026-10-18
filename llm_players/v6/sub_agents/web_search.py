"""Web search backend for V6 InternetContextAgent.

Uses Gemini grounding with the Google Search tool. Authentication uses an
environment API key or explicitly configured Vertex AI Application Default
Credentials, as for the other Gemini roles. No evaluator is required.

Returns the same shape the old DDG wrapper did:
    [{"title": ..., "snippet": ..., "url": ...}, ...]

so the InternetContextAgent integration stays unchanged.
"""
from __future__ import annotations

import os
import time as _time
from typing import List, Optional

# Lazy import — google.genai is heavy and may not be present in every env
_client_cache = None


def _extract_answer(resp) -> str:
    """Pull text out of a grounded response. `resp.text` is sometimes empty even
    when the answer is present in `candidate.content.parts` (observed under load),
    so fall back to concatenating the parts before giving up."""
    txt = (getattr(resp, "text", None) or "").strip()
    if txt:
        return txt
    try:
        for cand in (getattr(resp, "candidates", None) or []):
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) or []
            joined = " ".join(
                (getattr(p, "text", None) or "").strip() for p in parts
            ).strip()
            if joined:
                return joined
    except (AttributeError, IndexError, TypeError):
        pass
    return ""


def _finish_reason(resp) -> str:
    try:
        return str(getattr(resp.candidates[0], "finish_reason", "") or "")
    except (AttributeError, IndexError, TypeError):
        return ""


def _get_client():
    """Cache a client using an environment key or explicitly configured Vertex ADC."""
    global _client_cache
    if _client_cache is not None:
        return _client_cache

    from google import genai  # type: ignore
    api_key = None
    try:
        from llm_players.llm_wrapper import get_api_key
        api_key = get_api_key("GOOGLE_API_KEY") or get_api_key("GEMINI_API_KEY")
    except Exception:
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

    if api_key:
        print("[web_search auth] Using API key for Gemini grounding", flush=True)
        _client_cache = genai.Client(api_key=api_key)
    else:
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION")
        if not project or not location:
            raise RuntimeError(
                "Gemini search authentication missing. Set GOOGLE_API_KEY or GEMINI_API_KEY, "
                "or explicitly set GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION "
                "for your own Vertex AI project with Application Default Credentials."
            )
        print(f"[web_search auth] Using Vertex AI (project={project}, location={location})", flush=True)
        _client_cache = genai.Client(vertexai=True, project=project, location=location)
    return _client_cache


def gemini_grounded_search(
    query: str,
    max_results: int = 3,
    model: str = "gemini-2.5-flash",
    today_str: Optional[str] = None,
) -> List[dict]:
    """Run a grounded Gemini call with Google Search and return ranked sources.

    The first entry in the returned list contains Gemini's synthesised answer
    (title="answer", snippet=<1–2 sentences>, url=""). Subsequent entries are
    grounding sources extracted from the response metadata.

    Returns an empty list when the model produces no answer (a genuine soft
    "no results"). REAL errors — a failed import, auth/quota/region problems, or
    the Google-Search tool not being enabled on the Vertex project — are RAISED
    so the caller can log the actual reason. Previously every exception was
    swallowed into [], which made a broken-on-server search look identical to an
    empty one and was impossible to diagnose.
    """
    from google.genai import types  # type: ignore
    client = _get_client()
    config = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
        temperature=0.3,
    )
    # Pin the current date as the PRESENT. Without this the grounded model
    # treats the real game date as "the future" and refuses ("the year 2026 has
    # not yet occurred", "that's a future date") instead of using the live search
    # results — those refusals were then stored as facts (game 9059).
    date_pin = (
        f"Today is {today_str} — this is the PRESENT, the current real date, NOT "
        f"the future. Treat today and everything earlier as already happened. Base "
        f"your answer ONLY on the Google Search results, never on assumptions about "
        f"what year it is.\n"
        if today_str else ""
    )
    prompt = (
        f"{date_pin}"
        f"Search for: {query}\n"
        "Reply with 1–2 short factual sentences. No preamble, no citations."
    )
    # In a live game the main generator plus all five sub-agents hit Gemini at
    # once, and under that load the grounded call sometimes comes back with an
    # empty answer even though the same query returns real results on a fresh
    # call (verified: basketball/war queries that logged "0 results" in-game
    # answer fine standalone). One immediate retry recovers those transient
    # blanks instead of letting them surface as a silent "no results".
    answer = ""
    resp = None
    finish_reasons = []
    attempts = 3
    for _attempt in range(attempts):
        resp = client.models.generate_content(
            model=model, contents=prompt, config=config,
        )
        answer = _extract_answer(resp)
        if answer:
            break
        fr = _finish_reason(resp)
        if fr:
            finish_reasons.append(fr)
        _time.sleep(0.5 * (_attempt + 1))   # backoff between transient blanks
    if not answer:
        # Distinguish empty-under-load from a true error: RAISE with the
        # finish_reason so the caller logs WHY (instead of a silent "0 results").
        raise RuntimeError(
            f"grounding returned empty text after {attempts} tries "
            f"(finish_reasons={finish_reasons or 'unknown'})"
        )

    results: List[dict] = [{"title": "answer", "snippet": answer, "url": "", "grounded": False}]

    # Try to extract grounding sources (URLs Gemini used). This metadata is
    # best-effort, so its own parse errors stay swallowed. Its PRESENCE also tells
    # us whether the answer was actually grounded in live search results vs.
    # produced from the model's training prior — an ungrounded answer to a
    # volatile/current query is unreliable (it tends to be a "that date is in the
    # future" refusal), and the caller drops it (game 9059 fix).
    try:
        cand = resp.candidates[0]
        gm = getattr(cand, "grounding_metadata", None)
        if gm:
            chunks = getattr(gm, "grounding_chunks", None) or []
            if chunks:
                results[0]["grounded"] = True
            for ch in chunks[: max_results - 1]:
                web = getattr(ch, "web", None)
                if web:
                    title = getattr(web, "title", "") or ""
                    uri = getattr(web, "uri", "") or ""
                    if title or uri:
                        results.append({
                            "title": title,
                            "snippet": "",
                            "url": uri,
                        })
    except (AttributeError, IndexError):
        pass

    return results[:max_results]


# Backward-compatible alias — the InternetContextAgent still calls
# duckduckgo_search(...). It now routes through Gemini grounding.
def duckduckgo_search(query: str, max_results: int = 3,
                      today_str: Optional[str] = None) -> List[dict]:
    """Legacy name kept for backward compatibility.

    Despite the name, this now uses Gemini grounding (see module docstring).
    """
    return gemini_grounded_search(query, max_results=max_results, today_str=today_str)
