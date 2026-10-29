"""Shared formatting helpers for the per-component agent logs.

The master log (``logs/{name}_log.txt``) stays in the raw, append-only
``# NEW LOG`` form because the offline analysis (``analyze_agents_game.py``)
parses it block-by-block. The *per-component* split files
(``logs/{name}_{component}.txt``) are meant to be read by a human, so they show
each LLM call as ONE consolidated block — prompt, output and timing together —
instead of a stream of scattered ``# NEW LOG`` entries.

This module provides:

* ``render_note`` — a compact one-liner for an auxiliary note (used live by the
  logger when it appends a side-note to a component file).
* ``render_grouped`` — turn a list of role-tagged events (prompt / output /
  latency / note) into the consolidated per-call view. A call keeps only its
  first output plus tightly-following notes; later periodic notes are emitted as
  their own lines so unrelated entries don't collapse into a single block.
* ``parse_blocks`` / ``consolidate_text`` — parse a raw ``# NEW LOG`` stream and
  render it consolidated. Already-consolidated text is returned unchanged, so
  the function is idempotent.
"""

import re

_RULE = "=" * 80
_SUBRULE = "-" * 80

# How close (seconds) a trailing note must be to a call's output to be treated
# as part of that call rather than a separate standalone note.
_ATTACH_WINDOW_S = 4

# A single raw entry written by Logger.log(): captures time / operation and the
# (possibly multi-line) content up to the next entry or the end of the text.
_NEWLOG_RE = re.compile(
    r"# NEW LOG\n"
    r"## TIME: (?P<time>.*?)\n"
    r"## OPERATION: (?P<op>.*?)\n"
    r"## CONTENT: (?P<content>.*?)"
    r"(?=\n# NEW LOG\n|\Z)",
    re.DOTALL,
)

# Operation-name hints used to classify an output block.
_OUTPUT_HINTS = (
    "response", "raw", "claims", "advice", "output", "_update",
    "summary", "memory", "result",
)


def _to_secs(t: str):
    """Parse an ``HH:MM:SS`` timestamp prefix into seconds (or ``None``)."""
    m = re.match(r"\s*(\d{1,2}):(\d{2}):(\d{2})", t or "")
    if not m:
        return None
    h, mi, s = (int(x) for x in m.groups())
    return h * 3600 + mi * 60 + s


def _first_number(text: str):
    m = re.search(r"[-+]?\d*\.?\d+", text or "")
    return m.group(0) if m else None


def infer_role(op: str) -> str:
    """Best-effort role for a raw operation name.

    Returns one of ``prompt`` / ``output`` / ``latency`` / ``note``.
    """
    low = op.lower()
    if "prompt" in low:
        return "prompt"
    if "latency" in low:
        return "latency"
    if any(h in low for h in _OUTPUT_HINTS):
        return "output"
    return "note"


def render_note(time: str, operation: str, content) -> str:
    """Compact, single ``·`` line for an auxiliary note.

    Multi-line content is indented under the operation so the component file
    stays readable without the ``# NEW LOG`` boilerplate.
    """
    op = str(operation).strip()
    body = "" if content is None else str(content).strip()
    if not body:
        return f"\u00b7 [{time}] {op}\n"
    lines = body.splitlines()
    if len(lines) == 1 and len(body) <= 200:
        return f"\u00b7 [{time}] {op}: {body}\n"
    indented = "\n".join("    " + ln for ln in lines)
    return f"\u00b7 [{time}] {op}:\n{indented}\n"


def _render_call(call: dict, label: str | None) -> str:
    """Render one grouped call (prompt + output + notes) as a ``====`` block."""
    prompt = call["prompt"]
    header_bits = []
    if call.get("latency"):
        header_bits.append(f"latency {call['latency']}")
    header_extra = f"  ({' | '.join(header_bits)})" if header_bits else ""

    t_start = prompt["time"]
    t_end = call.get("last_time") or t_start
    when = t_start if t_end == t_start else f"{t_start} \u2192 {t_end}"
    title = (label or call.get("label") or "call").upper()

    parts = [
        _RULE,
        f"[{when}]  {title}{header_extra}",
        _SUBRULE,
        "### PROMPT",
        (prompt["content"] or "").strip(),
        "",
    ]
    if call.get("output") is not None:
        parts.append("### OUTPUT")
        parts.append((call["output"]["content"] or "").strip())
        parts.append("")
    for note in call.get("notes", []):
        parts.append(render_note(note["time"], note["op"], note["content"]).rstrip("\n"))
    parts.append(_RULE)
    parts.append("")
    parts.append("")
    return "\n".join(parts)


def render_grouped(events: list[dict], label: str | None = None) -> str:
    """Render role-tagged events as consolidated per-call blocks.

    Each event is a dict ``{time, op, content, role}`` where ``role`` is one of
    ``prompt`` / ``output`` / ``latency`` / ``note``. A call starts at a
    ``prompt`` and absorbs the first ``output``, the ``latency`` and any notes
    that arrive within ``_ATTACH_WINDOW_S`` seconds of the output. A second
    output, or a note that arrives later, flushes the current call and is
    emitted as its own standalone ``\u00b7`` line so unrelated periodic entries
    don't get swallowed into one block.
    """
    out: list[str] = []
    call = None

    def flush():
        nonlocal call
        if call is not None:
            out.append(_render_call(call, label))
            call = None

    for ev in events:
        time = ev["time"]
        op = ev["op"]
        content = ev["content"]
        role = ev.get("role") or infer_role(op)

        if role == "prompt":
            flush()
            call = {"prompt": {"time": time, "content": content},
                    "output": None, "latency": None, "notes": [],
                    "last_time": time}
            continue

        if call is None:
            out.append(render_note(time, op, content))
            continue

        if role == "latency":
            num = _first_number(content)
            call["latency"] = f"{num}s" if num is not None else (content or "").strip()
            call["last_time"] = time
            continue

        if role == "output":
            if call["output"] is None:
                call["output"] = {"op": op, "content": content}
                call["last_time"] = time
            else:
                # A second output means a new, unrelated entry.
                flush()
                out.append(render_note(time, op, content))
            continue

        # role == "note": attach only if it arrives close to the call's output.
        gap = None
        if call.get("output") is not None:
            a, b = _to_secs(time), _to_secs(call["last_time"])
            if a is not None and b is not None:
                gap = abs(a - b)
        if gap is not None and gap > _ATTACH_WINDOW_S:
            flush()
            out.append(render_note(time, op, content))
        else:
            call["notes"].append({"time": time, "op": op, "content": content})
            call["last_time"] = time

    flush()
    result = "".join(out)
    result = re.sub(r"\n{4,}", "\n\n\n", result)
    return result


def parse_blocks(text: str) -> list[dict]:
    """Parse a raw ``# NEW LOG`` stream into ``{time, op, content}`` dicts."""
    return [
        {"time": m.group("time").strip(),
         "op": m.group("op").strip(),
         "content": m.group("content")}
        for m in _NEWLOG_RE.finditer(text or "")
    ]


def consolidate_text(text: str, label: str | None = None) -> str:
    """Convert a raw ``# NEW LOG`` stream into consolidated per-call blocks.

    Roles are inferred from operation names. Already-consolidated text (no raw
    ``# NEW LOG`` entries) is returned unchanged, so the function is idempotent.
    """
    if not text or not _NEWLOG_RE.search(text):
        return text
    events = [{**b, "role": infer_role(b["op"])} for b in parse_blocks(text)]
    return render_grouped(events, label)
