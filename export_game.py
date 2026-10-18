"""Offline two-layer export for NEW games. Raw runtime files are never modified.

Creates a separate, privacy-PENDING projection in the existing study schemas;
it never adds games to the historical cohort or authorizes publication.
Only the Python standard library is used. No runtime/model imports or API calls.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile


class ExportError(ValueError):
    """Invalid, unsupported, inconsistent or unsafe export input."""


# The exact historical public projection, not arbitrary raw event dictionaries.
EVENT_FIELDS = {
    "v4_pipeline_complete": "final review_risk scheduler_decision send_reason",
    "v5_behavior_advice_updated": "advice",
    "v5_overused_words_updated": "words",
    "v5_phase_summary_tick": "n_messages phase summary",
    "v5_repetition_llm_pass": "note",
    "v5_repetition_note_updated": "note",
    "v5_scheduler_blocked": "reason",
    "v6_delivery_phase_drop": "message outcome reason stage",
    "v6_delivery_sent": "message outcome",
    "v6_generator_candidates": "candidates count",
    "v6_reviewer_analysis": "num_candidates selected_index verdicts",
    "v6_reviewer_demote_response_required": "demoted pool_size tag",
    "v6_reviewer_demote_same_dodge_mode": "demoted dodge_streak mode pool_size tag",
    "v6_reviewer_drop": "num_candidates review_reason_codes review_status source",
    "v6_reviewer_filter_known_accusation": "kept removed",
    "v6_reviewer_selected": "num_candidates review_reason_codes review_status selected_index selected_text selected_word_count source",
    "v6_scheduler_decision": "decision send_reason",
    "v6_stale_send_drop": "new_messages reason suppressed_message",
    "v6_state_guard_filter": "kept removed",
}
VERDICT_FIELDS = "index pool_index text type pros cons score red_flags verdict selected".split()
STRING_FIELDS = set("final review_risk scheduler_decision send_reason advice phase summary note reason message outcome stage tag mode selected_text source review_status suppressed_message decision".split())
INTEGER_FIELDS = set("count num_candidates selected_index selected_word_count pool_size dodge_streak kept new_messages n_messages".split())
LIST_FIELDS = {"candidates", "words", "review_reason_codes", "demoted"}
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024
ID_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}\Z")
CODENAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}\Z")
RESERVED = {"CON", "PRN", "AUX", "NUL", "CLOCK$", "CONIN$", "CONOUT$"} | {
    f"{prefix}{i}" for prefix in ("COM", "LPT") for i in range(1, 10)}
PATTERNS = [
    ("SECRET", re.compile(r"-----BEGIN [^-]*PRIVATE KEY-----[\s\S]*?-----END [^-]*PRIVATE KEY-----|\bAIza[0-9A-Za-z_-]{35}\b|\b(?:sk-|ghp_|github_pat_)[0-9A-Za-z_-]{20,}", re.I)),
    ("CREDENTIAL", re.compile(r"\b(?:api[_ -]?key|password|access[_ -]?token|refresh[_ -]?token|bearer)\s*[:=]\s*[^\s,;]+", re.I)),
    ("EMAIL", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)),
    ("URL", re.compile(r"\b(?:https?://|www\.)[^\s<>\"']+", re.I)),
    ("HANDLE", re.compile(r"(?<!\w)@[\w.-]+|\b(?:u|r)/[\w-]+", re.I)),
    ("PATH", re.compile(r"(?:\b[A-Z]:[\\/]|\\\\)[^\s<>\"']+|(?<!\w)/(?:home|Users|users|tmp|var|mnt|srv|workspace|app|root)/[^\s<>\"']+", re.I)),
    ("IDENTIFIER", re.compile(r"\b[0-9a-f]{24,}\b|\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b", re.I)),
    ("PHONE", re.compile(r"(?<!\w)(?:\+?\d[\d ().-]{5,}\d)(?!\w)")),
    ("DATE", re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[T ][0-9:.+Z-]+)?\b")),
]


def require(condition, message):
    if not condition:
        raise ExportError(message)


def no_links(path):
    """Reject symlinks and Windows junctions, including existing ancestors."""
    for item in (path, *path.parents):
        require(not item.is_symlink(), "Symlink inputs/destinations are unsupported")
        if item.exists():
            require(not getattr(item.lstat(), "st_file_attributes", 0) & 0x400,
                    "Reparse-point inputs/destinations are unsupported")


def object_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON field")
        result[key] = value
    return result


def decode_json(text, label, diagnostic_numbers=False):
    try:
        return json.loads(text, object_pairs_hook=object_pairs,
                          parse_constant=(lambda value: float(value)) if diagnostic_numbers else
                          (lambda _: require(False, "Nonfinite JSON number")))
    except (ValueError, TypeError) as exc:
        if isinstance(exc, ExportError):
            raise
        raise ExportError(f"Malformed JSON: {label}") from None


class Inputs:
    def __init__(self, root):
        self.root, self.content = root, {}

    def text(self, name, optional=False):
        if name in self.content:
            return self.content[name].decode("utf-8-sig") if self.content[name] is not None else ""
        path = self.root / name
        no_links(path)
        if optional and not path.exists():
            self.content[name] = None
            return ""
        require(path.is_file(), f"Required runtime input missing: {name}")
        require(stat.S_ISREG(path.stat().st_mode) and path.stat().st_size <= MAX_FILE_BYTES,
                f"Input is not a bounded regular file: {name}")
        raw = path.read_bytes()
        require(len(raw) <= MAX_FILE_BYTES, f"Input exceeds size bound: {name}")
        require(sum(len(v) for v in self.content.values() if v is not None) + len(raw)
                <= MAX_TOTAL_BYTES, "Input exceeds total size bound")
        self.content[name] = raw
        try:
            return raw.decode("utf-8-sig")
        except UnicodeError:
            raise ExportError(f"Input is not UTF-8: {name}") from None

    def json(self, name):
        return decode_json(self.text(name), name)

    def jsonl(self, name, diagnostic_numbers=False):
        result = []
        for index, line in enumerate(self.text(name).splitlines(), 1):
            if line.strip():
                row = decode_json(line, f"{name} row {index}", diagnostic_numbers)
                require(isinstance(row, dict), f"Expected JSON object: {name} row {index}")
                result.append(row)
        return result

    def unchanged(self):
        for name, raw in self.content.items():
            path = self.root / name
            no_links(path)
            require(not path.exists() if raw is None else path.is_file() and path.read_bytes() == raw,
                    "Runtime inputs changed during export; stop writers and try a fresh destination")


class Redactor:
    def __init__(self, source_id, players, turns):
        self.turns = turns
        # Only explicit identity fields, never a public registry or mapping export.
        self.private = [str(p["real_name"]) for p in players
                        if isinstance(p.get("real_name"), str) and p["real_name"].strip()
                        and p["real_name"] != p["name"]]
        self.source_id = source_id
        self.counts = Counter()

    def text(self, value, internal=False):
        require(isinstance(value, str), "Expected event/free-text string")
        for raw, mapped in sorted(self.turns.items(), key=lambda item: -len(item[0])):
            value = value.replace(raw, mapped)
        for private in sorted(set(self.private), key=len, reverse=True):
            value, count = re.subn(r"(?<!\w)" + re.escape(private) + r"(?!\w)",
                                   "[REDACTED_NAME]", value, flags=re.I)
            self.counts["NAME"] += count
        value, count = re.subn(r"(?<!\w)" + re.escape(self.source_id) + r"(?!\w)",
                               "[REDACTED_SOURCE_ID]", value)
        self.counts["SOURCE_ID"] += count
        # Underscores delimit machine identifiers too (e.g. session_0003).
        value, count = re.subn(r"(?<![A-Za-z0-9])" + re.escape(self.source_id) + r"(?![A-Za-z0-9])",
                       "[REDACTED_SOURCE_ID]", value)
        self.counts["SOURCE_ID"] += count
        for label, pattern in PATTERNS:
            value, count = pattern.subn(f"[REDACTED_{label}]", value)
            self.counts[label] += count
        if internal:
            value, count = re.subn(r"\b\d{2}:\d{2}:\d{2}\b", "[REDACTED_CLOCK]", value)
            self.counts["CLOCK"] += count
        return value


def roster_and_outcome(inputs):
    config = inputs.json("config.json")
    require(isinstance(config, dict), "Malformed config object")
    players = config.get("players")
    require(isinstance(players, list) and len(players) >= 3, "Expected at least two humans and one AI")
    require(all(isinstance(p, dict) and isinstance(p.get("name"), str)
                and CODENAME.fullmatch(p["name"]) and p["name"].upper() not in RESERVED
                and type(p.get("is_ai")) is bool for p in players), "Invalid runtime codename/role")
    roster = [p["name"] for p in players]
    require(len({n.casefold() for n in roster}) == len(roster), "Duplicate runtime codename")
    ais = [p["name"] for p in players if p["is_ai"]]
    require(len(ais) == 1, "Export requires exactly one AI")
    ai = ais[0]
    require(inputs.text("ai_player.txt").strip() == ai, "AI identity/config mismatch")
    require(inputs.text("player_names.txt").splitlines() == roster, "Roster/config mismatch")
    require(not inputs.text("dropped_players.txt", optional=True).strip(),
            "Dropped-player games need a separately audited exporter; unsupported here")
    audit = inputs.jsonl("events/elimination_audit.jsonl")
    require(bool(audit), "Completed elimination audit is required")
    info = {}
    for line in inputs.text("ai_elimination_info.txt").splitlines():
        if line.strip():
            key, sep, val = line.partition(" - ")
            require(sep and key not in info, "Malformed outcome summary")
            info[key] = val.strip()
    require(info.get("TOTAL_ROUNDS") == str(len(audit)), "Outcome/audit round mismatch")
    return config, players, roster, ai, audit, info


def ballots(inputs, roster, ai, audit, info):
    streams = {n: inputs.text(f"votes/{n}_vote.txt").splitlines() for n in roster}
    cursors, active, result = Counter(), list(roster), []
    abstentions = 0
    for index, row in enumerate(audit, 1):
        require(row.get("source") in (None, ""), "Rescue tally audits are unsupported; review manually")
        tally = row.get("tally")
        require(isinstance(tally, dict) and set(tally) == set(active)
                and all(type(v) is int and 0 <= v <= len(active) for v in tally.values()),
                f"Invalid active roster/tally in round {index}")
        require(ai in active and len(active) >= 3, "Audit continues after terminal round")
        counted = Counter()
        for voter in active:
            require(cursors[voter] < len(streams[voter]), f"Missing ballot slot in round {index}")
            target = streams[voter][cursors[voter]].strip()
            cursors[voter] += 1
            if target in ("", "<ABSTAIN>"):
                abstentions += 1
                continue
            require(target in active and target != voter, f"Invalid ballot in round {index}")
            counted[target] += 1
            result.append({"round": index, "voter": voter, "voted_for": target, "is_ai_voter": voter == ai})
        require(counted == Counter({n: v for n, v in tally.items() if v}), f"Ballot/tally mismatch in round {index}")
        maximum = max(tally.values())
        tied = {n for n, v in tally.items() if v == maximum}
        require(row.get("max_votes") == maximum and type(row.get("max_votes")) is int
                and isinstance(row.get("tied_candidates"), list)
                and len(row["tied_candidates"]) == len(tied) and set(row["tied_candidates"]) == tied,
                f"Invalid tie metadata in round {index}")
        require(row.get("voted_out") in tied and row.get("tie_break") == ("random" if len(tied) > 1 else "single_max"),
                f"Invalid recorded elimination in round {index}")
        active.remove(row["voted_out"])
        if index < len(audit):
            require(ai in active and len(active) > 2, "Nonterminal round ended the game")
    require(all(cursors[n] == len(streams[n]) for n in roster), "Unused/extra ballot lines")
    require(set(inputs.text("remaining_players.txt").splitlines()) == set(active), "Final remaining roster mismatch")
    eliminated = ai not in active
    require(eliminated or len(active) == 2, "Recorded game is not complete")
    expected = str(len(audit)) if eliminated else "survived"
    require(info.get("AI_ELIMINATED_ROUND") == expected, "AI elimination/outcome mismatch")
    require(inputs.text("who_wins.txt").strip() == ("Humans win!" if eliminated else "AI wins!"),
            "Winner file/outcome mismatch")
    return result, abstentions, eliminated


def displayed_transcript(inputs, audit, roster, anonymous, votes):
    lines = inputs.text("transcripts/chat_displayed_full.txt").splitlines()
    require(lines and lines[0].startswith("# Full chat AS USERS SAW IT"), "Canonical displayed transcript header missing")
    body = "\n".join(lines[1:]).strip("\n") + "\n"
    speakers = re.findall(r"^\[\d{2}:\d{2}:\d{2}\] ([^:]+):", body, re.M)
    require(set(speakers) <= {*roster, "Game-Manager"}, "Unknown transcript speaker")
    # Tally blocks are the authoritative round delimiters; never order by clocks.
    markers = list(re.finditer(r"^\[\d{2}:\d{2}:\d{2}\] Game-Manager:.*Vote results:\s*$", body, re.M))
    require(len(markers) == len(audit), "Displayed tally/audit count mismatch")
    for index, marker in enumerate(markers):
        end = markers[index+1].start() if index+1 < len(markers) else len(body)
        tail = body[marker.end():end]
        matches = list(re.finditer(r"^\s*→ eliminated: ([A-Za-z0-9_-]+)\s*$", tail, re.M))
        require(len(matches) == 1 and matches[0][1] == audit[index]["voted_out"], "Displayed elimination/audit mismatch")
        match = matches[0]
        block = tail[:match.start()]
        if anonymous:
            pairs = re.findall(r"^\s+([A-Za-z0-9_-]+): (\d+)\s*$", block, re.M)
            require(len(pairs) == len({n for n, _ in pairs}), "Duplicate displayed tally")
            require({n: int(v) for n, v in pairs} == {n: v for n, v in audit[index]["tally"].items() if v},
                    "Displayed tally/audit mismatch")
        else:
            pairs = re.findall(r"^\s+([A-Za-z0-9_-]+) voted for ([A-Za-z0-9_-]+)\s*$", block, re.M)
            expected = [(v["voter"], v["voted_for"]) for v in votes if v["round"] == index+1]
            require(Counter(pairs) == Counter(expected), "Displayed ballots/audit mismatch")
    return body


def survey(inputs, name):
    text = inputs.text(f"surveys/{name}_survey.txt", optional=True)
    if not text.strip():
        return None
    lines = text.splitlines()
    require(lines.count("Comments:") == 1, "Malformed survey Comments delimiter")
    split = lines.index("Comments:")
    preamble, comments = lines[:split], lines[split + 1:]

    def rating(prefix):
        found = [line[len(prefix) + 3:].strip() for line in preamble if line.startswith(prefix + " - ")]
        require(len(found) == 1 and re.fullmatch("[1-5]", found[0]), "Malformed/out-of-range survey rating")
        return int(found[0])

    row = {"participant": name, "human_likeness": rating("similarity to human behavior"),
           "timing": rating("timing of messaging"), "relevance": rating("relevance of messages")}
    compat = [line for line in preamble if line.startswith("Was the LLM identified")]
    require(compat in ([], ["Was the LLM identified - 0"]), "Unsupported identification survey field")
    if not any(line.startswith("enjoyment of game") for line in preamble):
        return {**row, "survey_schema": "legacy_terminal_v1", "comments": "\n".join(comments)}
    first, second = "Suspicion reasoning: ", "Other comments or suggestions: "
    indices = [i for i, line in enumerate(comments) if line.startswith(second)]
    require(comments and comments[0].startswith(first) and len(indices) == 1 and indices[0] > 0,
            "Malformed web survey free-text fields")
    pos = indices[0]
    suspicion = "\n".join([comments[0][len(first):], *comments[1:pos]])
    additional = "\n".join([comments[pos][len(second):], *comments[pos + 1:]])
    require(suspicion.strip() and additional.strip(), "Empty required web survey text")
    return {**row, "survey_schema": "web_v1", "enjoyment": rating("enjoyment of game"),
            "suspicion_reason": suspicion, "additional_comments": additional}


def project_events(raw, export_id, redactor, turns):
    projected, omitted = [], Counter()

    def strings(value):
        require(isinstance(value, list) and all(isinstance(v, str) for v in value), "Expected event string list")
        return [redactor.text(v, internal=True) for v in value]

    for row in raw:
        kind = row.get("event_type")
        require(isinstance(kind, str), "Event type missing or malformed")
        if kind not in EVENT_FIELDS:
            omitted[kind] += 1
            continue
        out = {"game_id": export_id, "event_type": kind, "event_order": len(projected) + 1}
        if row.get("turn_id") is not None:
            out["turn_id"] = turns[row["turn_id"]]
        for key in EVENT_FIELDS[kind].split():
            if key not in row:
                continue  # Missing observation is not a default value or zero.
            value = row[key]
            if key in STRING_FIELDS:
                out[key] = redactor.text(value, internal=True)
            elif key in INTEGER_FIELDS:
                require(type(value) is int, "Expected integer event field")
                out[key] = value
            elif key in LIST_FIELDS or (key == "removed" and kind != "v6_state_guard_filter"):
                out[key] = strings(value)
            elif key == "removed":
                require(isinstance(value, list) and all(isinstance(v, dict) for v in value), "Malformed guard removals")
                out[key] = [{k: redactor.text(v[k], internal=True) for k in ("text", "reason") if k in v} for v in value]
            elif key == "verdicts":
                require(isinstance(value, list) and all(isinstance(v, dict) for v in value), "Malformed reviewer verdicts")
                verdicts = []
                for verdict in value:
                    clean = {}
                    for field in VERDICT_FIELDS:
                        if field not in verdict:
                            continue
                        v = verdict[field]
                        if field in {"index", "pool_index"}:
                            require(type(v) is int, "Malformed verdict index")
                        elif field == "score":
                            require(type(v) in (int, float) and math.isfinite(v), "Malformed verdict score")
                        elif field == "selected":
                            require(type(v) is bool, "Malformed verdict selection")
                        elif field == "red_flags":
                            v = strings(v)
                        else:
                            v = redactor.text(v, internal=True)
                        clean[field] = v
                    verdicts.append(clean)
                out[key] = verdicts
        if kind == "v6_scheduler_decision":
            require(out.get("decision") in {"send", "wait"}, "Unknown scheduler decision")
        if kind == "v6_generator_candidates":
            require(out.get("count") == len(out.get("candidates", [])), "Candidate count mismatch")
        if kind in {"v6_delivery_sent", "v6_delivery_phase_drop"}:
            require(out.get("outcome") == ("sent" if kind == "v6_delivery_sent" else "phase_drop"), "Delivery outcome mismatch")
        projected.append(out)
    return projected, omitted


def publish_noreplace(staging, target):
    """Atomically publish a directory, never replace even an empty destination."""
    if os.name == "nt":
        os.rename(staging, target)  # Windows rename never replaces a destination.
        return
    import ctypes
    import sys
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        result = libc.renameat2(-100, os.fsencode(staging), -100, os.fsencode(target), 1)
    elif sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        result = libc.renamex_np(os.fsencode(staging), os.fsencode(target), 4)
    else:
        raise ExportError("Atomic no-replace directory publication is unsupported on this platform")
    if result != 0:
        import errno
        code = ctypes.get_errno()
        raise OSError(code, errno.errorcode.get(code, "Directory publication failed"))


def export_game(game_dir: Path, output: Path, export_id: str, kind: str) -> dict:
    """Write a fresh independent, privacy-pending two-layer projection."""
    require(isinstance(export_id, str) and ID_PATTERN.fullmatch(export_id)
            and export_id.upper() not in RESERVED and not re.fullmatch(r"game_\d{3}", export_id),
            "Use a safe, independent export ID (not historical game_NNN)")
    require(kind in {"local_game", "scripted_test"}, "Explicit collection kind required")
    source = Path(os.path.abspath(game_dir))
    target = Path(os.path.abspath(output))
    no_links(source)
    no_links(target)
    require(source.is_dir(), "Runtime game directory missing")
    require(not target.exists(), "Output already exists; overwriting is forbidden")
    require(target != source and source not in target.parents and target not in source.parents,
            "Output must not overlap runtime source")
    package = Path(__file__).resolve().parent
    if target == package or package in target.parents:
        require(package / "local_exports" in target.parents, "Inside the code checkout, export only under local_exports/")
    require(not any((parent / "game_001/metadata.json").is_file() and
                    (parent / "game_050/metadata.json").is_file() for parent in target.parents),
            "Cannot export inside a frozen 50-game dataset")
    inputs = Inputs(source)
    config, players, roster, ai, audit, info = roster_and_outcome(inputs)
    votes, abstentions, eliminated = ballots(inputs, roster, ai, audit, info)
    transcript = displayed_transcript(inputs, audit, roster, config.get("anonymous_voting", True), votes)
    # The runtime writer can emit Infinity for an unbounded cadence diagnostic.
    # Parse discarded diagnostics without copying them. Retained numeric fields
    # are explicitly finite/type checked, and serialized with allow_nan=False.
    raw_events = inputs.jsonl("events/game_events.jsonl", diagnostic_numbers=True)
    turns = {}
    for row in raw_events:
        turn = row.get("turn_id")
        if turn is not None:
            require(isinstance(turn, str) and bool(turn), "Invalid raw turn identifier")
            if turn not in turns:
                turns[turn] = f"{export_id}-turn{len(turns)+1:03d}"
    redactor = Redactor(source.name, players, turns)
    surveys, missing = [], []
    for name in roster:
        if name == ai:
            continue
        row = survey(inputs, name)
        if row is None:
            missing.append(name)
        else:
            for field in ("comments", "suspicion_reason", "additional_comments"):
                if field in row:
                    row[field] = redactor.text(row[field])
            surveys.append(row)
    events, omitted = project_events(raw_events, export_id, redactor, turns)
    metadata = {"game_id": export_id, "ai_pseudonym": ai, "num_human_players": len(roster)-1,
                "total_rounds": len(audit), "ai_eliminated": eliminated,
                "ai_elimination_round": len(audit) if eliminated else None}
    core = f"games/{export_id}/"
    internal = f"internal_outputs/games/{export_id}/events/game_events.jsonl"
    payloads = {core + "metadata.json": metadata, core + "votes.json": votes, core + "surveys.json": surveys}
    files = {path: json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n" for path, obj in payloads.items()}
    files[core + "transcript.txt"] = redactor.text(transcript)
    files[internal] = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in events)
    internal_manifest = {"schema_version": 1, "games": 1, "events": len(events),
                         "event_types": dict(sorted(Counter(row["event_type"] for row in events).items())),
                         "scope": "Same historical field allowlist; selected worker updates, not full memory",
                         "privacy_review": {"status": "pending"}, "publication_authorized": False}
    files["internal_outputs/manifest.json"] = json.dumps(internal_manifest, indent=2) + "\n"
    manifest = {"schema_version": 1, "export_id": export_id, "collection_kind": kind,
                "historical_cohort_member": False, "publication_authorized": False,
                "privacy_review": {"status": "pending", "automatic_redaction_is_not_anonymity_guarantee": True,
                                   "requires": ["Review transcript and surveys", "Review rejected candidates, reviewer and worker text", "Check consent and redistribution rights"]},
                "missing_surveys": missing, "omitted_abstention_slots": abstentions,
                "omitted_event_count": sum(omitted.values()),
                "automatic_redactions": {k: v for k, v in sorted(redactor.counts.items()) if v},
                "limitations": ["Raw runtime files remain local and unchanged", "Selection is not delivery; missing events are not synthesized", "No paid annotations, historical word counts or judge decisions are created", "Stopped/dropped/rescue/ambiguous games fail rather than infer missing ballots"],
                "files": {name: {"sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "bytes": len(text.encode("utf-8"))} for name, text in files.items()}}
    files["export_manifest.json"] = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    files["README.md"] = ("# New local game export — PRIVACY REVIEW PENDING\n\n"
        "NOT authorized for publication. NOT a member of the historical 50-game cohort.\n"
        f"Collection kind: {kind}.\n\n"
        "games/ contains the four observational files in the historical schemas.\n"
        "internal_outputs/ contains only allowlisted candidates, reviewer decisions,\n"
        "delivery events and selected worker updates, not full memory or prompts.\n"
        "Automatic masking is incomplete: manually review ALL text, including\n"
        "rejected candidates and derived worker/reviewer text, and verify consent\n"
        "and rights before any redistribution. Missing events are unknown, not zero.\n"
        "The full raw game remains at its original LOCAL location; do not upload it.\n")
    # Validate everything before touching output. Same-parent staging + rename:
    # either a complete tree appears or nothing does. Exclusive sibling lock
    # serializes exporter writers; never merge into an existing destination.
    inputs.unchanged()
    target.parent.mkdir(parents=True, exist_ok=True)
    no_links(target.parent)
    lock = target.with_name(target.name + ".export.lock")
    require(not lock.exists(), "Export destination is locked")
    staging, owned_lock = None, False
    try:
        with lock.open("x", encoding="utf-8"):
            owned_lock = True
            require(not target.exists(), "Output already exists; overwriting is forbidden")
            staging = Path(tempfile.mkdtemp(prefix=".game-export-", dir=target.parent))
            for name, text in files.items():
                path = staging / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(text.encode("utf-8"))
            inputs.unchanged()
            require(not target.exists(), "Output appeared during export; refusing overwrite")
            publish_noreplace(staging, target)
            staging = None
    except OSError as exc:
        raise ExportError(f"Could not write fresh export ({type(exc).__name__}); existing output was not merged") from None
    finally:
        if staging is not None:
            shutil.rmtree(staging)
        if owned_lock:
            lock.unlink(missing_ok=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-dir", type=Path, required=True, help="Explicit completed raw runtime directory")
    parser.add_argument("--output", type=Path, required=True, help="New independent output directory; never overwritten")
    parser.add_argument("--export-id", required=True, help="Independent ID, e.g. local_game_001; never a historical game_NNN")
    parser.add_argument("--kind", required=True, choices=("local_game", "scripted_test"))
    args = parser.parse_args()
    try:
        result = export_game(args.game_dir, args.output, args.export_id, args.kind)
    except (ExportError, OSError) as exc:
        parser.exit(2, f"Export refused: {exc}\n")
    print(json.dumps({"export_id": result["export_id"], "privacy_review": "pending",
                      "publication_authorized": False, "historical_cohort_member": False}, indent=2))


if __name__ == "__main__":
    main()