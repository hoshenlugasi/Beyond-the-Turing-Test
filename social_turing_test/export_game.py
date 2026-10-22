"""Offline two-layer export for NEW games. Raw runtime files are never modified.

Creates a separate, privacy-PENDING projection in the existing study schemas;
it never adds games to the historical cohort or authorizes publication.
Only the Python standard library is used. No runtime/model imports or API calls.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile

from social_turing_test.core.dataset_tables import write_tables
from social_turing_test.core.event_schema import classify_event, make_manifest, project_event, render_text
from social_turing_test.core.hub_features import dataset_infos


class ExportError(ValueError):
    """Invalid, unsupported, inconsistent or unsafe export input."""


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
    segments = {}
    for row in raw:
        disposition = classify_event(row)
        require(disposition in {"included", "retired_diagnostic"},
                "Unsupported or malformed neutral event; export requires explicit schema review")
        if disposition != "included":
            omitted[disposition] += 1
            continue
        try:
            out = project_event(row, export_id, len(projected) + 1,
                                lambda text: redactor.text(text, internal=True),
                                turns, timing=event_timing(row, segments))
        except (ValueError, TypeError, OverflowError):
            raise ExportError("Invalid retained event payload or timing") from None
        require(out is not None, "Included event did not project")
        # Counts and delivery claims must agree with the recorded observation.
        kind, payload = out["action"], out["payload"]
        if kind == "generator_candidates":
            candidates = payload["candidates"]
            require(candidates["candidates"] is not None and
                    candidates["count"] == len(candidates["candidates"]),
                    "Candidate count mismatch")
        if kind == "scheduler_decision":
            require(payload["scheduler"]["decision"] in {"send", "wait"}, "Unknown scheduler decision")
        if kind in {"delivery_sent", "delivery_phase_drop"}:
            require(payload["delivery"]["outcome"] ==
                    ("sent" if kind == "delivery_sent" else "phase_drop"), "Delivery outcome mismatch")
        projected.append(out)
    return projected, omitted


def event_timing(row, segments):
    """Accept only the logger's paired monotonic-segment elapsed contract.

    elapsed_seconds is measured since this logger segment's anchor, NOT START.
    Wall occurrence dates validate syntax only and never supply relative time.
    Bare elapsed, epoch/wall_t and ambiguous old HHMMSS records remain unknown;
    no rollover, timezone bridge, missing interval or game anchor is inferred.
    Segment identifiers/anchors remain private. Segment restarts are independent.
    """
    occurred = row.get("occurred_at_utc")
    if occurred is not None:
        require(isinstance(occurred, str), "Occurrence timestamp must be offset ISO")
        try:
            parsed = datetime.fromisoformat(occurred.replace("Z", "+00:00"))
        except ValueError:
            raise ExportError("Occurrence timestamp must be offset ISO") from None
        require(parsed.tzinfo is not None and parsed.utcoffset() is not None,
                "Occurrence timestamp must carry an offset")
    paired = {"elapsed_seconds", "time_basis", "time_precision_seconds",
              "segment_id", "monotonic_ns"}
    if (not paired <= row.keys() or row.get("time_basis") != "monotonic_segment"
            or row.get("elapsed_seconds") is None):
        return None
    elapsed, mono = row["elapsed_seconds"], row["monotonic_ns"]
    segment, precision = row["segment_id"], row["time_precision_seconds"]
    require(isinstance(segment, str) and bool(segment), "Invalid clock segment")
    require(type(mono) is int and mono >= 0, "Invalid monotonic clock")
    require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0,
            "Invalid elapsed clock")
    require(type(precision) in (int, float) and precision == .001,
            "Unsupported monotonic precision")
    if segment in segments:
        previous_mono, previous_elapsed = segments[segment]
        require(mono >= previous_mono and elapsed >= previous_elapsed and
                abs((mono - previous_mono) / 1e9 - (elapsed - previous_elapsed)) <= .002,
                "Inconsistent paired monotonic timing")
    segments[segment] = (mono, elapsed)
    return dict(elapsed_seconds=elapsed, time_basis="monotonic_segment",
                time_precision_seconds=precision)


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
    package = Path(__file__).resolve().parent.parent
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
    timeline = core + "events/timeline.jsonl"
    payloads = {core + "metadata.json": metadata, core + "votes.json": votes, core + "surveys.json": surveys}
    files = {path: json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n" for path, obj in payloads.items()}
    files[core + "transcript.txt"] = redactor.text(transcript)
    files[timeline] = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in events)
    files[core + "events/timeline.txt"] = render_text(events)
    files[core + "events/manifest.json"] = json.dumps(
        make_manifest(events, omitted, privacy_status="pending"), indent=2, allow_nan=False) + "\n"
    manifest = {"schema_version": 1, "export_id": export_id, "collection_kind": kind,
                "historical_cohort_member": False, "publication_authorized": False,
                "privacy_review": {"status": "pending", "automatic_redaction_is_not_anonymity_guarantee": True,
                                   "requires": ["Review transcript and surveys", "Review rejected candidates, reviewer and worker text", "Check consent and redistribution rights"]},
                "missing_surveys": missing, "omitted_abstention_slots": abstentions,
                "omitted_event_count": sum(omitted.values()),
                "omitted_event_reason_counts": dict(sorted(omitted.items())),
                "automatic_redactions": {k: v for k, v in sorted(redactor.counts.items()) if v},
                "limitations": ["Raw runtime files remain local and unchanged", "Selection is not delivery; missing events are not synthesized", "No paid annotations, historical word counts or judge decisions are created", "Stopped/dropped/rescue/ambiguous games fail rather than infer missing ballots"],
                "files": {}}
    files["dataset_infos.json"] = json.dumps(dataset_infos(), indent=2, ensure_ascii=False) + "\n"
    configuration = "---\nconfigs:\n" + "".join(
        f"  - config_name: {name}\n" + ("    default: true\n" if name == "games" else "")
        + f"    data_files:\n      - split: data\n        path: "
        + ("games/*/events/timeline.jsonl" if name == "events" else f"tables/{name}.jsonl") + "\n"
        for name in ("games", "messages", "votes", "surveys", "events")) + "---\n\n"
    files["README.md"] = (configuration + "# New local game export — PRIVACY REVIEW PENDING\n\n"
        "NOT authorized for publication. NOT a member of the historical 50-game cohort.\n"
        f"Collection kind: {kind}.\n\n"
        "games/<id>/ contains the four observational files in the historical schemas.\n"
        "Its events/ directory contains timeline.jsonl, its full derived timeline.txt,\n"
        "and a coverage manifest for foreground, system and all six worker outputs.\n"
        "tables/ contains four deterministic access views derived from the core files.\n"
        "Only schema-approved fields are projected, not full memory or prompts.\n"
        "Event order is retained write order, not causal order. Monotonic timing is\n"
        "since a logger segment anchor, not game START; ambiguous clocks stay null.\n"
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
            try:
                write_tables(staging)
            except (ValueError, TypeError, OverflowError):
                raise ExportError("Canonical core/table validation failed") from None
            # Hash the complete payload, including tables built from staging and
            # the readable view. The manifest cannot hash itself.
            manifest["files"] = {
                path.relative_to(staging).as_posix(): {
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "bytes": path.stat().st_size}
                for path in sorted(staging.rglob("*")) if path.is_file()}
            (staging / "export_manifest.json").write_bytes(
                (json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
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