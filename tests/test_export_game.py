"""Offline contract tests for exporting NEW local games, never the cohort.

All input records are synthetic and written under tmp_path. No runtime/SDK is
imported, and no approved game data is read. The CLI subprocess is offline and
uses only the exporter; these tests do not launch a game or change its timing.
"""

import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

import export_game as exporter
from export_game import ExportError, export_game


EXPORT_ID = "local_game_001"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CORE_FILES = {"metadata.json", "transcript.txt", "votes.json", "surveys.json"}
META_KEYS = {"game_id", "ai_pseudonym", "num_human_players", "total_rounds",
             "ai_eliminated", "ai_elimination_round"}
VOTE_KEYS = {"round", "voter", "voted_for", "is_ai_voter"}
# Copied deliberately from test_release_data.py, not from the implementation.
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
VERDICT_KEYS = {"index", "text", "type", "pros", "cons", "score", "red_flags",
                "verdict", "selected", "pool_index"}


def write_text(base, relative, text):
    path = base / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_json(base, relative, value):
    return write_text(base, relative, json.dumps(value, ensure_ascii=False) + "\n")


def write_jsonl(base, relative, rows):
    return write_text(base, relative, "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def snapshot(base):
    """Check bytes AND directory inventory; the exporter must not mutate raw data."""
    return {p.relative_to(base).as_posix(): p.read_bytes() if p.is_file() else None
            for p in base.rglob("*")}


def audit(tally, voted_out, clock="12:01:00"):
    maximum = max(tally.values())
    tied = [name for name, count in tally.items() if count == maximum]
    return {"timestamp": clock, "tally": tally, "max_votes": maximum,
            "tied_candidates": tied, "voted_out": voted_out,
            "tie_break": "random" if len(tied) > 1 else "single_max"}


def transcript(rows, include_announcement=True):
    lines = [
        '# Full chat AS USERS SAW IT — canonical displayed transcript',
        '[12:00:01] Blue: hello everyone',
        '[12:00:02] AI: hello there',
        '[12:00:03] Green: nice to meet you',
    ]
    for row in rows:
        lines.append(f'[{row["timestamp"]}] Game-Manager: 🗳️ Vote results:')
        lines.extend(f'    {name}: {count}' for name, count in row["tally"].items()
                     if count)
        lines.append(f'    → eliminated: {row["voted_out"]}')
        if include_announcement:
            lines.append(f'[{row["timestamp"]}] Game-Manager: '
                         f'{row["voted_out"]} was eliminated!')
    return "\n".join(lines) + "\n"


def web_survey():
    # The historical web compatibility value is not a fourth rating of zero.
    return ("TIME - 12:02:00\nWas the LLM identified - 0\n"
            "similarity to human behavior - 4\ntiming of messaging - 3\n"
            "relevance of messages - 5\nenjoyment of game - 4\nComments:\n"
            "Suspicion reasoning: short answers\n"
            "Other comments or suggestions: enjoyable\n")


def legacy_survey():
    return ("similarity to human behavior - 3\ntiming of messaging - 4\n"
            "relevance of messages - 5\nComments:\ninteresting game\n")


def event_rows():
    candidates = ["hello there", "nice weather", "fair enough", "sounds good"]
    raw_turn = "synthetic-source-8123-private-turn-29"
    common = {"schema_version": 1, "timestamp": "12:00:02", "player": "AI",
              "turn_id": raw_turn, "turn_index": 29,
              "source_game_id": "synthetic-source-8123",
              "api_key": "synthetic-metadata-secret", "real_name": "Private Blue",
              "prompt": {"system": "never-publish-prompt-marker"}}
    verdicts = [{
        "index": i, "pool_index": i, "text": text, "type": "reply",
        "pros": f"brief response {i}", "cons": f"generic wording {i}",
        "score": 0, "selected": i == 0, "red_flags": [], "verdict": "keep",
        "scores": {"private_metric": 5},
        "prompt": {"nested": "never-publish-prompt-marker"},
        "email": "hidden@example.invalid",
    } for i, text in enumerate(candidates)]
    payloads = [
        ("v6_generator_candidates", {"count": 4, "candidates": candidates}),
        ("v6_reviewer_analysis", {"num_candidates": 4, "selected_index": 0,
                                   "verdicts": verdicts}),
        ("v6_reviewer_selected", {"selected_text": candidates[0],
                                 "selected_word_count": 2, "selected_index": 0,
                                 "num_candidates": 4, "source": "model",
                                 "review_status": "ok", "review_reason_codes": []}),
        ("v6_delivery_sent", {"message": candidates[0], "outcome": "sent"}),
        ("v5_snapshot_read_full", {"snapshot": {"prompt": "omit-entire-event"}}),
        ("v5_behavior_advice_updated", {"advice": "keep answers brief"}),
    ]
    return [dict(common, event_type=kind, **payload) for kind, payload in payloads]


@pytest.fixture
def raw_game(tmp_path):
    base = tmp_path / "raw" / "synthetic-source-8123"
    players = [{"name": name, "real_name": f"Private {name}",
                "is_ai": name == "AI", "is_llm": name == "AI",
                "llm_config": {"model_name": "synthetic-model",
                               "async_type": "langgraph_v6"} if name == "AI" else {}}
               for name in ("AI", "Blue", "Green")]
    write_json(base, "config.json", {"players": players, "is_test": True,
                                   "api_key": "synthetic-config-secret"})
    write_text(base, "ai_player.txt", "AI\n")
    write_text(base, "player_names.txt", "AI\nBlue\nGreen\n")
    write_text(base, "remaining_players.txt", "AI\nBlue\n")
    write_text(base, "ai_elimination_info.txt",
               "AI_ELIMINATED_ROUND - survived\nTOTAL_ROUNDS - 1\n")
    write_text(base, "who_wins.txt", "AI wins!\n")
    rows = [audit({"AI": 0, "Blue": 1, "Green": 2}, "Green")]
    write_jsonl(base, "events/elimination_audit.jsonl", rows)
    write_text(base, "transcripts/chat_displayed_full.txt", transcript(rows))
    # Personal chat is authoritative delivery evidence, not reviewer selection.
    write_text(base, "messages/AI_chat.txt", "hello there\n")
    for voter, target in {"AI": "Green", "Blue": "Green", "Green": "Blue"}.items():
        write_text(base, f"votes/{voter}_vote.txt", target + "\n")
    write_text(base, "surveys/Blue_survey.txt", web_survey())
    write_text(base, "surveys/Green_survey.txt", legacy_survey())
    write_jsonl(base, "events/game_events.jsonl", event_rows())
    write_text(base, "logs/AI_generator.txt", "never-publish-raw-log-marker\n")
    write_text(base, "real_names.txt", "Private Blue\nPrivate Green\n")
    write_text(base, "dropped_players.txt", "")
    return base


def exported_events(output):
    return read_jsonl(output / "internal_outputs" / "games" / EXPORT_ID /
                      "events" / "game_events.jsonl")


def assert_manifest(output, kind, missing=()):
    manifest = read_json(output / "export_manifest.json")
    assert manifest["privacy_review"]["status"] == "pending"
    assert manifest["publication_authorized"] is False
    assert manifest["historical_cohort_member"] is False
    assert manifest["collection_kind"] == kind
    assert isinstance(manifest["missing_surveys"], list)
    assert sorted(manifest["missing_surveys"]) == sorted(missing)


@pytest.mark.parametrize("kind", ["local_game", "scripted_test"])
def test_layout_exact_core_schemas_and_source_unchanged(raw_game, tmp_path, kind):
    before = snapshot(raw_game)
    output = tmp_path / "export"
    result = export_game(raw_game, output, EXPORT_ID, kind)
    assert isinstance(result, dict)
    assert snapshot(raw_game) == before
    core = output / "games" / EXPORT_ID
    assert {p.name for p in core.iterdir()} == CORE_FILES
    assert {p.name for p in (output / "games").iterdir()} == {EXPORT_ID}
    assert (output / "internal_outputs" / "manifest.json").is_file()
    assert isinstance(read_json(output / "internal_outputs" / "manifest.json"), dict)
    assert (output / "README.md").read_text(encoding="utf-8").strip()
    assert_manifest(output, kind)
    meta = read_json(core / "metadata.json")
    assert set(meta) == META_KEYS
    assert meta == {"game_id": EXPORT_ID, "ai_pseudonym": "AI",
                    "num_human_players": 2, "total_rounds": 1,
                    "ai_eliminated": False, "ai_elimination_round": None}
    votes = read_json(core / "votes.json")
    assert len(votes) == 3
    assert all(set(row) == VOTE_KEYS for row in votes)
    assert {(v["round"], v["voter"], v["voted_for"], v["is_ai_voter"])
            for v in votes} == {(1, "AI", "Green", True), (1, "Blue", "Green", False),
                                 (1, "Green", "Blue", False)}
    for row in votes:
        assert type(row["round"]) is int
        assert type(row["is_ai_voter"]) is bool
    text = (core / "transcript.txt").read_text(encoding="utf-8")
    assert "[12:00:02] AI: hello there" in text
    assert "Vote results:" in text
    assert "→ eliminated: Green" in text
    assert "Green: 2" in text
    surveys = {row["participant"]: row for row in read_json(core / "surveys.json")}
    common = {"participant", "survey_schema", "human_likeness", "timing", "relevance"}
    assert set(surveys) == {"Blue", "Green"}
    assert set(surveys["Blue"]) == common | {
        "enjoyment", "suspicion_reason", "additional_comments"}
    assert set(surveys["Green"]) == common | {"comments"}
    assert surveys["Blue"]["survey_schema"] == "web_v1"
    assert surveys["Green"]["survey_schema"] == "legacy_terminal_v1"
    assert surveys["Blue"]["enjoyment"] == 4
    assert surveys["Green"]["comments"] == "interesting game"
    for row in surveys.values():
        for field in ("human_likeness", "timing", "relevance"):
            assert type(row[field]) is int and 1 <= row[field] <= 5
    forbidden_names = {"config.json", "real_names.txt", "AI_generator.txt"}
    assert not any(p.name in forbidden_names for p in output.rglob("*"))
    assert not any(p.name == "logs" for p in output.rglob("*"))


def test_internal_allowlists_order_false_zero_and_turn_remapping(raw_game, tmp_path):
    output = tmp_path / "export"
    export_game(raw_game, output, EXPORT_ID, "local_game")
    rows = exported_events(output)
    assert [row["event_type"] for row in rows] == [
        row["event_type"] for row in event_rows()
        if row["event_type"] in EVENT_FIELDS]
    for order, row in enumerate(rows, 1):
        assert row["game_id"] == EXPORT_ID
        assert type(row["event_order"]) is int
        assert row["event_order"] == order
        allowed = {"game_id", "event_type", "event_order", "turn_id"}
        allowed.update(EVENT_FIELDS[row["event_type"]].split())
        assert set(row) <= allowed
        assert row["turn_id"] == EXPORT_ID + "-turn001"
        assert not re.search(r"\b\d{2}:\d{2}:\d{2}\b", json.dumps(row))
    generated = next(r for r in rows if r["event_type"] == "v6_generator_candidates")
    assert generated["count"] == 4
    assert generated["candidates"] == ["hello there", "nice weather", "fair enough", "sounds good"]
    analysis = next(r for r in rows if r["event_type"] == "v6_reviewer_analysis")
    assert analysis["selected_index"] == 0
    assert len(analysis["verdicts"]) == 4
    for index, verdict in enumerate(analysis["verdicts"]):
        assert set(verdict) <= VERDICT_KEYS
        assert verdict["index"] == index
        assert verdict["score"] == 0 and type(verdict["score"]) in (int, float)
        assert verdict["selected"] is (index == 0)
        assert verdict["pros"] and verdict["cons"]
    selected = next(r for r in rows if r["event_type"] == "v6_reviewer_selected")
    delivery = next(r for r in rows if r["event_type"] == "v6_delivery_sent")
    assert selected["selected_text"] == delivery["message"] == "hello there"
    assert delivery["outcome"] == "sent"


def test_distinct_raw_turns_get_sequential_public_ids(raw_game, tmp_path):
    rows = event_rows()
    rows.append(dict(rows[0], turn_id="another-private-source-turn", turn_index=98))
    write_jsonl(raw_game, "events/game_events.jsonl", rows)
    output = tmp_path / "export"
    export_game(raw_game, output, EXPORT_ID, "local_game")
    public = exported_events(output)
    assert public[0]["turn_id"] == EXPORT_ID + "-turn001"
    assert public[-1]["turn_id"] == EXPORT_ID + "-turn002"


def test_selection_without_delivery_does_not_synthesize_an_outcome(raw_game, tmp_path):
    rows = [row for row in event_rows() if row["event_type"] != "v6_delivery_sent"]
    write_jsonl(raw_game, "events/game_events.jsonl", rows)
    output = tmp_path / "export"
    export_game(raw_game, output, EXPORT_ID, "local_game")
    public = exported_events(output)
    assert any(row["event_type"] == "v6_reviewer_selected" for row in public)
    assert not any("delivery" in row["event_type"] for row in public)
    assert not any("outcome" in row for row in public)


def test_free_text_redaction_and_private_fields_never_escape(raw_game, tmp_path):
    email = "synthetic.person@example.invalid"
    key = "sk-" + "z" * 32
    private_name = "Private Blue"
    sensitive = f"contact {email} token {key} name {private_name}"
    path = raw_game / "transcripts/chat_displayed_full.txt"
    write_text(raw_game, "transcripts/chat_displayed_full.txt",
               path.read_text(encoding="utf-8") + f"[12:00:04] Blue: {sensitive}\n")
    write_text(raw_game, "surveys/Blue_survey.txt",
               web_survey().replace("enjoyable", sensitive))
    rows = event_rows()
    rows[-1]["advice"] = sensitive
    rows[1]["verdicts"][0]["pros"] = sensitive
    write_jsonl(raw_game, "events/game_events.jsonl", rows)
    before = snapshot(raw_game)
    output = tmp_path / "export"
    export_game(raw_game, output, EXPORT_ID, "scripted_test")
    # Boolean assertions avoid echoing synthetic secret values in failure diffs.
    texts = [p.read_text(encoding="utf-8") for p in output.rglob("*") if p.is_file()]
    forbidden = [email, key, private_name, "synthetic-config-secret",
                 "synthetic-metadata-secret", "never-publish-prompt-marker",
                 "never-publish-raw-log-marker", "synthetic-source-8123"]
    assert not any(token.lower() in text.lower() for token in forbidden for text in texts), \
        "private source content escaped export"
    assert not any('"real_name"' in text or '"api_key"' in text for text in texts), \
        "private metadata field escaped export"
    core = output / "games" / EXPORT_ID
    assert "[REDACTED" in (core / "transcript.txt").read_text(encoding="utf-8")
    assert "[REDACTED" in (core / "surveys.json").read_text(encoding="utf-8")
    assert "[REDACTED" in json.dumps(exported_events(output))
    assert snapshot(raw_game) == before


@pytest.mark.parametrize("damage", [
    "wrong_tally", "missing_ballot", "extra_ballot", "self_vote", "missing_audit",
    "winner_inconsistent", "incomplete", "dropped_player", "wrong_total_rounds",
    "wrong_remaining_roster", "wrong_max_votes", "wrong_tied_candidates",
])
def test_inconsistent_or_incomplete_sources_rejected(raw_game, tmp_path, damage):
    if damage == "wrong_tally":
        write_jsonl(raw_game, "events/elimination_audit.jsonl",
                    [audit({"AI": 0, "Blue": 0, "Green": 3}, "Green")])
    elif damage == "missing_ballot":
        (raw_game / "votes/Blue_vote.txt").unlink()
    elif damage == "extra_ballot":
        write_text(raw_game, "votes/Blue_vote.txt", "Green\nGreen\n")
    elif damage == "self_vote":
        write_text(raw_game, "votes/Blue_vote.txt", "Blue\n")
        rows = [audit({"AI": 0, "Blue": 2, "Green": 1}, "Blue")]
        write_jsonl(raw_game, "events/elimination_audit.jsonl", rows)
        write_text(raw_game, "transcripts/chat_displayed_full.txt", transcript(rows))
        write_text(raw_game, "remaining_players.txt", "AI\nGreen\n")
    elif damage == "missing_audit":
        (raw_game / "events/elimination_audit.jsonl").unlink()
    elif damage == "winner_inconsistent":
        write_text(raw_game, "who_wins.txt", "Humans win!\n")
    elif damage == "incomplete":
        (raw_game / "who_wins.txt").unlink()
    elif damage == "dropped_player":
        write_text(raw_game, "dropped_players.txt", "Green\n")
    elif damage == "wrong_total_rounds":
        write_text(raw_game, "ai_elimination_info.txt",
                   "AI_ELIMINATED_ROUND - survived\nTOTAL_ROUNDS - 2\n")
    elif damage == "wrong_remaining_roster":
        write_text(raw_game, "remaining_players.txt", "AI\nGreen\n")
    else:
        rows = read_jsonl(raw_game / "events/elimination_audit.jsonl")
        rows[0]["max_votes" if damage == "wrong_max_votes" else "tied_candidates"] = (
            9 if damage == "wrong_max_votes" else ["Blue", "Green"])
        write_jsonl(raw_game, "events/elimination_audit.jsonl", rows)
    before = snapshot(raw_game)
    with pytest.raises(ExportError):
        export_game(raw_game, tmp_path / "export", EXPORT_ID, "local_game")
    assert snapshot(raw_game) == before


@pytest.mark.parametrize("slot", ["", "<ABSTAIN>"])
def test_abstention_slots_reconcile_without_becoming_ballots(raw_game, tmp_path, slot):
    write_text(raw_game, "votes/Green_vote.txt", slot + "\n")
    rows = [audit({"AI": 0, "Blue": 0, "Green": 2}, "Green")]
    write_jsonl(raw_game, "events/elimination_audit.jsonl", rows)
    write_text(raw_game, "transcripts/chat_displayed_full.txt", transcript(rows))
    output = tmp_path / "export"
    export_game(raw_game, output, EXPORT_ID, "local_game")
    votes = read_json(output / "games" / EXPORT_ID / "votes.json")
    assert len(votes) == 2
    assert {row["voter"] for row in votes} == {"AI", "Blue"}
    assert all(row["voted_for"] == "Green" for row in votes)


@pytest.mark.parametrize("empty", [False, True])
def test_missing_or_empty_survey_is_omitted_and_reported(raw_game, tmp_path, empty):
    path = raw_game / "surveys/Green_survey.txt"
    if empty:
        path.write_text("", encoding="utf-8")
    else:
        path.unlink()
    output = tmp_path / "export"
    export_game(raw_game, output, EXPORT_ID, "local_game")
    surveys = read_json(output / "games" / EXPORT_ID / "surveys.json")
    assert [row["participant"] for row in surveys] == ["Blue"]
    assert_manifest(output, "local_game", missing=["Green"])


@pytest.mark.parametrize("rating", ["0", "6", "3.5", "true", "not a rating", ""])
def test_malformed_rating_rejected(raw_game, tmp_path, rating):
    write_text(raw_game, "surveys/Blue_survey.txt", web_survey().replace(
        "similarity to human behavior - 4", f"similarity to human behavior - {rating}"))
    with pytest.raises(ExportError):
        export_game(raw_game, tmp_path / "export", EXPORT_ID, "local_game")


@pytest.mark.parametrize("bad_line", ['{"event_type":', "not-json", "[]", "null"])
def test_malformed_event_jsonl_rejected(raw_game, tmp_path, bad_line):
    path = raw_game / "events/game_events.jsonl"
    write_text(raw_game, "events/game_events.jsonl",
               path.read_text(encoding="utf-8") + bad_line + "\n")
    with pytest.raises(ExportError):
        export_game(raw_game, tmp_path / "export", EXPORT_ID, "local_game")


@pytest.mark.parametrize("directory", [False, True])
def test_existing_output_never_overwritten(raw_game, tmp_path, directory):
    output = tmp_path / "export"
    if directory:
        write_text(output, "sentinel.txt", "keep existing bytes\n")
        before = snapshot(output)
    else:
        output.write_text("keep existing bytes\n", encoding="utf-8")
        before = output.read_bytes()
    with pytest.raises(ExportError):
        export_game(raw_game, output, EXPORT_ID, "local_game")
    assert (snapshot(output) if directory else output.read_bytes()) == before


@pytest.mark.parametrize("export_id", ["", "../escape", "a/b", "a\\b", ".", "..",
                                       "C:escape", "has space", "bad\nname", "évil"])
def test_unsafe_export_id_rejected(raw_game, tmp_path, export_id):
    with pytest.raises(ExportError):
        export_game(raw_game, tmp_path / "export", export_id, "local_game")


def test_invalid_collection_kind_rejected(raw_game, tmp_path):
    with pytest.raises(ExportError):
        export_game(raw_game, tmp_path / "export", EXPORT_ID, "historical_cohort")


@pytest.mark.parametrize("relation", ["inside", "ancestor", "same"])
def test_output_cannot_overlap_source(raw_game, relation):
    output = {"inside": raw_game / "new_export", "ancestor": raw_game.parent,
              "same": raw_game}[relation]
    before = snapshot(raw_game.parent)
    with pytest.raises(ExportError):
        export_game(raw_game, output, EXPORT_ID, "local_game")
    assert snapshot(raw_game.parent) == before


def test_protected_package_games_destination_rejected(raw_game, tmp_path, monkeypatch):
    # Simulate the installed package root; never write into the real cohort tree.
    package = tmp_path / "pretend_package"
    (package / "games").mkdir(parents=True)
    monkeypatch.setattr(exporter, "__file__", str(package / "export_game.py"))
    before = snapshot(package)
    with pytest.raises(ExportError):
        export_game(raw_game, package / "games" / "new_export", EXPORT_ID, "local_game")
    assert snapshot(package) == before


def test_symlink_input_guard_without_platform_privileges(raw_game, tmp_path, monkeypatch):
    # Exercise the guard without requiring Windows symlink privileges or skips.
    original = Path.is_symlink
    suspect = raw_game / "config.json"

    def is_symlink(path):
        return path == suspect or original(path)

    monkeypatch.setattr(Path, "is_symlink", is_symlink)
    with pytest.raises(ExportError):
        export_game(raw_game, tmp_path / "export", EXPORT_ID, "local_game")


def test_two_round_random_tie_then_ai_elimination(raw_game, tmp_path):
    config = read_json(raw_game / "config.json")
    config["players"].append({"name": "Red", "real_name": "Private Red",
                              "is_ai": False, "is_llm": False, "llm_config": {}})
    write_json(raw_game, "config.json", config)
    write_text(raw_game, "player_names.txt", "AI\nBlue\nGreen\nRed\n")
    rows = [audit({"AI": 0, "Blue": 0, "Green": 2, "Red": 2}, "Green"),
            audit({"AI": 2, "Blue": 1, "Red": 0}, "AI", "12:04:00")]
    write_jsonl(raw_game, "events/elimination_audit.jsonl", rows)
    write_text(raw_game, "transcripts/chat_displayed_full.txt", transcript(rows))
    for voter, stream in {"AI": "Green\nBlue\n", "Blue": "Red\nAI\n",
                          "Green": "Red\n", "Red": "Green\nAI\n"}.items():
        write_text(raw_game, f"votes/{voter}_vote.txt", stream)
    write_text(raw_game, "remaining_players.txt", "Blue\nRed\n")
    write_text(raw_game, "ai_elimination_info.txt",
               "AI_ELIMINATED_ROUND - 2\nTOTAL_ROUNDS - 2\n")
    write_text(raw_game, "who_wins.txt", "Humans win!\n")
    write_text(raw_game, "surveys/Red_survey.txt", legacy_survey())
    before = snapshot(raw_game)
    output = tmp_path / "export"
    export_game(raw_game, output, EXPORT_ID, "scripted_test")
    meta = read_json(output / "games" / EXPORT_ID / "metadata.json")
    assert meta["num_human_players"] == 3
    assert meta["total_rounds"] == meta["ai_elimination_round"] == 2
    assert meta["ai_eliminated"] is True
    votes = read_json(output / "games" / EXPORT_ID / "votes.json")
    assert len(votes) == 7
    assert {r["voter"] for r in votes if r["round"] == 1} == {"AI", "Blue", "Green", "Red"}
    assert {r["voter"] for r in votes if r["round"] == 2} == {"AI", "Blue", "Red"}
    assert sum(r["voted_for"] == "AI" for r in votes if r["round"] == 2) == 2
    assert snapshot(raw_game) == before


def test_canonical_tally_without_separate_elimination_announcement(raw_game, tmp_path):
    rows = read_jsonl(raw_game / "events/elimination_audit.jsonl")
    write_text(raw_game, "transcripts/chat_displayed_full.txt",
               transcript(rows, include_announcement=False))
    output = tmp_path / "export"
    export_game(raw_game, output, EXPORT_ID, "local_game")
    assert "→ eliminated: Green" in (
        output / "games" / EXPORT_ID / "transcript.txt").read_text(encoding="utf-8")


def test_cli_exports_scripted_test_offline(raw_game, tmp_path):
    output = tmp_path / "cli_export"
    before = snapshot(raw_game)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    # -S excludes site initialization and SDKs; -B avoids package cache writes.
    completed = subprocess.run(
        [sys.executable, "-S", "-B", str(PACKAGE_ROOT / "export_game.py"),
         "--game-dir", str(raw_game), "--output", str(output),
         "--export-id", EXPORT_ID, "--kind", "scripted_test"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
        check=False,
    )
    assert completed.returncode == 0, "offline exporter CLI failed"
    assert_manifest(output, "scripted_test")
    assert {p.name for p in (output / "games" / EXPORT_ID).iterdir()} == CORE_FILES
    assert snapshot(raw_game) == before


def test_export_error_is_a_value_error():
    assert issubclass(ExportError, ValueError)


def test_nonfinite_discarded_diagnostic_not_copied(raw_game, tmp_path):
    rows = event_rows()
    rows.append({"event_type": "v6_scheduler_decision", "decision": "wait",
                 "send_reason": "cooldown", "private_cadence": float("inf")})
    write_jsonl(raw_game, "events/game_events.jsonl", rows)
    output = tmp_path / "export"
    export_game(raw_game, output, EXPORT_ID, "local_game")
    text = (output / "internal_outputs" / "games" / EXPORT_ID /
            "events/game_events.jsonl").read_text(encoding="utf-8")
    assert "Infinity" not in text and "private_cadence" not in text
    assert exported_events(output)[-1]["decision"] == "wait"


def test_nonfinite_retained_value_rejected(raw_game, tmp_path):
    rows = event_rows()
    rows[1]["verdicts"][0]["score"] = float("nan")
    write_jsonl(raw_game, "events/game_events.jsonl", rows)
    with pytest.raises(ExportError):
        export_game(raw_game, tmp_path / "export", EXPORT_ID, "local_game")


def test_nonanonymous_displayed_ballots_reconcile(raw_game, tmp_path):
    config = read_json(raw_game / "config.json")
    config["anonymous_voting"] = False
    write_json(raw_game, "config.json", config)
    rows = read_jsonl(raw_game / "events/elimination_audit.jsonl")
    text = transcript(rows).replace("    Blue: 1\n    Green: 2\n",
        "    AI voted for Green\n    Blue voted for Green\n    Green voted for Blue\n")
    write_text(raw_game, "transcripts/chat_displayed_full.txt", text)
    export_game(raw_game, tmp_path / "good", EXPORT_ID, "local_game")
    write_text(raw_game, "transcripts/chat_displayed_full.txt", text.replace("Blue voted for Green", "Blue voted for AI"))
    with pytest.raises(ExportError):
        export_game(raw_game, tmp_path / "bad", EXPORT_ID, "local_game")


def test_embedded_source_id_redacted(raw_game, tmp_path):
    rows = event_rows()
    rows[-1]["advice"] = "session_" + raw_game.name
    write_jsonl(raw_game, "events/game_events.jsonl", rows)
    output = tmp_path / "export"
    export_game(raw_game, output, EXPORT_ID, "local_game")
    assert raw_game.name not in json.dumps(exported_events(output))


def test_atomic_publication_never_replaces_new_destination(raw_game, tmp_path, monkeypatch):
    output = tmp_path / "export"
    original = exporter.publish_noreplace

    def race(staging, target):
        target.mkdir()
        return original(staging, target)

    monkeypatch.setattr(exporter, "publish_noreplace", race)
    with pytest.raises(ExportError):
        export_game(raw_game, output, EXPORT_ID, "local_game")
    assert output.is_dir() and not list(output.iterdir())