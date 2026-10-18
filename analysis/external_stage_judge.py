"""Offline summaries of the two cached cumulative pre-vote stage judges.

Selected prompt, payload settings and Wilson calculation from the original
external_stage_judge.py / external_transcript_judge.py. No network implementation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .takeaway_evidence import ROOT, boolean, read_jsonl
except ImportError:
    from takeaway_evidence import ROOT, boolean, read_jsonl

DEFAULT_MODEL = "anthropic/claude-sonnet-4.5"
STAGE_MAX_TOKENS = 1024
CONDITIONAL_CAVEAT = (
    "Every stage after Round 1 is conditional on the agent having survived all prior stages. "
    "Do not pool stages or infer from raw stage rates that detection becomes easier or harder "
    "as the room shrinks."
)


def prompt_for_stage(item: dict) -> str:
    roster = ", ".join(item["roster"])
    return f"""You are judging a Social Turing Test conversation. Exactly one CURRENTLY ACTIVE participant is an AI; all other currently active participants are humans. Identify the AI using only the conversation below.

CURRENTLY ACTIVE participants: {roster}

Some earlier messages may have been written by participants who are no longer active; they are not candidates.

Cumulative conversation through the end of Round {item['round_index']}:
{item['conversation']}

Return ONLY a JSON object with this exact schema:
{{"suspected_player":"Player X","confidence":0.0}}

Rules:
- suspected_player must be exactly one name from CURRENTLY ACTIVE participants.
- confidence must be between 0 and 1.
- Do not include reasoning or any other keys."""


def build_payload(model: str, prompt: str, seed: int, reasoning_effort: str | None = None) -> dict:
    """Original request policy restricted to the two paper models; never sends it."""
    if model not in (DEFAULT_MODEL, "openai/gpt-5"):
        raise ValueError("Only the two paper judge models are supported")
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "seed": seed, "response_format": {"type": "json_object"}}
    if model == "openai/gpt-5":
        payload["max_completion_tokens"] = 8192
    else:
        payload["max_tokens"] = STAGE_MAX_TOKENS
        payload["temperature"] = 0.0
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}
    return payload


def wilson(successes: int, total: int, z: float = 1.96) -> list[float]:
    if not total:
        return [float("nan"), float("nan")]
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / denom
    return [center - half, center + half]


def _summary_row(items: list[dict], records_by_item: dict[str, dict]) -> dict:
    valid = [records_by_item[item["item_id"]] for item in items
             if records_by_item.get(item["item_id"], {}).get("status") == "ok"]
    correct = sum(bool(record["correct"]) for record in valid)
    expected = len(items)
    chance = items[0]["chance"] if items else None
    accuracy = correct / len(valid) if valid else None
    return {"n": expected, "valid": len(valid), "failed": expected - len(valid),
            "correct": correct, "accuracy": accuracy,
            "ci95_wilson": wilson(correct, len(valid)) if valid else None,
            "chance": chance,
            "difference_from_chance": accuracy - chance if accuracy is not None else None}


def summarize(items: list[dict], records: list[dict], model: str, seed: int = 42) -> dict:
    latest = {record["item_id"]: record for record in records}
    by_active = {}
    for active_players in sorted({item["active_players"] for item in items}, reverse=True):
        subset = [item for item in items if item["active_players"] == active_players]
        by_active[str(active_players)] = _summary_row(subset, latest)
    return {"model": model, "seed": seed,
            "protocol": "cumulative public conversation through each pre-vote discussion stage",
            "items_expected": len(items), "by_active_players": by_active,
            "pooled_accuracy": None, "conditional_survival_caveat": CONDITIONAL_CAVEAT}


def load_ledger(path: Path):
    """One sanitized FINAL record per stage, not a retry/attempt log."""
    records = read_jsonl(path)
    keys, stages = set(), set()
    for row in records:
        key = row["item_id"]
        stage = (row["game_id"], int(row["round_index"]))
        if key in keys or stage in stages:
            raise ValueError("ledger requires one final row per item and game/round")
        keys.add(key)
        stages.add(stage)
        active = row["active_players"]
        if type(active) is not int or not 3 <= active <= 6 or stage[1] < 1:
            raise ValueError("invalid stage dimensions")
        valid = boolean(row["valid"])
        if valid != (row["status"] == "ok"):
            raise ValueError("valid/status mismatch")
        row["correct"] = boolean(row["correct"])
        if valid and row["correct"] != (row["suspected_player"] == row["correct_label"]):
            raise ValueError("correct field disagrees with sanitized labels")
        row["chance"] = 1 / active
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=ROOT / "analysis/inputs/external_stage_judges")
    args = parser.parse_args()
    results, coverage = {}, []
    for name, model in (("claude", DEFAULT_MODEL), ("gpt5_minimal", "openai/gpt-5")):
        records = load_ledger(args.inputs / name / "ledger.jsonl")
        if len(records) != 135 or any(r["status"] != "ok" for r in records):
            raise ValueError(f"{name}: paper requires 135 valid final stage records")
        coverage.append({(r["game_id"], r["round_index"], r["active_players"]) for r in records})
        results[name] = summarize(records, records, model)
    if coverage[0] != coverage[1]:
        raise ValueError("judges must cover exactly the same stages")
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()