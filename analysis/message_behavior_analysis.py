"""Selected cached-label calculation from message_behavior_analysis.py.

No annotation API, surface-feature analyses, plots, or extra comparisons.
The ten conversation labels remain inputs to validation; only four are compared.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean

try:
    from .takeaway_evidence import ROOT, boolean
except ImportError:
    from takeaway_evidence import ROOT, boolean

LABELS = (
    "asks_question", "answers_question_or_request", "opens_new_topic",
    "accuses_or_suspects_other_player", "responds_to_ai_accusation",
    "deflects_or_evades", "brief_reaction", "greeting_or_social_ritual",
    "shares_personal_detail", "uses_slang_or_informal_abbreviation",
)
RETAINED_LABELS = (
    "answers_question_or_request", "asks_question",
    "greeting_or_social_ritual", "shares_personal_detail",
)


def load_messages(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    seen = set()
    for row in rows:
        # Export contract uses game_id (gid is accepted as the explicit short alias).
        row["game_id"] = row.get("game_id") or row["gid"]
        key = (row["game_id"], row["id"])
        if key in seen:
            raise ValueError("duplicate game/message identifier")
        seen.add(key)
        if row["role"] not in ("agent", "human") or not row["speaker_blind"]:
            raise ValueError("expected agent/human role and neutral speaker label")
        row["round"] = int(row["round"])
        if row["round"] < 1:
            raise ValueError("round must be positive")
        for label in LABELS:
            row[label] = boolean(row[label])
    return rows


def summarize(items: list[dict]):
    """Preserve original player-share -> human-within-game -> across-game math.

    Silent players have no defined message share and do not enter this estimand,
    unlike the roster-inclusive participation calculation.
    """
    players = {}
    for item in items:
        key = (item["game_id"], item["speaker_blind"])
        player = players.setdefault(key, {"game_id": item["game_id"],
                    "player": item["speaker_blind"], "role": item["role"],
                    "n_messages": 0, "labels": Counter()})
        if player["role"] != item["role"]:
            raise ValueError("role changes within game/speaker")
        player["n_messages"] += 1
        for label in RETAINED_LABELS:
            player["labels"][label] += int(item[label])
    rows = []
    for player in players.values():
        n_messages = player["n_messages"]
        rows.append({"game_id": player["game_id"], "player": player["player"],
                     "role": player["role"], "n_messages": n_messages,
                     **{f"{label}_share": player["labels"][label] / n_messages
                        for label in RETAINED_LABELS}})
    game_ids = sorted({row["game_id"] for row in rows})
    paired = {}
    for label in RETAINED_LABELS:
        metric = f"{label}_share"
        triples = []
        for gid in game_ids:
            agents = [row for row in rows if row["game_id"] == gid and row["role"] == "agent"]
            humans = [row for row in rows if row["game_id"] == gid and row["role"] == "human"]
            if len(agents) != 1 or not humans:
                raise ValueError(f"{gid}: require one speaking agent and at least one speaking human")
            agent = agents[0]
            human_mean = mean(row[metric] for row in humans)
            triples.append((agent[metric], human_mean, agent[metric] - human_mean))
        paired[metric] = {"agent": mean(triple[0] for triple in triples),
                          "humans": mean(triple[1] for triple in triples),
                          "difference": mean(triple[2] for triple in triples),
                          "denominator": "player-positive messages / own public messages; "
                                         "humans averaged within game, then games equally"}
    return {"games": len(game_ids), "messages": len(items),
            "player_instances": len(rows), "paired": paired}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "analysis/inputs/message_behavior_message_level.csv")
    args = parser.parse_args()
    rows = load_messages(args.input)
    if len({r["game_id"] for r in rows}) != 50:
        raise ValueError("paper analysis requires exactly 50 games")
    print(json.dumps(summarize(rows), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()