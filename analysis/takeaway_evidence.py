"""Selected paper calculations from takeaway_evidence.py; public-data readers only.

No API calls, private-tree fallback, plots, or unreported robustness analyses.
See README.md for the safe auxiliary-input contract.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev, stdev

ROOT = Path(__file__).resolve().parents[1]
MSG_RE = re.compile(r"^\[(\d\d):(\d\d):(\d\d)\]\s+([^:]+):\s*(.*)$")


def secs(stamp: str) -> int:
    h, m, s = (int(x) for x in stamp.strip().split(":"))
    return 3600 * h + 60 * m + s


def boolean(value):
    if isinstance(value, bool):
        return value
    if str(value).strip().lower() in ("true", "1"):
        return True
    if str(value).strip().lower() in ("false", "0"):
        return False
    raise ValueError(f"Not a Boolean: {value!r}")


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_cohort(path: Path, expected_games=50):
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [r["game_id"] for r in rows]
    if len(ids) != expected_games or len(set(ids)) != expected_games:
        raise ValueError(f"Expected {expected_games} unique games; found {len(ids)} rows")
    # Preserve CSV order: the seeded whole-game bootstrap depends on it.
    return rows


def read_public_game(base: Path):
    """Read the submitted four-file format without treating manager text as chat.

    Displayed text is never the authority for original word counts or round rosters.
    """
    meta = json.loads((base / "metadata.json").read_text(encoding="utf-8"))
    votes = json.loads((base / "votes.json").read_text(encoding="utf-8"))
    records = json.loads((base / "surveys.json").read_text(encoding="utf-8"))
    messages = []
    for line in (base / "transcript.txt").read_text(encoding="utf-8").splitlines():
        match = MSG_RE.match(line)
        if match:
            h, mi, s, player, text = match.groups()
            if player.strip() != "Game-Manager" and text.strip():
                messages.append({"t": 3600 * int(h) + 60 * int(mi) + int(s),
                                 "player": player.strip(), "text": text.strip()})
    return {"metadata": meta, "votes": votes, "surveys": records,
            "display_messages": messages}


def load(cohort, root: Path, inputs: Path):
    """Adapt safe exports to the existing g[row/ai/audit/ends/msgs/surveys] shape.

    rounds.jsonl is mandatory: votes alone cannot recover silent rosters, ties,
    actual eliminations, or original audit boundaries. Counts are already the
    original deduplicated public-player messages; never deduplicate by word count.
    """
    audits, counts = defaultdict(list), defaultdict(list)
    ids = {r["game_id"] for r in cohort}
    for a in read_jsonl(inputs / "rounds.jsonl"):
        if a["game_id"] not in ids:
            raise ValueError("rounds.jsonl contains a game outside games.csv")
        audits[a["game_id"]].append(a)
    for m in read_jsonl(inputs / "message_counts.jsonl"):
        if m["game_id"] not in ids:
            raise ValueError("message_counts.jsonl contains a game outside games.csv")
        counts[m["game_id"]].append(m)
    games = {}
    for row in cohort:
        gid, ai = row["game_id"], row["ai_name"].strip()
        public = read_public_game(root / gid)
        meta = public["metadata"]
        if (meta["game_id"] != gid or meta["ai_pseudonym"] != ai
                or int(meta["num_human_players"]) != int(row["initial_humans"])
                or int(row["initial_total"]) != int(row["initial_humans"]) + 1
                or int(meta["total_rounds"]) != int(row["rounds_played"])
                or boolean(meta["ai_eliminated"]) == boolean(row["ai_won"])):
            raise ValueError(f"{gid}: metadata/CSV mismatch")
        audit = sorted(audits[gid], key=lambda a: int(a["round"]))
        if [int(a["round"]) for a in audit] != list(range(1, int(row["rounds_played"]) + 1)):
            raise ValueError(f"{gid}: missing/duplicate audit rounds")
        ends = [int(a.get("cutoff_seconds", secs(a["timestamp"]))) for a in audit]
        # Historical scripts used time-of-day seconds, including one game that
        # crossed midnight. Preserve those exact windows for paper reproduction;
        # do not silently repair clock rollover and change the reported estimand.
        if any(t < 0 or t >= 86400 for t in ends):
            raise ValueError(f"{gid}: audit timestamps must be valid time-of-day seconds")
        active = set(audit[0]["active_players"])
        if len(active) != int(row["initial_total"]):
            raise ValueError(f"{gid}: incomplete initial roster (include silent humans)")
        for a in audit:
            roster = a["active_players"]
            if (len(roster) != len(set(roster)) or set(roster) != active
                    or set(a["tally"]) != active or ai not in active
                    or a["voted_out"] not in active):
                raise ValueError(f"{gid}: inconsistent active roster/tally/elimination")
            if any(type(n) is not int or n < 0 for n in a["tally"].values()):
                raise ValueError(f"{gid}: invalid tally")
            ballots = [v for v in public["votes"] if int(v["round"]) == int(a["round"])]
            if (len({v["voter"] for v in ballots}) != len(ballots)
                    or any(v["voter"] not in active or v["voted_for"] not in active
                           or v["voter"] == v["voted_for"] for v in ballots)
                    or any(boolean(v["is_ai_voter"]) != (v["voter"] == ai) for v in ballots)
                    or sum(v["voter"] == ai for v in ballots) != 1):
                raise ValueError(f"{gid}: expected one non-self AI vote, valid unique human votes")
            tally = Counter(v["voted_for"] for v in ballots)
            if any(tally[p] != a["tally"][p] for p in active):
                raise ValueError(f"{gid}: public ballots disagree with audit tally")
            active = active - {a["voted_out"]}
        eliminated = audit[-1]["voted_out"] == ai
        survived = len(audit) - int(eliminated)
        if (survived != int(row["eliminations_survived"])
                or eliminated != boolean(meta["ai_eliminated"])
                or (eliminated and (int(row["ai_eliminated_round"]) != len(audit)
                                    or int(meta["ai_elimination_round"]) != len(audit)))
                or (not eliminated and len(active) != 2)):
            raise ValueError(f"{gid}: outcome/elimination-depth mismatch")
        first = [v for v in public["votes"] if int(v["round"]) == 1 and v["voter"] != ai]
        if (len(first) != int(row["first_round_ballots"])
                or sum(v["voted_for"] == ai for v in first) != int(row["first_round_votes_for_ai"])
                or abs(float(row["first_round_chance"]) - 1 / int(row["initial_humans"])) > 1e-12):
            raise ValueError(f"{gid}: first-round fields disagree with public ballots/roster")
        msgs = []
        for m in counts[gid]:
            ri, t, words = int(m["round"]), int(m["timestamp_seconds"]), int(m["word_count"])
            if (not 1 <= ri <= len(audit) or words < 1 or t < 0
                    or m["player"] not in audit[ri - 1]["active_players"]
                    or t > ends[ri - 1] or (ri > 1 and t <= ends[ri - 2])):
                raise ValueError(f"{gid}: invalid original message-count row")
            msgs.append({"round": ri, "t": t, "player": m["player"], "words": words})
        if len(msgs) != len(public["display_messages"]):
            raise ValueError(f"{gid}: original counts/displayed player-message count mismatch")
        games[gid] = {"row": row, "ai": ai, "audit": audit, "ends": ends,
                      "msgs": msgs, "surveys": public["surveys"]}
    return games


def boot_mean(xs, n=10000, seed=42):
    if not xs:
        return (None, None)
    rng = random.Random(seed)
    k = len(xs)
    vals = sorted(sum(xs[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return vals[int(0.025 * n)], vals[min(n - 1, int(0.975 * n))]


def identification(games):
    per = []
    for gid, g in games.items():
        r = g["row"]
        per.append({"game_id": gid, "ballots": int(r["first_round_ballots"]),
                    "votes_for_ai": int(r["first_round_votes_for_ai"]),
                    "chance": float(r["first_round_chance"])})
    votes = sum(p["votes_for_ai"] for p in per)
    ballots = sum(p["ballots"] for p in per)
    expected = sum(p["ballots"] * p["chance"] for p in per)
    return {"votes_for_ai": votes, "human_ballots": ballots,
            "observed_rate": votes / ballots, "expected_votes": expected,
            "expected_rate": expected / ballots}


def stage_breakdown(games):
    """Original tally calculation; later human stages are survival-conditional.

    sum(tally)-1 removes the single AI ballot. AI never self-votes, so tally[AI]
    counts human votes only. Report terminal and nonterminal rounds separately.
    """
    stages = {}
    for gid, g in games.items():
        ai = g["ai"]
        for a in g["audit"]:
            tally = a["tally"]
            active_players = len(tally)
            active_humans = active_players - 1
            ballots = sum(tally.values()) - 1
            if ballots <= 0:
                continue
            terminal = a.get("voted_out") == ai
            d = stages.setdefault(active_humans, {
                "active_humans": active_humans, "rounds": 0, "ballots": 0, "votes_for_ai": 0,
                "expected": 0.0, "terminal_rounds": 0,
                "nonterminal_rounds": 0, "nonterminal_ballots": 0,
                "nonterminal_votes_for_ai": 0, "nonterminal_expected": 0.0})
            chance = 1 / active_humans
            d["rounds"] += 1
            d["ballots"] += ballots
            d["votes_for_ai"] += tally.get(ai, 0)
            d["expected"] += ballots * chance
            if terminal:
                d["terminal_rounds"] += 1
            else:
                d["nonterminal_rounds"] += 1
                d["nonterminal_ballots"] += ballots
                d["nonterminal_votes_for_ai"] += tally.get(ai, 0)
                d["nonterminal_expected"] += ballots * chance
    out = []
    for h in sorted(stages, reverse=True):
        d = stages[h]
        row = dict(d)
        row["observed_rate"] = d["votes_for_ai"] / d["ballots"]
        row["chance_rate"] = 1 / h
        row["excess_pp"] = 100 * (d["votes_for_ai"] - d["expected"]) / d["ballots"]
        if d["nonterminal_ballots"]:
            row["nonterminal_observed_rate"] = d["nonterminal_votes_for_ai"] / d["nonterminal_ballots"]
            row["nonterminal_excess_pp"] = 100 * (
                d["nonterminal_votes_for_ai"] - d["nonterminal_expected"]) / d["nonterminal_ballots"]
        else:
            row["nonterminal_observed_rate"] = None
            row["nonterminal_excess_pp"] = None
        out.append(row)
    return {"by_stage": out,
            "totals": {k: sum(r[k] for r in out) for k in ("rounds", "ballots", "votes_for_ai")},
            "note": "Later human stages are conditional on prior AI survival; neither inclusion "
                    "nor exclusion of terminal rounds removes the stopping-rule selection."}


def survival(games):
    rows = [g["row"] for g in games.values()]
    ladder = []
    # Exact selected original ladder calculation, including eligible-game filtering,
    # per-game n/T reference, bootstrap seed, and discrete percentile indices.
    for n in (5, 4, 3, 2):
        reached, nulls = [], []
        for r in rows:
            T = int(r["initial_total"])
            if n >= T:
                continue
            reached.append(1 if int(r["eliminations_survived"]) >= T - n else 0)
            nulls.append(n / T)
        if not reached:
            continue
        obs = sum(reached) / len(reached)
        ref = sum(nulls) / len(nulls)
        diffs = [a - b for a, b in zip(reached, nulls)]
        excess_lo, excess_hi = boot_mean(diffs)
        olo, ohi = boot_mean(reached)
        ladder.append({"players_remaining": n, "eligible_games": len(reached),
                       "reached": sum(reached), "observed": obs, "reference": ref,
                       "observed_ci95": [olo, ohi],
                       "excess_pp": 100 * (obs - ref),
                       "excess_ci95_pp": [100 * excess_lo, 100 * excess_hi]})
    return {"final_two_n": sum(boolean(r["ai_won"]) for r in rows),
            "final_two_rate": sum(boolean(r["ai_won"]) for r in rows) / len(rows),
            "survival_ladder": ladder}


def participation(games):
    player_rounds = []
    pooled_words = {"agent": [], "human": []}
    for gid, g in games.items():
        ai, ends = g["ai"], g["ends"]
        for i, a in enumerate(g["audit"]):
            hi = ends[i]
            lo = None if i == 0 else ends[i - 1]
            active = list(a["tally"].keys())
            window = [m for m in g["msgs"]
                      if m["player"] in active and m["t"] <= hi and (lo is None or m["t"] > lo)]
            counts = {p: 0 for p in active}
            for m in window:
                counts[m["player"]] += 1
                pooled_words["agent" if m["player"] == ai else "human"].append(m["words"])
            for p in active:
                kind = "agent" if p == ai else "human"
                player_rounds.append({"game_id": gid, "round": i + 1, "player": p,
                                      "kind": kind, "n_messages": counts[p]})

    def side(kind):
        xs = [r for r in player_rounds if r["kind"] == kind]
        msgs = [r["n_messages"] for r in xs]
        return {"active_player_rounds": len(xs),
                "messages": sum(msgs),
                "mean_messages_per_active_round": mean(msgs),
                "mean_words_per_message": mean(pooled_words[kind]) if pooled_words[kind] else None}

    return {"agent": side("agent"), "human": side("human")}


def surveys(games):
    allrec = [s for g in games.values() for s in g["surveys"]]
    out = {"n_responses": len(allrec), "sd_convention": "sample (n-1)",
           "schema_counts": dict(Counter(s["survey_schema"] for s in allrec))}
    for key in ("human_likeness", "timing", "relevance", "enjoyment"):
        vals = [s[key] for s in allrec if s.get(key) is not None]
        if any(type(v) is not int or not 1 <= v <= 5 for v in vals):
            raise ValueError(f"{key}: ratings must be integers 1..5")
        counts = Counter(vals)
        out[key] = {"n": len(vals), "mean": mean(vals) if vals else None,
                    "sd": stdev(vals) if len(vals) > 1 else None,
                    "distribution": {str(k): counts[k] for k in range(1, 6)}}
    # No free-text output. Both SD conventions are available to a local audit,
    # but public output explicitly reports the conventional sample SD, not a fit.
    return out


def survey_sd_audit(games):
    """Local audit helper, not emitted by main; never chooses a convention by fit."""
    records = [s for g in games.values() for s in g["surveys"]]
    out = {}
    for key in ("human_likeness", "timing", "relevance", "enjoyment"):
        vals = [s[key] for s in records if s.get(key) is not None]
        out[key] = {"sample_sd": stdev(vals) if len(vals) > 1 else None,
                    "population_sd": pstdev(vals) if vals else None}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=ROOT / "analysis/inputs")
    parser.add_argument("--games", type=Path, default=ROOT / "games")
    args = parser.parse_args()
    games = load(read_cohort(args.inputs / "games.csv"), args.games, args.inputs)
    result = {"survival": survival(games), "identification": identification(games),
              "human_stage_votes": stage_breakdown(games),
              "participation": participation(games), "surveys": surveys(games)}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()