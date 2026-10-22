# Paper analyses (offline)

These are selected calculations from the original scripts with narrow readers
for the submitted public dataset. Python standard library only. No network calls,
API keys, private-corpus discovery, plots, paid reruns, or new analysis framework.
Run from the repository root; paths also default relative to the scripts, not the
current working directory. Each command prints its computed results as JSON:

```console
python analysis/takeaway_evidence.py
python analysis/external_stage_judge.py
python analysis/message_behavior_analysis.py
python analysis/analyze_message_behavior_interannotator.py
```

All paper numerical summaries use the frozen historical decisions and outputs;
no paid judge or annotation reruns are planned. The agent has Scheduler, Generator
and Reviewer plus six paper workers: Internet Context, Round Summarizer, Persona
Monitor, Social State Tracker, Behavior Adapter and Repetition Monitor. The Reviewer
selects a generated candidate verbatim, without rewriting. Recorded study diagnostic
evaluation used Gemini 2.5 Pro (`gemini-2.5-pro`) with feedback disabled; its 1,558
records are not part of the public dataset. The cached judgments remain frozen.
A new live game is not historical-analysis reproduction.

## Inputs: four-file public core plus safe calculation exports

Each of the 50 game directories retains the submitted four-file layout. These
are not runnable game directories. After manual review of all transcripts and
survey files, the public core has four additional redacted spans across three
files (197/200 game files unchanged); no rows, ratings, ballots or outcomes changed.
See the [public redaction record](https://github.com/hoshenlugasi/Beyond-the-Turing-Test/blob/main/data/public_redactions.json) for categories and
locations, without the removed strings. The public core is not wholly identical
to the submitted baseline. See [release notes](https://github.com/hoshenlugasi/Beyond-the-Turing-Test/blob/main/guides/RELEASE_NOTES.md) for the
current GitHub/Hugging Face publication status.

- `metadata.json`: `game_id`, `ai_pseudonym`, `num_human_players`, `total_rounds`,
  `ai_eliminated` (Boolean), `ai_elimination_round` (integer or null).
- `transcript.txt`: submitted readable transcript, including manager notices.
  Only timestamped non-manager speaker messages are counted by the reader.
- `votes.json`: array of `{round, voter, voted_for, is_ai_voter}`. Non-abstaining
  ballots only; exactly one non-self AI ballot per played round.
- `surveys.json`: array of participant responses with `survey_schema` and available
  integer 1-5 ratings `human_likeness`, `timing`, `relevance`, `enjoyment`.
  The five terminal-format surveys lack enjoyment. Missing ratings are not zeros.

The four files alone do **not** recover silent active players, original audit
boundaries, elimination tie resolution, or pre-redaction word counts. The following
safe exports are required under `analysis/inputs/`; no private-tree fallback exists:

### Outcomes, active rosters and participation

- `games.csv`: exactly 50 rows in **the original cohort CSV order**, with columns
  `game_id,initial_humans,initial_total,ai_name,rounds_played,ai_eliminated_round,ai_won,first_round_ballots,first_round_votes_for_ai,first_round_chance,eliminations_survived`.
  IDs match public directory names; AI names match public pseudonyms. Boolean
  values accept `True`/`False` or `1`/`0`. `ai_eliminated_round` is blank for wins.
  Preserve row order: the exact seeded bootstrap draws depend on it.
- `rounds.jsonl`: one original-schema safe audit per played stage:
  `{game_id, round, timestamp: "HH:MM:SS", tally: {pseudonym: integer}, voted_out: pseudonym, active_players: [pseudonym, ...], cutoff_seconds: integer}`.
  Round numbers are 1-based. Include **all** active players, including zero-vote
  and silent players, as tally keys. The explicit `cutoff_seconds` is preferred;
  if omitted, `timestamp` is converted to seconds. All times must share the same
  within-game clock as the message counts. Exports must preserve the original
  audit membership/windows, not substitute a later or earlier cutoff.
- `message_counts.jsonl`: one row per original deduplicated nonempty public
  player message: `{game_id, round, player, word_count, timestamp_seconds}`.
  `word_count` is the original `len(text.split())` before public redaction.
  There is **no original text** in this export. Preserve messages, including two
  different messages having the same speaker/time/word count: those are not
  duplicates. These counts, **not displayed redacted text**, reproduce original
  participation words. The submitted transcript's message count is cross-checked.

The adapter verifies rosters, actual vote tallies, first-round fields and outcomes
against the four-file data. Round windows preserve the original `(previous_end,
end]` membership (first round has no lower bound). Zero-message player-rounds
remain in the denominator. Means are total messages / active player-rounds and
pooled original words / public messages separately for agent and humans.

**Historical midnight limitation:** one game crosses midnight. The original
time-of-day window behavior is preserved, including its original counts and
memberships, for reproduction. This is not a corrected chronological-window
analysis; do not silently repair it or describe the windows as unaffected by
midnight rollover. See [REPRODUCIBILITY_NOTES.md](REPRODUCIBILITY_NOTES.md).

`survival()` retains the original ladder for 5, 4, 3, 2 players remaining. A game
is eligible only if its starting total is greater than the target; observed
survival uses `eliminations_survived >= initial_total - target`; the per-game
identity-blind reference is `target / initial_total`. `boot_mean()` is the original
10,000 whole-game resamples with `random.Random(42)` and original discrete
percentile indices. The final-two difference CI should round to −17.3,+7.2
percentage points; it is not a binomial interval or a fitted constant.

First-round human identification sums 51 AI votes / 227 cast human ballots. The
reference is ballot-weighted `sum(ballots * first_round_chance) / sum(ballots)`,
not a single six-player chance rate. It should round to 21.4%.

Human stage ballots use `sum(tally.values()) - 1`, removing the single AI ballot;
because the AI does not self-vote, its tally contains only human votes. Later
stages are conditional on AI survival, and the original terminal/nonterminal split
is retained. Neither grouping removes the stopping-rule selection; do not pool
stages as an unconditional detection estimate.

### Cached external stage judges

`external_stage_judges/claude/ledger.jsonl` and
`external_stage_judges/gpt5_minimal/ledger.jsonl` contain **one final record per
item**, each with `{item_id, game_id, round_index, active_players, correct, valid,
status, suspected_player, correct_label, reasoning_effort}`. `active_players` is
the integer count, not a roster list. `correct` and `valid` are Booleans;
`status == "ok"` iff valid. Neutral labels must agree with correctness. Failed
attempts/retry envelopes are not additional items. Each ledger covers the same
135 stages. No confidence or reasoning text is needed for these summaries.

The exact original `prompt_for_stage()` is retained. Inputs used neutral labels
fixed within game, removed timestamps and manager/vote/outcome metadata, retained
visible round boundaries and cumulative prior public discussion, and restricted
candidates to the current active roster. Cached original judgments are the
authority; this script does not recreate them from privacy-redacted text.

`build_payload()` records the original request settings for the two paper models
without sending requests: Claude Sonnet 4.5 uses temperature 0, JSON-object format,
seed 42 and 1,024 max tokens; GPT-5 omits temperature, uses JSON-object format,
seed 42 and 8,192 max completion tokens, normally with minimal reasoning effort.
One retained GPT result used a low-effort retry; preserve that row's `low` value.
The helper does not default GPT effort to minimal: it accepts the recorded effort
explicitly. Wilson intervals use the original helper (`z=1.96`). No pooled
accuracy is produced, because later stages condition on survival.

### Message behavior and raw agreement

- `message_behavior_message_level.csv`: `game_id` (or `gid`), `id`, `round`,
  `speaker_blind`, `role` (`agent`/`human`), and all ten Boolean conversation
  columns listed in `LABELS` in `message_behavior_analysis.py`. No text or surface
  evidence columns. IDs identify messages within game; speaker keys are
  **(game_id, speaker_blind)**, never the blind label across games.
- `validation_annotations.csv`: exactly 1,000 rows, with columns
  `id,label,human_1,human_2,model`: the same 100 item IDs × ten labels, each with
  three raw Boolean decisions. This is 1,000 decisions **per rater**, not 1,000
  total across raters. Original validation sampling was balanced 50 agent / 50
  human across five relative positions. The minimal decision CSV cannot verify
  that sampling design; the exporting manifest must document it.

The ten labels are asks question, answers question/request, opens new topic,
accuses/suspects another, responds to accusation, deflects/evades, brief reaction,
greeting/social ritual, personal detail, and slang/informal abbreviation.
Validation reports raw binary Cohen kappa for human1-human2, model-human1 and
model-human2. Retention requires **all three defined point estimates >= 0.6**.
No recoding, composites, OR labels, or bootstrap kappa intervals. Constant-marginal
kappa is undefined, not perfect agreement. All ten categories remain visible.

Only four retained categories are compared: answers question/request, asks
question, greeting/social ritual, and personal detail. The original calculation
first divides each speaking player's positives by their own messages, then
averages humans within game, then averages games equally. Silent players have
undefined message shares and are excluded here, unlike participation. Do not
replace this calculation with pooled message proportions.

**Supplied annotation provenance:** the original model annotation prompt had 13
independent labels (these ten plus three surface labels), with full prior public
history and no future context. The exact annotation prefix/compact protocol is in
[inputs/message_behavior_prompt.txt](inputs/message_behavior_prompt.txt), with
[inputs/annotation_settings.json](inputs/annotation_settings.json), the frozen
100-item [inputs/validation_manifest.json](inputs/validation_manifest.json) and
[inputs/PROVENANCE.md](inputs/PROVENANCE.md). These supply the recorded provenance;
they do not promise byte-identical reannotation from redacted display text.
The release contains no optional paid API-rerun code.

## Surveys and SD convention

All available ratings in the public surveys are used: 229 responses for likeness,
timing and relevance, 224 for enjoyment. Public output declares conventional
**sample SD (n−1)** and each 1-5 distribution. The paper's sample SDs are
0.94, 1.08, 0.90 and 0.78, respectively. The reader handles both survey schemas
and uses sample SD. `survey_sd_audit(games)` retains sample and population SD for audit
and is not emitted by the entry point. No survey free text is emitted.

[inputs/demographics_summary.json](inputs/demographics_summary.json) supplies
paper-reported aggregates for 226 records. It cannot reproduce individual linkage
or deduplication. The 234 starting human seats are not a verified count of 234
distinct participants.

## Four public access views

The current release includes four core files per game and four derived tables:
50 games, 2,954 messages (2,414 player / 540 system), 618 votes and 229 surveys.
See the [dataset guide](https://github.com/hoshenlugasi/Beyond-the-Turing-Test/blob/main/data/README.md)
and [dictionary](https://github.com/hoshenlugasi/Beyond-the-Turing-Test/blob/main/data/DATA_DICTIONARY.md).
Historical component event traces are not distributed. The four calculations
do not read them; frozen numeric inputs, judgments, annotations and reviewed
paper examples remain unchanged. Local runtime diagnostics and manual exports
are separate from the historical public dataset and remain privacy-pending.

## Reproduction scope and missing inputs

All required public inputs must be present for successful reproduction.
No new model-backed game or scientific-quality validation is claimed.
Linux/macOS and remote Hub behavior are not certified here. Offline analysis
needs no provider key; model access is conditional as described in
[../README.md](../README.md).

This GitHub release includes the refreshed package. Author permission is conditional on not identifying
individuals; review is not anonymity proof or independent consent/IRB certification.
See the [release notes](https://github.com/hoshenlugasi/Beyond-the-Turing-Test/blob/main/guides/RELEASE_NOTES.md) for scope.