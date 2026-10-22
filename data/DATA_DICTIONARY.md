# Data dictionary

Both data packages use this dictionary. UTF-8 JSONL has one object per line;
core ballot/survey files are JSON arrays. Public IDs and pseudonyms support
within-game joins only. No raw-to-public mapping is released.

## Authority and missingness

The four core files are canonical; tables are access views. The companion code
package's `social_turing_test/core/dataset_tables.py` defines their projection.
Historical component event traces and their coverage fields are not included.

Notation below: `s` string, `i` integer, `b` Boolean. Nullable fields distinguish
missing or unrecorded values from zero, false and empty submitted text. Integer
fields do not accept Booleans. Masked text is not a missing-value null.

## Core records and four access tables

### Games (50 rows)

[Metadata](games/game_001/metadata.json) contains the first six fields below;
[tables/games.jsonl](tables/games.jsonl) adds the last two.

| Field | Type | Meaning |
|---|---|---|
| `game_id` | s | Public directory identifier |
| `ai_pseudonym` | s | Agent's in-game name |
| `num_human_players` | i | Starting human seats, four or five in this cohort |
| `total_rounds` | i | Played discussion/vote rounds |
| `ai_eliminated` | b | Whether the AI was eliminated |
| `ai_elimination_round` | i or null | Elimination round; null for AI wins |
| `human_start_count` | i | Access-view alias of starting human count |
| `path` | s | Public relative game directory |

### Messages (2,954 rows)

[tables/messages.jsonl](tables/messages.jsonl) parses occurrences in the
[transcript](games/game_001/transcript.txt); it does not deduplicate or sort clocks.
Multiline continuations belong to the preceding message, with LF separators;
headers before the first timestamped record are excluded.

| Field | Type | Meaning |
|---|---|---|
| `game_id` | s | Within-game join key |
| `message_order` | i | One-based transcript occurrence order, including notices |
| `recorded_time` | s | Original HH:MM:SS token, without date/timezone inference |
| `speaker` | s | Pseudonym or Game-Manager |
| `speaker_role` | s | `agent`, `human` or `system` |
| `text` | s | Public message text, possibly multiline/redacted |
| `elapsed_seconds` | null | No shared chat/event clock has been established |
| `time_basis` | s | `unknown` for these message rows |

There are 2,414 player messages (519 agent, 1,895 human) and 540 system notices.
Filter `speaker_role != "system"` for player-message denominators. Event text,
generated candidates and reviewer selections are not additional chat messages.

### Votes (618 rows)

[Core ballots](games/game_001/votes.json) contain `round` (i, one-based), `voter`
(s), `voted_for` (s) and `is_ai_voter` (b). [tables/votes.jsonl](tables/votes.jsonl)
adds `game_id` (s). These are cast ballots, not all potential voting opportunities;
explicit abstentions are omitted, not imputed. AI ballots affect elimination but
must be excluded from human-only identification. Silent rostered humans can vote.

### Surveys (229 rows)

[Core surveys](games/game_001/surveys.json) contain available fields below;
[tables/surveys.jsonl](tables/surveys.jsonl) adds `game_id` and emits every listed
survey field, using null for absent ones.

| Field | Type | Meaning |
|---|---|---|
| `game_id` | s | Table-only game join key |
| `participant` | s | Respondent's within-game pseudonym |
| `survey_schema` | s | `legacy_terminal_v1` or `web_v1` |
| `human_likeness` | i | 1-5 likeness rating, 229 available |
| `timing` | i | 1-5 timing/pacing rating, 229 available |
| `relevance` | i | 1-5 relevance/coherence rating, 229 available |
| `enjoyment` | i or null | 1-5 enjoyment rating, 224 available |
| `comments` | s or null | Terminal-format optional free text |
| `suspicion_reason` | s or null | Web suspicion free text |
| `additional_comments` | s or null | Web additional free text |

Higher ratings are more favorable. Five terminal-format responses did not ask enjoyment;
five human seats have no submitted survey row. Empty submitted text differs from
an absent field. Starting seats do not establish unique people across games.

## Privacy and analysis boundaries

The four public views preserve the core records, recorded chat clocks and
missingness. They do not add private identifier maps or individual demographics.
The public-redaction record describes categories without the removed strings.
Review does not certify anonymity or consent. See the [dataset guide](README.md)
for rights, linkage risks and intended use.

Access views do not replace the frozen calculation inputs in `analysis/inputs/`.
Preserve original cohort order, word counts, audit boundaries and midnight-window
behavior; never silently revise the scientific method from displayed clocks.
