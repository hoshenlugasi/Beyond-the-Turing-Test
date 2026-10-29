---
configs: [{config_name: games, default: true, data_files: [{split: data, path: "tables/games.jsonl"}]}, {config_name: messages, data_files: [{split: data, path: "tables/messages.jsonl"}]}, {config_name: votes, data_files: [{split: data, path: "tables/votes.jsonl"}]}, {config_name: surveys, data_files: [{split: data, path: "tables/surveys.jsonl"}]}]
---

# Social Imitation Game Dataset

The 50-game historical condition reported in **Beyond the Turing Test:
Evaluating AI Agents in Group Conversation**. This guide covers both the
four-file observational core and four access-table views.

Navigate to [loading](#loading), [cohort and core records](#cohort-and-core-records),
[field definitions](#field-definitions), [scientific reproduction](#scientific-reproduction)
or [privacy and rights](#review-intended-use-and-rights).

## Layout and authority

Each directory under [games](games) contains four core files: metadata, readable
transcript, cast ballots and submitted surveys. For example, [game_001](games/game_001)
contains exactly those four core files.

The [games](tables/games.jsonl), [messages](tables/messages.jsonl),
[votes](tables/votes.jsonl) and [surveys](tables/surveys.jsonl) tables are deterministic
access views, not replacement scientific inputs. The [field definitions](#field-definitions)
below describe types and missingness. Frozen calculation inputs and provenance
remain in the [code repository's analysis guide](https://github.com/hoshenlugasi/Beyond-the-Turing-Test/blob/main/analysis/README.md)
and its `analysis/inputs/` directory; they are not part of the Hugging Face
dataset-only package.
Historical component event traces are not part of the current release.

## Loading

From the GitHub repository root, use
`load_dataset("data", name="messages", split="data")`. On Hugging Face, use
`load_dataset("hoshen/Beyond-the-Turing-Test", name="messages", split="data")`.
The four configs are `games`, `messages`, `votes` and `surveys`.
`dataset_infos.json` supplies explicit nullable types. Filter message rows by
`speaker_role != "system"` for the 2,414 player-message denominator.

## Publication and access

All four hosted configs passed full and streaming loading with exact canonical
rows: 50 games, 2,954 messages, 618 votes and 229 surveys. This verifies library
loading, not the four-view visual Viewer. The automatic Hugging Face Viewer is
temporarily disabled by the author's choice because its server cache still
showed the retired Events preview. Files and all four loading configs remain
usable; no automatic re-enablement is scheduled.

## Cohort and core records

The message access table has 2,954 rows: 2,414 player messages and 540 system
notices. There are 50 games, 135 rounds, 2,414 player messages (1,895 human / 519 AI),
618 cast ballots (483 human / 135 AI), and 229 surveys. Sixteen sessions began
with four humans and 34 with five; inclusion of four-human starts was accepted
before outcomes were inspected and no player was removed after START in those
sessions. Use roster-adjusted references and four-/five-human sensitivity results.
The 50 games involved 234 unique human participants.
Do not pool alternative model conditions or test games into this cohort.

Identifying details in the public text were redacted. No rows, ratings, ballots
or outcomes changed. Real-name mappings, recruitment IDs and raw configurations
are not released. Pseudonyms support within-game joins, not external identity
inference. Text review cannot guarantee anonymity.

The transcript includes Game-Manager notices and absolute HH:MM:SS clock tokens.
Its messages describe discussion, voting, aggregate tally, elimination and the
reveals visible to players. Each round is discussion then voting; a tied tally
can use random tie-breaking. The AI wins by reaching the final two, or loses on
elimination. Metadata identifies the AI pseudonym, not a real person's identity.

Ballots retain the AI-voter flag because AI votes counted toward elimination.
Exclude those rows for human-only identification. Explicit abstentions are omitted;
no target is inferred. Five rostered humans sent no chat messages, so legitimate
ballot pseudonyms need not appear as transcript speakers. Silent players still
belong in active-player participation denominators.

Of the 234 unique human participants, 229 submitted surveys; missing responses are omitted. Five
responses use `legacy_terminal_v1`: likeness, timing and relevance ratings plus
optional comments, without enjoyment. The other 224 use `web_v1`: likeness,
timing, relevance, enjoyment, suspicion reason and additional comments. Ratings
are integers 1-5. Missing enjoyment is not zero; empty submitted text is distinct
from an absent field. Demographic information is aggregate-only for 226 records.

## Scientific reproduction

Keep the original cohort order, audit boundaries, numeric word counts and frozen
judgments/annotations provided by the
[code repository's analysis guide](https://github.com/hoshenlugasi/Beyond-the-Turing-Test/blob/main/analysis/README.md).
Its `analysis/inputs/` files are required for the paper calculations but are not
needed for reading these 50 games or loading the four dataset views. One game crosses midnight; the
original time-of-day-window limitation is preserved. The four calculation
scripts do not require historical component traces. Four-file core alone cannot
recover all calculation inputs or fresh model decisions. Fixed, reviewed paper
examples and numerical inputs remain separate from unrestricted runtime logs.

## Field definitions

Both data packages use these definitions. UTF-8 JSONL has one object per line;
core ballot/survey files are JSON arrays. Public IDs and pseudonyms support
within-game joins only. No raw-to-public mapping is released.

### Authority and missingness

The four core files are canonical; tables are access views. The companion code
package's [dataset table projection](https://github.com/hoshenlugasi/Beyond-the-Turing-Test/blob/main/social_turing_test/core/dataset_tables.py)
defines their derivation. Historical component event traces and their coverage
fields are not included.

Notation below: `s` string, `i` integer, `b` Boolean. Nullable fields distinguish
missing or unrecorded values from zero, false and empty submitted text. Integer
fields do not accept Booleans. Masked text is not a missing-value null.

Jump to [games](#games-50-rows), [messages](#messages-2954-rows),
[votes](#votes-618-rows) or [surveys](#surveys-229-rows).

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
five participants have no submitted survey row. Empty submitted text differs from
an absent field.

### Privacy and analysis boundaries

The four public views preserve the core records, recorded chat clocks and
missingness. They do not add private identifier maps or individual demographics.
Identifying details were redacted without publishing the removed strings.
Review does not certify anonymity or consent. See [privacy and rights](#review-intended-use-and-rights)
for linkage risks and intended use.

Access views do not replace the
[code repository's frozen calculation inputs](https://github.com/hoshenlugasi/Beyond-the-Turing-Test/blob/main/analysis/README.md).
Preserve original cohort order, word counts, audit boundaries and midnight-window
behavior; never silently revise the scientific method from displayed clocks.

## Review, intended use and rights

Text review reduces disclosure risk but is not an anonymity guarantee.

Use for group human-AI interaction research and reported numerical reproduction.
Do not re-identify participants, link pseudonyms to outside identities, or treat
persona/social output as verified demographics, personality or clinical traits.
The author permits release conditional on not identifying individuals. This is
not independent consent/IRB certification or permission for unrelated personal-data
use. Applicable MIT terms cover the distributor's rights; participant privacy and
other third-party rights remain. Review reduces risk but is not anonymity proof.
