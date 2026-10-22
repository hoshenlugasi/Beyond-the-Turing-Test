---
configs: [{config_name: games, default: true, data_files: [{split: data, path: "tables/games.jsonl"}]}, {config_name: messages, data_files: [{split: data, path: "tables/messages.jsonl"}]}, {config_name: votes, data_files: [{split: data, path: "tables/votes.jsonl"}]}, {config_name: surveys, data_files: [{split: data, path: "tables/surveys.jsonl"}]}]
---

# Social Imitation Game Dataset

The 50-game historical condition reported in **Beyond the Turing Test:
Evaluating AI Agents in Group Conversation**. This guide covers both the
four-file observational core and four access-table views.

## Layout and authority

Each directory under [games](games) contains four core files: metadata, readable
transcript, cast ballots and submitted surveys. For example, [game_001](games/game_001)
contains exactly those four core files.

The [games](tables/games.jsonl), [messages](tables/messages.jsonl),
[votes](tables/votes.jsonl) and [surveys](tables/surveys.jsonl) tables are deterministic
access views, not replacement scientific inputs. [DATA_DICTIONARY.md](DATA_DICTIONARY.md)
defines fields, types and missingness. Frozen calculation inputs and provenance
are in the repository's `analysis/inputs/` directory, separate from these views.
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
usable; no automatic re-enablement is scheduled. Older public Git history is
not rewritten.

## Cohort and core records

The message access table has 2,954 rows: 2,414 player messages and 540 system
notices. There are 50 games, 135 rounds, 2,414 player messages (1,895 human / 519 AI),
618 cast ballots (483 human / 135 AI), and 229 surveys. Sixteen sessions began
with four humans and 34 with five; inclusion of four-human starts was accepted
before outcomes were inspected and no player was removed after START in those
sessions. Use roster-adjusted references and four-/five-human sensitivity results.
The 234 starting human seats are not a verified count of distinct people.
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

Of 234 human seats, 229 submitted surveys; missing responses are omitted. Five
responses use `legacy_terminal_v1`: likeness, timing and relevance ratings plus
optional comments, without enjoyment. The other 224 use `web_v1`: likeness,
timing, relevance, enjoyment, suspicion reason and additional comments. Ratings
are integers 1-5. Missing enjoyment is not zero; empty submitted text is distinct
from an absent field. Demographic information is aggregate-only for 226 records.

## Scientific reproduction

Keep the original cohort order, audit boundaries, numeric word counts and frozen
judgments/annotations in `analysis/inputs/`. One game crosses midnight; the
original time-of-day-window limitation is preserved. The four calculation
scripts do not require historical component traces. Four-file core alone cannot
recover all calculation inputs or fresh model decisions. Fixed, reviewed paper
examples and numerical inputs remain separate from unrestricted runtime logs.

## Review, intended use and rights

Text review reduces disclosure risk but is not an anonymity guarantee.

Use for group human-AI interaction research and reported numerical reproduction.
Do not re-identify participants, link pseudonyms to outside identities, or treat
persona/social output as verified demographics, personality or clinical traits.
The author permits release conditional on not identifying individuals. This is
not independent consent/IRB certification or permission for unrelated personal-data
use. Applicable MIT terms cover the distributor's rights; participant privacy and
other third-party rights remain. Review reduces risk but is not anonymity proof.
