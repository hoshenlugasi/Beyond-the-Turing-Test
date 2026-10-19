# Social Imitation Game Dataset

The 50-game historical condition reported in **Beyond the Turing Test:
Evaluating AI Agents in Group Conversation**. This guide covers both the
four-file observational core and its reviewed component timelines.

## Layout and authority

Each directory under [games](games) contains four core files: metadata, readable
transcript, cast ballots and submitted surveys. For example, [game_001](games/game_001)
also contains [events/timeline.jsonl](games/game_001/events/timeline.jsonl),
[events/timeline.txt](games/game_001/events/timeline.txt) and
[events/manifest.json](games/game_001/events/manifest.json). JSONL is the event
authority; text is its deterministic full rendering. There is no parallel
internal-output game tree and no mandatory HTML viewer.

[dataset_manifest.json](dataset_manifest.json) summarizes timeline coverage.
Root [games](tables/games.jsonl), [messages](tables/messages.jsonl),
[votes](tables/votes.jsonl) and [surveys](tables/surveys.jsonl) tables are deterministic
access views, not replacement scientific inputs. Events are loaded directly from
the per-game timelines. [DATA_DICTIONARY.md](DATA_DICTIONARY.md) defines fields,
types and missingness. [analysis/inputs/PROVENANCE.md](analysis/inputs/PROVENANCE.md)
documents the separate frozen calculation inputs.

## Cohort and core records

The message access table has 2,954 rows: 2,414 player messages and 540 system
notices. There are 50 games, 135 rounds, 2,414 player messages (1,895 human / 519 AI),
618 cast ballots (483 human / 135 AI), and 229 surveys. Sixteen sessions began
with four humans and 34 with five; inclusion of four-human starts was accepted
before outcomes were inspected and no player was removed after START in those
sessions. Use roster-adjusted references and four-/five-human sensitivity results.
The 234 starting human seats are not a verified count of distinct people.
Do not pool alternative model conditions or test games into this cohort.

Relative to the submitted core, four additional identifying/linkable spans across
three files were redacted;
197 of 200 files were unchanged. [public_redactions.json](public_redactions.json)
records categories and locations, not removed strings. No rows, ratings, ballots
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
are integers 1–5. Missing enjoyment is not zero; empty submitted text is distinct
from an absent field. Demographic information is aggregate-only for 226 records.

## Reviewed timelines and omissions

The timelines retain **53,360 of 54,918 source records**. Recorded study diagnostic
evaluation used Gemini 2.5 Pro (`gemini-2.5-pro`) with feedback disabled; its
**1,558 records are excluded from released timelines**. These diagnostic outputs
are not outputs of the six paper workers.

The foreground has Scheduler, Generator and Reviewer. Six workers contribute
recorded outputs: Internet Context (8,760 events), Round Summarizer (4,750),
Persona Monitor (694), Social State Tracker (2,897), Behavior Adapter (4,900),
and Repetition Monitor (815). System events are controls/observations, not a
seventh worker. Aggregate and per-game manifests retain component/action counts,
omission reasons and timing quality, including zero versus unknown coverage.

The Generator emitted four candidates per normal batch; the Reviewer selects
one verbatim, without rewriting. There are 582 candidate-batch events and 580
reviewer selections retained. Short reviewer pros/cons are
explicit emitted assessments, not hidden reasoning. A selection or pipeline
completion is not delivery. There are 519 sent events; the AI personal-chat
append is the delivery authority. Later stale, timeout, phase, reviewer, parse
and duplicate gates can suppress a selection. Rejected candidates are not player
messages and must not inflate chat denominators.

Internet records distinguish plans, requests, returned results, rejection/error,
stored facts and observed snapshot references. A recorded returned-item count
can include a synthesized answer item; it is **not** an independent cited-source
count. A seed is not live-search evidence, a stored/available fact does not prove
foreground use, and grounding is recorded only when explicitly supplied. Persona
claims and social observations are model interpretations, not verified traits.
Full prompts, full snapshots, raw provider results, billing envelopes, private
source mappings and unrestricted runtime logs are not distributed. Missing events
do not prove inactivity, WAIT, failure or a zero-valued output.

## Clocks, order and scientific limits

All 50 historical event streams were vetted for same-stream intervals, with
sample inspection. Their `time_basis` is `since_first_event`, and precision is
one second. `elapsed_seconds` starts at the **first source event, not game START**;
the first retained event need not define that anchor. No inferred date, chat-clock
merge or timezone correction is applied. In particular no three-hour adjustment
is used to align events with chat. Core transcript clocks remain HH:MM:SS.

`event_order` is deterministic retained source-write order, not a causal ordering
of concurrent work. Same-second events remain ordered by that field. Background
records are not assigned an unrelated foreground turn, and no inferred round is
added to the event envelope. Unknown timing stays null under `unknown`.

Future local exports use verified monotonic logger-segment intervals where
available, starting at a segment anchor rather than START, with 0.001-second
precision. There is NO public segment-index column: segment identifiers and
anchors remain private. Only the logger's complete paired elapsed/monotonic/
segment/basis/precision contract is accepted. Bare elapsed, wall/epoch clocks,
unpaired clock tokens and restart-ambiguous records without that contract stay
null/unknown. Invalid paired timing is refused. Valid restarted segments are
independent; do not stitch their intervals into a continuous game clock. Future
exports remain privacy-pending, outside the cohort; full raw files stay local.

Event intervals do not change scientific calculations. Preserve original
cohort order, audit boundaries, message memberships, original numeric word counts
and frozen judgments/annotations. One game crosses midnight; the original
time-of-day-window limitation is retained, not silently corrected by these traces.
Four-file core alone cannot recover all calculation inputs or fresh model decisions.

## Review, intended use and rights

A first pass reviewed 7,408 unique text values; 52 were withheld at 135 payload
occurrences. A further 22 withheld values occur at 111 payload locations.
Derived text repeats payload content and is not counted
as additional independent occurrences. Contextual verification was selective, not
exhaustive; no re-identification test or anonymity guarantee is claimed.

Use for group human–AI interaction research and reported numerical reproduction.
Do not re-identify participants, link pseudonyms to outside identities, or treat
persona/social output as verified demographics, personality or clinical traits.
The author permits release conditional on not identifying individuals. This is
not independent consent/IRB certification or permission for unrelated personal-data
use. Applicable MIT terms cover the distributor's rights; participant privacy and
other third-party rights remain. Review reduces risk but is not anonymity proof.

The final outside-ZIP validation report identifies the exact artifacts checked,
including 12 offline scientific reproductions, tests in a new external Windows
virtual environment, server checks and local dataset loading. Its evidence does
not automatically certify edited files or rebuilt packages. No new model-backed
game, scientific-quality validation or remote Hub viewer validation is claimed.
