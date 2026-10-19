# Data dictionary

Both data packages use this dictionary. UTF-8 JSONL has one object per line;
core ballot/survey files are JSON arrays. Public IDs and pseudonyms support
within-game joins only. No raw-to-public mapping is released.

## Authority and missingness

The four core files and nested timelines are canonical. Root tables are access
views. The companion code package's core/event_schema.py contains the executable
typed payload dictionary (`SCHEMAS`), action/field allowlists (`EVENT_SPECS`) and
envelope validator; core/dataset_tables.py defines access-table projection.
Those modules are code-package resources, not extra files in the data-only package.

Notation below: `s` string, `i` integer, `f` finite floating-point number, `b`
Boolean; `T[]` list of T; named structs have fixed fields. **Every payload field,
struct and list is nullable.** Every present struct emits all its defined keys;
inactive payload branches are null. List items cannot be null. Null means absent,
unknown or unrecorded—not zero, false, empty text or an empty list. Those four
observed values remain distinct. Integer fields do not accept Booleans. Privacy
masking may replace text; a masked string is not a missing-value null.

## Core records and four access tables

### Games (50 rows)

[Metadata](games/game_001/metadata.json) contains the first six fields below;
[tables/games.jsonl](tables/games.jsonl) adds the last three.

| Field | Type | Meaning |
|---|---|---|
| `game_id` | s | Public directory identifier |
| `ai_pseudonym` | s | Agent's in-game name |
| `num_human_players` | i | Starting human seats, four or five in this cohort |
| `total_rounds` | i | Played discussion/vote rounds |
| `ai_eliminated` | b | Whether the AI was eliminated |
| `ai_elimination_round` | i or null | Elimination round; null for AI wins |
| `event_count` | i or null | Manifest retained count; unavailable coverage is null |
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
| `human_likeness` | i | 1–5 likeness rating, 229 available |
| `timing` | i | 1–5 timing/pacing rating, 229 available |
| `relevance` | i | 1–5 relevance/coherence rating, 229 available |
| `enjoyment` | i or null | 1–5 enjoyment rating, 224 available |
| `comments` | s or null | Terminal-format optional free text |
| `suspicion_reason` | s or null | Web suspicion free text |
| `additional_comments` | s or null | Web additional free text |

Higher ratings are more favorable. Five terminal-format responses did not ask enjoyment;
five human seats have no submitted survey row. Empty submitted text differs from
an absent field. Starting seats do not establish unique people across games.

## Event envelope (53,360 rows)

[Nested timelines](games/game_001/events/timeline.jsonl) contain exactly these
fields. There is no public segment identifier, segment index or inferred round.

| Field | Type | Meaning |
|---|---|---|
| `event_schema` | i | Format revision, currently 1, not an agent/model identity |
| `game_id` | s | Public game key |
| `event_id` | s | Game key plus deterministic event ordinal |
| `event_order` | i | Contiguous one-based retained source-write order |
| `component` | s | Paper component label or System |
| `action` | s | Neutral snake-case recorded action |
| `turn_id` | s or null | Public mapped foreground turn when recorded |
| `elapsed_seconds` | f or null | Relative interval under the declared basis |
| `time_basis` | s | Basis described below |
| `time_precision_seconds` | f or null | Declared quantization, not latency accuracy |
| `event_text` | s | Derived component/action plus full payload display |
| `payload` | struct | All branches below; exactly one active branch |

Components: Scheduler, Generator, Reviewer; Internet Context, Round Summarizer,
Persona Monitor, Social State Tracker, Behavior Adapter, Repetition Monitor;
System (controls/observations, not a seventh worker). `action`, not component
alone, determines the active payload branch; generator boundary events can use
the system branch. Only action-allowlisted fields are copied into that branch.

### Timing contract

All historical rows use `since_first_event`, precision 1.0 second, anchored to
the first source event, **not START** or necessarily the first retained record.
No date, chat-clock merge, timezone correction or concurrency causality is inferred.
Use `event_order` for retained order, including same-second records.

The schema also accepts `unknown`, `since_game_start`, `verified_relative` and
`monotonic_segment`; accepting a basis does not establish its use in this cohort.
Unknown timing has null elapsed/precision. Future exports accept verified paired
monotonic logger timing at 0.001-second precision, relative to a private segment
anchor, not START. Segment IDs/anchors are not public columns. Bare elapsed,
unpaired wall/epoch/clock-token timing and restart-ambiguous records without a verified
paired contract remain null/unknown; inconsistent paired values are refused.
Valid restarted segments are independent, not a continuous public game clock.

### Complete typed payload branches

All names/types below mirror the executable dictionary. Types are nullable as
defined above; unused fields of an active branch are null, not inferred.

| Branch | Fields and types |
|---|---|
| `scheduler` | `decision:s`, `send_reason:s`, `reason:s`, `latency_ms:i`, `last_send_ago_s:f`, `nudges:s` |
| `candidates` | `count:i`, `candidates:s[]`, `reason:s`, `history_lines:i`, `current_info_chars:i` |
| `review` | `num_candidates:i`, `selected_index:i`, `verdicts:Verdict[]`, `selected_text:s`, `selected_word_count:i`, `source:s`, `review_status:s`, `review_risk:s`, `review_reason_codes:s[]` |
| `filter` | `removed:Removed[]`, `removed_count:i`, `kept:i`, `demoted:s[]`, `tag:s`, `pool_size:i`, `mode:s`, `dodge_streak:i`, `streak:i` |
| `delivery` | `message:s`, `outcome:s`, `reason:s`, `stage:s`, `new_messages:i`, `suppressed_message:s`, `pipeline_elapsed_s:f`, `timeout_s:f` |
| `internet` | `query:s`, `query_in:s`, `corrected_query:s`, `summary:s`, `summary_out:s`, `topic:s`, `stored:Stored`, `returned_item_count:i`, `origin:s`, `volatility:s`, `confidence:s`, `grounded:b`, `needs_search:b`, `dropped:b`, `reason:s`, `retrying:b`, `year_corrected:b`, `error_category:s`, `plans:Plan[]`, `history_len:i`, `phase_summary_count:i`, `latency_s:f`, `tick_latency_s:f`, `searches_run:i`, `entries_written:i`, `replaced:b`, `total_entries_after:i`, `expired_fact_count:i`, `stale_search_count:i`, `remaining_count:i`, `snapshot_revision:i`, `pruned_count:i` |
| `persona` | `claims:s[]`, `count:i`, `own_msgs:i`, `first_person:i`, `latency_s:f`, `facts:Facts`, `parse_ok:b`, `status:s` |
| `social_state` | `suspects_me:s[]`, `i_pointed_at:s[]`, `voted_out:s[]`, `prime_suspect:s`, `open_threads:s[]`, `group_mood:s`, `players:Player[]`, `room_norm:s`, `room_vibe:s`, `pressure_type:s`, `interrogation_pressure:f`, `attention_on_me:f`, `msgs:i`, `had_signal:b`, `parse_ok:b`, `latency_s:f`, `status:s` |
| `behavior` | `advice:s`, `players_left:i`, `suspicion_on_self:i`, `talk_share:f`, `llm_refreshed:b`, `warnings:s[]`, `recommendation:s`, `participation_assessment:s`, `agreement_ratio:f`, `deflection_ratio:f`, `latency_s:f`, `cadence:Cadence`, `status:s` |
| `repetition` | `words:s[]`, `overused:s[]`, `note:s`, `own_msgs:i`, `latency_s:f`, `status:s` |
| `round_summary` | `phase:s`, `phase_label:s`, `n_messages:i`, `summary:s`, `pending_msgs_count:i`, `latency_s:f`, `tick_latency_s:f`, `status:s` |
| `system` | `snapshot_revision:i`, `entry_count:i`, `turn_index:i`, `scheduler_decision:s`, `send_reason:s`, `final:s`, `review_risk:s`, `pipeline_latency_s:f`, `final_message_len:i`, `dropped:b`, `message_history_len:i`, `status:s`, `reason:s`, `mode:s`, `generator_ok:b`, `reviewer_ok:b`, `scheduler_ok:b` |

| Nested struct | Fields and types |
|---|---|
| Verdict | `index:i`, `pool_index:i`, `text:s`, `type:s`, `pros:s`, `cons:s`, `score:f`, `red_flags:s[]`, `verdict:s`, `selected:b` |
| Removed | `text:s`, `reason:s` |
| Plan | `query:s`, `volatility:s`, `confidence:s`, `reason:s`, `needs_search:b` |
| Stored | `topic:s`, `volatility:s`, `confidence:s`, `returned_item_count:i`, `origin:s` |
| Player | `pseudonym:s`, `observation:s` |
| Facts | `location:s`, `city:s`, `country:s`, `timezone:s`, `nationality:s`, `age:s`, `occupation:s`, `name:s` |
| Cadence | `level:s`, `reason:s`, `talk_share:f` |

Indices and scores retain their recorded action-specific meaning, not a newly
standardized ranking scale. Latencies/counts/ratios are recorded observations;
field names do not certify complete activity or causal influence. Facts are
configured persona claims, not participant identity measurements. `players` is a
list of fixed structs, not dynamic per-person columns. Returned-item counts may
include a synthesized answer item; they are not independent cited-source counts.
`grounded` is recorded only when available, never inferred. Short reviewer pros/
cons are emitted assessments, not hidden reasoning. The Reviewer selects a
generated candidate verbatim, without rewriting. Selection, completion and
context availability do not establish delivery; personal-chat append is authority.

## Manifests, privacy and analysis boundaries

[Per-game manifests](games/game_001/events/manifest.json) and the
[dataset manifest](dataset_manifest.json) report retained counts, component/action
coverage, omissions and time quality. Aggregate totals: 54,918 source events,
53,360 retained, 1,558 excluded diagnostic events. Missing activity is not zero
or WAIT; unknown coverage is not an observed empty stream.

A first pass reviewed 7,408 unique text values; 52 were withheld at 135 payload
occurrences. A further 22 withheld values occur at 111 payload locations.
Rendering repetition is not counted again. Contextual verification
was selective; `reviewed_with_withholding` is not an anonymity/rights guarantee.
No full prompts, snapshots, raw provider responses or private mappings are supplied.
The [dataset guide](DATASET.md) explains rights, linkage risks and intended use.

Access views and event intervals do not replace [frozen calculation inputs](analysis/inputs).
Preserve original cohort order, word counts, audit boundaries and midnight-window
behavior; never silently revise the scientific method from displayed clocks.