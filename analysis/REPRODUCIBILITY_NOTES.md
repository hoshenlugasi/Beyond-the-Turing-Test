# Reproduction scope and recorded limitations

The four calculation commands use the submitted public records plus the small
privacy-minimized inputs in this directory. They do not read the research
repository, request paid inference, regenerate charts, or manufacture results.
All paper numerical summaries are recalculated offline from frozen outputs;
no paid judge or annotation reruns are planned.

## Exact historical analysis versus clock rollover

One recorded game crosses midnight. The original analysis compared time-of-day
seconds directly rather than unwrapping the clock. The exported calculation
preserves the original audit boundaries and message-to-round assignments so that
the published statistics are reproducible. In that game the second audit cutoff
is numerically earlier than the first. A later round can consequently have no
messages in the original `(previous_end, end]` participation window even when the
displayed transcript continues. Do not interpret that window as a new empirical
finding about participant silence. Correcting rollover would be a methodological
revision, not an equivalent packaging change; no correction was silently made.

## Public text and original numerical features

The baseline is the 50-game supplementary release, not raw server directories.
The public version additionally redacts four identifying/linkable spans across
three files, documented in the root public-redaction manifest. Original message
word counts are provided as numerical features without the removed source text.
They support the paper's original word-length calculation. The participant
rosters include silent humans for participation denominators; a silent player's
undefined message-label proportion is not included in the message-share metric.
The public core is not wholly byte-identical to the submitted baseline. Redacted
display text cannot recover removed text or authenticate a fresh reannotation;
do not substitute its recalculated word counts for the frozen numeric features.

## Frozen model judgments and annotations

The stage-judge ledgers contain one retained valid decision per stage and model.
They are the original selections, not fresh inference on redacted public text.
The two judges cover the same 135 stages and are never pooled across room sizes.
GPT-5 uses the recorded minimal-effort setting except the one retained low-effort
retry. The complete fixed message-annotation prefix is supplied separately from
historical participant text. Boolean model and human labels are sufficient for
the four-label rates and all ten conversational-label agreement calculations.
Original human JSON/CSV pairs were checked before this minimal projection;
their names, notes, save times and local file paths are not distributed.

## Demographics and study provenance

The demographic file contains verified source aggregates for 226 available
records, not participant-level records or an independently rerunnable recruitment
API export. The 234 initial human seats are not, by themselves, evidence of 234
unique individuals. No cross-game identity linkage is released.

The agent has Scheduler, Generator and Reviewer plus six paper workers: Internet
Context, Round Summarizer, Persona Monitor, Social State Tracker, Behavior Adapter
and Repetition Monitor. The reference model is Gemini 2.5 Flash
(`gemini-2.5-flash`); the Reviewer selects a generated candidate verbatim, without
rewriting. Recorded study diagnostic evaluation used Gemini 2.5 Pro
(`gemini-2.5-pro`) with feedback disabled; its 1,558 records are excluded from
released timelines. Historical data, cached judgments and annotations are frozen.

Provider-side model behavior, availability, latency and live web-search results
can change. Google's [official model catalog](https://ai.google.dev/gemini-api/docs/models/gemini),
fetched 2026-10-15, limits Gemini 2.5 access to prior active users; new accounts
or projects may lack access. Entitlement must be checked with the chosen provider;
there is no automatic-upgrade or continued-availability guarantee. A replacement
model is a new experiment, not the original model condition.

The final outside-ZIP validation report identifies the exact artifacts checked,
including 12 offline scientific reproductions, tests in a new external Windows
virtual environment, server checks and local dataset loading. Its evidence applies
only to those artifacts, not automatically to edited files or rebuilt packages.
Isolation used same-account external directories and guarded processes, not a VM.
No new model-backed game, scientific-quality validation or remote Hub viewer
validation is claimed here.

## Component observability, clocks and privacy

Every game has four core files plus nested timeline JSONL, a full derived text
rendering and coverage manifest. Four root tables provide access views, not new
scientific inputs. See [dataset guide](../DATASET.md),
[dictionary](../DATA_DICTIONARY.md) and [manifest](../dataset_manifest.json).
Timelines retain 53,360 of 54,918 source events, excluding 1,558 historical
diagnostic records, and cover all six paper workers. This is not a full execution
log: full prompts/snapshots, raw provider responses and private mappings are omitted.

Historical timing is one-second precision since the first source event, not START.
The first retained event need not be the anchor. No date, chat-clock merge,
timezone correction or causal concurrency order is inferred. Future exports
accept verified paired monotonic logger-segment intervals, not a game-start clock.
There is no public segment-index column. Incomplete or restart-ambiguous unpaired
timing stays null/unknown; restarted segments are independent and cannot be
stitched into a continuous elapsed axis. Invalid paired timing is refused.
None of this revises historical analysis windows or original numeric word counts.

Missing events are not inactivity, WAIT or zeros. Candidate generation, selection
and confirmed delivery remain distinct; personal-chat append is delivery authority.
Emitted short assessments are not hidden reasoning or causal effectiveness proof.
Recorded returned-item counts are not independent sources; available facts do not
prove grounding or use. Persona/social output is not verified human biography.

A first pass reviewed 7,408 unique text values; 52 were withheld at 135 payload
occurrences. A further 22 withheld values occur at 111 payload locations.
Rendered repetition is not counted again. Contextual checks were
selective, not exhaustive. Author permission is conditional on not identifying
individuals; review is not an anonymity guarantee, independent consent/IRB
certification or clearance for unrestricted redistribution.