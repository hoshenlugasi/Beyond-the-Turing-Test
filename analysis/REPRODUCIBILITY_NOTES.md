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
three files, documented in the public-redaction record alongside the dataset. Original message
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
the public dataset. Historical data, cached judgments and annotations are frozen.

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

## Dataset scope and privacy

Every published game has four core files. Four tables provide access views, not
new scientific inputs. See the
[dataset guide](https://github.com/hoshenlugasi/Beyond-the-Turing-Test/blob/main/data/README.md) and
[dictionary](https://github.com/hoshenlugasi/Beyond-the-Turing-Test/blob/main/data/DATA_DICTIONARY.md).
Historical component event traces are
not included. Their removal leaves the calculation inputs and reference outputs
unchanged; none of the four calculations requires those traces.

Local runtime logging and manual diagnostic exports remain available for new
games, but are not historical data or publication-authorized outputs. Fixed paper
examples are documented separately in [input provenance](inputs/PROVENANCE.md).
Author permission remains conditional on not identifying individuals; review
does not certify anonymity, consent or unrestricted redistribution rights.
