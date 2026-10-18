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

The public source retains the six paper workers and legacy orchestration, but
intentionally omits the original research checkout's diagnostic Live Evaluator,
its model, feedback path and UI control. There is no default Pro role. The
diagnostic ran historically with feedback off; omission is not evidence it never
ran. Historical data, cached judgments and annotations remain unchanged. This
does not establish that every historical server build was byte-identical to
today's checkout.

Provider-side model behavior, availability, latency and live web-search results
can change. Google's [official model catalog](https://ai.google.dev/gemini-api/docs/models/gemini),
fetched 2026-10-15, limits Gemini 2.5 access to prior active users; new accounts
or projects may lack access. Entitlement must be checked with the chosen provider;
there is no automatic-upgrade or continued-availability guarantee. A replacement
model is a new experiment, not the original model condition.

Fresh Windows archive validation passed 427 tests and 27,987 subtests; two
private-source checks skipped while all public-data integrations ran. Twelve
stdlib-only analysis runs reproduced bundled results without network calls or
original-tree reads. The fresh venv and external extraction used the same Windows
account, not a VM. One separately approved real Flash functional game passed
with two scripted human-seat clients; it did not replay cohort outcomes or
establish scientific quality, live multi-round or reliable Search coverage.
Further paid tests require separate approval.

## Internal observability

The separately documented structured-output export is not a full execution log.
Missing events are not inferred. Candidate generation, reviewer selection and
confirmed delivery are distinct stages. Explicit short reviewer assessments are
not a complete explanation of model cognition or evidence of causal worker
effectiveness. Full prompts, full snapshots, private profiles and billing
envelopes are excluded. Privacy review is not consent or legal clearance for
unrestricted redistribution; authors must confirm those terms before uploading.