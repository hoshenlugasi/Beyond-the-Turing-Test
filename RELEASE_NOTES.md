# Release notes — 2026-10-18

GitHub release updated from the 2026-10-18 package. The existing
[Hugging Face mirror](https://huggingface.co/datasets/hoshen/Beyond-the-Turing-Test)
now matches this revision's shared dataset payloads. The author's release permission
is conditional on not identifying participants. Text review and filtering reduce
disclosure risk but do not guarantee anonymity or establish independent consent/IRB
approval. New local-game exports always need their own review; they do not inherit
the historical dataset's release permission.

## Current package and validation boundary

This package provides one agent and one nested game-data layout.
Scheduler, Generator and Reviewer retain the paper policies. The six background
workers are Internet Context, Round Summarizer, Persona Monitor, Social State
Tracker, Behavior Adapter and Repetition Monitor. System delivery controls and
the shared-memory snapshot are not additional workers.

Recorded study diagnostic evaluation used Gemini 2.5 Pro (`gemini-2.5-pro`) with
feedback disabled; its 1,558 records are excluded from released timelines.
Of 54,918 source records, 53,360 reviewed timeline records remain. The diagnostic
outputs are not outputs of the six paper workers. Scientific inputs, decisions
and results are frozen.

The final outside-ZIP validation report identifies the exact artifacts checked:
12 offline scientific reproductions, tests in a new external Windows virtual
environment, server checks and local dataset loading. Its evidence applies only
to those artifacts, not automatically to edited files or rebuilt packages.
No new model-backed game or scientific-quality validation is claimed.
Linux/macOS, Docker and a remote Hub viewer remain unvalidated for this package.
No full lockfile or perpetual model entitlement is claimed.

Hosted loading of the published dataset at
[858633f](https://huggingface.co/datasets/hoshen/Beyond-the-Turing-Test/commit/858633fdbf303f9243bb416c7e84c6d8a84ebd5b)
passed with `datasets==5.1.0` for all five configs in normal and streaming modes:
50 games, 2,954 messages, 618 votes, 229 surveys and 53,360 events. Every row and
nullable schema matched the release. This verifies hosted library loading, not
the visual Hugging Face Data Studio/Viewer, whose confirmation remains pending.

## Data, privacy and timing

The [dataset guide](DATASET.md), [dictionary](DATA_DICTIONARY.md) and
[dataset manifest](dataset_manifest.json) describe the released records.
Each game has four core files and three nested event files. Four root JSONL
tables provide game/message/ballot/survey access; events read per-game timelines.
The Hugging Face card declares five configs and only the split `data`.

A first pass reviewed 7,408 unique text values; 52 were withheld at 135 payload
occurrences. A further 22 withheld values occur at 111 payload locations.
Contextual verification was selective, not exhaustive; no
re-identification test establishes anonymity. Full prompts/snapshots, raw provider
responses, billing envelopes and private mappings are excluded.

All 50 same-stream historical event clocks were vetted, with sample inspection.
Intervals have one-second precision **since the first source event, not START**.
No date, event/chat alignment or three-hour timezone correction is inferred.
Core chat retains its HH:MM:SS tokens; original scientific midnight windows and
numeric word counts remain unchanged. Write order is not causal concurrency order.
Future monotonic intervals start at a private logger-segment anchor, not START.
There is no public segment-index column. Incomplete/restart-ambiguous unpaired
timing stays null/unknown; invalid paired timing is refused. Valid restarted
segments are independent and must not be stitched into a continuous game clock.

The public core has four additional redacted spans across three files relative
to the submitted core (197/200 unchanged), recorded in
[public_redactions.json](public_redactions.json). No rows, ratings, ballots or
outcomes changed. The historical cohort remains 50 games, including 16 four-human
starts and 34 five-human starts. Review does not establish distinct-person linkage.

## Reference settings and interpretation

The reference defaults retain the paper agent's orchestration, Swiss student
persona, Gemini 2.5 Flash (`gemini-2.5-flash`) for Scheduler/Generator/Reviewer and all six workers,
four generated candidates, generator/reviewer temperatures 1.3/0.3, 35-second
pipeline timeout and 2.5-minute discussion. Reviewer scoring, ranking, guards,
strategic voting, question policy, timing and model defaults define the reference condition.
The reviewer selects a candidate verbatim, not a rewrite.

Selection is not delivery: stale, timeout, phase, reviewer, parse and deduplication
gates may suppress output. The AI personal-chat append is delivery authority.
Recorded Internet returned-item counts are not independent sources or proof of
grounding; seeds are not live retrieval. Available context does not prove use.
Persona/social observations are model claims, not verified human attributes.

The local game permits at least two humans plus one AI and manual START, not the
historical recruitment protocol. Changing roster/models/settings creates a new
experiment, not a replay. Full raw runtime saving under games/real/&lt;id&gt; or
games/test/&lt;id&gt; is automatic. [export_game.py](export_game.py) is a separate,
offline manual step after stopping the game and collecting desired surveys.
Its fresh atomic 14-file output has four core files, three nested events, four
access tables, loading metadata, a README and export manifest. It never overwrites, modifies source
files, adds historical games, authorizes publication or creates new scientific
judgments. Automatic masking leaves privacy status pending.

## Reproduction scope

The four [analysis commands](analysis/README.md) print JSON from frozen public
inputs without API calls or figures. Exact annotation protocol/settings and
cached decisions are supplied; no paid reannotation is claimed. Original cohort
order, numeric word counts and midnight-window behavior must be preserved.
Enjoyment is missing in five terminal-format surveys, not zero. Demographic aggregates
cover 226 records and do not independently verify cross-game identity linkage.
See [reproduction notes](analysis/REPRODUCIBILITY_NOTES.md).

## Other ways to run

### Credentials and model access

`GOOGLE_API_KEY` may replace `GEMINI_API_KEY`. For your own Vertex AI endpoint,
leave both keys unset, set `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`,
and configure Application Default Credentials outside the repository/container.
No author project, key or credential file is supplied. The environment reference
is not automatically loaded. Check current provider entitlement; Gemini 2.5
access can be restricted and there is no automatic model replacement.

Optional `OPENROUTER_API_KEY` and `TOGETHER_API_KEY` support the corresponding
per-role UI choices. InternetContext still needs Google authentication for Gemini
grounding even when other roles use another provider. Live requests may incur costs.

### Linux/macOS (not validated)

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
printf 'Choose your admin password: '
stty -echo
IFS= read -r STT_ADMIN_PASSWORD
stty echo
printf '\nYour Gemini API key: '
stty -echo
IFS= read -r GEMINI_API_KEY
stty echo
printf '\n'
export STT_ADMIN_PASSWORD GEMINI_API_KEY
python web_player.py
```

Use `python` in place of the Windows Python commands in the main guide.

### Gunicorn and Docker (not validated)

Gunicorn is for Linux, including Linux containers, not Windows. Use exactly
one worker because state and locks are process-local. Do not add `--preload`:

```sh
gunicorn --workers 1 --bind 127.0.0.1:5000 --timeout 120 web_player:app
```

With your admin password and Google API key already exported:

```sh
docker build -t social-turing-test .
docker run --rm -p 127.0.0.1:5000:5000 --env STT_ADMIN_PASSWORD --env GEMINI_API_KEY social-turing-test
```

The Python 3.12 slim image runs as a non-root user and excludes the study data,
analysis inputs and credentials. This example is ephemeral: removing the
container removes its new runtime games. No private host folders or ADC volumes
are mounted. Pass `GOOGLE_API_KEY` instead if that is the variable you use.

The requirements are not a full lockfile. No local model weights or PyTorch are
needed. The offline analyses use only the standard library; runtime dependencies
may install NumPy even though the analyses do not require it.