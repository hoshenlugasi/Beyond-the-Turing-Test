# Release notes — updated 2026-10-18

Initial GitHub release of the reviewed code and data. The Hugging Face dataset
repository has been created, but its study-data upload is pending. The author permits
release provided individual participants cannot be identified. Text review and
filtering reduce disclosure risk but do not guarantee anonymity or establish
independent consent/IRB approval. New local-game exports always need their own
review; they do not inherit the historical dataset's release permission.

## Code and validation

- 2026-10-18: Shortened both package READMEs into practical step-by-step guides.
  Clarified that the four-file game core and the filtered internal events are
  separate. Removed disposable staging caches and the unused duplicate V5 search
  adapter/package. Updated credential/absence tests and two stale docstrings.
  Active runtime statements, study data, frozen inputs and results are unchanged.

- 2026-10-18: Added a standard-library-only explicit two-layer exporter for new
  local games. Full runtime logging is unchanged; a fresh separate destination
  receives the four core schemas and historical selected-event projection.
  Exports remain privacy-pending/non-cohort/non-publication-authorized. No
  historical data, analysis inputs, prompts, voting or runtime policies changed.
  Earlier real Flash validation applies to the unchanged runtime, not to an
  additional paid game; exporter checks are offline against synthetic fixtures
  and the saved completed scripted game.

- The public runtime retains six paper workers: InternetContext, PhaseSummarizer,
  PersonaMonitor, SocialMemory, BehaviorEvolution and RepetitionMonitor. The
  original research checkout's diagnostic Live Evaluator is intentionally removed,
  including its model, feedback path and UI control; no Pro role is a default.
  This is not a claim that the diagnostic never ran historically: it ran with
  feedback off. Historical data, judgments and annotations are unchanged.
- Fresh Windows Python 3.12.10 installation and `pip check` passed. Extracted
  archives passed **427 tests / 27,987 subtests**, with two private-source checks
  skipped; the staging suite passed 429 tests. All public-data integrations ran.
  Twelve offline calculation runs matched the bundled results. External ZIP
  extraction and a fresh venv were used under the same account, not a VM.
  Real unmodified server startup and admin/config-only HTTP checks passed;
  no AI game or model call was made. Current test scope is Windows only.
  Linux/macOS and Docker remain unvalidated.
- Earlier browser checks covered admin login, lobby creation, six human joins
  and blocked START without AI readiness, with AI launch disabled. Separately,
  one real Flash game passed with two scripted human-seat clients: 51 Flash-only
  SDK responses, four full candidate/rank/delivery chains, all three ballots and
  normal end. This revision removes three unused V5 files and updates two
  docstrings; all retained executable runtime statements match that live-tested
  archive. The new exporter ran offline on the saved game, with all 51 raw files
  unchanged and 198 selected internal events exported. Grounded Search reliability
  and live multi-round coverage remain unproven. Further paid tests need approval.
- No Windows-verified lockfile is claimed by these notes.

Use your own admin password and provider key; the environment reference is not
automatically loaded. Google's [official model catalog](https://ai.google.dev/gemini-api/docs/models/gemini),
fetched 2026-10-15, limits Gemini 2.5 access to prior active users. New accounts or
projects may lack access; provider entitlement and continued availability are
conditional. No automatic upgrade is guaranteed, and choosing a replacement
model does not reproduce the original model condition. Explicit custom model
choices remain distinct from the Flash defaults.

The manual local game requires at least two humans plus one AI, with no fixed
five-human cap, configured target or automatic start. Codenames have a neutral
unique fallback; the operator waits for AI readiness and presses START. This is
not a new collection of the historical study condition.

## Public core and mirror identity

The submitted **201-file baseline tree** matched the workspace baseline. After
full manual review of all 50 transcripts and 50 survey files, the public core
adds four redacted spans across three files:

| Game | Public file | Additional redaction |
|---|---|---|
| game_030 | [transcript](games/game_030/transcript.txt) | Two work-project tokens |
| game_006 | [transcript](games/game_006/transcript.txt) | One location echo |
| game_045 | [survey](games/game_045/surveys.json) | One family-event reference in an additional comment |

**197 of 200 game files are unchanged** from the submitted baseline. No rows,
ratings, ballots or outcomes changed.
[public_redactions.json](public_redactions.json) records categories and locations,
not the removed original strings. This GitHub release includes the reviewed core;
the Hugging Face study-data upload is pending. The public core is not wholly byte-identical to the
submitted baseline. The copied [DATASET.md](DATASET.md) describes the submitted
four-file core; its exclusion of internal outputs is scoped to that core.

## Reviewed internal-output export

The separate [manifest](internal_outputs/manifest.json) and
[scope guide](internal_outputs/README.md) accompany per-game event files, for
example [game_001 events](internal_outputs/games/game_001/events/game_events.jsonl).
The export contains **26,277 events across 50 games**. All **5,385 unique text
values** in the initial projection were manually reviewed; **22 flagged or
uncertain values were withheld at all 54 occurrences**. The conservative
field-level privacy filter remains in place.

Included: scheduler decisions, generated candidates, short explicit reviewer
assessments/selections, deterministic guards, delivery statuses and selected
behavior-advice, repetition and phase updates. This is **not all workers**, full
social/persona memory, full prompts or snapshots. Cost/billing metadata and
private identifiers are excluded. Omitted records are not zeros or WAITs;
selection is not delivery, and candidates are not public-chat messages.

## Reproduction scope and limitations

- Four standard-library analysis scripts compute JSON from public numeric inputs
  and cached model outputs. They generate no figures and make no API calls;
  optional paid API-rerun code is not included. All paper numerical summaries use
  frozen outputs offline; no paid analysis reruns are planned.
- The exact annotation prefix/protocol, settings, frozen validation manifest and
  [input provenance](analysis/inputs/PROVENANCE.md) are supplied; these are no
  longer outstanding provenance gaps.
- One game crosses midnight. Reproduction preserves the original time-of-day
  window behavior and original counts/memberships, rather than silently fixing
  the historical limitation. See
  [analysis/REPRODUCIBILITY_NOTES.md](analysis/REPRODUCIBILITY_NOTES.md).
- Tests confirm that survey sample SDs match the paper; no SD mismatch remains
  outstanding. Missing enjoyment ratings in five legacy surveys are not zeros.
- Demographic aggregates cover 226 records. The 234 starting human seats do not
  establish 234 distinct people; public data does not verify individual linkage
  or deduplication.

See [README.md](README.md) for setup and password commands, and
[analysis/README.md](analysis/README.md) for calculation inputs and methods.
Copyright notices, license scope and citation metadata are unchanged.

## Reference settings

The defaults are V6 legacy orchestration, a Swiss student persona, Gemini 2.5
Flash for generator/scheduler/reviewer and the six background workers, four
candidates, generator/reviewer temperatures 1.3/0.3, a 35-second pipeline timeout
and 2.5-minute discussions. Model/time controls remain available. Changing them
or the roster creates a new experiment, not a replay of the study condition.

The reviewer's choice is not proof of delivery. Stale, timeout, phase, parse,
reviewer and deduplication checks can suppress it; the AI personal-chat append
is the delivery record. Full runtime saving and the explicit exporter are both
included. The exporter emits only allowlisted event fields, never full memory
or prompts. Its manifest records missingness, omitted events, masking and hashes.

## Other ways to run

### Google credentials and other providers

`GOOGLE_API_KEY` can be used instead of `GEMINI_API_KEY`. For Vertex AI, leave
both API-key variables unset, set `GOOGLE_CLOUD_PROJECT` and
`GOOGLE_CLOUD_LOCATION` to your own project and supported location, and configure
Application Default Credentials for that account. Keep credential files outside
the repository and container. No author project or credentials are supplied.

Optional `OPENROUTER_API_KEY` and `TOGETHER_API_KEY` support the corresponding
per-role UI choices. InternetContext still uses Gemini for grounding and needs
Google authentication even when other roles use another provider.

### Linux/macOS shell (not validated)

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