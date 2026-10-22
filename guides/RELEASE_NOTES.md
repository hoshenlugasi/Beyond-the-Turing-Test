# Release notes - 2026-10-22

## Local browser release

Runtime code, packaged configuration and templates are grouped in
`social_turing_test/`. Run `python run_game.py` from the repository root.
The local HTTP service binds to `127.0.0.1` by default. Use `--lan` explicitly
for other devices on a trusted network; authentication remains required.
Container files, Gunicorn and deployment-only manifest tooling are not included.
This still uses a local Flask backend, not a static browser-only application.

## Published dataset scope

The current dataset contains 50 games with four core files each, plus the
`games`, `messages`, `votes` and `surveys` access views. Historical component event
traces, their manifests, the events loading config and event-count fields are
removed from this revision. Older public Git history is not rewritten.
See the [dataset guide](../data/README.md) and [dictionary](../data/DATA_DICTIONARY.md).
The 200 core files and frozen scientific inputs/results retain their bytes.
No model-backed game, new scientific result or new anonymity guarantee is claimed.
Earlier hosted five-config validation concerned the prior release, not this one.

## Local diagnostic exports

Runtime logging, all six background workers and the manual diagnostic exporter
remain available for new local games. A new export includes filtered event traces
and five loading configs; this is separate from the four-view historical dataset.
Every new export is privacy-review-pending and not publication-authorized.
Review all free text and confirm redistribution rights before sharing.
Full prompts, raw provider responses and private identifier maps are not exported.

Future exported intervals use verified logger-segment timing, not game START.
Unknown or restart-ambiguous unpaired timing remains null. Valid restarted segments
are independent and must not be stitched into a continuous game clock.

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
experiment, not a replay. Full raw runtime saving under local_data/games/real/&lt;id&gt; or
local_data/games/test/&lt;id&gt; is automatic. [export_game.py](../social_turing_test/export_game.py) is a separate,
offline manual step after stopping the game and collecting desired surveys.
Its fresh atomic 14-file output has four core files, three nested events, four
access tables, loading metadata, a README and export manifest. It never overwrites, modifies source
files, adds historical games, authorizes publication or creates new scientific
judgments. Automatic masking leaves privacy status pending.

## Reproduction and testing

The four [analysis commands](../analysis/README.md) use the four-file game core
and frozen calculation inputs, not historical component traces. Keep cohort order,
numeric word counts, missing survey values and midnight-window behavior unchanged.
See [reproduction limits](../analysis/REPRODUCIBILITY_NOTES.md).

The public offline regression suite is retained because it covers local HTTP/auth,
configuration, imports, credentials, private diagnostic exports and reproduction.
Historical-trace integration assertions have been replaced by core-only dataset
checks. Tests use synthetic fixtures and mocked providers, not private study tooling.

Author permission is conditional on not identifying participants. Review reduces
disclosure risk but does not establish anonymity or independent consent/IRB approval.
New local games do not inherit the historical dataset's release permission.

## Other ways to run

### Credentials and model access

`GOOGLE_API_KEY` may replace `GEMINI_API_KEY`. For your own Vertex AI endpoint,
leave both keys unset, set `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`,
and configure Application Default Credentials outside the repository.
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
python run_game.py
```

Use `python` in place of the Windows Python commands in the main guide.


The requirements specify direct dependencies, not a full lockfile. No local model
weights or PyTorch are needed. Live provider access remains conditional and may
incur costs; offline analysis and exports use only the standard library.
