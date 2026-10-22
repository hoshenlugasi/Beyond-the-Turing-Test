# Social Turing Test

A group chat game with one AI player. Humans win by voting out the AI; the AI
wins by reaching the final two. Includes one playable agent, 50 study games,
four core data views and offline calculations for **Beyond the Turing
Test: Evaluating AI Agents in Group Conversation**, Hoshen Lugasi, Uri Berger,
Niv Eckhaus, Itay Itzhak and Gabriel Stanovsky (2026).

Code: [GitHub](https://github.com/hoshenlugasi/Beyond-the-Turing-Test).
Data mirror: [Hugging Face](https://huggingface.co/datasets/hoshen/Beyond-the-Turing-Test).
[Project website](https://hoshenlugasi.github.io/Beyond-the-Turing-Test/).
See [release notes](guides/RELEASE_NOTES.md) for this dataset revision's publication status.

## 1. Install on Windows

Install Python 3.12 and open PowerShell in this package's root. All commands and
loading examples below use that working directory; activation is unnecessary.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Reading files needs no credentials or installation. Offline paper calculations
need only the standard library; skip to section 6 for those.

## 2. Set credentials and start the local game

In the same terminal, supply your own strong admin password and Gemini API key
without putting their values in command history:

```powershell
$adminSecret = Read-Host "Choose your admin password" -AsSecureString
$env:STT_ADMIN_PASSWORD = [System.Net.NetworkCredential]::new('', $adminSecret).Password
$googleSecret = Read-Host "Your Gemini API key" -AsSecureString
$env:GEMINI_API_KEY = [System.Net.NetworkCredential]::new('', $googleSecret).Password
Remove-Variable adminSecret, googleSecret
.\.venv\Scripts\python.exe run_game.py
```

Missing password disables admin access; restart after changing it.
[.env.example](.env.example) is a reference, **not automatically loaded**. Never
commit secrets. Live requests incur costs; check your account's Gemini 2.5 Flash
entitlement. Availability is conditional; replacement models are new conditions.
See [other ways to run](guides/RELEASE_NOTES.md#other-ways-to-run).

## 3. Play

1. Open <http://localhost:5000> and log into the admin panel with your password.
2. Create a game with default agent settings. For players on other devices,
   restart with `python run_game.py --lan` and share your reachable LAN address
   rather than localhost. Use this option only on a trusted network.
3. Have **at least two humans** join, wait for AI readiness, then press **START**.
   There is no fixed five-human cap or automatic start.
4. Chat and vote until the game ends; players can then submit surveys.

The local service listens on `127.0.0.1` by default. `--lan` explicitly enables
all interfaces; do not expose this development service directly to the Internet. Defaults retain the paper's orchestration, Swiss student
persona, 2.5-minute discussions, four candidates and verbatim reviewer selection.
Selection is not delivery. Scheduler, Generator and Reviewer work with Internet
Context, Round Summarizer, Persona Monitor, Social State Tracker, Behavior Adapter
and Repetition Monitor. The Reviewer selects a candidate verbatim, without
rewriting. Changing models, settings or roster is not a replay of the study.

## 4. Save and export

Full raw saving is automatic under local_data/games/real/&lt;id&gt; or local_data/games/test/&lt;id&gt;:
configuration, chats, votes, events, logs and memory traces. Keep raw files local.
After the manager and AI stop and desired surveys arrive, export manually:

```powershell
.\.venv\Scripts\python.exe -m social_turing_test.export_game --game-dir local_data/games/real/0001 --output local_exports/my_game --export-id local_game_001 --kind local_game
```

Replace the source with your completed directory. Scripted seats use
`--kind scripted_test`. [export_game.py](social_turing_test/export_game.py) needs only the standard
library, makes no network calls, leaves source unchanged and publishes atomically
to a fresh destination without overwriting. Its 14 files comprise four core
records, three nested event files, four root tables, loading schemas, a README
and export manifest. This is a private diagnostic export for new local games,
not the four-view historical dataset published in this repository.
It never extends the historical cohort. Export late surveys to another fresh
output. Use the ignored local_exports directory inside this checkout.

**Every new export is privacy-review-PENDING and not publication-authorized.**
Review all text, including rejected candidates, reviewer assessments and worker
observations, and check consent/rights. Automatic masking is not clearance.
Missing surveys remain missing; unfinished, inconsistent or unsupported drop/rescue
records are refused. Future timing uses verified logger segments, not START.
There is no public segment-index column; ambiguous unpaired timing is null and
restart intervals must not be stitched. See [local export timing](guides/RELEASE_NOTES.md#local-diagnostic-exports).

## 5. Read study data

Every [game](data/games/game_001) has four files:
[metadata](data/games/game_001/metadata.json),
[transcript](data/games/game_001/transcript.txt),
[ballots](data/games/game_001/votes.json) and
[surveys](data/games/game_001/surveys.json).

Counts: 50 games, 135 rounds, 2,414 player messages, 618 ballots, 229 surveys and
224 enjoyment ratings. Sixteen games started with four humans and 34 with five.
The four access tables contain 50 games, 2,954 messages (2,414 player / 540 system),
618 votes and 229 surveys. Historical component event traces are not included
in the current release. Their removal does not change the scientific inputs/results.

See the [dataset guide](data/README.md) and [dictionary](data/DATA_DICTIONARY.md).
Local standard-library reading:

```python
import json
from pathlib import Path
with Path("data/tables/messages.jsonl").open(encoding="utf-8") as handle:
    messages = [json.loads(line) for line in handle]
```

Optional `datasets` loading, not a runtime dependency:

```python
from datasets import load_dataset
messages = load_dataset("data", name="messages", split="data")
players = messages.filter(lambda row: row["speaker_role"] != "system")
```

[dataset_infos.json](data/dataset_infos.json) supplies nullable types for the
four configs: `games`, `messages`, `votes` and `surveys`, each with split `data`.

## 6. Run paper calculations

```powershell
py -3.12 analysis/takeaway_evidence.py
py -3.12 analysis/external_stage_judge.py
py -3.12 analysis/message_behavior_analysis.py
py -3.12 analysis/analyze_message_behavior_interannotator.py
```

Each prints JSON without credentials, API calls or figures, using bundled
[inputs](analysis/inputs) and frozen decisions. They do not overwrite
[reference results](analysis/results). Four core records alone are insufficient.
Preserve original numeric word counts, cohort order and midnight-window behavior;
do not infer scientific corrections from displayed clocks. See [analysis guide](analysis/README.md)
and [reproduction limits](analysis/REPRODUCIBILITY_NOTES.md).

## Repository layout

| Directory | Purpose |
|---|---|
| `social_turing_test/` | Local game, agent, templates and packaged configuration |
| `data/` | Four-view historical dataset, schemas and dataset documentation |
| `analysis/` | Offline calculations, frozen inputs and reference results |
| `guides/` | Release and local-use details |
| `docs/` | Research website (GitHub Pages) |

`run_game.py` is the local launcher. Other tools run as modules from the repository
root: `python -m social_turing_test.prepare_config --help`,
`python -m social_turing_test.prepare_game --help`, and
`python -m social_turing_test.export_game --help`. Private runtime games,
generated configurations and sessions stay in ignored `local_data/`; manual
exports belong in ignored `local_exports/`. No container or production deployment
setup is distributed.

## Privacy, license and citation

Core text was reviewed and identifying spans withheld; the
[public redaction record](data/public_redactions.json) records categories, not
removed text. Do not re-identify or externally link participants. Review is not
anonymity proof. Author permission is conditional on not identifying individuals,
not independent consent/IRB certification.

Code: [MIT](LICENSE), preserving upstream/contributor notices. Data:
[MIT within the distributor's rights](DATA_LICENSE); privacy and third-party
rights still apply. Please cite [CITATION.cff](CITATION.cff).