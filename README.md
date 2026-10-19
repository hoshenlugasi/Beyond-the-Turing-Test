---
configs: [{config_name: games, default: true, data_files: [{split: data, path: "tables/games.jsonl"}]}, {config_name: messages, data_files: [{split: data, path: "tables/messages.jsonl"}]}, {config_name: votes, data_files: [{split: data, path: "tables/votes.jsonl"}]}, {config_name: surveys, data_files: [{split: data, path: "tables/surveys.jsonl"}]}, {config_name: events, data_files: [{split: data, path: "games/*/events/timeline.jsonl"}]}]
---

# Social Turing Test

A group chat game with one AI player. Humans win by voting out the AI; the AI
wins by reaching the final two. Includes one playable agent, 50 study games,
reviewed component timelines and offline calculations for **Beyond the Turing
Test: Evaluating AI Agents in Group Conversation**, Hoshen Lugasi, Uri Berger,
Niv Eckhaus, Itay Itzhak and Gabriel Stanovsky (2026).

Code: [GitHub](https://github.com/hoshenlugasi/Beyond-the-Turing-Test).
Data mirror: [Hugging Face](https://huggingface.co/datasets/hoshen/Beyond-the-Turing-Test).
[Project website](https://hoshenlugasi.github.io/Beyond-the-Turing-Test/).
See [release notes](RELEASE_NOTES.md) for this dataset revision's mirror-refresh status.

## 1. Install on Windows

Install Python 3.12 and open PowerShell in this package's root. All commands and
loading examples below use that working directory; activation is unnecessary.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Reading files needs no credentials or installation. Offline paper calculations
need only the standard library; skip to section 6 for those.

## 2. Set credentials and start the server

In the same terminal, supply your own strong admin password and Gemini API key
without putting their values in command history:

```powershell
$adminSecret = Read-Host "Choose your admin password" -AsSecureString
$env:STT_ADMIN_PASSWORD = [System.Net.NetworkCredential]::new('', $adminSecret).Password
$googleSecret = Read-Host "Your Gemini API key" -AsSecureString
$env:GEMINI_API_KEY = [System.Net.NetworkCredential]::new('', $googleSecret).Password
Remove-Variable adminSecret, googleSecret
.\.venv\Scripts\python.exe web_player.py
```

Missing password disables admin access; restart after changing it.
[.env.example](.env.example) is a reference, **not automatically loaded**. Never
commit secrets. Live requests incur costs; check your account's Gemini 2.5 Flash
entitlement. Availability is conditional; replacement models are new conditions.
See [other ways to run](RELEASE_NOTES.md#other-ways-to-run).

## 3. Play

1. Open <http://localhost:5000> and log into the admin panel with your password.
2. Create a game with default agent settings and share the lobby link. Other
   devices need your reachable LAN address rather than localhost.
3. Have **at least two humans** join, wait for AI readiness, then press **START**.
   There is no fixed five-human cap or automatic start.
4. Chat and vote until the game ends; players can then submit surveys.

The server listens on all interfaces. Use a trusted network, not direct public
Internet exposure. Defaults retain the paper's orchestration, Swiss student
persona, 2.5-minute discussions, four candidates and verbatim reviewer selection.
Selection is not delivery. Scheduler, Generator and Reviewer work with Internet
Context, Round Summarizer, Persona Monitor, Social State Tracker, Behavior Adapter
and Repetition Monitor. The Reviewer selects a candidate verbatim, without
rewriting. Changing models, settings or roster is not a replay of the study.

## 4. Save and export

Full raw saving is automatic under games/real/&lt;id&gt; or games/test/&lt;id&gt;:
configuration, chats, votes, events, logs and memory traces. Keep raw files local.
After the manager and AI stop and desired surveys arrive, export manually:

```powershell
.\.venv\Scripts\python.exe export_game.py --game-dir games/real/0001 --output local_exports/my_game --export-id local_game_001 --kind local_game
```

Replace the source with your completed directory. Scripted seats use
`--kind scripted_test`. [export_game.py](export_game.py) needs only the standard
library, makes no network calls, leaves source unchanged and publishes atomically
to a fresh destination without overwriting. Its 14 files comprise four core
records, three nested event files, four root tables, loading schemas, a README
and export manifest.
It never extends the historical cohort. Export late surveys to another fresh
output. Use the ignored local_exports directory inside this checkout.

**Every new export is privacy-review-PENDING and not publication-authorized.**
Review all text, including rejected candidates, reviewer assessments and worker
observations, and check consent/rights. Automatic masking is not clearance.
Missing surveys remain missing; unfinished, inconsistent or unsupported drop/rescue
records are refused. Future timing uses verified logger segments, not START.
There is no public segment-index column; ambiguous unpaired timing is null and
restart intervals must not be stitched. See [clock details](DATASET.md#clocks-order-and-scientific-limits).

## 5. Read study data

Every [game](games/game_001) has seven files: [metadata](games/game_001/metadata.json),
[transcript](games/game_001/transcript.txt), [ballots](games/game_001/votes.json),
[surveys](games/game_001/surveys.json), [timeline JSONL](games/game_001/events/timeline.jsonl),
[full text rendering](games/game_001/events/timeline.txt) and
[coverage manifest](games/game_001/events/manifest.json).

Counts: 50 games, 135 rounds, 2,414 player messages, 618 ballots, 229 surveys and
224 enjoyment ratings. Sixteen games started with four humans and 34 with five.
Root tables have 50 games, 2,954 messages (2,414 player / 540 system), 618 votes
and 229 surveys. Timelines retain 53,360 of 54,918 source events, excluding 1,558
diagnostic records. All six paper workers are represented, not unrestricted logs.
Historical timing is one-second precision since the **first source event, not
START**; no chat-clock merge or timezone correction is asserted.

See [DATASET.md](DATASET.md), [DATA_DICTIONARY.md](DATA_DICTIONARY.md) and
[dataset_manifest.json](dataset_manifest.json). Local standard-library reading:

```python
import json
from pathlib import Path
with Path("games/game_001/events/timeline.jsonl").open(encoding="utf-8") as handle:
    events = [json.loads(line) for line in handle if line.strip()]
```

Optional `datasets` loading, not a runtime dependency:

```python
from datasets import load_dataset
messages = load_dataset(".", name="messages", split="data")
events = load_dataset(".", name="events", split="data")
players = messages.filter(lambda row: row["speaker_role"] != "system")
```

[dataset_infos.json](dataset_infos.json) supplies explicit nullable nested types;
do not infer event types from the first shard, where some fields are all null.
The same metadata and five loading configurations are included in new exports.

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
event clocks are not scientific corrections. See [analysis guide](analysis/README.md)
and [reproduction limits](analysis/REPRODUCIBILITY_NOTES.md).

## 7. Optional checks and validation scope

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest tests -q
```

The final outside-ZIP validation report identifies the exact artifacts checked:
12 offline scientific reproductions, tests in a new external Windows virtual
environment, server checks and local dataset loading. Its evidence applies only
to those artifacts, not automatically to edited files or rebuilt packages. No new
model-backed game, scientific-quality validation or remote Hub viewer validation
is claimed. See [release notes](RELEASE_NOTES.md).

## Privacy, license and citation

A first pass reviewed 7,408 unique text values; 52 were withheld at 135 payload
occurrences. A further 22 withheld values occur at 111 payload locations.
Contextual checks were selective, not anonymity proof. Do not
re-identify or externally link participants. Author permission is conditional on
not identifying individuals, not independent consent/IRB certification.

Code: [MIT](LICENSE), preserving upstream/contributor notices. Data:
[MIT within the distributor's rights](DATA_LICENSE); privacy and third-party
rights still apply. Please cite [CITATION.cff](CITATION.cff).