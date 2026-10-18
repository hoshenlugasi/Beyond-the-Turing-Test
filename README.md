# Social Turing Test

A chat game where humans try to find one AI player. After each discussion,
players vote someone out. Humans win by removing the AI; the AI wins by reaching
the final two.

This repository includes the V6 agent, a local game, the 50 released study games,
and code for the paper's numerical analyses. It does not include the private
recruitment or deployment system.

Paper: **Beyond the Turing Test: Evaluating AI Agents in Group Conversation**,
Hoshen Lugasi, Uri Berger, Niv Eckhaus, Itay Itzhak and Gabriel Stanovsky (2026).

## 1. Install

Install Python 3.12, download or clone this repository, and open a terminal in its
root folder. On Windows, run these commands in PowerShell. No activation is needed:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Only want to read the data or run the offline analyses? Skip the game dependencies
and credentials. Go straight to [the data](#5-the-50-study-games) or
[the analysis commands](#6-run-the-paper-calculations).

## 2. Set credentials and start the server

You need your own Gemini API key and a strong admin password. In the same
PowerShell terminal, run:

```powershell
$adminSecret = Read-Host "Choose your admin password" -AsSecureString
$env:STT_ADMIN_PASSWORD = [System.Net.NetworkCredential]::new('', $adminSecret).Password
$googleSecret = Read-Host "Your Gemini API key" -AsSecureString
$env:GEMINI_API_KEY = [System.Net.NetworkCredential]::new('', $googleSecret).Password
Remove-Variable adminSecret, googleSecret
.\.venv\Scripts\python.exe web_player.py
```

The password is read when the server starts. Without it, the admin panel is
disabled; restart the server after changing it. [.env.example](.env.example)
lists the settings, but is **not loaded automatically**. Never commit real keys.

Live games use paid external services. The default model is **Gemini 2.5 Flash**;
check that your account can access it before playing. Availability depends on the
provider. There is no automatic replacement model; changing models creates a new
experiment. [Other credential options and Linux/Docker instructions](RELEASE_NOTES.md#other-ways-to-run)
are available, but Linux/macOS and Docker have not been tested for this release.

## 3. Play

1. Open <http://localhost:5000>, enter the admin panel, and log in with your password.
2. Create a game using the default V6 settings. Share its lobby link with players.
   Players on another device need your computer's reachable LAN address instead
   of `localhost`.
3. Have **at least two humans** join. There is no fixed player cap or automatic
   start. Wait for the AI to be ready, then press **START**.
4. Chat, vote, and continue until the AI is eliminated or reaches the final two.
   Players can then submit the post-game survey.

The server listens on all network interfaces. Use a trusted network; do not put
this development server or its admin panel directly on the public Internet.

The defaults are V6 legacy orchestration, a Swiss student persona, 2.5-minute
discussions and four generated candidates. The reviewer selects one candidate
without rewriting it. Selection does not guarantee delivery: later checks can
still suppress it. The public runtime has six background workers and **no Live
Evaluator**. See [release details](RELEASE_NOTES.md) for the full settings and
the difference from the historical research runtime.

## 4. Save and export a new game

**The runtime provides two storage layers:**

- **Full local saving is automatic.** Configurations, chats, votes, events, logs
  and memory traces stay under `games/real/v6/` or `games/test/v6/`. These files
  are for running/debugging the game, not for public upload.
- **A separate export is manual.** After the game manager and AI have stopped
  and the surveys you want have arrived, run [export_game.py](export_game.py).

For example, if your completed game is in `games/real/v6/0001`:

```powershell
.\.venv\Scripts\python.exe export_game.py --game-dir games/real/v6/0001 --output local_exports/my_game --export-id local_game_001 --kind local_game
```

Replace the source path with your actual game directory. Use a fresh destination
and export ID each time. For scripted test seats, use `--kind scripted_test`.
The example source is not an included game.

The output contains four core files under `games/local_game_001/` and selected
events under `internal_outputs/games/local_game_001/events/`, plus a README and
manifests. The command needs no API key, makes no network requests, and leaves
the full source files unchanged. It never adds games to the historical cohort.
Inside this checkout, use the ignored `local_exports` folder.

**New exports still need privacy review before sharing.** Automatic masking is
not enough: check transcripts, surveys, rejected candidates and reviewer/worker
text, and confirm consent and redistribution rights. Every new export is marked
privacy-review-pending and not authorized for publication. Missing surveys stay
missing. Existing output is never overwritten. Unfinished or inconsistent games,
including unsupported drop/rescue records, are rejected rather than guessed.
To include a late survey, export again to a new destination.

## 5. The 50 study games

Start with [game_001](games/game_001). Every released game has four core files:

| File | What it contains |
|---|---|
| [metadata.json](games/game_001/metadata.json) | AI pseudonym, starting human count, rounds and outcome |
| [transcript.txt](games/game_001/transcript.txt) | Player-visible conversation and game notices |
| [votes.json](games/game_001/votes.json) | Cast ballots, with AI votes marked |
| [surveys.json](games/game_001/surveys.json) | Submitted ratings and comments |

There is also a **separate filtered event export** under
[internal_outputs](internal_outputs/README.md): candidates, short reviewer
assessments, selections, delivery checks and selected worker updates. These are
not all raw logs. Full prompts, full memory snapshots, credentials, recruitment
identifiers and private ID mappings are excluded. Rejected candidates are not
public chat messages.

The dataset has 50 games, 135 rounds, 2,414 player messages, 618 cast ballots and
229 surveys. Sixteen games began with four humans and 34 with five. Keep this
difference in analyses; do not combine these games with Pro, custom or test games.
Five surveys are missing, and the five legacy surveys have no enjoyment rating;
missing values are not zeros. Demographic data is aggregate-only.

The reviewed data is published in this GitHub repository and on
[Hugging Face](https://huggingface.co/datasets/hoshen/Beyond-the-Turing-Test).
Shared dataset files are byte-identical; the repositories' root README and
license files differ. See the
[project website](https://hoshenlugasi.github.io/Beyond-the-Turing-Test/) for an overview.

Text was reviewed and identifying spans removed; [public_redactions.json](public_redactions.json)
records the additional public redactions. Review reduces risk, but cannot
guarantee that no person could ever be identified from context. Do not attempt
re-identification. [DATASET.md](DATASET.md) describes **only the four-file core**;
its exclusion of internal traces does not refer to the separate filtered events.

## 6. Run the paper calculations

Use Python 3.12. These commands need only the standard library: no dependency
installation, credentials, model calls or paid services.

```powershell
py -3.12 analysis/takeaway_evidence.py
py -3.12 analysis/external_stage_judge.py
py -3.12 analysis/message_behavior_analysis.py
py -3.12 analysis/analyze_message_behavior_interannotator.py
```

Each command **prints JSON**. The checked-in [reference results](analysis/results)
are not overwritten. The scripts use the included [calculation inputs](analysis/inputs)
and frozen model judgments; they do not call judges again or create figures.
The four core game files alone are not sufficient. Both release packages already
include the same game data and inputs, so no copying is needed with this checkout.

Use the supplied original numeric word counts, not counts recomputed from redacted
text. One game crosses midnight; the calculations preserve the original window
behavior. See [analysis/README.md](analysis/README.md) for methods, inputs and limits.

## 7. Run the tests

After installing the game dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest tests -q
```

Validation used Windows Python 3.12.10 and fresh extracted packages. It covered
the offline calculations, code tests and one real Flash game with two scripted
human-seat clients. That is a functional check, not a study with real participants
or proof of reliable web grounding. Details are in [RELEASE_NOTES.md](RELEASE_NOTES.md).

## License and citation

Code: [MIT](LICENSE), preserving Niv Eckhaus's 2025 notice and adding Hoshen
Lugasi's 2026 contributions. Data: [MIT within the distributor's rights](DATA_LICENSE).
Participant privacy and other third-party rights still apply; a license is not
consent for unrelated personal-data use. The author's release permission is
conditional on not identifying individual participants.

Please cite the paper using [CITATION.cff](CITATION.cff).