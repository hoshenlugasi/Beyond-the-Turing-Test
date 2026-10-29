# Social Turing Test

A group chat game with one AI player. Humans win by voting out the AI; the AI
wins by reaching the final two. Code, 50 study games and offline analyses for
**Beyond the Turing Test: Evaluating AI Agents in Group Conversation**, Hoshen Lugasi, Uri Berger,
Niv Eckhaus, Itay Itzhak and Gabriel Stanovsky (2026).

[Project website](https://hoshenlugasi.github.io/Beyond-the-Turing-Test/) |
[Data on Hugging Face](https://huggingface.co/datasets/hoshen/Beyond-the-Turing-Test)

The [BGE-M3 + LDA message classifier](bge_m3_detector/README.md) includes trained
coefficients and inference code. It uses a separate environment from the game;
see its guide for installation and evaluation limitations.

## 1. Install on Windows

Install **Python 3.12** and open PowerShell in the repository root.
The commands below use the virtual environment directly; no activation is needed.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Reading data needs no installation. For standard-library-only analyses,
skip to [paper calculations](#6-run-paper-calculations).

## 2. Set credentials and start the local game

The reference setup uses **Gemini 2.5 Flash**. Set a strong admin password
and your key securely:

**Use a Google project with sufficient model quota (we recommend at least
100 requests per minute for one all-Gemini game). Also check token limits
and billing.**

```powershell
$adminSecret = Read-Host "Choose your admin password" -AsSecureString
$env:STT_ADMIN_PASSWORD = [System.Net.NetworkCredential]::new('', $adminSecret).Password
$googleSecret = Read-Host "Your Gemini API key" -AsSecureString
$env:GEMINI_API_KEY = [System.Net.NetworkCredential]::new('', $googleSecret).Password
Remove-Variable adminSecret, googleSecret
.\.venv\Scripts\python.exe run_game.py
```

An admin password is required; restart to change it. [.env.example](.env.example)
is **not automatically loaded**. Never commit secrets. Model calls and preflight
checks can cost money.

### Use other models

| Provider | Additional environment variable |
|---|---|
| OpenRouter | `OPENROUTER_API_KEY` |
| Together AI | `TOGETHER_API_KEY` |

Set the variable with the secure `Read-Host` pattern above, or enter the key
under localhost **API Keys** and click **Save Keys**.

In **Create New Game > AI Model Configuration**:
1. Set **Generator > Provider/Model** and **Reviewer > Mode/Model** separately.
2. For **Scheduler** and **Sub-agents**, set **Provider/Model**, or choose
   **(same as generator)** with an empty **Model**.
3. Check every role before **Open Lobby**: changing Generator does not change
   the other role selections. Use exact model IDs your account can access;
   menu entries do not guarantee compatibility.

For old saved configs, recreate roles with explicit providers if routing is wrong.

Internet Context always uses direct **Gemini 2.5 Flash + Google Search**.
It needs working Google authentication even when other roles use OpenRouter or
Together; changing their models does not replace it (Google/Vertex options below).

Check the **Agent** tab: missing Google auth means configure Google; **402** means
check the named provider's billing, credits or spending limit; **429** means check
request and token rate limits.

Across the 50 reference games, one AI recorded about **29 calls/minute** during
discussion and a peak of **74 in 60 seconds**, excluding the evaluator. Retries,
failed calls and setup/search may add load. Leave headroom for token limits and
quotas shared across games; this is not a guaranteed maximum.

<details>
<summary>Other Google authentication and operating systems</summary>

`GOOGLE_API_KEY` can replace `GEMINI_API_KEY`. For Vertex AI, leave both keys unset,
set `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`, and configure Application
Default Credentials outside the repository.

Linux/macOS are not validated. Use `python3.12 -m venv .venv`, then
`. .venv/bin/activate` and `python -m pip install -r requirements.txt`.
Securely set the same environment variables before `python run_game.py`.
Requirements list direct dependencies, not a full lockfile.

</details>

## 3. Play

1. Open <http://localhost:5000> and log into the admin panel with your password.
2. Create a game with the default agent settings.
3. Have **at least two humans** join, wait for AI readiness, then press **START**.
4. Chat and vote until the game ends; players can then submit surveys.

The default is localhost (`127.0.0.1`). For other devices on a **trusted LAN**,
restart with `.\.venv\Scripts\python.exe run_game.py --lan` and share your LAN
address instead of localhost. **Do not expose the service to the public Internet.**
Changing models, settings or the roster is not a replay of the study.

## 4. Save and export

New games are saved automatically under `local_data/games/`. Keep these raw files
private. After a game is complete, the manager and AI have stopped, and the surveys
you want have arrived, export it to a **fresh** destination:

```powershell
.\.venv\Scripts\python.exe -m social_turing_test.export_game --game-dir local_data/games/real/0001 --output local_exports/my_game --export-id local_game_001 --kind local_game
```

Replace the example source with your completed game directory. The offline exporter
leaves the source unchanged and never overwrites an existing destination.
This is a **private diagnostic export**, not an addition to the published study dataset.

**Review all exported text and confirm consent and redistribution rights before
sharing. Automatic masking does not authorize publication.** Keep exports in
the ignored `local_exports/` directory.

## 5. Read study data

The **50 study games** are in [data/games](data/games), each with `metadata.json`,
`transcript.txt`, `votes.json` and `surveys.json`. Historical component event traces
are not included. The [dataset guide](data/README.md) covers field definitions,
four table views, loading examples, missing values and privacy limitations.

## 6. Run paper calculations

```powershell
py -3.12 analysis/takeaway_evidence.py
py -3.12 analysis/external_stage_judge.py
py -3.12 analysis/message_behavior_analysis.py
py -3.12 analysis/analyze_message_behavior_interannotator.py
```

These standard-library-only commands need no credentials or model calls. Each
prints JSON from the bundled [inputs](analysis/inputs) without overwriting the
frozen [reference outputs](analysis/results). See the [analysis guide](analysis/README.md)
for cohort, original word-count, midnight-window and sampling limitations.

## Repository layout

| Directory | Purpose |
|---|---|
| `social_turing_test/` | Local game, agent, templates and configuration |
| `data/` | Study games and dataset guide |
| `analysis/` | Paper calculations, frozen inputs and reference outputs |
| `bge_m3_detector/` | Standalone trained message classifier |
| `docs/` | Research website (GitHub Pages) |

## Privacy, license and citation

Identifying details were redacted without changing ratings, ballots or outcomes.
Review does not guarantee anonymity or establish consent/IRB approval. Do not
re-identify participants or link them to outside identities.
Release permission remains conditional on not identifying individuals.

Code: [MIT](LICENSE), preserving upstream/contributor notices. Data:
[MIT within the distributor's rights](DATA_LICENSE); privacy and third-party
rights still apply. Please cite [CITATION.cff](CITATION.cff).