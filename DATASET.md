# Social Imitation Game Dataset

This supplementary dataset contains the 50 games reported in the paper.

## Anonymization

Real participant identities and recruitment-system identifiers have been removed. No Prolific participant IDs or demographics are included. Randomized in-game pseudonyms are retained so that messages, ballots, and surveys can be linked within each game. These pseudonyms are local to a game and do not identify participants. Internal game IDs have been replaced with `game_001` through `game_050`.

Free-text content was reviewed for identifying information. Identifying spans were replaced with `[REDACTED]`; participant language was otherwise left unchanged.

## Directory structure

Each game is stored under `games/game_XXX/` and contains:

- `transcript.txt`: participant-facing chronological transcript.
- `metadata.json`: minimal game-level metadata and the AI's in-game pseudonym.
- `votes.json`: per-round ballot records.
- `surveys.json`: anonymous post-game survey responses.

## Transcripts

Timestamps are retained to preserve conversational timing. `Game-Manager` is the game system rather than a participant. Its messages mark discussion phases, voting, aggregate vote results, eliminations, and Human/AI reveals shown to participants.

Only participant-visible content is included. Prompts, model reasoning, candidate generations, review decisions, worker memory, and execution logs are not part of the release.

## Rounds

Each round consists of a timed discussion phase followed by a vote. The displayed aggregate vote result determines one elimination, with a random tie-break when necessary. The game ends when the AI is eliminated or reaches the final two players.

## AI identity

`metadata.json["ai_pseudonym"]` gives the in-game pseudonym controlled by the agent. This field is included to support agent-versus-human analyses and does not correspond to a real participant identity.

## Votes

Each ballot contains `round`, `voter`, `voted_for`, and `is_ai_voter`. AI ballots are included and explicitly flagged because they counted in the actual elimination tally and are needed to reproduce the reported survival analyses. Human-only identification analyses should exclude rows where `is_ai_voter` is `true`. Explicit abstentions are omitted from `votes.json`; no missing vote target was inferred.

Five rostered human participants sent no chat message. Their pseudonyms may therefore appear in `votes.json` without appearing as transcript speakers. Their ballots are retained because they belong to the game roster and match the participant-visible aggregate vote tally. No separate roster or identity file is included.

## Surveys

Of 234 human participants, 229 submitted a post-game survey. Participants without a submitted response are omitted from `surveys.json`; missing values are not imputed.

Two survey versions are represented. Both include ratings of the AI's human-likeness, timing, and relevance.

### `legacy_terminal_v1`

Used for five responses. Participants rated the AI's similarity to human behavior, timing of messaging, and relevance of messages from 1 (`worst`) to 5 (`best`), followed by one optional comments field. This version did not include an enjoyment rating or separate free-text fields for suspicion and additional comments.

### `web_v1`

Used for 224 responses. It contains the following fields:

- `human_likeness`: "How human-like was the AI's behavior?" (`1 = Not human-like`, `5 = Very human-like`).
- `timing`: "How natural was the AI's timing and message pacing?" (`1 = Not natural`, `5 = Very natural`).
- `relevance`: "How relevant and coherent were the AI's responses?" (`1 = Not relevant`, `5 = Very relevant`).
- `enjoyment`: "How much did you enjoy the game?" (`1 = Not at all`, `5 = Very much`).
- `suspicion_reason`: "What made you suspect (or not suspect) the AI?"
- `additional_comments`: "Any other comments or suggestions?"

## Data provenance

All files come from the 50-game evaluation cohort reported in the paper. Raw game directories were not copied directly into the release. The supplementary data excludes real-name mappings, raw configurations, recruitment records, demographics, internal agent traces, hidden memory, status files, simulations, and development/test games.
