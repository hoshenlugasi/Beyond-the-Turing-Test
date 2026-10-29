# Released BGE-M3 message classifier

This is the **BGE-M3 + linear discriminant analysis (LDA) classifier with
macro-F1 = 0.937093034266331 (0.937 rounded)** described in *Beyond the Turing
Test: Evaluating AI Agents in Group Conversation*. It is a standalone add-on:
no game server, credentials, or training data are required for inference.

**The 0.937 score is in-sample:** the default scikit-learn
`LinearDiscriminantAnalysis()` was fitted and evaluated on the same 2,414
messages, following the fit-and-predict protocol used in the public LLMafia
code of Eckhaus et al. (2025). It is not a held-out test score and must not be
presented as performance on new conversations.

## What is released

- `classifier.npz`: the trained LDA coefficients, intercept, and class labels.
  This is a data-only NumPy archive, loaded with `allow_pickle=False`.
- `detector.py`: preprocessing, pinned BGE-M3 encoding, and inference.
- `model_card.json`: exact model revision, training settings, provenance,
  evaluation, and the classifier checksum.
- `requirements.txt`, this README, and the MIT license.

The original training script retained metrics rather than the fitted object.
This release refits the same classifier on the original cohort, verifies the
original macro-F1, and exports its linear decision rule. Exported predictions
and posterior probabilities are checked against scikit-learn on every
training message. See `model_card.json` for the verification results.

**BGE-M3 itself was not fine-tuned.** The trained component is the LDA layer
on top of the frozen `BAAI/bge-m3` encoder. The encoder weights are downloaded
separately from Hugging Face on first use (approximately 2.3 GB) and cached.
The pinned encoder revision is
`5617a9f61b028005a4858fdac845db406aefb181`.

## Install

Python 3.12 is the reference runtime. From this directory:

```shell
python -m venv .venv
```

Activate it (`.venv\Scripts\activate` on Windows, or
`source .venv/bin/activate` on Linux/macOS), then:

```shell
python -m pip install -r requirements.txt
```

PyTorch is installed as a dependency of sentence-transformers. CPU inference
is the default; `--device cuda` can use a compatible local GPU installation.
The game and classifier environments are independent: do not replace the
game's dependency file with this add-on's requirements.

## Classify messages

```shell
python detector.py --text "yeah fair enough" --names Ashton Elliot Jamie
```

Supply **all player pseudonyms for that conversation**, including silent
players, using `--names`. Names are replaced with `<name>` using
case-insensitive word-boundary matching, longest names first. Text is
stripped and lowercased, and embeddings are normalized, exactly as in the
original training preprocessing. No persona-term masking is applied.

Without names the tool still runs, but cannot mask unknown names. Do not
include timestamps, speaker prefixes, system messages, or whole transcripts
as message text: the classifier was trained on individual delivered messages.
Names supplied to the API should be the exact roster strings.

For several messages, repeat `--text`, or use a UTF-8 JSONL file with one
object per line:

```json
{"text": "Elliot what do you think?", "player_names": ["Ashton", "Elliot", "Jamie"]}
{"text": "not sure yet", "player_names": ["Ashton", "Elliot", "Jamie"]}
```

```shell
python detector.py --input messages.jsonl
```

`--input -` reads standard input. Per-record `player_names` overrides the
shared `--names` roster. Output is one JSON object per input message, in
order, with:

- `label`: `human` or `agent`.
- `agent_probability`: the LDA posterior for the agent class; **not a
  separately calibrated confidence or a general probability of AI authorship**.
- `decision_score`: the linear score; positive means `agent`, otherwise
  `human`.

Text is not echoed in the output. Input validation errors produce a nonzero
exit status rather than a prediction.

### Python API

With this directory on your import path:

```python
from detector import MessageDetector

detector = MessageDetector()
result = detector.predict_records([
    {"text": "yeah fair enough", "player_names": ["Ashton", "Elliot", "Jamie"]}
])
print(result)
```

For already normalized embeddings produced by the exact pinned encoder,
`predict_embeddings(matrix)` applies the exported linear rule without
loading the encoder.

## Training scope and limitations

The original historical post-W15 Flash cohort contains 50 games and 2,414
delivered messages: 1,895 human and 519 agent messages. The cohort includes
34 five-human and 16 four-human games. It does not pool Pro or custom-model
conditions. No additional training examples, balancing, shrinkage,
threshold tuning, or new detector variants were introduced for this release.

The result characterizes one specific agent condition, not arbitrary AI
systems. Short messages, new topics, different populations, or different
agents can behave differently. Do not use this detector alone to verify
human identity, moderate users, or impose sanctions.

No raw messages, participant identifiers, private game logs, embeddings,
or credentials are included in this add-on. The public dataset has
privacy redactions; refitting on that dataset is not claimed to recover
the exact original weights or score. This release intentionally provides
the fitted classifier so users do not need to retrain it.

## License and references

This add-on is distributed under the MIT license in `LICENSE`. BGE-M3
is obtained from its own repository and remains subject to its upstream
license and model documentation.

- [Project code and data](https://github.com/hoshenlugasi/Beyond-the-Turing-Test)
- [BGE-M3 model and license](https://huggingface.co/BAAI/bge-m3)
- [BGE M3-Embedding, Chen et al. (2024)](https://arxiv.org/abs/2402.03216)
- [LLMafia reference implementation](https://github.com/niveck/LLMafia)
