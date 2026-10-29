"""Inference for the released BGE-M3 + LDA message classifier."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit


MODEL_DIR = Path(__file__).resolve().parent


def preprocess(text: str, player_names: list[str] | tuple[str, ...] = ()) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    if not isinstance(player_names, (list, tuple)) or any(
        not isinstance(name, str) or not name.strip() for name in player_names
    ):
        raise ValueError("player_names must be a list of non-empty strings")
    text = text.strip()
    for name in sorted(player_names, key=len, reverse=True):
        text = re.sub(r"(?i)\b" + re.escape(name) + r"\b", "<name>", text)
    return text.lower()


class MessageDetector:
    def __init__(self, model_dir: Path = MODEL_DIR, device: str = "cpu"):
        model_dir = Path(model_dir)
        self.card = json.loads((model_dir / "model_card.json").read_text(encoding="utf-8"))
        if self.card["format_version"] != 1:
            raise ValueError("Unsupported detector format version")
        weights = model_dir / "classifier.npz"
        actual_hash = hashlib.sha256(weights.read_bytes()).hexdigest()
        if actual_hash != self.card["classifier"]["weights_sha256"]:
            raise ValueError("Classifier checksum differs from model_card.json")
        with np.load(weights, allow_pickle=False) as arrays:
            self.coef = arrays["coef"]
            self.intercept = arrays["intercept"]
            self.classes = arrays["classes"]
        dimension = self.card["classifier"]["embedding_dimensions"]
        if (
            self.coef.shape != (1, dimension)
            or self.intercept.shape != (1,)
            or not np.array_equal(self.classes, [0, 1])
            or not np.isfinite(self.coef).all()
            or not np.isfinite(self.intercept).all()
        ):
            raise ValueError("Invalid binary classifier parameters")
        self.device = device
        self._encoder = None

    def predict_embeddings(self, embeddings: np.ndarray) -> list[dict]:
        matrix = np.asarray(embeddings)
        if (
            matrix.ndim != 2
            or matrix.shape[1] != self.coef.shape[1]
            or not np.isfinite(matrix).all()
        ):
            raise ValueError(f"Expected finite embeddings with shape (N, {self.coef.shape[1]})")
        scores = (matrix @ self.coef.T + self.intercept).ravel()
        probabilities = expit(scores)
        return [
            {
                "label": "agent" if score > 0 else "human",
                "agent_probability": float(probability),
                "decision_score": float(score),
            }
            for score, probability in zip(scores, probabilities)
        ]

    def predict_records(self, records: list[dict]) -> list[dict]:
        if not isinstance(records, list) or not records:
            raise ValueError("Provide at least one message record")
        texts = []
        for index, record in enumerate(records, start=1):
            if not isinstance(record, dict) or "text" not in record:
                raise ValueError(f"Record {index} must be an object containing text")
            try:
                texts.append(preprocess(record["text"], record.get("player_names", [])))
            except ValueError as exc:
                raise ValueError(f"Record {index}: {exc}") from exc
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            encoder = self.card["encoder"]
            self._encoder = SentenceTransformer(
                encoder["repository"], revision=encoder["revision"], device=self.device
            )
        embeddings = self._encoder.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=self.card["encoder"]["normalize_embeddings"],
        )
        return self.predict_embeddings(embeddings)


def read_jsonl(stream, default_names: list[str]) -> list[dict]:
    records = []
    for number, line in enumerate(stream, start=1):
        if not line.strip():
            raise ValueError(f"Line {number}: blank lines are not message records")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Line {number}: invalid JSON ({exc.msg})") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Line {number}: expected a JSON object")
        record.setdefault("player_names", default_names)
        records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", action="append", help="Message text; may be repeated")
    source.add_argument("--input", type=Path, help="JSONL file, or - for standard input")
    parser.add_argument("--names", nargs="*", default=[], help="All conversation pseudonyms")
    parser.add_argument("--device", default="cpu", help="Encoder device (default: cpu)")
    args = parser.parse_args()
    try:
        if args.text is not None:
            records = [{"text": text, "player_names": args.names} for text in args.text]
        elif str(args.input) == "-":
            records = read_jsonl(sys.stdin, args.names)
        else:
            with args.input.open(encoding="utf-8") as stream:
                records = read_jsonl(stream, args.names)
        results = MessageDetector(device=args.device).predict_records(records)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    for result in results:
        print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
