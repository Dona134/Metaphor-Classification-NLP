from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


# Basic sentence splitter that works reasonably well for long-form plain text.
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
# Extract words while preserving internal apostrophes, e.g., don't.
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]
    return sentences


def sentence_words(sentence: str) -> list[str]:
    return WORD_RE.findall(sentence)


def iter_books(root: Path) -> Iterable[tuple[str, str, Path]]:
    for author_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for book_path in sorted(author_dir.glob("*.txt")):
            yield author_dir.name, book_path.stem, book_path


def build_dataset(root: Path, output_csv: Path, window_size: int, seed: int) -> None:
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []

    for author, document_name, book_path in iter_books(root):
        text = book_path.read_text(encoding="utf-8", errors="ignore")
        sentences = split_sentences(text)

        if not sentences:
            continue

        if len(sentences) >= window_size:
            start = rng.randint(0, len(sentences) - window_size)
            selected = sentences[start : start + window_size]
        else:
            # If a document has fewer than the requested window size, keep all.
            selected = sentences

        for sentence in selected:
            rows.append(
                {
                    "document_name": document_name,
                    "author": author,
                    "words": json.dumps(sentence_words(sentence), ensure_ascii=False),
                }
            )

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["document_name", "author", "words"])
        writer.writeheader()
        writer.writerows(rows)

    df = pd.read_csv(output_csv)
    df['words'] = df['words'].apply(ast.literal_eval)
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a sentence-level dataset from books by sampling consecutive windows."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Root folder containing author subfolders with .txt books.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sentence_dataset.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=100,
        help="Number of consecutive sentences sampled per document.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )

    args = parser.parse_args()
    build_dataset(args.root, args.output, args.window_size, args.seed)
