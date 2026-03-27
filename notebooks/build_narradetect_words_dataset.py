from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from pathlib import Path

import pandas as pd

# Basic sentence splitter for plain text.
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Extract words while preserving internal apostrophes, e.g., don't.
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]


def sentence_words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def build_words_dataset(input_csv: Path, output_csv: Path, max_sentences_per_id: int) -> None:
    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header row.")

        required = {"ID", "Genre", "Text", "Label"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Input CSV is missing required columns: {', '.join(sorted(missing))}"
            )

        rows: list[dict[str, str]] = []
        sentence_counts: dict[str, int] = {}
        for row in reader:
            text = row.get("Text", "") or ""
            row_id = row.get("ID", "")

            current_count = sentence_counts.get(row_id, 0)
            if current_count >= max_sentences_per_id:
                continue

            sentences = split_sentences(text)
            remaining = max_sentences_per_id - current_count
            for sentence in sentences[:remaining]:
                rows.append(
                    {
                        "document_name": row.get("Genre", ""),
                        "id": row_id,
                        "words": json.dumps(sentence_words(sentence), ensure_ascii=False),
                    }
                )

            sentence_counts[row_id] = current_count + min(len(sentences), remaining)

    # Convert to DataFrame for resampling
    df = pd.DataFrame(rows)
    
    # Balance genres (document_name) by resampling to match the minimum genre count
    genre_counts = df['document_name'].value_counts()
    min_count = genre_counts.min()
    
    # Resample each genre to have equal representation
    balanced_dfs = []
    for genre in df['document_name'].unique():
        genre_df = df[df['document_name'] == genre]
        # Sample with replacement if the genre has fewer samples than min_count
        # Sample without replacement if it has more
        resampled = genre_df.sample(n=min_count, replace=(len(genre_df) < min_count), random_state=42)
        balanced_dfs.append(resampled)
    
    df_balanced = pd.concat(balanced_dfs, ignore_index=True)
    
    # Convert words from JSON strings to actual lists
    df_balanced['words'] = df_balanced['words'].apply(ast.literal_eval)
    
    # Save to CSV
    df_balanced.to_csv(output_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert NarraDetect text into one sentence per row with token lists in a words column."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("NarraDetect_Large.csv"),
        help="Input NarraDetect CSV path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("NarraDetect.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--max-sentences-per-id",
        type=int,
        default=100,
        help="Maximum number of sentence rows emitted for each ID.",
    )

    args = parser.parse_args()
    build_words_dataset(args.input, args.output, args.max_sentences_per_id)
