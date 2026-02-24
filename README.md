# Metaphor Classification with NLP

A token-level metaphor detection system that uses **XLM-RoBERTa** (a multilingual transformer model) to classify individual words as metaphorical or literal in English and Russian text. The project was developed as part of a research investigation into cross-lingual metaphor identification, combining fine-tuned pre-trained language models with POS-aware analysis and class-imbalance handling.

---

## Table of Contents

1. [Key Features](#key-features)
2. [Repository Structure](#repository-structure)
3. [Setup / Installation](#setup--installation)
4. [Usage](#usage)
5. [Data](#data)
6. [Methodology / Approach](#methodology--approach)
7. [Results](#results)
8. [Reproducibility Notes](#reproducibility-notes)
9. [License](#license)
10. [Contributing](#contributing)
11. [Citation](#citation)

---

## Key Features

- **Token-level metaphor classification** — labels each word in a sentence as `Metaphor` or `Literal`.
- **Multilingual (EN + RU)** — trained on the English VUAMC corpus and the Russian LCC metaphor corpus.
- **XLM-RoBERTa backbone** — leverages `xlm-roberta-base` for robust cross-lingual representations.
- **Class-imbalance mitigation** — weighted cross-entropy loss with configurable class weights (EN: `[0.232, 1.384]`; RU: `[0.53, 10.07]`).
- **POS-aware analysis** — per-POS F1 breakdown (verb, noun, adj, adv) and optional POS-stratified dataset balancing.
- **Five experiment configurations** — `EN`, `EN_BALANCED_POS`, `RU_ZERO`, `RU_HEAD`, `RU_FULL` (zero-shot / head-only / full fine-tuning cross-lingual transfer).
- **Visualisations** — confusion matrices and F1-vs-metaphor-sample scatter plots per POS.

---

## Repository Structure

```
Metaphor-Classification-NLP/
├── data/
│   ├── raw/                        # Original, unprocessed datasets
│   │   ├── 0000.parquet            # VUAMC English corpus (Parquet)
│   │   ├── Datasets_ACL2014.xlsx   # Additional ACL 2014 metaphor datasets
│   │   └── ru_large.zip            # Russian LCC metaphor corpus (XML, zipped)
│   └── processed/                  # Pre-tokenised JSONL files (pipeline output)
│       ├── vuamc_token.jsonl       # English VUAMC — 7 850 sentences
│       └── ru_large_token.jsonl    # Russian LCC — 11 878 sentences
├── notebooks/
│   ├── vuamc_token.ipynb           # EN preprocessing: Parquet → JSONL
│   ├── ru_large_token.ipynb        # RU preprocessing: XML → JSONL + spaCy POS
│   ├── main.ipynb                  # Earlier experiment notebook
│   ├── main_upd.ipynb              # Intermediate updated notebook
│   └── Main_fin.ipynb              # Final, complete experiment notebook ★
├── plots/
│   ├── confusion_matrix.png
│   ├── f1_pos_1.png
│   ├── f1_pos_2.png
│   └── sample size of pos vs. f1 score.png
├── MetaphorClassification_AidanaAkkaziyeva.pdf   # Project report / paper
└── README.md
```

The primary entry point is **`notebooks/Main_fin.ipynb`**.

---

## Setup / Installation

### Environment

The notebooks are designed to run on **Google Colab** (GPU runtime recommended). All heavy training was performed on a Colab T4 GPU.

> **Local execution** is also possible with the dependencies below, but a CUDA-capable GPU is strongly recommended for training.

### Python Dependencies

Install with `pip`:

```bash
pip install torch transformers datasets scikit-learn pandas numpy \
            matplotlib seaborn spacy openpyxl pyarrow
# For Russian POS tagging (ru_large_token.ipynb only)
python -m spacy download ru_core_news_lg
```

| Package | Purpose |
|---------|---------|
| `torch` | Deep learning backend |
| `transformers` | XLM-RoBERTa model + `Trainer` API |
| `scikit-learn` | Metrics, class-weight computation, train/test split |
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn` | Visualisation |
| `spacy` + `ru_core_news_lg` | Russian POS tagging |
| `openpyxl`, `pyarrow` | Reading Excel / Parquet raw files |

### Google Drive (Colab)

All notebooks mount Google Drive and expect the project to live at:

```
/content/drive/MyDrive/Metaphor-Classification-NLP/
```

Clone or copy the repository to that path before running the notebooks.

---

## Usage

### Step 1 — Preprocessing (optional, processed files included)

Run the preprocessing notebooks once to (re-)generate the JSONL files:

| Notebook | Input | Output |
|----------|-------|--------|
| `notebooks/vuamc_token.ipynb` | `data/raw/0000.parquet` | `data/processed/vuamc_token.jsonl` |
| `notebooks/ru_large_token.ipynb` | `data/raw/ru_large.zip` (extract to `ru_large.xml`) | `data/processed/ru_large_token.jsonl` |

### Step 2 — Training & Evaluation

Open **`notebooks/Main_fin.ipynb`** in Colab (or locally) and:

1. Set the experiment in **Config** cell:
   ```python
   EXPERIMENT = "RU_FULL"   # choose one below
   ```

   | Value | Description |
   |-------|-------------|
   | `"EN"` | English-only fine-tuning on VUAMC (3 000 samples) |
   | `"EN_BALANCED_POS"` | English fine-tuning with POS-stratified balancing |
   | `"RU_ZERO"` | Russian zero-shot (EN-trained model, no RU fine-tuning) |
   | `"RU_HEAD"` | Russian — fine-tune classification head only |
   | `"RU_FULL"` | Russian — full model fine-tuning |

2. Run all cells. Training takes ~1–3 minutes per epoch on a Colab T4 GPU.

**Expected outputs:**
- Per-epoch evaluation metrics printed to stdout (`eval_f1`, `eval_metaphor_f1`, etc.)
- Classification report (Literal / Metaphor precision, recall, F1)
- Confusion matrix plot
- F1 vs. metaphorical-sample scatter plot per POS
- Qualitative examples of True Positives, False Positives, and False Negatives

---

## Data

### English — VUAMC (VU Amsterdam Metaphor Corpus)

- **File:** `data/raw/0000.parquet`  
- **Source:** [VU Amsterdam Metaphor Corpus](http://www.vismet.org/metcor/manual/index.php)  
- **Size:** 7 850 sentences after filtering  
- **Labels:** `mrw/met` (indirect), `mrw/lit` (direct), `mrw/met/double`, `mrw/met/PP` (personification)  
- **Pre-processed file:** `data/processed/vuamc_token.jsonl`

### Russian — LCC Metaphor Corpus

- **File:** `data/raw/ru_large.zip`  
- **Source:** [LCC Metaphor Corpus](https://github.com/lcc-api/metaphor)  
- **Size:** 11 878 sentences after filtering (~5 % metaphorical token rate)  
- **Pre-processed file:** `data/processed/ru_large_token.jsonl`

### JSONL Schema (processed files)

```json
{
  "document_name": "a8m-fragment02",
  "words": ["The", "cat", "sat", "..."],
  "labels": [0, 0, 1, 0],
  "metaphor_type": ["", "", "Indirect", ""],
  "pos": ["nan", "noun", "verb", "nan"]
}
```

> **Note:** The processed JSONL files are included in the repository. If you need to regenerate them from raw sources, run the preprocessing notebooks first.

---

## Methodology / Approach

1. **Preprocessing** — Raw corpora (Parquet / XML) are converted to a sentence-level JSONL format. Each record contains token lists, binary metaphor labels, metaphor type strings, and coarse POS tags (verb, noun, adj, adv).

2. **Tokenisation** — `XLMRobertaTokenizerFast` (subword) aligns subword tokens back to original words; labels are assigned to the first subword of each word and set to `-100` for subsequent subwords (ignored in loss).

3. **Model** — `xlm-roberta-base` with a token-classification head (`XLMRobertaForTokenClassification`, 2 output classes).

4. **Class-imbalance handling** — Class weights are computed from the training-set distribution and passed to a custom `WeightedTrainer` that applies weighted cross-entropy loss.

5. **POS balancing (optional)** — For `EN_BALANCED_POS`, sentences are resampled so that metaphorical tokens are roughly equally represented across POS categories.

6. **Train / Val / Test split** — Document-level stratified split (70 / 15 / 15) to prevent sentence-level leakage.

7. **Training** — Hugging Face `Trainer` with 3 epochs, batch size 16, `xlm-roberta-base` pre-trained weights, learning-rate schedule, and early-stopping via validation F1.

8. **Cross-lingual transfer (RU)** — The EN-trained model is used as initialisation for Russian experiments with three fine-tuning strategies: zero-shot inference, head-only, and full fine-tuning.

---

## Results

Results for the **RU_FULL** experiment (full fine-tuning on Russian data, reported in the final notebook run):

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Literal | 0.98 | 0.79 | 0.87 |
| Metaphor | 0.17 | 0.75 | 0.28 |
| **Overall (weighted)** | **0.94** | **0.78** | **0.84** |

**Per-POS F1 (Metaphor class):**

| POS | Total Samples | Metaphor Samples | Metaphor % | F1 Metaphor |
|-----|---------------|-----------------|------------|-------------|
| noun | 9 752 | 741 | 7.6 % | 0.309 |
| verb | 3 868 | 334 | 8.6 % | 0.323 |
| adj  | 3 537 | 187 | 5.3 % | 0.229 |
| adv  | 1 312 |  20 | 1.5 % | 0.103 |

Plots are saved in the `plots/` directory and also displayed inline in the notebook.

For a full discussion of results across all five experiments, see **`MetaphorClassification_AidanaAkkaziyeva.pdf`**.

---

## Reproducibility Notes

- **Random seed** is fixed at `SEED = 123` throughout.
- **Document-level splits** ensure no sentence from the same document appears in both train and test sets.
- **Class weights** are hard-coded in the Config cell (`EN_CLASS_WEIGHTS`, `RU_CLASS_WEIGHTS`) based on corpus statistics; recompute them if you use a different data split or dataset.
- **GPU/CPU parity** — results may differ slightly between GPU and CPU runs due to floating-point non-determinism.
- The notebook was last executed on **Google Colab** with `transformers==4.x`, `torch==2.x`. Pin your package versions if exact reproducibility is required.
- Processed JSONL files are committed to the repository so training can be reproduced without re-running the preprocessing notebooks.

---

## License

No license file is currently present in this repository. All rights are reserved by the author unless otherwise stated. If you wish to use this code or data for research or educational purposes, please contact the repository owner.

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`.
3. Commit your changes with clear messages.
4. Open a Pull Request describing what you changed and why.

Please follow the existing code style and include notebook outputs when submitting changes to notebooks.

---

## Citation

If you use this code or build on this work, please cite:

```
Akkaziyeva, A. (2024). Metaphor Classification with XLM-RoBERTa:
Cross-lingual Token-level Detection in English and Russian.
[Unpublished project report]. GitHub: https://github.com/Dona134/Metaphor-Classification-NLP
```

The accompanying project report is available as `MetaphorClassification_AidanaAkkaziyeva.pdf`.
