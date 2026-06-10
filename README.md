# Metaphor Detection & Stylistic Fingerprinting

A two-stage NLP pipeline that (1) trains RoBERTa-based classifiers for metaphor detection and (2) uses those classifiers to extract stylistic features and cluster documents by authorial/genre fingerprint.

---

## Project Structure

```
├── Training and saving the models.ipynb   # Fine-tune & publish sentence/token classifiers
└── Met_style_fingerprint.ipynb            # Inference, feature engineering & clustering
```

---

## Notebooks

### 1. `Training and saving the models.ipynb`

Trains two metaphor detection models on the VUAMC dataset and pushes them to the Hugging Face Hub.

**Pipeline overview:**

- **Data loading & preprocessing** — Loads VUAMC (labeled), Authors (unlabeled), and NarraDetect datasets. Converts word columns from strings to Python lists and splits VUAMC by document name to prevent data leakage.
- **Sentence classifier** — Fine-tunes `RobertaForSequenceClassification` to label sentences as `LITERAL` or `METAPHOR`. Uses Hugging Face `Trainer` with early stopping. Evaluated via confusion matrix on the held-out test set.
- **Token classifier** — Fine-tunes `RobertaForTokenClassification` with a custom `CustomTokenTrainer` that incorporates class weights in the loss function. Evaluated with a full classification report and confusion matrix.
- **Model publishing** — Both models are pushed to the Hugging Face Hub via `huggingface_hub`.

**Key configuration:**

| Parameter | Description |
|-----------|-------------|
| `MAX_LEN` | Maximum token sequence length |
| `BATCH_SIZE` | Training batch size |
| `SEED` | Random seed for reproducibility |

---

### 2. `Met_style_fingerprint.ipynb`

Loads the trained models and applies them to extract stylistic features, then clusters documents to reveal authorial and genre groupings.

**Pipeline overview:**

- **Model loading** — Pre-trained sentence and token classifiers are loaded from Hugging Face and wrapped in `transformers` pipelines.
- **Optimal threshold search** — Uses the VUAMC validation set to find per-classifier probability thresholds that maximise F1-score for metaphor detection.
- **Error analysis** — Identifies and categorises false positives and false negatives from the token classifier on the test set.
- **Feature engineering** (`compute_enriched_features`) — Computes a rich stylistic profile per document:

  | Feature | Description |
  |---------|-------------|
  | Metaphor Density | Proportion of metaphorical subword tokens |
  | MTLD | Measure of Textual Lexical Diversity |
  | Type-Token Ratio | Vocabulary richness |
  | Avg. Sentence Length | Mean words per sentence |
  | Sentence Length Variance | Variability of sentence lengths |
  | Noun-Verb Ratio | Syntactic balance indicator |
  | Content Density | Ratio of content to function words |
  | Function Word Ratio | Proportion of grammatical function words |

- **Inference pipeline** (`pipeline_on_test_df`) — Runs sentence classification first, then applies token classification only to metaphorical sentences, and aggregates metaphorical tokens at the document level.
- **Clustering & visualisation:**
  - Agglomerative (hierarchical) clustering with dendrograms
  - Silhouette Score for cluster quality evaluation
  - Author / genre composition analysis per cluster
  - Cosine similarity heatmaps for pairwise document comparison
  - Random Forest feature importance to identify the most discriminative stylistic features

---

## Datasets

| Dataset | Type | Purpose |
|---------|------|---------|
| VUAMC | Labeled | Training, validation, and testing |
| Authors | Unlabeled | Stylistic analysis target |
| NarraDetect | Unlabeled | Stylistic analysis / genre evaluation |

---

## Dependencies

```
torch
transformers
pandas
scikit-learn
spacy
nltk
lexical-diversity
huggingface_hub
```

Install with:

```bash
pip install torch transformers pandas scikit-learn spacy nltk lexical-diversity huggingface_hub
python -m spacy download en_core_web_sm
```

---

## Quickstart

**Step 1 — Train and publish models**

Open `Training and saving the models.ipynb` and run all cells. Make sure you are logged in to Hugging Face (`huggingface-cli login`) before the saving step.

**Step 2 — Run the stylistic analysis**

Open `Met_style_fingerprint.ipynb`. Update the model identifiers at the top to point to your published models, then run all cells.

---

## Future Work

- Advanced fine-tuning strategies (e.g. layer-wise learning rate decay, adversarial training)
- Richer semantic features using contextual RoBERTa embeddings
- Alternative clustering algorithms (HDBSCAN, spectral clustering)
- Cross-lingual metaphor detection
