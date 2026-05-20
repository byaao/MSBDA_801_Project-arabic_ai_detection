# Scalable Real-time Detection of AI-Generated Arabic Text
### MSBDA-801 Big Data Analytics — Final Project

A distributed Big Data pipeline that detects AI-generated Arabic academic abstracts
using Apache Spark (MLlib + Structured Streaming) on Google Colab.

---

## Project Structure

```
arabic_ai_detection/
├── src/
│   ├── __init__.py                  — package marker (committed)
│   ├── utils.py                     — shared constants, Spark factory, helpers
│   ├── data_preparation.py          — download, labelling, preprocessing, split
│   ├── feature_engineering.py       — 17 stylometric features, TF-IDF, Word2Vec
│   ├── modeling.py                  — LR, RF, SVM training + evaluation
│   └── streaming_pipeline.py        — Structured Streaming pipeline + StreamWriter
├── notebooks/
│   ├── 00_environment_setup.ipynb   — Phase 1: install packages, start Spark
│   ├── 01_data_acquisition.ipynb    — Phase 1: download dataset, initial EDA
│   ├── 02_preprocessing.ipynb       — Phase 2: Arabic NLP pipeline, MapReduce stats
│   ├── 03_eda.ipynb                 — Phase 2: word clouds, n-grams, TTR, charts
│   ├── 04_feature_engineering.ipynb — Phase 3: stylometric + TF-IDF + split + assemble
│   ├── 05_modeling.ipynb            — Phase 3/4: train LR/RF/SVM, evaluate, scalability
│   ├── 06_streaming.ipynb           — Phase 4: Structured Streaming demo
│   └── 07_analysis_reporting.ipynb  — Phase 5: final analysis, error analysis, figures
├── data/
│   ├── raw/           — raw HuggingFace parquet files (stored on GDrive, not git)
│   └── processed/     — Parquet checkpoints (stored on GDrive, not git)
├── models/            — saved Spark ML pipelines (stored on GDrive, not git)
├── reports/
│   └── figures/       — saved plots and CSV result tables
├── stream/
│   ├── input/         — stream simulation input JSON files
│   ├── output/        — stream output Parquet batches
│   └── checkpoint/    — Spark Structured Streaming checkpoint
├── requirements.txt
└── .gitignore
```

---

## Dataset

**KFUPM-JRCAI/arabic-generated-abstracts** ([HuggingFace](https://huggingface.co/datasets/KFUPM-JRCAI/arabic-generated-abstracts))

| Generation Method | Samples | Description |
|---|---|---|
| by_polishing | 2,851 | Refining existing human abstracts |
| from_title | 2,963 | Free-form from paper title only |
| from_title_and_content | 2,574 | Content-aware from title + paper |
| **Total** | **8,388** | Across all methods |

**Labels**: 0 = human-written, 1 = AI-generated  
**AI models**: ALLAM, JAIS, LLaMA, OpenAI GPT

---

## Assigned Stylometric Features

Feature formula: `f_((k×n)+i)`, where `n=21`, `k=0,1,2,…`

| Student | k=0 | k=1 | k=2 | k=3 | k=4 | k=5 |
|---|---|---|---|---|---|---|
| i=1 | f1 (total chars) | f22 (word entropy) | f43 (# nouns) | f64 (# nominatives) | f85 (sent len variance) | f106 (tanween freq) |
| i=4 | f4 (whitespace ratio) | f25 (# single quotes) | f46 (# adverbs) | f67 (# singular words) | f88 (semantic sim) | f109 (link freq) |
| i=13 | f13 (hapax ratio) | f34 (# sentences) | f55 (noun/verb ratio) | f76 (# passive sents) | f97 (BERT sim) | — |

---

## Setup and Execution

### 1. Google Colab

1. Open and run notebooks in order: `00` → `01` → `02` → `03` → `04` → `05` → `06` → `07`
2. Each notebook starts with a bootstrap cell that mounts Drive and creates the SparkSession.
3. Notebooks use GDrive checkpoints — safe across session reconnects.

### 2. Local (testing only)

```bash
pip install -r requirements.txt

# Set project root (defaults to cwd if not on Colab)
export PROJECT_ROOT=/path/to/arabic_ai_detection

# Run notebooks in order with Jupyter
jupyter notebook notebooks/
```

---

## Pipeline Phases

| Phase | Notebooks | Key outputs |
|---|---|---|
| 1 — Env + Acquisition | 00, 01 | `data/raw/*.parquet`, `labelled_raw` checkpoint |
| 2 — Preprocessing + EDA | 02, 03 | `preprocessed` checkpoint, figures |
| 3 — Feature Eng + Modeling | 04, 05 | `features_*`, split checkpoints, saved models |
| 4 — Streaming | 06 | `stream/output/batch_*/` Parquet, throughput charts |
| 5 — Analysis + Report | 07 | Final figures, comparison tables |

---

## Team

Amani Fahad Aloufi       | 4725097

Asmaa Raja Allah Alharbi | 4725073

Ohud Sulaiman Alraddadi  | 4725435

