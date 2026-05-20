"""
feature_engineering.py
======================
17 assigned stylometric features for team members i=1, i=4, i=13 (n=21).

  Student i=1  → f1,  f22, f43, f64,  f85,  f106
  Student i=4  → f4,  f25, f46, f67,  f88,  f109
  Student i=13 → f13, f34, f55, f76,  f97

┌───────────┬────────────────────────────────────────────────────────────────┐
│  Feature  │  Description                                                   │
├───────────┼────────────────────────────────────────────────────────────────┤
│  f1       │  Total number of characters (C)                                │
│  f4       │  Number of white spaces / C  (whitespace ratio)                │
│  f13      │  Hapax legomena ratio                                          │
│  f22      │  Entropy of word frequencies                                   │
│  f25      │  Number of single quotes                                       │
│  f34      │  Total number of sentences (S)                                 │
│  f43      │  Number of nouns                                               │
│  f46      │  Number of adverbs                                             │
│  f55      │  Noun-to-Verb ratio                                            │
│  f64      │  Number of nominatives                                         │
│  f67      │  Number of singular words                                      │
│  f76      │  Number of passive voice sentences                             │
│  f85      │  Sentence length variance                                      │
│  f88      │  Semantic similarity between sentences (mean cosine)           │
│  f97      │  BERT embedding similarity                                     │
│  f106     │  Tanween frequency (nunation count)                            │
│  f109     │  Link frequency (number of hyperlinks / URLs)                  │
└───────────┴────────────────────────────────────────────────────────────────┘

Each feature is implemented as:
  (a) A pure-Python row function (fast, testable, used inside Spark UDFs)
  (b) A registry entry binding the function to its return type and column name
  (c) A Spark UDF object in _UDFS dict
"""

from __future__ import annotations

import re
import math
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F          # single canonical import
from pyspark.sql.types import DoubleType, IntegerType, StructType, StructField
from pyspark.sql.functions import pandas_udf

from src.utils import logger, FEAT_PREFIX


# ============================================================
# Arabic regex helpers (shared across feature functions)
# ============================================================

_SENT_BOUNDARY = re.compile(r"[.!?؟।…]+")
_WHITESPACE    = re.compile(r"\s")
_SINGLE_QUOTE  = re.compile(r"[''\u2018\u2019]")
_TANWEEN_RE    = re.compile(r"[\u064B\u064C\u064D]")   # fathatan, dammatan, kasratan
_URL_RE        = re.compile(r"https?://\S+|www\.\S+|ftp://\S+", re.IGNORECASE)


# ============================================================
# CAMeL Tools — lazy global (loaded once, used by all morph features)
# ============================================================

_CAMEL_AVAILABLE = False
_analyzer        = None

try:
    from camel_tools.morphology.database import MorphologyDB  # type: ignore
    from camel_tools.morphology.analyzer import Analyzer      # type: ignore
    _db           = MorphologyDB.builtin_db()
    _analyzer     = Analyzer(_db)
    _CAMEL_AVAILABLE = True
    logger.info("CAMeL Tools morphology analyzer loaded.")
except Exception as _e:
    logger.warning(
        "CAMeL Tools not available (%s) — morphology features default to 0.", _e
    )


def _get_camel_analyses(token: str) -> List[dict]:
    """Return CAMeL morphological analyses for one Arabic token."""
    if not _CAMEL_AVAILABLE or _analyzer is None:
        return []
    try:
        return _analyzer.analyze(token)
    except Exception:
        return []


# ============================================================
# ── STUDENT i=1  (k=0..5) ───────────────────────────────────
# ============================================================

def feat_f1_total_chars(text: str) -> int:
    """f1: Total number of characters in the raw text."""
    return len(text) if text else 0


def feat_f22_word_entropy(text: str) -> float:
    """
    f22: Shannon entropy of the word-frequency distribution.

    H = -Σ p(w) * log2(p(w))

    High entropy → more uniform (diverse) vocabulary.
    Low entropy  → repetitive word usage (possible AI trait).

    MapReduce note (see report):
      Job1 (word count):  MAP word→(word,1) | REDUCE sum → (word, count)
      Job2 (entropy):     MAP (word, count) → -p*log2(p) | REDUCE sum → H
    """
    if not text:
        return 0.0
    words = text.split()
    if not words:
        return 0.0
    counts = Counter(words)
    total = len(words)
    return round(-sum((c / total) * math.log2(c / total) for c in counts.values()), 6)


def feat_f43_num_nouns(text: str) -> int:
    """
    f43: Count of nouns (POS tags containing 'noun') via CAMeL Tools.
    Tags targeted: 'noun', 'noun_prop', 'noun_quant'.
    Returns 0 if CAMeL not available.
    """
    if not _CAMEL_AVAILABLE or not text:
        return 0
    return sum(
        1 for token in text.split()
        if any("noun" in a.get("pos", "") for a in _get_camel_analyses(token))
    )


def feat_f64_num_nominatives(text: str) -> int:
    """
    f64: Count tokens whose morphological case is nominative (marfoo').
    CAMeL Tools 'cas' field == 'n' or 'nom'.
    """
    if not _CAMEL_AVAILABLE or not text:
        return 0
    return sum(
        1 for token in text.split()
        if any(a.get("cas", "") in ("n", "nom", "nominative")
               for a in _get_camel_analyses(token))
    )


def feat_f85_sentence_length_variance(text: str) -> float:
    """
    f85: Variance in the number of words per sentence.

    High variance → inconsistent sentence structure (may signal AI).

    MapReduce note (see report):
      Job1: MAP (doc, sent) → (doc_id, sent_len) | REDUCE collect lengths
      Job2: MAP → (doc_id, variance)             | REDUCE pass-through
    """
    if not text:
        return 0.0
    sentences = [s.strip() for s in _SENT_BOUNDARY.split(text) if s.strip()]
    if len(sentences) < 2:
        return 0.0
    lengths = [len(s.split()) for s in sentences]
    mean_len = sum(lengths) / len(lengths)
    return round(sum((l - mean_len) ** 2 for l in lengths) / len(lengths), 6)


def feat_f106_tanween_frequency(text: str) -> int:
    """
    f106: Count of tanween (nunation) characters.
    Tanween diacritics: fathatan ً (U+064B), dammatan ٌ (U+064C), kasratan ٍ (U+064D).
    Marks indefiniteness; frequency may differ between human and AI Arabic text.
    """
    return len(_TANWEEN_RE.findall(text)) if text else 0


# ============================================================
# ── STUDENT i=4  (k=0..5) ───────────────────────────────────
# ============================================================

def feat_f4_whitespace_ratio(text: str) -> float:
    """f4: Ratio of whitespace characters to total characters."""
    if not text:
        return 0.0
    return round(len(_WHITESPACE.findall(text)) / len(text), 6)


def feat_f25_single_quotes(text: str) -> int:
    """f25: Count of single-quote characters (ASCII ' and Unicode variants)."""
    return len(_SINGLE_QUOTE.findall(text)) if text else 0


def feat_f46_num_adverbs(text: str) -> int:
    """
    f46: Count of adverbs (CAMeL POS tag starting with 'adv').
    Returns 0 if CAMeL not available.
    """
    if not _CAMEL_AVAILABLE or not text:
        return 0
    return sum(
        1 for token in text.split()
        if any(a.get("pos", "").startswith("adv") for a in _get_camel_analyses(token))
    )


def feat_f67_num_singular(text: str) -> int:
    """
    f67: Count tokens whose morphological number is singular.
    CAMeL Tools 'num' field == 's'.
    """
    if not _CAMEL_AVAILABLE or not text:
        return 0
    return sum(
        1 for token in text.split()
        if any(a.get("num", "") == "s" for a in _get_camel_analyses(token))
    )


# Lazy sentence-transformer model for f88
_SENT_MODEL = None

def _get_sentence_model():
    global _SENT_MODEL
    if _SENT_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            _SENT_MODEL = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
        except Exception as e:
            logger.warning("sentence-transformers not available (%s); f88 defaults to 0.", e)
    return _SENT_MODEL


def feat_f88_semantic_similarity(text: str) -> float:
    """
    f88: Mean pairwise cosine similarity between sentence embeddings.

    Uses 'paraphrase-multilingual-MiniLM-L12-v2' (supports Arabic).
    High similarity → repetitive / semantically uniform content (AI trait).
    Returns 0.0 if sentence-transformers not installed.
    Capped at 10 sentences to limit runtime.
    """
    if not text:
        return 0.0
    sentences = [s.strip() for s in _SENT_BOUNDARY.split(text) if s.strip()]
    if len(sentences) < 2:
        return 0.0
    model = _get_sentence_model()
    if model is None:
        return 0.0
    try:
        embeddings = model.encode(sentences[:10], normalize_embeddings=True)
        n = len(embeddings)
        sims = [
            float(np.dot(embeddings[i], embeddings[j]))
            for i in range(n) for j in range(i + 1, n)
        ]
        return round(float(np.mean(sims)), 6) if sims else 0.0
    except Exception:
        return 0.0


def feat_f109_link_frequency(text: str) -> int:
    """
    f109: Count of hyperlinks / URLs in the text.
    Academic abstracts typically have 0 links; AI models sometimes inject references.
    """
    return len(_URL_RE.findall(text)) if text else 0


# ============================================================
# ── STUDENT i=13  (k=0..4) ──────────────────────────────────
# ============================================================

def feat_f13_hapax_ratio(text: str) -> float:
    """
    f13: Hapax legomena ratio = (words appearing exactly once) / total words.

    High ratio → richer, less repetitive vocabulary.

    MapReduce design (see Methodology section of report):
      Job1 (word count):
        MAP  (doc_id, text) → (word, 1)
        REDUCE             → (word, total_count)
      Job2 (hapax identification):
        MAP  filter total_count == 1 → ("HAPAX_KEY", 1)
        REDUCE                       → ("HAPAX_KEY", hapax_total)
      Final:  hapax_ratio = hapax_total / total_word_count
    """
    if not text:
        return 0.0
    words = text.split()
    if not words:
        return 0.0
    freq = Counter(words)
    hapax = sum(1 for c in freq.values() if c == 1)
    return round(hapax / len(words), 6)


def feat_f34_num_sentences(text: str) -> int:
    """
    f34: Total number of sentences.
    Splits on Arabic (؟) and Latin (. ! ? …) sentence terminators.
    """
    if not text:
        return 0
    return len([s for s in _SENT_BOUNDARY.split(text) if s.strip()])


def feat_f55_noun_verb_ratio(text: str) -> float:
    """
    f55: Ratio of nouns to verbs.
    = count_nouns / max(count_verbs, 1)  (avoids division by zero).
    High ratio → nominal style; low ratio → verbal style.
    """
    if not _CAMEL_AVAILABLE or not text:
        return 0.0
    nouns = verbs = 0
    for token in text.split():
        analyses = _get_camel_analyses(token)
        best = analyses[0] if analyses else {}
        pos = best.get("pos", "")
        if "noun" in pos:
            nouns += 1
        elif "verb" in pos:
            verbs += 1
    return round(nouns / max(verbs, 1), 6)


_PASSIVE_RE = re.compile(r"يُ\S*ِ|تُ\S*ِ|نُ\S*ِ|مُ\S*ِ")


def feat_f76_passive_sentences(text: str) -> int:
    """
    f76: Count of sentences containing passive voice constructions.

    Detection strategy (two-level):
      1. If CAMeL available: check 'voice' == 'p' on any token in sentence.
      2. Regex fallback: match common Arabic passive verb patterns (يُفِعَل).
    """
    if not text:
        return 0
    sentences = [s.strip() for s in _SENT_BOUNDARY.split(text) if s.strip()]
    count = 0
    for sent in sentences:
        if _CAMEL_AVAILABLE:
            if any(
                any(a.get("voice", "") == "p" for a in _get_camel_analyses(tok))
                for tok in sent.split()
            ):
                count += 1
        else:
            if _PASSIVE_RE.search(sent):
                count += 1
    return count


# Lazy BERT model for f97
_BERT_TOKENIZER = None
_BERT_MODEL     = None


def _get_bert():
    global _BERT_TOKENIZER, _BERT_MODEL
    if _BERT_TOKENIZER is None:
        for model_name in [
            "aubmindlab/bert-base-arabertv2",
            "bert-base-multilingual-cased",
        ]:
            try:
                from transformers import AutoTokenizer, AutoModel  # type: ignore
                import torch
                _BERT_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
                _BERT_MODEL     = AutoModel.from_pretrained(model_name)
                _BERT_MODEL.eval()
                logger.info("BERT model '%s' loaded for f97.", model_name)
                break
            except Exception as e:
                logger.warning("Could not load '%s': %s", model_name, e)
    return _BERT_TOKENIZER, _BERT_MODEL


def feat_f97_bert_embedding_similarity(text: str) -> float:
    """
    f97: Mean cosine similarity between BERT [CLS] embeddings of sentences.

    Sentences encoded individually (capped at 6 for runtime).
    High inter-sentence BERT similarity → semantically uniform / repetitive text.
    Falls back to 0.0 if transformers not installed.
    """
    if not text:
        return 0.0
    sentences = [s.strip() for s in _SENT_BOUNDARY.split(text) if len(s.strip()) > 10]
    if len(sentences) < 2:
        return 0.0
    tok, model = _get_bert()
    if tok is None or model is None:
        return 0.0
    try:
        import torch
        embeddings = []
        for sent in sentences[:6]:
            inputs = tok(sent, return_tensors="pt", truncation=True,
                         max_length=128, padding=True)
            with torch.no_grad():
                out = model(**inputs)
            cls = out.last_hidden_state[:, 0, :].squeeze().numpy()
            norm = np.linalg.norm(cls)
            if norm > 0:
                embeddings.append(cls / norm)
        if len(embeddings) < 2:
            return 0.0
        sims = [
            float(np.dot(embeddings[i], embeddings[j]))
            for i in range(len(embeddings))
            for j in range(i + 1, len(embeddings))
        ]
        return round(float(np.mean(sims)), 6)
    except Exception as exc:
        logger.warning("f97 BERT similarity failed: %s", exc)
        return 0.0


# ============================================================
# Feature registry + UDF objects
# ============================================================

_FEATURE_REGISTRY: Dict[str, Tuple] = {
    # key: (python_func, spark_return_type, output_column_name)
    # ── i=1 ─────────────────────────────────────────────────
    "f1_total_chars":          (feat_f1_total_chars,              IntegerType(), f"{FEAT_PREFIX}f1_total_chars"),
    "f22_word_entropy":        (feat_f22_word_entropy,            DoubleType(),  f"{FEAT_PREFIX}f22_word_entropy"),
    "f43_num_nouns":           (feat_f43_num_nouns,               IntegerType(), f"{FEAT_PREFIX}f43_num_nouns"),
    "f64_num_nominatives":     (feat_f64_num_nominatives,         IntegerType(), f"{FEAT_PREFIX}f64_num_nominatives"),
    "f85_sent_len_variance":   (feat_f85_sentence_length_variance,DoubleType(),  f"{FEAT_PREFIX}f85_sent_len_variance"),
    "f106_tanween_freq":       (feat_f106_tanween_frequency,      IntegerType(), f"{FEAT_PREFIX}f106_tanween_freq"),
    # ── i=4 ─────────────────────────────────────────────────
    "f4_whitespace_ratio":     (feat_f4_whitespace_ratio,         DoubleType(),  f"{FEAT_PREFIX}f4_whitespace_ratio"),
    "f25_single_quotes":       (feat_f25_single_quotes,           IntegerType(), f"{FEAT_PREFIX}f25_single_quotes"),
    "f46_num_adverbs":         (feat_f46_num_adverbs,             IntegerType(), f"{FEAT_PREFIX}f46_num_adverbs"),
    "f67_num_singular":        (feat_f67_num_singular,            IntegerType(), f"{FEAT_PREFIX}f67_num_singular"),
    "f88_semantic_similarity": (feat_f88_semantic_similarity,     DoubleType(),  f"{FEAT_PREFIX}f88_semantic_sim"),
    "f109_link_frequency":     (feat_f109_link_frequency,         IntegerType(), f"{FEAT_PREFIX}f109_link_freq"),
    # ── i=13 ────────────────────────────────────────────────
    "f13_hapax_ratio":         (feat_f13_hapax_ratio,             DoubleType(),  f"{FEAT_PREFIX}f13_hapax_ratio"),
    "f34_num_sentences":       (feat_f34_num_sentences,           IntegerType(), f"{FEAT_PREFIX}f34_num_sentences"),
    "f55_noun_verb_ratio":     (feat_f55_noun_verb_ratio,         DoubleType(),  f"{FEAT_PREFIX}f55_noun_verb_ratio"),
    "f76_passive_sents":       (feat_f76_passive_sentences,       IntegerType(), f"{FEAT_PREFIX}f76_passive_sents"),
    "f97_bert_similarity":     (feat_f97_bert_embedding_similarity,DoubleType(), f"{FEAT_PREFIX}f97_bert_sim"),
}

_UDFS = {
    key: F.udf(func, ret_type)
    for key, (func, ret_type, _) in _FEATURE_REGISTRY.items()
}

FEATURE_COLS = [col_name for (_, _, col_name) in _FEATURE_REGISTRY.values()]

# ============================================================
# Feature tier groups (by computational cost)
# ============================================================
# Tier 1 — pure text ops, no external model needed (~seconds)
LIGHT_KEYS: List[str] = [
    "f1_total_chars", "f4_whitespace_ratio", "f13_hapax_ratio",
    "f22_word_entropy", "f25_single_quotes", "f34_num_sentences",
    "f85_sent_len_variance", "f106_tanween_freq", "f109_link_frequency",
]
# Tier 2 — CAMeL Tools morphological analysis (~minutes)
CAMEL_KEYS: List[str] = [
    "f43_num_nouns", "f46_num_adverbs", "f55_noun_verb_ratio",
    "f64_num_nominatives", "f67_num_singular", "f76_passive_sents",
]
# Tier 3 — BERT / sentence-transformer embeddings (~long; FAST_MODE skips)
DEEP_KEYS: List[str] = [
    "f88_semantic_similarity", "f97_bert_similarity",
]


# ============================================================
# Batch helper functions for deep features
# ============================================================

def _compute_f88_batch(texts: list) -> list:
    """
    Batch-compute f88 (semantic similarity) for a list of texts.

    All sentences from all texts in the partition are gathered and
    encoded in one model.encode() call, then cosine similarities are
    computed per text.  This amortises model-load and GPU/CPU overhead
    across the whole batch.
    """
    model = _get_sentence_model()
    if model is None:
        return [0.0] * len(texts)

    all_sents: List[str] = []
    ranges: List[Tuple[int, int]] = []

    for text in texts:
        sents = [s.strip() for s in _SENT_BOUNDARY.split(text) if s.strip()][:10]
        start = len(all_sents)
        all_sents.extend(sents)
        ranges.append((start, len(all_sents)))

    if not all_sents:
        return [0.0] * len(texts)

    try:
        all_embs = model.encode(
            all_sents, normalize_embeddings=True,
            batch_size=128, show_progress_bar=False,
        )
        results = []
        for start, end in ranges:
            embs = all_embs[start:end]
            if len(embs) < 2:
                results.append(0.0)
                continue
            sims = [
                float(np.dot(embs[a], embs[b]))
                for a in range(len(embs))
                for b in range(a + 1, len(embs))
            ]
            results.append(round(float(np.mean(sims)), 6))
        return results
    except Exception as exc:
        logger.warning("_compute_f88_batch failed: %s", exc)
        return [0.0] * len(texts)


def _compute_f97_batch(texts: list) -> list:
    """
    Batch-compute f97 (BERT embedding similarity) for a list of texts.

    All sentences from the partition are tokenised and encoded together
    in one forward pass (chunked by batch_size=32), then cosine
    similarities are computed per text.
    """
    tok, bert = _get_bert()
    if tok is None or bert is None:
        return [0.0] * len(texts)

    import torch

    all_sents: List[str] = []
    ranges: List[Tuple[int, int]] = []

    for text in texts:
        sents = [s.strip() for s in _SENT_BOUNDARY.split(text)
                 if len(s.strip()) > 10][:6]
        start = len(all_sents)
        all_sents.extend(sents)
        ranges.append((start, len(all_sents)))

    if not all_sents:
        return [0.0] * len(texts)

    batch_size = 32
    all_cls_parts = []
    try:
        for i in range(0, len(all_sents), batch_size):
            batch = all_sents[i: i + batch_size]
            inputs = tok(
                batch, return_tensors="pt",
                truncation=True, max_length=128, padding=True,
            )
            with torch.no_grad():
                out = bert(**inputs)
            cls = out.last_hidden_state[:, 0, :].numpy()   # (B, H)
            norms = np.linalg.norm(cls, axis=1, keepdims=True).clip(min=1e-8)
            all_cls_parts.append(cls / norms)

        all_cls = np.vstack(all_cls_parts)   # (total_sents, H)

        results = []
        for start, end in ranges:
            embs = all_cls[start:end]
            if len(embs) < 2:
                results.append(0.0)
                continue
            sims = [
                float(np.dot(embs[a], embs[b]))
                for a in range(len(embs))
                for b in range(a + 1, len(embs))
            ]
            results.append(round(float(np.mean(sims)), 6))
        return results
    except Exception as exc:
        logger.warning("_compute_f97_batch failed: %s", exc)
        return [0.0] * len(texts)


# ============================================================
# Vocabulary-caching batch function for all 6 CAMeL features
# ============================================================

CAMEL_STRUCT_TYPE = StructType([
    StructField(f"{FEAT_PREFIX}f43_num_nouns",       IntegerType(), True),
    StructField(f"{FEAT_PREFIX}f46_num_adverbs",     IntegerType(), True),
    StructField(f"{FEAT_PREFIX}f55_noun_verb_ratio", DoubleType(),  True),
    StructField(f"{FEAT_PREFIX}f64_num_nominatives", IntegerType(), True),
    StructField(f"{FEAT_PREFIX}f67_num_singular",    IntegerType(), True),
    StructField(f"{FEAT_PREFIX}f76_passive_sents",   IntegerType(), True),
])

# Column names in insertion order — matches CAMEL_STRUCT_TYPE fields
_CAMEL_COLS = [f.name for f in CAMEL_STRUCT_TYPE.fields]


def compute_camel_features_pandas(texts: list) -> pd.DataFrame:
    """
    Compute all 6 CAMeL morphology features for a list of texts with
    a partition-level vocabulary cache.

    Algorithm
    ---------
    1. Walk all texts once → collect every unique token into a vocab set.
    2. Call CAMeL analyzer once per unique token (not per occurrence).
    3. Walk all texts a second time → look up cached analyses to tally
       all 6 feature counts in a single loop per text.

    This reduces total analyzer calls from
        N_rows × avg_tokens × 6  (old: ~6 000 000 for 42 000 rows)
    to
        N_unique_tokens           (new: ~50 000 – 80 000 for academic Arabic)

    Returns
    -------
    pd.DataFrame  with columns matching CAMEL_STRUCT_TYPE field names.
    """
    col_f43, col_f46, col_f55, col_f64, col_f67, col_f76 = _CAMEL_COLS

    # ── Step 1: collect unique tokens from the whole partition ────────────
    vocab: set = set()
    for text in texts:
        if text:
            vocab.update(text.split())
    logger.debug("CAMeL batch: %d unique tokens from %d texts", len(vocab), len(texts))

    # ── Step 2: analyze each unique token exactly once ────────────────────
    token_cache: dict = {}
    if _CAMEL_AVAILABLE:
        for token in vocab:
            token_cache[token] = _get_camel_analyses(token)

    # ── Step 3: compute all 6 features per text using cached analyses ─────
    rows: List[dict] = []
    for text in texts:
        if not text:
            rows.append({col_f43: 0, col_f46: 0, col_f55: 0.0,
                         col_f64: 0, col_f67: 0, col_f76: 0})
            continue

        tokens = text.split()
        sentences = [s.strip() for s in _SENT_BOUNDARY.split(text) if s.strip()]

        nouns = verbs = adverbs = nominatives = singulars = 0

        for token in tokens:
            analyses = token_cache.get(token, [])
            best     = analyses[0] if analyses else {}
            pos      = best.get("pos", "")
            cas      = best.get("cas", "")
            num      = best.get("num", "")

            if "noun" in pos:
                nouns += 1
            if "verb" in pos:
                verbs += 1
            if pos.startswith("adv"):
                adverbs += 1
            if cas in ("n", "nom", "nominative"):
                nominatives += 1
            if num == "s":
                singulars += 1

        # f76 passive: use CAMeL 'voice' if available, else regex fallback
        passive = 0
        for sent in sentences:
            sent_tokens = sent.split()
            if _CAMEL_AVAILABLE and sent_tokens:
                if any(
                    token_cache.get(tok, [{}])[0].get("voice", "") == "p"
                    for tok in sent_tokens
                    if token_cache.get(tok)
                ):
                    passive += 1
            elif _PASSIVE_RE.search(sent):
                passive += 1

        rows.append({
            col_f43: nouns,
            col_f46: adverbs,
            col_f55: round(nouns / max(verbs, 1), 6),
            col_f64: nominatives,
            col_f67: singulars,
            col_f76: passive,
        })

    return pd.DataFrame(rows, columns=_CAMEL_COLS)


@pandas_udf(CAMEL_STRUCT_TYPE)
def _pudf_all_camel(texts: pd.Series) -> pd.DataFrame:
    """
    Single Arrow-optimised pandas_udf computing all 6 CAMeL features at once.

    Benefits over 6 separate UDF calls:
    • 1 data pass instead of 6 (6× less I/O)
    • Vocabulary cache shared across all 6 features in the partition
    • CAMeL analyzer initialised once per executor process (not per row)
    """
    return compute_camel_features_pandas(texts.fillna("").tolist())


# ============================================================
# Pandas UDFs — Arrow-optimised, processes a full partition at once
# ============================================================

def _make_pandas_udf(func, ret_type):
    """Wrap a Python scalar function as a partition-level pandas_udf."""
    @pandas_udf(ret_type)
    def _inner(texts: pd.Series) -> pd.Series:
        return texts.fillna("").apply(func)
    return _inner


# Light and CAMeL features — simple series-apply
_PANDAS_UDFS: Dict[str, Any] = {
    key: _make_pandas_udf(func, ret_type)
    for key, (func, ret_type, _) in _FEATURE_REGISTRY.items()
    if key not in DEEP_KEYS
}

# Deep features — whole-partition batch UDFs
@pandas_udf(DoubleType())
def _pudf_f88(texts: pd.Series) -> pd.Series:
    return pd.Series(_compute_f88_batch(texts.fillna("").tolist()))

@pandas_udf(DoubleType())
def _pudf_f97(texts: pd.Series) -> pd.Series:
    return pd.Series(_compute_f97_batch(texts.fillna("").tolist()))

_PANDAS_UDFS["f88_semantic_similarity"] = _pudf_f88
_PANDAS_UDFS["f97_bert_similarity"]     = _pudf_f97


# ============================================================
# Master feature extraction function  (kept for streaming use)
# ============================================================

def extract_all_features(
    df: DataFrame,
    input_col: str = "clean_text",
) -> DataFrame:
    """
    Apply all 17 assigned stylometric features to the DataFrame using
    Arrow-optimised pandas_udf (much faster than row-level F.udf).

    For large batch jobs prefer `extract_features_tiered()` which adds
    intermediate checkpoints between cost tiers.

    Parameters
    ----------
    df        : Spark DataFrame containing at least `input_col`.
    input_col : Column with preprocessed Arabic text.

    Returns
    -------
    DataFrame with 17 new feature columns (prefix 'feat_').
    """
    logger.info("Extracting %d stylometric features (pandas_udf) …",
                len(_FEATURE_REGISTRY))
    result = df
    for key, (_, _, col_name) in _FEATURE_REGISTRY.items():
        result = result.withColumn(col_name, _PANDAS_UDFS[key](F.col(input_col)))
        logger.info("  ✓ %s", col_name)
    return result


# ============================================================
# Tiered extraction (notebook use — checkpoints between tiers)
# ============================================================

def extract_features_tiered(
    df: DataFrame,
    input_col: str = "clean_text",
    light_partitions: int = 0,    # 0 = keep current partitioning
    camel_partitions: int = 4,    # small → CAMeL DB loads fewer times
    deep_partitions:  int = 2,    # small → BERT loads fewer times
    fast_mode: bool = False,      # True → set deep features to 0.0
) -> Dict[str, DataFrame]:
    """
    Extract features in three cost tiers.  Returns a dict of DataFrames
    so the caller can checkpoint each tier independently.

    Dict keys: 'light', 'camel', 'deep'

    Parameters
    ----------
    light_partitions : Spark partitions for Tier 1 (0 = unchanged).
    camel_partitions : Partitions for Tier 2 (fewer = less model reload).
    deep_partitions  : Partitions for Tier 3.
    fast_mode        : Skip Tier 3 — fill f88 and f97 with 0.0.
    """
    # ── Tier 1: light (pure text) ────────────────────────────────────────
    logger.info("Tier 1 — %d light features …", len(LIGHT_KEYS))
    base = df.repartition(light_partitions) if light_partitions > 0 else df
    for key in LIGHT_KEYS:
        _, _, col_name = _FEATURE_REGISTRY[key]
        base = base.withColumn(col_name, _PANDAS_UDFS[key](F.col(input_col)))
    light_df = base

    # ── Tier 2: CAMeL morphology ─────────────────────────────────────────
    logger.info("Tier 2 — %d CAMeL features …", len(CAMEL_KEYS))
    base = light_df.repartition(camel_partitions)
    for key in CAMEL_KEYS:
        _, _, col_name = _FEATURE_REGISTRY[key]
        base = base.withColumn(col_name, _PANDAS_UDFS[key](F.col(input_col)))
    camel_df = base

    # ── Tier 3: deep learning ─────────────────────────────────────────────
    logger.info("Tier 3 — %d deep features (fast_mode=%s) …",
                len(DEEP_KEYS), fast_mode)
    if fast_mode:
        result = camel_df
        for key in DEEP_KEYS:
            _, ret_type, col_name = _FEATURE_REGISTRY[key]
            result = result.withColumn(col_name, F.lit(0.0).cast(ret_type))
        deep_df = result
    else:
        base = camel_df.repartition(deep_partitions)
        for key in DEEP_KEYS:
            _, _, col_name = _FEATURE_REGISTRY[key]
            base = base.withColumn(col_name, _PANDAS_UDFS[key](F.col(input_col)))
        deep_df = base

    return {"light": light_df, "camel": camel_df, "deep": deep_df}


# ============================================================
# TF-IDF pipeline  (Task 3.2 — advanced features)
# ============================================================

def build_tfidf_pipeline(
    num_features: int = 20_000,
    min_df: int = 2,
    input_col: str = "clean_text",
    output_col: str = "tfidf_features",
):
    """
    Return an unfitted Spark MLlib Pipeline for TF-IDF feature extraction.

    Stages: Tokenizer → HashingTF → IDF

    Parameters
    ----------
    num_features : Hash buckets for HashingTF (vocabulary size).
    min_df       : Minimum document frequency for IDF weighting.
    input_col    : Input text column name.
    output_col   : Output sparse vector column name.
    """
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import Tokenizer, HashingTF, IDF

    tokenizer = Tokenizer(inputCol=input_col,  outputCol="_tokens")
    htf       = HashingTF(inputCol="_tokens",  outputCol="_tf_raw", numFeatures=num_features)
    idf       = IDF(inputCol="_tf_raw",        outputCol=output_col, minDocFreq=min_df)
    return Pipeline(stages=[tokenizer, htf, idf])


def build_word2vec_pipeline(
    vector_size: int = 100,
    min_count: int = 2,
    input_col: str = "clean_text",
    output_col: str = "w2v_features",
):
    """
    Return an unfitted Spark MLlib Pipeline for Word2Vec embeddings.

    Parameters
    ----------
    vector_size : Dimensionality of word vectors.
    min_count   : Minimum word frequency to include in vocabulary.
    """
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import Tokenizer, Word2Vec

    tokenizer = Tokenizer(inputCol=input_col, outputCol="_tokens")
    w2v       = Word2Vec(
        inputCol="_tokens", outputCol=output_col,
        vectorSize=vector_size, minCount=min_count, seed=42,
    )
    return Pipeline(stages=[tokenizer, w2v])


# ============================================================
# Feature vector assembler (stylometric + TF-IDF → 'features')
# ============================================================

def assemble_feature_vector(
    df: DataFrame,
    tfidf_col: str = "tfidf_features",
    output_col: str = "features",
) -> DataFrame:
    """
    Combine all stylometric feature columns and the TF-IDF vector into a
    single 'features' DenseVector column required by Spark MLlib classifiers.

    Handles missing stylometric columns gracefully (warns and skips).
    """
    from pyspark.ml.feature import VectorAssembler

    available = [c for c in FEATURE_COLS if c in df.columns]
    missing   = [c for c in FEATURE_COLS if c not in df.columns]

    if missing:
        logger.warning("Missing feature columns (skipped): %s", missing)

    if tfidf_col not in df.columns:
        raise ValueError(f"Required TF-IDF column '{tfidf_col}' not found in DataFrame.")

    # Cast all stylometric columns to Double (VectorAssembler requirement)
    result = df
    for col_name in available:
        result = result.withColumn(col_name, F.col(col_name).cast(DoubleType()))

    assembler = VectorAssembler(
        inputCols=available + [tfidf_col],
        outputCol=output_col,
        handleInvalid="skip",
    )
    return assembler.transform(result)


# ============================================================
# MapReduce pseudocode — for Methodology section of report
# ============================================================

MAPREDUCE_PSEUDOCODE = """
══════════════════════════════════════════════════════════════
  MapReduce Design — Hapax Legomena Ratio (f13)
══════════════════════════════════════════════════════════════

Job 1 — Word Count
  MAP:    (doc_id, text)  →  (word, 1)  for each word
  REDUCE: (word, [1,1,…]) →  (word, total_count)

Job 2 — Hapax Identification
  MAP:    (word, total_count) | filter count==1
                              → ("HAPAX_KEY", 1)
  REDUCE: ("HAPAX_KEY", [1,1,…]) → hapax_total

Final Calculation
  hapax_ratio = hapax_total / total_word_count

Spark equivalent:
  Job1 → df.explode + groupBy("word").count()
  Job2 → filter(count == 1).count()
  Final → hapax_count / total_count
══════════════════════════════════════════════════════════════

══════════════════════════════════════════════════════════════
  MapReduce Design — Word Entropy (f22)
══════════════════════════════════════════════════════════════

Job 1 — Word Count  (same as above)

Job 2 — Entropy Calculation
  MAP:    (word, count) → p = count/N
                          emit (DOC_ID, -p * log2(p))
  REDUCE: (DOC_ID, [-p*log2(p), …]) → H = sum(values)

Spark equivalent:
  Job1 → groupBy("word").count()
  Job2 → withColumn("p", col("count")/N)
          .withColumn("h", -col("p")*log2(col("p")))
          .agg(sum("h"))
══════════════════════════════════════════════════════════════
"""


def print_mapreduce_design() -> None:
    """Print the MapReduce pseudocode for inclusion in the project report."""
    print(MAPREDUCE_PSEUDOCODE)
