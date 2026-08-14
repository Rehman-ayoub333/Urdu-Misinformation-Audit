"""TF-IDF classical baselines — Experiments A (Logistic Regression) and B (linear SVM).

Implements `EXPERIMENT_PLAN.md` Section 2 step 1. These are not throwaway numbers
to get past: `MASTER_PROJECT_BLUEPRINT.md` Part 32 records that classical models
have beaten transformers on this task family (FIRE2021), so A and B are a genuine
comparison point for C/D, and the whole config is tuned to give them a fair
showing rather than a token one.

Everything is driven by `research/configs/model_classical.yaml`. Determinism comes
from that file's `random_state` (`REPRODUCIBILITY.md` Section 2) — no library
default is relied on anywhere.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

__all__ = [
    "build_vectorizer",
    "build_classifier",
    "build_pipeline",
    "fit_predict",
]


def build_vectorizer(config: dict[str, Any]) -> FeatureUnion | TfidfVectorizer:
    """Assemble the TF-IDF feature space from config.

    A word + character-n-gram union by default. Character n-grams matter for Urdu
    specifically (`MASTER_PROJECT_BLUEPRINT.md` Part 7 item 3): the language is
    morphologically rich and the corpora are inconsistently spaced, so word
    boundaries are a lossy unit on their own.
    """
    tfidf = config["features"]["tfidf"]
    shared = {
        "lowercase": tfidf.get("lowercase", False),
        "strip_accents": tfidf.get("strip_accents"),
    }

    blocks: list[tuple[str, TfidfVectorizer]] = []

    word_cfg = tfidf.get("word", {})
    if word_cfg.get("enabled", False):
        blocks.append(
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=tuple(word_cfg["ngram_range"]),
                    min_df=word_cfg["min_df"],
                    max_df=word_cfg["max_df"],
                    max_features=word_cfg.get("max_features"),
                    sublinear_tf=word_cfg.get("sublinear_tf", True),
                    **shared,
                ),
            )
        )

    char_cfg = tfidf.get("char", {})
    if char_cfg.get("enabled", False):
        blocks.append(
            (
                "char",
                TfidfVectorizer(
                    analyzer=char_cfg.get("analyzer", "char_wb"),
                    ngram_range=tuple(char_cfg["ngram_range"]),
                    min_df=char_cfg["min_df"],
                    max_df=char_cfg["max_df"],
                    max_features=char_cfg.get("max_features"),
                    sublinear_tf=char_cfg.get("sublinear_tf", True),
                    **shared,
                ),
            )
        )

    if not blocks:
        raise ValueError("model_classical.yaml enables neither word nor char TF-IDF features")
    if len(blocks) == 1:
        return blocks[0][1]
    return FeatureUnion(blocks)


def build_classifier(kind: str, params: dict[str, Any], random_state: int):  # noqa: ANN201
    """Instantiate a classifier with the seed passed explicitly, never defaulted."""
    params = dict(params)
    if kind == "logistic_regression":
        return LogisticRegression(random_state=random_state, **params)
    if kind == "linear_svc":
        return LinearSVC(random_state=random_state, **params)
    raise ValueError(f"unknown classifier {kind!r} (expected logistic_regression or linear_svc)")


def build_pipeline(config: dict[str, Any], experiment_id: str) -> Pipeline:
    """Full vectorizer -> classifier pipeline for experiment A or B."""
    experiment = config["experiments"][experiment_id]
    return Pipeline(
        [
            ("features", build_vectorizer(config)),
            (
                "classifier",
                build_classifier(
                    experiment["classifier"],
                    experiment.get("params", {}),
                    config["random_state"],
                ),
            ),
        ]
    )


def fit_predict(
    pipeline: Pipeline,
    train_texts: list[str],
    train_labels: list[str],
    eval_texts: list[str],
    *,
    positive_label: str,
) -> tuple[list[str], list[float] | None]:
    """Fit on the training split and predict on an evaluation split.

    Returns predictions plus continuous scores for the positive class, used for
    ROC/PR. `LinearSVC` exposes no `predict_proba`, so its `decision_function` is
    used instead — deliberately not wrapped in `CalibratedClassifierCV`, which
    would change the reported model from "linear SVM" to "calibrated linear SVM"
    without helping the comparison this baseline exists to provide.
    """
    pipeline.fit(train_texts, train_labels)
    predictions = list(pipeline.predict(eval_texts))

    classifier = pipeline.named_steps["classifier"]
    scores: list[float] | None = None

    if hasattr(classifier, "predict_proba"):
        probabilities = pipeline.predict_proba(eval_texts)
        classes = list(classifier.classes_)
        if positive_label in classes:
            scores = [float(row[classes.index(positive_label)]) for row in probabilities]
    elif hasattr(classifier, "decision_function"):
        raw = np.asarray(pipeline.decision_function(eval_texts))
        classes = list(classifier.classes_)
        # Binary decision_function is signed toward classes_[1]; flip if the
        # configured positive label is classes_[0].
        if raw.ndim == 1 and len(classes) == 2:
            scores = [float(v) for v in (raw if classes[1] == positive_label else -raw)]

    return predictions, scores
