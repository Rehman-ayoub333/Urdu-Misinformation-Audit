"""Experiment H — the length-only baseline. The direct operationalisation of RQ3.

`MASTER_PROJECT_BLUEPRINT.md` Part 32: a classical baseline is "the fastest way to
isolate the length shortcut — a linear model on word count alone". That is exactly
what this is.

**This model never sees a single word of article content.** Its only inputs are
counts derived from the text — how long it is, not what it says. Whatever macro-F1
it achieves is therefore predictive power available from length *alone*, with no
linguistic signal of any kind involved.

That number is the floor for interpreting every other experiment in the project. A
content-based model scoring at or near H has not been shown to have learned
anything beyond article length, however high its absolute accuracy looks. If H
scores near chance, the length confound is not doing meaningful work in this
dataset and RQ3's premise weakens.

Milestone 2's audit measured Ax-to-Grind's fake class at p95 = 724 subword tokens
against real's p95 = 77. That makes a high H score *plausible* — but nothing here
assumes it. The features are extracted, the model is fitted, and the number is
whatever it is (`CLAUDE.md` rule 2).

Feature extraction deliberately duplicates nothing from `audit.py`: both derive
counts from the same Tier-2-cleaned text, and `extract_length_features` is the
single definition used for training, evaluation and the determinism test.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

__all__ = [
    "SUPPORTED_FEATURES",
    "extract_length_features",
    "build_length_pipeline",
    "fit_predict_length",
]

_WORD = re.compile(r"\S+")

# Every feature this model may use. All are pure functions of text LENGTH; none
# inspects which words are present. Keeping the registry explicit means a content
# feature cannot be added by accident — which would silently invalidate the whole
# point of the experiment.
SUPPORTED_FEATURES: dict[str, Any] = {
    "n_chars": lambda text: float(len(text)),
    "n_words": lambda text: float(len(_WORD.findall(text))),
    "mean_word_length": lambda text: (
        float(np.mean([len(w) for w in _WORD.findall(text)])) if _WORD.findall(text) else 0.0
    ),
}


def extract_length_features(
    texts: Sequence[str], feature_names: Sequence[str]
) -> np.ndarray:
    """Build the (n_samples, n_features) length-feature matrix.

    Deterministic by construction: a pure function of the input strings with no
    randomness, no fitted vocabulary and no ordering dependence. The determinism
    test asserts this rather than trusting it.
    """
    unknown = [name for name in feature_names if name not in SUPPORTED_FEATURES]
    if unknown:
        raise ValueError(
            f"unknown length features {unknown}; supported: {sorted(SUPPORTED_FEATURES)}. "
            "Only length-derived features are permitted — adding a content-based "
            "feature here would invalidate experiment H."
        )

    return np.asarray(
        [[SUPPORTED_FEATURES[name](text) for name in feature_names] for text in texts],
        dtype=np.float64,
    )


def length_relationship_diagnostic(
    texts: Sequence[str],
    labels: Sequence[str],
    *,
    feature: str = "n_words",
    positive_label: str = "fake",
    n_bins: int = 10,
) -> dict[str, Any]:
    """Measure whether the length↔label relationship is monotonic.

    Written because H's headline number cannot be interpreted without it. A linear
    logistic model can only express "more length ⇒ more likely fake"; if the true
    relationship is non-monotonic, H's macro-F1 is a **floor on the available
    length shortcut, not a measure of it**, and reading it as the latter would
    understate how much of a content model's performance length alone explains.

    Reports the per-bin positive rate and the rank-based ROC-AUC of the raw
    feature. An AUC near 0.5 combined with strongly varying per-bin rates is the
    signature of a non-monotonic relationship — the two together say "length is
    predictive, but not in the direction a linear model can use".
    """
    from sklearn.metrics import roc_auc_score

    values = np.asarray([SUPPORTED_FEATURES[feature](text) for text in texts])
    binary = np.asarray([1 if label == positive_label else 0 for label in labels])

    edges = np.percentile(values, np.linspace(0, 100, n_bins + 1))
    bins = []
    for i in range(n_bins):
        lower, upper = edges[i], edges[i + 1]
        mask = (values >= lower) & (values <= upper if i == n_bins - 1 else values < upper)
        if not mask.any():
            continue
        bins.append(
            {
                "bin": i + 1,
                "range": [float(lower), float(upper)],
                "n": int(mask.sum()),
                f"{positive_label}_rate": round(float(binary[mask].mean()), 4),
            }
        )

    auc = (
        float(roc_auc_score(binary, values)) if len(set(binary.tolist())) == 2 else None
    )
    rates = [b[f"{positive_label}_rate"] for b in bins]
    monotonic = all(x <= y for x, y in zip(rates, rates[1:])) or all(
        x >= y for x, y in zip(rates, rates[1:])
    )

    return {
        "feature": feature,
        "roc_auc_of_raw_feature": round(auc, 4) if auc is not None else None,
        "bins": bins,
        "is_monotonic": bool(monotonic),
        "interpretation": (
            "Non-monotonic: the positive rate is high at BOTH extremes and low in "
            "the middle. Rank-based AUC near 0.5 therefore understates the "
            "relationship, and a LINEAR model cannot express it — H's macro-F1 is a "
            "floor on the available length shortcut, not a measurement of it."
            if not monotonic
            else "Monotonic: a linear model can express this relationship."
        ),
    }


def build_length_pipeline(config: dict[str, Any]) -> Pipeline:
    """StandardScaler -> LogisticRegression over length features only.

    Scaling matters here: raw counts span roughly 10 to 5,000, which makes
    logistic regression converge poorly and renders the fitted coefficients
    uninterpretable — and the coefficient sign is part of what makes this
    experiment readable ("longer => more likely fake").
    """
    experiment = config["experiments"]["H"]
    steps = []
    if experiment.get("standardise", True):
        steps.append(("scaler", StandardScaler()))
    steps.append(
        (
            "classifier",
            LogisticRegression(
                random_state=config["random_state"], **experiment.get("params", {})
            ),
        )
    )
    return Pipeline(steps)


def fit_predict_length(
    pipeline: Pipeline,
    train_texts: Sequence[str],
    train_labels: Sequence[str],
    eval_texts: Sequence[str],
    feature_names: Sequence[str],
    *,
    positive_label: str,
) -> tuple[list[str], list[float] | None, dict[str, float]]:
    """Fit and predict from length features alone.

    Also returns the fitted coefficient per feature, so the direction of the
    relationship is recorded in the metrics file rather than left to be inferred
    from the score.
    """
    x_train = extract_length_features(train_texts, feature_names)
    x_eval = extract_length_features(eval_texts, feature_names)

    pipeline.fit(x_train, list(train_labels))
    predictions = list(pipeline.predict(x_eval))

    classifier = pipeline.named_steps["classifier"]
    classes = list(classifier.classes_)
    scores = None
    if hasattr(classifier, "predict_proba") and positive_label in classes:
        probabilities = pipeline.predict_proba(x_eval)
        scores = [float(row[classes.index(positive_label)]) for row in probabilities]

    # Coefficients are signed toward classes_[1] in scikit-learn's binary case.
    raw = np.asarray(classifier.coef_).ravel()
    oriented = raw if classes[1] == positive_label else -raw
    coefficients = {
        name: float(value) for name, value in zip(feature_names, oriented)
    }

    return predictions, scores, coefficients
