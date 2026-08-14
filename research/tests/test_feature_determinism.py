"""Feature-extraction determinism (`ROADMAP.md` Milestone 3 test requirement).

Same input must produce the same features on every run. If it does not, a metrics
file cannot be reproduced from its config and `REPRODUCIBILITY.md` Section 4's
whole premise fails — a reviewer re-running the experiment would get different
numbers with no way to tell whether the difference is real.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from research.src.experiments.run_in_domain import CONFIG_DIR
from research.src.models.classical import build_pipeline, build_vectorizer
from research.src.models.length_baseline import (
    SUPPORTED_FEATURES,
    extract_length_features,
    length_relationship_diagnostic,
)

TEXTS = [
    "پاکستان کی معاشی صورتحال بہتر ہو رہی ہے۔",
    "وزیراعظم نے COVID-19 کے بارے میں ۲۰۲۱ میں بات کی۔",
    "کراچی میں آج موسم خوشگوار رہا اور بارش کا امکان ہے۔",
    "یہ ایک مختصر خبر ہے۔",
]
LABELS = ["fake", "real", "real", "fake"]


@pytest.fixture(scope="module")
def model_config() -> dict:
    return yaml.safe_load((CONFIG_DIR / "model_classical.yaml").read_text(encoding="utf-8"))


# --- Length features -----------------------------------------------------------


def test_length_features_are_deterministic() -> None:
    names = ["n_chars", "n_words"]
    first = extract_length_features(TEXTS, names)
    for _ in range(5):
        np.testing.assert_array_equal(first, extract_length_features(TEXTS, names))


def test_length_features_do_not_depend_on_input_order() -> None:
    """Row i's features must depend only on row i's text.

    A vectoriser that fitted state across rows would break this, and would make
    a train/serve mismatch possible at Milestone 6.
    """
    names = ["n_chars", "n_words"]
    forward = extract_length_features(TEXTS, names)
    reverse = extract_length_features(list(reversed(TEXTS)), names)
    np.testing.assert_array_equal(forward, reverse[::-1])


def test_length_features_have_expected_values() -> None:
    """Pins the actual definitions, so a silent change to what is counted fails."""
    features = extract_length_features(["ab cd ef"], ["n_chars", "n_words"])
    assert features[0][0] == 8.0
    assert features[0][1] == 3.0


def test_length_feature_registry_rejects_unknown_features() -> None:
    """Only length-derived features are permitted — a content feature would
    invalidate experiment H entirely."""
    with pytest.raises(ValueError, match="unknown length features"):
        extract_length_features(TEXTS, ["n_words", "tfidf_of_the_word_breaking"])


def test_every_registered_feature_is_length_derived() -> None:
    """Guardrail on the registry itself: features must be insensitive to WHICH
    words appear, only to how many/how long."""
    for name, fn in SUPPORTED_FEATURES.items():
        same_shape_different_words = fn("aaa bbb ccc")
        assert fn("xxx yyy zzz") == same_shape_different_words, (
            f"feature {name!r} changes with word identity — it is not length-only"
        )


def test_length_diagnostic_is_deterministic() -> None:
    first = length_relationship_diagnostic(TEXTS * 10, LABELS * 10, n_bins=2)
    second = length_relationship_diagnostic(TEXTS * 10, LABELS * 10, n_bins=2)
    assert first == second


# --- TF-IDF features -----------------------------------------------------------


def test_tfidf_vectorizer_is_deterministic(model_config: dict) -> None:
    """Same corpus in, byte-identical matrix out."""
    first = build_vectorizer(model_config).fit_transform(TEXTS)
    second = build_vectorizer(model_config).fit_transform(TEXTS)
    np.testing.assert_array_equal(first.toarray(), second.toarray())


def test_tfidf_transform_is_deterministic_after_fit(model_config: dict) -> None:
    """A fitted vectoriser must transform the same text identically every call."""
    vectorizer = build_vectorizer(model_config)
    vectorizer.fit(TEXTS)
    np.testing.assert_array_equal(
        vectorizer.transform(TEXTS).toarray(), vectorizer.transform(TEXTS).toarray()
    )


def test_classical_pipeline_predictions_are_deterministic(model_config: dict) -> None:
    """Two pipelines built from the same config and data must agree exactly.

    This is what makes A and B safe to run once rather than over the transformer
    seed triple (EXPERIMENT_PLAN.md Section 5).
    """
    texts = TEXTS * 8
    labels = LABELS * 8
    predictions = []
    for _ in range(2):
        pipeline = build_pipeline(model_config, "A")
        pipeline.fit(texts, labels)
        predictions.append(list(pipeline.predict(texts)))
    assert predictions[0] == predictions[1]


def test_classical_pipeline_uses_the_configured_seed(model_config: dict) -> None:
    """random_state must come from config, never a library default."""
    pipeline = build_pipeline(model_config, "A")
    assert pipeline.named_steps["classifier"].random_state == model_config["random_state"]
    assert model_config["random_state"] == 42
