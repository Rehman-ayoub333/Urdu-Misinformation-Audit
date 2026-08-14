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
    SUPPORTED_CLASSIFIERS,
    SUPPORTED_FEATURES,
    build_length_pipeline,
    depth_sweep_diagnostic,
    extract_length_features,
    fit_predict_length,
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


# --- Experiment H2 — the nonlinear length-only baseline (M3-1) -----------------


def test_h2_uses_exactly_the_same_feature_registry_as_h(model_config: dict) -> None:
    """The load-bearing guard on the whole floor/ceiling comparison.

    H2 exists to isolate ONE difference from H — linear versus nonlinear. If the
    two feature lists ever drift apart, the gap between their scores stops being
    attributable to model shape and the pair reported for RQ3 becomes meaningless.
    """
    assert (
        model_config["experiments"]["H2"]["features"]
        == model_config["experiments"]["H"]["features"]
    )


def test_h2_features_are_all_length_only(model_config: dict) -> None:
    """Same content-blindness guarantee as H — asserted, not assumed."""
    for name in model_config["experiments"]["H2"]["features"]:
        assert name in SUPPORTED_FEATURES


def test_length_tree_predictions_are_deterministic(model_config: dict) -> None:
    """What makes H2 safe to run once, like H (EXPERIMENT_PLAN.md Section 5).

    Decision trees choose among equally-good splits using the RNG, so this is a
    real risk here rather than a formality: an unseeded tree can produce different
    thresholds — and a different macro-F1 — on identical data.
    """
    texts, labels = TEXTS * 8, LABELS * 8
    runs = []
    for _ in range(3):
        pipeline = build_length_pipeline(model_config, "H2")
        predictions, _, attribution = fit_predict_length(
            pipeline, texts, labels, texts, ["n_chars", "n_words"], positive_label="fake"
        )
        runs.append((predictions, attribution))
    assert runs[0] == runs[1] == runs[2]


def test_length_tree_uses_the_configured_seed_and_depth_cap(model_config: dict) -> None:
    """Seed from config, never a library default (REPRODUCIBILITY.md Section 2),
    and the depth cap is what keeps H2 a length-only *baseline* rather than an
    unconstrained model."""
    tree = build_length_pipeline(model_config, "H2").named_steps["classifier"]
    assert tree.random_state == model_config["random_state"]
    assert tree.max_depth == model_config["experiments"]["H2"]["params"]["max_depth"]
    assert tree.max_depth is not None and tree.max_depth <= 5


def test_length_tree_pipeline_does_not_standardise(model_config: dict) -> None:
    """H2's learned thresholds are reported in raw word/character counts.

    Scaling cannot change a tree's score, but it would turn "real articles sit
    between N and M words" into a statement about standard deviations, which is
    not readable as a finding.
    """
    assert "scaler" not in build_length_pipeline(model_config, "H2").named_steps
    assert "scaler" in build_length_pipeline(model_config, "H").named_steps


def test_length_pipeline_rejects_an_unsupported_classifier(model_config: dict) -> None:
    """H/H2 are a matched pair; a third variant appearing by accident would blur
    the floor/ceiling reading."""
    config = {
        **model_config,
        "experiments": {**model_config["experiments"], "H2": {"classifier": "random_forest"}},
    }
    with pytest.raises(ValueError, match="unsupported length-only classifier"):
        build_length_pipeline(config, "H2")


def test_supported_classifiers_map_to_their_experiment_ids() -> None:
    assert SUPPORTED_CLASSIFIERS == {
        "logistic_regression": "H",
        "decision_tree": "H2",
    }


def test_depth_sweep_diagnostic_is_deterministic(model_config: dict) -> None:
    texts, labels = TEXTS * 8, LABELS * 8
    first = depth_sweep_diagnostic(
        model_config, texts, labels, texts, labels, ["n_chars", "n_words"], depths=[2, 5]
    )
    second = depth_sweep_diagnostic(
        model_config, texts, labels, texts, labels, ["n_chars", "n_words"], depths=[2, 5]
    )
    assert first == second
    assert set(first["macro_f1_by_depth"]) == {"max_depth_2", "max_depth_5"}


def test_depth_sweep_does_not_mutate_the_config(model_config: dict) -> None:
    """The sweep overrides max_depth per probe. If it did so in place, H2's
    reported result would silently become whichever depth was probed last."""
    reported = model_config["experiments"]["H2"]["params"]["max_depth"]
    texts, labels = TEXTS * 8, LABELS * 8
    depth_sweep_diagnostic(
        model_config, texts, labels, texts, labels, ["n_chars", "n_words"], depths=[5]
    )
    assert model_config["experiments"]["H2"]["params"]["max_depth"] == reported


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
