"""Tests for Experiment I — length ablation (RQ3, replicating Haroon's 50-word cap).

Grouped by what would hurt most if it broke:

1. **The cap itself.** I's claim is "the only difference from the uncapped
   baseline is that every article is at most N words". If the rule is not exactly
   right-truncation on whitespace-delimited words, applied to both sides, the
   delta measures something other than the length confound.
2. **Scope**, which resolves a real spec ambiguity — the sweep must contain the
   primary cap 50, must be in-domain only, and the classical half must be the
   *measured* best baseline rather than a remembered one.
3. **Wiring** — shared config with D, branch naming inside the existing repo, and
   `clean.py` left alone.
4. **Committed output**, if any exists yet; skipped until the Kaggle run lands.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from research.src.data.split import load_split_frame
from research.src.evaluation.metrics import METRICS_DIR, predictions_filename
from research.src.experiments.length_ablation import (
    CAP_VALUES,
    CLASSICAL_BASELINE_ID,
    CLASSICAL_EXPERIMENT_ID,
    DATASET,
    EVAL_SPLIT,
    EXPERIMENT_ID,
    HAROON_CAP50_MACRO_F1_DROP,
    PRIMARY_CAP,
    TRANSFORMER_BASELINE_ID,
    TRANSFORMER_CONFIG_NAME,
    ablation_metadata,
    cap_frame,
    cap_words,
    classical_baseline_filename,
    classical_metrics_filename,
    comparison_metadata,
    length_profile,
    transformer_baseline_filename,
    transformer_metrics_filename,
)

REQUIRED_SEEDS = [42, 123, 2026]


@pytest.fixture(scope="module")
def train_frame():  # noqa: ANN201 - pandas DataFrame
    """Loaded once for the module.

    `load_split_frame` re-reads the raw corpus and re-cleans every row, so the
    per-test cost is seconds; several tests here need the FULL split (truncation
    monotonicity, per-label profiles) rather than a slice, and reloading it for
    each was the dominant cost of this file.
    """
    return load_split_frame(DATASET, "train")


def _load_config(name: str) -> dict:
    import yaml

    from research.src.models.transformer import CONFIG_DIR

    return yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# the cap rule
# --------------------------------------------------------------------------


def test_cap_keeps_the_first_n_words() -> None:
    """Right-truncation, matching the 512-subword ceiling (M2-1/M4-1)."""
    assert cap_words("a b c d e", 3) == "a b c"


def test_cap_drops_from_the_right_not_the_left() -> None:
    """The direction is the whole point of the M2-1/M4-1 consistency argument.

    Left-truncation would keep article endings while the subword ceiling keeps
    beginnings — two length limits in one pipeline disagreeing about which end of
    an article survives.
    """
    capped = cap_words("first second third fourth", 2)
    assert capped == "first second"
    assert not capped.endswith("fourth")


def test_short_articles_are_untouched() -> None:
    """The ablation may only remove length information, never add it."""
    assert cap_words("a b", 50) == "a b"


def test_cap_counts_the_same_words_the_length_baselines_do(train_frame) -> None:
    """H/H2 count `n_words` by whitespace split; the cap must use the same unit.

    If the two disagreed, "capped at 50 words" and "n_words = 50" would mean
    different things in the same thesis chapter.
    """
    text = train_frame["text"].iloc[0]
    assert len(cap_words(text, 25).split()) == min(25, len(text.split()))


@pytest.mark.parametrize("cap", CAP_VALUES)
def test_no_capped_article_exceeds_the_cap(cap: int, train_frame) -> None:
    frame = cap_frame(train_frame.head(200), cap)
    assert max(len(text.split()) for text in frame["text"]) <= cap


def test_zero_or_negative_cap_is_rejected() -> None:
    """A cap of 0 would silently produce an empty corpus and a meaningless result."""
    with pytest.raises(ValueError, match="positive word count"):
        cap_words("a b c", 0)


def test_non_string_input_does_not_crash() -> None:
    assert cap_words(None, 50) == ""  # type: ignore[arg-type]


def test_cap_frame_leaves_labels_and_ids_untouched(train_frame) -> None:
    frame = train_frame.head(50)
    capped = cap_frame(frame, 25)
    assert capped["row_id"].tolist() == frame["row_id"].tolist()
    assert capped["label"].tolist() == frame["label"].tolist()


def test_cap_frame_does_not_mutate_its_input(train_frame) -> None:
    """The recorded before/after profile would be identical under in-place edits."""
    frame = train_frame.head(50).copy()
    original = frame["text"].tolist()
    cap_frame(frame, 10)
    assert frame["text"].tolist() == original


# --------------------------------------------------------------------------
# the profile — the confound is a per-label property, so the record must be too
# --------------------------------------------------------------------------


def test_profile_reports_truncation_per_label(train_frame) -> None:
    """M3-1: length correlates with the label, so an aggregate figure hides which
    class the cap actually bit into."""
    profile = length_profile(train_frame, PRIMARY_CAP)
    assert set(profile["by_label"]) == {"fake", "real"}
    for stats in profile["by_label"].values():
        assert stats["mean_words_after"] <= stats["mean_words_before"]
        assert 0 <= stats["pct_truncated"] <= 100


def test_a_tighter_cap_truncates_at_least_as_much(train_frame) -> None:
    """Monotonicity: a smaller cap cannot truncate fewer rows."""
    counts = [length_profile(train_frame, cap)["n_truncated"] for cap in sorted(CAP_VALUES)]
    assert counts == sorted(counts, reverse=True)


def test_ablation_metadata_records_both_sides(train_frame) -> None:
    """Capping only at eval time would measure a distribution shift, not an ablation."""
    train = train_frame.head(100)
    evaluation = load_split_frame(DATASET, EVAL_SPLIT).head(100)
    ablation = ablation_metadata(cap=PRIMARY_CAP, train_frame=train, eval_frame=evaluation)[
        "ablation"
    ]

    assert ablation["type"] == "length_capping"
    assert ablation["applied_to_both_sides"] is True
    assert ablation["train_profile"]["cap_words"] == PRIMARY_CAP
    assert ablation["eval_profile"]["cap_words"] == PRIMARY_CAP
    assert ablation["is_primary_cap"] is True


def test_non_primary_cap_is_flagged_as_such(train_frame) -> None:
    """A curve point must never be mistaken for the replication."""
    frame = train_frame.head(20)
    ablation = ablation_metadata(cap=200, train_frame=frame, eval_frame=frame)["ablation"]
    assert ablation["is_primary_cap"] is False


# --------------------------------------------------------------------------
# scope — the spec ambiguity, resolved and pinned
# --------------------------------------------------------------------------


def test_sweep_contains_the_primary_replication_cap() -> None:
    """Part 1/11 specify Haroon's 50-word cap; Part 31 figure 10 wants a curve.

    The sweep satisfies both, but only if 50 is actually in it — without it the
    experiment is an extension with no replication in it, and the 0.0067 figure
    the blueprint cites has nothing to be compared against.
    """
    assert PRIMARY_CAP == 50
    assert PRIMARY_CAP in CAP_VALUES


def test_sweep_has_enough_points_to_draw_a_curve() -> None:
    """Part 31 figure 10 is 'F1 vs word cap' — one point is not a curve."""
    assert len(CAP_VALUES) >= 3
    assert list(CAP_VALUES) == sorted(CAP_VALUES), "caps should read left-to-right on the x axis"


def test_experiment_is_in_domain_only() -> None:
    """Cross-dataset transfer is F/G/Q's question. Mixing them would make the
    delta uninterpretable — two confounds moving at once."""
    assert DATASET == "ax_to_grind"
    assert EVAL_SPLIT == "test"


def test_classical_half_is_the_measured_best_baseline() -> None:
    """'Best classical' must trace to the committed numbers, not to memory.

    If A ever overtakes B on a re-run, this fails rather than letting I quietly
    keep using the weaker baseline.
    """
    a_path = METRICS_DIR / "A_tfidf_logreg_ax_to_grind_test.json"
    b_path = METRICS_DIR / "B_tfidf_svm_ax_to_grind_test.json"
    if not (a_path.exists() and b_path.exists()):
        pytest.skip("A/B in-domain results not committed")

    a = json.loads(a_path.read_text(encoding="utf-8"))["metrics"]["macro_f1"]
    b = json.loads(b_path.read_text(encoding="utf-8"))["metrics"]["macro_f1"]
    assert b > a, f"B {b:.4f} no longer beats A {a:.4f} — I's classical half must follow"
    assert CLASSICAL_EXPERIMENT_ID == "B"


def test_only_one_classical_model_is_run() -> None:
    """Part 11 says 'best classical', singular — not both A and B."""
    assert isinstance(CLASSICAL_EXPERIMENT_ID, str)
    assert CLASSICAL_BASELINE_ID == CLASSICAL_EXPERIMENT_ID


def test_transformer_half_uses_d_s_config_and_seeds() -> None:
    """The delta is attributable to the cap only if nothing else moved."""
    assert TRANSFORMER_CONFIG_NAME == "model_xlmr.yaml"
    config = _load_config(TRANSFORMER_CONFIG_NAME)
    assert config["model"]["experiment_id"] == TRANSFORMER_BASELINE_ID
    assert config["training"]["seeds"] == REQUIRED_SEEDS


# --------------------------------------------------------------------------
# filenames and the comparison block
# --------------------------------------------------------------------------


def test_transformer_filename_matches_the_agreed_pattern() -> None:
    assert transformer_metrics_filename(42, 50, "xlm-roberta-base") == (
        "I_xlm-roberta-base_ax_to_grind_test_seed42_cap50.json"
    )


def test_classical_filename_matches_the_agreed_pattern() -> None:
    assert classical_metrics_filename(100, "tfidf_svm") == (
        "I_tfidf_svm_ax_to_grind_test_cap100.json"
    )


def test_filenames_are_unique_per_cap() -> None:
    """Without the cap in the name, four sweep points would overwrite each other."""
    names = {transformer_metrics_filename(42, cap, "xlm-roberta-base") for cap in CAP_VALUES}
    assert len(names) == len(CAP_VALUES)


@pytest.mark.parametrize("seed", REQUIRED_SEEDS)
def test_transformer_comparison_matches_the_committed_d_file(seed: int) -> None:
    filename = transformer_baseline_filename(seed, "xlm-roberta-base")
    path = METRICS_DIR / filename
    if not path.exists():
        pytest.skip(f"{filename} not committed")

    recorded = comparison_metadata(
        cap=PRIMARY_CAP,
        model_name="xlm-roberta-base",
        seed=seed,
        baseline_filename=filename,
        baseline_id=TRANSFORMER_BASELINE_ID,
    )["compares_against"]["uncapped_baseline"]

    expected = json.loads(path.read_text(encoding="utf-8"))["metrics"]["macro_f1"]
    assert recorded["macro_f1"] == pytest.approx(expected, abs=1e-12)
    assert recorded["seed"] == seed
    assert recorded["same_seed"] is True


def test_classical_comparison_matches_the_committed_b_file() -> None:
    filename = classical_baseline_filename("tfidf_svm")
    path = METRICS_DIR / filename
    if not path.exists():
        pytest.skip(f"{filename} not committed")

    recorded = comparison_metadata(
        cap=PRIMARY_CAP,
        model_name="tfidf_svm",
        seed=None,
        baseline_filename=filename,
        baseline_id=CLASSICAL_BASELINE_ID,
    )["compares_against"]["uncapped_baseline"]

    expected = json.loads(path.read_text(encoding="utf-8"))["metrics"]["macro_f1"]
    assert recorded["macro_f1"] == pytest.approx(expected, abs=1e-12)


def test_haroon_reference_is_attached_only_to_the_primary_cap() -> None:
    """The 0.0067 figure is about a 50-word cap specifically; attaching it to a
    200-word run would invite a comparison that does not exist."""
    primary = comparison_metadata(
        cap=PRIMARY_CAP,
        model_name="xlm-roberta-base",
        seed=42,
        baseline_filename="whatever.json",
        baseline_id="D",
    )["compares_against"]
    other = comparison_metadata(
        cap=200,
        model_name="xlm-roberta-base",
        seed=42,
        baseline_filename="whatever.json",
        baseline_id="D",
    )["compares_against"]

    assert "reference_finding" in primary
    assert "reference_finding" not in other


def test_haroon_reference_is_labelled_as_external() -> None:
    """`CLAUDE.md` rule 2: a literature value must never read as ours."""
    reference = comparison_metadata(
        cap=PRIMARY_CAP,
        model_name="xlm-roberta-base",
        seed=42,
        baseline_filename="whatever.json",
        baseline_id="D",
    )["compares_against"]["reference_finding"]

    assert reference["is_this_projects_measurement"] is False
    assert reference["reported_macro_f1_drop"] == HAROON_CAP50_MACRO_F1_DROP
    assert "arXiv:2607.14131" in reference["source"]


def test_missing_counterpart_records_null_not_a_guess() -> None:
    recorded = comparison_metadata(
        cap=25,
        model_name="no-such-model",
        seed=999,
        baseline_filename="D_no-such-model_ax_to_grind_test_seed999.json",
        baseline_id="D",
    )["compares_against"]["uncapped_baseline"]
    assert recorded["macro_f1"] is None


# --------------------------------------------------------------------------
# wiring
# --------------------------------------------------------------------------


def test_checkpoints_branch_inside_the_existing_staging_repo() -> None:
    """One repo per model, one branch per (seed, cap). Not a repo per cap."""
    from research.src.models.transformer import push_checkpoint

    config = _load_config(TRANSFORMER_CONFIG_NAME)
    result = push_checkpoint(
        None, None, config, seed=42, dry_run=True, branch_suffix="-length-cap-50"
    )
    assert result["would_use_branch"] == "seed-42-length-cap-50"
    assert result["would_push_to"].endswith("urdu-misinfo-xlmr-staging")


def test_clean_py_does_not_contain_the_cap() -> None:
    """`CLAUDE.md` rule 4, and Part 8's explicit non-decision: length capping in
    the shared preprocessing path would change every other experiment and
    production inference, and pre-empt RQ3's own measurement."""
    from pathlib import Path

    from research.src.data import clean

    source = Path(clean.__file__).read_text(encoding="utf-8")
    assert "cap_words" not in source
    assert "length_ablation" not in source


def test_clean_text_still_returns_full_length_text() -> None:
    """The behavioural half of the check above."""
    from research.src.data.clean import clean_text

    long_text = " ".join(["کلمہ"] * 300)
    assert len(clean_text(long_text).split()) == 300


def test_torch_is_never_imported_at_module_scope() -> None:
    """The classical half of I must stay runnable without torch."""
    from pathlib import Path

    from research.src.experiments import length_ablation

    source = Path(length_ablation.__file__).read_text(encoding="utf-8")
    for line in source.splitlines():
        if line.startswith("import ") or line.startswith("from "):
            assert "torch" not in line, f"torch imported at module scope: {line!r}"
            assert "models.transformer" not in line, f"training code at module scope: {line!r}"
            assert "models.classical" not in line, f"sklearn model at module scope: {line!r}"


def test_shortcut_analysis_declares_the_mode_as_implemented() -> None:
    from research.src.experiments.run_shortcut_analysis import (
        LENGTH_ABLATION_MODE,
        MILESTONE_5_MODES,
    )

    assert LENGTH_ABLATION_MODE == "length-ablation"
    assert LENGTH_ABLATION_MODE not in MILESTONE_5_MODES


def test_shortcut_analysis_dispatches_to_the_i_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Including that the cap and model-family flags survive the hop."""
    from research.src.experiments import length_ablation, run_shortcut_analysis

    captured: list[list[str]] = []

    def fake_main(argv: list[str]) -> int:
        captured.append(argv)
        return 0

    monkeypatch.setattr(length_ablation, "main", fake_main)

    exit_code = run_shortcut_analysis.main(
        [
            "--mode",
            "length-ablation",
            "--dry-run",
            "--caps",
            "50",
            "--ablation-models",
            "classical",
        ]
    )

    assert exit_code == 0
    assert captured == [
        ["--data-config", "data.yaml", "--dry-run", "--caps", "50", "--models", "classical"]
    ]


# --------------------------------------------------------------------------
# committed I results — skipped until the Kaggle run lands
# --------------------------------------------------------------------------


def _i_files() -> list:
    return sorted(METRICS_DIR.glob("I_*.json")) if METRICS_DIR.exists() else []


@pytest.mark.parametrize("path", _i_files(), ids=lambda p: p.name)
def test_committed_i_file_records_its_cap(path) -> None:
    record = json.loads(path.read_text(encoding="utf-8"))
    ablation = record["run_metadata"]["ablation"]

    assert ablation["type"] == "length_capping"
    assert ablation["applied_to_both_sides"] is True
    assert f"_cap{ablation['cap_words']}." in path.name, (
        f"{path.name} does not name the cap its metadata records"
    )
    assert f"capped_{ablation['cap_words']}_words" in record["run_metadata"]["train_split"]


@pytest.mark.parametrize("path", _i_files(), ids=lambda p: p.name)
def test_committed_i_file_records_a_computable_delta(path) -> None:
    record = json.loads(path.read_text(encoding="utf-8"))
    comparison = record["run_metadata"]["compares_against"]
    baseline_path = METRICS_DIR / comparison["uncapped_baseline"]["metrics_file"]

    assert comparison["macro_f1_capped"] == pytest.approx(
        record["metrics"]["macro_f1"], abs=1e-12
    )
    if not baseline_path.exists():
        pytest.skip(f"{baseline_path.name} not committed — nothing to cross-check")

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert baseline["dataset"] == record["dataset"]
    assert baseline["split"] == record["split"]
    if comparison["uncapped_baseline"]["same_seed"]:
        assert baseline["seed"] == record["seed"], "I must compare against the SAME seed"
    assert comparison["delta_macro_f1_i_minus_uncapped"] == pytest.approx(
        record["metrics"]["macro_f1"] - baseline["metrics"]["macro_f1"], abs=1e-6
    )


@pytest.mark.parametrize("path", _i_files(), ids=lambda p: p.name)
def test_committed_i_file_has_per_example_predictions(path) -> None:
    """I's eval split is `test`, so Section 6 applies. Asserted, not skipped —
    I is written with the M5-2 machinery already in place, so a missing sibling
    is a regression rather than a legacy gap."""
    sibling = path.parent / predictions_filename(path.name)
    assert sibling.exists(), f"{path.name} has no per-example predictions sibling"

    record = json.loads(path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in sibling.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == record["n_examples"]


def test_the_primary_cap_is_present_once_the_sweep_is_committed() -> None:
    """If any I result exists, cap 50's must be among them — it is the
    replication, and a sweep committed without it would misrepresent the
    experiment as done."""
    files = _i_files()
    if not files:
        pytest.skip("Experiment I has not been run yet — pending the GPU session")
    assert any(f"_cap{PRIMARY_CAP}." in path.name for path in files), (
        f"I results exist but none at the primary cap {PRIMARY_CAP}"
    )


def test_experiment_id_is_stable() -> None:
    assert EXPERIMENT_ID == "I"


# --------------------------------------------------------------------------
# the real-run guard — regression test for an actual incident
# --------------------------------------------------------------------------


def test_a_real_run_is_refused_without_explicit_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bare invocation must not start training or write committed results.

    This is a regression test, not a precaution. A test asserting that
    `--mode length-ablation` exits cleanly as an unimplemented mode kept naming
    that mode after I was implemented — so it launched a real run, wrote
    `I_tfidf_svm_ax_to_grind_test_cap25.json` into `research/results/metrics/`
    and started fine-tuning XLM-R on CPU. The guard makes an accidental caller
    exit 2 instead.

    Verified by outcome as well as exit code: nothing may be written, and neither
    training path may be entered.
    """
    from research.src.experiments import length_ablation

    def explode(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("a real run started without confirmation")

    monkeypatch.setattr(length_ablation, "run_length_ablation", explode)
    monkeypatch.setattr(length_ablation, "run_classical_cap", explode)
    monkeypatch.setattr(length_ablation, "run_transformer_cap_seed", explode)

    before = sorted(METRICS_DIR.glob("I_*")) if METRICS_DIR.exists() else []
    assert length_ablation.main([]) == 2
    after = sorted(METRICS_DIR.glob("I_*")) if METRICS_DIR.exists() else []
    assert after == before, "a refused run still touched research/results/metrics/"


def test_a_dry_run_is_not_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bug-check writes nothing and pushes nothing, so it stays casual to run."""
    from research.src.experiments import length_ablation

    called: list[bool] = []

    def fake_run(**kwargs):  # noqa: ANN003, ANN202
        called.append(kwargs["dry_run"] is not None)
        return []

    monkeypatch.setattr(length_ablation, "run_length_ablation", fake_run)
    assert length_ablation.main(["--dry-run"]) == 0
    assert called == [True]


def test_confirmed_real_run_is_allowed_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard must not block the GPU notebook, which passes the flag once."""
    from research.src.experiments import length_ablation

    called: list[Any] = []

    def fake_run(**kwargs):  # noqa: ANN003, ANN202
        called.append(kwargs["dry_run"])
        return []

    monkeypatch.setattr(length_ablation, "run_length_ablation", fake_run)
    assert length_ablation.main(["--confirm-real-run"]) == 0
    assert called == [None], "a confirmed run must be a real run, not a dry one"
