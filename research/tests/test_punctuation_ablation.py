"""Tests for Experiment Q — punctuation-surface-form ablation (M2-3).

Three groups, in order of what would hurt most if it broke:

1. **The transform itself.** Q's entire claim is "the only difference from F is
   that the training corpus has no punctuation". If the stripping rule is not
   exactly what the docstring says, the finding is about something else. The
   regex is therefore cross-checked against a literal, character-by-character
   reading of the definition across the Unicode range, and against the real
   Ax-to-Grind and Notri-Fact text.
2. **Wiring.** That Q reuses D's config and D's staging repo rather than a
   near-duplicate that could drift, and that it never touches `clean.py`.
3. **Committed output**, if any exists yet — skipped until the Kaggle run lands,
   the same pattern `test_transformer_pipeline.py` uses for checkpoints.
"""

from __future__ import annotations

import json

import pytest

from research.src.data.split import load_split_frame
from research.src.evaluation.metrics import METRICS_DIR, predictions_filename
from research.src.experiments.punctuation_ablation import (
    CHECKPOINT_BRANCH_SUFFIX,
    EXPERIMENT_ID,
    MODEL_CONFIG_NAME,
    Q_DIRECTION,
    _reference_strip,
    ablation_metadata,
    baseline_metrics_filename,
    comparison_metadata,
    in_domain_metrics_filename,
    non_alphanumeric_profile,
    strip_punctuation,
    strip_punctuation_frame,
)
from research.src.experiments.run_cross_dataset import DIRECTIONS, transfer_metadata

REQUIRED_SEEDS = [42, 123, 2026]


def _load_config(name: str) -> dict:
    import yaml

    from research.src.models.transformer import CONFIG_DIR

    return yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# the transform — exactly the documented rule, no more and no less
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("پاکستان کی معاشی صورتحال بہتر ہو رہی ہے۔", "پاکستان کی معاشی صورتحال بہتر ہو رہی ہے"),
        # Urdu full stop, comma and question mark — M2-3's named characters.
        ("الف۔ بے، جیم؟", "الف بے جیم"),
        # Digits survive: the rule keeps letters AND numbers.
        ("سال 2026 میں ۱۲ افراد", "سال 2026 میں ۱۲ افراد"),
        # Latin punctuation, quotes, brackets and symbols all go.
        ('He said "no" (twice) — 50% #tag @user', "He said no twice 50 tag user"),
        # Underscore is connector punctuation, not a letter or digit.
        ("snake_case_word", "snake case word"),
        ("", ""),
    ],
)
def test_strip_punctuation_matches_the_documented_rule(raw: str, expected: str) -> None:
    assert strip_punctuation(raw) == expected


def test_stripping_leaves_no_non_alphanumeric_characters() -> None:
    """The property that makes Q's surfaces match Notri-Fact's."""
    stripped = strip_punctuation('a۔b، c؟ d—e "f" (g) [h] 100% #i @j k_l')
    assert non_alphanumeric_profile([stripped])["n_non_alphanumeric"] == 0


def test_whitespace_runs_are_collapsed() -> None:
    """Requirement from M2-3's implementation note, not a tidiness preference.

    Replacing an inter-word character with nothing (or with a space, un-collapsed)
    would leave a double space exactly where punctuation used to be — trading the
    punctuation confound for a whitespace-density one that correlates with the
    same thing. `n_words` and the whitespace count must be unaffected by the
    ablation.
    """
    stripped = strip_punctuation("الف ۔ بے   ،   جیم")
    assert "  " not in stripped
    assert stripped == "الف بے جیم"
    assert stripped == stripped.strip()


def test_removed_characters_leave_a_word_boundary() -> None:
    """The one sub-decision inside Q's definition that M2-3 did not fix.

    Deleting outright fuses the words either side into tokens that appear in
    neither corpus (`ڈیبٹ/کریڈٹ` -> `ڈیبٹکریڈٹ`), injecting a new
    out-of-vocabulary artefact into the corpus whose confounds Q isolates. Pinned
    by test because it is a decision, not an implementation detail — it differs
    on 11.43% of the real training rows.
    """
    assert strip_punctuation("ڈیبٹ/کریڈٹ") == "ڈیبٹ کریڈٹ"
    assert strip_punctuation("COVID-19") == "COVID 19"
    assert strip_punctuation("پابندی'چکن") == "پابندی چکن"


def test_regex_agrees_with_the_literal_definition_across_unicode() -> None:
    """The regex must implement "letter, digit or whitespace", not something near it.

    Swept over the whole Basic Multilingual Plane rather than a handful of
    examples: a category the regex and the docstring disagree about is a silent
    change to what Q's training corpus is.
    """
    sample = "".join(
        chr(codepoint)
        for codepoint in range(0x0000, 0x10000, 7)
        if chr(codepoint).isprintable() or chr(codepoint).isspace()
    )
    assert strip_punctuation(sample) == _reference_strip(sample)


def test_regex_agrees_with_the_literal_definition_on_the_real_corpus() -> None:
    """The synthetic sweep above cannot cover real Urdu text's actual mix."""
    frame = load_split_frame("ax_to_grind", "train").head(300)
    for text in frame["text"]:
        assert strip_punctuation(text) == _reference_strip(text)


def test_non_string_input_does_not_crash() -> None:
    """Same contract as clean.py: a stray NaN must not kill a training run."""
    assert strip_punctuation(None) == ""  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# the ablation is applied to the right corpus, and only to it
# --------------------------------------------------------------------------


def test_strip_punctuation_frame_leaves_labels_and_ids_untouched() -> None:
    frame = load_split_frame("ax_to_grind", "train").head(50)
    ablated = strip_punctuation_frame(frame)

    assert ablated["row_id"].tolist() == frame["row_id"].tolist()
    assert ablated["label"].tolist() == frame["label"].tolist()
    assert len(ablated) == len(frame)


def test_strip_punctuation_frame_does_not_mutate_its_input() -> None:
    """Q records a before/after profile; an in-place map would make them identical."""
    frame = load_split_frame("ax_to_grind", "train").head(50)
    original = frame["text"].tolist()
    strip_punctuation_frame(frame)
    assert frame["text"].tolist() == original


def test_ax_to_grind_actually_has_punctuation_to_remove() -> None:
    """M2-3's premise for the source corpus, re-measured rather than quoted."""
    profile = non_alphanumeric_profile(load_split_frame("ax_to_grind", "train")["text"])
    assert profile["n_non_alphanumeric"] > 0
    assert profile["n_distinct_non_alphanumeric"] > 1


def test_notri_fact_is_already_punctuation_free() -> None:
    """M2-3's other premise. If this ever fails, Q's surfaces do not match after
    the ablation and the delta means something narrower than Q claims."""
    profile = non_alphanumeric_profile(load_split_frame("notri_fact", "holdout")["text"])
    assert profile["n_non_alphanumeric"] == 0, (
        "Notri-Fact contains non-alphanumeric characters: "
        f"{profile['most_common']} — M2-3 measured zero"
    )


def test_ablation_metadata_reports_a_matched_surface() -> None:
    train = load_split_frame("ax_to_grind", "train").head(200)
    target = load_split_frame("notri_fact", "holdout").head(200)
    metadata = ablation_metadata(
        train_before=train, train_after=strip_punctuation_frame(train), target=target
    )["ablation"]

    assert metadata["characters_removed"] > 0
    assert metadata["train_after"]["n_non_alphanumeric"] == 0
    assert metadata["surfaces_match_after_ablation"] is True
    assert "notri_fact_holdout" in metadata["not_applied_to"][0]


# --------------------------------------------------------------------------
# clean.py is the shared train/serve path and Q must not have touched it
# --------------------------------------------------------------------------


def test_clean_py_does_not_contain_the_ablation() -> None:
    """`CLAUDE.md` rule 4. Q's ablation lives beside its runner, never in the
    shared preprocessing module the backend imports — a change there would alter
    every other experiment and production inference alike."""
    from pathlib import Path

    from research.src.data import clean

    source = Path(clean.__file__).read_text(encoding="utf-8")
    assert "strip_punctuation" not in source
    assert "punctuation_ablation" not in source


def test_clean_text_still_preserves_punctuation() -> None:
    """The behavioural half of the check above: proof by output, not by grep."""
    from research.src.data.clean import clean_text

    assert "۔" in clean_text("الف۔ بے")


# --------------------------------------------------------------------------
# wiring — scope fixed by M2-3, and no config drift against D
# --------------------------------------------------------------------------


def test_q_uses_the_xlm_r_config_that_experiment_d_uses() -> None:
    """Q's delta is attributable to the surface form ONLY if nothing else moved.

    A separate Q config could drift from D's by one line and turn the finding into
    a hyperparameter artefact, so Q reuses D's file rather than copying it.
    """
    assert MODEL_CONFIG_NAME == "model_xlmr.yaml"
    config = _load_config(MODEL_CONFIG_NAME)
    assert config["model"]["experiment_id"] == "D"
    assert config["model"]["name"] == "xlm-roberta-base"


def test_q_runs_the_required_three_seeds() -> None:
    assert _load_config(MODEL_CONFIG_NAME)["training"]["seeds"] == REQUIRED_SEEDS


def test_q_direction_is_ax_to_grind_to_notri_fact_only() -> None:
    """M2-3 scoped Q to the direction where the mechanism applies."""
    assert Q_DIRECTION["source_dataset"] == "ax_to_grind"
    assert Q_DIRECTION["target_dataset"] == "notri_fact"
    assert Q_DIRECTION["target_split"] == "holdout"
    assert (Q_DIRECTION["source_dataset"], Q_DIRECTION["target_dataset"]) == (
        DIRECTIONS["F"]["source_dataset"],
        DIRECTIONS["F"]["target_dataset"],
    )


def test_q_declares_itself_zero_shot_on_an_untrained_target() -> None:
    transfer = transfer_metadata(EXPERIMENT_ID, Q_DIRECTION)["transfer"]
    assert transfer["direction_id"] == "Q"
    assert transfer["zero_shot"] is True
    assert transfer["target_used_for_training"] is False
    assert transfer["source_dataset"] != transfer["target_dataset"]


def test_q_is_not_added_to_the_cross_dataset_direction_table() -> None:
    """Q must not become selectable via `run_cross_dataset.py --directions`.

    That runner scores checkpoints against the corpus as released; running Q
    through it would produce a `Q_*.json` with no ablation applied — a file
    claiming to be the ablation result while being F's.
    """
    assert "Q" not in DIRECTIONS


def test_q_checkpoints_branch_inside_the_existing_staging_repo() -> None:
    """One repo per model, one branch per run. Not a new repo per experiment."""
    from research.src.models.transformer import push_checkpoint

    config = _load_config(MODEL_CONFIG_NAME)
    result = push_checkpoint(
        None, None, config, seed=123, dry_run=True, branch_suffix=CHECKPOINT_BRANCH_SUFFIX
    )
    assert result["would_use_branch"] == "seed-123-punct-ablation"
    assert result["would_push_to"].endswith("urdu-misinfo-xlmr-staging")


def test_default_branch_naming_is_unchanged_for_c_and_d() -> None:
    """The suffix is opt-in; Milestone 4's branches must not move."""
    from research.src.models.transformer import push_checkpoint

    result = push_checkpoint(None, None, _load_config(MODEL_CONFIG_NAME), seed=42, dry_run=True)
    assert result["would_use_branch"] == "seed-42"


# --------------------------------------------------------------------------
# the comparison block — the delta must be computable from Q's own file
# --------------------------------------------------------------------------


def test_comparison_names_the_same_seed_and_direction() -> None:
    comparison = comparison_metadata(42, "xlm-roberta-base")["compares_against"]
    baseline = comparison["cross_dataset_baseline"]

    assert baseline["experiment_id"] == "F"
    assert baseline["seed"] == 42
    assert baseline["same_direction"] is True
    assert baseline["same_seed"] is True
    assert baseline["metrics_file"] == "F_xlm-roberta-base_notri_fact_holdout_seed42.json"
    assert comparison["in_domain_reference"]["metrics_file"] == (
        "D_xlm-roberta-base_ax_to_grind_test_seed42.json"
    )


@pytest.mark.parametrize("seed", REQUIRED_SEEDS)
def test_comparison_baseline_matches_the_committed_f_file(seed: int) -> None:
    """The recorded baseline must be F's real number, not a remembered one."""
    filename = baseline_metrics_filename(seed, "xlm-roberta-base")
    path = METRICS_DIR / filename
    if not path.exists():
        pytest.skip(f"{filename} not committed")

    recorded = comparison_metadata(seed, "xlm-roberta-base")["compares_against"][
        "cross_dataset_baseline"
    ]["macro_f1"]
    expected = json.loads(path.read_text(encoding="utf-8"))["metrics"]["macro_f1"]
    assert recorded == pytest.approx(expected, abs=1e-12)


def test_missing_counterpart_records_null_not_a_guess() -> None:
    """`CLAUDE.md` rule 2: an absent file yields null, never a plausible number."""
    comparison = comparison_metadata(999, "no-such-model")["compares_against"]
    assert comparison["cross_dataset_baseline"]["macro_f1"] is None
    assert comparison["in_domain_reference"]["macro_f1"] is None


def test_in_domain_reference_points_at_the_test_split() -> None:
    assert in_domain_metrics_filename(2026, "xlm-roberta-base").endswith(
        "ax_to_grind_test_seed2026.json"
    )


# --------------------------------------------------------------------------
# dry-run plumbing
# --------------------------------------------------------------------------


def test_dry_run_refuses_to_write_into_the_real_metrics_directory() -> None:
    from research.src.models.transformer import resolve_dry_run_destination

    with pytest.raises(ValueError, match="may not write into"):
        resolve_dry_run_destination(METRICS_DIR)


def test_dry_run_destination_is_created(tmp_path) -> None:
    from research.src.models.transformer import resolve_dry_run_destination

    target = tmp_path / "scratch" / "q"
    assert resolve_dry_run_destination(target).is_dir()


def test_shortcut_analysis_declares_the_mode_as_implemented() -> None:
    """The CLI in `EXPERIMENT_PLAN.md` step 4b must no longer exit 2."""
    from research.src.experiments.run_shortcut_analysis import (
        MILESTONE_5_MODES,
        PUNCTUATION_ABLATION_MODE,
    )

    assert PUNCTUATION_ABLATION_MODE == "punctuation-ablation"
    assert PUNCTUATION_ABLATION_MODE not in MILESTONE_5_MODES


def test_shortcut_analysis_dispatches_to_the_q_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`EXPERIMENT_PLAN.md` step 4b names run_shortcut_analysis as Q's entry point.

    The dispatch is asserted rather than assumed, including that the GPU-only
    flags survive the hop — a silently dropped `--dry-run` would start a real
    training run on someone's laptop.
    """
    from research.src.experiments import punctuation_ablation, run_shortcut_analysis

    captured: list[list[str]] = []

    def fake_main(argv: list[str]) -> int:
        captured.append(argv)
        return 0

    monkeypatch.setattr(punctuation_ablation, "main", fake_main)

    exit_code = run_shortcut_analysis.main(
        [
            "--mode",
            "punctuation-ablation",
            "--dry-run",
            "--seeds",
            "42",
            "123",
            "--no-push",
        ]
    )

    assert exit_code == 0
    assert captured == [
        ["--data-config", "data.yaml", "--seeds", "42", "123", "--dry-run", "--no-push"]
    ]


def test_shortcut_analysis_still_refuses_the_unimplemented_modes() -> None:
    """I and J are not built yet and must keep exiting cleanly rather than half-run."""
    from research.src.experiments import run_shortcut_analysis

    assert run_shortcut_analysis.main(["--mode", "length-ablation"]) == 2


def test_torch_is_never_imported_at_module_scope() -> None:
    """H/H2 must stay runnable on a machine with scikit-learn and no torch.

    `run_shortcut_analysis` imports Q's module only inside the branch that needs
    it, and Q's module imports the training code only inside its functions — so a
    torch-free environment can still run the length-only experiments, and the
    transform stays importable from a test without pulling in a 2 GB dependency.
    Asserted structurally: a top-level import is an unindented `import` line.
    """
    from pathlib import Path

    from research.src.experiments import punctuation_ablation

    source = Path(punctuation_ablation.__file__).read_text(encoding="utf-8")
    top_level = [
        line
        for line in source.splitlines()
        if (line.startswith("import ") or line.startswith("from "))
    ]
    for line in top_level:
        assert "torch" not in line, f"torch imported at module scope: {line!r}"
        assert "models.transformer" not in line, f"training code at module scope: {line!r}"


# --------------------------------------------------------------------------
# committed Q results — skipped until the Kaggle run lands
# --------------------------------------------------------------------------


def _q_files() -> list:
    return sorted(METRICS_DIR.glob("Q_*.json")) if METRICS_DIR.exists() else []


@pytest.mark.parametrize("path", _q_files(), ids=lambda p: p.name)
def test_committed_q_file_records_its_ablation(path) -> None:
    metadata = json.loads(path.read_text(encoding="utf-8"))["run_metadata"]
    ablation = metadata["ablation"]

    assert ablation["type"] == "punctuation_stripping"
    assert ablation["train_after"]["n_non_alphanumeric"] == 0
    assert metadata["train_split"] == "ax_to_grind_train:punctuation_stripped"


@pytest.mark.parametrize("path", _q_files(), ids=lambda p: p.name)
def test_committed_q_file_records_a_computable_delta(path) -> None:
    """The requirement from M2-3: the delta must not need a manual cross-reference."""
    record = json.loads(path.read_text(encoding="utf-8"))
    comparison = record["run_metadata"]["compares_against"]
    baseline_path = METRICS_DIR / comparison["cross_dataset_baseline"]["metrics_file"]

    assert comparison["macro_f1_q"] == pytest.approx(record["metrics"]["macro_f1"], abs=1e-12)
    if not baseline_path.exists():
        pytest.skip(f"{baseline_path.name} not committed — nothing to cross-check")

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert baseline["seed"] == record["seed"], "Q must compare against the SAME seed"
    assert baseline["dataset"] == record["dataset"]
    assert baseline["split"] == record["split"]
    assert comparison["delta_macro_f1_q_minus_f"] == pytest.approx(
        record["metrics"]["macro_f1"] - baseline["metrics"]["macro_f1"], abs=1e-6
    )


@pytest.mark.parametrize("path", _q_files(), ids=lambda p: p.name)
def test_committed_q_file_has_per_example_predictions(path) -> None:
    """Q's holdout split is a cross-dataset test set, so Section 6 applies to it.

    Unlike the M5-2 backfill this is asserted rather than skipped: Q is being
    written for the first time with the predictions machinery already in place, so
    a missing sibling is a regression, not a legacy gap.
    """
    sibling = path.parent / predictions_filename(path.name)
    assert sibling.exists(), f"{path.name} has no per-example predictions sibling"

    record = json.loads(path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in sibling.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == record["n_examples"]
