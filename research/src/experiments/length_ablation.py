"""Experiment I — length ablation (RQ3), replicating and extending Haroon's cap design.

`EXPERIMENT_PLAN.md` Section 2 step 5, `MASTER_PROJECT_BLUEPRINT.md` Part 11 row I.
The question: **how much of Ax-to-Grind's in-domain performance survives once
article length is no longer available as a cue?**

Length is this corpus's central confound. Milestone 2/3 measured it directly:
fake articles sit at both extremes of the length distribution while real ones
occupy a narrow middle band, the top length decile is 100% fake, and a model
given *nothing but* length reaches macro-F1 0.7552 against a 0.3333 majority-class
floor (`DECISION_REGISTER.md` M3-1, Experiment H2). Capping every article at N
words removes most of that signal — every article becomes the same length, up to
the ones that were already shorter — and what remains of the score is the part
that was never about length.

Haroon (2026, arXiv:2607.14131) ran this at a **50-word cap** and reported only a
**0.0067 macro-F1 drop**, the basis for the claim that the confound *inflates*
in-domain performance without *solely driving* it. Cap 50 is therefore the
primary, required point here: it is what makes this a replication rather than a
loosely related sweep.

--------------------------------------------------------------------------------
Sweep, not a single point — resolving a spec ambiguity
--------------------------------------------------------------------------------
`MASTER_PROJECT_BLUEPRINT.md` Part 1 and Part 11 describe I in the singular ("cap
at a fixed word count, following Haroon's 50-word design"), while Part 31 figure
10 specifies a **"length-ablation curve (F1 vs. word cap)"** — which a single
point cannot draw. Both are satisfied by running the sweep with 50 designated as
the primary point: `{25, 50, 100, 200}`, four points, of which 50 is the
replication and the other three exist to give figure 10 a curve. The sweep is not
optional and 50 is not skippable; each is one file, and each records
`is_primary_cap` so a reader cannot mistake an extension point for the
replication.

--------------------------------------------------------------------------------
Scope
--------------------------------------------------------------------------------
* **Models:** XLM-R (3 seeds, `{42, 123, 2026}`, as every transformer run) plus
  the **best classical baseline**, which is **B / tfidf_svm** on the measured
  numbers — committed test macro-F1 **0.8835 against A's 0.8755**. B only; running
  both would double the classical half for no comparison Part 11 asks for.
* **Direction:** Ax-to-Grind, **in-domain only** — train on `ax_to_grind_train`,
  evaluate on `ax_to_grind_test`. Cross-dataset transfer is F/G/Q's question, not
  this one, and mixing them would make the delta uninterpretable.

--------------------------------------------------------------------------------
The cap rule, and why it is not in `clean.py`
--------------------------------------------------------------------------------
**Definition.** Keep the **first N whitespace-delimited words** and drop the rest —
right-truncation, matching how the 512-subword ceiling already truncates
everywhere else in this project (`DECISION_REGISTER.md` M2-1, M4-1), so the two
length limits in the pipeline do not disagree about which end of an article
survives.

**Applied to BOTH sides at every N** — the training text *and* the evaluation
text. The question is how much signal survives when length information is
destroyed on both sides, not how a length-aware model copes with shortened test
input. Capping only at eval time would measure a distribution shift instead of an
ablation. The val split is capped too, since it selects the checkpoint and must
match the surface the training saw.

**Not in `research/src/data/clean.py`** — `CLAUDE.md` rule 4, exactly as for
Experiment Q. `clean.py` is the shared train/serve path the backend imports
verbatim; length capping there would change every other experiment and production
inference alike. It is also a Part 8 "explicit non-decision": capping at cleaning
time would pre-empt the very measurement RQ3 exists to make. I's cap is an
ablation applied on top of already-cleaned text, for this one experiment.

Usage:
    python -m research.src.experiments.length_ablation --dry-run      # CPU bug-check
    python -m research.src.experiments.length_ablation --models classical
    python -m research.src.experiments.run_shortcut_analysis --mode length-ablation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from research.src.data.split import load_split_frame
from research.src.evaluation.metrics import (
    annotate_metrics_file,
    build_metrics_record,
    committed_macro_f1,
    validate_metrics_record,
    write_metrics,
    write_predictions,
)
from research.src.evaluation.results_push import push_result_files

EXPERIMENT_ID = "I"

# Part 31 figure 10's curve. 50 is the replication point and is REQUIRED; the
# other three exist so the figure has more than one point to plot.
CAP_VALUES: tuple[int, ...] = (25, 50, 100, 200)
PRIMARY_CAP = 50

# Haroon (2026, arXiv:2607.14131), quoted in MASTER_PROJECT_BLUEPRINT.md Part 1's
# evidence table. A LITERATURE value for the primary cap only — never this
# project's own measurement, and labelled as such wherever it is recorded.
HAROON_CAP50_MACRO_F1_DROP = 0.0067

# Q reuses D's config for the same reason I does: the delta is only attributable
# to the ablation if nothing else moved.
TRANSFORMER_CONFIG_NAME = "model_xlmr.yaml"
CLASSICAL_CONFIG_NAME = "model_classical.yaml"

# Best classical on the committed numbers: B (tfidf_svm) test macro-F1 0.8835
# against A (tfidf_logreg) 0.8755. Asserted against the committed files by test,
# so "best" stays a measurement rather than a memory.
CLASSICAL_EXPERIMENT_ID = "B"

# Uncapped baselines each capped run is measured against.
TRANSFORMER_BASELINE_ID = "D"
CLASSICAL_BASELINE_ID = "B"

DATASET = "ax_to_grind"
EVAL_SPLIT = "test"

# One branch per (seed, cap) inside the EXISTING XLM-R staging repo.
CHECKPOINT_BRANCH_PREFIX = "-length-cap-"

RESULTS_SUBDIR = "milestone5/metrics"


# --- The ablation -------------------------------------------------------------


def cap_words(text: str, max_words: int) -> str:
    """Keep the first `max_words` whitespace-delimited words; drop the rest.

    Right-truncation, consistent with the 512-subword ceiling elsewhere (M2-1,
    M4-1). Applied to text `clean.py` has ALREADY cleaned and whitespace-collapsed,
    so `split()` sees exactly the tokens `n_words` is counted from in Experiments
    H/H2 — the two length measures in this project therefore agree by construction
    rather than by coincidence.

    An article already shorter than the cap is returned unchanged; the ablation
    can only remove length information, never add it.
    """
    if not isinstance(text, str) or not text:
        return ""
    if max_words <= 0:
        raise ValueError(f"cap must be a positive word count, got {max_words!r}")
    return " ".join(text.split()[:max_words])


def cap_frame(frame, max_words: int):  # noqa: ANN001, ANN201 - pandas DataFrame in/out
    """Return a copy of a split frame with every `text` capped at `max_words`."""
    capped = frame.copy()
    capped["text"] = capped["text"].map(lambda text: cap_words(text, max_words))
    return capped


def length_profile(frame, max_words: int) -> dict[str, Any]:  # noqa: ANN001 - DataFrame
    """Measure what the cap actually does to this split, per label.

    Per label deliberately: the confound under study is that length CORRELATES
    with the label (M3-1), so "how much did the cap remove" is only meaningful
    split by class. If the cap bites overwhelmingly on one class, that is the
    confound being removed, and the file should say so in numbers.
    """
    words = [len(text.split()) for text in frame["text"]]
    labels = frame["label"].tolist()

    by_label: dict[str, dict[str, float]] = {}
    for label in sorted(set(labels)):
        # strict=True is not pedantry here: a length/label misalignment would
        # silently attribute one class's article lengths to the other, which is
        # the exact quantity this profile exists to report.
        lengths = [n for n, lab in zip(words, labels, strict=True) if lab == label]
        truncated = [n for n in lengths if n > max_words]
        by_label[label] = {
            "n_examples": len(lengths),
            "mean_words_before": round(sum(lengths) / len(lengths), 2) if lengths else 0.0,
            "mean_words_after": (
                round(sum(min(n, max_words) for n in lengths) / len(lengths), 2)
                if lengths
                else 0.0
            ),
            "n_truncated": len(truncated),
            "pct_truncated": round(100 * len(truncated) / len(lengths), 2) if lengths else 0.0,
            "max_words_before": max(lengths) if lengths else 0,
        }

    truncated_all = [n for n in words if n > max_words]
    return {
        "cap_words": max_words,
        "n_examples": len(words),
        "n_truncated": len(truncated_all),
        "pct_truncated": round(100 * len(truncated_all) / len(words), 2) if words else 0.0,
        "mean_words_before": round(sum(words) / len(words), 2) if words else 0.0,
        "mean_words_after": (
            round(sum(min(n, max_words) for n in words) / len(words), 2) if words else 0.0
        ),
        "by_label": by_label,
    }


def ablation_metadata(
    *,
    cap: int,
    train_frame,  # noqa: ANN001 - pandas DataFrame (UNCAPPED)
    eval_frame,  # noqa: ANN001 - pandas DataFrame (UNCAPPED)
) -> dict[str, Any]:
    """The self-describing record of what the cap did, on both sides."""
    return {
        "ablation": {
            "type": "length_capping",
            "cap_words": cap,
            "is_primary_cap": cap == PRIMARY_CAP,
            "rule": (
                "keep the first N whitespace-delimited words, drop the rest "
                "(right-truncation, consistent with the 512-subword ceiling in "
                "DECISION_REGISTER.md M2-1/M4-1)"
            ),
            "applied_to": [
                f"{DATASET}_train (fine-tuning / fitting corpus)",
                f"{DATASET}_val (checkpoint selection only, transformer runs)",
                f"{DATASET}_{EVAL_SPLIT} (evaluation corpus)",
            ],
            "applied_to_both_sides": True,
            "why_both_sides": (
                "The measurement is how much signal survives when length "
                "information is destroyed everywhere, not how a length-aware model "
                "copes with shortened test input. Capping only at evaluation time "
                "would measure a train/test distribution shift instead."
            ),
            "applied_on_top_of": (
                "research/src/data/clean.py's Tier-2 output. clean.py itself is "
                "UNCHANGED — it is the shared train/serve path (CLAUDE.md rule 4), "
                "and length capping there is a Part 8 explicit non-decision "
                "because it would pre-empt RQ3's own measurement."
            ),
            "sweep": {
                "cap_values": list(CAP_VALUES),
                "primary_cap": PRIMARY_CAP,
                "why": (
                    "MASTER_PROJECT_BLUEPRINT.md Part 1/11 specify a single "
                    "50-word cap (Haroon's design); Part 31 figure 10 specifies a "
                    "curve of F1 against word cap. The sweep satisfies both, with "
                    "50 as the required replication point and the rest supplying "
                    "the curve."
                ),
            },
            "train_profile": length_profile(train_frame, cap),
            "eval_profile": length_profile(eval_frame, cap),
        }
    }


# --- The comparison I exists to make ------------------------------------------


def transformer_metrics_filename(seed: int, cap: int, model_name: str) -> str:
    return f"{EXPERIMENT_ID}_{model_name}_{DATASET}_{EVAL_SPLIT}_seed{seed}_cap{cap}.json"


def classical_metrics_filename(cap: int, model_name: str) -> str:
    return f"{EXPERIMENT_ID}_{model_name}_{DATASET}_{EVAL_SPLIT}_cap{cap}.json"


def transformer_baseline_filename(seed: int, model_name: str) -> str:
    return f"{TRANSFORMER_BASELINE_ID}_{model_name}_{DATASET}_{EVAL_SPLIT}_seed{seed}.json"


def classical_baseline_filename(model_name: str) -> str:
    return f"{CLASSICAL_BASELINE_ID}_{model_name}_{DATASET}_{EVAL_SPLIT}.json"


def comparison_metadata(
    *, cap: int, model_name: str, seed: int | None, baseline_filename: str, baseline_id: str
) -> dict[str, Any]:
    """Name the uncapped run this capped one is measured against, with its number.

    The delta must be computable straight from I's own file — a reader should not
    have to work out which of D's nine files pairs with this one, nor re-derive
    that "same model, same seed, same split, uncapped" is the correct pairing.
    """
    block: dict[str, Any] = {
        "purpose": (
            "I minus its uncapped baseline is the in-domain macro-F1 cost of "
            "removing article-length information (RQ3). The two runs differ in "
            "exactly one respect: I's train and eval text are capped at "
            f"{cap} words. Same model, same config, same seed, same split."
        ),
        "uncapped_baseline": {
            "experiment_id": baseline_id,
            "metrics_file": baseline_filename,
            "model": model_name,
            "dataset": DATASET,
            "split": EVAL_SPLIT,
            "seed": seed,
            "same_seed": seed is not None,
            "macro_f1": committed_macro_f1(baseline_filename),
        },
        "null_macro_f1_means": (
            "the counterpart file was absent when I ran — recompute from the "
            "named file rather than treating the gap as a result"
        ),
    }

    if cap == PRIMARY_CAP:
        # Literature value, for the replication point only, clearly attributed.
        # It is NOT this project's measurement and must never be reported as one.
        block["reference_finding"] = {
            "source": "Haroon (2026), arXiv:2607.14131",
            "via": "MASTER_PROJECT_BLUEPRINT.md Part 1 evidence table",
            "claim": (
                "capping Ax-to-Grind articles at 50 words yields only a 0.0067 "
                "macro-F1 drop, used to argue the length confound inflates but "
                "does not solely drive in-domain performance"
            ),
            "reported_macro_f1_drop": HAROON_CAP50_MACRO_F1_DROP,
            "is_this_projects_measurement": False,
            "note": (
                "External literature value recorded for comparison only. This "
                "run's own delta is delta_macro_f1_i_minus_uncapped below."
            ),
        }

    return {"compares_against": block}


def finalise_metrics_file(
    path: Path, *, push_info: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Add the delta (and, for a transformer run, the pushed checkpoint).

    Both need facts that do not exist when the record is built: the delta needs
    the record's own macro-F1, and a revision SHA is only assigned once the push
    completes. Shares `metrics.annotate_metrics_file` with Experiment Q.
    """

    def annotate(record: dict[str, Any]) -> None:
        metadata = record["run_metadata"]

        comparison = metadata.get("compares_against")
        if comparison is not None:
            baseline = comparison["uncapped_baseline"]["macro_f1"]
            comparison["macro_f1_capped"] = record["metrics"]["macro_f1"]
            comparison["delta_macro_f1_i_minus_uncapped"] = (
                round(record["metrics"]["macro_f1"] - baseline, 6)
                if baseline is not None
                else None
            )

        if push_info and push_info.get("pushed"):
            metadata["checkpoint"] = {
                "repo_id": push_info.get("repo_id"),
                "revision_branch": push_info.get("revision_branch"),
                "revision_sha": push_info.get("revision_sha"),
            }

    return annotate_metrics_file(path, annotate)


def _report_delta(record: dict[str, Any], label: str) -> None:
    comparison = record["run_metadata"]["compares_against"]
    baseline = comparison["uncapped_baseline"]["macro_f1"]
    print(
        f"  delta vs uncapped {comparison['uncapped_baseline']['experiment_id']} "
        f"({label}): capped={comparison['macro_f1_capped']:.4f} "
        f"uncapped={baseline} delta={comparison['delta_macro_f1_i_minus_uncapped']}"
    )


# --- Classical half (B / tfidf_svm) -------------------------------------------


def run_classical_cap(
    cap: int,
    *,
    data_config: dict[str, Any],
    model_config: dict[str, Any],
    dry_run: Any = None,
    push_results: bool = True,
) -> Path | None:
    """Fit B on capped train text and score it on capped test text.

    Deterministic given `random_state`, so this runs once per cap rather than over
    the seed triple — the same rule `EXPERIMENT_PLAN.md` Section 5 applies to A/B.
    Refits from config rather than loading a pickle, exactly as the cross-dataset
    runner does, so the source of truth stays in config.
    """
    from research.src.models.classical import build_pipeline, fit_predict
    from research.src.models.transformer import dry_run_slice

    experiment = model_config["experiments"][CLASSICAL_EXPERIMENT_ID]
    model_name = experiment["name"]
    labels = data_config["labels"]["order"]
    positive = data_config["labels"]["positive"]

    train = load_split_frame(DATASET, "train")
    evaluation = load_split_frame(DATASET, EVAL_SPLIT)
    if dry_run is not None:
        # Balanced, not `head` — the split index sorts fake first, so a plain head
        # is single-class and LinearSVC cannot fit it at all.
        train = dry_run_slice(train, dry_run.n_train)
        evaluation = dry_run_slice(evaluation, dry_run.n_eval)

    ablation = ablation_metadata(cap=cap, train_frame=train, eval_frame=evaluation)
    capped_train = cap_frame(train, cap)
    capped_eval = cap_frame(evaluation, cap)

    pipeline = build_pipeline(model_config, CLASSICAL_EXPERIMENT_ID)
    predictions, scores = fit_predict(
        pipeline,
        capped_train["text"].tolist(),
        capped_train["label"].tolist(),
        capped_eval["text"].tolist(),
        positive_label=positive,
    )

    record = build_metrics_record(
        experiment_id=EXPERIMENT_ID,
        model=model_name,
        dataset=DATASET,
        split=EVAL_SPLIT,
        seed=model_config["random_state"],
        y_true=capped_eval["label"].tolist(),
        y_pred=predictions,
        labels=labels,
        scores=scores,
        positive_label=positive,
        config={
            "data": data_config,
            "model": {
                "random_state": model_config["random_state"],
                "features": model_config["features"],
                "experiment": experiment,
            },
        },
        extra_metadata={
            "n_train": len(capped_train),
            "train_split": f"{DATASET}_train:capped_{cap}_words",
            "base_experiment_id": CLASSICAL_BASELINE_ID,
            "best_classical_selected_on": (
                "committed Ax-to-Grind test macro-F1: B/tfidf_svm 0.8835 > "
                "A/tfidf_logreg 0.8755"
            ),
            "dry_run": bool(dry_run),
            **ablation,
            **comparison_metadata(
                cap=cap,
                model_name=model_name,
                seed=None,
                baseline_filename=classical_baseline_filename(model_name),
                baseline_id=CLASSICAL_BASELINE_ID,
            ),
        },
    )

    problems = validate_metrics_record(record)
    if problems:
        raise RuntimeError(
            f"{EXPERIMENT_ID}/{model_name}/cap{cap} metrics violate the contract: {problems}"
        )

    destination = dry_run.metrics_destination if dry_run else None
    if dry_run is not None and destination is None:
        print(
            f"  [dry-run] {model_name} cap{cap}: "
            f"macro-F1={record['metrics']['macro_f1']:.4f} "
            "(NOT written to research/results/metrics/)"
        )
        return None

    filename = classical_metrics_filename(cap, model_name)
    path = write_metrics(record, filename, destination)
    prediction_path = write_predictions(
        metrics_filename=filename,
        split=EVAL_SPLIT,
        row_ids=capped_eval["row_id"].tolist(),
        y_true=capped_eval["label"].tolist(),
        y_pred=predictions,
        scores=scores,
        destination=destination,
    )
    print(
        f"  {model_name} cap{cap}: macro-F1={record['metrics']['macro_f1']:.4f} "
        f"acc={record['metrics']['accuracy']:.4f} -> {path.name}"
        + (f" (+ {prediction_path.name})" if prediction_path is not None else "")
    )

    _report_delta(finalise_metrics_file(path), f"{model_name} cap{cap}")

    paths = [path] + ([prediction_path] if prediction_path is not None else [])
    push_result_files(
        paths, subdir=RESULTS_SUBDIR, dry_run=bool(dry_run) or not push_results
    )
    return path


# --- Transformer half (XLM-R, 3 seeds) ----------------------------------------


def run_transformer_cap_seed(
    cap: int,
    seed: int,
    *,
    data_config: dict[str, Any],
    model_config: dict[str, Any],
    dry_run: Any = None,
    push: bool = True,
) -> dict[str, Any]:
    """Fine-tune XLM-R on capped Ax-to-Grind at one cap and one seed.

    Runs through `train_one_seed`'s optional-parameter path — the same one
    Experiment Q uses — so training, scoring, the metrics contract and the
    per-example predictions are the existing implementations, not copies.
    """
    from research.src.models.transformer import (
        EvalTarget,
        dry_run_slice,
        push_checkpoint,
        train_one_seed,
    )

    model_name = model_config["model"]["name"]

    train = load_split_frame(DATASET, "train")
    val = load_split_frame(DATASET, "val")
    evaluation = load_split_frame(DATASET, EVAL_SPLIT)

    if dry_run is not None:
        # Sliced here so the recorded ablation profile describes the rows this run
        # actually used; `train_one_seed` re-applies the same slice as a no-op.
        train = dry_run_slice(train, dry_run.n_train)
        val = dry_run_slice(val, dry_run.n_eval)
        evaluation = dry_run_slice(evaluation, dry_run.n_eval)

    ablation = ablation_metadata(cap=cap, train_frame=train, eval_frame=evaluation)
    profile = ablation["ablation"]
    print(
        f"  cap {cap}: train {profile['train_profile']['pct_truncated']}% of rows "
        f"truncated (mean words {profile['train_profile']['mean_words_before']} -> "
        f"{profile['train_profile']['mean_words_after']}), "
        f"eval {profile['eval_profile']['pct_truncated']}%"
    )

    result = train_one_seed(
        EXPERIMENT_ID,
        model_config,
        data_config,
        seed,
        dry_run=dry_run,
        train_frame=cap_frame(train, cap),
        selection_frame=cap_frame(val, cap),
        selection_label=f"{DATASET}_val:capped_{cap}_words",
        eval_targets=[EvalTarget(DATASET, EVAL_SPLIT, cap_frame(evaluation, cap))],
        train_split_label=f"{DATASET}_train:capped_{cap}_words",
        filename_suffix=f"_cap{cap}",
        extra_metadata={
            "base_experiment_id": TRANSFORMER_BASELINE_ID,
            **ablation,
            **comparison_metadata(
                cap=cap,
                model_name=model_name,
                seed=seed,
                baseline_filename=transformer_baseline_filename(seed, model_name),
                baseline_id=TRANSFORMER_BASELINE_ID,
            ),
        },
    )

    should_push = push and model_config["checkpoint_push"].get("enabled", False)
    push_info = push_checkpoint(
        result.pop("trainer"),
        result.pop("tokenizer"),
        model_config,
        seed,
        dry_run=bool(dry_run) or not should_push,
        branch_suffix=f"{CHECKPOINT_BRANCH_PREFIX}{cap}",
    )
    result["push"] = push_info
    result["cap_words"] = cap
    print(f"  checkpoint: {push_info}")

    for path in result["metrics_files"]:
        _report_delta(
            finalise_metrics_file(Path(path), push_info=push_info),
            f"{model_name} seed {seed} cap{cap}",
        )

    # M4-6: metrics AND their per-example predictions leave the session disk as
    # each run finishes, never batched at the end.
    result["results_push"] = push_result_files(
        [*result["metrics_files"], *result.get("prediction_files", [])],
        subdir=RESULTS_SUBDIR,
        dry_run=bool(dry_run) or not should_push,
    )
    return result


# --- Runner -------------------------------------------------------------------


def run_length_ablation(
    *,
    data_config: dict[str, Any],
    transformer_config: dict[str, Any],
    classical_config: dict[str, Any],
    caps: list[int] | None = None,
    seeds: list[int] | None = None,
    models: str = "all",
    dry_run: Any = None,
    push: bool = True,
) -> list[dict[str, Any]]:
    """Run I across the cap sweep, for the selected model families."""
    caps = caps or list(CAP_VALUES)
    if PRIMARY_CAP not in caps:
        # Not a style rule: cap 50 is the point that replicates Haroon's design
        # and is the only one comparable to the 0.0067 figure the blueprint cites.
        # A sweep without it is an extension with no replication in it.
        print(
            f"  !! cap {PRIMARY_CAP} is NOT in this run's sweep {caps}. It is the "
            "required replication point (EXPERIMENT_PLAN.md step 5, "
            "MASTER_PROJECT_BLUEPRINT.md Part 1) — the other caps only draw "
            "figure 10's curve around it."
        )

    seeds = seeds or (
        list(dry_run.seeds) if dry_run else list(transformer_config["training"]["seeds"])
    )

    results: list[dict[str, Any]] = []
    for cap in caps:
        marker = " (PRIMARY — replicates Haroon's design)" if cap == PRIMARY_CAP else ""
        print(f"\n=== cap {cap} words{marker} ===")

        if models in ("classical", "all"):
            path = run_classical_cap(
                cap,
                data_config=data_config,
                model_config=classical_config,
                dry_run=dry_run,
                push_results=push,
            )
            results.append({"model_family": "classical", "cap_words": cap, "path": str(path)})

        if models in ("transformer", "all"):
            for seed in seeds:
                print(
                    f"\n--- {EXPERIMENT_ID} / {transformer_config['model']['name']} "
                    f"/ seed {seed} / cap {cap} ---"
                )
                results.append(
                    run_transformer_cap_seed(
                        cap,
                        seed,
                        data_config=data_config,
                        model_config=transformer_config,
                        dry_run=dry_run,
                        push=push,
                    )
                )
    return results


def main(argv: list[str] | None = None) -> int:
    from research.src.models.transformer import (
        CONFIRM_FLAG,
        DryRunSettings,
        load_config,
        require_real_run_confirmation,
        resolve_dry_run_destination,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        CONFIRM_FLAG,
        action="store_true",
        dest="confirm_real_run",
        help="Required for a real (non-dry) run: this trains models and writes committed results.",
    )
    parser.add_argument(
        "--models",
        default="all",
        choices=["classical", "transformer", "all"],
        help="Classical is B/tfidf_svm only (the best baseline on committed numbers).",
    )
    parser.add_argument(
        "--caps",
        nargs="+",
        type=int,
        default=None,
        help=f"Word caps to sweep. Defaults to {list(CAP_VALUES)}; {PRIMARY_CAP} is required.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--data-config", default="data.yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Real training steps on a tiny CPU-sized slice, to catch bugs before GPU time.",
    )
    parser.add_argument("--dry-run-max-length", type=int, default=None)
    parser.add_argument("--dry-run-train-rows", type=int, default=None)
    parser.add_argument(
        "--dry-run-output-dir",
        default=None,
        help=(
            "Scratch directory for the dry run's metrics and predictions, so the "
            "write path is exercised for real. Never research/results/metrics/."
        ),
    )
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args(argv)

    if not args.dry_run and not require_real_run_confirmation(
        EXPERIMENT_ID, confirmed=args.confirm_real_run
    ):
        return 2

    data_config = load_config(args.data_config)
    transformer_config = load_config(TRANSFORMER_CONFIG_NAME)
    classical_config = load_config(CLASSICAL_CONFIG_NAME)

    dry_run = None
    if args.dry_run:
        overrides: dict[str, Any] = {}
        if args.dry_run_max_length is not None:
            overrides["max_length"] = args.dry_run_max_length
        if args.dry_run_train_rows is not None:
            overrides["n_train"] = args.dry_run_train_rows
        if args.dry_run_output_dir is not None:
            overrides["metrics_destination"] = resolve_dry_run_destination(
                args.dry_run_output_dir
            )
        dry_run = DryRunSettings(**overrides)

    caps = args.caps or list(CAP_VALUES)
    print(f"=== Experiment {EXPERIMENT_ID} — length ablation (RQ3) ===")
    print(f"  dataset: {DATASET}, in-domain (train -> {EVAL_SPLIT}); no cross-dataset direction")
    print(f"  caps: {caps} (primary {PRIMARY_CAP} — replicates Haroon's 50-word design)")
    print(f"  models: {args.models} — XLM-R x 3 seeds + best classical "
          f"({classical_config['experiments'][CLASSICAL_EXPERIMENT_ID]['name']})")
    if dry_run:
        print("\n=== DRY RUN — real training steps, tiny slice, CPU ===")
        print(
            f"    {dry_run.n_train} train rows, {dry_run.n_eval} eval rows, "
            f"max_length={dry_run.max_length}, batch={dry_run.batch_size}, "
            f"epochs={dry_run.epochs}, seeds={dry_run.seeds}"
        )
        if dry_run.metrics_destination is None:
            print("    No metrics are written and no checkpoint is pushed.")
        else:
            print(
                f"    Metrics + predictions -> {dry_run.metrics_destination} "
                "(scratch, never research/results/metrics/). No checkpoint is pushed."
            )

    results = run_length_ablation(
        data_config=data_config,
        transformer_config=transformer_config,
        classical_config=classical_config,
        caps=args.caps,
        seeds=args.seeds,
        models=args.models,
        dry_run=dry_run,
        push=not args.no_push,
    )

    print("\n=== summary ===")
    print(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
