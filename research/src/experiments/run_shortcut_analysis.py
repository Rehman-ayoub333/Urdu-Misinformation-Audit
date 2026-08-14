"""Shortcut analysis — Experiments H and H2 (length-only) at Milestone 3.

`EXPERIMENT_PLAN.md` Section 2 steps 2 and 2b. Other modes listed in that table
(`length-ablation` = I, `length-buckets` = J, `punctuation-ablation` = Q) require
trained transformer checkpoints and belong to Milestones 4-5; they are declared in
the CLI so the surface matches the plan, and exit cleanly rather than pretending.

**What `--mode length-only` measures.** A model over article length features alone
— no words, no vocabulary, no content of any kind. Its macro-F1 is how much of
this dataset's label can be predicted from "how long is it?" and nothing else,
which is RQ3 stated as a number.

`--classifier` selects which of the pair runs:

* `logistic_regression` (default) → **H**, the linear model Part 32 specifies.
* `decision_tree` → **H2**, added by `DECISION_REGISTER.md` M3-1 because the
  length↔label relationship on Ax-to-Grind is U-shaped and a linear model cannot
  express it. Same features, same data, same seed.

They are a **floor/ceiling pair**, not alternatives: H bounds the length shortcut
from below, H2 from above, and RQ3 reports both. Running H2 writes the pair's
numbers side by side into its own metrics file (`length_shortcut_floor_ceiling`),
so neither has to be read alone.

Read the result against experiments A/B/C/D: whatever margin a content-based model
has over H2 is the part of its performance not explainable by length.

Usage:
    python -m research.src.experiments.run_shortcut_analysis --mode length-only
    python -m research.src.experiments.run_shortcut_analysis --mode length-only \\
        --classifier decision_tree
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from research.src.data.split import load_split_frame
from research.src.evaluation.metrics import (
    METRICS_DIR,
    build_metrics_record,
    validate_metrics_record,
    write_metrics,
)
from research.src.models.length_baseline import (
    SUPPORTED_CLASSIFIERS,
    build_length_pipeline,
    depth_sweep_diagnostic,
    describe_tree,
    fit_predict_length,
    length_relationship_diagnostic,
)

_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = _ROOT / "research" / "configs"

MILESTONE_5_MODES = {
    "length-ablation": "Experiment I — needs a trained checkpoint (Milestone 5)",
    "length-buckets": "Experiment J — needs trained checkpoints (Milestone 5.5)",
    "punctuation-ablation": "Experiment Q — needs XLM-R retraining (Milestone 5, M2-3)",
}


def load_config(name: str) -> dict[str, Any]:
    return yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))


def _counterpart_macro_f1(
    experiment_id: str, model_name: str, dataset: str, split: str
) -> float | None:
    """Read the paired experiment's macro-F1 off disk, if it has been run.

    Deliberately tolerant: H2 must still produce a valid result when H's file is
    absent (a fresh clone, or H re-run later). A missing counterpart records
    `null` rather than failing or — far worse — inventing a number.
    """
    path = METRICS_DIR / f"{experiment_id}_{model_name}_{dataset}_{split}.json"
    if not path.exists():
        return None
    try:
        return float(json.loads(path.read_text(encoding="utf-8"))["metrics"]["macro_f1"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def _floor_ceiling_pair(
    *,
    dataset: str,
    split: str,
    model_name: str,
    h2_macro_f1: float,
    majority_floor: float,
) -> dict[str, Any]:
    """H and H2's headline numbers, side by side, in H2's metrics file.

    The reason this block exists rather than living only in a write-up: the two
    numbers are individually misleading. H alone understates the length shortcut
    (M3-1), and H2 alone loses the fact that the *specified* linear baseline scores
    much lower. Anyone parsing H2's file for RQ3 gets both, plus the majority-class
    floor that makes them interpretable, without having to know to open a second
    file.
    """
    h_macro_f1 = _counterpart_macro_f1("H", model_name, dataset, split)
    pair: dict[str, Any] = {
        "majority_class_macro_f1": round(majority_floor, 4),
        "floor_H_linear_macro_f1": round(h_macro_f1, 4) if h_macro_f1 is not None else None,
        "ceiling_H2_nonlinear_macro_f1": round(h2_macro_f1, 4),
        "delta_H2_minus_H": (
            round(h2_macro_f1 - h_macro_f1, 4) if h_macro_f1 is not None else None
        ),
        "source_files": {
            "H": f"H_{model_name}_{dataset}_{split}.json",
            "H2": f"H2_{model_name}_{dataset}_{split}.json",
        },
    }
    if h_macro_f1 is None:
        pair["interpretation"] = (
            "H has not been run for this split, so no floor/ceiling pair is "
            "available. Run --mode length-only without --classifier first."
        )
    else:
        pair["interpretation"] = (
            "Both models see ONLY article length — identical features, data and "
            "seed. The gap between them is not extra information; it is the part "
            "of the length shortcut that a linear model structurally cannot "
            "express. Report the pair, not either number alone (DECISION_REGISTER "
            "M3-1). Content-based models must be compared against the CEILING to "
            "claim they learned something beyond length."
        )
    return pair


def run_length_only(
    data_config: dict[str, Any],
    model_config: dict[str, Any],
    *,
    dataset: str = "ax_to_grind",
    classifier: str = "logistic_regression",
) -> list[Path]:
    """Experiment H (linear) or H2 (nonlinear) over the configured length features."""
    experiment_id = SUPPORTED_CLASSIFIERS[classifier]
    experiment = model_config["experiments"][experiment_id]
    feature_names = experiment["features"]
    labels = data_config["labels"]["order"]
    positive = data_config["labels"]["positive"]

    train = load_split_frame(dataset, "train")
    written: list[Path] = []

    print(f"  features (length only, no content): {feature_names}")

    for split in model_config["evaluation"]["splits_evaluated"]:
        evaluation = load_split_frame(dataset, split)

        pipeline = build_length_pipeline(model_config, experiment_id)
        predictions, scores, attribution = fit_predict_length(
            pipeline,
            train["text"].tolist(),
            train["label"].tolist(),
            evaluation["text"].tolist(),
            feature_names,
            positive_label=positive,
        )

        extra_metadata: dict[str, Any] = {
            "n_train": len(train),
            "train_split": f"{dataset}_train",
            "features_used": list(feature_names),
            "uses_text_content": False,
            "classifier": classifier,
            # Required to interpret the score: if the relationship is
            # non-monotonic, a LINEAR model structurally cannot express it and H's
            # macro-F1 is a FLOOR on the length shortcut, not a measurement of it.
            # Recorded on both experiments — it is what makes the pair readable.
            "length_relationship_diagnostic": length_relationship_diagnostic(
                evaluation["text"].tolist(),
                evaluation["label"].tolist(),
                positive_label=positive,
            ),
        }

        if experiment_id == "H":
            # Signed toward the positive label ("fake"), so a positive value
            # reads directly as "more of this feature => more likely fake".
            extra_metadata["fitted_coefficients_toward_fake"] = attribution
        else:
            # A tree has no single direction to report — precisely why H2 exists.
            extra_metadata["feature_importances"] = attribution
            extra_metadata["tree_structure"] = describe_tree(pipeline, feature_names)
            extra_metadata["depth_sweep_diagnostic"] = depth_sweep_diagnostic(
                model_config,
                train["text"].tolist(),
                train["label"].tolist(),
                evaluation["text"].tolist(),
                evaluation["label"].tolist(),
                feature_names,
                experiment.get("diagnostic_depths", []),
            )

        record = build_metrics_record(
            experiment_id=experiment_id,
            model=experiment["name"],
            dataset=dataset,
            split=split,
            seed=model_config["random_state"],
            y_true=evaluation["label"].tolist(),
            y_pred=predictions,
            labels=labels,
            scores=scores,
            positive_label=positive,
            config={
                "data": data_config,
                "model": {
                    "random_state": model_config["random_state"],
                    "experiment": experiment,
                },
            },
            extra_metadata=extra_metadata,
        )

        if experiment_id == "H2":
            record["run_metadata"]["length_shortcut_floor_ceiling"] = _floor_ceiling_pair(
                dataset=dataset,
                split=split,
                model_name=experiment["name"],
                h2_macro_f1=record["metrics"]["macro_f1"],
                majority_floor=record["majority_class_baseline"]["macro_f1"],
            )

        problems = validate_metrics_record(record)
        if problems:
            raise RuntimeError(
                f"{experiment_id}/{split} metrics violate the contract: {problems}"
            )

        filename = f"{experiment_id}_{experiment['name']}_{dataset}_{split}.json"
        written.append(write_metrics(record, filename))

        metrics = record["metrics"]
        baseline = record["majority_class_baseline"]
        collapse = record["prediction_collapse"]
        print(
            f"  [{experiment_id}] {split:<5} "
            f"macro-F1={metrics['macro_f1']:.4f}  acc={metrics['accuracy']:.4f}  "
            f"(majority-class macro-F1={baseline['macro_f1']:.4f})  "
            f"dominant-class={collapse['dominant_class_share']:.2%}"
        )
        if experiment_id == "H":
            print(f"        coefficients toward fake: {attribution}")
        else:
            pair = record["run_metadata"]["length_shortcut_floor_ceiling"]
            print(f"        feature importances: {attribution}")
            print(
                f"        floor/ceiling: H(linear)="
                f"{pair['floor_H_linear_macro_f1']}  "
                f"H2(nonlinear)={pair['ceiling_H2_nonlinear_macro_f1']}  "
                f"delta={pair['delta_H2_minus_H']}"
            )
            print(
                "        depth sweep: "
                f"{extra_metadata['depth_sweep_diagnostic']['macro_f1_by_depth']}"
            )

    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=["length-only", *MILESTONE_5_MODES],
    )
    parser.add_argument(
        "--classifier",
        default="logistic_regression",
        choices=sorted(SUPPORTED_CLASSIFIERS),
        help=(
            "logistic_regression => experiment H (linear, the floor); "
            "decision_tree => experiment H2 (nonlinear, the ceiling — "
            "DECISION_REGISTER.md M3-1)"
        ),
    )
    parser.add_argument("--dataset", default="ax_to_grind")
    parser.add_argument("--data-config", default="data.yaml")
    parser.add_argument("--model-config", default="model_classical.yaml")
    args = parser.parse_args(argv)

    if args.mode in MILESTONE_5_MODES:
        print(f"--mode {args.mode} is not implemented yet: {MILESTONE_5_MODES[args.mode]}")
        return 2

    data_config = load_config(args.data_config)
    model_config = load_config(args.model_config)

    experiment_id = SUPPORTED_CLASSIFIERS[args.classifier]
    shape = "linear — the FLOOR" if experiment_id == "H" else "nonlinear — the CEILING"
    print(f"=== Experiment {experiment_id} — length-only baseline, {shape} (RQ3) ===")
    print(f"  dataset: {args.dataset}")
    print(f"  classifier: {args.classifier}")
    print(f"  random_state: {model_config['random_state']}")

    written = run_length_only(
        data_config, model_config, dataset=args.dataset, classifier=args.classifier
    )

    print()
    for path in written:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
