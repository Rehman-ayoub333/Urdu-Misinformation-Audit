"""Shortcut analysis — Experiment H (length-only) at Milestone 3.

`EXPERIMENT_PLAN.md` Section 2 step 2. Other modes listed in that table
(`length-ablation` = I, `length-buckets` = J, `punctuation-ablation` = Q) require
trained transformer checkpoints and belong to Milestones 4-5; they are declared in
the CLI so the surface matches the plan, and exit cleanly rather than pretending.

**What `--mode length-only` measures.** A logistic regression over article length
features alone — no words, no vocabulary, no content of any kind. Its macro-F1 is
how much of this dataset's label can be predicted from "how long is it?" and
nothing else, which is RQ3 stated as a number.

Read the result against experiments A/B/C/D: whatever margin a content-based model
has over H is the part of its performance not explainable by length. That framing
is the point of running H before the transformers rather than after.

Usage:
    python -m research.src.experiments.run_shortcut_analysis --mode length-only
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from research.src.data.split import load_split_frame
from research.src.evaluation.metrics import (
    build_metrics_record,
    validate_metrics_record,
    write_metrics,
)
from research.src.models.length_baseline import (
    build_length_pipeline,
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


def run_length_only(
    data_config: dict[str, Any],
    model_config: dict[str, Any],
    *,
    dataset: str = "ax_to_grind",
) -> list[Path]:
    """Experiment H over the configured length features."""
    experiment = model_config["experiments"]["H"]
    feature_names = experiment["features"]
    labels = data_config["labels"]["order"]
    positive = data_config["labels"]["positive"]

    train = load_split_frame(dataset, "train")
    written: list[Path] = []

    print(f"  features (length only, no content): {feature_names}")

    for split in model_config["evaluation"]["splits_evaluated"]:
        evaluation = load_split_frame(dataset, split)

        pipeline = build_length_pipeline(model_config)
        predictions, scores, coefficients = fit_predict_length(
            pipeline,
            train["text"].tolist(),
            train["label"].tolist(),
            evaluation["text"].tolist(),
            feature_names,
            positive_label=positive,
        )

        record = build_metrics_record(
            experiment_id="H",
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
            extra_metadata={
                "n_train": len(train),
                "train_split": f"{dataset}_train",
                "features_used": list(feature_names),
                "uses_text_content": False,
                # Signed toward the positive label ("fake"), so a positive value
                # reads directly as "more of this feature => more likely fake".
                "fitted_coefficients_toward_fake": coefficients,
                # Required to interpret the score above: if the relationship is
                # non-monotonic, this linear model structurally cannot express it
                # and its macro-F1 is a FLOOR on the length shortcut, not a
                # measurement of it.
                "length_relationship_diagnostic": length_relationship_diagnostic(
                    evaluation["text"].tolist(),
                    evaluation["label"].tolist(),
                    positive_label=positive,
                ),
            },
        )

        problems = validate_metrics_record(record)
        if problems:
            raise RuntimeError(f"H/{split} metrics violate the contract: {problems}")

        filename = f"H_{experiment['name']}_{dataset}_{split}.json"
        written.append(write_metrics(record, filename))

        metrics = record["metrics"]
        baseline = record["majority_class_baseline"]
        collapse = record["prediction_collapse"]
        print(
            f"  [H] {split:<5} "
            f"macro-F1={metrics['macro_f1']:.4f}  acc={metrics['accuracy']:.4f}  "
            f"(majority-class macro-F1={baseline['macro_f1']:.4f})  "
            f"dominant-class={collapse['dominant_class_share']:.2%}"
        )
        print(f"        coefficients toward fake: {coefficients}")

    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=["length-only", *MILESTONE_5_MODES],
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

    print("=== Experiment H — length-only baseline (RQ3) ===")
    print(f"  dataset: {args.dataset}")
    print(f"  random_state: {model_config['random_state']}")

    written = run_length_only(data_config, model_config, dataset=args.dataset)

    print()
    for path in written:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
