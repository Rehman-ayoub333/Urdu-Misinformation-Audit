"""Experiments A and B — classical in-domain baselines on Ax-to-Grind.

`EXPERIMENT_PLAN.md` Section 2 step 1. Trains on `ax_to_grind_train`, evaluates on
`ax_to_grind_val` and `ax_to_grind_test`, writes one metrics file per
experiment × split per the Section 4 output contract.

In-domain only. Cross-dataset transfer (F, G) is Milestone 5 and Notri-Fact is not
touched here — `R7` keeps the two pools separate, and `data.yaml` marks Notri-Fact
`train_on: false`.

Usage:
    python -m research.src.experiments.run_in_domain --models classical
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
    write_predictions,
)
from research.src.models.classical import build_pipeline, fit_predict

_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = _ROOT / "research" / "configs"

CLASSICAL_EXPERIMENTS = ("A", "B")


def load_config(name: str) -> dict[str, Any]:
    return yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))


def run_classical(
    experiment_id: str,
    data_config: dict[str, Any],
    model_config: dict[str, Any],
    *,
    dataset: str = "ax_to_grind",
) -> list[Path]:
    """Train one classical experiment and write its metrics files."""
    experiment = model_config["experiments"][experiment_id]
    labels = data_config["labels"]["order"]
    positive = data_config["labels"]["positive"]

    train = load_split_frame(dataset, "train")
    written: list[Path] = []

    for split in model_config["evaluation"]["splits_evaluated"]:
        evaluation = load_split_frame(dataset, split)

        # Refit per evaluation split so val and test are each produced by a model
        # fitted only on train — no state leaks between the two reports.
        pipeline = build_pipeline(model_config, experiment_id)
        predictions, scores = fit_predict(
            pipeline,
            train["text"].tolist(),
            train["label"].tolist(),
            evaluation["text"].tolist(),
            positive_label=positive,
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
                    "features": model_config["features"],
                    "experiment": experiment,
                },
            },
            extra_metadata={
                "n_train": len(train),
                "train_split": f"{dataset}_train",
            },
        )

        problems = validate_metrics_record(record)
        if problems:
            raise RuntimeError(f"{experiment_id}/{split} metrics violate the contract: {problems}")

        filename = f"{experiment_id}_{experiment['name']}_{dataset}_{split}.json"
        written.append(write_metrics(record, filename))

        # REPRODUCIBILITY.md Section 6 / DECISION_REGISTER.md M5-2. No-ops on val;
        # the qualifying-splits rule lives in metrics.py so it cannot drift.
        write_predictions(
            metrics_filename=filename,
            split=split,
            row_ids=evaluation["row_id"].tolist(),
            y_true=evaluation["label"].tolist(),
            y_pred=predictions,
            scores=scores,
        )

        metrics = record["metrics"]
        collapse = record["prediction_collapse"]
        print(
            f"  [{experiment_id}] {split:<5} "
            f"macro-F1={metrics['macro_f1']:.4f}  acc={metrics['accuracy']:.4f}  "
            f"weighted-F1={metrics['weighted_f1']:.4f}  "
            f"dominant-class={collapse['dominant_class_share']:.2%}"
            + ("  COLLAPSED" if collapse["is_collapsed"] else "")
        )

    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        choices=["classical", "transformer"],
        default="classical",
        help="Which model family to run. `transformer` (C, D) is Milestone 4.",
    )
    parser.add_argument("--dataset", default="ax_to_grind")
    parser.add_argument("--data-config", default="data.yaml")
    parser.add_argument("--model-config", default="model_classical.yaml")
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=["C", "D"],
        default=["C", "D"],
        help="Transformer experiments to run. Ignored for --models classical.",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Skip the staging checkpoint push (transformer runs only).",
    )
    args = parser.parse_args(argv)

    if args.models == "transformer":
        # EXPERIMENT_PLAN.md Section 2 step 3 names this script as C/D's entry point,
        # so the transformer path lives behind the same command rather than a
        # separate one. The training loop itself is research/src/models/transformer.py.
        # Imported lazily: torch and transformers are heavy, and the classical path
        # must stay runnable without them.
        from research.src.models.transformer import run_experiment

        data_config = load_config(args.data_config)
        print("=== In-domain transformer fine-tuning (Experiments C, D) ===")
        print(f"  experiments: {args.experiments}")

        summaries = []
        for experiment_id in args.experiments:
            summaries.extend(
                run_experiment(
                    experiment_id,
                    data_config=data_config,
                    dry_run=None,
                    push=not args.no_push,
                )
            )
        print("\n=== summary ===")
        for summary in summaries:
            print(
                f"  {summary['experiment_id']} seed={summary['seed']} "
                f"{summary['train_seconds']:.0f}s -> {summary['push']}"
            )
        return 0

    data_config = load_config(args.data_config)
    model_config = load_config(args.model_config)

    print(f"=== In-domain classical baselines on {args.dataset} ===")
    print(f"  configs: {args.data_config}, {args.model_config}")
    print(f"  random_state: {model_config['random_state']}")

    written: list[Path] = []
    for experiment_id in CLASSICAL_EXPERIMENTS:
        name = model_config["experiments"][experiment_id]["name"]
        print(f"\n{experiment_id} — {name}")
        written.extend(run_classical(experiment_id, data_config, model_config, dataset=args.dataset))

    print()
    for path in written:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
