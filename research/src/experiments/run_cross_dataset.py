"""Experiments F and G — cross-dataset zero-shot transfer (RQ2).

`EXPERIMENT_PLAN.md` Section 2 step 4. Strict zero-shot: a model trained on the
SOURCE corpus is evaluated on the TARGET corpus with no fine-tuning, no adaptation
and no hyperparameter tuning on the target (`MASTER_PROJECT_BLUEPRINT.md` Part 13).

Blocked on the Milestone 2 dedup gate, which is confirmed passed: the
`cross:ax_to_grind|notri_fact` scan in `research/data/audit/duplication_report.json`
reports **0 exact duplicate groups and 0 near-duplicate pairs** across the full
10,083 x 13,388 comparison (char-5-gram MinHash, 128 permutations, Jaccard >= 0.8),
and `research/data/splits/splits_manifest.json` independently records
`cross_dataset_overlap: 0`. Without that, "zero-shot" would be unprovable.

Preprocessing is not reimplemented here: `load_split_frame` is the single path from
a committed split index to model input, and it cleans through
`research/src/data/clean.py` — the same function the backend calls at serving time
(`CLAUDE.md` rule 4).

--------------------------------------------------------------------------------
ONLY EXPERIMENT F IS IMPLEMENTED. Experiment G is deliberately absent.
--------------------------------------------------------------------------------
G ("train Notri-Fact -> test Ax-to-Grind") cannot be implemented without resolving
a direct contradiction inside the research design itself:

  * `MASTER_PROJECT_BLUEPRINT.md` Part 11 line 237 specifies G as *training on*
    Notri-Fact with all four models.
  * `MASTER_PROJECT_BLUEPRINT.md` Part 9 line 205 states that Notri-Fact is "used
    in full, held out entirely — no portion of Notri-Fact is used for training or
    hyperparameter tuning at any point, to keep RQ2's zero-shot claim valid."
  * `research/configs/data.yaml` sides with Part 9: Notri-Fact has `train_on:
    false`, carries only a `holdout` split, and is annotated "R7: never trained on,
    never split".

Both cannot hold. Implementing G would also require a Notri-Fact train/test split
that does not exist, and — for C and D — transformer checkpoints trained on
Notri-Fact, which is a full Milestone-4-scale training job rather than the
inference-only pass F needs. `CLAUDE.md` rule 3 and the escalation rule forbid
guessing on anything affecting dataset handling or experiment design, so G is left
unimplemented pending a `DECISION_REGISTER.md` resolution.

Usage:
    python -m research.src.experiments.run_cross_dataset --models classical
    python -m research.src.experiments.run_cross_dataset --models transformer
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
from research.src.models.classical import build_pipeline, fit_predict

_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = _ROOT / "research" / "configs"

# Transfer directions. Only F is defined; G's row is intentionally not present —
# see the module docstring. Adding it is a methodology decision, not a code change.
DIRECTIONS: dict[str, dict[str, str]] = {
    "F": {
        "source_dataset": "ax_to_grind",
        "source_split": "train",
        "target_dataset": "notri_fact",
        "target_split": "holdout",
        "description": "train Ax-to-Grind -> test Notri-Fact (zero-shot)",
    },
}

# Experiment id in the in-domain runs -> the same model, reused here.
CLASSICAL_MODELS = ("A", "B")
TRANSFORMER_MODELS = {"C": "model_mbert.yaml", "D": "model_xlmr.yaml"}


def load_config(name: str) -> dict[str, Any]:
    return yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))


def _transfer_metadata(direction_id: str) -> dict[str, Any]:
    """Provenance that makes a cross-dataset file self-describing.

    A reader must never have to infer which way the transfer ran from the
    filename alone.
    """
    direction = DIRECTIONS[direction_id]
    return {
        "transfer": {
            "direction_id": direction_id,
            "description": direction["description"],
            "source_dataset": direction["source_dataset"],
            "source_split": direction["source_split"],
            "target_dataset": direction["target_dataset"],
            "target_split": direction["target_split"],
            "zero_shot": True,
            "target_used_for_training": False,
            "target_used_for_tuning": False,
            "dedup_gate": {
                "cross_dataset_near_duplicate_pairs": 0,
                "source": "research/data/audit/duplication_report.json",
            },
        }
    }


def run_classical_transfer(
    direction_id: str,
    base_experiment_id: str,
    *,
    data_config: dict[str, Any],
    model_config: dict[str, Any],
) -> Path:
    """Retrain a classical model on the source corpus, score it on the target.

    Retraining rather than persisting a pickle is deliberate and cheap: these
    models are fully deterministic given `random_state`, so refitting reproduces
    the in-domain model exactly while keeping the source of truth in config rather
    than in a binary artefact (`MASTER_PROJECT_BLUEPRINT.md` Part 13).
    """
    direction = DIRECTIONS[direction_id]
    experiment = model_config["experiments"][base_experiment_id]
    labels = data_config["labels"]["order"]
    positive = data_config["labels"]["positive"]

    train = load_split_frame(direction["source_dataset"], direction["source_split"])
    target = load_split_frame(direction["target_dataset"], direction["target_split"])

    pipeline = build_pipeline(model_config, base_experiment_id)
    predictions, scores = fit_predict(
        pipeline,
        train_texts=train["text"].tolist(),
        train_labels=train["label"].tolist(),
        eval_texts=target["text"].tolist(),
        positive_label=positive,
    )

    record = build_metrics_record(
        experiment_id=direction_id,
        model=experiment["name"],
        dataset=direction["target_dataset"],
        split=direction["target_split"],
        seed=model_config["random_state"],
        y_true=target["label"].tolist(),
        y_pred=list(predictions),
        labels=labels,
        scores=list(scores) if scores is not None else None,
        positive_label=positive,
        config={"data": data_config, "model": model_config},
        extra_metadata={
            "n_train": len(train),
            "train_split": f"{direction['source_dataset']}_{direction['source_split']}",
            "base_experiment_id": base_experiment_id,
            "retrained_from_config": True,
            **_transfer_metadata(direction_id),
        },
    )

    problems = validate_metrics_record(record)
    if problems:
        raise RuntimeError(
            f"{direction_id}/{base_experiment_id} metrics violate contract: {problems}"
        )

    filename = (
        f"{direction_id}_{experiment['name']}"
        f"_{direction['target_dataset']}_{direction['target_split']}.json"
    )
    path = write_metrics(record, filename)
    print(
        f"  [{direction_id}/{base_experiment_id}] {experiment['name']}: "
        f"macro-F1={record['metrics']['macro_f1']:.4f} "
        f"acc={record['metrics']['accuracy']:.4f} "
        f"dominant-class={record['prediction_collapse']['dominant_class_share']:.2%} "
        f"-> {path.name}"
    )
    return path


def run_transformer_transfer(
    direction_id: str,
    base_experiment_id: str,
    seed: int,
    *,
    data_config: dict[str, Any],
    model_config: dict[str, Any],
    push_results: bool = True,
) -> Path:
    """Score an already-trained checkpoint on the target corpus. No training.

    The checkpoint is pulled from the staging repo that Milestone 4 pushed to, by
    repo id + revision branch, which is what `REPRODUCIBILITY.md` Section 6
    requires a result be traceable to. Runs identically on CPU and GPU — it is a
    forward pass, so a GPU only changes how long it takes.
    """
    # Imported lazily: the classical path must stay usable without torch installed.
    from research.src.models.evaluate_checkpoint import evaluate_on_frame

    direction = DIRECTIONS[direction_id]
    target = load_split_frame(direction["target_dataset"], direction["target_split"])

    return evaluate_on_frame(
        experiment_id=direction_id,
        model_config=model_config,
        data_config=data_config,
        seed=seed,
        frame=target,
        dataset=direction["target_dataset"],
        split=direction["target_split"],
        extra_metadata={
            "base_experiment_id": base_experiment_id,
            **_transfer_metadata(direction_id),
        },
        push_results=push_results,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default="classical",
        choices=["classical", "transformer", "all"],
        help="Classical retrains on the source; transformer scores existing checkpoints.",
    )
    parser.add_argument(
        "--directions",
        nargs="+",
        default=["F"],
        choices=sorted(DIRECTIONS),
        help="Only F exists. G is blocked on a methodology decision — see the module docstring.",
    )
    parser.add_argument("--data-config", default="data.yaml")
    parser.add_argument("--classical-config", default="model_classical.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--no-push-results", action="store_true")
    args = parser.parse_args(argv)

    data_config = load_config(args.data_config)
    written: list[Path] = []

    for direction_id in args.directions:
        direction = DIRECTIONS[direction_id]
        print(f"\n=== Experiment {direction_id} — {direction['description']} ===")
        print(f"  source: {direction['source_dataset']}/{direction['source_split']}")
        print(f"  target: {direction['target_dataset']}/{direction['target_split']} (zero-shot)")

        if args.models in ("classical", "all"):
            classical_config = load_config(args.classical_config)
            print(f"\n  -- classical (retrained on source, random_state="
                  f"{classical_config['random_state']}) --")
            for base_experiment_id in CLASSICAL_MODELS:
                written.append(
                    run_classical_transfer(
                        direction_id,
                        base_experiment_id,
                        data_config=data_config,
                        model_config=classical_config,
                    )
                )

        if args.models in ("transformer", "all"):
            print("\n  -- transformer (existing checkpoints, inference only) --")
            for base_experiment_id, config_name in TRANSFORMER_MODELS.items():
                model_config = load_config(config_name)
                seeds = args.seeds or list(model_config["training"]["seeds"])
                for seed in seeds:
                    written.append(
                        run_transformer_transfer(
                            direction_id,
                            base_experiment_id,
                            seed,
                            data_config=data_config,
                            model_config=model_config,
                            push_results=not args.no_push_results,
                        )
                    )

    print(f"\n=== wrote {len(written)} cross-dataset metrics file(s) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
