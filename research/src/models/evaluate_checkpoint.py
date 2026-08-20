"""Re-score an already-trained checkpoint from the HF Hub, without retraining.

`DECISION_REGISTER.md` M4-6. Milestone 4's six runs completed and pushed their
checkpoints, then the Kaggle session's working directory vanished with the metrics
JSONs still on it. This recovers those numbers from the artefacts that survived.

**Why re-scoring is the faithful recovery, not merely the cheap one.** Both model
configs set `load_best_model_at_end: true`, and `push_checkpoint` uploads
`trainer.model` after training — so the checkpoint on the Hub IS the exact model
that produced the lost metrics. Scoring is `argmax` over logits with no sampling,
so re-running it reproduces those numbers. A retrain would NOT: GPU training is not
bit-deterministic, so a re-run of seed 42 yields different weights, and pushing it
would overwrite the seed branches and orphan the revision SHAs already recorded as
pushed. `REPRODUCIBILITY.md` Section 6 makes the Hub checkpoint plus revision the
canonical artefact; re-scoring it is exactly what an external verifier would do.

**What cannot be recovered, and is therefore recorded as null rather than
invented** (`CLAUDE.md` rule 2): `train_runtime_seconds` and `train_loss` are only
observable while training, and the training run's hardware is not observable from
its output. `run_metadata.hardware` here describes the machine doing the
EVALUATION, and `run_metadata.evaluation_only_recovery` says so explicitly so no
later reader mistakes it for the training host. None of these feed RQ1, RQ2 or RQ3.

Everything else is recovered exactly: all metrics, the confusion matrix, per-class
figures, `prediction_collapse`, the majority-class baseline, `truncation`
(recomputed deterministically from the same split and tokenizer) and the config.

This is a RECOVERY path, not part of the standard pipeline. The normal route is
`run_in_domain.py --models transformer`, which trains and evaluates in one pass.

Usage:
    python -m research.src.models.evaluate_checkpoint                    # all 6
    python -m research.src.models.evaluate_checkpoint --experiments D --seeds 42
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

from research.src.data.split import load_split_frame
from research.src.evaluation.metrics import (
    build_metrics_record,
    validate_metrics_record,
    write_metrics,
)
from research.src.evaluation.results_push import push_result_files

# Reused wholesale from the training module rather than reimplemented — the point
# of this path is to produce records identical in shape and derivation to the ones
# the training run wrote, so the scoring, dataset construction and label mapping
# must be literally the same code (CLAUDE.md rule 4's principle, applied here).
from research.src.models.transformer import (
    ENV_HF_STAGING_PREFIX,
    ENV_HF_TOKEN,
    MODEL_CONFIGS,
    _hardware_metadata,
    _label_maps,
    build_dataset,
    load_config,
    truncation_stats,
)

_ROOT = Path(__file__).resolve().parents[3]


def resolve_repo_id(model_config: dict[str, Any], prefix: str | None = None) -> str:
    """Same construction `push_checkpoint` used, so it points at the same repo."""
    prefix = (
        prefix if prefix is not None else os.environ.get(ENV_HF_STAGING_PREFIX, "")
    ).strip()
    suffix = model_config["checkpoint_push"]["staging_repo_suffix"]
    return f"{prefix}/{suffix}" if prefix else suffix


def evaluate_one_checkpoint(
    experiment_id: str,
    model_config: dict[str, Any],
    data_config: dict[str, Any],
    seed: int,
    *,
    repo_id: str | None = None,
    revision: str | None = None,
    push_results: bool = True,
) -> dict[str, Any]:
    """Download one checkpoint, score val/test, write the metrics files."""
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    token = os.environ.get(ENV_HF_TOKEN, "").strip() or None
    repo_id = repo_id or resolve_repo_id(model_config)
    revision = revision or f"seed-{seed}"

    training_cfg = dict(model_config["training"])
    label_order = data_config["labels"]["order"]
    positive_label = data_config["labels"]["positive"]
    label_to_id, id_to_label = _label_maps(label_order)
    max_length = model_config["tokenizer"]["max_length"]
    eval_batch_size = training_cfg["per_device_eval_batch_size"]
    # Match the ORIGINAL run's numerics: it evaluated under the same fp16 setting,
    # via the same Trainer. Changing it could shift logits at the margin.
    use_fp16 = bool(training_cfg.get("fp16", False)) and torch.cuda.is_available()

    print(f"  pulling {repo_id}@{revision}")
    tokenizer = AutoTokenizer.from_pretrained(repo_id, revision=revision, token=token)
    model = AutoModelForSequenceClassification.from_pretrained(
        repo_id, revision=revision, token=token
    )

    # The pinned revision SHA is what REPRODUCIBILITY.md Section 6 requires be
    # recorded next to the metrics. Read back from the Hub so it is the real one.
    revision_sha = None
    try:
        from huggingface_hub import HfApi

        revision_sha = getattr(
            HfApi(token=token).repo_info(repo_id=repo_id, revision=revision), "sha", None
        )
    except Exception as exc:  # noqa: BLE001 - provenance nicety, never fatal
        print(f"    (could not read revision sha: {exc})")

    train_frame = load_split_frame("ax_to_grind", "train")
    split_frames = {
        split: load_split_frame("ax_to_grind", split)
        for split in model_config["evaluation"]["splits_evaluated"]
    }

    # Deterministic given the same split and tokenizer, so this reproduces the
    # training run's figure rather than approximating it (M4-1).
    truncation = truncation_stats(train_frame, tokenizer, max_length)

    eval_datasets = {
        split: build_dataset(frame, tokenizer, max_length=max_length, label_to_id=label_to_id)
        for split, frame in split_frames.items()
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        args = TrainingArguments(
            output_dir=tmp_dir,
            per_device_eval_batch_size=eval_batch_size,
            fp16=use_fp16,
            dataloader_pin_memory=torch.cuda.is_available(),
            report_to=[],
            disable_tqdm=False,
        )
        trainer = Trainer(
            model=model,
            args=args,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        )

        written: list[str] = []
        for split, dataset in eval_datasets.items():
            output = trainer.predict(dataset)
            logits = (
                output.predictions[0]
                if isinstance(output.predictions, tuple)
                else output.predictions
            )
            probabilities = torch.softmax(torch.tensor(logits), dim=-1).numpy()
            predicted_ids = probabilities.argmax(axis=-1)

            y_pred = [id_to_label[int(i)] for i in predicted_ids]
            y_true = split_frames[split]["label"].tolist()
            scores = [float(row[label_to_id[positive_label]]) for row in probabilities]

            record = build_metrics_record(
                experiment_id=experiment_id,
                model=model_config["model"]["name"],
                dataset="ax_to_grind",
                split=split,
                seed=seed,
                y_true=y_true,
                y_pred=y_pred,
                labels=label_order,
                scores=scores,
                positive_label=positive_label,
                config={"data": data_config, "model": model_config},
                extra_metadata={
                    # True of the split, not of the training run, so still accurate.
                    "n_train": len(train_frame),
                    "train_split": "ax_to_grind_train",
                    # Describe how the checkpoint WAS trained; read from the same
                    # config that trained it, not from observation of this run.
                    "epochs": training_cfg["num_train_epochs"],
                    "effective_batch_size": training_cfg["per_device_train_batch_size"]
                    * training_cfg.get("gradient_accumulation_steps", 1),
                    "fp16": use_fp16,
                    "truncation": truncation,
                    # NOT observable from a checkpoint. Explicitly null, never a
                    # plausible-looking placeholder (CLAUDE.md rule 2).
                    "train_runtime_seconds": None,
                    "train_loss": None,
                    # This is the EVALUATION host, not the training host.
                    "hardware": _hardware_metadata(),
                    "evaluation_only_recovery": {
                        "recovered": True,
                        "reason": (
                            "Metrics lost with the Kaggle session's working directory "
                            "before download (DECISION_REGISTER.md M4-6). Re-scored "
                            "from the pushed checkpoint, which load_best_model_at_end "
                            "makes the exact model that produced the original numbers."
                        ),
                        "checkpoint_repo_id": repo_id,
                        "checkpoint_revision_branch": revision,
                        "checkpoint_revision_sha": revision_sha,
                        "unavailable_fields": [
                            "train_runtime_seconds",
                            "train_loss",
                            "training_hardware",
                        ],
                        "hardware_is": "evaluation host, not the training host",
                    },
                    "dry_run": False,
                },
            )

            problems = validate_metrics_record(record)
            if problems:
                raise RuntimeError(
                    f"{experiment_id}/{split}/seed{seed} metrics violate contract: {problems}"
                )

            filename = (
                f"{experiment_id}_{model_config['model']['name']}"
                f"_ax_to_grind_{split}_seed{seed}.json"
            )
            path = write_metrics(record, filename)
            written.append(str(path))
            print(
                f"  {split}: macro-F1={record['metrics']['macro_f1']:.4f} "
                f"acc={record['metrics']['accuracy']:.4f} -> {path.name}"
            )

    result: dict[str, Any] = {
        "experiment_id": experiment_id,
        "seed": seed,
        "metrics_files": written,
        "checkpoint_repo_id": repo_id,
        "checkpoint_revision_branch": revision,
        "checkpoint_revision_sha": revision_sha,
    }

    # M4-6 again: do not let a recovered metrics file sit only on this disk either.
    if push_results:
        result["results_push"] = push_result_files(written)

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments", nargs="+", default=["C", "D"], choices=["C", "D"]
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Defaults to every seed in the model config.",
    )
    parser.add_argument("--data-config", default="data.yaml")
    parser.add_argument(
        "--no-push-results",
        action="store_true",
        help="Write metrics locally without uploading them (not recommended -- M4-6).",
    )
    args = parser.parse_args(argv)

    data_config = yaml.safe_load(
        (_ROOT / "research" / "configs" / args.data_config).read_text(encoding="utf-8")
    )

    print("=== Eval-only recovery: re-scoring pushed checkpoints ===")
    print("    No training. Metrics are recomputed from the checkpoints on the Hub.")

    results = []
    for experiment_id in args.experiments:
        model_config = load_config(MODEL_CONFIGS[experiment_id])
        seeds = args.seeds or list(model_config["training"]["seeds"])
        for seed in seeds:
            print(f"\n--- {experiment_id} / {model_config['model']['name']} / seed {seed} ---")
            results.append(
                evaluate_one_checkpoint(
                    experiment_id,
                    model_config,
                    data_config,
                    seed,
                    push_results=not args.no_push_results,
                )
            )

    written = [path for r in results for path in r["metrics_files"]]
    print(f"\n=== recovered {len(written)} metrics files from {len(results)} checkpoints ===")

    failed = [
        r for r in results if r.get("results_push") and not r["results_push"].get("pushed")
    ]
    if failed:
        print(
            f"\n!! {len(failed)} checkpoint(s) had metrics that did NOT reach the Hub. "
            "They exist locally but are not durable yet — see M4-6."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
