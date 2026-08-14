"""Transformer fine-tuning — Experiments C (mBERT) and D (XLM-R), in-domain.

Implements `EXPERIMENT_PLAN.md` Section 2 step 3. Trains on Ax-to-Grind's train
split and evaluates on val/test, once per seed in `{42, 123, 2026}`
(`REPRODUCIBILITY.md` Section 2), writing metrics through
`research/src/evaluation/metrics.py`'s existing contract rather than a second
implementation.

**This module is not run in the authoring environment.** It has no GPU access, so
its real execution is a Colab handoff (`PROJECT_SPECIFICATION.md` Section 6):
Claude Code prepares the runnable pipeline, Rehman executes it and commits the
resulting metrics and checkpoint references back. What *is* verified here is
`--dry-run`, which performs real training steps on a tiny slice of real data on
CPU purely to catch bugs before Colab time is spent on them.

**Sequence length.** `max_length=512`, from config. `MASTER_PROJECT_BLUEPRINT.md`
Part 8 calls for "full, uncapped" text at training time, but both mBERT and
XLM-R-base have 512 learned position embeddings, so unlimited is not
architecturally expressible — 512 is the only possible reading of "uncapped".
About 4.7% of Ax-to-Grind still exceeds it, concentrated in the extreme-length
fake tail (max observed 1,447 tokens). `DECISION_REGISTER.md` M4-1 records what
that means for how RQ3's results may be phrased.

Usage:
    python -m research.src.experiments.run_in_domain --models transformer
    python -m research.src.models.transformer --dry-run          # CPU bug-check
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from research.src.data.split import load_split_frame
from research.src.evaluation.metrics import (
    build_metrics_record,
    validate_metrics_record,
    write_metrics,
)

_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = _ROOT / "research" / "configs"
CHECKPOINT_DIR = _ROOT / "checkpoints"  # gitignored; HF Hub is the real home (E17)

# Environment variables the Colab run needs. Never hardcoded, never committed —
# same treatment as the Kaggle credential at Milestone 1.
ENV_HF_TOKEN = "HF_TOKEN"
ENV_HF_STAGING_PREFIX = "HF_STAGING_PREFIX"

MODEL_CONFIGS = {"C": "model_mbert.yaml", "D": "model_xlmr.yaml"}


@dataclass
class DryRunSettings:
    """Overrides that make a real-but-tiny training run possible on CPU.

    Everything here shrinks the *work*, not the code path: the same tokenizer,
    model class, optimiser, loss and metrics run, so a bug in any of them still
    surfaces. `max_length` is reduced because CPU attention at 512 is quadratic
    and would turn a bug-check into a coffee break — the 512 path itself is
    covered by `--dry-run-max-length 512` when it needs exercising.
    """

    # Sized so the whole check finishes in a couple of minutes on a CPU-only box.
    # A measured ~75-190 s/step at seq 128 / batch 2 on the authoring machine made a
    # 16-step dry run useless as a pre-flight check — it took longer than reading the
    # code. Eight rows over four steps at seq 64 still exercises every line that can
    # break (tokenise, collate, forward, backward, optimiser step, LR schedule, eval,
    # metrics contract, push wiring) while costing ~2 minutes.
    n_train: int = 8
    n_eval: int = 4
    max_length: int = 64
    batch_size: int = 2
    epochs: int = 1
    seeds: tuple[int, ...] = (42,)
    push: bool = False


def load_config(name: str) -> dict[str, Any]:
    return yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))


def training_step_counts(
    *,
    n_train: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    epochs: int,
    warmup_ratio: float,
) -> tuple[int, int, int]:
    """Return (steps_per_epoch, total_optimizer_steps, warmup_steps).

    Pure arithmetic, factored out so the Colab handoff's runtime estimate is
    computed from the same numbers the scheduler actually uses rather than being
    guessed separately and drifting from it.
    """
    effective_batch = max(1, batch_size * max(1, gradient_accumulation_steps))
    steps_per_epoch = max(1, -(-n_train // effective_batch))  # ceil division
    total_steps = steps_per_epoch * max(1, int(epochs))
    return steps_per_epoch, total_steps, int(round(total_steps * warmup_ratio))


def set_all_seeds(seed: int) -> None:
    """Seed every RNG that can affect a run.

    Transformers seeds Python/NumPy/torch, but the environment variable and the
    cuDNN determinism flags are set explicitly too — `REPRODUCIBILITY.md` Section 2
    requires seeded runs, and a silently non-deterministic dataloader shuffle would
    make the 3-seed variance meaningless.
    """
    import torch
    from transformers import set_seed

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


class TextClassificationDataset:
    """Minimal torch Dataset over pre-tokenised encodings."""

    def __init__(self, encodings: dict[str, Any], labels: list[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        # Plain lists, not tensors: rows have ragged lengths under dynamic padding,
        # and the collator is what turns a batch into a rectangular tensor.
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def _label_maps(label_order: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    """Fix the label<->id mapping from config, so it never depends on row order."""
    label_to_id = {label: index for index, label in enumerate(label_order)}
    return label_to_id, {index: label for label, index in label_to_id.items()}


def build_dataset(
    frame,  # noqa: ANN001 - pandas DataFrame
    tokenizer,  # noqa: ANN001
    *,
    max_length: int,
    label_to_id: dict[str, int],
) -> TextClassificationDataset:
    """Tokenise WITHOUT padding; a collator pads each batch to its own longest row.

    Truncation still happens here at `max_length`, so what the model can see is
    unchanged. Only the padding strategy differs from `ML_SPECIFICATION.md`
    Section 1's `padding="max_length"`, which describes *serving* (single request,
    fixed shape) rather than training.

    The saving is real but modest here, and worth stating accurately rather than
    assuming: Ax-to-Grind's median article is 40 subword tokens, but the fake class
    has a heavy tail, so in a batch of 16 at least one long article usually pulls
    the batch maximum up. Measured over the real training split, E[batch max] is
    **401 tokens against the 512 ceiling — about a 1.3x saving**, not the order of
    magnitude the median length alone would suggest.

    `group_by_length=True` would capture much more of that gap by bucketing
    similar-length examples together, and is deliberately NOT used. Length predicts
    the label in this corpus (`DECISION_REGISTER.md` M3-1: the top length decile is
    100% fake), so length-homogeneous batches would also be label-homogeneous —
    biasing per-batch gradient estimates along exactly the axis this project is
    trying to measure. A modest speedup is not worth entangling the optimiser with
    the confound under study.

    Results are unaffected by dynamic padding itself: attention masks zero out pad
    positions either way, so the two strategies are mathematically equivalent.
    """
    encodings = tokenizer(
        frame["text"].tolist(),
        truncation=True,
        padding=False,
        max_length=max_length,
    )
    labels = [label_to_id[label] for label in frame["label"].tolist()]
    return TextClassificationDataset(dict(encodings), labels)


def truncation_stats(frame, tokenizer, max_length: int) -> dict[str, Any]:  # noqa: ANN001
    """Measure how much text this run actually loses to the 512-token ceiling.

    Recorded into every metrics file. `DECISION_REGISTER.md` M4-1 makes this a
    reporting requirement rather than a nicety: RQ3's transformer results describe
    signal *within* a 512-token window, and the share of examples truncated is what
    makes that qualification concrete instead of a hand-wave.
    """
    lengths = [
        len(ids)
        for ids in tokenizer(frame["text"].tolist(), truncation=False)["input_ids"]
    ]
    truncated = [n for n in lengths if n > max_length]
    by_label: dict[str, int] = {}
    for length, label in zip(lengths, frame["label"].tolist()):
        if length > max_length:
            by_label[label] = by_label.get(label, 0) + 1

    return {
        "max_length": max_length,
        "n_examples": len(lengths),
        "n_truncated": len(truncated),
        "pct_truncated": round(100 * len(truncated) / len(lengths), 3) if lengths else 0.0,
        "truncated_by_label": by_label,
        "max_observed_tokens": max(lengths) if lengths else 0,
    }


def _compute_metrics_for_trainer(label_order: list[str]):  # noqa: ANN202
    """Metric callback used ONLY for in-training checkpoint selection.

    Deliberately thin. The authoritative metrics that land in
    `research/results/metrics/` are built by `metrics.py` after training, so there
    is exactly one implementation of the reported numbers (`CLAUDE.md` rule 4's
    spirit applied to evaluation).
    """
    from sklearn.metrics import f1_score

    def compute(eval_prediction) -> dict[str, float]:  # noqa: ANN001
        logits = eval_prediction.predictions
        logits = logits[0] if isinstance(logits, tuple) else logits
        predictions = np.argmax(logits, axis=-1)
        return {
            "macro_f1": float(
                f1_score(eval_prediction.label_ids, predictions, average="macro", zero_division=0)
            )
        }

    return compute


def train_one_seed(
    experiment_id: str,
    model_config: dict[str, Any],
    data_config: dict[str, Any],
    seed: int,
    *,
    dry_run: DryRunSettings | None = None,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Fine-tune one model at one seed and write its val/test metrics.

    Returns a summary dict; metrics files are written as a side effect.
    """
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    training_cfg = dict(model_config["training"])
    label_order = data_config["labels"]["order"]
    positive_label = data_config["labels"]["positive"]
    label_to_id, id_to_label = _label_maps(label_order)

    max_length = model_config["tokenizer"]["max_length"]
    batch_size = training_cfg["per_device_train_batch_size"]
    eval_batch_size = training_cfg["per_device_eval_batch_size"]
    epochs = training_cfg["num_train_epochs"]
    use_fp16 = bool(training_cfg.get("fp16", False)) and torch.cuda.is_available()

    if dry_run:
        max_length = dry_run.max_length
        batch_size = dry_run.batch_size
        eval_batch_size = dry_run.batch_size
        epochs = dry_run.epochs
        use_fp16 = False  # fp16 is unsupported on CPU and errors rather than degrades

    set_all_seeds(seed)

    print(f"  loading tokenizer + model: {model_config['model']['hf_model_id']}")
    tokenizer = AutoTokenizer.from_pretrained(model_config["model"]["hf_model_id"])
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config["model"]["hf_model_id"],
        num_labels=model_config["model"]["num_labels"],
        id2label=id_to_label,
        label2id=label_to_id,
    )

    train_frame = load_split_frame("ax_to_grind", "train")
    split_frames = {
        split: load_split_frame("ax_to_grind", split)
        for split in model_config["evaluation"]["splits_evaluated"]
    }

    if dry_run:
        train_frame = train_frame.head(dry_run.n_train)
        split_frames = {k: v.head(dry_run.n_eval) for k, v in split_frames.items()}

    truncation = truncation_stats(train_frame, tokenizer, max_length)
    print(
        f"  truncation at max_length={max_length}: "
        f"{truncation['n_truncated']}/{truncation['n_examples']} "
        f"({truncation['pct_truncated']}%), max observed {truncation['max_observed_tokens']} tokens"
    )

    train_dataset = build_dataset(
        train_frame, tokenizer, max_length=max_length, label_to_id=label_to_id
    )
    eval_datasets = {
        split: build_dataset(frame, tokenizer, max_length=max_length, label_to_id=label_to_id)
        for split, frame in split_frames.items()
    }

    run_dir = (output_root or CHECKPOINT_DIR) / f"{experiment_id}_{model_config['model']['name']}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # transformers 5.x removed `warmup_ratio` in favour of an absolute
    # `warmup_steps`, so the configured ratio is converted here. Keeping the ratio
    # in config is the right level of abstraction — it stays correct when the batch
    # size or epoch count changes, whereas a hardcoded step count would silently
    # become a different fraction of training.
    steps_per_epoch, total_steps, warmup_steps = training_step_counts(
        n_train=len(train_frame),
        batch_size=batch_size,
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 1),
        epochs=epochs,
        warmup_ratio=training_cfg.get("warmup_ratio", 0.1),
    )
    print(
        f"  schedule: {steps_per_epoch} steps/epoch x {epochs} epochs = {total_steps} "
        f"optimizer steps, {warmup_steps} warmup"
    )

    args = TrainingArguments(
        output_dir=str(run_dir),
        seed=seed,
        data_seed=seed,
        learning_rate=float(training_cfg["learning_rate"]),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 1),
        num_train_epochs=epochs,
        weight_decay=training_cfg.get("weight_decay", 0.01),
        adam_epsilon=float(training_cfg.get("adam_epsilon", 1e-8)),
        max_grad_norm=training_cfg.get("max_grad_norm", 1.0),
        warmup_steps=warmup_steps,
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "linear"),
        optim=training_cfg.get("optimizer", "adamw_torch"),
        fp16=use_fp16,
        eval_strategy="epoch" if not dry_run else "no",
        save_strategy="epoch" if not dry_run else "no",
        load_best_model_at_end=bool(training_cfg.get("load_best_model_at_end", True))
        and not dry_run,
        metric_for_best_model=training_cfg.get("metric_for_best_model", "macro_f1"),
        greater_is_better=bool(training_cfg.get("greater_is_better", True)),
        save_total_limit=training_cfg.get("save_total_limit", 1),
        logging_steps=training_cfg.get("logging_steps", 50) if not dry_run else 1,
        dataloader_num_workers=0 if dry_run else training_cfg.get("dataloader_num_workers", 2),
        # Pinned memory is a GPU-transfer optimisation and warns noisily on CPU.
        dataloader_pin_memory=torch.cuda.is_available(),
        report_to=[],  # local JSON logging only — W&B deferred
        disable_tqdm=False,
    )

    from transformers import DataCollatorWithPadding

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets.get("val"),
        compute_metrics=_compute_metrics_for_trainer(label_order),
        # Pads each batch to its own longest sequence rather than to max_length.
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    started = time.time()
    train_result = trainer.train()
    train_seconds = time.time() - started
    print(f"  training finished in {train_seconds:.1f}s")

    written: list[str] = []
    for split, dataset in eval_datasets.items():
        output = trainer.predict(dataset)
        logits = output.predictions[0] if isinstance(output.predictions, tuple) else output.predictions
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
                "n_train": len(train_frame),
                "train_split": "ax_to_grind_train",
                "train_runtime_seconds": round(train_seconds, 2),
                "train_loss": float(train_result.training_loss),
                "epochs": epochs,
                "effective_batch_size": batch_size
                * training_cfg.get("gradient_accumulation_steps", 1),
                "fp16": use_fp16,
                # M4-1: what this run could not see.
                "truncation": truncation,
                "hardware": _hardware_metadata(),
                "dry_run": bool(dry_run),
            },
        )

        problems = validate_metrics_record(record)
        if problems:
            raise RuntimeError(f"{experiment_id}/{split}/seed{seed} metrics violate contract: {problems}")

        if dry_run:
            # A dry run must never write into research/results/metrics/ — those are
            # committed research artefacts and a 32-row toy result has no business
            # there (CLAUDE.md rule 2).
            print(
                f"  [dry-run] {split}: macro-F1={record['metrics']['macro_f1']:.4f} "
                f"(NOT written to research/results/metrics/)"
            )
        else:
            filename = f"{experiment_id}_{model_config['model']['name']}_ax_to_grind_{split}_seed{seed}.json"
            path = write_metrics(record, filename)
            written.append(str(path))
            print(
                f"  {split}: macro-F1={record['metrics']['macro_f1']:.4f} "
                f"acc={record['metrics']['accuracy']:.4f} -> {path.name}"
            )

    return {
        "experiment_id": experiment_id,
        "seed": seed,
        "train_seconds": train_seconds,
        "metrics_files": written,
        "checkpoint_dir": str(run_dir),
        "trainer": trainer,
        "tokenizer": tokenizer,
    }


def _hardware_metadata() -> dict[str, Any]:
    """Auto-captured, never hand-typed (REPRODUCIBILITY.md Section 5)."""
    import torch

    meta: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    if torch.cuda.is_available():
        meta["gpu_name"] = torch.cuda.get_device_name(0)
        meta["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
        )
        meta["cuda_version"] = torch.version.cuda
    return meta


def push_checkpoint(
    trainer,  # noqa: ANN001
    tokenizer,  # noqa: ANN001
    model_config: dict[str, Any],
    seed: int,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Push one trained checkpoint to a STAGING Hugging Face repo.

    Staging, never production: the deployed checkpoint is chosen deliberately at
    Milestone 9 on defensibility rather than raw in-domain F1
    (`EXPERIMENT_PLAN.md` Section 7), so nothing here writes to `MODEL_REPO_ID`.

    Credentials come from the environment only. `HF_TOKEN` is a write token and
    `HF_STAGING_PREFIX` is the destination namespace (Rehman's HF username or org,
    which this environment cannot know). Neither is ever committed — identical
    treatment to the Kaggle credential at Milestone 1.

    Returns the repo id and commit revision, which are what `REPRODUCIBILITY.md`
    Section 6 requires be recorded alongside the metrics.
    """
    prefix = os.environ.get(ENV_HF_STAGING_PREFIX, "").strip()
    token = os.environ.get(ENV_HF_TOKEN, "").strip()
    suffix = model_config["checkpoint_push"]["staging_repo_suffix"]
    repo_id = f"{prefix}/{suffix}" if prefix else suffix
    revision_branch = f"seed-{seed}"

    if dry_run:
        return {
            "pushed": False,
            "reason": "dry run",
            "would_push_to": repo_id,
            "would_use_branch": revision_branch,
            "prefix_set": bool(prefix),
            "token_set": bool(token),
        }

    if not prefix or not token:
        missing = [
            name
            for name, value in ((ENV_HF_STAGING_PREFIX, prefix), (ENV_HF_TOKEN, token))
            if not value
        ]
        raise RuntimeError(
            f"cannot push checkpoint: {missing} not set in the environment. "
            "Set them in the Colab session before running "
            "(see research/scripts/MILESTONE_4_COLAB_HANDOFF.md). Never hardcode them."
        )

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        private=bool(model_config["checkpoint_push"].get("private", True)),
        exist_ok=True,
    )
    # One branch per seed, so all three seeds live in one repo without overwriting
    # each other and each is addressable by a pinned revision.
    try:
        api.create_branch(repo_id=repo_id, branch=revision_branch, exist_ok=True)
    except Exception as exc:  # noqa: BLE001 - branch may already exist on some backends
        print(f"    (branch note: {exc})")

    trainer.model.push_to_hub(repo_id, token=token, revision=revision_branch)
    tokenizer.push_to_hub(repo_id, token=token, revision=revision_branch)

    info = api.repo_info(repo_id=repo_id, revision=revision_branch)
    return {
        "pushed": True,
        "repo_id": repo_id,
        "revision_branch": revision_branch,
        "revision_sha": getattr(info, "sha", None),
    }


def run_experiment(
    experiment_id: str,
    *,
    data_config: dict[str, Any],
    dry_run: DryRunSettings | None = None,
    push: bool = True,
) -> list[dict[str, Any]]:
    """Run every seed for one experiment (C or D)."""
    model_config = load_config(MODEL_CONFIGS[experiment_id])
    seeds = dry_run.seeds if dry_run else tuple(model_config["training"]["seeds"])

    results = []
    for seed in seeds:
        print(f"\n--- {experiment_id} / {model_config['model']['name']} / seed {seed} ---")
        result = train_one_seed(
            experiment_id, model_config, data_config, seed, dry_run=dry_run
        )

        should_push = push and model_config["checkpoint_push"].get("enabled", False)
        push_info = push_checkpoint(
            result.pop("trainer"),
            result.pop("tokenizer"),
            model_config,
            seed,
            dry_run=bool(dry_run) or not should_push,
        )
        result["push"] = push_info
        print(f"  checkpoint: {push_info}")
        results.append(result)

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="+", default=["C", "D"], choices=["C", "D"])
    parser.add_argument("--data-config", default="data.yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Real training steps on a tiny CPU-sized slice, to catch bugs before Colab.",
    )
    # Default to DryRunSettings' own values; these flags only override when passed,
    # so the fast defaults are not silently undone by argparse.
    parser.add_argument("--dry-run-max-length", type=int, default=None)
    parser.add_argument("--dry-run-train-rows", type=int, default=None)
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args(argv)

    data_config = load_config(args.data_config)
    dry_run = None
    if args.dry_run:
        overrides: dict[str, Any] = {}
        if args.dry_run_max_length is not None:
            overrides["max_length"] = args.dry_run_max_length
        if args.dry_run_train_rows is not None:
            overrides["n_train"] = args.dry_run_train_rows
        dry_run = DryRunSettings(**overrides)

    if dry_run:
        print("=== DRY RUN — real training steps, tiny slice, CPU ===")
        print(f"    {dry_run.n_train} train rows, {dry_run.n_eval} eval rows, "
              f"max_length={dry_run.max_length}, batch={dry_run.batch_size}, "
              f"epochs={dry_run.epochs}, seeds={dry_run.seeds}")
        print("    No metrics are written and no checkpoint is pushed.\n")

    summaries = []
    for experiment_id in args.experiments:
        summaries.extend(
            run_experiment(
                experiment_id,
                data_config=data_config,
                dry_run=dry_run,
                push=not args.no_push,
            )
        )

    print("\n=== summary ===")
    print(json.dumps(summaries, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
