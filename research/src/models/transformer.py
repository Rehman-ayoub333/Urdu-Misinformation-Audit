"""Transformer fine-tuning — Experiments C (mBERT) and D (XLM-R), in-domain.

Implements `EXPERIMENT_PLAN.md` Section 2 step 3. Trains on Ax-to-Grind's train
split and evaluates on val/test, once per seed in `{42, 123, 2026}`
(`REPRODUCIBILITY.md` Section 2), writing metrics through
`research/src/evaluation/metrics.py`'s existing contract rather than a second
implementation.

**This module is not run in the authoring environment.** It has no GPU access, so
its real execution is a hosted-GPU handoff (`PROJECT_SPECIFICATION.md` Section 6),
on Colab or Kaggle:
Claude Code prepares the runnable pipeline, Rehman executes it and commits the
resulting metrics and checkpoint references back. What *is* verified here is
`--dry-run`, which performs real training steps on a tiny slice of real data on
CPU purely to catch bugs before hosted GPU time is spent on them.

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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from research.src.data.split import load_split_frame
from research.src.evaluation.metrics import (
    build_metrics_record,
    validate_metrics_record,
    write_metrics,
    write_predictions,
)
from research.src.evaluation.results_push import push_result_files

_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = _ROOT / "research" / "configs"
CHECKPOINT_DIR = _ROOT / "checkpoints"  # gitignored; HF Hub is the real home (E17)

# Environment variables the hosted-GPU run needs. Never hardcoded, never committed —
# same treatment as the Kaggle credential at Milestone 1. The notebook reads HF_TOKEN
# from whichever platform it is on via research/src/notebook_env.py.
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

    # Where a dry run's metrics and per-example predictions are written, if
    # anywhere. `None` (the default) writes nothing at all, which is the original
    # behaviour. Pointing it at a scratch directory makes the dry run exercise
    # `write_metrics`/`write_predictions` for real — the only way to *prove* those
    # land rather than infer it from the code — while keeping a toy result out of
    # `research/results/metrics/`, which `CLAUDE.md` rule 2 forbids. It must never
    # be set to the real metrics directory.
    metrics_destination: Path | None = None


@dataclass
class EvalTarget:
    """One frame to score after training, plus how its metrics file is named.

    C and D evaluate the Ax-to-Grind splits their training corpus came from, so
    the dataset name could once be assumed. Experiment Q reuses this same training
    path but scores **Notri-Fact** afterwards
    (`DECISION_REGISTER.md` M2-3), so the dataset a frame belongs to now travels
    with the frame rather than being inferred from the training corpus — which is
    what keeps the filename and the `dataset` field in the record honest.
    """

    dataset: str
    split: str
    frame: Any  # pandas DataFrame


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

    Pure arithmetic, factored out so the GPU handoff's runtime estimate is
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
    train_frame=None,  # noqa: ANN001 - pandas DataFrame
    selection_frame=None,  # noqa: ANN001 - pandas DataFrame
    selection_label: str | None = None,
    eval_targets: list[EvalTarget] | None = None,
    train_split_label: str = "ax_to_grind_train",
    filename_suffix: str = "",
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fine-tune one model at one seed and write its metrics.

    Returns a summary dict; metrics files are written as a side effect.

    Every keyword after `output_root` defaults to the C/D behaviour this function
    was written for — train on `ax_to_grind_train`, select the checkpoint on
    `ax_to_grind_val`, score the splits named in the model config. They exist so
    **Experiment Q** can reuse this exact training path with a punctuation-stripped
    training corpus and a Notri-Fact evaluation target
    (`DECISION_REGISTER.md` M2-3) rather than growing a second, drifting copy of it:

        train_frame:       rows to fine-tune on. Q passes an ablated copy.
        selection_frame:   the in-training eval set behind `load_best_model_at_end`.
                           Never written to a metrics file — it drives model
                           selection, which `REPRODUCIBILITY.md` Section 6 scopes
                           out of the reported set.
        eval_targets:      what gets scored and written afterwards.
        train_split_label: what `run_metadata.train_split` records. It must name
                           the ablation when one was applied, or the file would
                           claim to have trained on the corpus as released.
        filename_suffix:   appended before `.json`, so an experiment that runs the
                           same model/split/seed at several settings gets one file
                           per setting. Experiment I passes `_cap{N}`; the
                           predictions sibling follows automatically.
        extra_metadata:    merged into `run_metadata` for every written record.
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

    if train_frame is None:
        train_frame = load_split_frame("ax_to_grind", "train")
    if eval_targets is None:
        eval_targets = [
            EvalTarget("ax_to_grind", split, load_split_frame("ax_to_grind", split))
            for split in model_config["evaluation"]["splits_evaluated"]
        ]

    if dry_run:
        train_frame = dry_run_slice(train_frame, dry_run.n_train)
        eval_targets = [
            replace(target, frame=dry_run_slice(target.frame, dry_run.n_eval))
            for target in eval_targets
        ]
        if selection_frame is not None:
            selection_frame = dry_run_slice(selection_frame, dry_run.n_eval)

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
        target.split: build_dataset(
            target.frame, tokenizer, max_length=max_length, label_to_id=label_to_id
        )
        for target in eval_targets
    }
    # M4-1 asks that a result state what it could not see. For an in-domain run the
    # training corpus's figure above answers that; for a cross-dataset one it does
    # not, because the corpus being scored is a different corpus with a different
    # length distribution. Recorded per target so both questions have an answer in
    # the file rather than only the one that happened to matter for C and D.
    truncation_evaluated = {
        target.split: truncation_stats(target.frame, tokenizer, max_length)
        for target in eval_targets
    }

    # Falls back to the "val" target, which is exactly what C and D have always
    # selected on — the parameter only exists so Q can select on an ablated copy of
    # the same split instead of one whose surface form its training never saw.
    if selection_frame is None:
        selection_dataset = eval_datasets.get("val")
        selection_label = selection_label or ("ax_to_grind_val" if selection_dataset else None)
    else:
        selection_dataset = build_dataset(
            selection_frame, tokenizer, max_length=max_length, label_to_id=label_to_id
        )

    # The suffix is in the directory name too, so an experiment running the same
    # seed at several settings does not have its runs overwrite each other on disk.
    run_dir = (
        (output_root or CHECKPOINT_DIR)
        / f"{experiment_id}_{model_config['model']['name']}_seed{seed}{filename_suffix}"
    )
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
        eval_dataset=selection_dataset,
        compute_metrics=_compute_metrics_for_trainer(label_order),
        # Pads each batch to its own longest sequence rather than to max_length.
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    started = time.time()
    train_result = trainer.train()
    train_seconds = time.time() - started
    print(f"  training finished in {train_seconds:.1f}s")

    # A dry run must never write into research/results/metrics/ — those are
    # committed research artefacts and an 8-row toy result has no business there
    # (CLAUDE.md rule 2). It may write to an explicit scratch directory, which is
    # how the write path itself gets exercised rather than assumed.
    destination = dry_run.metrics_destination if dry_run else None
    writes_files = destination is not None or not dry_run

    written: list[str] = []
    prediction_files: list[str] = []
    for target in eval_targets:
        output = trainer.predict(eval_datasets[target.split])
        logits = output.predictions[0] if isinstance(output.predictions, tuple) else output.predictions
        probabilities = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        predicted_ids = probabilities.argmax(axis=-1)

        y_pred = [id_to_label[int(i)] for i in predicted_ids]
        y_true = target.frame["label"].tolist()
        scores = [float(row[label_to_id[positive_label]]) for row in probabilities]

        record = build_metrics_record(
            experiment_id=experiment_id,
            model=model_config["model"]["name"],
            dataset=target.dataset,
            split=target.split,
            seed=seed,
            y_true=y_true,
            y_pred=y_pred,
            labels=label_order,
            scores=scores,
            positive_label=positive_label,
            config={"data": data_config, "model": model_config},
            extra_metadata={
                "n_train": len(train_frame),
                "train_split": train_split_label,
                "train_runtime_seconds": round(train_seconds, 2),
                "train_loss": float(train_result.training_loss),
                "epochs": epochs,
                "effective_batch_size": batch_size
                * training_cfg.get("gradient_accumulation_steps", 1),
                "fp16": use_fp16,
                # M4-1: what this run could not see, on both corpora it touched.
                "truncation": truncation,
                "truncation_evaluated": truncation_evaluated[target.split],
                "checkpoint_selection": {
                    "selected_on": selection_label,
                    "metric": training_cfg.get("metric_for_best_model", "macro_f1"),
                    "best_metric": (
                        float(trainer.state.best_metric)
                        if trainer.state.best_metric is not None
                        else None
                    ),
                    "load_best_model_at_end": bool(args.load_best_model_at_end),
                },
                "hardware": _hardware_metadata(),
                "dry_run": bool(dry_run),
                **(extra_metadata or {}),
            },
        )

        problems = validate_metrics_record(record)
        if problems:
            raise RuntimeError(
                f"{experiment_id}/{target.split}/seed{seed} metrics violate contract: {problems}"
            )

        if not writes_files:
            print(
                f"  [dry-run] {target.dataset}/{target.split}: "
                f"macro-F1={record['metrics']['macro_f1']:.4f} "
                f"(NOT written to research/results/metrics/)"
            )
            continue

        filename = (
            f"{experiment_id}_{model_config['model']['name']}"
            f"_{target.dataset}_{target.split}_seed{seed}{filename_suffix}.json"
        )
        path = write_metrics(record, filename, destination)
        written.append(str(path))

        # REPRODUCIBILITY.md Section 6 / DECISION_REGISTER.md M5-2.
        prediction_path = write_predictions(
            metrics_filename=filename,
            split=target.split,
            row_ids=target.frame["row_id"].tolist(),
            y_true=y_true,
            y_pred=y_pred,
            scores=scores,
            destination=destination,
        )
        if prediction_path is not None:
            prediction_files.append(str(prediction_path))
        print(
            f"  {target.dataset}/{target.split}: "
            f"macro-F1={record['metrics']['macro_f1']:.4f} "
            f"acc={record['metrics']['accuracy']:.4f} -> {path.name}"
            + (f" (+ {prediction_path.name})" if prediction_path is not None else "")
        )

    return {
        "experiment_id": experiment_id,
        "seed": seed,
        "train_seconds": train_seconds,
        "metrics_files": written,
        "prediction_files": prediction_files,
        "checkpoint_dir": str(run_dir),
        "trainer": trainer,
        "tokenizer": tokenizer,
    }


def dry_run_slice(frame, n_rows: int):  # noqa: ANN001, ANN201 - pandas DataFrame in/out
    """Take a LABEL-BALANCED slice of `n_rows` for a dry run.

    A plain `head(n)` is not good enough, and this is not hypothetical: the
    committed split indices are sorted by `row_id`, and Ax-to-Grind's fake rows
    sort first, so `head(8)` is **eight `fake` rows and no `real` ones**. A
    transformer trains on that without complaint (it just learns nothing), but
    `LinearSVC` raises `"needs samples of at least 2 classes"` — which is how this
    surfaced, in Experiment I's classical half.

    A single-class dry run is a weak bug-check even where it does not crash: it
    never exercises the label→id mapping in both directions, never produces a
    two-class confusion matrix, and cannot catch an inverted label. Balanced costs
    nothing and checks more.

    Dry-run-only. Real runs use the full split and never come through here.
    """
    labels = sorted(frame["label"].unique())
    if not labels:
        return frame.head(n_rows)

    per_label = max(1, n_rows // len(labels))
    taken = [frame.loc[frame["label"] == label].head(per_label) for label in labels]

    import pandas as pd

    return pd.concat(taken).head(n_rows)


CONFIRM_FLAG = "--confirm-real-run"


def require_real_run_confirmation(experiment_id: str, *, confirmed: bool) -> bool:
    """Gate a GPU-scale run behind an explicit flag. Returns True if it may proceed.

    The same guard `research/scripts/MILESTONE_5_GPU_HANDOFF.md` puts in front of
    the notebook's expensive cell (`CONFIRM_EVAL`), applied to the CLI, and for the
    same stated reason: a stray invocation must not start a multi-hour job.

    It is here because that failure actually happened rather than as a
    precaution. A unit test asserting that `--mode length-ablation` exits cleanly
    as an unimplemented mode kept passing that mode after Experiment I was
    implemented — so instead of returning 2, it started a real run, wrote
    `I_tfidf_svm_ax_to_grind_test_cap25.json` into `research/results/metrics/` and
    began fine-tuning XLM-R on a laptop CPU. The test was the immediate bug and is
    fixed, but nothing structural stopped an accidental caller from producing
    committed research artefacts, which is what this closes.

    `--dry-run` is deliberately NOT gated: it writes nothing to the results
    directory, pushes nothing, and exists to be run casually.
    """
    if confirmed:
        return True
    print(
        f"Experiment {experiment_id}: refusing to start a real run without "
        f"{CONFIRM_FLAG}.\n"
        "  This trains model(s) and writes committed research artefacts into\n"
        "  research/results/metrics/. Pass --dry-run for the CPU bug-check, or\n"
        f"  {CONFIRM_FLAG} when you mean it (e.g. from the GPU notebook)."
    )
    return False


def resolve_dry_run_destination(path: str | Path) -> Path:
    """Validate a dry run's scratch output directory and create it.

    Refuses `research/results/metrics/` outright. A dry run's numbers come from a
    handful of rows and one epoch; committing them beside real results would be
    exactly the "plausible-looking placeholder" `CLAUDE.md` rule 2 forbids, and a
    mistyped flag is the realistic way that happens.
    """
    from research.src.evaluation.metrics import METRICS_DIR

    resolved = Path(path).expanduser().resolve()
    if resolved == METRICS_DIR.resolve():
        raise ValueError(
            "a dry run may not write into research/results/metrics/ — those are "
            "committed research artefacts (CLAUDE.md rule 2). Use a scratch directory."
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


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
    branch_suffix: str = "",
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

    `branch_suffix` extends the per-seed branch name so a variant run lands beside
    the original rather than overwriting it — Experiment Q pushes to
    `seed-{n}-punct-ablation` in the same XLM-R staging repo, keeping one repo per
    model instead of one per experiment.
    """
    prefix = os.environ.get(ENV_HF_STAGING_PREFIX, "").strip()
    token = os.environ.get(ENV_HF_TOKEN, "").strip()
    suffix = model_config["checkpoint_push"]["staging_repo_suffix"]
    repo_id = f"{prefix}/{suffix}" if prefix else suffix
    revision_branch = f"seed-{seed}{branch_suffix}"

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
            "Set them in the notebook session before running "
            "(see research/scripts/MILESTONE_4_GPU_HANDOFF.md). Never hardcode them."
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

        # M4-6: get the metrics off this machine NOW, seed by seed — the same
        # guarantee the checkpoint above has had all along. Milestone 4's first run
        # lost every metrics file precisely because they were batched into a single
        # zip at the very end and the session's disk did not outlive the run, while
        # all six checkpoints survived. Never fatal: a failed upload must not
        # destroy a completed training run, so this reports and continues.
        result["results_push"] = push_result_files(
            result["metrics_files"], dry_run=bool(dry_run) or not should_push
        )
        results.append(result)

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="+", default=["C", "D"], choices=["C", "D"])
    parser.add_argument("--data-config", default="data.yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Real training steps on a tiny CPU-sized slice, to catch bugs before GPU time.",
    )
    # Default to DryRunSettings' own values; these flags only override when passed,
    # so the fast defaults are not silently undone by argparse.
    parser.add_argument("--dry-run-max-length", type=int, default=None)
    parser.add_argument("--dry-run-train-rows", type=int, default=None)
    parser.add_argument(
        "--dry-run-output-dir",
        default=None,
        help=(
            "Scratch directory for the dry run's metrics and predictions. Off by "
            "default (nothing is written). Never point this at "
            "research/results/metrics/ — a toy result must not land there."
        ),
    )
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
        if args.dry_run_output_dir is not None:
            overrides["metrics_destination"] = resolve_dry_run_destination(
                args.dry_run_output_dir
            )
        dry_run = DryRunSettings(**overrides)

    if dry_run:
        print("=== DRY RUN — real training steps, tiny slice, CPU ===")
        print(f"    {dry_run.n_train} train rows, {dry_run.n_eval} eval rows, "
              f"max_length={dry_run.max_length}, batch={dry_run.batch_size}, "
              f"epochs={dry_run.epochs}, seeds={dry_run.seeds}")
        if dry_run.metrics_destination is None:
            print("    No metrics are written and no checkpoint is pushed.\n")
        else:
            print(
                f"    Metrics + predictions -> {dry_run.metrics_destination} "
                "(scratch, never research/results/metrics/). No checkpoint is pushed.\n"
            )

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
