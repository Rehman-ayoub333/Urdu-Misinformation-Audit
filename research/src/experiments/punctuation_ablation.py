"""Experiment Q — punctuation-surface-form ablation (`DECISION_REGISTER.md` M2-3).

`EXPERIMENT_PLAN.md` Section 2 step 4b. Milestone 2's audit found that the two
corpora do not share a surface form: **Notri-Fact contains zero non-alphanumeric
characters anywhere** — no Urdu full stop, comma or question mark, evidently
stripped before release — while Ax-to-Grind retains 53 distinct non-alphanumeric
characters, at a density that itself differs by label (real 0.0147 vs fake
0.0102). Any punctuation-linked cue a model learns on Ax-to-Grind is therefore
*systematically unavailable* at test time on Notri-Fact, so Experiment F's
cross-dataset collapse is confounded by surface form on top of the domain and
length shifts already logged (M2-2, M3-1).

Q isolates that confound by removing it from the training side: retrain XLM-R on a
punctuation-stripped copy of Ax-to-Grind's training split, re-run the same
zero-shot evaluation on Notri-Fact, and report the macro-F1 delta against F. This
is option (c) of M2-3 — "run both, report the delta as its own finding" — chosen
over stripping both corpora, which would have destroyed a potentially genuine
signal to make the surfaces match.

**Scope, fixed by M2-3 and deliberately narrow.** XLM-R only (the primary model,
not all four), Ax-to-Grind → Notri-Fact only (the direction where the mechanism
actually applies: training exposed to a punctuation↔label correlation, testing on
a corpus with none), 3 seeds `{42, 123, 2026}`. Kept proportionate to the time
budget (R8) — a targeted confound-isolation check, not a second full cross-dataset
matrix.

--------------------------------------------------------------------------------
What "punctuation-stripped" means here, and why it is not in `clean.py`
--------------------------------------------------------------------------------
**Definition.** Replace every character that is not a Unicode letter, digit or
whitespace with a single space, then collapse whitespace runs and strip the ends.
Not just sentence punctuation: M2-3's audit measured *zero* non-alphanumeric
characters of any kind in Notri-Fact, so matching that surface means removing
brackets, quotes, dashes, symbols and underscores too. The whitespace collapse is
not cosmetic — leaving the gaps would trade a punctuation confound for a
whitespace-density one that correlates with exactly the same thing.

**Removed characters become a word boundary rather than vanishing, and that is a
deliberate choice made on measurement.** The character *class* is fixed by M2-3's
audit; whether a removed character leaves a space behind is not, and the two
readings are not equivalent: on the real Ax-to-Grind training split they produce
different text for **732 of 6,405 rows (11.43%)**. Deleting outright fuses the
words either side into tokens that exist in neither corpus — `ڈیبٹ/کریڈٹ` becomes
`ڈیبٹکریڈٹ`, `پابندی'چکن` becomes `پابندیچکن`, `COVID-19` becomes `COVID19` —
which would inject a brand-new out-of-vocabulary artefact into exactly the corpus
whose confounds this experiment is trying to isolate. Substituting a space keeps
the word boundary (`COVID 19`, `F 16`) and costs nothing: measured word counts go
493,866 → 486,333 under substitution against 484,961 under deletion, and the
residual drop is punctuation that stood alone as its own token. If this is ever
revisited, it is the one sub-decision inside Q's definition that was made here
rather than in M2-3.

**One visible consequence, stated rather than discovered later:** `clean.py`'s
placeholder tokens lose their brackets, so `[URL]` becomes `URL` and `[EMOJI]`
becomes `EMOJI`. They survive as words and stay distinguishable from Urdu content
(they are Latin-script), so the *presence* signal Part 8 wanted preserved is
preserved — but they are no longer bracketed, and Notri-Fact's own text could in
principle contain the bare word "URL". Recorded in each run's metadata.

**This transform is Q-specific and must never move into
`research/src/data/clean.py`.** `clean.py` is the single train/serve preprocessing
path (`CLAUDE.md` rule 4, `ML_SPECIFICATION.md` Section 9): the backend imports it
verbatim, so anything added there would silently change what every other
experiment trains on *and* what production sees at inference time. Q's stripping is
an ablation applied **on top of** already-cleaned text, for one experiment, which
is why it lives here beside the runner that uses it. `collapse_whitespace` is
imported from `clean.py` rather than re-implemented, so the two cannot disagree
about what a whitespace run is.

**Notri-Fact is not touched.** It is already punctuation-free as released, and it
is the held-out corpus — `data.yaml` marks it `train_on: false` (R7). The
ablation applies to the training side only, which is what keeps Q zero-shot in the
same sense F is.

Usage:
    python -m research.src.experiments.punctuation_ablation --dry-run   # CPU bug-check
    python -m research.src.experiments.punctuation_ablation             # real run, GPU
    python -m research.src.experiments.run_shortcut_analysis --mode punctuation-ablation
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from research.src.data.clean import collapse_whitespace
from research.src.data.split import load_split_frame
from research.src.evaluation.metrics import (
    METRICS_DIR,
    validate_metrics_record,
)
from research.src.evaluation.results_push import push_result_files
from research.src.experiments.run_cross_dataset import transfer_metadata

EXPERIMENT_ID = "Q"

# Q reuses D's config file wholesale rather than getting its own. That is the
# point: the delta Q reports is only attributable to the surface form if every
# other setting — learning rate, batch size, epochs, seeds, max_length, the
# checkpoint-selection metric — is identical to the run it is compared against. A
# near-duplicate config could drift from D's by one line and silently turn the
# finding into a hyperparameter artefact.
MODEL_CONFIG_NAME = "model_xlmr.yaml"

# The cross-dataset run Q's delta is measured against: same model, same direction,
# same seed, original surface form.
BASELINE_EXPERIMENT_ID = "F"
# The in-domain run both descend from, recorded so a reader can see what the model
# scored on the corpus it trained on without hunting for the file.
IN_DOMAIN_EXPERIMENT_ID = "D"

# One branch per seed inside the EXISTING XLM-R staging repo, never a new repo.
CHECKPOINT_BRANCH_SUFFIX = "-punct-ablation"

# Milestone 5's results subdirectory on the Hub (M4-6), matching Experiment F's.
RESULTS_SUBDIR = "milestone5/metrics"

# Same transfer as F. Written here rather than added to `run_cross_dataset.py`'s
# DIRECTIONS table — see `transfer_metadata`'s docstring for why.
Q_DIRECTION: dict[str, str] = {
    "source_dataset": "ax_to_grind",
    "source_split": "train",
    "target_dataset": "notri_fact",
    "target_split": "holdout",
    "description": (
        "train punctuation-stripped Ax-to-Grind -> test Notri-Fact (zero-shot); "
        "surface-form-matched counterpart of Experiment F"
    ),
}


# --- The ablation -------------------------------------------------------------

# "Not a Unicode letter, digit or whitespace." Python's `\w` is letters + digits +
# underscore, so the underscore is removed by the second alternative — it is
# Unicode category Pc (connector punctuation), not a letter or digit, and
# Notri-Fact has none. `_reference_strip` below is the literal reading of the
# definition, and a test asserts the two agree across the Unicode range, so this
# regex can never quietly diverge from what the docstring promises.
_NON_ALPHANUMERIC = re.compile(r"[^\w\s]|_", re.UNICODE)


def strip_punctuation(text: str) -> str:
    """Replace every non-alphanumeric, non-whitespace character with a space.

    Then collapse whitespace runs, so the substitution cannot leave a
    whitespace-density signal behind where the punctuation was. See the module
    docstring for why a removed character leaves a word boundary rather than
    vanishing.

    Applied to text that `clean.py` has ALREADY cleaned — this is an ablation on
    top of the shared preprocessing path, not a replacement for any part of it.
    """
    if not isinstance(text, str) or not text:
        return ""
    return collapse_whitespace(_NON_ALPHANUMERIC.sub(" ", text))


def _reference_strip(text: str) -> str:
    """The definition written out character by character, for cross-checking.

    Deliberately slow and literal: a character survives iff it is a Unicode letter
    (category `L*`), a Unicode number (`N*`) or whitespace; anything else becomes a
    space. Never used in a run — it exists so a test can prove
    `_NON_ALPHANUMERIC` implements the documented rule rather than something close
    to it.
    """
    kept = [
        character
        if character.isspace() or unicodedata.category(character)[0] in {"L", "N"}
        else " "
        for character in text
    ]
    return collapse_whitespace("".join(kept))


def strip_punctuation_frame(frame):  # noqa: ANN001, ANN201 - pandas DataFrame in/out
    """Return a copy of a split frame with the ablation applied to `text`."""
    ablated = frame.copy()
    ablated["text"] = ablated["text"].map(strip_punctuation)
    return ablated


def non_alphanumeric_profile(texts) -> dict[str, Any]:  # noqa: ANN001 - iterable of str
    """Count what the ablation removes (or, on Notri-Fact, confirm there is nothing).

    Measured on the corpus in hand at run time rather than quoted from M2-3's
    audit, so a metrics file records the state of the data that run actually saw.
    """
    total_characters = 0
    non_alphanumeric = 0
    distinct: dict[str, int] = {}

    for text in texts:
        if not isinstance(text, str):
            continue
        total_characters += len(text)
        for character in text:
            if character.isspace() or unicodedata.category(character)[0] in {"L", "N"}:
                continue
            non_alphanumeric += 1
            distinct[character] = distinct.get(character, 0) + 1

    ranked = sorted(distinct.items(), key=lambda item: (-item[1], item[0]))
    return {
        "n_characters": total_characters,
        "n_non_alphanumeric": non_alphanumeric,
        "density": round(non_alphanumeric / total_characters, 6) if total_characters else 0.0,
        "n_distinct_non_alphanumeric": len(distinct),
        # Capped: this is a provenance record, not a character-frequency dataset.
        "most_common": [{"character": c, "count": n} for c, n in ranked[:15]],
    }


def ablation_metadata(
    *,
    train_before,  # noqa: ANN001 - pandas DataFrame
    train_after,  # noqa: ANN001
    target,  # noqa: ANN001
) -> dict[str, Any]:
    """The self-describing record of what Q did to the data, and what it did not."""
    before = non_alphanumeric_profile(train_before["text"])
    after = non_alphanumeric_profile(train_after["text"])
    target_profile = non_alphanumeric_profile(target["text"])

    words_before = sum(len(text.split()) for text in train_before["text"])
    words_after = sum(len(text.split()) for text in train_after["text"])

    return {
        "ablation": {
            "type": "punctuation_stripping",
            "decision": "DECISION_REGISTER.md M2-3 (option c — run both, report the delta)",
            "definition": (
                "replace every character that is not a Unicode letter (L*), digit "
                "(N*) or whitespace with a single space, then collapse whitespace "
                "runs. Not only sentence punctuation: M2-3 measured zero "
                "non-alphanumeric characters of ANY kind in Notri-Fact, so "
                "matching that surface means removing all of them."
            ),
            "word_boundary_policy": {
                "policy": "substitute_space",
                "why": (
                    "A removed character leaves a word boundary rather than "
                    "vanishing. Deleting outright fuses the words either side "
                    "into tokens present in neither corpus, injecting a new "
                    "out-of-vocabulary artefact into the corpus whose confounds "
                    "this experiment isolates. Not fixed by M2-3 — decided when Q "
                    "was implemented; see punctuation_ablation.py's docstring."
                ),
                "n_words_before": words_before,
                "n_words_after": words_after,
            },
            "applied_to": [
                "ax_to_grind_train (the fine-tuning corpus)",
                "ax_to_grind_val (checkpoint selection only, so the selection "
                "split's surface form matches the training one)",
            ],
            "not_applied_to": [
                "notri_fact_holdout — already punctuation-free as released, and "
                "held out entirely (data.yaml train_on: false, R7)"
            ],
            "applied_on_top_of": (
                "research/src/data/clean.py's Tier-2 output. clean.py itself is "
                "UNCHANGED — it is the shared train/serve path (CLAUDE.md rule 4)."
            ),
            "placeholder_token_effect": (
                "clean.py's [URL]/[EMOJI] placeholders lose their brackets and "
                "become the bare words URL/EMOJI. They survive as Latin-script "
                "tokens, so their presence is still available to the model."
            ),
            "train_before": before,
            "train_after": after,
            "characters_removed": before["n_non_alphanumeric"] - after["n_non_alphanumeric"],
            "target_corpus": target_profile,
            # M2-3's premise, re-measured rather than assumed. If this is ever
            # non-zero the surfaces do NOT match after the ablation and Q's delta
            # means something narrower than the docstring claims.
            "surfaces_match_after_ablation": (
                after["n_non_alphanumeric"] == 0 and target_profile["n_non_alphanumeric"] == 0
            ),
        }
    }


# --- The comparison Q exists to make ------------------------------------------


def baseline_metrics_filename(seed: int, model_name: str) -> str:
    return f"{BASELINE_EXPERIMENT_ID}_{model_name}_notri_fact_holdout_seed{seed}.json"


def in_domain_metrics_filename(seed: int, model_name: str) -> str:
    return f"{IN_DOMAIN_EXPERIMENT_ID}_{model_name}_ax_to_grind_test_seed{seed}.json"


def _macro_f1_from(filename: str) -> float | None:
    """Read a committed result's macro-F1, or `None` if it is not on disk.

    Tolerant by design: a missing counterpart records `null`, never a guess
    (`CLAUDE.md` rule 2). Same pattern as `run_shortcut_analysis.py`'s H/H2 pair.
    """
    path = METRICS_DIR / filename
    if not path.exists():
        return None
    try:
        return float(json.loads(path.read_text(encoding="utf-8"))["metrics"]["macro_f1"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def comparison_metadata(seed: int, model_name: str) -> dict[str, Any]:
    """Name the runs Q's delta is measured against, and carry their numbers along.

    The point is that the delta must be computable straight from Q's own file. A
    reader should not have to know which of F's nine metrics files pairs with this
    one, nor re-derive that "same seed, same direction, same model" is the correct
    pairing.
    """
    baseline_file = baseline_metrics_filename(seed, model_name)
    in_domain_file = in_domain_metrics_filename(seed, model_name)

    return {
        "compares_against": {
            "purpose": (
                "Q minus F isolates the punctuation surface-form confound in the "
                "Ax-to-Grind -> Notri-Fact transfer (DECISION_REGISTER.md M2-3). "
                "The two runs differ in exactly one respect: Q's training corpus "
                "had every non-alphanumeric character removed. Same model, same "
                "config, same seed, same direction, same evaluation corpus."
            ),
            "cross_dataset_baseline": {
                "experiment_id": BASELINE_EXPERIMENT_ID,
                "metrics_file": baseline_file,
                "model": model_name,
                "dataset": "notri_fact",
                "split": "holdout",
                "seed": seed,
                "same_direction": True,
                "same_seed": True,
                "macro_f1": _macro_f1_from(baseline_file),
            },
            "in_domain_reference": {
                "experiment_id": IN_DOMAIN_EXPERIMENT_ID,
                "metrics_file": in_domain_file,
                "model": model_name,
                "dataset": "ax_to_grind",
                "split": "test",
                "seed": seed,
                "macro_f1": _macro_f1_from(in_domain_file),
                "note": (
                    "Context only. D trained on the corpus as released, so it is "
                    "not surface-matched to Q and the two are not a controlled pair."
                ),
            },
            "null_macro_f1_means": (
                "the counterpart file was absent when Q ran — recompute from the "
                "named file rather than treating the gap as a result"
            ),
        }
    }


# --- Post-write annotation ----------------------------------------------------


def finalise_metrics_file(
    path: Path, *, push_info: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Add the two facts that do not exist until after the record is written.

    Both are genuinely unavailable at `build_metrics_record` time, which is why
    this is a second pass rather than sloppiness:

    * `delta_macro_f1_q_minus_f` needs Q's own macro-F1, which is a product of the
      record being built.
    * `checkpoint` needs the revision SHA, which the Hub only assigns once the push
      completes — and the push cannot precede training.

    Nothing computed by `metrics.py` is touched; only `run_metadata` gains fields,
    and the record is re-validated afterwards so a malformed edit cannot survive.
    """
    record = json.loads(path.read_text(encoding="utf-8"))
    metadata = record["run_metadata"]

    comparison = metadata.get("compares_against")
    if comparison is not None:
        baseline = comparison["cross_dataset_baseline"]["macro_f1"]
        comparison["macro_f1_q"] = record["metrics"]["macro_f1"]
        comparison["delta_macro_f1_q_minus_f"] = (
            round(record["metrics"]["macro_f1"] - baseline, 6) if baseline is not None else None
        )

    if push_info and push_info.get("pushed"):
        metadata["checkpoint"] = {
            "repo_id": push_info.get("repo_id"),
            "revision_branch": push_info.get("revision_branch"),
            "revision_sha": push_info.get("revision_sha"),
        }

    problems = validate_metrics_record(record)
    if problems:
        raise RuntimeError(
            f"{path.name} violates the metrics contract after annotation: {problems}"
        )

    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return record


# --- Runner -------------------------------------------------------------------


def run_seed(
    seed: int,
    *,
    data_config: dict[str, Any],
    model_config: dict[str, Any],
    dry_run: Any = None,
    push: bool = True,
) -> dict[str, Any]:
    """Train Q at one seed and score it zero-shot on Notri-Fact.

    Imported lazily inside the function so that importing this module — for the
    transform, or from a test — does not require torch.
    """
    from research.src.models.transformer import (
        EvalTarget,
        push_checkpoint,
        train_one_seed,
    )

    model_name = model_config["model"]["name"]

    train = load_split_frame("ax_to_grind", "train")
    val = load_split_frame("ax_to_grind", "val")
    target = load_split_frame(
        Q_DIRECTION["target_dataset"], Q_DIRECTION["target_split"]
    )

    if dry_run is not None:
        # Slice here rather than leaving it to `train_one_seed`, so the ablation
        # profile recorded in the metadata describes the rows this run actually
        # trained on. `train_one_seed` re-applies the same `head` and the second
        # call is a no-op.
        train = train.head(dry_run.n_train)
        val = val.head(dry_run.n_eval)
        target = target.head(dry_run.n_eval)

    stripped_train = strip_punctuation_frame(train)
    # The selection split is stripped too. D selected its checkpoint on
    # Ax-to-Grind val in its original surface form; for Q the faithful analogue is
    # the same split in the surface form Q actually trained on, so that model
    # selection is not made on a distribution the training never saw. Val is
    # selection only — it is never written to a metrics file
    # (REPRODUCIBILITY.md Section 6 scopes the reported set to test splits).
    stripped_val = strip_punctuation_frame(val)

    ablation = ablation_metadata(
        train_before=train, train_after=stripped_train, target=target
    )
    profile = ablation["ablation"]
    print(
        f"  ablation: removed {profile['characters_removed']:,} non-alphanumeric "
        f"characters from {len(train):,} training rows "
        f"({profile['train_before']['n_distinct_non_alphanumeric']} distinct -> "
        f"{profile['train_after']['n_distinct_non_alphanumeric']})"
    )
    if not profile["surfaces_match_after_ablation"]:
        print(
            "    !! surfaces do NOT match after the ablation: train_after="
            f"{profile['train_after']['n_non_alphanumeric']}, target="
            f"{profile['target_corpus']['n_non_alphanumeric']} non-alphanumeric "
            "characters. M2-3's premise does not hold for this data — Q's delta "
            "measures something narrower than a clean surface-form match."
        )

    result = train_one_seed(
        EXPERIMENT_ID,
        model_config,
        data_config,
        seed,
        dry_run=dry_run,
        train_frame=stripped_train,
        selection_frame=stripped_val,
        selection_label="ax_to_grind_val:punctuation_stripped",
        eval_targets=[
            EvalTarget(
                Q_DIRECTION["target_dataset"], Q_DIRECTION["target_split"], target
            )
        ],
        train_split_label="ax_to_grind_train:punctuation_stripped",
        extra_metadata={
            "base_experiment_id": IN_DOMAIN_EXPERIMENT_ID,
            **ablation,
            **transfer_metadata(EXPERIMENT_ID, Q_DIRECTION),
            **comparison_metadata(seed, model_name),
        },
    )

    should_push = push and model_config["checkpoint_push"].get("enabled", False)
    push_info = push_checkpoint(
        result.pop("trainer"),
        result.pop("tokenizer"),
        model_config,
        seed,
        dry_run=bool(dry_run) or not should_push,
        branch_suffix=CHECKPOINT_BRANCH_SUFFIX,
    )
    result["push"] = push_info
    print(f"  checkpoint: {push_info}")

    for path in result["metrics_files"]:
        record = finalise_metrics_file(Path(path), push_info=push_info)
        comparison = record["run_metadata"]["compares_against"]
        print(
            f"  delta vs {BASELINE_EXPERIMENT_ID} (seed {seed}): "
            f"Q={comparison['macro_f1_q']:.4f} "
            f"F={comparison['cross_dataset_baseline']['macro_f1']} "
            f"delta={comparison['delta_macro_f1_q_minus_f']}"
        )

    # M4-6: metrics and their per-example predictions leave this disk as each seed
    # finishes, not batched at the end. Predictions are included because Q's
    # holdout file is a cross-dataset test set, which is precisely what
    # REPRODUCIBILITY.md Section 6 requires be recoverable without re-running
    # inference — losing it would cost the same GPU hour to rebuild as the metrics.
    result["results_push"] = push_result_files(
        [*result["metrics_files"], *result.get("prediction_files", [])],
        subdir=RESULTS_SUBDIR,
        dry_run=bool(dry_run) or not should_push,
    )
    return result


def run_punctuation_ablation(
    *,
    data_config: dict[str, Any],
    model_config: dict[str, Any],
    seeds: list[int] | None = None,
    dry_run: Any = None,
    push: bool = True,
) -> list[dict[str, Any]]:
    """Run Q across its seed set."""
    seeds = seeds or (list(dry_run.seeds) if dry_run else list(model_config["training"]["seeds"]))

    results = []
    for seed in seeds:
        print(f"\n--- {EXPERIMENT_ID} / {model_config['model']['name']} / seed {seed} ---")
        results.append(
            run_seed(
                seed,
                data_config=data_config,
                model_config=model_config,
                dry_run=dry_run,
                push=push,
            )
        )
    return results


def main(argv: list[str] | None = None) -> int:
    from research.src.models.transformer import (
        DryRunSettings,
        load_config,
        resolve_dry_run_destination,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default="data.yaml")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Defaults to the config's seed set {42, 123, 2026}. Not a tunable.",
    )
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

    data_config = load_config(args.data_config)
    model_config = load_config(MODEL_CONFIG_NAME)

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

    print(f"=== Experiment {EXPERIMENT_ID} — punctuation ablation (M2-3, RQ2) ===")
    print(f"  {Q_DIRECTION['description']}")
    print(f"  model: {model_config['model']['name']} (config {MODEL_CONFIG_NAME}, shared with D)")
    print(f"  checkpoint branches: seed-<n>{CHECKPOINT_BRANCH_SUFFIX}")
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

    results = run_punctuation_ablation(
        data_config=data_config,
        model_config=model_config,
        seeds=args.seeds,
        dry_run=dry_run,
        push=not args.no_push,
    )

    print("\n=== summary ===")
    print(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
