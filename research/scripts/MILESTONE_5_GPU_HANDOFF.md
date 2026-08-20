# Milestone 5 — Experiment F handoff: cross-dataset zero-shot (transformers)

**For: Rehman. Hands-on: ~5 minutes of setup, then ~10–20 minutes unattended on a GPU.**

Notebook: `research/notebooks/05_cross_dataset_eval.ipynb`.

## Why this is a separate doc from `MILESTONE_4_GPU_HANDOFF.md`

The environment setup is identical and is *copied verbatim* from notebook 04 — the
sync, stale-guard, dependency-install and environment-gate cells carry the M4-3,
M4-4 and M4-5 fixes and must not drift. What differs is everything around them, and
it differs enough that folding it into the Milestone 4 doc would have made both
harder to follow:

| | Milestone 4 (notebook 04) | Milestone 5 / F (notebook 05) |
|---|---|---|
| Work | **Trains** 6 checkpoints | **Inference only** — scores 6 existing checkpoints |
| Corpora | Ax-to-Grind **only** (deliberately) | **Both** — evaluates on Notri-Fact |
| Credentials | `HF_TOKEN` | `HF_TOKEN` **+ `KAGGLE_API_TOKEN`** (new) |
| Guard | `CONFIRM_TRAIN` | `CONFIRM_EVAL` (deliberately a different flag) |
| GPU | Hard requirement | Strongly preferred, **not** required |
| Runtime | ~35–60 min | ~10–20 min on GPU; hours on CPU |

---

## 1. Before you open it

### Two secrets, not one

Both go in **Add-ons → Secrets**, attached to this notebook (Colab: 🔑 sidebar).

| Secret | Why | Notes |
|---|---|---|
| `HF_TOKEN` | Pull the 6 checkpoints; push results back | **Needs WRITE.** Read alone is not enough — results are pushed to the Hub as each seed finishes (M4-6). |
| `KAGGLE_API_TOKEN` | Download Notri-Fact | **New for this notebook.** From kaggle.com/settings/api. Milestone 4 never needed it because it only touched Ax-to-Grind. |

> ⚠️ The local read-only token used for verification is **not sufficient here** — it
> can pull checkpoints but cannot push results, which would silently defeat M4-6's
> durability guarantee. Use a write token in the Kaggle secret.

### Settings

- **Accelerator → GPU T4 x2** (or P100). Missing GPU is a warning, not a stop —
  inference on CPU gives identical numbers, just slowly.
- **Internet → On.** Off by default; nothing works without it.

---

## 2. Cells to run, in order

Run top to bottom, **except** that the expensive cell needs one deliberate action.

| # | Section | Cell | Note |
|---|---|---|---|
| 1 | 0 | Restore state | Edit `HF_STAGING_PREFIX` and `REPO_URL` first. **After any kernel restart, this is the only cell to re-run.** |
| 2 | 1 | Pre-flight | GPU + internet. Pins `CUDA_VISIBLE_DEVICES=0`. |
| 3 | 2 | Repo sync | |
| 4 | 2 | Stale-notebook guard | Stops if your tab is older than the repo. |
| 5 | 2 | Dependency install | torchvision removal, pinned set, numpy/scipy force-reinstall. |
| 6 | 2 | Environment gate | Runs in a subprocess (M4-5). **Must print "All environment checks passed."** |
| 7 | 3 | Data check | Downloads **both** corpora. Needs `KAGGLE_API_TOKEN`. |
| 8 | — | **`CONFIRM_EVAL = True`** | **Type this into a new cell yourself.** It is not set anywhere by default. |
| 9 | 4 | Experiment F | The real run. 6 scoring passes, ~5.4 GB of downloads. |
| 10 | 5 | Summary | The RQ2 table — copy this back. |
| 11 | 6 | Package | Pushes + verifies + zips. **Stops the notebook if the Hub cannot confirm.** |

If you run everything without doing step 8, cell 9 prints `SKIPPED` and changes
nothing. That is intended: a stray "Run All" must not start a multi-gigabyte job.

### Why `CONFIRM_EVAL` rather than reusing `CONFIRM_TRAIN`

Not pedantry — reusing it would have been unsafe. `CONFIRM_TRAIN` is a single
global, so setting it to run an *evaluation* would simultaneously arm every
training cell in the same kernel, and a later "Run All" could start hours of
unintended fine-tuning. It would also normalise flipping the training flag for
routine work, which is exactly how a guard stops being a guard. Same mechanism,
separate flag, independent: setting one never arms the other.

---

## 3. What will be produced

Six metrics files in `research/results/metrics/`:

```
F_bert-base-multilingual-cased_notri_fact_holdout_seed{42,123,2026}.json
F_xlm-roberta-base_notri_fact_holdout_seed{42,123,2026}.json
```

pushed to `<HF_STAGING_PREFIX>/urdu-misinfo-results-staging` under
`milestone5/metrics/` **as each seed completes**, not batched at the end. A
disconnect mid-run therefore costs only the seed in flight.

No checkpoints are written or modified. This notebook cannot alter the Milestone 4
staging repos — it only reads them.

---

## 4. What to bring back

1. **The summary table from section 5.** In-domain vs zero-shot macro-F1 per model
   per seed, the delta, and `dominant %`.
2. **Anything that looked wrong** — a seed far from the other two, an OOM, a
   download stall.

The metrics themselves should already be on the Hub; the Output-tab zip is a
convenience copy, never the copy (`DECISION_REGISTER.md` M4-6).

---

## 5. Reading the result

- **The delta is RQ2's headline.** In-domain, C reached macro-F1 0.9211 ± 0.0025 and
  D 0.9269 ± 0.0042 on Ax-to-Grind test. The question is how much survives transfer.
- **Read `dominant %` at least as closely as macro-F1.** A model that learned a
  dataset-specific shortcut typically collapses toward one class on an unseen
  corpus, and macro-F1 alone can understate that. The classical baselines already
  show it: A and B fell to 0.4948 / 0.4787 with 68.6% / 71.5% dominant-class share —
  near-chance on a balanced binary task.
- **A transformer landing near the classical numbers is itself the finding**, not a
  failure of the run. That comparison is the point of the experiment
  (`MASTER_PROJECT_BLUEPRINT.md` Part 13).
- **This is one direction only.** Experiment G (Notri-Fact → Ax-to-Grind) is
  resolved-but-deferred (`DECISION_REGISTER.md` M5-1) and needs its own split plus
  6 further training runs. Until it lands, F measures a drop; it does **not**
  evidence the *asymmetry* Part 13 hypothesises.
