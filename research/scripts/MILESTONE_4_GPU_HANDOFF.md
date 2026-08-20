# Milestone 4 — hosted-GPU handoff (Experiments C and D)

**For: Rehman. Estimated hands-on time: ~10 minutes of setup, then ~1–2 hours unattended.**

This is the GPU step Claude Code cannot run (`PROJECT_SPECIFICATION.md` Section 6 —
hosted GPU access is not available to the coding agent). Everything is prepared and
dry-run tested on CPU; your job is to execute it on a GPU and bring the results back.

**Runs on Colab or Kaggle.** One notebook covers both
(`research/notebooks/04_transformer_training.ipynb`) — the platforms differ in only
three mechanical ways, all handled by `research/src/notebook_env.py`. Section 1.1
below is the only part you read differently depending on which you use; everything
from section 2 onward is identical.

> **Looking for Experiment F (cross-dataset zero-shot)?** That is a different job —
> inference only, both corpora, an extra Kaggle credential — and has its own doc and
> notebook: `research/scripts/MILESTONE_5_GPU_HANDOFF.md` and
> `research/notebooks/05_cross_dataset_eval.ipynb`. This doc covers Milestone 4's
> **training** runs only. The environment cells are shared verbatim between the two
> notebooks so they cannot drift.

> **Current status: Kaggle.** Colab's free GPU quota was exhausted on 2026-08-19
> (a usage-limit error, not a bug — confirmed live), so this run moves to Kaggle.
> The Colab path is kept working and documented, not deleted, since the quota
> refills.

---

## 0. Two habits this notebook now enforces

Both were added after Milestone 4 lost a run's results and needed a same-day
recovery (`DECISION_REGISTER.md` M4-6). They are the default now, not something to
remember.

**Durability first.** Every metrics file is pushed to
`<HF_STAGING_PREFIX>/urdu-misinfo-results-staging` the moment it is written — per
seed, per split — never batched to the end. The packaging cell then verifies with
the Hub that everything arrived and **stops the notebook if it cannot confirm**. The
session disk is never the only copy, not even briefly.

**One cell to restore state, and a hard guard on expensive cells.** Section 0 is a
single idempotent cell that re-establishes `REPO_DIR`, `HF_STAGING_PREFIX`,
`os.environ["HF_TOKEN"]` and `sys.path` from scratch, re-syncing the repo if needed.
After any kernel restart that is the **only** cell to re-run — not five, in the right
order. It is safe to run repeatedly and at any point.

It also sets `CONFIRM_TRAIN = False`. The two cells that spend real GPU time check
that flag and print a clear "SKIPPED" message instead of running, so an accidental
**Run All** or a misclick cannot start hours of training. To train deliberately, put
`CONFIRM_TRAIN = True` in a cell of your own and re-run the training cell. Restoring
state resets it to `False` — recovering from a crash must never re-arm training.

---

## 1. Before you open the notebook

### Secrets you must create

| What | Where to get it | Used for |
|---|---|---|
| **Hugging Face write token** | huggingface.co → Settings → Access Tokens → **New token, role: Write** | Pushing the 6 trained checkpoints to a staging repo |

That is the only credential needed. The data needs no Kaggle *API* token (it is
already committed as split index files, and `download.py` fetches Ax-to-Grind
without credentials), and no Weights & Biases account is required — logging is
local JSON only, as you asked.

### Environment variables

Both are read from the environment; neither is ever written into a file or committed.

| Variable | Value | Notes |
|---|---|---|
| `HF_TOKEN` | your HF **write** token | Read from the platform's secret store — never typed into a cell, so it cannot end up in saved notebook output. See 1.1 for where that store is. |
| `HF_STAGING_PREFIX` | your HF username or org, e.g. `rehman-ayoub` | Not secret. Set it directly in the notebook's config cell. Claude Code cannot know this. |

---

## 1.1 Platform setup — the only part that differs

| | **Kaggle** | **Colab** |
|---|---|---|
| **GPU** | **Settings → Accelerator → GPU T4 x2** (or P100). There is no "Runtime → Change runtime type" menu; the accelerator is a *notebook setting*. | **Runtime → Change runtime type → T4 GPU → Save** |
| **Internet** | ⚠️ **Settings → Internet → On.** Kaggle disables outbound internet **by default**. Requires a phone-verified account. | Always on; nothing to set. |
| **Secret store** | **Add-ons → Secrets** → add `HF_TOKEN`, and tick it to **attach it to this notebook**. Read via `kaggle_secrets.UserSecretsClient`. | **Secrets** (🔑 icon, left sidebar) → add `HF_TOKEN`, enable **"Notebook access"**. Read via `google.colab.userdata`. |
| **Working dir** | `/kaggle/working` | `/content` |
| **Getting the zip back** | The session's **Output** tab (right-hand panel). Kaggle has no browser-download call. | Downloads automatically via the browser. |
| **Session limits** | ~30 GPU-hours/week; 12 h max per session; 20 min idle timeout. | Free tier disconnects after ~90 min idle; quota varies. |

**The internet toggle is the single easiest thing to miss.** With it off, the repo
clone, `pip install` and the HF checkpoint push all fail — several cells apart, with
three unrelated-looking errors. The notebook's first cell now checks it explicitly
and fails with one clear message instead.

### ⚠️ Kaggle's "T4 x2" and batch size

Kaggle's default GPU option gives you **two** T4s. With two GPUs visible, the
Hugging Face `Trainer` silently wraps the model in `nn.DataParallel` and treats
`per_device_train_batch_size: 16` as *per device* — making the effective batch **32**
and cutting the 401 optimizer steps/epoch that this project's runtime estimate and
3-seed design assume down to ~201. That is a change to the training dynamics, not a
free speed-up.

The notebook therefore sets `CUDA_VISIBLE_DEVICES=0` in its pre-flight cell, so the
run uses one GPU and stays numerically comparable to the single-T4 design. **Nothing
in the config or the training loop was changed** — restricting visibility is what
*preserves* the frozen design. Selecting **P100** (a single GPU) instead is equally
fine and avoids the issue by construction.

---

## 2. What will be produced

Two staging repos, three branches each — one per seed, so all three are addressable
by a pinned revision and none overwrites another:

```
<HF_STAGING_PREFIX>/urdu-misinfo-mbert-staging   branches: seed-42, seed-123, seed-2026
<HF_STAGING_PREFIX>/urdu-misinfo-xlmr-staging    branches: seed-42, seed-123, seed-2026
```

These are **staging**, deliberately not the production `MODEL_REPO_ID`. Promotion to
production is a separate decision at Milestone 9, made on defensibility rather than
raw in-domain F1 (`EXPERIMENT_PLAN.md` Section 7).

Plus one **dataset** repo holding the metrics, written to as each seed finishes so
the results are never only on the session disk (`DECISION_REGISTER.md` M4-6):

```
<HF_STAGING_PREFIX>/urdu-misinfo-results-staging   milestone4/metrics/*.json
```

Plus 12 metrics files in `research/results/metrics/`:

```
C_bert-base-multilingual-cased_ax_to_grind_{val,test}_seed{42,123,2026}.json
D_xlm-roberta-base_ax_to_grind_{val,test}_seed{42,123,2026}.json
```

---

## 3. Runtime estimate

Computed from the actual configs, not guessed:

- Training split after dedup: **6,405 rows**
- Batch size 16, gradient accumulation 1 → **401 optimizer steps/epoch**
- 3 epochs → **1,203 steps per seed**
- 2 models × 3 seeds = **6 runs → 7,218 optimizer steps total**

Batches use **dynamic padding** (each batch padded to its own longest row rather
than to 512). Measured over the real training split, the average batch maximum is
**401 tokens** — a ~1.3× saving against a fixed 512, not the large one the median
article length (40 tokens) might suggest, because the fake class's heavy tail means
most batches of 16 contain at least one long article.

Compute per optimizer step at batch 16, average padded length 401, for a base-size
encoder ≈ **4.17 TFLOPs**. A T4 peaks at ~65 TFLOPS fp16; real sustained utilisation
for this workload is typically 20–35%:

| Utilisation | s/step | per seed (1,203 steps) | 6 runs |
|---|---|---|---|
| 20% (conservative) | 0.32 | ~6.4 min | ~39 min |
| 35% (optimistic) | 0.18 | ~3.7 min | ~22 min |

Add per-seed evaluation (3 epoch-evals on val, plus final val+test — forward-only,
~1–2 min), checkpoint upload (~1.1 GB XLM-R / ~0.7 GB mBERT, ~30 s each on a hosted
link), and a one-off ~2 min model download per architecture:

**Total realistic range: ~35–60 minutes for all 6 runs. Budget 1 hour.**

The table above is for one T4. A **P100** is roughly 1.5–2× a T4 for this workload,
so treat those figures as an upper bound there. The `T4 x2` option does **not** halve
the time — the notebook deliberately uses one of the two GPUs (see 1.1).

⚠️ Sessions time out when idle: Colab free tier after ~90 min, Kaggle after ~20 min
(and 12 h maximum per session regardless). Keep the tab open, or run the two models
in separate sessions (`--experiments C` then `--experiments D`) if you would rather
not babysit a single long run. Kaggle also caps free GPU use at ~30 hours/week —
ample for this job, which needs about one.

The real numbers land in each metrics file as `run_metadata.train_runtime_seconds` —
replace this estimate with those when you report back.

---

## 4. Running it

Open `research/notebooks/04_transformer_training.ipynb` and run every cell top to
bottom. It:

1. checks a GPU **and outbound internet**, and pins the run to one GPU,
2. syncs the repo to the branch tip and prints the commit it landed on,
3. checks the notebook you are running is not older than the committed one,
4. reads `HF_TOKEN` from the platform's secret store (Colab Secrets / Kaggle
   Add-ons → Secrets, chosen automatically) and `HF_STAGING_PREFIX` from the cell,
5. installs `research/requirements.txt` (pinned) and asserts the versions took,
   **including Python 3.12** (`DECISION_REGISTER.md` M4-2),
6. runs a fast **smoke test** (a few real steps on a tiny slice) before the real job —
   if this fails, stop and send the error rather than burning an hour,
7. runs all 6 real training runs,
8. pushes checkpoints and prints a summary table,
9. zips the metrics files and hands them back the platform's way.

**To open it on Kaggle:** New Notebook → File → Import Notebook → GitHub (or upload
the `.ipynb`). On Colab: `File → Open notebook → GitHub → this repo`.

### If the checkpoints exist but the metrics are gone (recovery mode)

Section **5b** of the notebook. Skip it on a normal run.

If a session is lost after training completed but before the metrics were secured,
you do **not** need to retrain. The pushed checkpoints are the exact models that
produced those numbers (`load_best_model_at_end` means the best-epoch model is what
got pushed), and scoring is deterministic, so re-scoring reproduces them.

1. **Step 0 — inventory.** CPU, seconds, no GPU cost. Confirms all six seed branches
   carry config, weights *and* tokenizer. Read its output before going further: if it
   reports some seeds missing, only those need retraining.
2. **Step 1 — re-score.** Downloads each checkpoint and re-evaluates val/test.
   Minutes, not hours — the compute is a forward pass over ~2,750 rows per seed; the
   wall-clock is mostly the ~5.4 GB of downloads.

Two fields cannot be recovered this way and are written as `null` rather than
guessed: `train_runtime_seconds` and `train_loss`. Neither feeds RQ1, RQ2 or RQ3.
Recovered files declare themselves in `run_metadata.evaluation_only_recovery`, and
the summary table flags them in a `recov` column.

**Do not retrain to recover metrics.** GPU training is not bit-deterministic, so a
retrain produces different weights and would overwrite the seed branches, orphaning
the revision SHAs already recorded — leaving numbers that match no artefact you hold.

### If a fix has landed since you opened the notebook

**Re-open the notebook from GitHub. Do not just restart the session.**

Restarting the session restarts the *kernel*; it does **not** reload the notebook
source in a tab you already have open (true on both Colab's `Runtime → Restart
session` and Kaggle's `Run → Restart session` / Factory reset). So after a fix is
pushed, an open tab keeps executing the old cells while step 2 above pulls the new
repo code — the two drift apart and the failure looks like the fix did not work.
This happened on 2026-08-15: the data check ran unscoped and the smoke test hit the
torchvision error, both because the tab predated the fix.

Re-open it from GitHub and run from the top. Section 3's stale-notebook cell compares
the notebook you are running against the committed one and stops with an explicit
message if yours is older, so you will not have to diagnose this by hand again.

---

## 5. What to bring back

> ### ⚠️ The run is not done until the results are on Hugging Face
>
> **A Kaggle draft session's working directory is not storage.** On 2026-08-19 all
> six runs completed, all six checkpoints pushed successfully, the metrics zip was
> packaged — and then the Output-tab download silently failed (Kaggle was returning
> 503s), the session went idle, the tab was reloaded, and `/kaggle/working` came
> back **empty**. Every metrics file was lost. The checkpoints survived only because
> they had been pushed to HF as each seed finished. `DECISION_REGISTER.md` M4-6.
>
> The notebook now pushes each seed's metrics to
> `<HF_STAGING_PREFIX>/urdu-misinfo-results-staging` the moment they are written,
> and the packaging cell **stops the notebook** if it cannot confirm with the Hub
> that every file arrived. If that cell raises: **do not close the session.** Re-run
> it. The metrics still exist in `research/results/metrics/` at that point, but only
> there — which is exactly the state that lost them last time.
>
> Treat the Output-tab zip as a convenience copy, never as the copy.

1. **The 12 metrics JSON files.** They should already be on HF in the results repo
   above — check there first; that is the durable copy. The zip from the session's
   **Output** tab is a convenience.
2. **The staging repo IDs and, for each seed, the revision SHA** printed in the
   summary table. `REPRODUCIBILITY.md` Section 6 requires a checkpoint be identified
   by repo ID **plus** revision, not repo ID alone — a branch name moves, a SHA does not.
3. **Anything that looked wrong** — a run that diverged, an OOM, a seed whose
   macro-F1 sits far from the other two.

Commit the metrics files, or paste them into the next session and Claude Code will.

---

## 6. Things worth knowing before you look at the results

- **Every run truncates at 512 tokens.** Both architectures cap there, so
  "uncapped training text" is not physically available (`DECISION_REGISTER.md` M4-1).
  About **4.7% of Ax-to-Grind is truncated**, concentrated in the extreme-length fake
  tail. Each metrics file records exactly how many of its own rows were affected under
  `run_metadata.truncation`. This constrains how RQ3's results may be phrased — see M4-1.
- **Watch `prediction_collapse` in each file.** A model predicting one class for
  ≥95% of inputs is flagged there. That is the exact failure mode Haroon (2026)
  reported at 99.7%, and it is why macro-F1 rather than accuracy is the primary metric.
- **Compare against the Milestone 3 baselines**, which are already committed:
  classical TF-IDF reached macro-F1 **0.8755** (A) and **0.8835** (B) on the same test
  split, and a length-only model with **no content access** reached **0.6145** linear /
  ~**0.76** nonlinear. A transformer that lands near those numbers has not
  demonstrated much genuine linguistic signal — that comparison is the point of the
  whole experiment, not a formality.
