# Milestone 4 — Colab handoff (Experiments C and D)

**For: Rehman. Estimated hands-on time: ~10 minutes of setup, then ~1–2 hours unattended.**

This is the GPU step Claude Code cannot run (`PROJECT_SPECIFICATION.md` Section 6 —
Colab GPU access is not available to the coding agent). Everything is prepared and
dry-run tested on CPU; your job is to execute it on a T4 and bring the results back.

---

## 1. Before you open the notebook

### Secrets you must create

| What | Where to get it | Used for |
|---|---|---|
| **Hugging Face write token** | huggingface.co → Settings → Access Tokens → **New token, role: Write** | Pushing the 6 trained checkpoints to a staging repo |

That is the only credential needed. There is no Kaggle step here (the data is
already committed as split index files) and no Weights & Biases account is required
— logging is local JSON only, as you asked.

### Environment variables to set in the Colab session

Both are read from the environment; neither is ever written into a file or committed.

| Variable | Value | Notes |
|---|---|---|
| `HF_TOKEN` | your HF **write** token | Set via Colab **Secrets** (🔑 icon in the left sidebar), name it `HF_TOKEN`, and enable "Notebook access". The notebook reads it from there. |
| `HF_STAGING_PREFIX` | your HF username or org, e.g. `rehman-ayoub` | The namespace the staging repos are created under. Claude Code cannot know this. |

The notebook has a cell for both — set `HF_STAGING_PREFIX` there directly (it is not
secret), and put `HF_TOKEN` in Colab Secrets rather than typing it into a cell, so it
never ends up in the notebook's saved output.

### Runtime

**Runtime → Change runtime type → T4 GPU → Save.** The notebook asserts a GPU is
present and stops immediately if not, rather than silently spending hours on CPU.

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
~1–2 min), checkpoint upload (~1.1 GB XLM-R / ~0.7 GB mBERT, ~30 s each on Colab's
link), and a one-off ~2 min model download per architecture:

**Total realistic range: ~35–60 minutes for all 6 runs. Budget 1 hour.**

⚠️ Colab free-tier sessions disconnect after ~90 min idle. Keep the tab open, or run
the two models in separate sessions (`--experiments C` then `--experiments D`) if you
would rather not babysit a single long run.

The real numbers land in each metrics file as `run_metadata.train_runtime_seconds` —
replace this estimate with those when you report back.

---

## 4. Running it

Open `research/notebooks/04_transformer_training.ipynb` in Colab and run every cell
top to bottom. It:

1. checks a GPU is attached,
2. syncs the repo to the branch tip and prints the commit it landed on,
3. installs `research/requirements.txt` (pinned),
4. reads `HF_TOKEN` from Colab Secrets and `HF_STAGING_PREFIX` from the cell,
5. runs a fast **smoke test** (a few real steps on a tiny slice) before the real job —
   if this fails, stop and send the error rather than burning an hour,
6. runs all 6 real training runs,
7. pushes checkpoints and prints a summary table,
8. zips the metrics files for download.

### If a fix has landed since you opened the notebook

**Re-open the notebook from GitHub. Do not just restart the runtime.**

`Runtime → Restart session` restarts the *kernel*; it does **not** reload the
notebook source in a tab you already have open. So after a fix is pushed, an open
tab keeps executing the old cells while step 2 above pulls the new repo code — the
two drift apart and the failure looks like the fix did not work. This happened on
2026-08-15: the data check ran unscoped and the smoke test hit the torchvision
error, both because the tab predated the fix.

`File → Open notebook → GitHub → this repo → 04_transformer_training.ipynb`, then
run from the top. Section 3's second cell now compares the notebook you are running
against the committed one and stops with an explicit message if yours is older, so
you will not have to diagnose this by hand again.

---

## 5. What to bring back

1. **The 12 metrics JSON files** from `research/results/metrics/` (step 8 zips them).
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
