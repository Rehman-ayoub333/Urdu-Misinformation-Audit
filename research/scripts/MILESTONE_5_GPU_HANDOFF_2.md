# Milestone 5 — combined GPU handoff: predictions backfill + Experiments Q and I

**For: Rehman. Hands-on: ~5 minutes of setup, then ~1.5–2.5 hours, mostly unattended
but you must stay attached (see §6).**

Notebook: `research/notebooks/06_milestone5_combined.ipynb`.

## Why one notebook and not three

Four pieces of work that each need the same environment and the same ~6.5 GB of
model downloads. Run separately they would pay that cost three times and burn
three sessions' setup. Run together they share it.

| § | Work | Trains? | Guard | New artefact |
|---|---|---|---|---|
| **a** | C/D predictions backfill (in-domain) | No | none | `.predictions.jsonl` × 6 |
| **b** | F predictions backfill (cross-dataset) | No | none | `.predictions.jsonl` × 6 |
| **c** | Experiment Q — punctuation ablation | **Yes**, 3 runs | `CONFIRM_Q` | `Q_*.json` × 3 + predictions |
| **d** | Experiment I — length ablation, XLM-R half | **Yes**, 12 runs | `CONFIRM_I` | `I_xlm-roberta-base_*.json` × 12 + predictions |

**Experiment I's classical half is already done.** B/tfidf_svm is deterministic and
needs no GPU, so all four caps were run locally on 2026-08-21 and are committed.
The notebook runs I's XLM-R half only (`--models transformer`), deliberately — see
§7.

---

## 1. Before you open it

### Two secrets — the same pair notebook 05 used

Both in **Add-ons → Secrets**, attached to this notebook (Colab: 🔑 sidebar).

| Secret | Why | Notes |
|---|---|---|
| `HF_TOKEN` | Pull the 6 existing checkpoints; push 15 new ones; push every result file | **Needs WRITE.** A read token gets you through sections a and b and then fails at the first checkpoint push in section c, ~40 minutes in. |
| `KAGGLE_API_TOKEN` | Download Notri-Fact | From kaggle.com/settings/api. Sections b and c evaluate on it. |

> ⚠️ The read-only token used for local verification is **not** sufficient. This
> notebook pushes 15 checkpoints and ~40 result files.

### Settings

- **Accelerator → GPU T4 x2** (or P100). **Effectively mandatory**, unlike notebook
  05: sections c and d fine-tune 15 models. The notebook *refuses* to start them
  without a GPU rather than warning, because 15 fine-tunes on CPU cannot finish
  inside Kaggle's 12-hour session cap and would waste the whole session.
- **Internet → On.** Off by default; nothing works without it.
- Set `HF_STAGING_PREFIX` and `REPO_URL` in the restore-state cell.

---

## 2. Cells to run, in order

| # | Section | Cell | Note |
|---|---|---|---|
| 1 | 0 | Restore state | Edit `HF_STAGING_PREFIX` and `REPO_URL` first. **After any kernel restart, this is the only cell to re-run.** |
| 2 | 1 | Pre-flight | GPU + internet. Sets `HAS_GPU`, pins `CUDA_VISIBLE_DEVICES=0`. |
| 3 | 2 | Repo sync | |
| 4 | 2 | Stale-notebook guard | Stops if your tab is older than the repo. |
| 5 | 2 | Dependency install | torchvision removal, pinned set, numpy/scipy force-reinstall. |
| 6 | 2 | Environment gate | Runs in a subprocess (M4-5). **Must print "All environment checks passed."** |
| 7 | 3 | Data check | Downloads **both** corpora. Needs `KAGGLE_API_TOKEN`. |
| 8 | a | C/D backfill | Ungated. ~5.4 GB of downloads. |
| 9 | a | **a-verify** | **Must print "all C/D aggregates reproduce…". If it raises, stop.** |
| 10 | b | F backfill | Ungated. Checkpoints already cached. |
| 11 | b | **b-verify** | Same. **If it raises, stop.** |
| 12 | — | **`CONFIRM_Q = True`** | **Type this into a new cell yourself.** |
| 13 | c | Experiment Q | 3 fine-tunes. |
| 14 | — | **`CONFIRM_I = True`** | **Type this into a new cell yourself.** |
| 15 | d | Experiment I | 12 fine-tunes. Local checkpoints are pruned automatically as each push is verified (§5.1). |
| 16 | e | Disk check | Should report **nothing left**. Anything listed is a run whose upload could not be confirmed — read it, don't just delete it. |
| 17 | f | Summary | The three tables — copy these back. |
| 18 | g | Package | Pushes + verifies + zips. **Stops the notebook if the Hub cannot confirm.** |

A plain **Run All** does sections a and b and then stops at c and d with a
`SKIPPED` message, changing nothing. That is intended.

### Why two confirm flags rather than one

`CONFIRM_Q` and `CONFIRM_I` are independent, for the reason notebook 05 refused to
reuse notebook 04's `CONFIRM_TRAIN`: a single global would mean confirming Q also
arms I, so a later Run All could start 12 unintended fine-tunes. They are also
independently restartable — if Q succeeds and I dies, you re-arm only I.

**Each guarded cell ALSO passes `--confirm-real-run` to the runner** (added by
`DECISION_REGISTER.md` M5-6, after a unit test started a real training run because
nothing structural stopped it). Belt and suspenders on purpose: the notebook flag
alone is one Run-All away from being ignored, and the CLI flag alone would be
satisfied by this cell every time.

---

## 3. Runtime budget

Computed from the actual configs and a measured sequence-length profile, not
guessed — same method as `MILESTONE_4_GPU_HANDOFF.md` §3.

**Base unit.** One uncapped XLM-R fine-tune is 6,405 rows / batch 16 = 401 steps ×
3 epochs = **1,203 optimizer steps**, at **0.18–0.32 s/step** on a T4 → **3.6–6.4
min** of training.

**Section d is not 12 full-length runs.** Capping shortens sequences and attention
is superlinear in length. Measured `E[batch max]` over the real training split at
batch 16, with the XLM-R tokenizer:

| cap | mean words | mean subwords | E[batch max] | relative cost |
|---|---|---|---|---|
| uncapped | 77.1 | 83.3 | **405.0** | 1.00 |
| 200 | 50.3 | 66.0 | 235.6 | 0.58 |
| 100 | 39.5 | 52.5 | 131.2 | 0.32 |
| 50 | 31.1 | 42.0 | 72.6 | 0.18 |
| 25 | 21.3 | 29.8 | 41.3 | 0.10 |

(The uncapped 405 independently reproduces the 401 recorded in `build_dataset`'s
docstring.) The four caps together cost **1.19×** one uncapped run, so all twelve
runs are about **3.6 uncapped-run equivalents**, not 12.

| § | Work | Estimate |
|---|---|---|
| setup | deps, data, env gate | 8–12 min |
| a | 6 checkpoints × 2,748 rows + **5.4 GB download** | 8–20 min |
| b | 6 checkpoints × 13,355 rows, cached | 8–15 min |
| c | 3 × (train 3.6–6.4 + eval + 1.1 GB upload) | 18–34 min |
| d | 3.6 uncapped-equivalents + 12 × (eval + 1.1 GB upload) | 24–52 min |
| | **total** | **~66–133 min** |

**Call it 1.5–2.5 hours.**

### Does this exceed a single Kaggle session? No — and here is the check

- **Session cap: 12 h.** We need ~2 h. **5–6× headroom.**
- **Weekly quota: ~30 GPU-h.** We use ~2 h, about **7%**.
- **Idle timeout: 20 min.** This is the binding constraint, and it is about your
  attention rather than the quota — see §6.

So **do not split this into two sessions.** Splitting would double the setup and
re-download the 5.4 GB of checkpoints for no benefit. The things that could
genuinely kill the run are disk and storage, not quota — §5.

If you would rather de-risk anyway, the natural seam is **after section b**:
sections a and b are the cheap, must-reproduce half, and c/d are the expensive new
half. Everything a/b produces is on the Hub before c starts.

---

## 4. What will be produced

**Backfill (a, b) — no new numbers, only new files.** 12 `.predictions.jsonl`
siblings:

```
C_bert-base-multilingual-cased_ax_to_grind_test_seed{42,123,2026}.predictions.jsonl
D_xlm-roberta-base_ax_to_grind_test_seed{42,123,2026}.predictions.jsonl
F_bert-base-multilingual-cased_notri_fact_holdout_seed{42,123,2026}.predictions.jsonl
F_xlm-roberta-base_notri_fact_holdout_seed{42,123,2026}.predictions.jsonl
```

The matching `.json` metrics files are **rewritten with identical numbers**. The
verify cells prove that by walking every numeric leaf against the copy in git and
stopping the notebook on any difference. Expect `git status` afterwards to show
those `.json` files as modified with only `timestamp_utc`, `git_commit` and the
`hardware` block changed.

**New results (c, d) — 15 files plus predictions:**

```
Q_xlm-roberta-base_notri_fact_holdout_seed{42,123,2026}.json
I_xlm-roberta-base_ax_to_grind_test_seed{42,123,2026}_cap{25,50,100,200}.json
```

**New checkpoint branches, all inside the existing `urdu-misinfo-xlmr-staging`
repo** — no new repos:

```
seed-{42,123,2026}-punct-ablation                      (3)
seed-{42,123,2026}-length-cap-{25,50,100,200}          (12)
```

Everything is pushed to `<HF_STAGING_PREFIX>/urdu-misinfo-results-staging` under
`milestone5/metrics/` **as each run completes**, not batched (M4-6). A disconnect
therefore costs only the run in flight.

---

## 5. The two real risks — read this before starting

Neither is quota. Both can kill the run late, which is the expensive way to fail.

### 5.1 Local disk — handled automatically, but know what the signal looks like

Fifteen XLM-R checkpoints at ~1.1 GB would be **~16.5 GB** against Kaggle's
**~19.5 GB** working quota, which would have run section d out of disk part-way.

**This is now automatic and needs nothing from you.** Each training run deletes its
own local checkpoint directory as soon as the push to the Hub has been
**verified** — `push_verify_prune` asks the Hub to list the files at the pushed
revision and requires config **and** weights **and** tokenizer before removing
anything, using the same completeness definition
`inventory_staging_checkpoints.py` audits with. Peak local usage is therefore about
one checkpoint, not fifteen.

**The precondition is the point.** A disk fix that deleted a checkpoint whose
upload had not actually landed would recreate M4-6 — a completed training run whose
only artefact is gone — and would do it while looking like housekeeping. So an
attempted, skipped, dry-run, partially-uploaded or merely unverifiable push all
leave the local copy alone, and the run prints:

```
local checkpoint KEPT: checkpoint upload NOT verified on the Hub — ...
```

**If you see that line, do not ignore it.** It means that run's model exists only
in the session's working directory, and the session is ephemeral. Either re-push it
or download it before the session ends.

Section e lists whatever survived. Anything in that list is a signal, not clutter —
read the reason before deleting it by hand, and if it never reached the Hub, do not
delete it at all.

> Sections a and b need no equivalent: they *download* checkpoints (into the shared
> HF cache, which section b reuses) and run the Trainer inside a
> `tempfile.TemporaryDirectory`, so they write no persistent local checkpoint state
> and have nothing to prune. That is asserted by test rather than assumed, so if
> either ever starts pushing checkpoints the test fails and the prune gets wired in.

### 5.2 Hugging Face storage for a *private* repo

This session pushes **15 new branches × ~1.1 GB ≈ 16.5 GB** into
`urdu-misinfo-xlmr-staging`, which is **private** (M4-7). Free-tier private storage
is finite. **Check the repo's storage before you start.**

If it is a problem, the honest ranking is: **I's 12 checkpoints are the most
disposable.** They are fully reproducible from config + seed + cap, they are not
inputs to any later experiment, and I's result is the metrics, not the model. Q's 3
are more worth keeping (Q is a candidate for the explainability work). To skip I's
checkpoint pushes, add `--no-push` to the section-d command — metrics and
predictions still get pushed; only the model weights are skipped.

That is your call, not one to make silently: `REPRODUCIBILITY.md` §6 wants results
traceable to a checkpoint, so skipping the push means I's files record a branch
that does not exist. If you go that way, tell me and I will make the metadata say
so rather than leave a dangling reference.

---

## 6. Staying attached

Kaggle's **20-minute idle timeout** is the constraint that actually bites here. The
longest unattended stretch is section d at ~25–50 minutes, which is fine — the
kernel is busy, not idle. The risk is the gaps: between finishing b and typing
`CONFIRM_Q`, and between c and d.

Set the two confirm cells up **before** you start, then it is one uninterrupted
pass.

---

## 7. Why section d runs `--models transformer`

I's classical half (B/tfidf_svm, all four caps) is **already committed**, computed
locally on the pinned CPython 3.12.11 with the exact package pins — and verified by
reproducing the committed uncapped B baseline bit-identically before any new number
was written. Re-running it on Kaggle would overwrite four committed results with
identical numbers, and would do so from a *different* environment for no benefit.

Committed classical results, for comparison against what section d produces:

| cap | B macro-F1 | delta vs uncapped B (0.8835) |
|---|---|---|
| 25 | 0.8551 | −0.0284 |
| 50 | 0.8675 | −0.0160 |
| 100 | 0.8726 | −0.0109 |
| 200 | 0.8770 | −0.0065 |

---

## 8. What to bring back

1. **The three tables from section f** — Q's delta against F, I's cap curve
   (XLM-R beside the committed classical numbers), and the M5-2 backfill
   completeness check, which should end with **"M5-2 is CLOSED"**.
2. **The `git status` / `git diff --stat` output**, so the backfill's rewritten
   `.json` files can be checked before committing.
3. **The zip from the Output tab** — a convenience copy. The Hub is the copy.
4. **Anything that looked wrong**: a seed far from its two siblings, an OOM, a
   disk-full error, a failed push.

Do **not** commit anything until the two verify cells have passed. If either
raised, bring back its output instead — a changed aggregate in a backfill is a
real finding and needs investigating before anything lands.

---

## 9. Reading the results

**Q — is F's collapse about punctuation?** Q trains on punctuation-stripped
Ax-to-Grind, so its training surface matches Notri-Fact's (which ships with zero
non-alphanumeric characters — M2-3). Compare Q's macro-F1 against F's same-seed
XLM-R number, already committed at **0.3393 ± 0.0031** with >99% dominant-class
share.

- **Delta near zero** → the surface mismatch was *not* what drove the collapse, and
  F's result stands as a domain/length finding. This is the more likely outcome and
  is a perfectly good result: it *eliminates* a confound rather than finding one.
- **Large positive delta** → punctuation was doing real work, and F's headline
  number needs restating with that caveat.

Either way it is reported as its own finding (M2-3 option (c)), not folded silently
into F.

**I — how much of in-domain performance is length?** Read the cap curve against
uncapped D (**0.9269 ± 0.0042**). Cap 50 is the **replication** point: Haroon (2026)
reports a **0.0067** macro-F1 drop there. That is a *literature* value, recorded in
each cap-50 file under `compares_against.reference_finding` with
`is_this_projects_measurement: false` — do not report it as ours.

Watch for the classical/transformer comparison: B lost 0.0160 at cap 50. If XLM-R
loses substantially more, the transformer was leaning on length harder than the
classical baseline — which would be a genuine contribution, since Part 1's evidence
table has only Haroon's single-model number.

Also note **the cap is not neutral preprocessing**: at cap 50 it truncates **35.26%
of fake training rows against 13.18% of real ones**, and mean words fall 120.75 →
30.2 for fake but only 33.48 → 32.08 for real. The ablation removes far more from
one class than the other, which is the length signal itself. Recorded per run under
`ablation.train_profile.by_label`, and it must be stated in the write-up rather than
discovered by a reader.

**Both Q and I remain within a 512-subword window** (M4-1), and I's caps are far
below it, so for I the word cap — not the subword ceiling — is the binding limit.
