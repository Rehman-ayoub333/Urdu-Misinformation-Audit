# Master Project Blueprint
## Urdu Misinformation Detection: Cross-Dataset Generalization & Shortcut Analysis

**Prepared for:** Rehman Ayoub
**Date:** August 12, 2026
**Status:** Phase 2 — Research & Engineering Design (no implementation yet)

This document is the single source of truth for the project until superseded by an explicit, logged decision (see Part 37). It assumes Phase 0/1 (`RESEARCH_VALIDATION_REPORT.md`) as prior context and does not repeat that material except where a claim needed re-verification.

---

## PART 1 — Re-Validation of Load-Bearing Claims

The Phase 0/1 report rested on a small number of claims that, if wrong, would invalidate the whole pivot. Those were re-checked today against primary sources (direct fetch of the arXiv abstract page and cross-referencing the Nature paper across five independent sources: Nature, PMC, Heriot-Watt's research portal, TechXplore, and ResearchGate), not re-run as a generic search.

| Claim | Primary source | Evidence | Confidence | Effect on project |
|---|---|---|---|---|
| A July 2026 paper is the first systematic cross-dataset generalization study for Urdu FND, using Ax-to-Grind (10,083 articles) and Notri-Fact (13,388 articles) | arXiv:2607.14131, Haroon, submitted 7 Jul 2026, cs.CL, CC-BY 4.0 — fetched directly | Abstract confirms: Notri-Fact→Ax-to-Grind transfer F1 0.771; Ax-to-Grind→Notri-Fact transfer **collapses to F1 0.005**, predicting "fake" for 99.7% of test articles; root cause is a length confound (fake avg. 117 words vs. real avg. 35 words, 3.4x); **a length-ablation experiment capping articles at 50 words yields only a 0.0067 F1 drop**, which the author uses to argue the confound *inflates* but does not *solely drive* in-domain performance | **High** — verified directly from the primary source, not a secondary summary | This is the single most important empirical anchor for the whole pivot. It is real, recent, and single-authored (worth noting for your own literature framing: it's a preprint, not yet peer-reviewed, so cite it as such) |
| A February 2026 Scientific Reports paper reports 96.2% accuracy / F1 0.956 on a new 14,178-article, 15-domain Urdu dataset (UFND) using XLM-RoBERTa + concatenated GloVe | Nature (s41598-026-36771-0), cross-checked against PMC (PMC12923771), Heriot-Watt Research Portal, TechXplore (Mar 2026), ResearchGate | Confirmed: authors Feroz, Abbasi, Babar, Aljohani (Superior University, University of Lahore, Heriot-Watt, Taibah University); dataset spans 2017–2023, 15 domains including politics/religion; XLM-R+GloVe: F1 0.956, accuracy 0.962, precision 0.932, recall 0.940 | **High** — corroborated across 5 independent sources | Confirms the "field already hit ~96%" claim is not overstated. Reinforces that a plain classifier proposal would be redundant |
| UFND dataset is publicly/openly accessible | Same sources as above | Co-author Dr. Waseem Abbasi is quoted stating "We've made the dataset open access so that we can continually improve its performance" — but **no direct GitHub/Kaggle/Zenodo download link was found** in this pass | **Medium** — access is claimed by the authors but not independently confirmed | Do not commit engineering time to UFND as primary until you've located and tested the actual download link via the paper's Data Availability Statement. Treat as an optional upgrade path, not a dependency |
| Ax-to-Grind Urdu is 10,083 articles, 15 domains, expert-journalist annotated, publicly hosted on GitHub with usable CSV files | arXiv:2403.14037 abstract + GitHub repo (github.com/Sheetal83/Ax-to-Grind-Urdu-Dataset), corroborated via ResearchGate PDF and a mirror repo (HjH-Whu-CRC/Ax-to-Grind-Urdu) | Confirmed: `Fake News.csv` and `True News.csv` exist and are fetchable directly from the repo; no explicit OSS license file was located — treat as "available for research use with citation," not redistribution-cleared | **High** | Confirms primary-dataset choice is sound; license ambiguity means: cite it, don't redistribute the raw CSVs in your own repo — link to source instead (see Part 28) |
| An additional, previously unseen related paper exists: "Detection of Human and Machine-Authored Fake News in Urdu" | arXiv:2410.19517 (surfaced during this verification pass) | Title and existence confirmed via search; full content not read in this pass | **Medium** — existence confirmed, content not yet reviewed | Flag as a literature-review item to read before finalizing your Related Work chapter; it may be relevant to a future-work note on LLM-generated Urdu misinformation, not core to the current experiment design |
| Notri-Fact Urdu has no accompanying methodology paper | Kaggle page (tridata/notri-fact-real-and-unreal-urdu-news), fetched directly | Page confirms dataset exists and is live on Kaggle; no linked paper or methodology description found on the page itself | **Medium-High** | Its low provenance is real, not an artifact of insufficient searching — use it only as a diagnostic/stress-test set (as already planned), and say so explicitly in the thesis rather than treating it as equally authoritative to Ax-to-Grind |

**Nothing in Phase 0/1 needed to be retracted.** The one adjustment: the length-ablation detail (capping at 50 words → only 0.0067 F1 drop) was not in the original summary and is important — it means the length confound *inflates* Ax-to-Grind's in-domain numbers and *fully explains* the cross-dataset collapse, but the model isn't purely a length detector even in-domain. This nuance should be stated precisely in your own methodology; overclaiming "the dataset is only measuring length" would itself be a defensible-in-a-viva mistake.

---

## PART 2 — Final Research Gap Statement

Not "nobody has studied Urdu fake news" — that's false, as Part 1 of the prior report showed extensively. The defensible, evidence-backed gap:

> **Existing Urdu misinformation detection research reports high in-domain accuracy (92–98%) but has not systematically evaluated — across multiple model families and multiple datasets — whether that performance reflects genuine misinformation-relevant signal or dataset-specific shortcuts such as article length, and there is no publicly deployed, evaluation-transparent Urdu misinformation tool that communicates this uncertainty to end users.** A single 2026 preprint (Haroon) has demonstrated the length-confound effect for one model (XLM-R) on one dataset pair (Ax-to-Grind ↔ Notri-Fact). It has not been (a) replicated across a classical-ML baseline and a second transformer (mBERT), (b) extended with an explainability layer that checks whether "important" tokens correlate with length/formatting artifacts rather than content, or (c) turned into a deployed system that surfaces this uncertainty rather than a bare accuracy number.

What this gap is **not**: it is not "build a better classifier that beats 96.2%." Chasing a higher number on a benchmark shown to be confound-inflated would be scientifically counterproductive and is explicitly excluded from this project's goals.

---

## PART 3 — Final Research Title

| # | Candidate title | Assessment |
|---|---|---|
| 1 | "Urdu Fake News Detection Using XLM-RoBERTa" | Original idea. Technically accurate but describes the *saturated* formulation. Rejected. |
| 2 | "A Robust AI System for Detecting Fake News in Urdu" | Marketing-toned ("robust," "AI system"), overpromises certainty the project explicitly avoids claiming. Rejected. |
| 3 | "Cross-Dataset Generalization and Shortcut Learning in Urdu Misinformation Detection" | Precise, matches the actual experiments (Direction A + F), but omits the explainability/deployment contribution. Strong candidate. |
| 4 | "Beyond In-Domain Accuracy: Auditing Shortcut Learning and Cross-Dataset Generalization in Urdu Misinformation Detection" | Same substance as #3, framed as an audit/critique, which matches the actual contribution better and reads well as a thesis title. Strongest candidate. |
| 5 | "Towards Trustworthy Urdu Misinformation Detection: A Cross-Dataset, Explainability-Aware Evaluation" | Broader, includes explainability and the "trustworthy AI" framing, but "trustworthy" risks sounding aspirational/marketing if the thesis can't fully deliver on it. |

**Recommended final title:**

> **"Beyond In-Domain Accuracy: Auditing Shortcut Learning and Cross-Dataset Generalization in Urdu Misinformation Detection"**

It is specific, names the actual method (auditing, shortcut learning, cross-dataset), is not overclaiming, and reads as a legitimate NLP-venue paper title, not a product tagline. Use "Urdu Misinformation Detection: Cross-Dataset Generalization & Shortcut Analysis" as the informal/GitHub-facing short name (already used as this document's subtitle).

---

## PART 4 — Research Questions

### RQ1 — In-domain baseline performance
**Question:** How do classical ML (TF-IDF+LR/SVM) and multilingual transformers (mBERT, XLM-R) perform on Urdu misinformation detection under standard in-domain evaluation?
**Motivation:** Establishes a faithful replication baseline before any generalization claims are made; needed to confirm your pipeline reproduces literature-reported ranges before trusting downstream experiments.
**Hypothesis:** Transformers will roughly match literature-reported ranges (F1 0.90–0.96 on Ax-to-Grind); classical baselines will be competitive but somewhat lower, consistent with FIRE2021 and PeerJ findings.
**Dataset:** Ax-to-Grind (primary), in-domain train/val/test split.
**Variables:** Model family (independent); Macro-F1, accuracy, precision/recall (dependent).
**Experiment:** Train/evaluate all four models under identical preprocessing and splits.
**Metrics:** Macro-F1 (primary), accuracy, per-class precision/recall, confusion matrix.
**Expected outcome:** Transformers ≥ classical baselines by a modest margin; all models score "high" in-domain.
**Supports hypothesis if:** Results land within ~5 points of literature ranges under a matched setup.
**Rejects/complicates hypothesis if:** A large, unexplained gap from literature appears — triggers a pipeline audit before proceeding to RQ2.

### RQ2 — Cross-dataset generalization
**Question:** How well do models trained on one Urdu misinformation dataset generalize to a distinct, unseen dataset?
**Motivation:** Directly extends Haroon (2026), which tested only XLM-R; this project adds classical ML and mBERT to check whether the collapse is model-specific or systemic.
**Hypothesis:** All models trained on Ax-to-Grind will show a substantial F1 drop when evaluated zero-shot on Notri-Fact, and vice versa in the opposite (milder) direction, replicating the asymmetry Haroon found for XLM-R.
**Dataset:** Train on Ax-to-Grind → test on Notri-Fact (zero-shot); train on Notri-Fact → test on Ax-to-Grind (zero-shot).
**Variables:** Model family, transfer direction (independent); cross-dataset Macro-F1, prediction-class distribution (dependent).
**Experiment:** No fine-tuning/retraining on the target dataset — strict zero-shot transfer, matching Haroon's protocol for direct comparability.
**Metrics:** Macro-F1, per-class recall, prediction-collapse indicator (% predicted-fake).
**Expected outcome:** Ax-to-Grind→Notri-Fact collapses for all four models (not just XLM-R); Notri-Fact→Ax-to-Grind holds up better.
**Supports hypothesis if:** The asymmetry replicates across model families.
**Rejects hypothesis if:** Classical models (which cannot "see" the same length-driven embedding shortcuts as easily, or arguably even more easily via raw token counts) behave differently — itself an interesting, reportable finding either way.

### RQ3 — Shortcut/confound quantification
**Question:** To what extent does the article-length confound (and other candidate shortcuts: source-style artifacts, formatting, vocabulary) drive model predictions?
**Motivation:** Haroon quantified this for length alone on XLM-R; this project must independently replicate it and extend the shortcut inventory (Part 7's risk register) rather than assume length is the only artifact.
**Hypothesis:** Length is the dominant, but not sole, confound; a length-only baseline (e.g., logistic regression on word-count alone) will achieve surprisingly high F1 on Ax-to-Grind in-domain, well above chance, but below the full model's in-domain score.
**Dataset:** Ax-to-Grind (where the confound was found); Notri-Fact as a length-balanced control.
**Variables:** Feature set (length-only vs. full text) (independent); Macro-F1, calibration (dependent).
**Experiment:** Train a trivial length-only classifier; compare against full-text models; run the length-ablation replication (cap articles at a fixed word count, following Haroon's 50-word design) on Ax-to-Grind.
**Metrics:** Macro-F1 of length-only baseline vs. full model; F1 delta under length-capping.
**Expected outcome:** Length-only baseline scores well above the 50% chance rate but below the full model; length-capping reduces in-domain F1 only modestly (replicating Haroon's 0.0067 finding) — meaning length inflates but doesn't solely explain in-domain scores, while it fully explains the cross-dataset collapse.
**Supports hypothesis if:** Both replicate as above.
**Rejects hypothesis if:** Length-only baseline is near-chance (confound weaker than reported) or explains nearly all in-domain variance too (confound stronger/broader than reported) — either is a legitimate, reportable correction to the literature.

### RQ4 — Mitigation
**Question:** Can shortcut-aware mitigation (length balancing/stratification, not full dataset re-engineering) improve cross-dataset generalization without destroying in-domain performance?
**Motivation:** A diagnosis without a mitigation attempt is a weaker thesis contribution; even a negative result ("mitigation didn't fully fix it") is publishable and honest.
**Hypothesis:** Length-stratified training (balancing word-count distribution between classes in the training set) will partially recover cross-dataset F1 on the Ax-to-Grind→Notri-Fact direction, at a modest cost to in-domain F1.
**Dataset:** Ax-to-Grind, length-stratified resample.
**Variables:** Training-set construction (standard vs. length-stratified) (independent); in-domain F1, cross-dataset F1 (dependent).
**Experiment:** Retrain the best-performing model (likely XLM-R) on a length-stratified subset; re-run RQ2's transfer test.
**Metrics:** Macro-F1 (both settings), delta vs. RQ2 baseline.
**Expected outcome:** Partial, not full, recovery — a realistic, defensible finding.
**Supports hypothesis if:** Cross-dataset F1 improves measurably (e.g., >0.05 absolute) without in-domain F1 collapsing.
**Rejects hypothesis if:** No improvement, or in-domain performance collapses along with it — still reportable as "length-stratification alone is insufficient; the confound may be entangled with other features," a legitimate and specific finding for Discussion/Future Work.

### RQ5 — Explainability and shortcut visibility
**Question:** Do model explanations (SHAP/Integrated Gradients) reveal reliance on meaningful linguistic signals, or on artifacts like length/formatting/boilerplate tokens?
**Motivation:** Connects the quantitative shortcut-finding to token-level evidence, and gives the deployed demo a legitimate, defensible explanation feature instead of a black box.
**Hypothesis:** For Ax-to-Grind-trained models, explanations on misclassified or borderline cross-dataset examples will disproportionately highlight boilerplate/positional tokens (e.g., trailing sentences, repeated phrases) rather than content-bearing claims, especially for the confound-affected model.
**Dataset:** A stratified sample (see Part 14) of correctly- and incorrectly-classified articles from RQ1–RQ2.
**Variables:** Model (confound-affected vs. length-stratified) (independent); qualitative explanation patterns, token-attribution concentration (dependent).
**Experiment:** Run SHAP (or Integrated Gradients — final choice justified in Part 15) over a sampled set; manually categorize top-attributed tokens as content-relevant vs. artifact-like, with a simple inter/intra-rater consistency check since you are the sole annotator (documented limitation).
**Metrics:** Qualitative categorization rate (% of top-K tokens judged artifact-like vs. content-like); example-level case studies.
**Expected outcome:** A measurable, if imperfect, difference in explanation "cleanliness" between the confound-affected and mitigated models.
**Supports hypothesis if:** The pattern is visible and consistent across a reasonable sample (aim for n≥40 examples, documented in Part 14).
**Rejects hypothesis if:** No clear pattern emerges — still valuable: report as "explanation methods did not clearly surface the confound, suggesting token-attribution alone is an insufficient diagnostic," which is itself a finding about explainability limitations relevant to Part 15/16.

---

## PART 5 — Thesis Contributions

**Scientific contribution:** An independent replication of the Ax-to-Grind length confound across an additional model family (mBERT) and a classical baseline (not just XLM-R as in Haroon 2026), plus a mitigation attempt (length stratification) and its measured effect — extending a single-author, single-model preprint into a broader, multi-model empirical picture.

**Methodological contribution:** A reusable audit protocol — bidirectional cross-dataset transfer + length-only baseline + length-ablation + explanation-pattern categorization — packaged as scripts others can point at a new Urdu (or other low-resource-language) misinformation dataset pair to check for shortcut learning before trusting reported accuracy.

**Engineering contribution:** A working, deployed, Urdu-native misinformation *analysis* tool (not "detector," per the responsible-AI framing in Part 16) that surfaces model confidence and a shortcut-aware explanation, filling the verified gap that no such public demo currently exists.

**Dataset contribution:** None claimed. No new dataset is being created, annotated, or released as a primary contribution — the project uses existing public datasets (Ax-to-Grind, Notri-Fact) as-is. If a length-stratified resample is constructed for RQ4, that resampling script/config is released for reproducibility, but this is explicitly a derived artifact, not a new dataset contribution, and should be labeled as such everywhere it's mentioned.

**Practical contribution:** Primarily useful to (a) other Urdu-NLP researchers who want a shortcut-check before trusting a new dataset, and (b) as a triage aid for resource-constrained Urdu fact-checkers (Soch Fact Check, Geo Fact Check) — framed honestly as a first-pass filter, not a verdict engine, per the documented English-vs-Urdu fact-checking gap in Phase 0.

What this thesis is **not**: it is not a new SOTA claim (explicitly reject any framing built around beating 96.2%), not a new dataset paper, and not a claim that the system can determine ground truth. It is diagnostic/methodological research plus a responsibly-scoped engineering artifact.

---

## PART 6 — Final Dataset Strategy

| Dataset | Size | Real/Fake | Domains | Script | Annotation | License/Access | Duplicates | Leakage risk | Length imbalance | Source imbalance | MT artifacts | Research usage | Reproducibility | Suitability |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ax-to-Grind Urdu | 10,083 | ~50/50 | 15 | Urdu (Nastaliq) | Expert journalists, κ=0.94 | Public GitHub, cite-only (no explicit OSS license) | Not independently checked yet — planned in Part 7 | Source style not flagged as an issue in literature | **Severe** (117 vs 35 words, 3.4x) — confirmed via primary source | Not confirmed, to be audited | None reported | High (used in 3+ papers surveyed) | High — public CSVs, stable repo | **Primary training set** |
| Notri-Fact Urdu | 13,388 | ~50/50 | Undocumented | Urdu (Nastaliq) | Undocumented (no paper) | Kaggle standard terms | Not independently checked | Unknown provenance is itself a risk, but length-balanced | Balanced (per Haroon's use as control) | Not confirmed | Not confirmed | Used in Haroon (2026) as cross-dataset test set | Medium — Kaggle-hosted, no paper trail | **Cross-dataset test set** |
| UFND (Feb 2026) | 14,178 | ~58/42 | 15 (incl. politics/religion) | Urdu (Nastaliq) | Manually curated | Claimed open access, **link unconfirmed** | Unknown | Unknown | Unknown | Unknown | Not reported | Newest, from a corroborated Scientific Reports paper | Unconfirmed pending located download link | **Optional upgrade primary** — pursue only after confirming the link works |
| Bend the Truth / FIRE2020–21 | 900–1,600 | Similar split | 5 | Urdu (Nastaliq) | Real verified; fake written-to-deceive by journalists (synthetic) | Public GitHub, citation required | Small enough to check exhaustively if used | Synthetic-fake vs organic-real is itself a construct-validity concern, flagged explicitly | Not the focus of the confound literature | Not confirmed | None (native Urdu, no MT) | Field standard benchmark | High | **Optional — literature comparability only** |
| Farooq et al. 2023 (PeerJ) | 4,097 | ~60/40 | 9 | Urdu (Nastaliq) | Partially MT'd from English, not fully re-verified | Zenodo, CC-BY 4.0 | Not checked | MT artifacts flagged by later papers | Not confirmed | Not confirmed | **Yes, partial** | Moderate | High (Zenodo DOI) | **Not selected** — MT artifacts + imbalance make it a weaker fit for a shortcut-analysis-focused thesis |
| Urdu News 1M (Kaggle/Mendeley) | 1M+ | N/A — unlabeled | 4 categories | Urdu (Nastaliq) | N/A | Public | N/A | N/A | N/A | N/A | N/A | Used for domain-adaptive pretraining in one 2025 paper | High | **Optional unlabeled corpus** — only if RQ4/5 timeline allows a domain-adaptive-pretraining stretch experiment |

**Final assignment:**
- **PRIMARY TRAINING DATASET:** Ax-to-Grind Urdu — largest well-documented, expert-annotated, non-MT dataset with a directly-relevant, verified confound to investigate.
- **CROSS-DATASET TEST DATASET:** Notri-Fact Urdu — the exact pairing Haroon used, enabling direct comparison of your replicated numbers against a published reference point.
- **SECONDARY DATASET:** UFND (Feb 2026), conditional — pursue once its download link is confirmed; if inaccessible by the dataset-acquisition milestone (Part 36, Phase 3), drop it without blocking the timeline.
- **OPTIONAL BENCHMARK DATASET:** Bend the Truth / FIRE2020–21, for a literature-comparability table only (not part of the core cross-dataset experiment, since its synthetic-fake construction is a different labeling paradigm that would confound the confound-analysis itself).
- **OPTIONAL UNLABELED CORPUS:** Urdu News 1M, stretch-only, for domain-adaptive pretraining if time permits after RQ1–RQ5 core results are in.

**Explicitly not combining datasets for training** — concatenating Ax-to-Grind and Notri-Fact would eliminate the ability to run the core RQ2/RQ3 cross-dataset experiments, which depend on the datasets remaining distinct train/test pools.

---

## PART 7 — Dataset Audit & Shortcut/Bias Risk Register

### Dataset Quality Report — required measurements (per dataset, before any training)

1. **Size & class distribution:** row counts, class balance, per-domain counts (for the 15-domain datasets).
2. **Length distribution:** character count, word count, and tokenizer-subword count (using the actual XLM-R tokenizer, since subword count is what the model "sees"), each broken down by label — this is the direct, primary check for the known confound.
3. **Vocabulary analysis:** top-N word frequencies per class, top character 3-/4-grams per class (per literature, character n-grams are informative for Urdu), and a simple log-odds/PMI ranking of words most associated with each label — a fast, classical way to surface source-style or template artifacts.
4. **Source distribution:** publisher/domain field (if present) cross-tabulated against label — directly checks the Hook-and-Bait-style source-leakage pattern flagged in Phase 0/1, even though Hook and Bait itself isn't the primary dataset.
5. **Structural artifacts:** presence of boilerplate strings (bylines, "read more," fixed disclaimers), heading/formatting patterns, punctuation-density differences by class.
6. **Language mixing:** rough proportion of Latin-script tokens (code-mixed English) per article, by class.
7. **Duplication:** exact-duplicate detection (hash-based) and near-duplicate detection (e.g., MinHash/simhash or TF-IDF cosine similarity above a threshold) within each dataset and across the Ax-to-Grind/Notri-Fact pair (cross-dataset duplication would silently invalidate the "zero-shot" claim in RQ2 if any articles overlap).
8. **Leakage checklist:** are there any temporal fields, and if so, does the intended train/val/test split cross a suspicious pattern (e.g., all "fake" from one time window)? Is there any author/byline field that trivially predicts label?

### Dataset Shortcut / Bias Risk Register

| Risk | Evidence | Severity | Detection method | Mitigation | Residual risk |
|---|---|---|---|---|---|
| Article-length confound (Ax-to-Grind) | Directly confirmed, primary source (Haroon 2026) | **Critical** | Length-by-label histogram + length-only baseline (RQ3) | Length-stratified resampling (RQ4); always report length-bucketed performance (Part 14) | High even after mitigation — this is the central subject of the thesis, not a side note, so "residual risk" is expected and should be reported, not hidden |
| Source-style leakage | Not confirmed for Ax-to-Grind specifically (was flagged for Hook and Bait, which isn't primary) | Medium — plausible, unverified | Source-field cross-tab (if metadata exists) + top-vocabulary-per-class check | If confirmed, report explicitly; consider a source-blind text-only ablation | Unknown until audited — flagged as a required Phase-4 task, not assumed away |
| Exact/near duplicates within a dataset | Not yet checked | Medium | Hash + near-dup similarity scan | De-duplicate before splitting | Low after mitigation, if caught early |
| Cross-dataset duplication (Ax-to-Grind articles appearing in Notri-Fact or vice versa) | Not yet checked; would be a serious validity problem for RQ2 if present | **High if present, currently unknown** | Cross-dataset near-duplicate scan before running any transfer experiment | Remove overlapping articles from the test set if found | Must be resolved to zero before RQ2 results can be trusted — this is a go/no-go gate, not optional cleanup |
| Synthetic/non-organic fake class (Bend the Truth only) | Confirmed by design (journalist-written-to-deceive) | Medium, scoped only to the optional benchmark dataset | N/A — known by construction | Use only for literature comparability, never for the core confound analysis | Documented limitation if the optional dataset is used at all |
| Machine-translation artifacts | Not applicable to Ax-to-Grind/Notri-Fact (no MT reported); applies to the excluded Farooq et al. dataset | Low (not using that dataset) | N/A | Dataset excluded from primary/secondary/cross-dataset roles | None |
| Undocumented Notri-Fact provenance | Confirmed — no paper found | Medium | N/A — inherent to the dataset | Use strictly as a diagnostic/stress-test set, state its provenance limitation explicitly wherever cited | Cannot be fully resolved; must be disclosed, not hidden |
| Boilerplate/template tokens (bylines, fixed phrases) | Not yet checked | Medium | Structural-artifact scan (item 5 above) | Consider whether to strip obvious boilerplate in the "clean" tier (Part 8) — decide based on evidence, not by default | Depends on audit outcome |

---

## PART 8 — Data Preprocessing Strategy

Three immutable tiers, never overwritten in place:

**Tier 1 — Raw:** exact as-downloaded files (`Fake News.csv`, `True News.csv` for Ax-to-Grind; original Kaggle export for Notri-Fact), stored read-only, hashed (SHA256 manifest) so any downstream corruption is detectable.

**Tier 2 — Clean:** Unicode normalization (NFC, plus Urdu/Arabic character-variant normalization — e.g., normalizing Arabic yeh ي vs. Urdu yeh ی, kaf ك vs. keheh ک, which is a well-documented Urdu-text inconsistency), whitespace collapsing, HTML entity/tag stripping (news scrapes often carry residual markup), URL and emoji normalization (replace with a placeholder token rather than deleting, so their presence/absence remains a feature the model can use if genuinely informative — deleting outright risks removing signal, per the brief's explicit warning against over-cleaning). Deduplication (exact + near-duplicate removal, per Part 7) happens at this tier, before any splitting.

**Tier 3 — Model-ready:** tokenized (XLM-R SentencePiece / mBERT WordPiece / TF-IDF vocabulary, produced separately per model), truncated/padded to a max sequence length decided empirically from the length audit (Part 7) — likely 256–512 subword tokens given the length distributions involved, finalized once the audit numbers are in rather than guessed upfront.

**Explicit non-decisions, pending audit evidence:** whether to strip boilerplate/bylines is deferred until Part 7's structural-artifact scan shows whether they're a meaningful shortcut; whether to cap article length as a *default* preprocessing step (vs. only as the RQ4 mitigation experiment) is deferred, because doing it by default would silently pre-empt RQ3's measurement of the undisturbed confound. **The undisturbed, only-Tier-2-cleaned version must exist and be used for RQ1–RQ3; length-capping is applied only as an explicit, separately-tracked experimental condition for RQ4.**

---

## PART 9 — Data Splitting Strategy

- **Within-dataset splits (Ax-to-Grind):** stratified 70/15/15 train/val/test by label, with deduplication applied *before* splitting (Part 8) so no near-duplicate pair straddles train and test. If a source/publisher field survives the audit, consider grouped splitting (all articles from one outlet in only one split) to prevent source leakage — decide based on Part 7's source-distribution findings, not by default.
- **No temporal field was confirmed as usable for exact-date stratification** in this pass; if the audit finds one, a temporal holdout (train on earlier years, test on later) becomes a strongly-recommended additional robustness check (Part 12), since it directly tests generalization to "emerging narratives," a limitation the Feb 2026 paper's own authors flagged.
- **Cross-dataset test (Notri-Fact):** used in full, held out entirely — no portion of Notri-Fact is used for training or hyperparameter tuning at any point, to keep RQ2's zero-shot claim valid.
- **Cross-dataset de-contamination check:** before finalizing any split, run the near-duplicate scan (Part 7) between Ax-to-Grind and Notri-Fact; if overlap is found, remove overlapping items from Notri-Fact (the test side), not from Ax-to-Grind (the training side), to preserve training-set size.
- **Article-level leakage:** since both datasets are article-level (no known multi-row-per-article structure), the main leakage vector is duplication, already covered above, rather than fragment-level leakage.

---

## PART 10 — Model Strategy

| Model | Architecture | Urdu capability | Params | Memory (fine-tune, T4) | Training cost | Inference cost | Research relevance | Colab feasibility |
|---|---|---|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | Linear, bag-of-words/n-grams | Adequate for surface patterns; can trivially learn length via feature count | N/A | Trivial (CPU) | Minutes | Milliseconds | High — the FIRE2021 result where classical beat some transformers makes this a mandatory, not optional, baseline; also the fastest way to smoke-test the length-only shortcut (RQ3) | Excellent |
| TF-IDF + SVM | Linear/kernel | Same as above | N/A | Trivial (CPU) | Minutes | Milliseconds | Second classical baseline, consistent with nearly every paper surveyed | Excellent |
| mBERT | 12-layer transformer, WordPiece | Covers Urdu among 104 languages; known higher subword fertility on non-Latin scripts | ~178M | Comfortable | ~30–90 min/epoch depending on dataset size | Fast (<100ms/article on GPU) | Second transformer, used across nearly every paper for comparison | Comfortable |
| XLM-RoBERTa-base | 12-layer transformer, SentencePiece BPE | Stronger multilingual pretraining than mBERT, the model every recent Urdu-FND paper converges on, including Haroon (2026) and the Feb 2026 SOTA paper | ~270M | Comfortable | ~45–120 min/epoch | Fast (<150ms/article on GPU) | Highest — direct comparability to the paper you're extending | Comfortable |

**PRIMARY MODEL:** XLM-RoBERTa-base — matches the literature you're directly extending (Haroon 2026), keeps results comparable.
**BASELINE MODELS:** TF-IDF+LR and TF-IDF+SVM — mandatory per the FIRE2021 precedent and useful diagnostically for RQ3.
**SECONDARY MODEL:** mBERT — the second model needed to test whether the shortcut/generalization pattern is XLM-R-specific or systemic (this is *the* novel extension over Haroon 2026, who tested XLM-R alone).
**OPTIONAL MODEL:** None added beyond these four. XLM-R-large, XLNet, or LLaMA-2-scale fine-tuning (as in Hook and Bait) are explicitly rejected — they would consume disproportionate Colab budget for a question (generalization/shortcuts) that doesn't require a bigger model to answer, and adding them "to look more impressive" is exactly the kind of scope inflation this blueprint exists to prevent.

---

## PART 11 — Complete Experiment Matrix

| ID | Experiment | Models | Data | Purpose |
|---|---|---|---|---|
| A | TF-IDF + Logistic Regression, in-domain | LR | Ax-to-Grind train/test | RQ1 baseline |
| B | TF-IDF + SVM, in-domain | SVM | Ax-to-Grind train/test | RQ1 baseline |
| C | mBERT, in-domain | mBERT | Ax-to-Grind train/test | RQ1 |
| D | XLM-RoBERTa, in-domain | XLM-R | Ax-to-Grind train/test | RQ1 (primary reference point) |
| E | All four models, in-domain on Notri-Fact | LR/SVM/mBERT/XLM-R | Notri-Fact train/test | RQ1 replication on second dataset, needed as the in-domain reference before interpreting E's cross-dataset counterpart |
| F | Cross-dataset: train Ax-to-Grind → test Notri-Fact (zero-shot) | All four | — | RQ2 |
| G | Cross-dataset: train Notri-Fact → test Ax-to-Grind (zero-shot) | All four | — | RQ2 |
| H | Length-only baseline (word count as sole feature) | LR | Ax-to-Grind | RQ3 |
| I | Length-ablation (cap at fixed word count, replicate Haroon's 50-word design) | XLM-R (+ best classical) | Ax-to-Grind | RQ3 |
| J | Length-bucketed performance breakdown | Best model(s) from A–D | Ax-to-Grind, Notri-Fact | Feeds Part 14 error analysis and RQ3 |
| K | Source/vocabulary shortcut scan (log-odds word association, source cross-tab if available) | N/A (data analysis, not a trained model) | Both datasets | RQ3, Part 7 |
| L | Mitigation: length-stratified retraining | XLM-R | Ax-to-Grind (stratified resample) | RQ4 |
| M | Re-run F/G with the mitigated model | XLM-R (mitigated) | — | RQ4 |
| N | Explainability: SHAP/IG over sampled correct + incorrect predictions | XLM-R (confound and mitigated versions) | Sampled subset (Part 14) | RQ5 |
| O | Error analysis categorization | All models' predictions | Sampled subset | Part 14 |
| P | Optional: domain-adaptive pretraining before fine-tuning | XLM-R | Urdu News 1M (unlabeled) + Ax-to-Grind | Stretch only |

---

## PART 12 — Experiment Priority

**REQUIRED (thesis cannot be defended without these):** A, B, C, D, F, G, H, I, N, O — this is the direct, minimal experimental backbone for RQ1–RQ3 and RQ5, plus the error-analysis and explainability layer needed to make the diagnosis credible rather than purely statistical.

**STRONGLY RECOMMENDED (materially strengthens the thesis, should be attempted before considering the core "done"):** E, J, K, L, M — the second in-domain reference point, length-bucketed breakdown, the broader shortcut scan, and the mitigation experiment (RQ4). These are what separates "we found a problem" from "we characterized and partly addressed a problem."

**STRETCH (only if the required + recommended set is fully complete with time/compute to spare):** P (domain-adaptive pretraining), an additional temporal-holdout split if metadata allows (Part 9), extending the shortcut scan to the optional UFND dataset if its access is confirmed.

**FUTURE WORK (explicitly not implemented in this thesis):** evidence-retrieval/fact-verification (Direction E from Phase 0/1), Roman Urdu / social-media-register modeling (no dataset exists — building one is a thesis-scale project on its own), multimodal (image/audio) misinformation, multi-class (beyond binary) veracity labeling, and any attempt to chase a new SOTA number.

---

## PART 13 — Evaluation Framework

**Primary metric: Macro-F1.** Justification: the datasets are near-balanced but not perfectly so (Ax-to-Grind ~50/50, some UFND-family datasets ~58/42), and accuracy alone would obscure the exact failure mode central to this thesis — prediction collapse to a single class (as in Haroon's 99.7%-predicted-fake finding). Macro-F1 penalizes exactly that failure mode, unlike accuracy, which can look deceptively reasonable under class imbalance combined with collapse. Accuracy is still reported as a secondary, literature-comparability metric since every surveyed paper reports it.

**Full metric set:** accuracy, macro-F1, weighted-F1, per-class precision/recall, confusion matrix (every experiment); ROC-AUC and PR-AUC for the in-domain experiments only (less meaningful once a model has collapsed to near-constant predictions, as expected in some cross-dataset runs, so reported but not over-interpreted there); a basic calibration check (reliability diagram / expected calibration error) for the primary model, directly supporting the "model confidence ≠ factual truth" product principle (Part 16) — if the model is poorly calibrated, that itself is a reportable, responsible-AI-relevant finding.

**Statistical rigor:** run each transformer experiment (C, D, and their cross-dataset counterparts) across **3 random seeds**, report mean ± standard deviation — not a full bootstrap/significance-testing apparatus, which would be disproportionate for a single-institution student thesis with fixed, small-ish test sets, but enough to distinguish a real effect from training-run noise, which matters a great deal here since the central finding (F1 0.005 vs 0.771) is a huge, not marginal, effect and should be shown to be robust to seed variation. Classical baselines (A, B) are deterministic enough that a single run per configuration is acceptable, noted explicitly as a simplification.

---

## PART 14 — Error Analysis Framework

**Categories used (data-supported, not assumed):** short articles (bottom length decile), long articles (top length decile — directly relevant to the confound), politics/religion domain (flagged as an admitted weak spot by the Feb 2026 paper's own authors), health domain (documented real-world harm category from Phase 0), code-mixed articles (if the language-mixing audit in Part 7 finds a meaningful proportion), cross-dataset-only failures (correct in-domain, wrong cross-dataset — the most theoretically interesting bucket), and source-style outliers if Part 7's source audit surfaces a pattern. Categories like "satire" or "clickbait" are **not** used unless the dataset audit turns up a way to identify them (e.g., a domain/tag field) — inventing a category you can't actually sample against would be an unsupported claim.

**Sampling method:** stratified random sample of ~15–20 misclassified examples per category above, capped around 100–120 total examples for manual review — a size that's actually reviewable by one person (you) in the available time, logged with a fixed random seed for reproducibility. This same sample feeds the RQ5 explainability review (Part 4), so the two efforts aren't duplicated.

**Methodology:** for each sampled example, record: true label, predicted label, confidence, length bucket, domain (if available), and a short free-text note on the likely failure reason, categorized post-hoc into a small fixed taxonomy (e.g., "length-shortcut-consistent," "ambiguous/borderline content," "possible label-noise," "topic/domain shift," "other"). Because you are the sole reviewer, state this as a limitation explicitly in the thesis (no inter-annotator agreement is computable) rather than presenting the categorization as more rigorous than it is.

---

## PART 15 — Explainability

**Comparison:**

| Method | Basis | Cost | Limitations |
|---|---|---|---|
| Raw attention weights | Model-internal attention scores | Free (already computed) | Broadly criticized in interpretability literature as not reliably corresponding to feature importance — **explicitly excluded** per the brief's own instruction, and for good methodological reason |
| Integrated Gradients | Path-integral of gradients from a baseline (e.g., all-padding) input to the actual input | Moderate — a handful of forward/backward passes per example | Requires a sensible choice of baseline input, which is somewhat arbitrary for text; can be noisy on subword tokens |
| SHAP (via a model-agnostic or gradient-based Shapley approximation, e.g., Partition SHAP for text) | Game-theoretic Shapley value approximation over token coalitions | Higher — more forward passes than IG, feasible only on a sampled subset, not the full test set | Approximation quality depends on the sampling budget; slower per-example than IG |

**Selected primary method: Integrated Gradients**, run via `captum` (PyTorch-native, straightforward integration with a Hugging Face XLM-R model). Reasoning: better computational fit for a Colab-budget project (needed on a ~100-example sample per Part 14, not the full test set, so cost is manageable either way, but IG's lower per-example cost gives headroom for the seed-repetition and mitigated-vs-original model comparisons RQ5 requires), and it has a cleaner mathematical justification (a formal completeness axiom: attributions sum to the difference between the model's output on the input and on the baseline) that's easier to defend precisely in a viva than a sampling-based Shapley approximation. SHAP is kept as a documented alternative/future-work item, not implemented in the core thesis, to avoid the "two explainability methods that don't fully agree" complexity trap for a first attempt.

**Validation of explanations:** cross-check IG's top-attributed tokens against the length/vocabulary/structural artifacts identified in Part 7's audit (a token-level sanity check: do high-attribution tokens cluster near the end of long articles, consistent with a length/position confound, or are they distributed across content-bearing claims?) — this is the direct link between RQ3 (quantitative shortcut evidence) and RQ5 (token-level evidence), and is more defensible than presenting IG scores as inherently trustworthy on their own.

**User-facing communication (product):** explanations are shown as "these text features contributed most strongly to the model's prediction" with the top-attributed spans highlighted — never phrased as "this proves the article is fake," per the brief's explicit instruction, enforced at the UI-copy level (Part 17/18) and in the model card (Part 26).

---

## PART 16 — Responsible AI

**False positives/negatives:** both are explicitly possible and disclosed; false positives (real news flagged as misinformation) carry a specific defamation/reputational risk if a source institution is named, so the UI never displays a source-level accusation, only an article-level analysis.

**Political/religious sensitivity:** given the Feb 2026 paper's own authors' admission that their model "may misclassify satire or political dissent," and given Phase 0's documented pattern of religious-minority-targeted misinformation and hate speech in Pakistan, the system must not be framed as capable of adjudicating politically or religiously sensitive claims — the UI includes a standing disclaimer on any politics/religion-domain input (detectable via a simple keyword/domain heuristic, not a hard gate) recommending the user consult a human fact-checker.

**Defamation:** no feature names or accuses a specific outlet/author of dishonesty; source-style analysis (if implemented) is described in aggregate, statistical terms only.

**Satire:** explicitly out of scope for reliable detection; disclosed as a known failure mode, consistent with the literature's own admission.

**Emerging events / out-of-distribution content:** the cross-dataset generalization findings (RQ2) are the direct evidence base for this warning — the UI states that confidence may be unreliable for topics or events not resembling the training data's time range (2017–2023) or domains.

**Model uncertainty communication:** confidence is always labeled "model confidence" (a measure of the model's certainty in its own pattern-match), never "probability the article is false" — this distinction is stated in the UI, the model card, and the thesis's Discussion/Ethical Considerations chapter, directly implementing the brief's explicit requirement.

**Standing disclaimer text (used verbatim across UI, README, and model card):** *"This system provides an AI-assisted analysis of linguistic patterns associated with misinformation in the datasets it was trained on. It does not verify facts, and a prediction is not equivalent to a determination of truth or falsehood. Always consult multiple credible sources, especially for political, religious, or health-related claims."*

---

## PART 17 — Final Product Requirements

**MUST HAVE:** Urdu text input (paste/type); prediction (Real/Fake-risk label) with model confidence, clearly labeled as such; Integrated-Gradients-based explanation with highlighted contributing text spans; the standing responsible-AI disclaimer (Part 16), shown persistently, not just once; graceful error handling (empty input, non-Urdu input detection, extremely short input).

**SHOULD HAVE:** URL-based article extraction (with the security controls in Part 23); a methodology/"How this works" page describing the cross-dataset findings in plain language (this is a major differentiator — turning the research finding into user-facing transparency); a dataset-and-limitations page (states training data, time range, known weak spots); a handful of curated example articles (including at least one from each dataset, one long, one short) so a visitor can try the tool without needing their own Urdu text.

**STRETCH (only if core research + must-haves are done with time to spare):** prediction history (session-local, not account-based, to avoid unnecessary data-handling complexity); a simple public inference API for programmatic access (Part 22), rate-limited.

**FUTURE WORK (not built now):** evidence retrieval/related-article search, multi-language support beyond Urdu, an account system, analytics dashboards, batch/bulk analysis. Building these now would shift effort away from the research core this project is actually being evaluated on.

---

## PART 18 — Frontend Architecture

Technology decision deferred to Phase 11 per the brief ("decide the technology only after comparing alternatives") — at that point, compare a lightweight Streamlit/Gradio app (fastest path to a Hugging Face Spaces deployment, weaker custom UI control) against a small React/Next.js app calling a separate API (more engineering-portfolio value, more work). Given this project's stated goal of demonstrating both research *and* software engineering breadth (per the brief's own framing), the recommendation to carry into Phase 11 is a small React frontend + FastAPI backend (Part 19), deployed as two linked Hugging Face Spaces (one Space per service, or a single Space running both) — Gradio/Streamlit is the documented fallback if time runs short, not the default.

**Screens (wireframe-level):**

1. **Landing/Home** — Purpose: explain what the tool does and does not do (leads with the responsible-AI disclaimer, not the demo). Layout: hero section with disclaimer, "Try it" CTA, three-card summary of methodology/dataset/limitations pages. Components: disclaimer banner, nav, example-article picker. State: none (static). API calls: none.
2. **Detection interface** — Purpose: primary input screen. Layout: large Urdu-aware (RTL) textarea, optional URL field, example-article dropdown, submit button. Components: textarea, URL input, submit, loading spinner. State: input text, loading flag, validation errors. API calls: `POST /api/v1/analyze` on submit. Loading state: spinner + disabled submit. Error state: inline message for empty/invalid input or extraction failure. Empty state: placeholder Urdu-script prompt text.
3. **Results** — Purpose: show prediction, confidence, and a link into the explanation view. Layout: label + confidence gauge, standing disclaimer restated, "See why" expand. Components: confidence bar (never framed as a percentage-true), label badge. State: result object from API. Loading/error/empty states inherited from the request lifecycle.
4. **Explanation** — Purpose: show IG-highlighted spans. Layout: original text with highlighted contributing spans (color-intensity = attribution magnitude), plain-language caption ("these features contributed most"). Components: highlighted-text renderer. State: explanation payload. Error state: fallback message if explanation computation times out (explanation is more expensive than prediction — must degrade gracefully, not block the whole result).
5. **History (stretch only)** — session-local list of past analyses, cleared on refresh; no account system, no persistent storage of user-submitted text beyond the session, for privacy reasons given the sensitive (political/religious) nature of possible inputs.
6. **Methodology** — Purpose: the differentiator page. Plain-language explanation of the cross-dataset generalization finding, with a simple chart (in-domain vs. cross-dataset F1) — this is where the research becomes visible to a non-technical visitor.
7. **About/Research** — links to the thesis/paper, GitHub repo, model card, dataset card, author info.
8. **Responsible-use disclaimer** — a dedicated, permanently linked page (not just a banner) with the full text from Part 16, FAQ-style ("Can this tell me if something is 100% fake?" "No — here's why.").

Accessibility: RTL layout support is a functional requirement, not a nice-to-have, given the script; sufficient color contrast for the explanation-highlighting (color alone must not be the only signal — pair with underline/weight for colorblind accessibility).

---

## PART 19 — Backend Architecture

**Stack (to be finalized at Phase 11, current recommendation):** FastAPI (Python) — natural fit since the ML stack (transformers, captum) is Python-native, avoiding a cross-language serving layer.

**Services:** `InferenceService` (loads the fine-tuned XLM-R checkpoint, runs prediction), `ExplainabilityService` (runs Integrated Gradients, separate from base inference since it's more expensive and should be requestable independently), `ExtractionService` (URL → article text, with the security controls in Part 23), `PreprocessingService` (shared Tier-2 cleaning logic, imported by both training and inference code so preprocessing never silently drifts between train and serve — a common, serious bug class).

**Endpoints (see Part 22 for full contract):** `POST /api/v1/analyze` (text or URL in, prediction + confidence out), `POST /api/v1/explain` (text + prediction id in, explanation spans out — separate from analyze so the UI can show the result immediately and load the explanation asynchronously), `GET /api/v1/examples` (curated example articles), `GET /api/v1/health` (liveness/readiness, model-loaded check), `GET /api/v1/model-info` (model card summary, version, training data description — supports the transparency requirement from Part 16/17).

**Validation:** input length limits (min to avoid meaningless single-word submissions, max to bound compute cost and avoid abuse), basic script detection (warn, don't hard-block, if input doesn't look like Urdu — avoids false confidence on out-of-scope input), URL scheme/format validation before extraction is attempted.

**Error handling:** structured error responses (Part 22) distinguishing user error (400s: bad input) from system error (500s: model/extraction failure) from safety refusals (422: URL blocked by SSRF policy).

**Logging:** request metadata (timestamp, input length, prediction, confidence, latency) — explicitly **not** logging raw submitted text by default, given the sensitive-content privacy stance from Part 18; an opt-in debug-logging flag for local development only.

**Security & rate limiting:** covered fully in Part 23.

---

## PART 20 — Complete ML Pipeline

```
Raw Data (Tier 1, immutable, hashed)
  ↓
Validation (schema check: expected columns, label values, non-empty text; row-count sanity check against known dataset size)
  ↓
Cleaning (Tier 2: Unicode/Urdu normalization, whitespace, HTML/URL/emoji handling — Part 8)
  ↓
Deduplication (exact + near-duplicate removal, within- and cross-dataset — Part 7/9)
  ↓
Dataset Audit (length/vocabulary/source/structure/leakage measurements — Part 7; produces the Dataset Quality Report + Risk Register as versioned artifacts, not just a one-off notebook)
  ↓
Split (stratified train/val/test per Part 9; separate, untouched cross-dataset test pool)
  ↓
Tokenization (Tier 3, model-specific: SentencePiece/WordPiece/TF-IDF vocab — Part 8)
  ↓
Training (Experiments A–E, L; 3-seed repetition for transformers — Part 11/13)
  ↓
Checkpointing (per-epoch or best-val-F1 checkpoint, saved with config + seed — Part 25/26)
  ↓
Evaluation (in-domain metrics — Part 13)
  ↓
Cross-Dataset Testing (Experiments F, G, M — Part 11)
  ↓
Shortcut Analysis (Experiments H, I, K — Part 7/11)
  ↓
Mitigation (Experiment L — Part 11)
  ↓
Error Analysis (Experiment O — Part 14)
  ↓
Explainability (Experiment N — Part 15)
  ↓
Model Selection (choose the checkpoint deployed to the demo — likely the length-stratified XLM-R from Experiment L, since it's the most defensible to present to end users, not necessarily the highest raw in-domain-F1 checkpoint)
  ↓
Model Card (Part 26)
  ↓
Deployment (Part 27)
```

Every stage below "Dataset Audit" produces a versioned artifact (CSV/JSON metrics, saved figures, saved checkpoints) referenced by path in Part 21's folder structure — nothing exists only inside a notebook's transient output.

---

## PART 21 — Complete Folder & File Architecture

> **Superseded notice (added during the Phase 3 pre-coding audit):** the tree below was drafted before the frontend/backend technology stack was finalized. **`ARCHITECTURE.md` Section 4 is now the authoritative repository tree** (it nests the research pipeline under `research/`, and adds the concrete `frontend/`/`backend/` structures this tree only sketched). Read this tree for the conceptual grouping (data/ML/backend/frontend/docs/thesis separation, which is unchanged) but implement against `ARCHITECTURE.md`, not against the literal paths below.

```
urdu-misinfo-audit/
├── README.md                          # Project overview, findings summary, quickstart, links to demo/thesis
├── LICENSE                            # MIT for code; data licensing handled separately (Part 28)
├── CITATION.cff                       # Citable metadata once results are finalized
├── .gitignore
├── .env.example                       # Env var names only, no secrets
├── pyproject.toml / requirements.txt  # Pinned versions (Part 25)
├── environment.yml                    # Conda alt, optional
│
├── data/
│   ├── raw/                           # Tier 1 — immutable, gitignored, fetched via scripts/download_data.py; SHA256 manifest committed
│   │   ├── ax_to_grind/
│   │   ├── notri_fact/
│   │   └── ufnd/                      # populated only if link confirmed (Part 6)
│   ├── clean/                         # Tier 2 — gitignored, regenerable via scripts/preprocess.py
│   ├── processed/                     # Tier 3 — tokenized, model-ready, gitignored
│   ├── splits/                        # train/val/test/cross_dataset_test index files (row IDs, not full text — small, committed to git for reproducibility)
│   └── audit/                         # Dataset Quality Report + Risk Register outputs (CSV/JSON + figures) — committed, since these are core research artifacts
│
├── src/
│   ├── data/
│   │   ├── download.py                # Fetch raw data from source URLs, verify checksums
│   │   ├── validate.py                # Schema/row-count checks
│   │   ├── clean.py                   # Tier-2 preprocessing (shared by training AND inference — imported by src/inference/preprocess.py to avoid train/serve skew)
│   │   ├── dedup.py                   # Exact + near-duplicate detection, within- and cross-dataset
│   │   ├── audit.py                   # Length/vocabulary/source/structure measurements → data/audit/
│   │   └── split.py                   # Stratified splitting, leakage checks
│   ├── models/
│   │   ├── classical.py               # TF-IDF + LR/SVM training and inference
│   │   ├── transformer.py             # mBERT/XLM-R fine-tuning (shared training loop, model name as config param)
│   │   └── length_baseline.py         # RQ3's length-only classifier
│   ├── experiments/
│   │   ├── run_in_domain.py           # Experiments A–E
│   │   ├── run_cross_dataset.py       # Experiments F, G, M
│   │   ├── run_shortcut_analysis.py   # Experiments H, I, K
│   │   ├── run_mitigation.py          # Experiment L
│   │   └── run_all.py                 # Orchestrates the full matrix given a config file
│   ├── evaluation/
│   │   ├── metrics.py                 # Macro-F1, accuracy, calibration, etc. (Part 13)
│   │   └── error_analysis.py          # Sampling + categorization support (Part 14)
│   ├── explainability/
│   │   └── integrated_gradients.py    # Part 15 implementation via captum
│   └── inference/
│       ├── preprocess.py              # Imports src/data/clean.py — guarantees train/serve parity
│       ├── predict.py                 # Loads a checkpoint, runs prediction
│       └── explain.py                 # Serving-side wrapper around explainability/integrated_gradients.py
│
├── configs/
│   ├── data.yaml                      # Dataset paths, split ratios, seed
│   ├── model_xlmr.yaml                # Hyperparameters (Part 6 training strategy — later phase)
│   ├── model_mbert.yaml
│   ├── model_classical.yaml
│   └── experiment_matrix.yaml         # Maps experiment IDs (Part 11) to configs
│
├── notebooks/                         # Exploration only — no result that matters lives only here
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_audit_walkthrough.ipynb
│   └── 03_explainability_examples.ipynb
│
├── scripts/
│   ├── download_data.py               # CLI wrapper around src/data/download.py
│   ├── run_full_pipeline.sh           # Part 20 end-to-end, Colab-friendly
│   └── export_model_card.py           # Generates model card from eval artifacts (Part 26)
│
├── results/
│   ├── metrics/                       # Per-experiment JSON/CSV metrics — committed (small, core evidence)
│   ├── figures/                       # Part 31 figures — committed
│   └── error_samples/                 # Part 14 sampled/annotated examples — committed (with any PII/sensitive-content review)
│
├── checkpoints/                       # gitignored (large); tracked via a MODEL_CARD.md pointer + external storage (HF Hub) instead
│
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI app entrypoint
│   │   ├── routers/analyze.py         # Part 22 endpoints
│   │   ├── routers/explain.py
│   │   ├── services/inference_service.py
│   │   ├── services/explainability_service.py
│   │   ├── services/extraction_service.py   # URL fetch + article extraction (Part 23 security controls live here)
│   │   └── schemas/                   # Pydantic request/response models (Part 22 contract)
│   └── tests/
│
├── frontend/
│   ├── src/ (React app — screens per Part 18)
│   └── tests/
│
├── tests/                             # Cross-cutting: data validation tests, ML sanity tests (Part 24)
│
├── docs/
│   ├── methodology.md                 # Feeds the frontend Methodology page and thesis Methodology chapter
│   ├── dataset_card.md
│   ├── model_card.md
│   ├── responsible_ai.md              # Part 16, source of truth for all disclaimer text
│   └── architecture.md                # This blueprint's living summary, updated as decisions are logged (Part 37)
│
├── thesis/                            # LaTeX/Markdown source for the written thesis (Part 29), kept in-repo for full reproducibility
│
└── .github/
    └── workflows/
        ├── tests.yml                  # Run tests/ + backend/tests + frontend/tests on PR
        └── lint.yml
```

Every file above has a single, stated responsibility; nothing is created ad hoc during implementation without first being added here (and logged if it changes the plan, per Part 37).

---

## PART 22 — API Specification

Versioned under `/api/v1/`. No authentication for the public demo (rate-limited instead, Part 23); a future authenticated tier is future work, not built now.

| Method | Path | Purpose | Request | Response | Validation | Errors |
|---|---|---|---|---|---|---|
| POST | `/api/v1/analyze` | Predict misinformation-risk for submitted text or URL | `{ "text": string? , "url": string? }` (exactly one required) | `{ "id": string, "label": "real_pattern"\|"fake_pattern", "confidence": float, "disclaimer": string }` | text length 10–5000 chars; url must be http(s), pass SSRF checks (Part 23) | 400 (missing/both fields, text too short/long), 422 (URL blocked by security policy), 502 (extraction failed), 500 (model error) |
| POST | `/api/v1/explain` | Get IG-based explanation for a prior prediction | `{ "id": string }` | `{ "id": string, "spans": [{ "text": string, "attribution": float }], "note": string }` | id must reference a recent analyze call (short-lived in-memory cache, not persistent storage) | 404 (id not found/expired), 500 (explanation computation error), 504 (explanation timeout — degrade gracefully per Part 18) |
| GET | `/api/v1/examples` | Curated example articles for demo purposes | — | `{ "examples": [{ "id": string, "title": string, "text": string, "source_dataset": string }] }` | — | 500 |
| GET | `/api/v1/model-info` | Model/version/training-data transparency info | — | `{ "model": string, "version": string, "training_data": string, "known_limitations": [string] }` | — | 500 |
| GET | `/api/v1/health` | Liveness/readiness probe | — | `{ "status": "ok", "model_loaded": bool }` | — | 503 (model not loaded) |

**Latency targets:** `/analyze` under ~2s on CPU inference (acceptable for a Spaces free-tier deployment given XLM-R-base's size), `/explain` under ~5s (IG is more expensive; async/polling considered if this proves too slow in practice — a decision deferred to Phase 13 integration testing, logged per Part 37 if it changes).

**Versioning strategy:** path-based (`/api/v1/`); a breaking change to response schemas requires `/api/v2/`, old version kept alive during any transition — realistic given this is a portfolio/thesis project, not a policy over-engineered for a system with no other consumers yet.

---

## PART 23 — Security

**URL extraction (SSRF prevention, the main real risk here):** resolve and validate the URL's IP before fetching, reject requests to private/loopback/link-local address ranges (RFC1918, 127.0.0.0/8, 169.254.0.0/16, etc.) and to non-standard ports; disallow redirects to a different resolved IP class than the original (re-validate after each redirect hop, cap at 3 hops); enforce a request timeout (e.g., 10s) and a maximum response size (e.g., 2MB) before parsing; strip and never execute any script content from fetched pages (text-extraction only, via a library like `trafilatura` or `readability`, never a headless browser that could execute page JS).

**Input validation:** length bounds (Part 22), rejection of binary/non-text payloads, basic HTML/script-tag stripping on any extracted text before it reaches the model (defense in depth, even though the model itself doesn't execute anything).

**Secret management:** `.env`, never committed; `.env.example` documents required variable names only; any HF Hub tokens used for model download are read from environment/secrets store, not hardcoded.

**Dependency security:** `pip-audit` or GitHub Dependabot enabled on the repo (Part 28); pinned versions (Part 25) reduce surprise transitive-dependency changes.

**Rate limiting:** IP-based sliding-window limit (e.g., 20 requests/minute) on `/analyze` and `/explain`, implemented at the FastAPI middleware level — sufficient for a free-tier public demo; no enterprise WAF/CDN-level protection, which would be disproportionate infrastructure for this project's scale and explicitly rejected per the brief's "no unnecessary enterprise infrastructure" instruction.

---

## PART 24 — Testing Strategy

**Unit tests:** `src/data/clean.py` normalization functions (known input → known output pairs, including Urdu-character-variant cases), `src/evaluation/metrics.py` metric computations against hand-computed small examples, `src/data/dedup.py` against synthetic near-duplicate pairs.

**Data validation tests:** schema checks on raw data (Part 20's Validation stage) run as actual test-suite assertions, not just a notebook cell — e.g., "Ax-to-Grind raw file has exactly 2 label values," "no null text fields survive Tier 2 cleaning."

**ML sanity tests:** model outputs valid probabilities summing to 1; inference is deterministic given a fixed seed and identical input; empty-string input is rejected before reaching the model, not passed through; extremely long input is truncated, not crashed on; a saved checkpoint loads without error and produces the same prediction on a fixed regression-test example every time (catches silent checkpoint/environment drift); explanation output is non-empty and its span attributions sum to a value consistent with IG's completeness axiom within a numerical tolerance (a genuine correctness check, not just a smoke test).

**Backend/API tests:** each endpoint in Part 22 tested for its documented success and error paths, including the SSRF-blocked-URL case explicitly (a security regression test, not just a happy-path test).

**Frontend tests:** component-level tests for the Detection and Results screens' state transitions (loading/error/empty), not full E2E browser automation, which is disproportionate for a solo student project — logged as a scope decision (Part 37).

**End-to-end test:** one full-path test (submit example article → receive prediction → receive explanation) run in CI against a lightweight/mocked model if the real checkpoint is too large for CI, or against the real small checkpoint if feasible.

**Acceptance criteria:** all REQUIRED experiments (Part 12) reproduce within the documented seed-variation band (Part 13) before a checkpoint is promoted to "deployed"; all `tests/`, `backend/tests`, `frontend/tests` pass in CI before merge to main.

---

## PART 25 — Reproducibility

Python 3.11 (specific patch version pinned in CI); all package versions pinned in `requirements.txt`/`pyproject.toml` (transformers, torch, captum, scikit-learn, fastapi versions recorded exactly, not as ranges); fixed random seeds (`{42, 123, 2026}` for the 3-seed transformer runs, documented in `configs/`); dataset versions pinned via the SHA256 manifest in `data/raw/` (Part 21) plus the exact download date/source URL recorded in `docs/dataset_card.md`; every experiment run reads its full configuration from a `configs/*.yaml` file, and that exact config file is saved alongside the resulting checkpoint/metrics (no experiment is "reproducible" only in someone's memory of what flags they passed); hardware documented (Google Colab T4, 16GB VRAM, specific driver/CUDA version noted at time of the final training run, since this can silently affect numerics); all saved predictions (not just aggregate metrics) for the test and cross-dataset test sets are stored in `results/metrics/` so a reviewer can recompute any derived statistic without rerunning inference.

---

## PART 26 — MLOps / Model Versioning

Model naming convention: `urdu-misinfo-{model}-{dataset}-{purpose}-v{n}` (e.g., `urdu-misinfo-xlmr-axtogrind-mitigated-v1`); checkpoints pushed to the Hugging Face Model Hub (free, and directly compatible with an HF Spaces deployment — Part 27) rather than stored in git, with each Hub model repo carrying its own model card (auto-generated by `scripts/export_model_card.py` from the evaluation artifacts, so the model card can never silently drift out of sync with actual measured results). Experiment tracking: **Weights & Biases free tier** (or, as a zero-dependency fallback, structured local logging to `results/metrics/*.json` if W&B integration adds more setup overhead than value at the project's scale) — recommended because it's free for public/academic projects and gives seed-variance and cross-experiment comparison views for free, directly useful for Part 13's 3-seed reporting; no paid infrastructure introduced anywhere in this pipeline. Dataset version: the SHA256 manifest (Part 25) doubles as the dataset version identifier, referenced in every experiment config and model card.

---

## PART 27 — Deployment

```
Training (Colab) → Push checkpoint + tokenizer to Hugging Face Model Hub
                          ↓
                  Backend (FastAPI) loads model from Hub at startup
                          ↓
                  Backend deployed as a Hugging Face Space (Docker SDK, CPU basic tier)
                          ↓
                  Frontend (React) deployed as a second HF Space (static) or served by the same backend Space if simpler
                          ↓
                  Public demo URL
```

**Comparison of options considered:** HF Spaces (free CPU tier, generous for a portfolio demo, some cold-start latency and a ~16GB RAM ceiling on the free CPU tier — XLM-R-base fits comfortably) vs. Streamlit Community Cloud (simpler for a pure-Python single-file app, weaker fit if the React-frontend direction from Part 18 is taken) vs. a self-hosted VM (rejected — introduces cost and maintenance burden disproportionate to project scope). **Recommendation: Hugging Face Spaces**, consistent with the verified product-layer gap from Phase 0/1 (no Urdu FND Space currently exists) and with the "free/low-cost" constraint running through the whole brief.

**Memory/cold-start handling:** model loaded once at container startup (not per-request), quantization (e.g., to fp16 or int8 via `optimum`) considered if CPU-tier memory or latency becomes a real bottleneck during Phase 15 integration testing — a decision deferred and logged (Part 37) rather than pre-committed to now, since it depends on measured, not assumed, performance.

---

## PART 28 — GitHub Repository Design

`README.md`: problem statement, the core finding (with the F1 0.771 vs 0.005 headline result once your own numbers replace/confirm Haroon's), architecture diagram, quickstart, links to the live demo, thesis, and model card — written to be legible to a hiring manager skimming for 60 seconds, not just to an examiner. `LICENSE`: MIT for all original code. Data is **not** redistributed in the repo (per Part 1's license-ambiguity finding on Ax-to-Grind) — `data/raw/` is gitignored and populated via `scripts/download_data.py`, which fetches from the original sources and cites them, avoiding any relicensing risk. `docs/model_card.md` and `docs/dataset_card.md`: follow the standard HF model-card/dataset-card templates, auto-populated where possible (Part 26). `results/figures/`: committed, since these are core evidence a reader should see without cloning and rerunning anything. `CITATION.cff`: added once the thesis/paper text is stable, so the project is citable by others. Reproduction instructions: a single `scripts/run_full_pipeline.sh` documented in the README, plus an explicit "expected runtime and hardware" note (Colab T4, approximate hours) so a reader knows what they're committing to before running it.

---

## PART 29 — Thesis Structure

1. Abstract
2. Introduction
3. Problem Statement
4. Motivation (Phase 0/1 findings: Pakistan misinformation landscape, English/Urdu fact-checking gap)
5. Research Questions (Part 4)
6. Objectives
7. Literature Review (the verified table from Part 1, expanded)
8. Research Gap (Part 2)
9. Dataset (Part 6)
10. Dataset Audit (Part 7)
11. Methodology (Parts 8–10, 13)
12. Models (Part 10)
13. Experimental Design (Parts 11–12)
14. Shortcut Analysis (RQ3, Experiments H/I/K)
15. Cross-Dataset Evaluation (RQ2, Experiments F/G)
16. Mitigation (RQ4, Experiment L/M)
17. Explainability (RQ5, Part 15)
18. Results (consolidated, all experiments)
19. Error Analysis (Part 14)
20. Discussion
21. Limitations (explicit: single-annotator error analysis, Notri-Fact's undocumented provenance, no formal significance testing beyond seed variance, no Roman Urdu coverage)
22. Ethical Considerations (Part 16)
23. System Implementation (Parts 18–24, summarized — full detail lives in repo docs, not duplicated at length in the thesis body)
24. Deployment (Part 27)
25. Conclusion
26. Future Work (Part 12's Future Work list + anything else surfaced during the project)
27. References
28. Appendices (full hyperparameter tables, additional figures, extended error-analysis samples)

No sections merged from the brief's suggested list — the 27-chapter structure maps cleanly onto the work already scoped above, and forcing merges would create awkward chapters mixing unrelated content (e.g., combining Methodology with Models would bury the model-selection justification inside preprocessing detail).

---

## PART 30 — Research-Paper Structure (for later conversion)

Main contribution to lead with: the multi-model, mitigation-inclusive extension of the length-confound finding (not "we detect Urdu fake news"). Core experiment to headline: the F/G cross-dataset transfer matrix (Part 11) — this is the single result that carries the paper. Key figures: length-by-label distribution (Part 7/31), the 4-model × 2-direction cross-dataset F1 heatmap, the length-ablation curve (F1 vs. word-count cap), a before/after mitigation comparison. Key tables: in-domain vs. cross-dataset metrics per model, the length-only-baseline comparison. Main findings to state plainly: (1) the confound replicates beyond XLM-R, (2) mitigation partially but not fully recovers generalization, (3) explanations partially surface the artifact. Limitations to state plainly, not bury: single-author-equivalent error analysis, one dataset pair, binary-only framing. This structure is prepared now so that, once results exist, paper-writing is mostly reformatting rather than new synthesis — no paper text is written at this stage, per the brief.

---

## PART 31 — Figures and Tables

| # | Figure/Table | Purpose | Data source | Required experiment |
|---|---|---|---|---|
| 1 | Dataset size & class distribution (both datasets) | Orient the reader | Part 7 audit | Data acquisition |
| 2 | Length distribution by label, both datasets | Visualize the confound directly | Part 7 audit | Data acquisition |
| 3 | Top vocabulary/log-odds words by class | Surface possible source/style shortcuts | Part 7 audit | Data acquisition |
| 4 | System architecture diagram | Explain the deployed pipeline | Part 20/27 | None (design artifact) |
| 5 | ML training pipeline diagram | Explain the experimental pipeline | Part 20 | None (design artifact) |
| 6 | In-domain model comparison table (4 models × 2 datasets) | RQ1 results | Experiments A–E | A–E |
| 7 | Cross-dataset F1 heatmap (4 models × 2 directions) | RQ2 headline result | Experiments F, G | F, G |
| 8 | Length-bucketed performance chart | Connects RQ2/RQ3 | Experiment J | J |
| 9 | Length-only-baseline vs. full-model F1 | RQ3 headline result | Experiment H | H |
| 10 | Length-ablation curve (F1 vs. word cap) | RQ3, replicates/extends Haroon | Experiment I | I |
| 11 | Mitigation before/after comparison | RQ4 headline result | Experiments L, M | L, M |
| 12 | Confusion matrices (per model, in-domain and cross-dataset) | Supports Results chapter | All model experiments | A–G |
| 13 | Explainability example spreads (2–3 worked examples) | RQ5, makes the abstract finding concrete | Experiment N | N |
| 14 | Error-category breakdown chart | Part 14 | Experiment O | O |
| 15 | Calibration/reliability diagram | Part 13, supports Part 16's confidence-communication claim | Primary model | D or L |

---

## PART 32 — FYP / Thesis Defense Preparation

**ML:** *Why XLM-R?* — matches the paper being directly extended (Haroon 2026), keeping results comparable; strongest multilingual pretraining among the free-tier-feasible options surveyed. *Why mBERT?* — the genuine novel extension over Haroon, who tested only XLM-R; needed to know if the collapse is model-general or XLM-R-specific. *Why a classical baseline?* — FIRE2021 showed classical models can beat transformers on this task family; also the fastest way to isolate the length shortcut (a linear model on word count alone). *What is fine-tuning / transfer learning?* — standard definitions, tied concretely to "we start from XLM-R's multilingual pretraining and adapt its final layers to the binary Urdu-misinformation task."

**Dataset:** *Why this dataset?* — largest well-documented, non-machine-translated, expert-annotated Urdu FND dataset with a directly-relevant, independently-confirmed confound to study (Part 1/6). *How were labels created?* — expert journalists, κ=0.94 (Ax-to-Grind); explicitly note Notri-Fact's undocumented labeling as a stated limitation, not a strength. *How did you detect leakage?* — Part 7's exact/near-duplicate scan, both within-dataset and cross-dataset, run before any split. *Why not combine datasets?* — doing so would destroy the ability to run the core zero-shot cross-dataset experiment that is the thesis's central method.

**Research:** *What is your contribution?* — Part 5, stated precisely: replication + multi-model extension + mitigation attempt + explainability layer, not a new SOTA or new dataset. *What is novel?* — extending a single-model (XLM-R-only) 2026 preprint's finding to a classical baseline and a second transformer, plus the mitigation and explainability layers that preprint didn't attempt. *Why cross-dataset testing?* — because in-domain accuracy alone cannot distinguish genuine signal from shortcut learning, which the field's own 2026 literature has just started to acknowledge. *What did you discover?* — answered empirically once results exist; the blueprint predicts (Part 4) but does not presuppose the answer.

**Evaluation:** *Why F1 (macro) over accuracy?* — Part 13: penalizes exactly the prediction-collapse failure mode central to this thesis, which accuracy can mask under class imbalance. *Why not accuracy alone?* — same reasoning, stated from the other direction; accuracy is still reported for literature comparability. *What does cross-dataset failure mean?* — the model has learned patterns specific to one dataset's construction (e.g., its length distribution) rather than misinformation-general signal — explained with the F1 0.005/prediction-collapse concept directly.

**Explainability:** *Why Integrated Gradients over SHAP?* — Part 15: better Colab-budget fit, a cleaner formal completeness guarantee, easier to defend precisely than a sampling-based Shapley approximation. *Are explanations trustworthy?* — explicitly not treated as ground truth; validated only by cross-referencing against the independently-measured length/structural audit (Part 15), and this limitation is stated directly, not glossed over.

**Engineering:** *Why this architecture?* — Part 19/21: separation of inference and explainability services because they have different cost profiles; shared preprocessing module between training and serving specifically to prevent train/serve skew, a concrete engineering decision with a stated reason. *How is the model deployed?* — HF Hub + HF Spaces, chosen for free-tier feasibility and to fill the verified product-gap from Phase 0/1. *How are failures handled?* — Part 22's structured error taxonomy (400/422/500/502/504), and Part 18's explicit UI error states.

**Responsible AI:** *Can the system prove something is fake?* — No, stated plainly, with the standing disclaimer text from Part 16 quoted directly. *What happens with satire?* — disclosed as a known, literature-acknowledged failure mode, not silently ignored. *What about political content?* — a specific UI-level caution is triggered (Part 16), and the thesis's Ethical Considerations chapter addresses it directly, tied to Phase 0's documented religious/political misinformation patterns in Pakistan.

---

## PART 33 — Portfolio & Career Impact Plan

**AI/ML hiring manager view:** demonstrates the ability to read current literature critically (catching and extending a 2026 preprint's finding, not just citing it), design a multi-condition experiment matrix, and reason about failure modes (shortcut learning) rather than chase a leaderboard number — a materially stronger signal than "I fine-tuned a transformer and got 95%," which is now a commodity claim.

**Software engineering hiring manager view:** a full pipeline from data validation through a deployed, tested, documented API and frontend, with explicit train/serve-parity engineering (shared preprocessing module) and a real security control (SSRF prevention) — concrete, checkable evidence of engineering judgment, not just ML scripting.

**Scholarship committee view:** demonstrates initiative (pivoting away from a redundant idea after independent research, rather than building the first thing that came to mind), genuine social-impact grounding (the documented Urdu/English fact-checking gap in Pakistan), and responsible-AI awareness (the explicit confidence-communication and disclaimer design) — a narrative that shows judgment, not just technical output.

**Academic reviewer view:** a defensible methodology (Parts 4/11/13), honest scope limitations stated upfront rather than discovered by an examiner (Part 29's Limitations chapter), and reproducibility infrastructure (Part 25) that would let another researcher actually check the claims.

**CV project description (draft, ~40 words):** *"Investigated whether Urdu misinformation-detection models learn genuine linguistic signals or dataset-specific shortcuts (e.g., article length), across classical ML and transformer models (mBERT, XLM-RoBERTa), with cross-dataset generalization testing, mitigation experiments, and an explainability-based audit. Deployed as a public, responsibly-framed demo on Hugging Face Spaces."*

**GitHub presentation strategy:** lead the README with the headline empirical finding (once measured) and a link to the live demo, not with a generic "final year project" framing.

**Live demo strategy:** the Methodology page (Part 18) is the differentiator — most portfolio demos show only a prediction; this one visibly explains its own limitations, which is a more sophisticated signal to a technical reviewer than a bare accuracy badge.

**Scholarship/SOP narrative:** "I started with a standard classifier idea, independently found through literature review that it was already solved, and redesigned the project around a genuine, current gap I could defend" — a stronger initiative narrative than describing the final system alone.

**Interview talking points:** the F1 0.005 collapse number (a vivid, memorable, technically substantive detail); the decision to reject a "beat 96%" framing in favor of a diagnostic contribution (shows research maturity); the SSRF/train-serve-parity engineering decisions (shows engineering maturity beyond notebook-level ML).

No claims above assume a specific numeric result before the experiments are run — all are structured around the process and design decisions, which are already defensible regardless of final numbers.

---

## PART 34 — Scope Control

**CORE THESIS (must complete):** Experiments A, B, C, D, F, G, H, I, N, O (Part 12's REQUIRED set); the dataset audit and risk register (Part 7); the responsible-AI-framed deployed demo with at minimum the MUST HAVE features (Part 17); the thesis chapters 1–22, 25–27 (Part 29) at minimum, with 23–24 (system implementation/deployment) covered at whatever depth the built system actually reached.

**EXTENSIONS (strengthen if time permits):** Experiments E, J, K, L, M (Part 12's STRONGLY RECOMMENDED set); the SHOULD HAVE product features (Part 17); a full research-paper draft (Part 30).

**STRETCH (attempt only after core + extensions are solid):** Experiment P (domain-adaptive pretraining); the STRETCH product features (prediction history, public API); UFND as a confirmed secondary dataset if its access materializes.

**FUTURE WORK (explicitly not attempted in this project):** evidence retrieval, Roman Urdu modeling, multimodal input, multi-class veracity, any SOTA-chasing reformulation of the task.

This ordering is the actual defense against scope creep — if a deadline forces a cut, cut from the bottom of this list, never by quietly weakening the REQUIRED experiment set to make room for a product feature.

---

## PART 35 — Risk Register

| Risk | Probability | Impact | Mitigation | Fallback |
|---|---|---|---|---|
| UFND dataset link never materializes | High | Low | Already scoped as optional/conditional (Part 6) | Proceed with Ax-to-Grind + Notri-Fact only; no core-thesis impact |
| Ax-to-Grind license ambiguity becomes a real redistribution problem | Low | Medium | Never redistribute raw data in the repo (Part 28); link to source only | Cite dataset, request explicit permission from authors if redistribution is ever needed |
| Undetected cross-dataset duplication invalidates RQ2's zero-shot claim | Medium | **Critical** | Mandatory near-duplicate scan before any transfer experiment (Part 7/9) — a go/no-go gate | If overlap found, remove affected items from the test side and re-document the (slightly reduced) test set size |
| Mitigation experiment (RQ4) shows no improvement | Medium | Low (scientifically) / Medium (narrative) | Already framed as an honest possible outcome (Part 4/5) — a negative result is still a contribution | Reframe Discussion around "why simple length-stratification is insufficient," pointing to entangled confounds as future work |
| Colab session limits disrupt multi-seed transformer training | Medium | Medium | Checkpoint-and-resume built into the training loop from the start (Part 20/25) | Reduce to 2 seeds if 3 proves infeasible within the timeline, documented as a scope adjustment (Part 37) |
| Explainability (IG) proves too slow even on the sampled subset | Low-Medium | Medium | Sampled, not full-test-set, scope already (Part 14/15); batched computation | Reduce sample size further, document the smaller n explicitly as a limitation |
| Deployment memory limits on HF Spaces free CPU tier | Low-Medium | Medium | XLM-R-base is comfortably sized for this tier; fp16/int8 quantization available if needed (Part 27) | Fall back to a smaller/quantized checkpoint for the live demo while reporting full-precision results in the thesis |
| URL extraction fails on many real-world Urdu news sites (paywalls, JS-heavy pages) | Medium | Low | Text-input remains the primary, always-working path (Part 17); URL extraction is a SHOULD HAVE, not MUST HAVE | Document known-working example sources; don't block the thesis on universal URL support |
| Research novelty risk: another paper scoops this exact multi-model extension before submission | Low-Medium (the space is moving fast, evidenced by the Feb/Jul 2026 papers found in this pass) | Medium | Move the REQUIRED experiment set (Part 12) to the front of the roadmap (Part 36) rather than polishing product features first | If scooped, reframe as an independent replication/confirmation study — still a valid, honest contribution, and independent replication has its own scientific value |
| Thesis/FYP deadline pressure | Medium-High (typical for any FYP) | High | Part 34's strict scope-control ordering exists specifically for this | Cut from STRETCH → EXTENSIONS first; REQUIRED core is protected |

---

## PART 36 — Complete Project Roadmap

| Phase | Goal | Key tasks | Files involved | Validation criteria | Difficulty | Est. time |
|---|---|---|---|---|---|---|
| 0 | Validation | Completed | `RESEARCH_VALIDATION_REPORT.md` | Delivered | — | Done |
| 1 | Dataset research | Completed | Same | Delivered | — | Done |
| 2 | Architecture/research design | This document | `MASTER_PROJECT_BLUEPRINT.md` | Your approval | — | Done pending approval |
| 3 | Dataset acquisition | `src/data/download.py`, checksum manifest, attempt UFND link | `data/raw/` | Raw files present, checksums match, row counts match Part 6's documented sizes | Low | 2–4 days |
| 4 | Dataset audit | `src/data/clean.py`, `dedup.py`, `audit.py` | `data/clean/`, `data/audit/` | Dataset Quality Report + Risk Register produced (Part 7); cross-dataset duplication gate passed | Medium | 1–1.5 weeks |
| 5 | Baselines | `src/models/classical.py`, `length_baseline.py`, Experiments A/B/H | `results/metrics/` | In-domain classical F1 in plausible literature-comparable range | Low-Medium | 3–5 days |
| 6 | Transformer training | `src/models/transformer.py`, Experiments C/D/E, 3-seed runs | `checkpoints/` (HF Hub), `results/metrics/` | In-domain transformer F1 in plausible range; seed variance reported | High | 1.5–2.5 weeks (Colab session limits, Part 35) |
| 7 | Cross-dataset evaluation | Experiments F/G | `results/metrics/` | Bidirectional transfer numbers obtained for all 4 models | Medium | 3–5 days |
| 8 | Shortcut analysis | Experiments H (done in Phase 5)/I/K | `results/metrics/`, `results/figures/` | Length-ablation and vocabulary-shortcut results obtained | Medium | 1 week |
| 9 | Mitigation experiments | Experiment L/M | `results/metrics/` | Mitigated model's in-domain + cross-dataset F1 obtained | Medium | 3–5 days |
| 10 | Explainability | `src/explainability/integrated_gradients.py`, Experiment N, error analysis (O) | `results/error_samples/`, `results/figures/` | Sampled explanation review completed and categorized (Part 14/15) | Medium-High | 1–1.5 weeks |
| 11 | Backend | `backend/app/*` | Backend service tree (Part 21) | All Part 22 endpoints pass their tests (Part 24) | Medium | 1–1.5 weeks |
| 12 | Frontend | `frontend/src/*` | Frontend tree (Part 21) | All Part 18 screens implemented with documented states | Medium | 1.5–2 weeks |
| 13 | Integration | Wire frontend↔backend↔model | Both trees | End-to-end test passes (Part 24) | Medium | 3–5 days |
| 14 | Testing | Full `tests/` suite, CI | `.github/workflows/` | CI green on `tests.yml`/`lint.yml` | Low-Medium | 3–5 days |
| 15 | Deployment | HF Hub push, HF Spaces deploy | `scripts/`, HF Hub/Spaces | Public demo URL live, health check passes, memory/latency within targets (Part 27) | Medium | 3–5 days |
| 16 | Documentation | README, model/dataset cards, `docs/` | `docs/*`, `README.md` | A fresh reader can reproduce the pipeline from the README alone | Low-Medium | 4–6 days |
| 17 | Thesis | Full write-up per Part 29 | `thesis/` | Complete draft ready for supervisor review | High | 2–4 weeks |
| 18 | Paper (optional/extension) | Reformat per Part 30 | `thesis/` derivative | Submittable draft to a workshop/regional venue | Medium-High | 1–2 weeks (after thesis) |
| 19 | Defense preparation | Rehearse Part 32's Q&A bank against actual results | — | Can answer every Part 32 question with your own measured numbers, not the blueprint's predictions | Low-Medium | 1 week |

Total realistic estimate for Phases 3–17 (the core thesis path): roughly **3–4 months** of consistent part-time student effort, before paper/defense-prep phases — a number you should sanity-check against your actual FYP calendar and adjust the STRONGLY RECOMMENDED/STRETCH scope (Part 34) accordingly rather than compressing the REQUIRED core.

---

## PART 37 — Decision Register

| # | Decision | Evidence | Alternatives considered | Reason | Trade-offs | Status |
|---|---|---|---|---|---|---|
| D1 | Pivot from plain classification to cross-dataset generalization + shortcut analysis | Phase 0/1 literature review; Part 1 re-verification of Haroon (2026) and the Feb 2026 SOTA paper | Keep original plain-classifier framing; pursue evidence-retrieval (Direction E) instead | Plain classifier is redundant with published SOTA; evidence-retrieval lacks a supporting Urdu dataset and would dilute focus | Lower "flashy accuracy number" to show; higher research legitimacy | **Approved (Phase 0/1), carried forward** |
| D2 | Primary dataset: Ax-to-Grind Urdu | Part 6 comparison matrix | Notri-Fact as primary; UFND as primary; Hook and Bait as primary | Best-documented annotation methodology + directly relevant, independently-confirmed confound to study | Inherits the length confound as a property of the primary dataset — but that's the subject of the thesis, not a flaw to avoid | **Proposed — pending your approval** |
| D3 | Add mBERT as the required second transformer (beyond Haroon's XLM-R-only study) | Part 5/10 — this is the stated novel extension | XLM-R only (pure replication); add a third transformer (e.g., XLNet) instead | mBERT gives the minimum viable "is this model-general?" check without disproportionate added training cost | More Colab time than a single-model study | **Proposed — pending approval** |
| D4 | Primary metric: Macro-F1 | Part 13 | Accuracy as primary; AUC as primary | Directly sensitive to the prediction-collapse failure mode central to the thesis | Slightly less familiar to a general audience than accuracy (mitigated by reporting both) | **Proposed — pending approval** |
| D5 | Explainability method: Integrated Gradients (not SHAP, not attention) | Part 15 | SHAP; raw attention | Better Colab-budget fit, cleaner formal justification (completeness axiom) | SHAP's Shapley-theoretic framing is arguably more familiar to some reviewers — noted as a documented alternative, not implemented | **Proposed — pending approval** |
| D6 | No dataset concatenation, ever, across the primary/cross-dataset pair | Part 6/9 — explicit brief instruction, reinforced by the experiment design itself | Concatenate for a larger training set | Concatenation would destroy the core zero-shot cross-dataset experiment | Somewhat smaller effective training set than a merged corpus would offer | **Approved by design constraint, non-negotiable given the RQs** |
| D7 | Frontend/backend stack: React + FastAPI, with Streamlit/Gradio as documented fallback | Part 18/19 | Streamlit/Gradio as the primary choice | Better demonstrates the software-engineering breadth the brief explicitly asks the project to show | More implementation time than a Streamlit app | **Proposed — final call deferred to Phase 11 per the brief's own "decide only after comparing alternatives" instruction; logged here as the current leaning, not a locked decision** |
| D8 | Deployment: Hugging Face Spaces | Part 27 | Streamlit Community Cloud; self-hosted VM | Free tier, fills the verified product-gap, natural fit with HF Hub model storage | Some cold-start latency and a hard memory ceiling on free tier | **Proposed — pending approval** |
| D9 | No new dataset creation/annotation claimed as a contribution | Part 5, per the brief's explicit instruction not to overclaim | Manually annotate a small Roman Urdu supplementary set | Time budget better spent on the core REQUIRED experiment set (Part 12); Roman Urdu annotation is thesis-scale on its own | Leaves the verified Roman Urdu gap (Phase 0/1) unaddressed, explicitly deferred to Future Work | **Approved by scope-control logic (Part 34)** |

**Process for future changes:** if evidence during implementation (e.g., Phase 4's audit, Phase 6/7's actual results) contradicts a decision above, the change will be proposed here in this same table format (old decision → new evidence → new decision → trade-offs) and flagged to you explicitly before being acted on, per the brief's own rule — nothing above is silently overridden mid-implementation.

---

## PART 38 — Final Quality Review

**As FYP Supervisor:** Approvable. The project has a specific, evidence-backed research question, a realistic scope-control mechanism, and doesn't overreach into claims (SOTA, new dataset, fact-verification) it can't support. Required change before final approval: confirm your actual FYP calendar against Part 36's ~3–4 month core-path estimate before committing to the full STRONGLY RECOMMENDED set as "expected," not just "hoped for."

**As Thesis Examiner:** The methodology (Part 4/9/11/13) is defensible — hypotheses are falsifiable, splits are leakage-conscious, the primary metric is justified rather than assumed. Weakness to address before submission: the error-analysis and explanation-categorization steps (Parts 14/15) rely on a single annotator (you) with no inter-rater agreement measure — this is disclosed as a limitation, which is the right call, but be ready to defend why a second annotator wasn't feasible (time/scope, honestly stated) rather than implying it wasn't needed.

**As Research Reviewer:** A genuine, if modest, contribution exists (Part 5) — multi-model extension and mitigation attempt on a very recent single-model finding, not a replication-only or SOTA-chasing project. Risk: the field is moving fast (two major papers found in Q1/Q3 2026 during this project's own research phase); mitigated by Part 35's explicit "scooped" contingency (reframe as independent confirmation, still valid).

**As ML Engineer:** Technically sound. The train/serve preprocessing-parity decision (Part 19/21) and the mandatory cross-dataset duplication gate (Part 7/9/35) are the two design choices that most distinguish this from a typical student project — both are the kind of detail that silently breaks results if skipped, and both are made explicit here rather than left to be discovered late.

**As Software Architect:** Maintainable. The folder structure (Part 21) cleanly separates data/model/backend/frontend concerns, and the shared preprocessing module prevents a common, hard-to-debug class of production ML bug. Watch item: `notebooks/` must stay exploration-only in practice, not become a shadow home for results that should live in `results/` — worth a one-line CI or review-checklist reminder once implementation starts.

**As Hiring Manager:** Demonstrates real skill across both research reasoning (Part 1's independent primary-source verification, not just trusting a summary) and engineering judgment (Parts 19/23's specific, justified security and architecture decisions). This is a stronger portfolio signal than a bare accuracy-focused classifier project.

**As Scholarship Reviewer:** Demonstrates initiative (the pivot itself, and today's re-verification pass), social grounding (Phase 0's documented Urdu/English fact-checking gap), and responsible framing (Part 16) — a coherent, honest narrative rather than an inflated one.

**Revisions made based on this review, already incorporated above:** the length-ablation nuance from Part 1 (confound inflates but doesn't solely drive in-domain performance) is carried through consistently into RQ3, Part 15's explanation-validation approach, and Part 32's defense answers, rather than being stated once and then contradicted by an oversimplified "it's just measuring length" framing elsewhere in the document.

---

# PHASE 2 COMPLETE — AWAITING APPROVAL

This blueprint covers all 38 requested parts. Nothing in it has been implemented — no code, no training, no deployment. The Decision Register (Part 37) marks which calls are firm (D1, D6, D9) versus proposed and awaiting your explicit approval (D2–D5, D7, D8). Please review, in particular: the primary dataset and mBERT-as-second-transformer choice (D2/D3), the Macro-F1 and Integrated Gradients choices (D4/D5), and the frontend/deployment leanings (D7/D8) — and flag anything you want changed before implementation begins.
