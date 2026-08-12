# Research Paper Plan

Full reasoning: `MASTER_PROJECT_BLUEPRINT.md` Part 30. This document exists so that, once real results are in, converting the thesis into a paper draft is reformatting, not new synthesis — no paper text is written before then.

**Target venue class:** a regional/workshop-tier NLP venue (e.g., a South Asian NLP workshop) or a national CS conference — realistic given the scope (`RESEARCH_VALIDATION_REPORT.md` Section 0.6), not a top-tier ACL/EMNLP main track claim.

**Lead contribution (the paper's one sentence):** a multi-model (classical + mBERT + XLM-R), mitigation-inclusive extension of a single-model 2026 preprint's length-confound finding in Urdu fake-news detection — not "we detect Urdu fake news," and not a SOTA claim.

**Core experiment to headline:** the cross-dataset transfer matrix (`EXPERIMENT_PLAN.md` experiments F/G) — 4 models × 2 transfer directions. This single result set carries the paper's argument.

**Key figures (subset of `MASTER_PROJECT_BLUEPRINT.md` Part 31, paper-selected):** length-by-label distribution (both datasets), the cross-dataset F1 heatmap, the length-ablation curve, the before/after mitigation comparison.

**Key tables:** in-domain vs. cross-dataset metrics per model (the core results table), the length-only-baseline-vs-full-model comparison.

**Findings to state plainly (once measured, not presupposed):** (1) whether the confound replicates beyond XLM-R across mBERT and classical baselines, (2) whether length-stratified mitigation partially or fully recovers cross-dataset generalization, (3) whether Integrated Gradients explanations detectably surface the artifact.

**Limitations section (non-negotiable, stated plainly):** single-reviewer error analysis (no inter-annotator agreement computable), one dataset pair, binary-only framing, Notri-Fact's undocumented provenance.

**Source material location:** `research/results/` (all figures/tables), `thesis/` Chapters 13–21 (the paper is a condensed reformatting of these, not independently drafted from scratch).

**Authorship/citation note:** the July 2026 preprint (Haroon) being extended must be cited as a preprint (not yet peer-reviewed at time of writing, per `MASTER_PROJECT_BLUEPRINT.md` Part 1) — verify its publication status again before final submission, since it may have been published or updated between this planning phase and paper submission.
