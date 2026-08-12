# GitHub Repository Plan

Full reasoning: `MASTER_PROJECT_BLUEPRINT.md` Part 28. This document is the concrete content plan for `README.md` and repo hygiene, to be executed at `ROADMAP.md` Milestone 10.

## 1. README.md structure

1. One-paragraph problem statement + the headline empirical finding (once measured — e.g., "training on Dataset A and testing on Dataset B causes F1 to collapse from X to Y across four model families," filled in with real numbers, never a placeholder claim).
2. Architecture diagram (from `ARCHITECTURE.md` Section 1, rendered as an image or Mermaid diagram).
3. Live demo link (Vercel URL) + a screenshot/GIF of the Analyze flow.
4. Quickstart (local dev commands from `CLAUDE.md`'s quickstart section).
5. Repository structure (short version, linking to `ARCHITECTURE.md` for the full tree).
6. Key results (the in-domain vs. cross-dataset table, real numbers).
7. Links: live demo, thesis (once available), model card, dataset card, this document set (`docs/`).
8. Reproduction instructions: `research/scripts/run_full_pipeline.sh`, with an explicit "expected runtime and hardware" note (Colab T4, approximate hours) so a reader knows what they're committing to.
9. License, citation (`CITATION.cff`), contact.

Written to be legible to a hiring manager skimming for 60 seconds, not only to an academic examiner — this is the single most important piece of portfolio-facing writing in the whole project (`MASTER_PROJECT_BLUEPRINT.md` Part 33).

## 2. LICENSE

MIT, applied to all original code. **Data is never redistributed in this repository** — `research/data/raw/` is gitignored; `research/src/data/download.py` fetches from the original sources and cites them, avoiding any relicensing risk given Ax-to-Grind's ambiguous license status (`MASTER_PROJECT_BLUEPRINT.md` Part 1).

## 3. What's committed vs. gitignored

Committed: all source code, `research/data/splits/` (row-ID indexes only), `research/data/audit/` outputs, `research/results/` (metrics, figures, error samples), `research/configs/`, all `docs/`. Gitignored: `research/data/raw/`, `research/data/clean/`, `research/data/processed/`, `checkpoints/` (large binaries — live on the HF Hub instead), `.env`, `node_modules/`, `.venv/`.

## 4. CITATION.cff

Added once the thesis/paper text is stable (`ROADMAP.md` Milestone 12), so the project is citable by others — not before, since citation metadata for unfinished work is premature.

## 5. Repository hygiene

Dependabot enabled (`SECURITY.md` Section 4). Branch protection on `main` requiring CI (`DEPLOYMENT_PLAN.md` Section 5) to pass before merge. Commit convention: Conventional Commits style (`feat:`, `fix:`, `research:`, `docs:`, `chore:`, `test:`, `deploy:` — matches the prefixes already used in `ROADMAP.md`'s commit points) for a legible history, useful both for collaboration hygiene and as a portfolio signal of engineering discipline.

## 6. What NOT to do

No generic "Awesome Project" badges-for-badges'-sake. No stock-photo hero image. No claim in the README that isn't backed by a real, committed artifact in `research/results/` — the README's credibility is a direct extension of `CLAUDE.md` rule 2's no-fabrication rule.
