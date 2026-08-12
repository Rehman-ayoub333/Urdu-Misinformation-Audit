# Thesis Plan — Artifact-to-Chapter Mapping

Chapter list and full content guidance: `MASTER_PROJECT_BLUEPRINT.md` Part 29. This document maps each chapter to the exact repository artifact(s) that feed it, so writing the thesis becomes assembly-and-analysis of real, already-produced evidence rather than starting from a blank page.

| Chapter | Fed by |
|---|---|
| 1. Abstract | Written last, after Chapter 18 (Results) is final |
| 2. Introduction | `PROJECT_SPECIFICATION.md`, `RESEARCH_VALIDATION_REPORT.md` Phase 0 |
| 3. Problem Statement | `RESEARCH_VALIDATION_REPORT.md` Section 0.1 |
| 4. Motivation | `RESEARCH_VALIDATION_REPORT.md` Sections 0.1/0.3 (Pakistan misinformation landscape, English/Urdu fact-checking gap) |
| 5. Research Questions | `MASTER_PROJECT_BLUEPRINT.md` Part 4, verbatim RQ1–RQ5 |
| 6. Objectives | Derived from Part 4 + `PROJECT_SPECIFICATION.md` Section 1 |
| 7. Literature Review | `RESEARCH_VALIDATION_REPORT.md` Section 0.2 + `MASTER_PROJECT_BLUEPRINT.md` Part 1's verified-claims table, expanded with the full paper list |
| 8. Research Gap | `MASTER_PROJECT_BLUEPRINT.md` Part 2, verbatim |
| 9. Dataset | `DATASET_PLAN.md` Section 1, `docs/dataset_card.md` |
| 10. Dataset Audit | `research/data/audit/*` (real output, Milestone 2) |
| 11. Methodology | `DATASET_PLAN.md` + `EXPERIMENT_PLAN.md` + `MASTER_PROJECT_BLUEPRINT.md` Part 13 |
| 12. Models | `MASTER_PROJECT_BLUEPRINT.md` Part 10, `DECISION_REGISTER.md` R4 |
| 13. Experimental Design | `EXPERIMENT_PLAN.md` Sections 1–2 |
| 14. Shortcut Analysis | `research/results/metrics/H_*`, `I_*` + figures 9–10 (`MASTER_PROJECT_BLUEPRINT.md` Part 31), Milestone 5 output |
| 15. Cross-Dataset Evaluation | `research/results/metrics/F_*`, `G_*` + figure 7, Milestone 5 output |
| 16. Mitigation | `research/results/metrics/L_*`, `M_*` + figure 11, Milestone 5.5 output |
| 17. Explainability | `research/results/error_samples/`, IG output, figure 13, Milestone 5 output |
| 18. Results | Consolidated `research/results/metrics/` across all REQUIRED + EXTENSIONS experiments |
| 19. Error Analysis | `research/results/error_samples/`, figure 14 |
| 20. Discussion | Written analysis connecting Chapters 14–19 to RQ1–RQ5's support/reject criteria (`MASTER_PROJECT_BLUEPRINT.md` Part 4) — this is genuine authorial synthesis, not artifact assembly |
| 21. Limitations | `MASTER_PROJECT_BLUEPRINT.md` Part 29's explicit list (single-annotator error analysis, Notri-Fact provenance, no Roman Urdu coverage, etc.) — must not be watered down |
| 22. Ethical Considerations | `MASTER_PROJECT_BLUEPRINT.md` Part 16, `docs/responsible_ai.md` |
| 23. System Implementation | `ARCHITECTURE.md`, `FRONTEND_SPECIFICATION.md`, `BACKEND_SPECIFICATION.md` (summarized — full detail stays in the repo docs, not duplicated at length) |
| 24. Deployment | `DEPLOYMENT_PLAN.md`, actual Milestone 9 outcome |
| 25. Conclusion | Synthesis of Chapter 20 |
| 26. Future Work | `MASTER_PROJECT_BLUEPRINT.md` Part 12's FUTURE WORK list + anything genuinely surfaced during implementation |
| 27. References | Full bibliography from `RESEARCH_VALIDATION_REPORT.md` + `MASTER_PROJECT_BLUEPRINT.md` Part 1's sources, in the citation style the institution requires |
| 28. Appendices | Full hyperparameter tables (`research/configs/`), extended error-analysis samples, additional figures not in the main body |

**Rule:** no chapter above is drafted with placeholder numbers. Chapters 10, 13–19 specifically must wait for their feeding milestone (`ROADMAP.md`) to produce real artifacts — `CLAUDE.md` rule 2 applies to thesis content exactly as it applies to code and the model card.
