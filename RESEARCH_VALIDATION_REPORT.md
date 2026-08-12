# Urdu Misinformation Detection System — Research Validation Report

**Prepared for:** Rehman Ayoub
**Date:** August 12, 2026
**Scope:** Phase 0 (idea validation) + Phase 1 (dataset research) + verdict and recommended project direction, per the master research brief. Architecture and implementation are deliberately out of scope until this direction is approved.

A note on method before the findings: this report is built from live web research conducted today (academic papers, dataset repositories, GitHub/Hugging Face listings, Pakistani fact-checking and digital-rights sources), not from memorized knowledge. Every non-obvious claim below is cited. Where something could not be verified, that is stated explicitly rather than smoothed over. A handful of the cited papers are from Q1–Q3 2026 — recent enough that you should spot-check the primary source yourself before quoting them in a thesis or paper, since a single research pass can occasionally misattribute a detail even when search-grounded.

---

## Phase 0 — Independent Idea Validation

### 0.1 Is the problem real?

Yes, but the evidence supports a narrower and more specific claim than "Urdu misinformation is a huge unsolved crisis." What's actually documented:

Pakistan has roughly 117 million internet users and 80 million active social-media identities as of late 2025/2026, with WhatsApp usage estimated between 50–100 million (the wide range itself signals weak measurement infrastructure in-country) (DataReportal, Digital 2026: Pakistan). Concrete, sourced harm cases exist: an 8-month cross-platform study found misinformation meaningfully shaped events in Pakistan in 2020 through emotional appeals and impersonation of credible sources (arXiv:2106.09338); a study of 227 WhatsApp groups during COVID-19 found misinformation made up 14% of over 7,000 messages; and Digital Rights Monitor documented a case where false papaya-leaf-cures-dengue advice, spread via WhatsApp/YouTube/Facebook, led to actual hospitalization. Recurring themes across sources are polio-vaccine conspiracies, COVID/anti-vax content, financial rumors, religious-minority hate speech, and politically-motivated fabrications.

The most useful finding for validating *this specific project* is a documented service gap: Pakistan has only two IFCN-certified fact-checkers (Soch Fact Check and AFP Fact Check Pakistan), and both operate predominantly in English even though the people consuming and forwarding viral misinformation are overwhelmingly Urdu speakers. A fact-checker quoted directly in Digital Rights Monitor's reporting says this explicitly — Urdu-speaking audiences "use it as a political tool... instead of a critical thinking strategy" because the fact-checks aren't reaching them in their language. Geo Fact Check (launched 2022) is one of very few outlets publishing bilingually. This is a real, sourced, English-vs-Urdu asymmetry in Pakistan's fact-checking infrastructure — a stronger and more specific justification than "Urdu NLP is under-resourced," and one you can defend in a viva without hand-waving.

One honest caveat: Pakistan has been excluded from the Reuters Institute Digital News Report for lack of research/data transparency, so there is no authoritative national survey of misinformation *perception* or *trust* to cite. Claims about scale should stay grounded in the documented harm cases and platform-usage numbers above, not in invented survey statistics.

Realistic users, in order of how well-evidenced they are: (1) small, resource-constrained Pakistani fact-checking newsrooms who are currently English-first and could use an Urdu-language triage/first-pass tool; (2) NGOs like Digital Rights Foundation running misinformation tiplines; (3) ordinary WhatsApp users, per a November 2025 LUMS participatory-design study that prototyped an intervention with low-socioeconomic-status adults; (4) researchers. No evidence supports "millions of end users waiting for this app" — that would be overclaiming.

### 0.2 What does the existing research already cover?

This is the most important finding in the whole report, and it cuts against the project as originally framed: **Urdu fake-news classification is not an under-explored niche. It is an active, fairly crowded research area with a clear trajectory from 2020 to a February 2026 paper reporting 96%+ accuracy.**

| Year | Work | Dataset | Approach | Result |
|---|---|---|---|---|
| 2020 | Amjad et al., "Bend the Truth" (J. Intelligent & Fuzzy Systems) | 900 articles | Classical ML (SVM/RF/NB/AdaBoost) | Baseline benchmark |
| 2020–21 | UrduFake@FIRE shared tasks | 900–1,600 articles | 18+ teams: TF-IDF+SVM up to BERT/RoBERTa/MuRIL | Best F1 0.679 — notably, **transformer entries underperformed a simple SGD baseline** |
| 2023 | Farooq et al., PeerJ Computer Science | 4,097 articles | Stacked classical ensemble (no deep learning) | 93.8% accuracy |
| 2024 | Ax-to-Grind Urdu (arXiv:2403.14037) | 10,083 articles, 15 domains | mBERT/XLNet/XLM-R + ensemble | Ensemble F1 0.924, accuracy 0.956 |
| 2025 | Multi-domain ensemble, *Scientific Reports* | — | ELECTRA+mBERT+XLM-R stack | Accuracy 0.914 |
| 2025 | Hook and Bait Urdu, *Scientific Reports* | 78,409 articles | Fine-tuned LLaMA-2 | Accuracy 0.978 (see caveat below) |
| Dec 2025 | Domain-adaptive pretraining, U. of Waikato (arXiv:2512.22778) | 4 combined datasets | Domain-adaptive pretraining + XLM-R/mBERT | Domain adaptation consistently helps XLM-R |
| **Feb 2026** | Feroz, Abbasi, Babar et al., *Scientific Reports* 16:7352 | New 14,178-article dataset (adds politics/religion coverage) | XLM-R + concatenated GloVe embeddings | **F1 0.956, accuracy 0.962** — current published SOTA |
| **Jul 2026** | Haroon, cross-dataset generalization study (arXiv:2607.14131) | Ax-to-Grind vs. Notri-Fact | XLM-R, in-domain vs. zero-shot cross-domain | **In-domain scores look strong; cross-dataset transfer collapses to F1 0.005** |

That last row matters more than any accuracy number in the table above it. The July 2026 paper is, by its own claim, the first systematic cross-dataset generalization study in Urdu fake-news detection, and it found that Ax-to-Grind's fake articles average 117 words versus 35 for real articles — a 3.4x length asymmetry. A model trained on Ax-to-Grind and tested on a different, length-balanced dataset (Notri-Fact) collapses to predicting "fake" for 99.7% of inputs. In plain terms: much of the field's headline 92–98% accuracy may reflect models learning "article length" rather than "is this true," and nobody had systematically checked this until mid-2026.

What this means for you: proposing "fine-tune XLM-RoBERTa on an Urdu fake-news dataset, report accuracy" in 2026 would put you in direct, unfavorable competition with a well-resourced paper that already hit 96.2% two months before you started. That specific formulation scores low on novelty. But the shortcut-learning finding above is fresh (published the same month as this report), not yet widely absorbed into the field's practice, and opens a genuinely current, defensible research angle: rigorous cross-dataset evaluation and bias/shortcut auditing of Urdu misinformation models, which almost nobody in this literature has done systematically. This is Phase 2 material, addressed below.

Also documented, and worth knowing before you scope preprocessing: multiple papers report that character n-gram features outperform word n-grams for Urdu classical baselines, and that transformer models sometimes lose to well-tuned SVM/ensemble baselines on the smaller shared-task datasets (FIRE2021) — a reason to keep a strong classical baseline in your experiment design rather than assuming transformers auto-win.

Genuinely under-explored, verified by absence in this search: Roman Urdu (Latin-script, informal social-media text) has no dedicated fake-news dataset or paper at all — every major dataset found uses formal Nastaliq-script news articles, not the informal register people actually forward on WhatsApp. Explainability/rationale generation is absent from every paper found — all are pure classifiers with no explanation layer. And the Feb 2026 SOTA paper's own authors explicitly flag, as an *unresolved* limitation, that their model "may struggle with emerging narratives" and "could misclassify satire or political dissent" as misinformation — a real, admitted risk that argues for careful product framing (see Phase 0.5/product principle below), not for ignoring it.

### 0.3 What already exists as a system or tool?

The dataset and modeling layer is crowded, as shown above. The product/deployment layer is not. A targeted search of Hugging Face Spaces found no Urdu-language fake-news detection demo at all — every notable fake-news-classifier Space is English-only. GitHub repos that do exist (`Noman-Aziz/Urdu-Fake-News-Detection`, `Bushramjad/Fake-News-Detection-in-Urdu`) are small, single-notebook, Naive-Bayes-from-scratch projects with no live demo and no recent activity. `urduhack`, the most-used Urdu NLP preprocessing library (309 stars), is TensorFlow-based and officially caps at Python 3.6–3.7 — a signal it's stale relative to 2026 tooling and shouldn't be adopted wholesale without modification.

On the human side, existing "fact-checking tools" for Pakistan (Soch Fact Check, AFP Fact Check Pakistan, Geo Fact Check, DRF's WhatsApp tipline) are editorial/manual operations, not automated classifiers. No Urdu-language automated WhatsApp fact-checking bot was found (a Hindi-language equivalent exists elsewhere, per one source, but nothing Urdu-specific).

Net competitive read: you would not be building "yet another thing that already exists with a different UI." A live, usable, Urdu-native demo — even one that's honest about its confidence limits — would fill a real, verified gap at the product layer, even though the underlying classification task itself is well-trodden at the research layer.

### 0.4 Technical feasibility

Confirmed comfortable on Google Colab's free tier (T4 GPU, 16GB VRAM, ~12.7GB system RAM, sessions up to ~12 hours). XLM-RoBERTa-base (~270M params) and mBERT (~178M params) are exactly the model classes every paper surveyed above fine-tunes on standard single-GPU academic compute — no paper in this literature needed more than that. The realistic friction points are: (1) idle-timeout disconnects requiring checkpoint-and-resume across sessions if you replicate an ensemble (e.g., Ax-to-Grind's mBERT+XLM-R+XLNet stack) rather than a single model; (2) CPU RAM (not GPU RAM) spiking if you load multiple large checkpoints simultaneously; (3) Urdu-specific tokenizer inefficiency (multilingual BPE/SentencePiece tokenizers are known to need more subword tokens per word for non-Latin scripts than monolingual tokenizers, though an Urdu-specific fertility number wasn't confirmed in this pass — worth a follow-up check if you need a precise figure for a paper). None of these are blockers; they're things to plan around in your training strategy (Phase 6, later).

### 0.5 Portfolio, scholarship, and FYP value

As originally scoped (plain binary classifier, report accuracy), this project is a below-average portfolio piece for 2026 — reviewers who know the space (and AI/ML hiring panels increasingly do skim recent arXiv) will recognize it as replicating work a February 2026 paper already did better. As a project that explicitly engages with the shortcut-learning/length-confound problem, benchmarks multiple model families honestly, tests cross-dataset generalization, and ships a working, honestly-framed demo, it becomes a genuinely strong piece: it demonstrates you can read current literature, notice a documented flaw in how a field evaluates itself, design an experiment around it, and communicate uncertainty responsibly in a deployed product. That combination (empirical rigor + shipped artifact + responsible-AI framing) is exactly what strengthens scholarship applications and FYP defensibility, more than a high accuracy number would.

### 0.6 Research and publication potential

Realistic framing: this is workshop-paper or regional-conference territory (e.g., a South Asian NLP workshop, a national CS conference, or a solid FYP thesis chapter), not a top-tier ACL/EMNLP main-track paper — the bar there requires more novel modeling contributions than a student project on borrowed compute can realistically clear in one semester. But a focused contribution ("we replicate and extend the length-confound finding across three Urdu datasets, and show which mitigation strategies restore cross-dataset performance") is a legitimate, publishable-scope contribution if executed carefully, with honestly reported (not fabricated) numbers. No promise of publication is being made here — only that the scope is realistic for it, which the original plain-classifier framing was not.

---

## Phase 0 — Scorecard

Two sets of scores are given: for the project *as originally proposed* (plain Real/Fake classifier), and for the *recommended pivot* described in Phase 2 below. The gap between them is the main takeaway of this report.

| Dimension | As originally proposed | With recommended pivot |
|---|---|---|
| Problem significance | 70/100 | 72/100 |
| Research novelty | 30/100 | 78/100 |
| Dataset feasibility | 80/100 | 80/100 |
| Technical feasibility | 90/100 | 85/100 |
| AI/ML depth | 55/100 | 82/100 |
| NLP depth | 60/100 | 78/100 |
| Portfolio value | 55/100 | 85/100 |
| Scholarship value | 55/100 | 78/100 |
| FYP suitability | 65/100 | 88/100 |
| Research potential | 35/100 | 72/100 |
| Real-world usefulness | 55/100 | 62/100 |
| **Overall** | **~55/100** | **~78/100** |

### Final verdict: **B — Build, with modifications**

The underlying problem is real and reasonably well-documented, the datasets exist and are accessible, and the compute is free-tier feasible. But the project as originally framed ("classify Urdu news real/fake with XLM-R") is not worth building in 2026 — it would be visibly redundant with a February 2026 paper that already scored 96%+ on a larger, more domain-diverse dataset, and a July 2026 paper suggests the whole benchmark tradition it belongs to may be measuring the wrong thing. The fix is not a different topic — it's a different research question layered on the same technical foundation you already had in mind. See below.

---

## Phase 1 — Dataset Landscape

Every serious candidate found, evaluated for size, labeling methodology, access, and — critically — known quality/leakage risks, since the literature review above shows these risks are not hypothetical in this specific field.

| Dataset | Size / Balance | Labeling | Access | Key risk flags |
|---|---|---|---|---|
| **Ax-to-Grind Urdu** (2024) | 10,083 articles, 15 domains, ~50/50 balanced | Expert journalists, κ=0.94, no MT | Public GitHub, no explicit license | **Severe length confound**: fake articles average 3.4x longer than real. Independently shown to cause near-total collapse (F1 0.005) on cross-dataset transfer. Must be mitigated (length control/stratified analysis), not ignored. |
| **Notri-Fact Urdu** | 13,388 articles, ~50/50 | Undocumented — no paper, uploaded to Kaggle without methodology | Kaggle, standard terms | Low provenance, but length-balanced across classes — makes it valuable specifically as a shortcut-learning stress test, not as a sole primary dataset. |
| **UFND (Feb 2026)** — Feroz, Abbasi, Babar et al. | 14,178 articles, 15 domains incl. politics/religion, ~58/42 | Manually curated, explicitly designed to fix prior domain gaps | Claimed open access by authors; **direct download link not confirmed in this pass** — check the paper's Data Availability Statement before committing to it | Newest and most domain-diverse; verify accessibility first. |
| **Hook and Bait Urdu** (2025) | 78,409 articles, 15 domains — largest available | Real from news portals; fake scraped/MT'd from fact-check sites, manually edited | Public GitHub | **Source-leakage risk** (real vs. fake come from structurally different site types) and partial machine-translation artifacts in the fake class. Reported 0.978 accuracy should be treated with real skepticism given this. |
| **Bend the Truth / UrduFake@FIRE2020–21** | 900–1,600 articles | Real verified; fake was *written to deceive* by hired journalists (not organic) | Public GitHub, citation required | Small; synthetic fake class doesn't reflect organic misinformation patterns — but it's the field's standard benchmark, useful for literature comparability. |
| **Farooq et al. 2023 (PeerJ) corpus** | 4,097 articles, 9 domains, ~60/40 | Partially translated from English, not fully re-verified | Zenodo, CC-BY 4.0 | Class imbalance; MT artifacts flagged by later papers. |
| Urdu News Dataset 1M (Kaggle/Mendeley) | 1M+ articles, 4 categories | No veracity labels | Public | Not usable for classification directly — correct use is domain-adaptive pretraining before fine-tuning on a labeled set, as the Dec 2025 Waikato paper does. |

**Explicitly ruled out:** Kishwar & Zafar's Pakistani news dataset (English, not Urdu); "Roman-Urdu-Fake-Review-Detection" on GitHub (fake product reviews, not news — misleading name); `CSALT/deepfake_detection_dataset_urdu` (audio voice-cloning data, different modality entirely).

### Recommendation

**Primary:** Ax-to-Grind Urdu, used only with explicit length-confound mitigation built into the methodology (length-stratified splits, and/or reporting performance separately by length bucket). If you can confirm a working download link for the Feb 2026 UFND dataset, it's worth a direct comparison as an alternative primary, since it's newer and more domain-diverse — but don't block on it if the link doesn't pan out.

**Secondary (cross-dataset generalization test):** Notri-Fact Urdu. Training on Ax-to-Grind and evaluating zero-shot on Notri-Fact directly replicates the July 2026 shortcut-learning study — this is your core research experiment, not a side note.

**Optional:** UrduFake@FIRE2020/21 for literature comparability (every paper in this space reports numbers on it), and Hook and Bait for a scale-up experiment once you've audited it for source leakage.

**Do not** simply concatenate all of these to inflate sample count — the brief you gave was explicit about this, and it would also actively undermine the cross-dataset generalization experiment, which depends on the datasets staying separate.

---

## Phase 2 — Recommended Research Direction (high-level)

Given the literature findings, the strongest and most current direction is a combination of **Direction A (cross-dataset generalization)** and **Direction F (dataset bias / shortcut analysis)**, with **Direction B (explainability)** as a secondary layer and **Direction D (multi-model benchmark)** kept as your experimental backbone rather than an afterthought.

Concretely, the central research question becomes: *do Urdu misinformation classifiers trained on today's public datasets learn genuine linguistic signals of misinformation, or dataset-specific shortcuts like article length or source style — and does that change what "state of the art" actually means for this task?* You would benchmark classical ML, mBERT, and XLM-RoBERTa (Direction D) on Ax-to-Grind, then evaluate zero-shot transfer to Notri-Fact (Direction A), explicitly measure and report the length-confound effect described above (Direction F), and add a lightweight, scientifically-defensible explanation layer — not raw attention weights, which the literature and interpretability research broadly warn against over-trusting, but something like SHAP or integrated gradients over the final model (Direction B, kept intentionally modest in scope).

This is deliberately not Direction E (evidence-aware retrieval with sourcing) as the core contribution — that's a substantially larger system (retrieval infrastructure, evidence-source curation) that would dilute focus and isn't well-supported by the current Urdu dataset landscape, which has no evidence/claim-alignment resource to build on. It could be a stretch/future-work feature on the demo, framed honestly as "related coverage" rather than "proof," but shouldn't be the thesis.

Product framing follows directly from the brief's own principle and from the Feb 2026 paper's admitted limitation: this should be presented as an AI-assisted misinformation *analysis and triage* tool, not a truth oracle, with explicit UI/documentation language that a high confidence score reflects the model's pattern-match to its training data, not a verified fact-check — especially given the documented satire/political-dissent misclassification risk.

---

## Phase 3 — Model Recommendation (overview)

**Primary:** XLM-RoBERTa-base. It's the model every recent paper in this literature converges on, it's within comfortable Colab free-tier budget, and using it keeps your results directly comparable to the field.

**Baselines (mandatory, not optional):** TF-IDF + Logistic Regression/SVM. The FIRE2021 shared task result — where classical baselines beat several transformer submissions — is a documented reason not to skip this; it's also a cheap, fast sanity check for the length-confound problem itself (a bag-of-words model overfitting to length is a fast way to detect the shortcut before you spend GPU hours on it).

**Secondary comparison model:** mBERT, since it's the second model the domain-adaptation and ensemble papers consistently compare against, and gives you a same-scale multilingual comparison point to XLM-R.

**Optional/stretch:** domain-adaptive pretraining on the unlabeled Urdu News 1M corpus before fine-tuning, following the December 2025 Waikato paper's approach — a legitimate, citable extension if time allows, not required for a defensible core project.

---

## Answering the brief's direct questions

**Should you build this?** Yes — but not the version you originally described. The plain classifier is redundant with a two-months-old published result. The pivot toward cross-dataset generalization and shortcut/bias analysis, using the same technical foundation (XLM-R, Urdu datasets, Colab-feasible training) you already had in mind, is current, defensible, and substantially strengthens every dimension in the scorecard above.

**What should the best version of the project be?** An Urdu misinformation *detection and analysis* system whose central research contribution is auditing whether existing Urdu fake-news models generalize across datasets or merely exploit dataset-specific shortcuts (length, source style) — benchmarked across classical ML, mBERT, and XLM-R, evaluated with cross-dataset transfer as a first-class experiment (not an afterthought), paired with a modest, defensible explanation layer, and shipped as a live, honestly-framed demo (filling the verified product-layer gap on Hugging Face Spaces) that communicates uncertainty rather than asserting truth.

This report deliberately stops here, per the brief: architecture, preprocessing pipeline, training strategy, and the rest of the phases should only proceed once you've confirmed this direction is the one you want to commit to.

---

## Sources

Full citation list compiled from both research passes; every dataset, paper, and tool referenced above has a corresponding source here.

- Ax-to-Grind Urdu dataset & paper — https://arxiv.org/abs/2403.14037
- Cross-dataset generalization / length confound study — https://arxiv.org/abs/2607.14131
- Domain-adaptive pretraining for Urdu FND (Waikato) — https://arxiv.org/abs/2512.22778
- UFND / "Verifying Urdu news authenticity," Scientific Reports 16:7352 (2026) — https://www.nature.com/articles/s41598-026-36771-0
- Heriot-Watt press release on above — https://www.hw.ac.uk/news/2026/new-ai-could-stop-fake-news-in-urdu
- Hook and Bait Urdu, Scientific Reports (2025) — https://www.nature.com/articles/s41598-025-98271-x
- Notri-Fact Urdu dataset (Kaggle) — https://www.kaggle.com/datasets/tridata/notri-fact-real-and-unreal-urdu-news
- Farooq et al., PeerJ Computer Science 9:e1353 (2023) — https://peerj.com/articles/cs-1353/ ; dataset on Zenodo — https://zenodo.org/records/7773474
- Bend the Truth dataset / FIRE shared tasks — https://github.com/MaazAmjad/Datasets-for-Urdu-news ; https://arxiv.org/pdf/2207.05144
- Multi-domain pre-trained ensemble, Scientific Reports 15:8705 — https://pmc.ncbi.nlm.nih.gov/articles/PMC11906872/
- Urdu News Dataset 1M — https://www.kaggle.com/datasets/saurabhshahane/urdu-news-dataset
- urduhack library — https://github.com/urduhack/urduhack
- Investigating misinformation dissemination in Pakistan — https://arxiv.org/abs/2106.09338
- WhatsApp COVID-19 misinformation in Pakistan study — https://www.researchgate.net/publication/350789762_Fake_News_Shared_on_WhatsApp_During_Covid-19_An_Analysis_of_Groups_and_Statuses_in_Pakistan
- Towards Misinformation Resilience in Pakistan (LUMS, 2025) — https://arxiv.org/pdf/2511.06147
- Pakistan's fledgling fact-checking industry, Digital Rights Monitor — https://digitalrightsmonitor.pk/pakistans-fledgling-fact-checking-industry-struggles-to-gain-footing/
- Soch Fact Check — https://www.sochfactcheck.com/about-us/
- Digital Rights Foundation fake news report — https://digitalrightsfoundation.pk/january-2020-drf-released-its-latest-report-on-fake-news/
- Countering Disinformation in Pakistan, International Media Support (2023) — https://www.mediasupport.org/wp-content/uploads/2023/01/Countering-Disinformation-in-Pakistan-2023.pdf
- Pakistan excluded from Reuters Digital News Report 2025 — https://asianews.network/pakistan-once-again-excluded-from-reuters-annual-digital-news-report/
- Digital 2026: Pakistan, DataReportal — https://datareportal.com/reports/digital-2026-pakistan
- Google Colab free-tier GPU specs — https://www.hivenet.com/post/google-colaboratory-gpu-complete-guide-to-free-cloud-gpu-access-and-limitations

*A note on verification: two of the most consequential citations above — the February 2026 Scientific Reports SOTA paper and the July 2026 cross-dataset/length-confound study — are recent enough that you should open and re-read the primary sources yourself before relying on them in formal academic work. Everything else has been cross-checked against multiple mentions across the two research passes.*
