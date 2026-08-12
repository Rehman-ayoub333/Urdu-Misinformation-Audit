# Responsible AI

**This file is the single source of truth for all disclaimer and confidence-language copy in
this project** (`CLAUDE.md` rule 14). Every place that shows disclaimer text — frontend
components, `docs/model_card.md`, `README.md` — imports or quotes this file. It is never
retyped, paraphrased, or re-worded per surface. If the wording needs to change, it changes
here and everywhere else follows.

## Standing disclaimer

Copied verbatim from `MASTER_PROJECT_BLUEPRINT.md` Part 16's "Standing disclaimer text"
block. Used verbatim across UI, README, and model card:

> This system provides an AI-assisted analysis of linguistic patterns associated with
> misinformation in the datasets it was trained on. It does not verify facts, and a
> prediction is not equivalent to a determination of truth or falsehood. Always consult
> multiple credible sources, especially for political, religious, or health-related claims.

This text is shown **persistently**, not once — it is a MUST-HAVE product requirement
(`MASTER_PROJECT_BLUEPRINT.md` Part 17), and `backend/tests/test_analyze.py` asserts literal
string equality between the API's `disclaimer` field and this text
(`TESTING_STRATEGY.md` Section 1).

## Confidence language

Confidence is always labelled **"model confidence"** — a measure of the model's certainty in
its own pattern-match. It is **never** labelled "probability the article is false" or any
equivalent. This distinction is stated in the UI, the model card, and the thesis's
Discussion / Ethical Considerations chapter (`MASTER_PROJECT_BLUEPRINT.md` Part 16).

## Explanation language

Explanations are shown as **"these text features contributed most strongly to the model's
prediction"**, with the top-attributed spans highlighted. They are never phrased as "this
proves the article is fake" or any variant of that claim
(`MASTER_PROJECT_BLUEPRINT.md` Part 15).

## Known limitations, disclosed rather than hidden

Sourced from `MASTER_PROJECT_BLUEPRINT.md` Part 16:

- **False positives and false negatives are both possible** and are disclosed. Because a
  false positive (real news flagged as misinformation) carries defamation and reputational
  risk if a source institution is named, the UI never displays a source-level accusation —
  only an article-level analysis.
- **Politically and religiously sensitive claims are not adjudicable by this system.** A
  standing disclaimer recommending a human fact-checker is shown for politics/religion-domain
  input, detected by a simple keyword/domain heuristic — a soft warning, never a hard gate.
- **Satire is explicitly out of scope** for reliable detection and is disclosed as a known
  failure mode.
- **Defamation:** no feature names or accuses a specific outlet or author of dishonesty.
  Source-style analysis, if implemented, is described in aggregate statistical terms only.
- **Emerging events and out-of-distribution content:** confidence may be unreliable for
  topics, events, or domains that do not resemble the training data. The cross-dataset
  generalization findings (RQ2) are the direct evidence base for this warning.

## Placeholders pending real results

Two limitation statements above depend on measurements that do not exist yet and are
deliberately left unquantified rather than filled with plausible-looking numbers
(`CLAUDE.md` rule 2):

- Training-data time range: `TBD` — recorded from the dataset audit at Milestone 2.
- Cross-dataset F1 drop cited as evidence for the out-of-distribution warning: `TBD` —
  produced by Experiment F/G at Milestone 5.
