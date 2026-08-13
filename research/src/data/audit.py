"""Dataset Quality Report — the audit that decides what this project can claim.

Implements `DATASET_PLAN.md` Section 2 step 5 and `MASTER_PROJECT_BLUEPRINT.md`
Part 7's eight required measurements. Writes machine-readable JSON/CSV to
`research/data/audit/` and figures 1-3 of Part 31 to `research/results/figures/`.

Measurements, and which Part 7 item each covers:

  1. size & class distribution        item 1
  2. length distribution              item 2  — char, word, and REAL XLM-R subword
                                                counts, by label. Subword count is
                                                what the model actually sees, and is
                                                the evidence U2 is resolved from.
  3. vocabulary + log-odds            item 3
  4. source/domain cross-tab          item 4  — see the limitation below
  5. structural artefacts             item 5
  6. language mixing                  item 6
  7. duplication                      item 7  — computed by dedup.py, referenced here
  8. leakage checklist                item 8

**Known limitation, recorded rather than worked around (Part 7 item 4).** Part 7
assumes a 15-domain dataset with per-domain counts. Ax-to-Grind as published has
exactly three columns — `Sr. No.`, `News Items`, `Label` — with **no domain, source,
publisher, author or date field**. The 15 domains are described in the source
paper but are not present in the released data. Per `DATASET_PLAN.md` Section 2
step 5 ("source/domain cross-tabs *where a source field exists*") this is reported
as a real limitation, not fabricated and not silently skipped. It has two concrete
consequences: the source-leakage risk in Part 7's Risk Register **cannot be tested**
for Ax-to-Grind, and grouped splitting by publisher is impossible, so `split.py`
falls back to plain stratification. Notri-Fact does carry `Category`, so its
distribution is computed.

Usage:
    python -m research.src.data.audit
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # headless: figures are written to disk, never displayed
import matplotlib.pyplot as plt  # noqa: E402

from research.src.data.clean import clean_text  # noqa: E402
from research.src.data.validate import (  # noqa: E402
    LABEL_FAKE,
    LABEL_REAL,
    load_ax_to_grind,
    load_notri_fact,
)

_ROOT = Path(__file__).resolve().parents[3]
AUDIT_DIR = _ROOT / "research" / "data" / "audit"
FIGURES_DIR = _ROOT / "research" / "results" / "figures"

TOKENIZER_NAME = "xlm-roberta-base"

# Percentiles reported for the length distributions. The upper tail drives U2.
LENGTH_PERCENTILES = (50, 75, 90, 95, 97.5, 99, 99.5, 100)

TOP_N_VOCAB = 30
# Ignore very rare words in the log-odds ranking: a word appearing twice, both times
# in one class, gets an extreme score that is noise rather than signal.
MIN_WORD_COUNT_FOR_LOG_ODDS = 20

_LATIN = re.compile(r"[A-Za-z]")
_ARABIC_SCRIPT = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
_WORD = re.compile(r"\S+")

# Structural-artefact probes (Part 7 item 5). Urdu strings with an English gloss.
_BOILERPLATE_PROBES = {
    "read_more_ur": "مزید پڑھیں",
    "our_correspondent_ur": "نامہ نگار",
    "report_ur": "رپورٹ",
    "news_agency_ur": "خبر رساں ادارے",
    "according_to_sources_ur": "ذرائع کے مطابق",
    "file_photo_ur": "فائل فوٹو",
    "read_more_en": "read more",
    "click_here_en": "click here",
}


def _tokenizer():  # noqa: ANN202 - transformers types are heavy to import for typing
    """Load the real XLM-R tokenizer.

    Part 7 item 2 requires the *actual* tokenizer rather than a word-count proxy:
    subword count is what the model sees and what the sequence-length decision (U2)
    must be made from. A whitespace approximation would understate Urdu badly.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def _percentiles(series: pd.Series) -> dict[str, float]:
    return {f"p{p:g}": float(series.quantile(p / 100)) for p in LENGTH_PERCENTILES}


def _length_stats(series: pd.Series) -> dict[str, float]:
    return {
        "count": int(series.size),
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        **_percentiles(series),
    }


def compute_lengths(frame: pd.DataFrame, tokenizer) -> pd.DataFrame:  # noqa: ANN001
    """Per-row char / word / XLM-R subword counts on Tier-2-cleaned text."""
    cleaned = frame["text"].map(clean_text)
    subword = [
        len(ids)
        for ids in tokenizer(
            cleaned.tolist(), add_special_tokens=True, truncation=False
        )["input_ids"]
    ]
    return pd.DataFrame(
        {
            "row_id": frame["row_id"].to_numpy(),
            "label": frame["label"].to_numpy(),
            "clean_text": cleaned.to_numpy(),
            "n_chars": cleaned.map(len).to_numpy(),
            "n_words": cleaned.map(lambda s: len(_WORD.findall(s))).to_numpy(),
            "n_subwords": subword,
        }
    )


def size_and_class_distribution(lengths: pd.DataFrame) -> dict[str, object]:
    counts = lengths["label"].value_counts().to_dict()
    total = int(sum(counts.values()))
    return {
        "n_rows": total,
        "class_counts": {k: int(v) for k, v in counts.items()},
        "class_balance": {k: round(v / total, 4) for k, v in counts.items()},
    }


def length_distribution(lengths: pd.DataFrame) -> dict[str, object]:
    """Length stats overall and by label — the direct check for the confound."""
    out: dict[str, object] = {}
    for metric in ("n_chars", "n_words", "n_subwords"):
        by_label = {
            label: _length_stats(lengths.loc[lengths["label"] == label, metric])
            for label in sorted(lengths["label"].unique())
        }
        overall = _length_stats(lengths[metric])
        fake_median = by_label.get(LABEL_FAKE, {}).get("p50", 0.0)
        real_median = by_label.get(LABEL_REAL, {}).get("p50", 0.0)
        out[metric] = {
            "overall": overall,
            "by_label": by_label,
            "fake_minus_real_median": round(fake_median - real_median, 2),
            "median_ratio_fake_over_real": (
                round(fake_median / real_median, 3) if real_median else None
            ),
        }
    return out


def _log_odds_ratio(
    fake_counts: Counter[str], real_counts: Counter[str], min_count: int
) -> list[dict[str, object]]:
    """Log-odds ratio with add-one smoothing, per Part 7 item 3.

    Positive scores lean fake, negative lean real. Rare words are excluded because
    an add-one-smoothed ratio on a count of 2 is dominated by the smoothing term.
    """
    fake_total = sum(fake_counts.values())
    real_total = sum(real_counts.values())
    scores = []
    for word in set(fake_counts) | set(real_counts):
        f = fake_counts.get(word, 0)
        r = real_counts.get(word, 0)
        if f + r < min_count:
            continue
        odds = ((f + 1) / (fake_total + 1)) / ((r + 1) / (real_total + 1))
        scores.append(
            {
                "word": word,
                "log_odds": round(math.log(odds), 4),
                "fake_count": f,
                "real_count": r,
            }
        )
    scores.sort(key=lambda d: d["log_odds"])  # type: ignore[arg-type,return-value]
    return scores


def vocabulary_analysis(lengths: pd.DataFrame) -> dict[str, object]:
    fake_counts: Counter[str] = Counter()
    real_counts: Counter[str] = Counter()
    for text, label in zip(lengths["clean_text"], lengths["label"]):
        target = fake_counts if label == LABEL_FAKE else real_counts
        target.update(_WORD.findall(text))

    scores = _log_odds_ratio(fake_counts, real_counts, MIN_WORD_COUNT_FOR_LOG_ODDS)
    return {
        "vocab_size_fake": len(fake_counts),
        "vocab_size_real": len(real_counts),
        "top_words_fake": [
            {"word": w, "count": c} for w, c in fake_counts.most_common(TOP_N_VOCAB)
        ],
        "top_words_real": [
            {"word": w, "count": c} for w, c in real_counts.most_common(TOP_N_VOCAB)
        ],
        "min_count_for_log_odds": MIN_WORD_COUNT_FOR_LOG_ODDS,
        "most_fake_associated": list(reversed(scores[-TOP_N_VOCAB:])),
        "most_real_associated": scores[:TOP_N_VOCAB],
    }


def source_domain_analysis(frame: pd.DataFrame, dataset: str) -> dict[str, object]:
    """Part 7 item 4. Reports absence honestly instead of inventing a field."""
    candidates = ("Category", "Source", "Publisher", "Domain", "Author", "Date")
    present = [c for c in candidates if c in frame.columns]

    if not present:
        return {
            "has_source_or_domain_field": False,
            "columns_available": list(frame.columns),
            "limitation": (
                f"{dataset} as published carries no domain, source, publisher, author "
                "or date field. Part 7 item 4's source-label cross-tab CANNOT be "
                "computed, so the source-leakage risk in the Risk Register is "
                "UNTESTABLE for this dataset — not absent, untested. Grouped splitting "
                "by publisher is also impossible, so split.py uses plain stratification."
            ),
        }

    out: dict[str, object] = {"has_source_or_domain_field": True, "fields": present}
    for column in present:
        if column == "Date":
            out["date_field_present"] = True
            continue
        cross = pd.crosstab(frame[column], frame["label"])
        out[f"{column}_distribution"] = {
            str(k): int(v) for k, v in frame[column].value_counts().items()
        }
        out[f"{column}_by_label"] = {
            str(idx): {str(c): int(v) for c, v in row.items()}
            for idx, row in cross.iterrows()
        }
        # A category that is ~entirely one label is a trivially predictive shortcut.
        skew = {}
        for idx, row in cross.iterrows():
            total = int(row.sum())
            if total >= 20:
                dominant = float(row.max() / total)
                if dominant >= 0.9:
                    skew[str(idx)] = round(dominant, 3)
        out[f"{column}_label_skew_over_90pct"] = skew
    return out


def structural_artifacts(lengths: pd.DataFrame) -> dict[str, object]:
    """Part 7 item 5: boilerplate probes and punctuation density by label."""
    out: dict[str, object] = {"boilerplate": {}}
    for name, probe in _BOILERPLATE_PROBES.items():
        by_label = {}
        for label in sorted(lengths["label"].unique()):
            subset = lengths.loc[lengths["label"] == label, "clean_text"]
            hits = int(subset.str.contains(re.escape(probe), regex=True).sum())
            by_label[label] = {"count": hits, "rate": round(hits / len(subset), 5)}
        out["boilerplate"][name] = {"probe": probe, "by_label": by_label}

    punctuation = {}
    for label in sorted(lengths["label"].unique()):
        subset = lengths.loc[lengths["label"] == label]
        density = (
            subset["clean_text"].map(lambda s: sum(1 for c in s if c in "۔،؟!\"'()[]{}:;-"))
            / subset["n_chars"].clip(lower=1)
        )
        punctuation[label] = {
            "mean_punctuation_density": round(float(density.mean()), 5),
            "median_punctuation_density": round(float(density.median()), 5),
        }
    out["punctuation_density_by_label"] = punctuation

    # Surface-form inventory: which non-alphanumeric characters exist at all. A
    # corpus with none has been punctuation-stripped before release, which is a
    # property of the *distribution*, not of its articles.
    counter: Counter[str] = Counter()
    for text in lengths["clean_text"]:
        counter.update(ch for ch in text if not ch.isalnum() and not ch.isspace())
    out["non_alphanumeric_inventory"] = {
        "distinct_characters": len(counter),
        "total_occurrences": int(sum(counter.values())),
        "top_20": [{"char": ch, "count": n} for ch, n in counter.most_common(20)],
    }
    return out


def language_mixing(lengths: pd.DataFrame) -> dict[str, object]:
    """Part 7 item 6: Latin-script (code-mixed English) proportion by label."""
    out = {}
    for label in sorted(lengths["label"].unique()):
        subset = lengths.loc[lengths["label"] == label, "clean_text"]
        latin_share = subset.map(
            lambda s: len(_LATIN.findall(s)) / max(len(s), 1)
        )
        arabic_share = subset.map(
            lambda s: len(_ARABIC_SCRIPT.findall(s)) / max(len(s), 1)
        )
        out[label] = {
            "mean_latin_char_share": round(float(latin_share.mean()), 5),
            "mean_arabic_char_share": round(float(arabic_share.mean()), 5),
            "rows_with_any_latin": int((latin_share > 0).sum()),
            "rows_over_10pct_latin": int((latin_share > 0.10).sum()),
        }
    return out


def leakage_checklist(frame: pd.DataFrame, dataset: str) -> dict[str, object]:
    """Part 7 item 8."""
    return {
        "has_temporal_field": "Date" in frame.columns,
        "has_author_or_byline_field": any(
            c in frame.columns for c in ("Author", "Byline", "Reporter")
        ),
        "note": (
            f"{dataset}: no temporal field, so the temporal-holdout robustness check "
            "in blueprint Part 9 is not possible."
            if "Date" not in frame.columns
            else f"{dataset}: a Date field exists; a temporal holdout is feasible and "
            "is a strongly-recommended additional check (blueprint Part 9)."
        ),
    }


# --- Figures (Part 31 items 1-3) ----------------------------------------------

_PALETTE = {LABEL_FAKE: "#c2410c", LABEL_REAL: "#0e7490"}


# Fonts carrying Arabic-script glyphs, in preference order. matplotlib's default
# (DejaVu Sans) has none, and renders Urdu as empty "tofu" boxes.
_URDU_FONT_CANDIDATES = ("Arial", "Tahoma", "Segoe UI", "Nirmala UI", "Noto Naskh Arabic")


def _urdu_font() -> str | None:
    """First locally-available font with Arabic-script coverage, if any."""
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    return next((name for name in _URDU_FONT_CANDIDATES if name in available), None)


def shape_urdu(text: str) -> str:
    """Render Urdu for matplotlib: join letter forms, then apply the bidi algorithm.

    matplotlib has no Arabic shaping or bidirectional layout. Without this, Urdu
    appears as isolated, unjoined letters in logical (left-to-right) order — that is,
    wrong, and obviously so to any Urdu reader examining the thesis.

    `arabic-reshaper` and `python-bidi` are pulled in specifically for this: figure 3
    is a REQUIRED thesis figure (blueprint Part 31 item 3) in a project whose whole
    subject is Urdu, so legible Urdu is a stated need rather than a convenience
    (CLAUDE.md rule 16). Both are small and pure-Python.

    Falls back to the raw string if either package is unavailable, so the audit still
    runs — a slightly wrong figure beats a crashed pipeline, and the canonical
    strings are always in the JSON report regardless.
    """
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display

        return get_display(arabic_reshaper.reshape(text))
    except Exception:  # noqa: BLE001 - cosmetic only; never fail the audit over a label
        return text


def _style(ax) -> None:  # noqa: ANN001
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def figure_1_size_and_class(summaries: dict[str, dict], destination: Path) -> Path:
    """Part 31 figure 1: dataset size and class distribution, both datasets."""
    datasets = list(summaries)
    labels = [LABEL_FAKE, LABEL_REAL]
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    width = 0.36
    positions = range(len(datasets))

    for i, label in enumerate(labels):
        values = [summaries[d]["size_and_class"]["class_counts"].get(label, 0) for d in datasets]
        offsets = [p + (i - 0.5) * width for p in positions]
        bars = ax.bar(offsets, values, width, label=label, color=_PALETTE[label])
        ax.bar_label(bars, fmt="%d", fontsize=8, padding=2)

    ax.set_xticks(list(positions))
    ax.set_xticklabels([d.replace("_", "-") for d in datasets])
    ax.set_ylabel("articles")
    ax.set_title("Figure 1 — Dataset size and class distribution", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)
    return destination


def figure_2_length_by_label(
    lengths_by_dataset: dict[str, pd.DataFrame], destination: Path
) -> Path:
    """Part 31 figure 2: length distribution by label — visualises the confound.

    Plotted on a log x-axis and clipped at p99. The distribution is extremely
    right-skewed (max ~30x the median), so a linear axis would render every
    distribution as a single spike against a long empty tail and show nothing.
    """
    datasets = list(lengths_by_dataset)
    fig, axes = plt.subplots(1, len(datasets), figsize=(11, 4.2), dpi=150, sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, name in zip(axes, datasets):
        frame = lengths_by_dataset[name]
        upper = frame["n_subwords"].quantile(0.99)
        lower = max(1, int(frame["n_subwords"].min()))
        # Log-spaced bins, matching the log x-axis. Linear bins render as a few
        # enormous bars at the left end and hide the shape entirely.
        bins = np.logspace(np.log10(lower), np.log10(max(upper, lower + 1)), 60)
        for label in (LABEL_FAKE, LABEL_REAL):
            values = frame.loc[frame["label"] == label, "n_subwords"].clip(upper=upper)
            ax.hist(
                values,
                bins=bins,
                alpha=0.55,
                label=label,
                color=_PALETTE[label],
                edgecolor="none",
            )
            ax.axvline(
                values.median(), color=_PALETTE[label], linestyle="--", linewidth=1.3
            )
        ax.set_xscale("log")
        ax.set_xlabel("XLM-R subword tokens (log scale, clipped at p99)")
        ax.set_title(name.replace("_", "-"), fontsize=10, loc="left")
        _style(ax)

    axes[0].set_ylabel("articles")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Figure 2 — Length distribution by label (dashed = median)",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)
    return destination


def figure_3_log_odds(vocab: dict[str, object], dataset: str, destination: Path) -> Path:
    """Part 31 figure 3: words most associated with each class."""
    fake = list(vocab["most_fake_associated"])[:12][::-1]  # type: ignore[index]
    real = list(vocab["most_real_associated"])[:12]  # type: ignore[index]

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=150)
    entries = real + fake
    words = [e["word"] for e in entries]
    scores = [e["log_odds"] for e in entries]
    colours = [_PALETTE[LABEL_REAL] if s < 0 else _PALETTE[LABEL_FAKE] for s in scores]

    ax.barh(range(len(entries)), scores, color=colours)
    ax.set_yticks(range(len(entries)))
    # Properly shaped + bidi-ordered Urdu, in a font that actually has the glyphs.
    # The JSON report carries the canonical unshaped strings.
    font = _urdu_font()
    label_kwargs = {"fontsize": 9}
    if font:
        label_kwargs["fontfamily"] = font
    ax.set_yticklabels([shape_urdu(w) for w in words], **label_kwargs)
    ax.axvline(0, color="#334155", linewidth=0.9)
    ax.set_xlabel("← more associated with real      log-odds      more associated with fake →")
    ax.set_title(
        f"Figure 3 — Words most associated with each class ({dataset.replace('_', '-')})",
        loc="left",
        fontsize=11,
    )
    _style(ax)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)
    return destination


def audit_dataset(frame: pd.DataFrame, dataset: str, tokenizer) -> tuple[dict, pd.DataFrame]:  # noqa: ANN001
    lengths = compute_lengths(frame, tokenizer)
    summary = {
        "dataset": dataset,
        "tokenizer": TOKENIZER_NAME,
        "size_and_class": size_and_class_distribution(lengths),
        "length_distribution": length_distribution(lengths),
        "vocabulary": vocabulary_analysis(lengths),
        "source_domain": source_domain_analysis(frame, dataset),
        "structural_artifacts": structural_artifacts(lengths),
        "language_mixing": language_mixing(lengths),
        "leakage_checklist": leakage_checklist(frame, dataset),
    }
    return summary, lengths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer {TOKENIZER_NAME} ...")
    tokenizer = _tokenizer()

    summaries: dict[str, dict] = {}
    lengths_by_dataset: dict[str, pd.DataFrame] = {}

    for name, loader in (("ax_to_grind", load_ax_to_grind), ("notri_fact", load_notri_fact)):
        print(f"Auditing {name} ...")
        frame, _ = loader()
        summary, lengths = audit_dataset(frame, name, tokenizer)
        summaries[name] = summary
        lengths_by_dataset[name] = lengths

        lengths[["row_id", "label", "n_chars", "n_words", "n_subwords"]].to_csv(
            AUDIT_DIR / f"{name}_lengths.csv", index=False
        )

    # Cross-dataset comparison: the domain- and length-shift confound on RQ2.
    axg = summaries["ax_to_grind"]["length_distribution"]["n_subwords"]["overall"]
    ntf = summaries["notri_fact"]["length_distribution"]["n_subwords"]["overall"]
    axg_surface = summaries["ax_to_grind"]["structural_artifacts"]["non_alphanumeric_inventory"]
    ntf_surface = summaries["notri_fact"]["structural_artifacts"]["non_alphanumeric_inventory"]

    comparison = {
        "median_subwords_ax_to_grind": axg["p50"],
        "median_subwords_notri_fact": ntf["p50"],
        "median_ratio_notri_over_ax": round(ntf["p50"] / axg["p50"], 3) if axg["p50"] else None,
        "surface_form_comparability": {
            "ax_to_grind_distinct_non_alphanumeric": axg_surface["distinct_characters"],
            "notri_fact_distinct_non_alphanumeric": ntf_surface["distinct_characters"],
            "warning": (
                "Notri-Fact contains NO non-alphanumeric characters at all — not one "
                "Urdu full stop, comma or question mark — whereas Ax-to-Grind retains a "
                "full punctuation inventory. Notri-Fact was evidently punctuation-"
                "stripped before release. This is a THIRD cross-dataset confound "
                "alongside domain shift and length shift, and a consequential one: "
                "Ax-to-Grind's punctuation density itself differs by label (real "
                "~0.0147 vs fake ~0.0102), so a model trained on it may learn a "
                "punctuation-linked cue that is SYSTEMATICALLY ABSENT at cross-dataset "
                "test time. A cross-dataset F1 collapse could therefore reflect this "
                "surface-form mismatch rather than a failure to learn genuine "
                "linguistic signal. Whether to neutralise it — e.g. by stripping "
                "punctuation from both corpora — is a research-methodology change "
                "requiring a DECISION_REGISTER.md entry before it is made "
                "(CLAUDE.md rule 3); it is NOT decided here."
            ),
        },
        "domain_shift_warning": (
            "Ax-to-Grind carries no domain field, but its source paper describes 15 "
            "domains including politics, religion and health. Notri-Fact's Category "
            "field is dominated by Sports/Entertainment/Business & Economics. The two "
            "corpora therefore differ in BOTH topical domain and article length. A "
            "cross-dataset F1 drop (RQ2) is consequently NOT attributable to shortcut "
            "learning alone — domain shift and length shift are confounded with it. "
            "Any RQ2 claim must control for, or explicitly acknowledge, this."
        ),
    }

    report = {
        "tokenizer": TOKENIZER_NAME,
        "datasets": summaries,
        "cross_dataset_comparison": comparison,
        "duplication": "see research/data/audit/duplication_report.json (dedup.py)",
    }
    report_path = AUDIT_DIR / "dataset_quality_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {report_path}")

    f1 = figure_1_size_and_class(summaries, FIGURES_DIR / "fig01_dataset_size_class.png")
    f2 = figure_2_length_by_label(
        lengths_by_dataset, FIGURES_DIR / "fig02_length_by_label.png"
    )
    f3 = figure_3_log_odds(
        summaries["ax_to_grind"]["vocabulary"],
        "ax_to_grind",
        FIGURES_DIR / "fig03_log_odds_ax_to_grind.png",
    )
    for path in (f1, f2, f3):
        print(f"Wrote {path}")

    print("\n=== Subword length percentiles (evidence for U2) ===")
    for name in summaries:
        stats = summaries[name]["length_distribution"]["n_subwords"]["overall"]
        line = "  ".join(
            f"{k}={stats[k]:.0f}" for k in ("p50", "p90", "p95", "p99", "p99.5", "p100")
        )
        print(f"  {name:<14} {line}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
