"""Tier-2 text cleaning — the single source of truth for both training and serving.

Implements `MASTER_PROJECT_BLUEPRINT.md` Part 8's Tier 2. This module is imported
directly by `backend/app/ml/preprocess.py` at Milestone 6, which must call these
functions rather than reimplement them (`ML_SPECIFICATION.md` Section 9, `CLAUDE.md`
rule 4). `test_ml_sanity.py::test_preprocessing_parity` enforces that.

**Pure standard library, by constraint, not by preference** (`DECISION_REGISTER.md`
E18). The backend Docker image copies this one file out of `research/` and must not
drag `pandas`/`torch` into the serving image with it. Only `re`, `unicodedata` and
`html` are used. Do not add a third-party import here, however convenient — it will
break the deployment build.

**Design rule: normalise, don't delete.** URLs and emoji become placeholder tokens
rather than disappearing, so their *presence* stays available to the model as a
feature. `MASTER_PROJECT_BLUEPRINT.md` Part 8 is explicit that over-cleaning risks
removing genuine signal, which matters especially here — a project whose research
question is "what shortcuts does the model exploit?" must not quietly destroy the
evidence before measuring it.

**Deliberately NOT done at this tier:** boilerplate/byline stripping and any length
capping. Both are Part 8 "explicit non-decisions" pending audit evidence; length
capping in particular would pre-empt RQ3's measurement of the undisturbed length
confound, which is the thesis's central subject.
"""

from __future__ import annotations

import html
import re
import unicodedata

__all__ = [
    "URL_TOKEN",
    "EMOJI_TOKEN",
    "NUMBER_TOKEN",
    "clean_text",
    "normalise_urdu_characters",
    "strip_html",
    "collapse_whitespace",
]

# --- Placeholder tokens -------------------------------------------------------
# Latin-script and bracketed so they cannot collide with genuine Urdu content and
# survive tokenisation as recognisable units.
URL_TOKEN = "[URL]"
EMOJI_TOKEN = "[EMOJI]"
NUMBER_TOKEN = "[NUM]"

# --- Urdu / Arabic character-variant normalisation ----------------------------
# Urdu text scraped from mixed sources routinely mixes Arabic and Urdu codepoints
# for what readers treat as the same letter. Left alone these produce distinct
# subword tokens for identical words, which is noise the model would otherwise have
# to spend capacity on — and worse, the mix can correlate with *source*, which would
# be exactly the kind of source-style shortcut this project exists to detect.
#
# Each mapping is a same-letter variant substitution, never a semantic change.
_URDU_CHARACTER_VARIANTS = {
    # yeh family -> Urdu yeh / farsi yeh
    "ي": "ی",  # ARABIC YEH -> FARSI YEH
    "ى": "ی",  # ALEF MAKSURA -> FARSI YEH
    "ے": "ے",  # YEH BARREE, kept distinct: it is a real Urdu letter
    # kaf family -> keheh
    "ك": "ک",  # ARABIC KAF -> KEHEH
    # heh family -> heh goal
    "ه": "ہ",  # ARABIC HEH -> HEH GOAL
    "ھ": "ھ",  # HEH DOACHASHMEE, kept: distinct Urdu letter (aspiration)
    # alef variants -> plain alef (hamza/madda forms normalised)
    "آ": "ا",  # ALEF WITH MADDA ABOVE -> ALEF
    "أ": "ا",  # ALEF WITH HAMZA ABOVE -> ALEF
    "إ": "ا",  # ALEF WITH HAMZA BELOW -> ALEF
    # teh marbuta -> heh goal (Urdu convention)
    "ة": "ہ",
    # kashida / tatweel is decorative elongation carrying no meaning
    "ـ": "",
}

_VARIANT_TRANSLATION = str.maketrans(_URDU_CHARACTER_VARIANTS)

# Arabic diacritics (harakat). Optional in Urdu orthography and applied
# inconsistently across sources, so their presence is a source artefact rather than
# content. U+0610-U+061A, U+064B-U+065F, U+0670, U+06D6-U+06ED.
_DIACRITICS = re.compile(
    "[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]"
)

# Zero-width and bidi control characters. Invisible, inconsistently present, and a
# frequent cause of "identical" strings comparing unequal during deduplication.
_ZERO_WIDTH = re.compile("[​-‏‪-‮⁠-⁤﻿]")

_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")

# Emoji and pictographic symbol ranges.
_EMOJI = re.compile(
    "["
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f680-\U0001f6ff"  # transport & map
    "\U0001f700-\U0001f77f"
    "\U0001f780-\U0001f7ff"
    "\U0001f800-\U0001f8ff"
    "\U0001f900-\U0001f9ff"  # supplemental symbols
    "\U0001fa00-\U0001faff"
    "\U00002600-\U000026ff"  # misc symbols
    "\U00002700-\U000027bf"  # dingbats
    "\U0001f1e6-\U0001f1ff"  # regional indicators (flags)
    "\U0000fe0f"  # variation selector-16
    "\U00002b00-\U00002bff"
    "]+",
    flags=re.UNICODE,
)


def strip_html(text: str) -> str:
    """Unescape HTML entities, then remove tags.

    Order matters: entities are unescaped first so that an escaped `&lt;b&gt;`
    becomes a real tag and is then removed. Doing it the other way round would leave
    the literal text `<b>` behind in the article.
    """
    return _HTML_TAG.sub(" ", html.unescape(text))


def normalise_urdu_characters(text: str) -> str:
    """Fold Arabic/Urdu letter variants, drop diacritics and zero-width characters."""
    text = _ZERO_WIDTH.sub("", text)
    text = text.translate(_VARIANT_TRANSLATION)
    return _DIACRITICS.sub("", text)


def collapse_whitespace(text: str) -> str:
    """Collapse every run of whitespace to a single space and strip the ends."""
    return _WHITESPACE.sub(" ", text).strip()


def clean_text(text: str, *, normalise_numbers: bool = False) -> str:
    """Apply the full Tier-2 pipeline to one article.

    This is the function `backend/app/ml/preprocess.py` calls. Training and serving
    must run byte-identical logic, so any change here changes both — which is the
    entire point of the module living in one place.

    Args:
        text: raw article text. Non-string input returns an empty string rather than
            raising, so a stray NaN from a dataframe cannot crash a training run.
        normalise_numbers: replace digit runs with ``[NUM]``. **Off by default** —
            numeric density plausibly differs by label (dates, casualty figures,
            statistics), so it is a candidate shortcut the audit should be able to
            measure. Erasing it by default would destroy that evidence before it is
            looked at. Exposed as a flag for a future RQ4-style ablation.

    Returns:
        The cleaned article, or ``""`` for empty/non-string input.

    Order of operations, and why:
        1. NFC first — composes decomposed sequences so later per-character
           substitutions see one codepoint per letter rather than base+combining.
        2. HTML unescape/strip — before URL detection, since an ``href`` in a tag
           should vanish with the tag, not become a stray ``[URL]``.
        3. URLs -> token — before emoji, as URLs may contain symbol characters.
        4. Emoji -> token.
        5. Urdu variant folding and diacritic removal.
        6. Optional number folding.
        7. Whitespace collapse last, tidying the gaps every earlier step left.
    """
    if not isinstance(text, str) or not text:
        return ""

    text = unicodedata.normalize("NFC", text)
    text = strip_html(text)
    text = _URL.sub(f" {URL_TOKEN} ", text)
    text = _EMOJI.sub(f" {EMOJI_TOKEN} ", text)
    text = normalise_urdu_characters(text)
    if normalise_numbers:
        text = re.sub(r"[\d٠-٩۰-۹]+", f" {NUMBER_TOKEN} ", text)
    return collapse_whitespace(text)
