"""Tests for Tier-2 cleaning (`research/src/data/clean.py`).

These matter more than typical unit tests: this module is the shared train/serve
preprocessing path (`ML_SPECIFICATION.md` Section 9), so a regression here silently
skews both training data and live predictions at once.

Fixtures use real Urdu script, not Latin placeholders. A cleaning bug that mangles
Arabic-script codepoints would pass every ASCII-only test ever written — which is
precisely the risk `TESTING_STRATEGY.md` calls out for `test_urdu_text_roundtrip`.
"""

from __future__ import annotations

import unicodedata

import pytest

from research.src.data.clean import (
    EMOJI_TOKEN,
    NUMBER_TOKEN,
    URL_TOKEN,
    clean_text,
    collapse_whitespace,
    normalise_urdu_characters,
    strip_html,
)

# A real Urdu sentence: "Pakistan's economic situation is improving."
URDU_SENTENCE = "پاکستان کی معاشی صورتحال بہتر ہو رہی ہے۔"
# Urdu with code-switched English and Eastern Arabic-Indic numerals.
URDU_CODE_SWITCHED = "وزیراعظم نے COVID-19 کے بارے میں ۲۰۲۱ میں بات کی۔"


# --- Guardrail: the E18 constraint --------------------------------------------


def test_clean_module_imports_only_stdlib() -> None:
    """DECISION_REGISTER.md E18: the backend image copies this file alone.

    A third-party import here breaks the Docker build at Milestone 9, far from the
    commit that caused it, so it is asserted at the source level.
    """
    import ast
    import sys
    from pathlib import Path

    import research.src.data.clean as clean_module

    tree = ast.parse(Path(clean_module.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])

    non_stdlib = imported - set(sys.stdlib_module_names) - {"__future__"}
    assert not non_stdlib, f"clean.py must stay pure-stdlib per E18, found: {non_stdlib}"


# --- Urdu round-trip -----------------------------------------------------------


def test_urdu_text_roundtrip_preserves_script() -> None:
    """Real Urdu survives cleaning without corruption."""
    result = clean_text(URDU_SENTENCE)
    assert result, "cleaning returned empty output for valid Urdu"
    arabic = sum(1 for ch in result if "؀" <= ch <= "ۿ")
    assert arabic > 20, f"only {arabic} Arabic-script characters survived: {result!r}"
    assert "?" not in result, "cleaning introduced replacement question marks"
    assert "�" not in result, "cleaning introduced U+FFFD replacement characters"


def test_urdu_code_switching_and_eastern_numerals_survive() -> None:
    """Mixed Urdu/English with Eastern Arabic-Indic digits is not mangled.

    Direct regression test for DATASET_PLAN.md Section 2 step 3's normalisation
    rules, run against genuinely mixed-script input.
    """
    result = clean_text(URDU_CODE_SWITCHED)
    assert "COVID" in result, "code-switched Latin text was destroyed"
    assert "۲۰۲۱" in result, "Eastern Arabic-Indic numerals were destroyed"
    assert "�" not in result


def test_number_folding_is_off_by_default() -> None:
    """Numeric density is a candidate shortcut the audit must be able to measure."""
    assert NUMBER_TOKEN not in clean_text(URDU_CODE_SWITCHED)
    assert NUMBER_TOKEN in clean_text(URDU_CODE_SWITCHED, normalise_numbers=True)


def test_number_folding_covers_eastern_and_western_digits() -> None:
    folded = clean_text("سال ۲۰۲۱ اور 2022 میں", normalise_numbers=True)
    assert "۲۰۲۱" not in folded
    assert "2022" not in folded
    assert folded.count(NUMBER_TOKEN) == 2


# --- Normalise, don't delete ---------------------------------------------------


def test_urls_become_a_token_not_deleted() -> None:
    """Part 8: presence of a URL stays available as a feature."""
    result = clean_text(f"خبر یہاں ہے https://example.com/article ختم")
    assert URL_TOKEN in result
    assert "example.com" not in result


def test_www_urls_are_matched_too() -> None:
    assert URL_TOKEN in clean_text("دیکھیں www.example.com/x")


def test_emoji_become_a_token_not_deleted() -> None:
    result = clean_text("یہ خبر ہے 😀🔥")
    assert EMOJI_TOKEN in result
    assert "😀" not in result


# --- HTML ----------------------------------------------------------------------


def test_html_tags_are_removed() -> None:
    assert clean_text("<p>خبر</p>") == "خبر"


def test_html_entities_are_unescaped_before_tags_are_stripped() -> None:
    """Order matters: unescape first, so an escaped tag is removed, not left as text."""
    assert "<b>" not in strip_html("&lt;b&gt;خبر&lt;/b&gt;")


def test_html_entity_becomes_real_character() -> None:
    assert "&amp;" not in clean_text("A &amp; B")
    assert "&" in clean_text("A &amp; B")


# --- Urdu character-variant normalisation --------------------------------------


@pytest.mark.parametrize(
    ("variant", "canonical"),
    [
        ("ي", "ی"),  # Arabic yeh -> Farsi yeh
        ("ى", "ی"),  # alef maksura -> Farsi yeh
        ("ك", "ک"),  # Arabic kaf -> keheh
        ("ه", "ہ"),  # Arabic heh -> heh goal
        ("ة", "ہ"),  # teh marbuta -> heh goal
        ("آ", "ا"),  # alef madda -> alef
        ("أ", "ا"),
        ("إ", "ا"),
    ],
)
def test_character_variants_fold_to_canonical(variant: str, canonical: str) -> None:
    assert normalise_urdu_characters(variant) == canonical


def test_distinct_urdu_letters_are_not_collapsed() -> None:
    """Heh doachashmee and yeh barree are real letters, not variants.

    Folding these would change words rather than normalise them — a silent
    correctness bug that would be invisible to a non-Urdu-reading developer.
    """
    assert normalise_urdu_characters("ھ") == "ھ"
    assert normalise_urdu_characters("ے") == "ے"


def test_kashida_is_removed() -> None:
    assert normalise_urdu_characters("کــتاب") == "کتاب"


def test_zero_width_characters_are_removed() -> None:
    """Invisible characters make identical strings compare unequal during dedup."""
    assert normalise_urdu_characters("کتاب‌خانہ") == "کتابخانہ"


def test_diacritics_are_removed() -> None:
    assert normalise_urdu_characters("کِتاب") == "کتاب"


# --- Unicode normalisation -----------------------------------------------------


def test_output_is_nfc_normalised() -> None:
    decomposed = unicodedata.normalize("NFD", URDU_SENTENCE)
    assert clean_text(decomposed) == clean_text(URDU_SENTENCE)


# --- Whitespace and edge cases -------------------------------------------------


def test_whitespace_is_collapsed() -> None:
    assert collapse_whitespace("  a \n\t b  ") == "a b"


def test_cleaning_is_idempotent() -> None:
    """Cleaning twice equals cleaning once.

    Matters for train/serve parity: text may pass through preprocessing more than
    once across the pipeline, and a non-idempotent step would make training and
    serving diverge depending on call count.
    """
    once = clean_text(f"<p>خبر</p> https://x.com 😀 {URDU_CODE_SWITCHED}")
    assert clean_text(once) == once


@pytest.mark.parametrize("value", ["", None, 123, [], {}])
def test_non_string_and_empty_input_returns_empty_string(value: object) -> None:
    """A stray NaN from a dataframe must not crash a training run."""
    assert clean_text(value) == ""  # type: ignore[arg-type]


def test_whitespace_only_input_returns_empty_string() -> None:
    assert clean_text("   \n\t  ") == ""
