"""Structural tests for `research/notebooks/06_milestone5_combined.ipynb`.

A notebook is a committed artefact that nothing else executes in CI, so its
invariants rot silently — `DECISION_REGISTER.md` M5-6 is exactly that failure in a
test rather than a notebook. These assert the properties that would be expensive
to discover live on Kaggle, an hour into a session:

* every code cell parses as Python;
* the stale-notebook guard reads **this** notebook's filename, not the one it was
  copied from (a guard pointed at the wrong file silently never fires);
* both expensive sections are gated, and gated on *independent* flags;
* each guarded cell also passes the CLI's own `--confirm-real-run` (M5-6);
* the ungated sections cannot reach a training runner;
* Experiment I runs its transformer half only, since the classical half is
  already committed;
* the environment cells are byte-identical to notebook 05's, which is the whole
  reason they were copied rather than adapted.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = _ROOT / "research" / "notebooks" / "06_milestone5_combined.ipynb"
NOTEBOOK_05 = _ROOT / "research" / "notebooks" / "05_cross_dataset_eval.ipynb"


def _cells(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["cells"]


def _sources(path: Path) -> list[str]:
    return ["".join(cell["source"]) for cell in _cells(path)]


@pytest.fixture(scope="module")
def sources() -> list[str]:
    return _sources(NOTEBOOK)


@pytest.fixture(scope="module")
def joined(sources: list[str]) -> str:
    return "\n".join(sources)


def test_notebook_exists_and_is_valid_nbformat() -> None:
    document = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert document["nbformat"] == 4
    assert document["cells"], "notebook has no cells"
    for index, cell in enumerate(document["cells"]):
        assert cell["cell_type"] in ("code", "markdown"), index
        if cell["cell_type"] == "code":
            assert "outputs" in cell and "execution_count" in cell, index


def test_every_code_cell_parses_as_python() -> None:
    """A syntax error would only surface when that cell is reached, live."""
    for index, cell in enumerate(_cells(NOTEBOOK)):
        if cell["cell_type"] != "code":
            continue
        body = "".join(cell["source"])
        # `!pip ...` is IPython syntax, not Python.
        stripped = "\n".join(
            "pass" if line.lstrip().startswith("!") else line for line in body.splitlines()
        )
        try:
            ast.parse(stripped)
        except SyntaxError as exc:  # pragma: no cover - the assert reports it
            pytest.fail(f"cell {index}: SyntaxError line {exc.lineno}: {exc.msg}")


# --------------------------------------------------------------------------
# the stale-notebook guard
# --------------------------------------------------------------------------


def test_stale_guard_reads_this_notebook(sources: list[str]) -> None:
    """A guard copied from 05 and left pointing at 05 can never fire."""
    guards = [body for body in sources if "STALE NOTEBOOK" in body]
    assert len(guards) == 1, f"expected exactly one stale guard, found {len(guards)}"
    assert "06_milestone5_combined.ipynb" in guards[0]
    assert "05_cross_dataset_eval.ipynb" not in guards[0]


def test_notebook_revision_is_findable_by_the_guards_own_regex(sources: list[str]) -> None:
    """Asserted with the guard's own pattern, not an approximation of it."""
    marker = re.compile(r"NOTEBOOK" + r"_REVISION\s*=\s*(\d+)")
    found = [int(m.group(1)) for body in sources for m in marker.finditer(body)]
    assert found, "no NOTEBOOK_REVISION the stale guard could find"


# --------------------------------------------------------------------------
# the expensive sections are guarded
# --------------------------------------------------------------------------

GUARDS = {"CONFIRM_Q": "punctuation_ablation", "CONFIRM_I": "length_ablation"}


@pytest.mark.parametrize("flag", sorted(GUARDS))
def test_confirm_flag_is_reset_in_restore_state(flag: str, joined: str) -> None:
    """Recovering from a crash must never leave expensive work armed."""
    assert f"{flag} = False" in joined


@pytest.mark.parametrize("flag", sorted(GUARDS))
def test_confirm_flag_is_read_with_a_safe_default(flag: str, joined: str) -> None:
    """A bare `if CONFIRM_X:` raises NameError if restore-state was not run —
    which fails open in the worst case, since the traceback stops the cell but
    tells the reader nothing about why."""
    assert f'globals().get("{flag}", False)' in joined


@pytest.mark.parametrize(("flag", "module"), sorted(GUARDS.items()))
def test_guarded_cell_invokes_its_runner_and_the_cli_guard(
    flag: str, module: str, sources: list[str]
) -> None:
    """M5-6: the notebook flag and the CLI flag are both required, on purpose."""
    cells = [body for body in sources if f'globals().get("{flag}"' in body]
    assert cells, f"no cell guards on {flag}"
    body = cells[0]
    assert module in body, f"{flag} cell does not invoke {module}"
    assert "--confirm-real-run" in body, f"{flag} cell does not pass the CLI guard"


def test_the_two_flags_are_independent(joined: str) -> None:
    """One flag would mean confirming Q silently arms I as well."""
    assert "CONFIRM_Q" != "CONFIRM_I"
    assert joined.count("CONFIRM_Q = False") >= 1
    assert joined.count("CONFIRM_I = False") >= 1
    # Neither may be assigned from the other.
    assert "CONFIRM_I = CONFIRM_Q" not in joined
    assert "CONFIRM_Q = CONFIRM_I" not in joined


@pytest.mark.parametrize("module", sorted(set(GUARDS.values())))
def test_no_ungated_cell_can_reach_a_training_runner(module: str) -> None:
    """The whole point of the guards. An ungated Run All must not train."""
    for index, cell in enumerate(_cells(NOTEBOOK)):
        if cell["cell_type"] != "code":
            continue
        body = "".join(cell["source"])
        if module in body and "globals().get" not in body:
            pytest.fail(f"cell {index} invokes {module} with no confirm guard")


def test_training_sections_refuse_without_a_gpu(joined: str) -> None:
    """15 fine-tunes on CPU cannot finish inside Kaggle's 12-hour session cap."""
    assert 'globals().get("HAS_GPU", False)' in joined
    assert "ALLOW_CPU_TRAINING" in joined


# --------------------------------------------------------------------------
# scope
# --------------------------------------------------------------------------


def test_experiment_i_runs_the_transformer_half_only(sources: list[str]) -> None:
    """I's classical half is already committed, computed locally on the pinned
    3.12. Re-running it here would overwrite four committed results with numbers
    from a different environment, for no benefit."""
    cells = [b for b in sources if "length_ablation" in b and "subprocess" in b]
    assert cells, "no Experiment I cell"
    assert '"--models", "transformer"' in cells[0]


def test_backfill_sections_are_present(joined: str) -> None:
    assert "evaluate_checkpoint" in joined, "no C/D backfill"
    assert "run_cross_dataset" in joined, "no F backfill"


def test_backfill_is_verified_against_git(joined: str) -> None:
    """M5-2's lesson: a backfill can be wrong with every headline number intact,
    so the check walks the full JSON tree against the committed copy."""
    assert 'git", "show"' in joined
    assert "numeric_leaves" in joined
    assert joined.count("def numeric_leaves") >= 2, (
        "each verify cell should define its own helper — the sections are "
        "independently re-runnable and must not NameError after a restart"
    )


def test_results_are_pushed_and_verified(joined: str) -> None:
    """M4-6: a run is not done until its results are off the session disk."""
    assert "push_result_files" in joined
    assert "verify_results_uploaded" in joined
    assert "predictions.jsonl" in joined, "packaging must include the predictions files"


# --------------------------------------------------------------------------
# the copied environment cells must not drift
# --------------------------------------------------------------------------

# Cell indices in notebook 05 that carry the M4-3 / M4-4 / M4-5 fixes.
VERBATIM_FROM_05 = {
    "repo sync": 6,
    "dependency install": 8,
    "environment gate": 9,
    "data download": 11,
}


@pytest.mark.parametrize(("label", "index"), sorted(VERBATIM_FROM_05.items()))
def test_environment_cells_match_notebook_05(label: str, index: int) -> None:
    """They were copied rather than adapted precisely so they cannot diverge.

    Trailing newlines are normalised: nbformat stores source as a line list, so a
    cell identical in content can differ by one final newline.
    """
    if not NOTEBOOK_05.exists():  # pragma: no cover - notebook 05 is committed
        pytest.skip("notebook 05 not present")
    expected = _sources(NOTEBOOK_05)[index].rstrip("\n")
    actual = {body.rstrip("\n") for body in _sources(NOTEBOOK)}
    assert expected in actual, (
        f"the {label} cell has drifted from notebook 05's cell {index} — "
        "these carry the M4-3/M4-4/M4-5 fixes and must stay identical"
    )
