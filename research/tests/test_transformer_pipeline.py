"""Transformer pipeline tests, including the checkpoint-loading regression test.

Split by what each test needs:

* **Pure logic** — step arithmetic, seeding, config validity, push-credential
  handling. Runs everywhere, no model download, no GPU.
* **Checkpoint loading** — SKIPS until a real trained checkpoint exists. The
  checkpoint is produced by Rehman's Colab run (`PROJECT_SPECIFICATION.md`
  Section 6), so before that this cannot pass and must not pretend to. Same
  pattern as the Notri-Fact tests at Milestone 1, which skipped until Kaggle
  credentials existed.

Point the checkpoint tests at a real checkpoint by setting::

    TEST_CHECKPOINT_REPO=<hf-user>/urdu-misinfo-xlmr-staging
    TEST_CHECKPOINT_REVISION=<commit sha>      # optional but recommended

`REPRODUCIBILITY.md` Section 6 requires a checkpoint be identified by repo ID AND
revision — a branch name moves, a SHA does not — which is why the revision has its
own variable rather than being folded into the repo string.
"""

from __future__ import annotations

import os

import pytest
import yaml

from research.src.models.transformer import (
    ENV_HF_STAGING_PREFIX,
    ENV_HF_TOKEN,
    MODEL_CONFIGS,
    CONFIG_DIR,
    DryRunSettings,
    load_config,
    push_checkpoint,
    training_step_counts,
)

ENV_CHECKPOINT_REPO = "TEST_CHECKPOINT_REPO"
ENV_CHECKPOINT_REVISION = "TEST_CHECKPOINT_REVISION"

# EXPERIMENT_PLAN.md Section 5 / REPRODUCIBILITY.md Section 2 — not tunable.
REQUIRED_SEEDS = [42, 123, 2026]


# --- Config correctness --------------------------------------------------------


@pytest.mark.parametrize("experiment_id", sorted(MODEL_CONFIGS))
def test_model_config_loads(experiment_id: str) -> None:
    config = load_config(MODEL_CONFIGS[experiment_id])
    assert config["model"]["experiment_id"] == experiment_id


@pytest.mark.parametrize("experiment_id", sorted(MODEL_CONFIGS))
def test_config_uses_the_required_seed_triple(experiment_id: str) -> None:
    assert load_config(MODEL_CONFIGS[experiment_id])["training"]["seeds"] == REQUIRED_SEEDS


@pytest.mark.parametrize("experiment_id", sorted(MODEL_CONFIGS))
def test_max_length_is_512(experiment_id: str) -> None:
    """DECISION_REGISTER.md M4-1: 512 is both architectures' ceiling.

    Anything smaller would truncate more than the architecture forces, which would
    silently change what RQ3's results describe.
    """
    assert load_config(MODEL_CONFIGS[experiment_id])["tokenizer"]["max_length"] == 512


@pytest.mark.parametrize("experiment_id", sorted(MODEL_CONFIGS))
def test_binary_classification_head(experiment_id: str) -> None:
    assert load_config(MODEL_CONFIGS[experiment_id])["model"]["num_labels"] == 2


def test_c_and_d_share_hyperparameters() -> None:
    """C exists to answer "is the collapse model-general or XLM-R-specific?".

    If the two are trained under different budgets, any difference between them is
    confounded by the hyperparameters and that question becomes unanswerable — so
    the shared settings are asserted rather than left to reviewer diligence.
    """
    c = load_config(MODEL_CONFIGS["C"])["training"]
    d = load_config(MODEL_CONFIGS["D"])["training"]
    for key in (
        "learning_rate",
        "per_device_train_batch_size",
        "num_train_epochs",
        "warmup_ratio",
        "weight_decay",
        "seeds",
        "lr_scheduler_type",
    ):
        assert c[key] == d[key], f"C and D differ on {key}: {c[key]!r} vs {d[key]!r}"


def test_checkpoints_push_to_staging_not_production() -> None:
    """Promotion to production is a Milestone 9 decision (EXPERIMENT_PLAN.md S7)."""
    for experiment_id in MODEL_CONFIGS:
        suffix = load_config(MODEL_CONFIGS[experiment_id])["checkpoint_push"][
            "staging_repo_suffix"
        ]
        assert "staging" in suffix, f"{experiment_id} does not push to a staging repo"


# --- Step arithmetic -----------------------------------------------------------


def test_step_counts_match_the_real_training_split() -> None:
    """6,405 train rows at batch 16 => 401 steps/epoch, 1,203 over 3 epochs.

    Pins the numbers the Colab handoff's runtime estimate is built from, so the
    estimate and the scheduler cannot drift apart.
    """
    steps_per_epoch, total, warmup = training_step_counts(
        n_train=6405,
        batch_size=16,
        gradient_accumulation_steps=1,
        epochs=3,
        warmup_ratio=0.1,
    )
    assert steps_per_epoch == 401
    assert total == 1203
    assert warmup == 120


def test_step_counts_account_for_gradient_accumulation() -> None:
    _, total_plain, _ = training_step_counts(
        n_train=100, batch_size=10, gradient_accumulation_steps=1, epochs=1, warmup_ratio=0.1
    )
    _, total_accum, _ = training_step_counts(
        n_train=100, batch_size=10, gradient_accumulation_steps=2, epochs=1, warmup_ratio=0.1
    )
    assert total_plain == 10
    assert total_accum == 5


def test_step_counts_never_return_zero_steps() -> None:
    """A tiny dry-run slice must still produce at least one optimizer step."""
    steps_per_epoch, total, _ = training_step_counts(
        n_train=1, batch_size=16, gradient_accumulation_steps=1, epochs=1, warmup_ratio=0.1
    )
    assert steps_per_epoch >= 1 and total >= 1


# --- Dry-run settings ----------------------------------------------------------


def test_dry_run_defaults_are_small_enough_to_be_useful() -> None:
    """The dry run's whole purpose is being fast enough to actually run first."""
    settings = DryRunSettings()
    _, total, _ = training_step_counts(
        n_train=settings.n_train,
        batch_size=settings.batch_size,
        gradient_accumulation_steps=1,
        epochs=settings.epochs,
        warmup_ratio=0.1,
    )
    assert total <= 8, f"dry run would take {total} steps — too slow for a pre-flight check"
    assert settings.max_length <= 128
    assert settings.push is False, "a dry run must never push a checkpoint"


# --- Credential handling -------------------------------------------------------


def test_push_is_refused_without_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing credentials must fail loudly, not silently skip the push.

    A silent skip would end with Rehman's Colab session finished and no checkpoints
    anywhere, discovered only afterwards.
    """
    monkeypatch.delenv(ENV_HF_TOKEN, raising=False)
    monkeypatch.delenv(ENV_HF_STAGING_PREFIX, raising=False)
    config = load_config(MODEL_CONFIGS["D"])

    with pytest.raises(RuntimeError, match="cannot push checkpoint"):
        push_checkpoint(None, None, config, seed=42, dry_run=False)


def test_push_dry_run_reports_target_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV_HF_TOKEN, raising=False)
    monkeypatch.setenv(ENV_HF_STAGING_PREFIX, "someuser")
    result = push_checkpoint(None, None, load_config(MODEL_CONFIGS["D"]), seed=123, dry_run=True)

    assert result["pushed"] is False
    assert result["would_push_to"] == "someuser/urdu-misinfo-xlmr-staging"
    assert result["would_use_branch"] == "seed-123"
    assert result["token_set"] is False


def test_no_credential_is_hardcoded_in_the_source() -> None:
    """Rule 15: credentials come from the environment, never the repository."""
    source = (CONFIG_DIR.parent / "src" / "models" / "transformer.py").read_text(
        encoding="utf-8"
    )
    assert "hf_" + "abcdefg" not in source  # placeholder shape check
    for line in source.splitlines():
        if "HF_TOKEN" in line and "=" in line and "os.environ" not in line:
            assert "ENV_HF_TOKEN" in line or line.strip().startswith("#"), (
                f"possible hardcoded token assignment: {line.strip()!r}"
            )


@pytest.mark.parametrize("config_name", sorted(MODEL_CONFIGS.values()))
def test_configs_contain_no_secrets(config_name: str) -> None:
    raw = (CONFIG_DIR / config_name).read_text(encoding="utf-8")
    lowered = raw.lower()
    for marker in ("hf_token", "api_key", "password", "secret:"):
        assert marker not in lowered, f"{config_name} appears to contain a credential"
    yaml.safe_load(raw)


# --- Checkpoint-loading regression test (skips until a checkpoint exists) -------


def _checkpoint_reference() -> tuple[str, str | None] | None:
    repo = os.environ.get(ENV_CHECKPOINT_REPO, "").strip()
    if not repo:
        return None
    return repo, os.environ.get(ENV_CHECKPOINT_REVISION, "").strip() or None


def test_checkpoint_loads_and_predicts() -> None:
    """Regression test for a real trained checkpoint.

    Skips — deliberately, not silently — until Rehman's Colab run has produced one
    and its repo ID is exported. Asserting against a checkpoint that does not exist
    yet could only be done by fabricating one, which rule 2 forbids.

    Once a checkpoint exists this is what protects the deployment path: it proves
    the artefact loads, exposes the right label mapping, and produces a
    well-formed probability distribution over the two classes.
    """
    reference = _checkpoint_reference()
    if reference is None:
        pytest.skip(
            f"no checkpoint yet — set {ENV_CHECKPOINT_REPO} (and optionally "
            f"{ENV_CHECKPOINT_REVISION}) once Milestone 4's Colab run has pushed one"
        )

    repo_id, revision = reference
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    kwargs = {"revision": revision} if revision else {}
    tokenizer = transformers.AutoTokenizer.from_pretrained(repo_id, **kwargs)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(repo_id, **kwargs)
    model.eval()

    assert model.config.num_labels == 2
    assert set(model.config.id2label.values()) == {"fake", "real"}, (
        f"unexpected label mapping: {model.config.id2label}"
    )

    inputs = tokenizer(
        "پاکستان کی معاشی صورتحال بہتر ہو رہی ہے۔",
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        probabilities = torch.softmax(model(**inputs).logits, dim=-1)

    assert probabilities.shape == (1, 2)
    assert probabilities.sum().item() == pytest.approx(1.0, abs=1e-5)


def test_checkpoint_prediction_is_deterministic() -> None:
    """The same input must give the same output on every call.

    Serving depends on this: a non-deterministic model would return different
    verdicts for one article on refresh, which the responsible-AI framing in
    docs/responsible_ai.md could not survive.
    """
    reference = _checkpoint_reference()
    if reference is None:
        pytest.skip(f"no checkpoint yet — set {ENV_CHECKPOINT_REPO}")

    repo_id, revision = reference
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    kwargs = {"revision": revision} if revision else {}
    tokenizer = transformers.AutoTokenizer.from_pretrained(repo_id, **kwargs)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(repo_id, **kwargs)
    model.eval()

    inputs = tokenizer("کراچی میں آج موسم خوشگوار ہے۔", return_tensors="pt", truncation=True)
    with torch.no_grad():
        first = model(**inputs).logits
        second = model(**inputs).logits

    assert torch.allclose(first, second, atol=1e-6)
