# Security

Scope: this is a small, free-tier-deployed, unauthenticated public demo — the security posture is proportionate to that (`MASTER_PROJECT_BLUEPRINT.md` Part 23's "no unnecessary enterprise infrastructure" rule applies throughout). The one genuinely serious risk is **SSRF via the URL-extraction feature**; everything else here is standard, low-effort hardening.

## 1. SSRF prevention (`backend/app/security/ssrf.py`)

This is the concrete implementation spec for `test_ssrf.py` in `TESTING_STRATEGY.md`. Before fetching any user-submitted URL:

1. Parse the URL; reject any scheme other than `http`/`https`.
2. Resolve the hostname to an IP address **before** fetching.
3. Reject if the resolved IP falls in any of: `127.0.0.0/8`, `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`, `169.254.0.0/16` (this specifically blocks cloud metadata endpoints like `169.254.169.254`, a classic real-world SSRF target), `::1`, and other loopback/link-local/private ranges (IPv6 equivalents included).
4. Reject non-standard ports outside `{80, 443}` unless explicitly needed.
5. Fetch with a hard timeout (`EXTRACTION_TIMEOUT_SECONDS`, default 10s) and a max-bytes cap (`EXTRACTION_MAX_BYTES`, default 2MB) enforced while streaming, not only after the full response downloads.
6. If the response is a redirect, **re-resolve and re-validate the new target** before following it; cap total redirects at 3. A redirect to a blocked IP range fails the request with `422`, matching a real bypass technique (DNS rebinding / open redirect to an internal address) that a naive "check the URL once" implementation would miss.
7. Extract text via `trafilatura` (`DECISION_REGISTER.md` E11) — never a headless browser (no page-JS execution, eliminating an entire class of risk from untrusted, arbitrary web content).
8. Strip any residual HTML/script tags from the extracted text before it reaches `PreprocessService` (defense in depth, even though the model itself doesn't execute content).

## 2. Input validation

Text length bounds enforced both client-side (UX) and server-side (the actual boundary — never trust the client): `MIN_TEXT_LENGTH`/`MAX_TEXT_LENGTH` from `BACKEND_SPECIFICATION.md`. Non-text/binary payloads rejected by Pydantic's type validation before reaching any handler code.

## 3. Secret management

No secret is required on the frontend (`DECISION_REGISTER.md` E16 — only a public API base URL). Backend secrets (if any are ever needed, e.g., a private HF token for a gated model — not expected given the model repo is public) are read via `pydantic-settings` from environment variables set in the HF Spaces "Secrets" panel, never committed. `.env.example` lists variable names with placeholder/empty values only. A pre-commit or CI check (`git-secrets`-style pattern scan, or minimally a manual review step documented in `GITHUB_PLAN.md`) guards against accidental secret commits.

## 4. Dependency security

GitHub Dependabot enabled on the repo (free, zero-config) for both `backend/requirements.txt` and `frontend/package.json`. `pip-audit` run as an optional local/CI check for the backend. Pinned versions everywhere (`REPRODUCIBILITY.md`) reduce the chance of an unreviewed transitive-dependency change.

## 5. Rate limiting

`slowapi` middleware (`DECISION_REGISTER.md` E10), IP-based sliding window, `RATE_LIMIT_PER_MINUTE` (default 20) applied to `/api/v1/analyze` and `/api/v1/explain` — the two compute-costly endpoints. `/health`, `/examples`, `/model-info` are unlimited (cheap, static/near-static reads).

## 6. Privacy

Raw submitted article text is not logged by default (`BACKEND_SPECIFICATION.md` Section 5) — given plausible political/religious/sensitive content, this is a genuine privacy consideration, not just good hygiene. The `/explain` id-cache (`BACKEND_SPECIFICATION.md`) holds text only transiently, in-process memory, with a TTL — never written to disk.

## 7. Explicitly out of scope for this project

Authentication/authorization (no user accounts exist), a WAF/CDN security layer, DDoS protection beyond the platform's own free-tier defaults (Vercel/HF Spaces), and any enterprise compliance tooling — all correctly excluded per `DECISION_REGISTER.md` E8/E10's proportionality reasoning. If the project's scope ever grows to include accounts or paid infrastructure, this section must be revisited and a new `DECISION_REGISTER.md` row added before implementation.
