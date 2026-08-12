# Frontend Specification

Stack: Next.js 14+ (App Router), TypeScript, Tailwind CSS, shadcn/ui, Recharts (`DECISION_REGISTER.md` E2–E5). This document is authoritative for every page, component, and visual decision — Claude Code should not need to invent any of this during implementation.

## 1. Visual identity

The brief is explicit: this must not look like a generic AI-SaaS template. The identity to communicate is **"AI + Urdu NLP + Research + Trust,"** not "startup product."

**Color system** (define as Tailwind CSS variables / design tokens in `frontend/styles/globals.css`):

| Token | Role | Value (HSL, exact enough to implement directly) | Rationale |
|---|---|---|---|
| `--color-ink` | Primary text, primary UI color | `210 35% 12%` (near-black deep ink-teal) | Deep ink-teal instead of pure black or the ubiquitous violet/purple AI-SaaS primary — reads as serious/editorial (research, journalism), not "startup" |
| `--color-teal-600` | Primary interactive (buttons, links, focus rings) | `192 55% 28%` | Calm, trustworthy, distinct from alarmist red or hype purple |
| `--color-teal-50` | Primary-tint backgrounds (cards, highlights) | `192 40% 95%` | — |
| `--color-amber-500` | Accent (used sparingly: highlights in explanation view, key stat callouts) | `38 75% 52%` | A restrained nod to Urdu manuscript/illumination gold, not a bright "AI purple gradient" — used only as accent, never as a dominant fill |
| `--color-warm-gray-50..900` | Neutral scale for backgrounds/text/borders | Standard warm-gray ramp (e.g., Tailwind `stone` scale) | Warm, not cold-blue-gray — avoids the sterile SaaS-dashboard look |
| `--color-risk-elevated` | "Fake-pattern" prediction indicator | `18 55% 45%` (muted terracotta, not alarm-red) | Deliberately desaturated relative to a typical red — avoids an alarmist "red = danger" framing that would undercut the responsible-AI messaging (a stark red badge implies certainty the model doesn't have) |
| `--color-risk-low` | "Real-pattern" prediction indicator | `152 30% 38%` (muted sage green, not bright green) | Same reasoning — muted, not a stoplight |

**Typography:** `Noto Nastaliq Urdu` (Google Fonts, self-hosted via `next/font`) for all Urdu-script content (article text, results, explanations) — this is a functional requirement, not decoration, since a generic sans-serif renders Urdu (which is properly typeset in Nastaliq style) poorly. `Inter` for all Latin-script UI chrome (nav, buttons, labels, English body copy). Both loaded via `next/font/google` for performance and licensing simplicity.

Three specific, non-obvious Urdu-typography decisions (get these wrong and the product looks amateurish regardless of the color/component work):
1. **Line height:** Nastaliq's diagonal, stacked letterforms need materially more vertical breathing room than Latin text — set `line-height: 2` (not Tailwind's default `1.5`) on any Urdu-script container, and a correspondingly larger vertical rhythm between paragraphs of Urdu article text.
2. **Justification:** never use `text-align: justify` on Urdu content — justification breaks Nastaliq's calligraphic letter-to-letter connections by inserting irregular gaps. Urdu blocks are `text-align: right` (the natural RTL default), ragged edge, never justified.
3. **Numerals:** confidence percentages, statistics, and dates in the UI chrome are always rendered in Western (Latin) digits for universal legibility and consistent alignment in tables/charts, even inside an otherwise-RTL container — but numerals that appear *inside submitted or example Urdu article text* are left exactly as authored (Eastern Arabic-Indic or Western, whichever the source used) and never transliterated, since altering the source text would misrepresent what the model actually saw.

## 1.5 Grid, layout, and motion

**Grid:** a 12-column grid on desktop (≥1024px), collapsing to a single column below `768px` (Tailwind's default breakpoint scale is used as-is — `sm:640 / md:768 / lg:1024 / xl:1280` — no custom breakpoints invented). Max content width is capped, not full-bleed: `1200px` for content-dense pages (`/research`, `/methodology`, `/dataset`), a narrower `760px` reading column specifically for the `/analyze` textarea and results — long lines are already harder to scan in RTL Nastaliq than in Latin text, so the input/output column is kept deliberately narrower than the page's outer max-width, not stretched edge-to-edge.

**Motion (restrained, on purpose — "premium" here means motion that's barely noticed, not motion that performs):** a single transition-timing token (`150ms ease-out`) used consistently for all state changes (loading→result reveal, hover states, tab switches) — no bouncy/spring easing, no page-transition animations, no confetti or celebratory micro-interactions (consistent with Section 1's "no toasts/confetti" rule — this is a research tool, not a consumer app). Skeleton loading states use a subtle, slow shimmer (not a fast pulsing spinner) to match the restrained tone. Hover/focus states on cards and buttons are a soft border/background shift, never a scale/bounce transform. The one deliberate exception: `HighlightedText`'s explanation spans fade in with a slightly longer, staggered transition (~400ms, sequential per span) — this is the one moment in the product where a bit more motion is earned, because it's directly communicating the model's reasoning process, not decorating the UI.

**Mobile vs. desktop:** on mobile (`<768px`), `/analyze`'s input and results stack vertically full-width; `/research`'s `CrossDatasetHeatmap` and `BenchmarkTable` switch to a horizontally-scrollable container with a visible scroll affordance (not a squeezed, unreadable table) rather than being redesigned as separate mobile-only chart types — a research tool's data should look the same everywhere, just reflowed, not simplified away on small screens.

**Spacing/radius/shadow:** Tailwind's default spacing scale, unmodified (no need to reinvent it). Border radius: `--radius-card: 0.75rem` (rounded but not pill-shaped — reads as "document/card," not "bubbly app"). Shadows: minimal, single soft shadow tier (`shadow-sm` equivalent) — this is a research tool, not a marketing site; avoid heavy drop-shadows/glassmorphism.

**Components (shadcn/ui primitives, restyled with the tokens above):** Card, Button (primary=teal-600, secondary=outline, destructive reserved only for irreversible actions, of which there are none in this app), Input/Textarea (with RTL support — `dir="rtl"` conditionally applied when Urdu script is detected in the textarea), Alert (used for the disclaimer banner and error states), Tabs (used on the Research page to switch between benchmark views), Skeleton (loading state for prediction/explanation), Badge (prediction label).

**States:** Loading = skeleton placeholders matching the final content's shape (not a generic spinner-only screen, except for the initial page load). Empty = a specific illustration-free, text-first empty state ("Paste an Urdu article or choose an example to begin") — no decorative empty-state illustrations, consistent with the restrained visual identity. Error = an `Alert` with `variant="destructive"` styled in the terracotta risk color (not pure red), always paired with a specific next action ("Try a shorter article," "Check the URL and try again"). Success = no separate "success" chrome beyond the Results screen itself rendering — this app's "success" state is the normal, expected outcome, not a toast/confetti moment (toasts/confetti would undercut the serious tone).

## 2. Information architecture and page hierarchy

Two tiers, deliberately kept flat (no nested sub-navigation, no dashboard-style sidebar — a sidebar would signal "SaaS product," which is explicitly the look this project avoids): **Tier 1 (primary nav, always visible):** Home, Analyze, Research. **Tier 2 (secondary, reachable from Home's transparency cards, the footer, and cross-links from Tier 1 pages, but not cluttering the main nav bar):** Methodology, Dataset, Model Info, Responsible Use, About. This mirrors the actual task hierarchy: a visitor either wants to *use* the tool (Analyze) or *evaluate* it (Research, and from there the Tier 2 trio) — the nav should not force a first-time visitor to parse eight equally-weighted links.

## 3. Pages (final list)

| Route | Purpose | Rendering |
|---|---|---|
| `/` | Landing — explains what the tool does/doesn't do, leads with the disclaimer, links to Analyze and the three transparency pages | Server-rendered (static content, good for sharing/SEO) |
| `/analyze` | Primary interaction: input → prediction → (optional) explanation, all as progressive reveal within one page using client state (not three separate routes) — this is a deliberate UX call: forcing navigation between input/results/explanation would break the single coherent task the user is doing | Client component (interactive) |
| `/methodology` | Plain-language explanation of the cross-dataset generalization finding, with an in-domain-vs-cross-dataset F1 chart — the differentiator page (`MASTER_PROJECT_BLUEPRINT.md` Part 18) | Server-rendered, fetches metrics at build/revalidate time |
| `/dataset` | Dataset card content rendered as a page: what data trained the model, time range, domains, known limitations (Notri-Fact's undocumented provenance, Ax-to-Grind's length confound) | Server-rendered from `docs/dataset_card.md` |
| `/model` | Model card content: architecture, training setup, in-domain and cross-dataset metrics, calibration note | Server-rendered from `docs/model_card.md` |
| `/research` | Full benchmark tables and charts: in-domain comparison (4 models), cross-dataset F1 heatmap, length-bucketed performance, confusion matrices, shortcut-analysis figures | Server-rendered, pulls from `research/results/` at build time (static generation — this data changes only when new experiments are run, not per-request) |
| `/responsible-use` | The standing disclaimer, in full, FAQ-style | Server-rendered, static |
| `/about` | Author, links to thesis/paper/GitHub, project motivation | Server-rendered, static |
| `not-found` (404) | On-brand 404, links back to `/` and `/analyze` | Server-rendered, static |

No separate `/results` or `/explanation` route, and no `/history` route in the MVP (history is a stretch feature — if built later, it's a client-state panel within `/analyze`, not a new route, since it's not meaningfully a different "page").

## 4. Component tree

(Request-lifecycle logic used by these components is factored into `frontend/lib/hooks/useAnalyze.ts` and `useExplain.ts`, not duplicated inline — see `ARCHITECTURE.md` Section 4's frontend tree. Client-side input validation lives in `frontend/lib/validation.ts`, imported by `ArticleInput`/`UrlInput`, never re-implemented per-component.)

```
components/
├── layout/
│   ├── Navbar.tsx              # Props: none. Links to all pages. Sticky, minimal.
│   ├── Footer.tsx               # Props: none. License, GitHub, thesis links.
│   └── DisclaimerBanner.tsx     # Props: { dismissible?: boolean }. Renders docs/responsible_ai.md's standing text. Shown on Landing (persistent) and Analyze (persistent, non-dismissible — this one must never be dismissible, unlike a generic cookie-banner pattern, because it needs to stay visible exactly where predictions are shown)
│
├── analyze/
│   ├── ArticleInput.tsx         # Props: { value, onChange, dir }. State: local text value. RTL-aware textarea, live character count against the 10–5000 bound from BACKEND_SPECIFICATION.md. A11y: labeled, aria-describedby pointing to the count/limit hint.
│   ├── UrlInput.tsx              # Props: { value, onChange }. Validates http(s) shape client-side (server re-validates + SSRF-checks regardless — client validation is UX only, never trusted as the security boundary).
│   ├── ExampleArticlePicker.tsx  # Props: { onSelect }. Fetches GET /api/v1/examples once, renders a dropdown/list. Loading: skeleton row. Error: silently hides the picker rather than blocking the page (examples are a convenience, not critical path).
│   ├── AnalyzeButton.tsx         # Props: { onClick, disabled, loading }. Disabled when input is empty/invalid or a request is in flight.
│   └── AnalyzeForm.tsx           # Composes the above; owns the request lifecycle (idle → loading → success/error) via useState/useReducer (DECISION_REGISTER.md E4); calls POST /api/v1/analyze via lib/api-client.ts.
│
├── results/
│   ├── PredictionCard.tsx        # Props: { label, confidence, disclaimer }. Renders the Badge (risk-elevated/risk-low tokens) + ConfidenceGauge + the disclaimer text inline (never omitted).
│   ├── ConfidenceGauge.tsx       # Props: { confidence: number }. A simple horizontal bar, NOT a percentage framed as "% chance of being false" — labeled "model confidence."
│   ├── ExplainButton.tsx         # Props: { predictionId, onExplained }. Triggers POST /api/v1/explain lazily (not auto-fetched with the prediction — matches the latency decision in ARCHITECTURE.md).
│   ├── ExplanationView.tsx       # Props: { spans }. Renders HighlightedText + the fixed caption "These text features contributed most strongly to the model's prediction" (this exact phrasing is not to be altered — see CLAUDE.md rule 14).
│   └── HighlightedText.tsx       # Props: { text, spans: {text, attribution}[] }. Color-intensity (using --color-amber-500 at varying opacity) + underline weight (never color alone — a11y requirement from MASTER_PROJECT_BLUEPRINT.md Part 18).
│
├── research/
│   ├── BenchmarkTable.tsx        # Props: { rows }. In-domain 4-model comparison table, used on /research.
│   ├── CrossDatasetHeatmap.tsx   # Props: { matrix }. Recharts-based heatmap (4 models × 2 directions), the headline research figure.
│   ├── ConfusionMatrix.tsx       # Props: { matrix, labels }. Per-model confusion matrix visualization.
│   └── LengthDistributionChart.tsx # Props: { data }. Histogram, real vs. fake, both datasets — visualizes the confound directly, used on /methodology and /dataset.
│
└── ui/                            # shadcn/ui primitives (Button, Card, Input, Textarea, Alert, Tabs, Skeleton, Badge) — generated via the shadcn CLI, then restyled with the tokens in Section 1, not hand-rolled from scratch
```

Every component above lists its own props/state/loading/error behavior; a component not on this list should not be created without first adding it here (mirrors `CLAUDE.md` rule 1).

## 5. User flows

**Main flow:** Land on `/` → read the disclaimer + purpose → click "Try it" → arrive at `/analyze` → paste Urdu text (or pick an example) → click Analyze → loading skeleton on `PredictionCard`'s slot → result renders (label + confidence + disclaimer, always together) → user clicks "See why" → loading skeleton on `ExplanationView`'s slot → explanation renders → user can analyze another article (form resets) or navigate to `/methodology` to understand the limitation context.

**URL flow:** Same as above, but the user fills `UrlInput` instead of `ArticleInput` → backend extraction (`SECURITY.md`) runs server-side → on success, behaves identically to the text flow from the prediction step onward → on extraction failure, a specific error state (below) is shown instead of a generic error.

**Invalid input flow (empty/too short/too long):** `AnalyzeButton` stays disabled with an inline hint ("Enter at least 10 characters" / "Maximum 5000 characters") — this is caught client-side before a request is even sent, so there's no round-trip for this case.

**API failure flow (500/network error):** `Alert` (destructive/terracotta) reading "Something went wrong analyzing this article. Please try again." with a retry button that re-submits the same input — no silent failure, no console-only error.

**Model failure flow (health check fails / model not loaded):** Detected via a `GET /api/v1/health` check the frontend can optionally poll before allowing submission on cold start; if a request still hits this, the same generic API-failure `Alert` is shown (the user doesn't need to know the internal distinction between "model not loaded" and "model errored" — `BACKEND_SPECIFICATION.md` documents this distinction for developers, not end users).

**Extremely long article flow:** Client-side truncation warning shown at the 5000-character bound, input is hard-capped in the textarea (cannot type past it) rather than silently truncating server-side — this makes the limit visible and honest rather than surprising the user with a truncated analysis.

**Empty input flow:** Covered by "Invalid input flow" above.

**Unsupported/non-Urdu content flow:** Backend performs a lightweight script check (`ML_SPECIFICATION.md`) and returns a soft warning (not a hard block — the backend still attempts a prediction) surfaced as a non-blocking `Alert` above the result: "This text doesn't look like Urdu — results may be unreliable." This matches the responsible-AI principle of disclosing uncertainty rather than pretending the tool has scope it doesn't.

**404 flow:** Any unmatched route renders `not-found.tsx` with links back to `/` and `/analyze` — no dead ends.

## 6. Accessibility requirements (cross-cutting, applies to all components above)

RTL layout must be a real, tested state (not an afterthought) for any Urdu-text-containing component — set via `dir="rtl"` on the relevant container, not globally on `<html>`, since UI chrome (nav, buttons) stays LTR while content is RTL. Color is never the sole signal (explanation highlighting pairs color with underline weight; prediction badges pair color with a text label, never a color-only dot). All interactive elements are keyboard-navigable and carry appropriate ARIA labels, verified as part of the frontend test suite (`TESTING_STRATEGY.md`).
