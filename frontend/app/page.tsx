/**
 * Home page — Milestone 0 placeholder.
 *
 * The real landing page (hero leading with the disclaimer, "Try it" CTA, and the
 * three-card methodology/dataset/limitations summary) is built at Milestone 7 per
 * FRONTEND_SPECIFICATION.md. This exists only so `npm run dev` serves something
 * verifiable, which is Milestone 0's acceptance criterion.
 *
 * No disclaimer copy is reproduced here: it lives in `docs/responsible_ai.md` and
 * is imported through `lib/constants.ts` once that exists (CLAUDE.md rule 14).
 */
export default function HomePage() {
  return (
    <main className="mx-auto flex min-h-screen max-w-2xl flex-col justify-center gap-4 p-8">
      <h1 className="text-3xl font-semibold tracking-tight">Urdu Misinformation Audit</h1>
      <p className="text-muted-foreground">
        Scaffold placeholder. This project is at Milestone 0 of its roadmap — no analysis
        functionality is implemented yet.
      </p>
    </main>
  );
}
