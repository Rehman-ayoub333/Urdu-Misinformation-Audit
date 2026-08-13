import type { Metadata } from "next";

import "../styles/globals.css";

/**
 * Root layout — Milestone 0 scaffold.
 *
 * Deliberately bare. The Navbar, Footer and persistent DisclaimerBanner
 * (`components/layout/`) are added at Milestone 7 per FRONTEND_SPECIFICATION.md;
 * the disclaimer copy they render comes from `docs/responsible_ai.md` via
 * `lib/constants.ts` and is never retyped (CLAUDE.md rule 14).
 */
export const metadata: Metadata = {
  title: "Urdu Misinformation Audit",
  description:
    "A research platform analysing whether Urdu misinformation classifiers learn genuine " +
    "linguistic signal or dataset-specific shortcuts.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
