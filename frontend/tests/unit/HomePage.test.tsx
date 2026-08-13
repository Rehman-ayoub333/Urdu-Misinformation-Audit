import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import HomePage from "@/app/page";

/**
 * Milestone 0 trivial test — it exists so `npm run test` is genuinely green in CI
 * rather than a no-op. The real component suite (AnalyzeForm, PredictionCard,
 * ExplanationView, ArticleInput, ExampleArticlePicker) is specified in
 * TESTING_STRATEGY.md Section 2 and written at Milestone 7.
 */
describe("HomePage", () => {
  it("renders the placeholder heading", () => {
    render(<HomePage />);

    expect(screen.getByRole("heading", { name: "Urdu Misinformation Audit" })).toBeInTheDocument();
  });
});
