import { expect, test } from "@playwright/test";

/**
 * Milestone 0 smoke test.
 *
 * Its only job is to make `npx playwright test` green against real output rather
 * than a stubbed no-op, and to assert Milestone 0's acceptance criterion — that a
 * built, served frontend actually renders placeholder content — automatically
 * instead of by hand.
 *
 * The one REQUIRED end-to-end flow (load /analyze → pick an example article →
 * Analyze → assert prediction → "See why" → assert explanation) is specified in
 * TESTING_STRATEGY.md Section 3 and is implemented at Milestone 8, once there is a
 * backend and a checkpoint to run it against.
 */
test("home page serves placeholder content", async ({ page }) => {
  await page.goto("/");

  await expect(page.getByRole("heading", { name: "Urdu Misinformation Audit" })).toBeVisible();
});
