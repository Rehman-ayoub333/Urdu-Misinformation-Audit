import { defineConfig, devices } from "@playwright/test";

/**
 * Playwright config (DECISION_REGISTER.md E14, TESTING_STRATEGY.md Section 3).
 * Chromium only, by design — a broader matrix costs CI time this project's scope
 * does not justify.
 *
 * `webServer` builds and starts the app itself, so the E2E run needs no manually
 * started dev server locally or in CI. At Milestone 8 this gains the required
 * analyze → explain flow, which additionally needs a running backend.
 */
const PORT = 3000;

export default defineConfig({
  testDir: "./tests/e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: process.env.CI ? "github" : "list",
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    trace: "on-first-retry",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: `npm run build && npm run start -- --port ${PORT}`,
    url: `http://127.0.0.1:${PORT}`,
    reuseExistingServer: !process.env.CI,
    timeout: 180_000,
  },
});
