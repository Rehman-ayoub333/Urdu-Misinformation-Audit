import { fileURLToPath } from "node:url";

import { defineConfig } from "vitest/config";

/**
 * Vitest + React Testing Library config (DECISION_REGISTER.md E14).
 *
 * No `@vitejs/plugin-react`: the plugin exists for Fast Refresh, which tests do
 * not use, and its current major pulls in a Vite/rolldown version that conflicts
 * with Vitest's own. Vitest's esbuild transform handles JSX on its own — hence
 * `esbuild.jsx = "automatic"`, set explicitly so the unit suite does not depend on
 * whatever `jsx` value Next's build last wrote into tsconfig.json.
 *
 * The `@/` alias is declared here rather than via `vite-tsconfig-paths` for the
 * same reason: one alias does not justify a plugin (CLAUDE.md rule 16).
 *
 * Playwright specs under tests/e2e/ are excluded — they run via `npm run test:e2e`.
 */
export default defineConfig({
  esbuild: { jsx: "automatic" },
  resolve: {
    alias: {
      "@": fileURLToPath(new URL(".", import.meta.url)),
    },
  },
  test: {
    environment: "jsdom",
    include: ["tests/unit/**/*.test.{ts,tsx}"],
    setupFiles: ["./tests/unit/setup.ts"],
  },
});
