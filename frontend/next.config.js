/**
 * Next.js configuration.
 *
 * Milestone 0 scaffold: intentionally close to empty. `reactStrictMode` is on so
 * that unsafe lifecycle patterns surface during Milestone 7's component work
 * rather than in production.
 *
 * @type {import('next').NextConfig}
 */
const nextConfig = {
  reactStrictMode: true,
  // Next 16 otherwise writes frontend/AGENTS.md and frontend/CLAUDE.md on dev/build.
  // The repository's agent instructions are the root CLAUDE.md, and neither generated
  // file is in ARCHITECTURE.md Section 4's tree — a second, auto-regenerated CLAUDE.md
  // inside frontend/ would compete with the real one.
  agentRules: false,
};

module.exports = nextConfig;
