import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    globals: true,
    exclude: ["node_modules/**", "dist/**"],
    coverage: {
      provider: "v8",
      reporter: ["text"],
      thresholds: {
        statements: 80,
        branches: 60,
        functions: 80,
        lines: 80
      },
      include: ["src/**/*.ts"],
      exclude: ["src/cli.ts"]
    }
  }
});
