import { chmod, mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { checkPrerequisites } from "../src/check.js";

describe("checkPrerequisites", () => {
  const originalPath = process.env.PATH;
  const originalHeadless = process.env.AUTOTUNE_HEADLESS_BIN;

  afterEach(() => {
    process.env.PATH = originalPath;
    if (originalHeadless === undefined) {
      delete process.env.AUTOTUNE_HEADLESS_BIN;
    } else {
      process.env.AUTOTUNE_HEADLESS_BIN = originalHeadless;
    }
  });

  it("checks python, optuna, headless, and runtime from PATH without shell lookup", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-check-"));
    await writeExecutable(path.join(dir, "python3"), "if [ \"$1\" = \"--version\" ]; then echo 'Python 3.12.1'; else echo '3.6.1'; fi\n");
    await writeExecutable(path.join(dir, "headless"), "echo 'claude ok'\n");
    process.env.PATH = `${dir}${path.delimiter}${originalPath ?? ""}`;
    process.env.AUTOTUNE_HEADLESS_BIN = path.join(dir, "headless");

    await expect(
      checkPrerequisites({
        invocation: { language: "python", command: ["python3"], script: "/tmp/train.py" },
        agent: "claude"
      })
    ).resolves.toMatchObject({ python: "3.12.1", optuna: "3.6.1" });
  });

  it("reports missing script runtimes", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-check-missing-"));
    await writeExecutable(path.join(dir, "python3"), "if [ \"$1\" = \"--version\" ]; then echo 'Python 3.12.1'; else echo '3.6.1'; fi\n");
    process.env.PATH = `${dir}${path.delimiter}/usr/bin${path.delimiter}/bin`;

    await expect(
      checkPrerequisites({
        invocation: { language: "other", command: ["definitely-missing-autotune-runtime"], script: "/tmp/train.other" },
        agent: "claude",
        skipHeadless: true
      })
    ).rejects.toThrow(/runtime not found/);
  });
});

async function writeExecutable(filePath: string, body: string): Promise<void> {
  await writeFile(filePath, `#!/usr/bin/env bash\n${body}`, "utf8");
  await chmod(filePath, 0o755);
}
