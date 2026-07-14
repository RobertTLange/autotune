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

  it("checks supported Optuna and cmaes versions for Centaur", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-check-centaur-"));
    await writeExecutable(
      path.join(dir, "python3"),
      "if [ \"$1\" = \"--version\" ]; then echo 'Python 3.12.1'; else printf '4.8.0\\n0.12.0\\n'; fi\n"
    );
    await writeExecutable(path.join(dir, "headless"), "echo 'claude ok'\n");
    process.env.PATH = `${dir}${path.delimiter}${originalPath ?? ""}`;
    process.env.AUTOTUNE_HEADLESS_BIN = path.join(dir, "headless");

    await expect(
      checkPrerequisites({
        invocation: { language: "python", command: ["python3"], script: "/tmp/train.py" },
        agent: "claude",
        centaur: true
      })
    ).resolves.toMatchObject({ optuna: "4.8.0", cmaes: "0.12.0" });
  });

  it("requires an installed headless executable for Centaur", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-check-centaur-headless-"));
    await writeExecutable(
      path.join(dir, "python3"),
      "if [ \"$1\" = \"--version\" ]; then echo 'Python 3.12.1'; else printf '4.8.0\\n0.12.0\\n'; fi\n"
    );
    process.env.PATH = `${dir}${path.delimiter}/usr/bin${path.delimiter}/bin`;
    delete process.env.AUTOTUNE_HEADLESS_BIN;

    await expect(checkPrerequisites({
      invocation: { language: "python", command: ["python3"], script: "/tmp/train.py" },
      agent: "claude",
      centaur: true
    })).rejects.toThrow(/Centaur requires an installed headless executable/i);
  });

  it("rejects unsupported Optuna versions for Centaur", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-check-centaur-version-"));
    await writeExecutable(
      path.join(dir, "python3"),
      "if [ \"$1\" = \"--version\" ]; then echo 'Python 3.12.1'; else printf '4.7.0\\n0.12.0\\n'; fi\n"
    );
    await writeExecutable(path.join(dir, "headless"), "echo 'claude ok'\n");
    process.env.PATH = `${dir}${path.delimiter}${originalPath ?? ""}`;
    process.env.AUTOTUNE_HEADLESS_BIN = path.join(dir, "headless");

    await expect(
      checkPrerequisites({
        invocation: { language: "python", command: ["python3"], script: "/tmp/train.py" },
        agent: "claude",
        centaur: true
      })
    ).rejects.toThrow(/Optuna >= 4\.8\.0 and < 5/i);
  });

  it("rejects unsupported cmaes versions for Centaur", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-check-cmaes-version-"));
    await writeExecutable(
      path.join(dir, "python3"),
      "if [ \"$1\" = \"--version\" ]; then echo 'Python 3.12.1'; else printf '4.8.0\\n0.11.1\\n'; fi\n"
    );
    await writeExecutable(path.join(dir, "headless"), "echo 'claude ok'\n");
    process.env.PATH = `${dir}${path.delimiter}${originalPath ?? ""}`;
    process.env.AUTOTUNE_HEADLESS_BIN = path.join(dir, "headless");

    await expect(
      checkPrerequisites({
        invocation: { language: "python", command: ["python3"], script: "/tmp/train.py" },
        agent: "claude",
        centaur: true
      })
    ).rejects.toThrow(/cmaes >= 0\.12/i);
  });
});

async function writeExecutable(filePath: string, body: string): Promise<void> {
  await writeFile(filePath, `#!/usr/bin/env bash\n${body}`, "utf8");
  await chmod(filePath, 0o755);
}
