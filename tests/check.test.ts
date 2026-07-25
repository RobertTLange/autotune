import { chmod, mkdir, mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { checkPrerequisites } from "../src/check.js";

describe("checkPrerequisites", () => {
  const originalPath = process.env.PATH;
  const originalHeadless = process.env.AUTOTUNE_HEADLESS_BIN;
  const originalNpmExecPath = process.env.npm_execpath;

  afterEach(() => {
    process.env.PATH = originalPath;
    if (originalHeadless === undefined) {
      delete process.env.AUTOTUNE_HEADLESS_BIN;
    } else {
      process.env.AUTOTUNE_HEADLESS_BIN = originalHeadless;
    }
    if (originalNpmExecPath === undefined) {
      delete process.env.npm_execpath;
    } else {
      process.env.npm_execpath = originalNpmExecPath;
    }
  });

  it("checks python, optuna, headless, and runtime from PATH without shell lookup", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-check-"));
    await writeCompatiblePython(path.join(dir, "python3"));
    await writeExecutable(path.join(dir, "headless"), "echo '| claude | ✓ | oauth | 2.1.0 | model | - |'\n");
    process.env.PATH = `${dir}${path.delimiter}${originalPath ?? ""}`;
    process.env.AUTOTUNE_HEADLESS_BIN = path.join(dir, "headless");

    await expect(
      checkPrerequisites({
        invocation: { language: "python", command: ["python3"], script: "/tmp/train.py" },
        agent: "claude"
      })
    ).resolves.toMatchObject({ python: "3.12.1", optuna: "4.8.0" });
  });

  it("reports missing script runtimes", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-check-missing-"));
    await writeCompatiblePython(path.join(dir, "python3"));
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
    await writeCompatiblePython(path.join(dir, "python3"));
    await writeExecutable(
      path.join(dir, "headless"),
      "echo '| claude | ✓ | oauth | 2.1.0 | model | - |'; echo '| codex | ✗ | - | - | model | - |'\n"
    );
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

  it("uses the npx Headless fallback for Centaur", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-check-centaur-headless-"));
    await writeCompatiblePython(path.join(dir, "python3"));
    const npmBin = path.join(dir, "npm", "bin");
    const npxCli = path.join(npmBin, "npx-cli.js");
    await mkdir(npmBin, { recursive: true });
    await writeFile(npxCli, "console.log('| claude | ✓ | oauth | 2.1.0 | model | - |');\n", "utf8");
    await chmod(npxCli, 0o755);
    process.env.npm_execpath = path.join(npmBin, "npm-cli.js");
    process.env.PATH = `${dir}${path.delimiter}/usr/bin${path.delimiter}/bin`;
    delete process.env.AUTOTUNE_HEADLESS_BIN;

    await expect(checkPrerequisites({
      invocation: { language: "python", command: ["python3"], script: "/tmp/train.py" },
      agent: "claude",
      centaur: true
    })).resolves.toMatchObject({ headless: expect.stringContaining("npx") });
  });

  it("rejects an empty explicit Headless override", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-check-empty-headless-"));
    await writeCompatiblePython(path.join(dir, "python3"));
    process.env.PATH = `${dir}${path.delimiter}${originalPath ?? ""}`;
    process.env.AUTOTUNE_HEADLESS_BIN = "";

    await expect(checkPrerequisites({
      invocation: { language: "python", command: ["python3"], script: "/tmp/train.py" },
      agent: "claude",
      centaur: true
    })).rejects.toThrow(/configured headless executable must not be empty/i);
  });

  it("rejects unavailable Centaur proposal agents", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-check-centaur-agent-"));
    await writeCompatiblePython(path.join(dir, "python3"));
    await writeExecutable(
      path.join(dir, "headless"),
      "echo '| claude | ✓ | oauth | 2.1.0 | model | - |'; echo '| codex | ✗ | - | - | model | - |'\n"
    );
    process.env.PATH = `${dir}${path.delimiter}${originalPath ?? ""}`;
    process.env.AUTOTUNE_HEADLESS_BIN = path.join(dir, "headless");

    await expect(checkPrerequisites({
      invocation: { language: "python", command: ["python3"], script: "/tmp/train.py" },
      agent: "codex",
      centaur: true
    })).rejects.toThrow(/Centaur proposal agent.*codex.*not available/i);
  });

});

async function writeCompatiblePython(filePath: string): Promise<void> {
  await writeExecutable(filePath, [
    "if [[ \"$1\" == \"--version\" ]]; then",
    "  echo 'Python 3.12.1'",
    "elif [[ \"$*\" == *\"platform.machine\"* ]]; then",
    "  echo '{\"executable\":\"'\"$0\"'\",\"version\":\"3.12.1\",\"implementation\":\"cpython\",\"platform\":\"linux\",\"arch\":\"x86_64\"}'",
    "elif [[ \"$*\" == *\"import cmaes\"* ]]; then",
    "  echo '{\"optuna\":\"4.8.0\",\"cmaes\":\"0.12.0\"}'",
    "elif [[ \"$*\" == *\"import json, optuna\"* ]]; then",
    "  echo '{\"optuna\":\"4.8.0\"}'",
    "else",
    "  exit 1",
    "fi",
    ""
  ].join("\n"));
}

async function writeExecutable(filePath: string, body: string): Promise<void> {
  await writeFile(filePath, `#!/usr/bin/env bash\n${body}`, "utf8");
  await chmod(filePath, 0o755);
}
