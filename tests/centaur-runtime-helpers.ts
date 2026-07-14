import { chmod, mkdir, stat, writeFile } from "node:fs/promises";
import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import path from "node:path";
import { writeOptunaRunner } from "../src/generate.js";
import type { HeadlessOptions, SearchSpace } from "../src/types.js";

export const DEFAULT_TRAIN_SOURCE = `import argparse
p=argparse.ArgumentParser()
p.add_argument('--x', type=float)
p.add_argument('--y', type=float)
p.add_argument('--optimizer')
a=p.parse_args()
print(f'autotune_metric={a.x + a.y}')
`;

export async function centaurPython(): Promise<string | undefined> {
  const candidates = [process.env.AUTOTUNE_CENTAUR_PYTHON, "/tmp/autotune-centaur-venv/bin/python", "python3"]
    .filter((candidate): candidate is string => Boolean(candidate));
  for (const candidate of candidates) {
    try {
      await runPython(candidate, ["-c", "import optuna,cmaes; assert optuna.__version__.startswith('4.8.'); assert cmaes.__version__.startswith('0.12.')"]);
      return candidate;
    } catch {
      continue;
    }
  }
  if (process.env.CI) {
    throw new Error("CI requires Optuna 4.8.x and cmaes 0.12.x for Centaur runtime tests");
  }
  return undefined;
}

export async function writeRunner(
  dir: string,
  searchSpace: SearchSpace,
  studyName: string,
  python: string,
  trainSource = DEFAULT_TRAIN_SOURCE,
  headless: HeadlessOptions = { agent: "codex", model: "test-model", reasoningEffort: "low" }
): Promise<string> {
  const train = path.join(dir, "train.py");
  const runner = path.join(dir, "train_optuna.py");
  await mkdir(dir, { recursive: true });
  await writeFile(train, trainSource, "utf8");
  await writeOptunaRunner({
    invocation: { language: "python", command: [python], script: train },
    searchSpace,
    outputPath: runner,
    resultsPath: path.join(dir, "results.json"),
    studyName,
    headless
  });
  return runner;
}

export async function waitForFile(filePath: string): Promise<void> {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    try {
      await stat(filePath);
      return;
    } catch {
      await new Promise((resolve) => setTimeout(resolve, 50));
    }
  }
  throw new Error(`timed out waiting for ${filePath}`);
}

export function blockingTrainSource(marker: string, release: string): string {
  return `import argparse,time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--x', type=float)
p.add_argument('--y', type=float)
p.add_argument('--optimizer')
a=p.parse_args()
marker=Path(${JSON.stringify(marker)})
if not marker.exists():
    marker.write_text('started')
    release=Path(${JSON.stringify(release)})
    while not release.exists():
        time.sleep(0.05)
print(f'autotune_metric={a.x + a.y}')
`;
}

export async function writeFakeHeadless(
  dir: string,
  marker: string,
  proposal: Record<string, unknown>,
  delayMilliseconds = 0
): Promise<string> {
  const executable = path.join(dir, "fake-headless.mjs");
  await writeFile(executable, `#!/usr/bin/env node
import fs from "node:fs";
if (process.env.AUTOTUNE_TEST_FORBIDDEN) process.exit(7);
if (process.env.ANTHROPIC_API_KEY) process.exit(7);
if (process.argv[2] !== "codex") process.exit(7);
fs.appendFileSync(${JSON.stringify(marker)}, process.env.OPENAI_API_KEY === "selected-key" ? "k" : "x");
setTimeout(() => console.log(JSON.stringify(${JSON.stringify(proposal)})), ${delayMilliseconds});
`, "utf8");
  await chmod(executable, 0o755);
  return executable;
}

export function runnerArgs(runner: string, results: string, studyName: string, trials: number): string[] {
  return [runner, "--trials", String(trials), "--direction", "maximize", "--sampler", "centaur", "--pruner", "none", "--n-jobs", "1", "--study-name", studyName, "--output", results];
}

export function runPython(
  executable: string,
  args: string[],
  env: Record<string, string> = {},
  cwd?: string
): Promise<string> {
  return new Promise((resolve, reject) => {
    const child = spawn(executable, args, {
      stdio: ["ignore", "pipe", "pipe"],
      env: { ...process.env, ...env },
      cwd
    });
    let stdout = "";
    let stderr = "";
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) resolve(stdout);
      else reject(new Error(`${executable} ${args.join(" ")} failed with ${code}: ${stderr || stdout}`));
    });
  });
}

export function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}
