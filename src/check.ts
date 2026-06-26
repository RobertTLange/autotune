import { access } from "node:fs/promises";
import { constants } from "node:fs";
import path from "node:path";
import { runCommand } from "./process.js";
import type { Invocation } from "./types.js";

export interface PrerequisiteReport {
  python: string;
  optuna: string;
  headless: string;
  runtime: string;
}

export async function checkPrerequisites(input: {
  invocation: Invocation;
  agent: string;
  skipHeadless?: boolean;
}): Promise<PrerequisiteReport> {
  const python = await checkPython();
  const optuna = await checkOptuna();
  const headless = input.skipHeadless ? "skipped" : await checkHeadless(input.agent);
  const runtime = await checkRuntime(input.invocation);
  return { python, optuna, headless, runtime };
}

async function checkPython(): Promise<string> {
  const { stdout } = await runCommand("python3", ["--version"]);
  const version = stdout.trim().replace(/^Python\s+/, "");
  const [major = "0", minor = "0"] = version.split(".");
  if (Number(major) < 3 || (Number(major) === 3 && Number(minor) < 9)) {
    throw new Error(`python3 >= 3.9 required, found ${version}`);
  }
  return version;
}

async function checkOptuna(): Promise<string> {
  try {
    const { stdout } = await runCommand("python3", ["-c", "import optuna; print(optuna.__version__)"]);
    return stdout.trim();
  } catch (error) {
    throw new Error(`Optuna is required: python3 -m pip install optuna (${String(error)})`);
  }
}

async function checkHeadless(agent: string): Promise<string> {
  const bin = process.env.AUTOTUNE_HEADLESS_BIN ?? "headless";
  let stdout = "";
  let stderr = "";
  try {
    ({ stdout, stderr } = await runCommand(bin, ["--check"]));
  } catch (error) {
    if (bin === "headless" && isMissingExecutable(error)) {
      ({ stdout, stderr } = await runCommand("npx", ["-y", "@roberttlange/headless", "--check"]));
    } else {
      throw error;
    }
  }
  const output = `${stdout}\n${stderr}`;
  if (!output.toLowerCase().includes(agent.toLowerCase())) {
    return `${bin} (agent ${agent} not listed by --check)`;
  }
  return `${bin} (${agent})`;
}

function isMissingExecutable(error: unknown): boolean {
  return typeof error === "object" && error !== null && "code" in error && error.code === "ENOENT";
}

async function checkRuntime(invocation: Invocation): Promise<string> {
  const executable = invocation.command[0];
  if (!executable) {
    throw new Error("empty invocation command");
  }
  if (executable.includes("/") || executable.startsWith(".")) {
    await access(executable, constants.X_OK);
    return executable;
  }
  const resolved = await findOnPath(executable);
  if (!resolved) {
    throw new Error(`runtime not found on PATH: ${executable}`);
  }
  return resolved;
}

async function findOnPath(executable: string): Promise<string | undefined> {
  for (const directory of (process.env.PATH ?? "").split(path.delimiter)) {
    const candidate = path.join(directory, executable);
    try {
      await access(candidate, constants.X_OK);
      return candidate;
    } catch {
      // Keep scanning PATH.
    }
  }
  return undefined;
}
