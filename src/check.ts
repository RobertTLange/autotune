import { access } from "node:fs/promises";
import { constants } from "node:fs";
import path from "node:path";
import { runCommand } from "./process.js";
import { FALLBACK_HEADLESS_PACKAGE } from "./headless.js";
import type { Invocation } from "./types.js";

export interface PrerequisiteReport {
  python: string;
  optuna: string;
  cmaes?: string;
  headless: string;
  runtime: string;
}

export interface DoctorCheck {
  name: string;
  status: "ok" | "fail" | "skip";
  detail: string;
}

export async function checkPrerequisites(input: {
  invocation: Invocation;
  agent: string;
  centaur?: boolean;
  skipHeadless?: boolean;
}): Promise<PrerequisiteReport> {
  const python = await checkPython();
  const centaurPackages = input.centaur ? await checkCentaurPackages() : undefined;
  const optuna = centaurPackages?.optuna ?? await checkOptuna();
  const headless = input.skipHeadless
    ? "skipped"
    : await checkHeadless(input.agent, { allowFallback: !input.centaur });
  const runtime = await checkRuntime(input.invocation);
  return { python, optuna, cmaes: centaurPackages?.cmaes, headless, runtime };
}

export async function checkDoctorPrerequisites(input: {
  invocation?: Invocation;
  agent: string;
}): Promise<DoctorCheck[]> {
  const checks: DoctorCheck[] = [];
  checks.push(await runDoctorCheck("python3", checkPython));
  checks.push(await runDoctorCheck("optuna", checkOptuna));
  checks.push(await runDoctorCheck("headless", () => checkHeadless(input.agent)));
  checks.push(
    input.invocation
      ? await runDoctorCheck("runtime", () => checkRuntime(input.invocation as Invocation))
      : {
          name: "runtime",
          status: "skip",
          detail: "pass a script to check its runtime"
        }
  );
  return checks;
}

async function runDoctorCheck(name: string, check: () => Promise<string>): Promise<DoctorCheck> {
  try {
    return { name, status: "ok", detail: await check() };
  } catch (error) {
    return { name, status: "fail", detail: error instanceof Error ? error.message : String(error) };
  }
}

export async function checkPython(): Promise<string> {
  const { stdout } = await runCommand("python3", ["--version"]);
  const version = stdout.trim().replace(/^Python\s+/, "");
  const [major = "0", minor = "0"] = version.split(".");
  if (Number(major) < 3 || (Number(major) === 3 && Number(minor) < 9)) {
    throw new Error(`python3 >= 3.9 required, found ${version}`);
  }
  return version;
}

export async function checkOptuna(): Promise<string> {
  try {
    const { stdout } = await runCommand("python3", ["-c", "import optuna; print(optuna.__version__)"]);
    return stdout.trim();
  } catch (error) {
    throw new Error(`Optuna is required: python3 -m pip install optuna (${String(error)})`);
  }
}

async function checkCentaurPackages(): Promise<{ optuna: string; cmaes: string }> {
  let stdout: string;
  try {
    ({ stdout } = await runCommand("python3", [
      "-c",
      "import cmaes, optuna; print(optuna.__version__); print(cmaes.__version__)"
    ]));
  } catch (error) {
    throw new Error(`Centaur requires Optuna and cmaes: python3 -m pip install 'optuna>=4.8,<5' 'cmaes>=0.12' (${String(error)})`);
  }
  const [optuna = "", cmaes = ""] = stdout.trim().split(/\r?\n/);
  if (!isSupportedCentaurOptuna(optuna)) {
    throw new Error(`Centaur requires Optuna >= 4.8.0 and < 5, found ${optuna || "unknown"}`);
  }
  if (!isAtLeastVersion(cmaes, 0, 12)) {
    throw new Error(`Centaur requires cmaes >= 0.12, found ${cmaes || "unknown"}`);
  }
  return { optuna, cmaes };
}

function isSupportedCentaurOptuna(version: string): boolean {
  const [major, minor] = version.split(".").map(Number);
  return major === 4 && Number.isFinite(minor) && minor >= 8;
}

function isAtLeastVersion(version: string, minimumMajor: number, minimumMinor: number): boolean {
  const [major, minor] = version.split(".").map(Number);
  return Number.isFinite(major) && Number.isFinite(minor) && (major > minimumMajor || (major === minimumMajor && minor >= minimumMinor));
}

export async function checkHeadless(agent: string, options: { allowFallback?: boolean } = {}): Promise<string> {
  const bin = process.env.AUTOTUNE_HEADLESS_BIN ?? "headless";
  let stdout = "";
  let stderr = "";
  try {
    ({ stdout, stderr } = await runCommand(bin, ["--check"]));
  } catch (error) {
    if (bin === "headless" && isMissingExecutable(error) && options.allowFallback !== false) {
      ({ stdout, stderr } = await runCommand("npx", ["-y", FALLBACK_HEADLESS_PACKAGE, "--check"]));
    } else if (bin === "headless" && isMissingExecutable(error) && options.allowFallback === false) {
      throw new Error("Centaur requires an installed headless executable on PATH or AUTOTUNE_HEADLESS_BIN");
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

export async function checkRuntime(invocation: Invocation): Promise<string> {
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
