import { access } from "node:fs/promises";
import { constants } from "node:fs";
import path from "node:path";
import { runCommand } from "./process.js";
import { FALLBACK_HEADLESS_PACKAGE } from "./headless.js";
import { ensurePythonRuntime, inspectPythonInterpreter } from "./python-runtime.js";
import type { Invocation } from "./types.js";

export interface PrerequisiteReport {
  python: string;
  optuna: string;
  cmaes?: string;
  headless: string;
  runtime: string;
  managedPython: boolean;
  pythonExecutable: string;
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
  const pythonRuntime = await ensurePythonRuntime({ includeCmaes: true });
  const headless = input.skipHeadless
    ? "skipped"
    : input.centaur
      ? await checkCentaurHeadless(input.agent)
      : await checkHeadless(input.agent);
  const runtime = await checkRuntime(input.invocation);
  return {
    python: pythonRuntime.pythonVersion,
    optuna: pythonRuntime.optunaVersion,
    ...(pythonRuntime.cmaesVersion ? { cmaes: pythonRuntime.cmaesVersion } : {}),
    headless,
    runtime,
    managedPython: pythonRuntime.managed,
    pythonExecutable: pythonRuntime.python
  };
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
  return (await inspectPythonInterpreter()).pythonVersion;
}

export async function checkOptuna(): Promise<string> {
  return (await ensurePythonRuntime({ includeCmaes: true })).optunaVersion;
}

export async function checkHeadless(agent: string): Promise<string> {
  const bin = process.env.AUTOTUNE_HEADLESS_BIN ?? "headless";
  let stdout = "";
  let stderr = "";
  try {
    ({ stdout, stderr } = await runCommand(bin, ["--check"]));
  } catch (error) {
    if (bin === "headless" && isMissingExecutable(error)) {
      ({ stdout, stderr } = await runCommand("npx", ["-y", FALLBACK_HEADLESS_PACKAGE, "--check"]));
    } else {
      throw error;
    }
  }
  const output = `${stdout}\n${stderr}`;
  if (!headlessListsAgent(output, agent)) {
    return `${bin} (agent ${agent} not listed by --check)`;
  }
  return `${bin} (${agent})`;
}

async function checkCentaurHeadless(agent: string): Promise<string> {
  const bin = process.env.AUTOTUNE_HEADLESS_BIN ?? "headless";
  let output: string;
  try {
    const { stdout, stderr } = await runCommand(bin, ["--check"]);
    output = `${stdout}\n${stderr}`;
  } catch (error) {
    if (isMissingExecutable(error)) {
      throw new Error("Centaur requires an installed headless executable on PATH or AUTOTUNE_HEADLESS_BIN");
    }
    throw error;
  }
  if (!headlessAgentAvailable(output, agent)) {
    throw new Error(`Centaur proposal agent ${agent} is not available according to headless --check`);
  }
  return `${bin} (${agent})`;
}

function headlessAgentAvailable(output: string, agent: string): boolean {
  const normalizedAgent = agent.trim().toLowerCase();
  return output.split(/\r?\n/).some((line) => {
    const columns = line.split("|").slice(1, -1).map((column) => column.trim());
    return columns[0]?.toLowerCase() === normalizedAgent && columns[1] === "✓";
  });
}

function headlessListsAgent(output: string, agent: string): boolean {
  const available = output.toLowerCase().split(/[^a-z0-9_-]+/);
  return available.includes(agent.trim().toLowerCase());
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
