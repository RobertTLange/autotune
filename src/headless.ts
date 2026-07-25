import { createHash } from "node:crypto";
import { chmod, lstat, mkdtemp, readFile, realpath, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { performance } from "node:perf_hooks";
import type { SearchSpace } from "./types.js";
import { parseSearchSpaceText } from "./search-space.js";
import { isCommandInterruptedError, runCommand } from "./process.js";
import { resolveNpmCommand } from "./npx.js";
import { npmInstallEnvironment, npxHeadlessEnvironment } from "./headless-environment.js";

const HEADLESS_TIMEOUT_MS = 10 * 60 * 1000;
const HEADLESS_MAX_OUTPUT_BYTES = 2 * 1024 * 1024;
export const FALLBACK_HEADLESS_PACKAGE = "@roberttlange/headless@0.4.0";
const HEADLESS_RUNTIME_LOCK = new URL("../resources/headless-runtime/package-lock.json", import.meta.url);
const HEADLESS_RUNTIME_LOCK_SHA256 = "b12466c830d5f87d7fd4673dfb1a1b1260cc67e3b97068e87635800be999fa7c";
const HEADLESS_RUNTIME_PACKAGE_JSON = `${JSON.stringify({
  name: "@roberttlange/autotune-headless-runtime",
  private: true,
  dependencies: { "@roberttlange/headless": "0.4.0" }
})}\n`;

export function extractHeadlessJson(output: string): SearchSpace {
  const candidates = collectCandidates(output, "search-space");
  for (const candidate of candidates) {
    try {
      return parseSearchSpaceText(candidate);
    } catch {
      // Try the next candidate.
    }
  }
  throw new Error("headless output did not contain a valid search space JSON object");
}

export function extractHeadlessObject(output: string): Record<string, unknown> {
  const candidates = collectCandidates(output, "object");
  let fallback: Record<string, unknown> | undefined;
  for (const candidate of candidates) {
    try {
      const value = JSON.parse(candidate) as unknown;
      if (value && typeof value === "object" && !Array.isArray(value)) {
        const record = value as Record<string, unknown>;
        if (typeof record.code === "string") {
          return record;
        }
        fallback ??= record;
      }
    } catch {
      // Try the next candidate.
    }
  }
  if (fallback) {
    return fallback;
  }
  throw new Error("headless output did not contain a valid JSON object");
}

export async function runHeadless(
  args: string[],
  options: { cwd: string; bin?: string; resolveNpm?: typeof resolveNpmCommand }
): Promise<string> {
  const environmentConfigured = process.env.AUTOTUNE_HEADLESS_BIN !== undefined;
  const explicitlyConfigured = options.bin !== undefined || environmentConfigured;
  const configured = options.bin ?? process.env.AUTOTUNE_HEADLESS_BIN;
  if (explicitlyConfigured && !configured?.trim()) {
    throw new Error("configured headless executable must not be empty");
  }
  if (!explicitlyConfigured) {
    const environment = npxHeadlessEnvironment({ agent: args[0] ?? "", model: optionValue(args, "--model") });
    return runHeadlessFallback(args, environment, options.resolveNpm);
  }
  const bin = configured as string;
  try {
    return await spawnCapture(bin, args, options.cwd);
  } catch (error) {
    throw error;
  }
}

export async function runHeadlessFallback(
  args: string[],
  environment: NodeJS.ProcessEnv,
  resolveNpm: typeof resolveNpmCommand = resolveNpmCommand,
  timeoutMs = HEADLESS_TIMEOUT_MS,
  maxOutputBytes = HEADLESS_MAX_OUTPUT_BYTES
): Promise<string> {
  const deadline = performance.now() + timeoutMs;
  const executionDirectory = await mkdtemp(path.join(tmpdir(), "autotune-headless-npm-"));
  try {
    await chmod(executionDirectory, 0o700);
    const lock = await readFile(HEADLESS_RUNTIME_LOCK);
    if (createHash("sha256").update(lock).digest("hex") !== HEADLESS_RUNTIME_LOCK_SHA256) {
      throw new Error("Headless runtime lock failed its integrity check");
    }
    await Promise.all([
      writeFile(path.join(executionDirectory, "package.json"), HEADLESS_RUNTIME_PACKAGE_JSON, { mode: 0o600 }),
      writeFile(path.join(executionDirectory, "package-lock.json"), lock, { mode: 0o600 })
    ]);
    const npm = await resolveNpm();
    await runCommand(
      npm.command,
      [...npm.args, "ci", "--ignore-scripts", "--no-audit", "--no-fund"],
      {
        cwd: executionDirectory,
        env: {
          ...npmInstallEnvironment(),
          NPM_CONFIG_IGNORE_SCRIPTS: "true",
          NPM_CONFIG_AUDIT: "false",
          NPM_CONFIG_FUND: "false"
        },
        timeoutMs: remainingTimeout(deadline),
        maxOutputBytes
      }
    );
    const headlessCli = path.join(
      executionDirectory,
      "node_modules",
      "@roberttlange",
      "headless",
      "dist",
      "cli.js"
    );
    const modulesDirectory = path.join(executionDirectory, "node_modules");
    const [metadata, resolvedCli, resolvedModulesDirectory] = await Promise.all([
      lstat(headlessCli),
      realpath(headlessCli),
      realpath(modulesDirectory)
    ]);
    const relativeCli = path.relative(resolvedModulesDirectory, resolvedCli);
    if (
      !metadata.isFile()
      || metadata.isSymbolicLink()
      || relativeCli.startsWith("..")
      || path.isAbsolute(relativeCli)
    ) {
      throw new Error("installed Headless fallback entry point was unsafe");
    }
    return await spawnCapture(
      process.execPath,
      [resolvedCli, ...args],
      executionDirectory,
      environment,
      remainingTimeout(deadline),
      maxOutputBytes
    );
  } finally {
    await rm(executionDirectory, { recursive: true, force: true }).catch(() => undefined);
  }
}

async function spawnCapture(
  bin: string,
  args: string[],
  cwd: string,
  env?: NodeJS.ProcessEnv,
  timeoutMs = HEADLESS_TIMEOUT_MS,
  maxOutputBytes = HEADLESS_MAX_OUTPUT_BYTES
): Promise<string> {
  try {
    const { stdout, stderr } = await runCommand(bin, args, {
      cwd,
      env,
      timeoutMs,
      maxOutputBytes
    });
    return stdout + (stderr ? `\n${stderr}` : "");
  } catch (error) {
    if (isMissingExecutable(error) || isCommandInterruptedError(error)) {
      throw error;
    }
    throw new Error(`headless failed: ${error instanceof Error ? error.message : String(error)}`);
  }
}

function remainingTimeout(deadline: number): number {
  const remaining = deadline - performance.now();
  if (remaining <= 0) throw new Error("Headless fallback timed out during installation");
  return remaining;
}

function optionValue(args: string[], option: string): string | undefined {
  const index = args.indexOf(option);
  return index >= 0 ? args[index + 1] : undefined;
}

function isMissingExecutable(error: unknown): boolean {
  return typeof error === "object" && error !== null && "code" in error && error.code === "ENOENT";
}

function collectCandidates(output: string, mode: "search-space" | "object"): string[] {
  const structured: string[] = [];
  const fallback: string[] = [];
  const trimmed = output.trim();
  if (trimmed) {
    fallback.push(trimmed);
  }

  for (const line of output.split(/\r?\n/)) {
    const value = extractTextFromJsonLine(line, mode);
    if (value) {
      structured.push(value);
    }
  }

  const fenced = [...output.matchAll(/```(?:json)?\s*([\s\S]*?)```/gi)].map((match) => match[1] ?? "");
  fallback.push(...fenced);

  const object = extractBalancedObject(output);
  if (object) {
    fallback.push(object);
  }

  return [...structured.reverse(), ...fallback.reverse()].filter(Boolean);
}

function extractTextFromJsonLine(line: string, mode: "search-space" | "object"): string | undefined {
  try {
    const parsed = JSON.parse(line);
    return findText(parsed, mode);
  } catch {
    return undefined;
  }
}

function findText(value: unknown, mode: "search-space" | "object"): string | undefined {
  if (typeof value === "string") {
    const trimmed = value.trim();
    return mode === "object" ? (trimmed.startsWith("{") ? value : undefined) : value.includes("parameters") ? value : undefined;
  }
  if (Array.isArray(value)) {
    for (const item of value) {
      const found = findText(item, mode);
      if (found) {
        return found;
      }
    }
  }
  if (value && typeof value === "object") {
    const record = value as Record<string, unknown>;
    for (const key of ["text", "content", "message", "result", "final"]) {
      const found = findText(record[key], mode);
      if (found) {
        return found;
      }
    }
    for (const nested of Object.values(record)) {
      const found = findText(nested, mode);
      if (found) {
        return found;
      }
    }
  }
  return undefined;
}

function extractBalancedObject(text: string): string | undefined {
  const start = text.indexOf("{");
  if (start === -1) {
    return undefined;
  }

  let depth = 0;
  let quote: string | undefined;
  let escaping = false;
  for (let index = start; index < text.length; index += 1) {
    const char = text[index];
    if (escaping) {
      escaping = false;
      continue;
    }
    if (char === "\\") {
      escaping = true;
      continue;
    }
    if (char === "\"" && !escaping) {
      quote = quote ? undefined : char;
      continue;
    }
    if (quote) {
      continue;
    }
    if (char === "{") {
      depth += 1;
    } else if (char === "}") {
      depth -= 1;
      if (depth === 0) {
        return text.slice(start, index + 1);
      }
    }
  }
  return undefined;
}
