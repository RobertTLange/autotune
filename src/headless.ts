import type { SearchSpace } from "./types.js";
import { parseSearchSpaceText } from "./search-space.js";
import { runCommand } from "./process.js";

const HEADLESS_TIMEOUT_MS = 10 * 60 * 1000;
const HEADLESS_MAX_OUTPUT_BYTES = 2 * 1024 * 1024;
export const FALLBACK_HEADLESS_PACKAGE = "@roberttlange/headless@0.4.0";

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

export async function runHeadless(args: string[], options: { cwd: string; bin?: string }): Promise<string> {
  const bin = options.bin ?? process.env.AUTOTUNE_HEADLESS_BIN ?? "headless";
  try {
    return await spawnCapture(bin, args, options.cwd);
  } catch (error) {
    if (!options.bin && !process.env.AUTOTUNE_HEADLESS_BIN && isMissingExecutable(error)) {
      return spawnCapture("npx", ["-y", FALLBACK_HEADLESS_PACKAGE, ...args], options.cwd);
    }
    throw error;
  }
}

async function spawnCapture(bin: string, args: string[], cwd: string): Promise<string> {
  try {
    const { stdout, stderr } = await runCommand(bin, args, {
      cwd,
      timeoutMs: HEADLESS_TIMEOUT_MS,
      maxOutputBytes: HEADLESS_MAX_OUTPUT_BYTES
    });
    return stdout + (stderr ? `\n${stderr}` : "");
  } catch (error) {
    if (isMissingExecutable(error)) {
      throw error;
    }
    throw new Error(`headless failed: ${error instanceof Error ? error.message : String(error)}`);
  }
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
