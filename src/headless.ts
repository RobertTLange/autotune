import { spawn } from "node:child_process";
import type { SearchSpace } from "./types.js";
import { parseSearchSpaceText } from "./search-space.js";

export function extractHeadlessJson(output: string): SearchSpace {
  const candidates = collectCandidates(output);
  for (const candidate of candidates) {
    try {
      return parseSearchSpaceText(candidate);
    } catch {
      // Try the next candidate.
    }
  }
  throw new Error("headless output did not contain a valid search space JSON object");
}

export async function runHeadless(args: string[], options: { cwd: string; bin?: string }): Promise<string> {
  const bin = options.bin ?? process.env.AUTOTUNE_HEADLESS_BIN ?? "headless";
  try {
    return await spawnCapture(bin, args, options.cwd);
  } catch (error) {
    if (!options.bin && !process.env.AUTOTUNE_HEADLESS_BIN && isMissingExecutable(error)) {
      return spawnCapture("npx", ["-y", "@roberttlange/headless", ...args], options.cwd);
    }
    throw error;
  }
}

async function spawnCapture(bin: string, args: string[], cwd: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const child = spawn(bin, args, { cwd, stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";

    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
    });
    child.on("error", reject);
    child.on("close", (code) => {
      const output = stdout + (stderr ? `\n${stderr}` : "");
      if (code === 0) {
        resolve(output);
      } else {
        reject(new Error(`headless exited with ${code}: ${output.trim()}`));
      }
    });
  });
}

function isMissingExecutable(error: unknown): boolean {
  return typeof error === "object" && error !== null && "code" in error && error.code === "ENOENT";
}

function collectCandidates(output: string): string[] {
  const candidates: string[] = [];
  const trimmed = output.trim();
  if (trimmed) {
    candidates.push(trimmed);
  }

  for (const line of output.split(/\r?\n/)) {
    const value = extractTextFromJsonLine(line);
    if (value) {
      candidates.push(value);
    }
  }

  const fenced = [...output.matchAll(/```(?:json)?\s*([\s\S]*?)```/gi)].map((match) => match[1] ?? "");
  candidates.push(...fenced);

  const object = extractBalancedObject(output);
  if (object) {
    candidates.push(object);
  }

  return candidates.filter(Boolean).reverse();
}

function extractTextFromJsonLine(line: string): string | undefined {
  try {
    const parsed = JSON.parse(line);
    return findText(parsed);
  } catch {
    return undefined;
  }
}

function findText(value: unknown): string | undefined {
  if (typeof value === "string") {
    return value.includes("parameters") ? value : undefined;
  }
  if (Array.isArray(value)) {
    for (const item of value) {
      const found = findText(item);
      if (found) {
        return found;
      }
    }
  }
  if (value && typeof value === "object") {
    const record = value as Record<string, unknown>;
    for (const key of ["text", "content", "message", "result", "final"]) {
      const found = findText(record[key]);
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
