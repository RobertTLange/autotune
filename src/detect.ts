import { statSync } from "node:fs";
import path from "node:path";
import type { Invocation } from "./types.js";

export function splitCommand(command: string): string[] {
  const parts: string[] = [];
  let current = "";
  let quote: "'" | "\"" | undefined;
  let escaping = false;

  for (const char of command.trim()) {
    if (escaping) {
      current += char;
      escaping = false;
      continue;
    }

    if (char === "\\" && quote !== "'") {
      escaping = true;
      continue;
    }

    if (char === "'" || char === "\"") {
      if (quote === char) {
        quote = undefined;
      } else if (!quote) {
        quote = char;
      } else {
        current += char;
      }
      continue;
    }

    if (/\s/.test(char) && !quote) {
      if (current.length > 0) {
        parts.push(current);
        current = "";
      }
      continue;
    }

    current += char;
  }

  if (escaping) {
    current += "\\";
  }
  if (quote) {
    throw new Error("unterminated quote in command override");
  }
  if (current.length > 0) {
    parts.push(current);
  }
  if (parts.length === 0) {
    throw new Error("command override cannot be empty");
  }
  return parts;
}

export function detectInvocation(scriptPath: string, commandOverride?: string | string[]): Invocation {
  const extension = path.extname(scriptPath).toLowerCase();
  const command = Array.isArray(commandOverride)
    ? commandOverride
    : commandOverride
      ? splitCommand(commandOverride)
      : undefined;

  if (command) {
    return {
      language: languageForExtension(extension),
      command,
      script: scriptPath,
      scriptArgument: scriptArgumentForOverride(command, scriptPath, extension)
    };
  }

  switch (extension) {
    case ".py":
      return { language: "python", command: ["python3"], script: scriptPath, scriptArgument: "append" };
    case ".sh":
      return { language: "shell", command: ["bash"], script: scriptPath, scriptArgument: "append" };
    case ".jl":
      return { language: "julia", command: ["julia"], script: scriptPath, scriptArgument: "append" };
    case ".r":
      return { language: "r", command: ["Rscript"], script: scriptPath, scriptArgument: "append" };
    case ".rb":
      return { language: "ruby", command: ["ruby"], script: scriptPath, scriptArgument: "append" };
    case "":
      if (isExecutable(scriptPath)) {
        return { language: "executable", command: [scriptPath], script: scriptPath, scriptArgument: "included" };
      }
      throw new Error(`cannot auto-detect runtime for non-executable file: ${scriptPath}`);
    default:
      throw new Error(`cannot auto-detect runtime for extension '${extension}'. Use --command.`);
  }
}

function scriptArgumentForOverride(
  command: string[],
  scriptPath: string,
  extension: string
): Invocation["scriptArgument"] {
  if (command.some((arg) => path.resolve(arg) === path.resolve(scriptPath))) {
    return "included";
  }
  return shouldAppendScriptForExtension(extension) ? "append" : "none";
}

function shouldAppendScriptForExtension(extension: string): boolean {
  return [".py", ".sh", ".jl", ".r", ".rb"].includes(extension);
}

function languageForExtension(extension: string): string {
  switch (extension) {
    case ".py":
      return "python";
    case ".sh":
      return "shell";
    case ".jl":
      return "julia";
    case ".r":
      return "r";
    case ".rb":
      return "ruby";
    case "":
      return "executable";
    default:
      return extension.slice(1) || "unknown";
  }
}

function isExecutable(filePath: string): boolean {
  try {
    const stat = statSync(filePath);
    return stat.isFile() && (stat.mode & 0o111) !== 0;
  } catch {
    return false;
  }
}
