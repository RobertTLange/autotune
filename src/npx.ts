import { constants } from "node:fs";
import { access } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

export interface NpxCommand {
  command: string;
  args: string[];
}

export async function resolveNpxCommand(): Promise<NpxCommand> {
  const node = process.execPath;
  for (const cli of npxCliCandidates(node)) {
    if (await isExecutableFile(cli)) {
      return { command: node, args: [cli] };
    }
  }

  const executable = await findExecutableOnPath("npx");
  if (!executable) {
    throw new Error("npx was not found; install Node.js with npm to use the Headless fallback");
  }
  if (isWindowsBatchShim(executable)) {
    throw new Error("npx-cli.js was not found beside this Node.js/npm installation");
  }
  return { command: executable, args: [] };
}

type PathApi = Pick<typeof path, "dirname" | "join" | "resolve">;

export function npxCliCandidates(
  node: string,
  pathApi: PathApi = path
): string[] {
  const nodeDirectory = pathApi.dirname(node);
  const candidates = [
    pathApi.join(nodeDirectory, "node_modules", "npm", "bin", "npx-cli.js"),
    pathApi.resolve(nodeDirectory, "..", "lib", "node_modules", "npm", "bin", "npx-cli.js")
  ];
  return [...new Set(candidates.map((candidate) => pathApi.resolve(candidate)))];
}

export function isWindowsBatchShim(executable: string): boolean {
  return process.platform === "win32" && /\.(?:cmd|bat)$/i.test(executable);
}

export async function findExecutableOnPath(executable: string): Promise<string | undefined> {
  const extensions = process.platform === "win32"
    ? (process.env.PATHEXT ?? ".COM;.EXE;.BAT;.CMD").split(";")
    : [""];
  for (const directory of (process.env.PATH ?? "").split(path.delimiter)) {
    for (const extension of extensions) {
      const candidate = path.resolve(directory || ".", executable + extension.toLowerCase());
      if (await isExecutableFile(candidate)) {
        return candidate;
      }
    }
  }
  return undefined;
}

async function isExecutableFile(filePath: string): Promise<boolean> {
  try {
    await access(filePath, process.platform === "win32" ? constants.F_OK : constants.X_OK);
    return true;
  } catch {
    return false;
  }
}
