import { access, mkdir, readdir, readFile, rename, writeFile } from "node:fs/promises";
import { constants } from "node:fs";
import { randomUUID } from "node:crypto";
import path from "node:path";

const ARTIFACT_DIR_NAME = "autotune";
const RUNS_DIR_NAME = "runs";
const LATEST_FILE_NAME = "latest.json";

export interface RunArtifactLayout {
  workDir: string;
  latestRuns: LatestRunPointer[];
}

export interface LatestRunPointer {
  latestPath: string;
  metadata: LatestRunMetadata;
}

export interface LatestRunMetadata {
  run_dir: string;
  work_dir: string;
  script: string;
  created_at: string;
}

export function resolveRunArtifactLayout(scriptPath: string, workDir: string | undefined): RunArtifactLayout {
  if (workDir !== undefined) {
    return { workDir: path.resolve(workDir), latestRuns: [] };
  }
  const artifactRoot = path.join(path.dirname(scriptPath), ARTIFACT_DIR_NAME);
  const scriptName = scriptArtifactName(scriptPath);
  const scriptRoot = path.join(artifactRoot, scriptName);
  const createdAt = new Date().toISOString();
  const runName = `${timestampForPath(createdAt)}-${randomUUID()}`;
  const runDir = path.join(scriptRoot, RUNS_DIR_NAME, runName);
  return {
    workDir: runDir,
    latestRuns: [
      latestRunPointer(scriptRoot, runDir, scriptPath, createdAt),
      latestRunPointer(artifactRoot, runDir, scriptPath, createdAt)
    ]
  };
}

export async function writeLatestRun(layout: RunArtifactLayout): Promise<void> {
  for (const latest of layout.latestRuns) {
    await writeLatestRunPointer(latest);
  }
}

export async function resolveResultsFile(location: string): Promise<string> {
  if (location.endsWith(".json")) {
    return path.resolve(location);
  }
  const resolved = path.resolve(location);
  const direct = path.join(resolved, "results.json");
  if (await fileExists(direct)) {
    return direct;
  }
  const latestRun = await latestRunResultsFile(resolved);
  if (latestRun) {
    return latestRun;
  }
  const legacy = await legacyArtifactRoot(resolved);
  if (legacy) {
    const legacyDirect = path.join(legacy, "results.json");
    if (await fileExists(legacyDirect)) {
      return legacyDirect;
    }
    const legacyLatest = await latestRunResultsFile(legacy);
    if (legacyLatest) {
      return legacyLatest;
    }
  }
  return direct;
}

export async function resolveRunDirectory(location: string): Promise<string> {
  const resolved = path.resolve(location);
  if (await fileExists(path.join(resolved, "search_space.yaml"))) {
    return resolved;
  }
  const latest = await latestRunDirectory(resolved);
  if (latest) {
    return latest;
  }
  const legacy = await legacyArtifactRoot(resolved);
  if (legacy) {
    return resolveRunDirectory(legacy);
  }
  return resolved;
}

async function latestRunResultsFile(location: string): Promise<string | undefined> {
  const runDir = await latestRunDirectory(location);
  return runDir ? path.join(runDir, "results.json") : undefined;
}

async function latestRunDirectory(location: string): Promise<string | undefined> {
  const direct = await latestRunFromRoot(location);
  if (direct) {
    return direct.runDir;
  }
  const children = await latestChildRuns(location);
  if (children.length === 0) {
    return undefined;
  }
  children.sort((left, right) => right.createdAt.localeCompare(left.createdAt));
  return children[0]?.runDir;
}

async function latestRunFromRoot(root: string): Promise<LatestRun | undefined> {
  const latestPath = path.join(root, LATEST_FILE_NAME);
  if (!await fileExists(latestPath)) {
    return undefined;
  }
  const metadata = JSON.parse(await readFile(latestPath, "utf8")) as Partial<LatestRunMetadata>;
  if (typeof metadata.run_dir !== "string" || metadata.run_dir.length === 0) {
    throw new Error(`invalid latest run metadata: ${latestPath}`);
  }
  return {
    runDir: resolveLatestRunDir(root, metadata.run_dir),
    createdAt: typeof metadata.created_at === "string" ? metadata.created_at : ""
  };
}

async function latestChildRuns(root: string): Promise<LatestRun[]> {
  if (!await fileExists(root)) {
    return [];
  }
  const entries = await readdir(root, { withFileTypes: true });
  const latest = await Promise.all(
    entries
      .filter((entry) => entry.isDirectory())
      .map((entry) => latestRunFromRoot(path.join(root, entry.name)))
  );
  return latest.filter((entry): entry is LatestRun => entry !== undefined);
}

async function fileExists(filePath: string): Promise<boolean> {
  try {
    await access(filePath, constants.F_OK);
    return true;
  } catch {
    return false;
  }
}

async function legacyArtifactRoot(resolved: string): Promise<string | undefined> {
  if (path.basename(resolved) !== ARTIFACT_DIR_NAME) {
    return undefined;
  }
  const legacy = path.join(path.dirname(resolved), `.${ARTIFACT_DIR_NAME}`);
  return await fileExists(legacy) ? legacy : undefined;
}

function resolveLatestRunDir(scriptRoot: string, runDir: string): string {
  if (path.isAbsolute(runDir)) {
    throw new Error(`invalid latest run metadata: run_dir must be relative`);
  }
  const resolved = path.resolve(scriptRoot, runDir);
  const relative = path.relative(scriptRoot, resolved);
  if (relative.startsWith("..") || path.isAbsolute(relative)) {
    throw new Error(`invalid latest run metadata: run_dir escapes artifact directory`);
  }
  return resolved;
}

function scriptArtifactName(scriptPath: string): string {
  return path.basename(scriptPath);
}

function timestampForPath(createdAt: string): string {
  return createdAt.replaceAll(":", "").replace(".", "");
}

async function writeLatestRunPointer(latest: LatestRunPointer): Promise<void> {
  await mkdir(path.dirname(latest.latestPath), { recursive: true });
  const tempPath = path.join(
    path.dirname(latest.latestPath),
    `.${path.basename(latest.latestPath)}.${process.pid}.${randomUUID()}.tmp`
  );
  await writeFile(tempPath, `${JSON.stringify(latest.metadata, null, 2)}\n`, "utf8");
  await rename(tempPath, latest.latestPath);
}

function latestRunPointer(root: string, runDir: string, scriptPath: string, createdAt: string): LatestRunPointer {
  return {
    latestPath: path.join(root, LATEST_FILE_NAME),
    metadata: {
      run_dir: path.relative(root, runDir),
      work_dir: runDir,
      script: scriptPath,
      created_at: createdAt
    }
  };
}

interface LatestRun {
  runDir: string;
  createdAt: string;
}
