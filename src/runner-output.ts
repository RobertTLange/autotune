import { spawnSync } from "node:child_process";
import {
  accessSync,
  closeSync,
  constants,
  fstatSync,
  mkdtempSync,
  openSync,
  opendirSync,
  readFileSync,
  readSync,
  realpathSync,
  rmSync,
  statSync
} from "node:fs";
import os from "node:os";
import path from "node:path";
import { performance } from "node:perf_hooks";
import process from "node:process";
import { StringDecoder } from "node:string_decoder";

const PROCESS_SNAPSHOT_TIMEOUT_MS = 500;
const PROCESS_SNAPSHOT_MAX_BYTES = 1024 * 1024;
const OUTPUT_HOLDER_SCAN_TIMEOUT_MS = 1_000;
export const MAX_CAPTURE_BYTES = 16 * 1024 * 1024;
const MAX_PROCESSES_SCANNED = 4_096;
const MAX_FDS_PER_PROCESS = 1_024;
const MAX_TOTAL_FDS_SCANNED = 65_536;
const MAX_DRAIN_BYTES = 1024 * 1024;

interface FileIdentity {
  dev: number;
  ino: number;
  filePath: string;
}

export interface OutputCapture {
  directory: string;
  stdoutWriter: number | undefined;
  stderrWriter: number | undefined;
  stdout: CapturedStream;
  stderr: CapturedStream;
  identities: FileIdentity[];
}

interface CapturedStream {
  reader: number | undefined;
  decoder: StringDecoder;
  ended: boolean;
}

export interface CapturedOutput {
  text: string;
  bytes: number;
  ended: boolean;
}

interface ProcessIdentity {
  uid: number;
  generation: string;
}

export function createOutputCapture(): OutputCapture {
  const directory = realpathSync(mkdtempSync(path.join(os.tmpdir(), "autotune-runner-output-")));
  const stdoutPath = path.join(directory, "stdout.fifo");
  const stderrPath = path.join(directory, "stderr.fifo");
  const descriptors: number[] = [];
  try {
    createFifo(stdoutPath);
    createFifo(stderrPath);
    const [stdoutReader, stdoutWriter] = openFifo(stdoutPath);
    descriptors.push(stdoutReader, stdoutWriter);
    const [stderrReader, stderrWriter] = openFifo(stderrPath);
    descriptors.push(stderrReader, stderrWriter);
    return {
      directory,
      stdoutWriter,
      stderrWriter,
      stdout: { reader: stdoutReader, decoder: new StringDecoder("utf8"), ended: false },
      stderr: { reader: stderrReader, decoder: new StringDecoder("utf8"), ended: false },
      identities: [fileIdentity(stdoutReader, stdoutPath), fileIdentity(stderrReader, stderrPath)]
    };
  } catch (error) {
    for (const descriptor of descriptors) closeDescriptor(descriptor);
    removeCaptureDirectory(directory);
    throw error;
  }
}

function openFifo(filePath: string): [reader: number, writer: number] {
  const reader = openSync(filePath, constants.O_RDONLY | constants.O_NONBLOCK);
  try {
    const writer = openSync(filePath, constants.O_WRONLY);
    return [reader, writer];
  } catch (error) {
    closeDescriptor(reader);
    throw error;
  }
}

function createFifo(filePath: string): void {
  const command = resolveMkfifoCommand();
  if (!command) throw new Error("cannot create bounded runner output pipes: mkfifo unavailable");
  const result = spawnSync(command, [filePath], {
    encoding: "utf8",
    env: { PATH: path.dirname(command), LC_ALL: "C" },
    killSignal: "SIGKILL",
    maxBuffer: PROCESS_SNAPSHOT_MAX_BYTES,
    timeout: PROCESS_SNAPSHOT_TIMEOUT_MS,
    windowsHide: true
  });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(`mkfifo exited with ${result.status}: ${result.stderr.trim()}`);
  }
}

export function cleanupOutputCapture(capture: OutputCapture): void {
  closeOutputReaders(capture);
  closeOutputWriters(capture);
  removeCaptureDirectory(capture.directory);
}

function closeOutputReaders(capture: OutputCapture): void {
  for (const stream of [capture.stdout, capture.stderr]) {
    const reader = stream.reader;
    stream.reader = undefined;
    if (reader !== undefined) closeDescriptor(reader);
  }
}

export function closeOutputWriters(capture: OutputCapture): void {
  const stdoutWriter = capture.stdoutWriter;
  const stderrWriter = capture.stderrWriter;
  capture.stdoutWriter = undefined;
  capture.stderrWriter = undefined;
  if (stdoutWriter !== undefined) closeDescriptor(stdoutWriter);
  if (stderrWriter !== undefined) closeDescriptor(stderrWriter);
}

export function drainCapturedOutput(
  capture: OutputCapture,
  name: "stdout" | "stderr"
): CapturedOutput {
  const stream = capture[name];
  if (stream.ended || stream.reader === undefined) {
    return { text: "", bytes: 0, ended: stream.ended };
  }
  const chunks: string[] = [];
  let bytes = 0;
  while (bytes < MAX_DRAIN_BYTES) {
    const buffer = Buffer.allocUnsafe(Math.min(64 * 1024, MAX_DRAIN_BYTES - bytes));
    let count: number;
    try {
      count = readSync(stream.reader, buffer, 0, buffer.length, null);
    } catch (error) {
      if (isRetryableRead(error)) break;
      throw error;
    }
    if (count === 0) {
      stream.ended = true;
      chunks.push(stream.decoder.end());
      break;
    }
    bytes += count;
    chunks.push(stream.decoder.write(buffer.subarray(0, count)));
  }
  return { text: chunks.join(""), bytes, ended: stream.ended };
}

function isRetryableRead(error: unknown): boolean {
  return error instanceof Error && "code" in error
    && (error.code === "EAGAIN" || error.code === "EWOULDBLOCK");
}

function closeDescriptor(descriptor: number): void {
  try {
    closeSync(descriptor);
  } catch {
    // Descriptor may already be closed during spawn failure cleanup.
  }
}

function removeCaptureDirectory(directory: string): void {
  try {
    rmSync(directory, { recursive: true, force: true });
  } catch {
    // Capture cleanup must not prevent promise settlement.
  }
}

function fileIdentity(descriptor: number, filePath: string): FileIdentity {
  const metadata = fstatSync(descriptor);
  return { dev: metadata.dev, ino: metadata.ino, filePath };
}

export function signalOutputHolders(
  capture: OutputCapture | undefined,
  parentPid: number | undefined
): void {
  if (!capture) return;
  const deadline = performance.now() + OUTPUT_HOLDER_SCAN_TIMEOUT_MS;
  for (const pid of outputHolderProcessIds(capture, deadline)) {
    if (performance.now() >= deadline) break;
    if (pid === process.pid || pid === parentPid) continue;
    const identity = readProcessIdentity(pid, deadline);
    if (!identity || identity.uid !== process.getuid?.()) continue;
    if (!processHoldsCapture(pid, capture, deadline)) continue;
    const revalidatedIdentity = readProcessIdentity(pid, deadline);
    if (!sameProcessIdentity(identity, revalidatedIdentity)) continue;
    try {
      // Portable Node has no stable macOS process handle. The same-UID generation
      // and FIFO identity checks minimize the remaining PID/syscall race.
      process.kill(pid, "SIGKILL");
    } catch {
      // Holder may have exited between discovery and signaling.
    }
  }
}

function outputHolderProcessIds(capture: OutputCapture, deadline: number): number[] {
  if (process.platform === "linux") {
    return linuxFileHolderProcessIds(capture, deadline);
  }
  const command = resolveLsofCommand();
  if (!command) return [];
  const result = runLsof(command, undefined, deadline);
  if (!result.error && result.status === 0) {
    return parseLsofProcessIds(result.stdout, capture.identities);
  }
  return [];
}

function runLsof(command: string, pid: number | undefined, deadline: number) {
  const selector = pid ? ["-p", String(pid)] : ["-u", String(process.getuid?.())];
  return spawnSync(command, ["-n", "-P", "-F", "pDin", "-a", ...selector, "-d", "1,2"], {
    encoding: "utf8",
    env: { PATH: path.dirname(command), LC_ALL: "C" },
    killSignal: "SIGKILL",
    maxBuffer: PROCESS_SNAPSHOT_MAX_BYTES,
    timeout: Math.max(1, Math.ceil(deadline - performance.now())),
    windowsHide: true
  });
}

function linuxFileHolderProcessIds(capture: OutputCapture, deadline: number): number[] {
  const holders: number[] = [];
  let processes: ReturnType<typeof opendirSync>;
  try {
    processes = opendirSync("/proc");
  } catch {
    return [];
  }
  let scannedProcesses = 0;
  let scannedDescriptors = 0;
  try {
    let entry;
    while (
      performance.now() < deadline
      && scannedProcesses < MAX_PROCESSES_SCANNED
      && scannedDescriptors < MAX_TOTAL_FDS_SCANNED
      && (entry = processes.readSync()) !== null
    ) {
      if (!/^\d+$/.test(entry.name)) continue;
      scannedProcesses += 1;
      const pid = Number(entry.name);
      if (linuxProcessHoldsCapture(pid, capture, deadline, (count) => {
        scannedDescriptors += count;
        return scannedDescriptors < MAX_TOTAL_FDS_SCANNED;
      })) {
        holders.push(pid);
      }
    }
  } finally {
    processes.closeSync();
  }
  return holders;
}

function processHoldsCapture(pid: number, capture: OutputCapture, deadline: number): boolean {
  if (process.platform === "linux") {
    return linuxProcessHoldsCapture(pid, capture, deadline);
  }
  const command = resolveLsofCommand();
  if (!command) return false;
  const result = runLsof(command, pid, deadline);
  return !result.error
    && result.status === 0
    && parseLsofProcessIds(result.stdout, capture.identities).includes(pid);
}

function linuxProcessHoldsCapture(
  pid: number,
  capture: OutputCapture,
  deadline: number,
  countDescriptor: (count: number) => boolean = () => true
): boolean {
  let descriptors: ReturnType<typeof opendirSync>;
  try {
    descriptors = opendirSync(`/proc/${pid}/fd`);
  } catch {
    return false;
  }
  let scanned = 0;
  try {
    let entry;
    while (
      scanned < MAX_FDS_PER_PROCESS
      && performance.now() < deadline
      && (entry = descriptors.readSync()) !== null
    ) {
      scanned += 1;
      if (!countDescriptor(1)) return false;
      try {
        const metadata = statSync(`/proc/${pid}/fd/${entry.name}`);
        if (capture.identities.some((identity) => sameFileIdentity(metadata, identity))) return true;
      } catch {
        // Descriptor may close while it is being inspected.
      }
    }
  } finally {
    descriptors.closeSync();
  }
  return false;
}

function readProcessIdentity(pid: number, deadline: number): ProcessIdentity | undefined {
  if (performance.now() >= deadline) return undefined;
  if (process.platform === "linux") {
    try {
      const stat = readFileSync(`/proc/${pid}/stat`, "utf8");
      const status = readFileSync(`/proc/${pid}/status`, "utf8");
      const statFields = stat.slice(stat.lastIndexOf(")") + 2).trim().split(/\s+/);
      const uid = Number(/^Uid:\s+(\d+)/m.exec(status)?.[1]);
      const startTime = statFields[19];
      if (!Number.isInteger(uid) || !startTime) return undefined;
      return { uid, generation: startTime };
    } catch {
      return undefined;
    }
  }
  const command = resolvePosixPsCommand();
  if (!command) return undefined;
  const result = spawnSync(
    command,
    ["-o", "pid=,uid=,lstart=,pgid=,comm=", "-p", String(pid)],
    {
      encoding: "utf8",
      env: { PATH: path.dirname(command), LC_ALL: "C" },
      killSignal: "SIGKILL",
      maxBuffer: PROCESS_SNAPSHOT_MAX_BYTES,
      timeout: Math.max(1, Math.ceil(deadline - performance.now())),
      windowsHide: true
    }
  );
  if (result.error || result.status !== 0) return undefined;
  const line = result.stdout.trim();
  const [pidText, uidText] = line.split(/\s+/, 2);
  if (Number(pidText) !== pid || !Number.isInteger(Number(uidText))) return undefined;
  return { uid: Number(uidText), generation: line };
}

function sameProcessIdentity(
  expected: ProcessIdentity,
  actual: ProcessIdentity | undefined
): boolean {
  return actual !== undefined
    && actual.uid === expected.uid
    && actual.generation === expected.generation;
}

function parseLsofProcessIds(source: string, identities: FileIdentity[]): number[] {
  const holders = new Set<number>();
  let pid: number | undefined;
  let dev: number | undefined;
  let ino: number | undefined;
  for (const line of source.split("\n")) {
    if (line.startsWith("p")) {
      pid = Number(line.slice(1));
      dev = undefined;
      ino = undefined;
    } else if (line.startsWith("D")) {
      dev = Number(line.slice(1));
    } else if (line.startsWith("i")) {
      ino = Number(line.slice(1));
      if (pid && dev !== undefined && matchesFileIdentity(identities, dev, ino)) holders.add(pid);
    } else if (line.startsWith("n") && pid && ino !== undefined) {
      const filePath = line.slice(1);
      if (identities.some((identity) => identity.ino === ino && identity.filePath === filePath)) {
        holders.add(pid);
      }
    }
  }
  return [...holders];
}

function matchesFileIdentity(identities: FileIdentity[], dev: number, ino: number): boolean {
  return identities.some((identity) => identity.dev === dev && identity.ino === ino);
}

function sameFileIdentity(metadata: { dev: number; ino: number }, identity: FileIdentity): boolean {
  return metadata.dev === identity.dev && metadata.ino === identity.ino;
}

function resolveLsofCommand(): string | undefined {
  return resolveCommand(["/usr/sbin/lsof", "/usr/bin/lsof"]);
}

function resolveMkfifoCommand(): string | undefined {
  return resolveCommand(["/usr/bin/mkfifo", "/bin/mkfifo", "/run/current-system/sw/bin/mkfifo"]);
}

function resolvePosixPsCommand(): string | undefined {
  return resolveCommand(["/bin/ps", "/usr/bin/ps", "/run/current-system/sw/bin/ps"]);
}

function resolveCommand(candidates: string[]): string | undefined {
  for (const candidate of candidates) {
    try {
      accessSync(candidate, constants.X_OK);
      return candidate;
    } catch {
      // Keep scanning fixed operating-system locations.
    }
  }
  return undefined;
}
