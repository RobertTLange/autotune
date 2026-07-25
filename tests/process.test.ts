import { resolveWindowsSystemRoot, runCommand, windowsTaskkillInvocation } from "../src/process.js";

describe("Windows process-tree support", () => {
  it("prefers the standard trusted system root when taskkill exists", () => {
    expect(resolveWindowsSystemRoot(() => true, "D:\\Windows")).toBe("C:\\Windows");
  });

  it("accepts a validated nonstandard Windows system root", () => {
    const onlyDriveDTaskkillExists = (candidate: string) => candidate === "D:\\Windows\\System32\\taskkill.exe";
    expect(resolveWindowsSystemRoot(onlyDriveDTaskkillExists, "D:\\Windows")).toBe("D:\\Windows");
    expect(resolveWindowsSystemRoot(() => false, "relative\\Windows")).toBe("C:\\Windows");
    expect(resolveWindowsSystemRoot(() => false, "D:\\attacker")).toBe("C:\\Windows");
    expect(resolveWindowsSystemRoot(
      (candidate) => candidate === "D:\\temp\\Windows\\System32\\taskkill.exe",
      "D:\\temp\\Windows"
    )).toBe("C:\\Windows");
    expect(resolveWindowsSystemRoot(
      (candidate) => candidate === "\\\\server\\share\\Windows\\System32\\taskkill.exe",
      "\\\\server\\share\\Windows"
    )).toBe("C:\\Windows");
  });

  it("builds a bounded taskkill invocation from the validated system root", () => {
    const onlyDriveDTaskkillExists = (candidate: string) => candidate === "D:\\Windows\\System32\\taskkill.exe";
    const invocation = windowsTaskkillInvocation(42, onlyDriveDTaskkillExists, "D:\\Windows");

    expect(invocation).toEqual({
      command: "D:\\Windows\\System32\\taskkill.exe",
      args: ["/PID", "42", "/T", "/F"],
      options: {
        env: { SystemRoot: "D:\\Windows", windir: "D:\\Windows" },
        stdio: "ignore",
        timeout: 5_000,
        windowsHide: true
      }
    });
  });
});

describe("runCommand", () => {
  it("captures stdout", async () => {
    await expect(runCommand("node", ["-e", "console.log('ok')"])).resolves.toMatchObject({ stdout: "ok\n" });
  });

  it("rejects non-zero exits with stderr", async () => {
    await expect(runCommand("node", ["-e", "console.error('bad'); process.exit(2)"])).rejects.toThrow(/bad/);
  });

  it("truncates captured output", async () => {
    const result = await runCommand("node", ["-e", "process.stdout.write('x'.repeat(20))"], { maxOutputBytes: 8 });

    expect(result.stdout).toBe("xxxxxxxx");
  });

  it("keeps trailing output when truncating", async () => {
    const result = await runCommand("node", ["-e", "process.stdout.write('prefix-' + 'x'.repeat(20) + '-suffix')"], {
      maxOutputBytes: 10
    });

    expect(result.stdout).toBe("xxx-suffix");
  });

  it("times out long-running commands", async () => {
    await expect(runCommand("node", ["-e", "setTimeout(() => {}, 1000)"], { timeoutMs: 10 })).rejects.toThrow(
      /timed out/
    );
  });

  it("waits for signal cleanup instead of exiting synchronously", async () => {
    const previousExitCode = process.exitCode;
    const priorHandlers = new Set(process.rawListeners("SIGTERM"));
    const running = runCommand(process.execPath, ["-e", "setInterval(() => {}, 1000)"]);
    const outcome = running.catch((error: unknown) => error);
    const handler = process.rawListeners("SIGTERM").find((listener) => !priorHandlers.has(listener));
    const exit = vi.spyOn(process, "exit").mockImplementation(((code?: number) => {
      throw new Error(`unexpected process.exit(${code})`);
    }) as never);

    try {
      expect(handler).toBeDefined();
      expect(() => (handler as () => void)()).not.toThrow();
      expect(process.rawListeners("SIGTERM")).toContain(handler);
      expect(() => (handler as () => void)()).not.toThrow();
      await expect(outcome).resolves.toEqual(expect.objectContaining({
        code: "ERR_COMMAND_INTERRUPTED",
        message: expect.stringMatching(/interrupted by SIGTERM/i)
      }));
      expect(exit).not.toHaveBeenCalled();
      expect(process.exitCode).toBe(143);
    } finally {
      exit.mockRestore();
      process.exitCode = previousExitCode;
    }
  });

  it.runIf(process.platform === "win32")("times out commands whose grandchildren inherit output pipes", async () => {
    const source = `
const { spawn } = require("node:child_process");
spawn(process.execPath, ["-e", "setTimeout(() => {}, 30000)"], { stdio: ["ignore", "inherit", "inherit"] });
setTimeout(() => {}, 30000);
`;
    const started = Date.now();
    await expect(runCommand(process.execPath, ["-e", source], { timeoutMs: 25 }))
      .rejects.toThrow(/timed out/);
    expect(Date.now() - started).toBeLessThan(5_000);
  });
});
