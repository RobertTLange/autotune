import { runCommand } from "../src/process.js";

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
});
