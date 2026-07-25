import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { CommandInterruptedError } from "../src/process.js";

const runHeadlessMock = vi.hoisted(() => vi.fn());

vi.mock("../src/headless.js", () => ({
  extractHeadlessJson: vi.fn(),
  extractHeadlessObject: vi.fn(),
  runHeadless: runHeadlessMock
}));

import { analyzeScript } from "../src/analyze.js";

describe("analyzeScript interruption", () => {
  it("does not retry Headless after a forwarded signal", async () => {
    const interrupted = new CommandInterruptedError("headless", "SIGTERM");
    runHeadlessMock.mockRejectedValue(interrupted);

    await expect(
      analyzeScript({
        invocation: {
          language: "python",
          command: ["python3", "train.py"],
          script: "train.py"
        },
        workDir: await mkdtemp(path.join(tmpdir(), "autotune-interrupt-")),
        agent: "codex"
      })
    ).rejects.toBe(interrupted);
    expect(runHeadlessMock).toHaveBeenCalledTimes(1);
  });
});
