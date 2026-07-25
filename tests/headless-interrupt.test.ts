const runCommandMock = vi.hoisted(() => vi.fn());

vi.mock("../src/process.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/process.js")>();
  return { ...actual, runCommand: runCommandMock };
});

import { runHeadless } from "../src/headless.js";
import { CommandInterruptedError } from "../src/process.js";

describe("runHeadless interruption", () => {
  it("preserves forwarded signal errors", async () => {
    const interrupted = new CommandInterruptedError("headless", "SIGINT");
    runCommandMock.mockRejectedValue(interrupted);

    await expect(runHeadless(["codex"], { cwd: process.cwd(), bin: "headless" })).rejects.toBe(interrupted);
  });
});
