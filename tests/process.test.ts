import { runCommand } from "../src/process.js";

describe("runCommand", () => {
  it("captures stdout", async () => {
    await expect(runCommand("node", ["-e", "console.log('ok')"])).resolves.toMatchObject({ stdout: "ok\n" });
  });

  it("rejects non-zero exits with stderr", async () => {
    await expect(runCommand("node", ["-e", "console.error('bad'); process.exit(2)"])).rejects.toThrow(/bad/);
  });
});
