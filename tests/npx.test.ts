import { chmod, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import process from "node:process";
import { npmCliCandidates, npxCliCandidates, resolveNpmCommand } from "../src/npx.js";

describe("npx CLI resolution", () => {
  it("covers standard Windows and Unix Node.js layouts", () => {
    expect(npxCliCandidates("C:\\Program Files\\nodejs\\node.exe", path.win32))
      .toContain("C:\\Program Files\\nodejs\\node_modules\\npm\\bin\\npx-cli.js");
    expect(npxCliCandidates("/opt/node/bin/node"))
      .toContain("/opt/node/lib/node_modules/npm/bin/npx-cli.js");
  });

  it("only derives executable JavaScript from the Node installation", () => {
    expect(npxCliCandidates("/opt/node/bin/node"))
      .not.toContain("/tmp/forged/npm/bin/npx-cli.js");
  });

  it("finds npm beside the active Node.js runtime", () => {
    expect(npmCliCandidates("C:\\Program Files\\nodejs\\node.exe", path.win32))
      .toContain("C:\\Program Files\\nodejs\\node_modules\\npm\\bin\\npm-cli.js");
    expect(npmCliCandidates("/opt/node/bin/node"))
      .toContain("/opt/node/lib/node_modules/npm/bin/npm-cli.js");
  });

  it.skipIf(process.platform === "win32")("does not trust npm from PATH", async () => {
    const directory = await mkdtemp(path.join(tmpdir(), "autotune-hostile-npm-"));
    const hostileNpm = path.join(directory, "npm");
    const previousPath = process.env.PATH;
    await writeFile(hostileNpm, "#!/bin/sh\nexit 99\n", "utf8");
    await chmod(hostileNpm, 0o755);
    process.env.PATH = directory;

    try {
      const command = await resolveNpmCommand();
      expect(command.command).toBe(process.execPath);
      expect(command.args).not.toContain(hostileNpm);
    } finally {
      if (previousPath === undefined) delete process.env.PATH;
      else process.env.PATH = previousPath;
      await rm(directory, { recursive: true, force: true });
    }
  });
});
