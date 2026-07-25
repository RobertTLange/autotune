import path from "node:path";
import { npxCliCandidates } from "../src/npx.js";

describe("npx CLI resolution", () => {
  it("prefers the npx CLI adjacent to npm_execpath", () => {
    const npm = path.join(path.sep, "tools", "npm", "bin", "npm-cli.js");
    expect(npxCliCandidates(path.join(path.sep, "tools", "bin", "node"), npm)[0])
      .toBe(path.join(path.sep, "tools", "npm", "bin", "npx-cli.js"));
  });

  it("covers standard Windows and Unix Node.js layouts", () => {
    expect(npxCliCandidates("C:\\Program Files\\nodejs\\node.exe", undefined, path.win32))
      .toContain("C:\\Program Files\\nodejs\\node_modules\\npm\\bin\\npx-cli.js");
    expect(npxCliCandidates("/opt/node/bin/node", undefined))
      .toContain("/opt/node/lib/node_modules/npm/bin/npx-cli.js");
  });
});
