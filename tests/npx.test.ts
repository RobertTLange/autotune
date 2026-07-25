import path from "node:path";
import { npxCliCandidates } from "../src/npx.js";

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
});
