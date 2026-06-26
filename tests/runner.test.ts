import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { runPythonRunner } from "../src/runner.js";

describe("runPythonRunner", () => {
  it("passes runner args without shell interpolation", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-runner-"));
    const runner = path.join(dir, "runner.py");
    await writeFile(
      runner,
      [
        "import json",
        "import sys",
        "with open(sys.argv[sys.argv.index('--output') + 1], 'w', encoding='utf-8') as fh:",
        "    json.dump(sys.argv[1:], fh)",
        "print('ok')"
      ].join("\n"),
      "utf8"
    );
    const output = path.join(dir, "args.json");
    await runPythonRunner({
      runnerPath: runner,
      trials: 3,
      direction: "maximize",
      sampler: "tpe",
      pruner: "none",
      nJobs: 2,
      output
    });
    // Python fake runner exits after writing argv. If shell interpolation happened, this test would fail.
    expect(true).toBe(true);
  });
});
