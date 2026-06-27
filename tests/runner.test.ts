import { mkdtemp, readFile, writeFile } from "node:fs/promises";
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
      studyName: "train_autotune",
      output
    });
    const args = JSON.parse(await readFile(output, "utf8")) as string[];
    expect(args).toEqual(expect.arrayContaining(["--study-name", "train_autotune"]));
  });

  it("forwards runner stderr progress", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-runner-progress-"));
    const runner = path.join(dir, "runner.py");
    await writeFile(
      runner,
      [
        "import json",
        "import sys",
        "print('trial 1/3 complete', file=sys.stderr)",
        "with open(sys.argv[sys.argv.index('--output') + 1], 'w', encoding='utf-8') as fh:",
        "    json.dump({'ok': True}, fh)"
      ].join("\n"),
      "utf8"
    );
    const output = path.join(dir, "results.json");
    let forwarded = "";
    const original = process.stderr.write;
    process.stderr.write = ((chunk: string | Uint8Array) => {
      forwarded += chunk.toString();
      return true;
    }) as typeof process.stderr.write;
    try {
      await runPythonRunner({
        runnerPath: runner,
        trials: 3,
        direction: "maximize",
        sampler: "tpe",
        pruner: "none",
        nJobs: 1,
        output
      });
    } finally {
      process.stderr.write = original;
    }
    expect(forwarded).toContain("trial 1/3 complete");
  });
});
