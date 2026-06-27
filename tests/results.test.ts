import { mkdir, mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { readResults, renderResults } from "../src/results.js";

describe("results", () => {
  it("reads results files and renders best/top trials", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-results-"));
    await writeFile(
      path.join(dir, "results.json"),
      JSON.stringify({
        study_name: "study",
        direction: "minimize",
        n_trials: 2,
        best_trial: { number: 1, value: 0.1, params: { lr: 0.01 } },
        all_trials: [
          { number: 0, value: 0.2, params: { lr: 0.1 } },
          { number: 1, value: 0.1, params: { lr: 0.01 } }
        ]
      }),
      "utf8"
    );
    const result = await readResults(dir);
    const rendered = renderResults(result, 1);
    expect(rendered).toContain("Best trial: #1");
    expect(rendered).toContain("Trial  Value  Parameters");
    expect(rendered).toContain("#1     0.1    lr=0.01");
    expect(rendered).not.toContain("\t");
  });

  it("reads the latest timestamped run from an autotune script directory", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-results-latest-"));
    const scriptRoot = path.join(dir, "autotune", "train.py");
    const runDir = path.join(scriptRoot, "runs", "2026-06-27T181650000Z");
    await mkdir(runDir, { recursive: true });
    await writeFile(
      path.join(scriptRoot, "latest.json"),
      JSON.stringify({ run_dir: "runs/2026-06-27T181650000Z" }),
      "utf8"
    );
    await writeFile(
      path.join(runDir, "results.json"),
      JSON.stringify({
        study_name: "study",
        direction: "maximize",
        n_trials: 1,
        best_trial: { number: 0, value: 1, params: { x: 0.5 } },
        all_trials: [{ number: 0, value: 1, params: { x: 0.5 } }]
      }),
      "utf8"
    );

    const result = await readResults(scriptRoot);

    expect(result.best_trial?.params).toEqual({ x: 0.5 });
  });

  it("reads the only latest run from an autotune root directory", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-results-root-"));
    const autotuneRoot = path.join(dir, "autotune");
    const scriptRoot = path.join(autotuneRoot, "train.py");
    const runDir = path.join(scriptRoot, "runs", "2026-06-27T181650000Z");
    await mkdir(runDir, { recursive: true });
    await writeFile(
      path.join(autotuneRoot, "latest.json"),
      JSON.stringify({ run_dir: "train.py/runs/2026-06-27T181650000Z", created_at: "2026-06-27T18:16:50.000Z" }),
      "utf8"
    );
    await writeFile(
      path.join(runDir, "results.json"),
      JSON.stringify({
        study_name: "study",
        direction: "maximize",
        n_trials: 1,
        best_trial: { number: 0, value: 1, params: { x: 0.5 } },
        all_trials: [{ number: 0, value: 1, params: { x: 0.5 } }]
      }),
      "utf8"
    );

    const result = await readResults(autotuneRoot);

    expect(result.n_trials).toBe(1);
  });

  it("uses the newest child latest when a root latest pointer is missing", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-results-children-"));
    const autotuneRoot = path.join(dir, "autotune");
    const oldScriptRoot = path.join(autotuneRoot, "old.py");
    const newScriptRoot = path.join(autotuneRoot, "new.py");
    const oldRunDir = path.join(oldScriptRoot, "runs", "2026-06-27T181650000Z");
    const newRunDir = path.join(newScriptRoot, "runs", "2026-06-27T181700000Z");
    await mkdir(oldRunDir, { recursive: true });
    await mkdir(newRunDir, { recursive: true });
    await writeFile(
      path.join(oldScriptRoot, "latest.json"),
      JSON.stringify({ run_dir: "runs/2026-06-27T181650000Z", created_at: "2026-06-27T18:16:50.000Z" }),
      "utf8"
    );
    await writeFile(
      path.join(newScriptRoot, "latest.json"),
      JSON.stringify({ run_dir: "runs/2026-06-27T181700000Z", created_at: "2026-06-27T18:17:00.000Z" }),
      "utf8"
    );
    await writeFile(
      path.join(oldRunDir, "results.json"),
      JSON.stringify({
        study_name: "old",
        direction: "maximize",
        n_trials: 1,
        best_trial: { number: 0, value: 1, params: { script: "old" } },
        all_trials: [{ number: 0, value: 1, params: { script: "old" } }]
      }),
      "utf8"
    );
    await writeFile(
      path.join(newRunDir, "results.json"),
      JSON.stringify({
        study_name: "new",
        direction: "maximize",
        n_trials: 1,
        best_trial: { number: 0, value: 1, params: { script: "new" } },
        all_trials: [{ number: 0, value: 1, params: { script: "new" } }]
      }),
      "utf8"
    );

    const result = await readResults(autotuneRoot);

    expect(result.study_name).toBe("new");
  });

  it("falls back to legacy .autotune results for the default visible directory", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-results-legacy-"));
    await mkdir(path.join(dir, "autotune"), { recursive: true });
    const legacyRoot = path.join(dir, ".autotune");
    await mkdir(legacyRoot, { recursive: true });
    await writeFile(
      path.join(legacyRoot, "results.json"),
      JSON.stringify({
        study_name: "study",
        direction: "maximize",
        n_trials: 1,
        best_trial: { number: 0, value: 1, params: { x: 0.5 } },
        all_trials: [{ number: 0, value: 1, params: { x: 0.5 } }]
      }),
      "utf8"
    );

    const result = await readResults(path.join(dir, "autotune"));

    expect(result.n_trials).toBe(1);
  });
});
