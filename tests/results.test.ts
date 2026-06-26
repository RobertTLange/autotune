import { mkdtemp, writeFile } from "node:fs/promises";
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
    expect(renderResults(result, 1)).toContain("Best trial: #1");
    expect(renderResults(result, 1)).toContain("#1\t0.1\tlr=0.01");
  });
});
