import { mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { plotProgress, readVariantProgress } from "../src/progress-plot.js";

describe("progress plot", () => {
  it("includes result variants with unrecognized directory names", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "autotune-progress-variants-"));
    await writeResult(path.join(root, "04_fixed_params_only", "results.json"), "maximize", [trial(0, 1)]);
    await writeResult(path.join(root, "custom_experiment", "results.json"), "maximize", [trial(0, 2)]);

    const variants = await readVariantProgress(root);

    expect(variants.map((variant) => variant.label)).toEqual(["04_fixed_params_only", "custom_experiment"]);
  });

  it("builds maximize progress across resets and skips transferred seed trials", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "autotune-progress-max-"));
    await writeResult(path.join(root, "01_base_optuna", "results.json"), "maximize", [
      trial(0, 1),
      trial(1, 3)
    ]);
    await writeResult(path.join(root, "02_resets_no_trial_transfer", "results.round_0.json"), "maximize", [
      trial(0, 1),
      trial(1, 2)
    ]);
    await writeResult(path.join(root, "02_resets_no_trial_transfer", "results.round_1.json"), "maximize", [
      trial(0, 100, { autotune_failure_reason: "timeout" }),
      trial(1, 4)
    ]);
    await writeResult(path.join(root, "03_resets_trial_transfer", "results.round_0.json"), "maximize", [
      trial(0, 2)
    ]);
    await writeResult(path.join(root, "03_resets_trial_transfer", "results.round_1.json"), "maximize", [
      trial(0, 5, { autotune_transfer: true }),
      trial(1, 6)
    ]);

    const variants = await readVariantProgress(root);

    expect(variants.map((variant) => variant.label)).toEqual([
      "Base Optuna",
      "Resets, no transfer",
      "Resets + transfer"
    ]);
    expect(variants[1].resets).toEqual([2]);
    expect(variants[1].points).toEqual([
      { x: 1, y: 1, improved: true },
      { x: 2, y: 2, improved: true },
      { x: 3, y: 2, improved: false },
      { x: 4, y: 4, improved: true }
    ]);
    expect(variants[2].totalTrials).toBe(2);
    expect(variants[2].points).toEqual([
      { x: 1, y: 2, improved: true },
      { x: 2, y: 6, improved: true }
    ]);
  });

  it("builds minimize progress without letting timeout penalties set the y-domain", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "autotune-progress-min-"));
    await writeResult(path.join(root, "03_autotune_resets_trial_transfer", "results.round_0.json"), "minimize", [
      trial(0, 100, { autotune_failure_reason: "timeout" }),
      trial(1, 0.9)
    ]);
    await writeResult(path.join(root, "03_autotune_resets_trial_transfer", "results.round_1.json"), "minimize", [
      trial(0, 0.9, { autotune_transfer: true }),
      trial(1, 100, { autotune_failure_reason: "timeout" }),
      trial(2, 0.8)
    ]);

    const [variant] = await readVariantProgress(root);

    expect(variant.direction).toBe("minimize");
    expect(variant.resets).toEqual([2]);
    expect(variant.totalTrials).toBe(4);
    expect(variant.points).toEqual([
      { x: 2, y: 0.9, improved: true },
      { x: 3, y: 0.9, improved: false },
      { x: 4, y: 0.8, improved: true }
    ]);
  });

  it("writes an SVG progress plot with reset labels", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "autotune-progress-svg-"));
    const output = path.join(root, "plot.svg");
    await writeResult(path.join(root, "01_base_optuna", "results.json"), "maximize", [
      trial(0, 1),
      trial(1, 2)
    ]);
    await writeResult(path.join(root, "02_resets_no_trial_transfer", "results.round_0.json"), "maximize", [
      trial(0, 1)
    ]);
    await writeResult(path.join(root, "02_resets_no_trial_transfer", "results.round_1.json"), "maximize", [
      trial(0, 3)
    ]);

    await plotProgress(root, { output, title: "Synthetic progress", maxTrials: 4, yMin: 0, yMax: 4 });

    const svg = await readFile(output, "utf8");
    expect(svg).toContain("<svg");
    expect(svg).toContain("Synthetic progress");
    expect(svg).toContain("Base Optuna");
    expect(svg).toContain("Resets, no transfer");
    expect(svg).toContain("reset @ 1");
    expect(svg).toContain("<polygon");
    expect(svg).toContain("clip-path");
    expect(svg).toContain("legend-bg");
  });
});

function trial(number: number, value: number, userAttrs: Record<string, unknown> = {}) {
  return {
    number,
    value,
    params: { x: number },
    state: "COMPLETE",
    user_attrs: userAttrs
  };
}

async function writeResult(file: string, direction: "maximize" | "minimize", trials: ReturnType<typeof trial>[]) {
  await mkdir(path.dirname(file), { recursive: true });
  await writeFile(
    file,
    JSON.stringify({
      study_name: path.basename(path.dirname(file)),
      direction,
      n_trials: trials.length,
      best_trial: trials[0],
      all_trials: trials
    }),
    "utf8"
  );
}
