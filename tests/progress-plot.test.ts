import { mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { plotProgress, readVariantProgress } from "../src/progress-plot.js";

describe("progress plot", () => {
  it("labels the fourth ablation arm as Centaur", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "autotune-progress-centaur-"));
    await writeResult(path.join(root, "04_centaur", "results.json"), "maximize", [trial(0, 1)]);

    const [variant] = await readVariantProgress(root);

    expect(variant.label).toBe("Centaur");
  });

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
    expect(variants[1].resets).toEqual([{ round: 1, x: 2 }]);
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
    expect(variant.resets).toEqual([{ round: 1, x: 2 }]);
    expect(variant.totalTrials).toBe(4);
    expect(variant.points).toEqual([
      { x: 2, y: 0.9, improved: true },
      { x: 3, y: 0.9, improved: false },
      { x: 4, y: 0.8, improved: true }
    ]);
  });

  it("uses cumulative trial runtime while excluding transferred seeds", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "autotune-progress-runtime-"));
    await writeResult(path.join(root, "03_resets_trial_transfer", "results.round_0.json"), "maximize", [
      trial(0, 1, { autotune_duration_seconds: 3600 }),
      trial(1, 2, { autotune_duration_seconds: 1800 })
    ]);
    await writeResult(path.join(root, "03_resets_trial_transfer", "results.round_1.json"), "maximize", [
      trial(0, 2, { autotune_duration_seconds: 10000, autotune_transfer: true }),
      trial(1, 100, { autotune_duration_seconds: 900, autotune_failure_reason: "timeout" }),
      trial(2, 3, { autotune_duration_seconds: 2700 })
    ]);

    const [variant] = await readVariantProgress(root, { xAxis: "runtime" });

    expect(variant.resets).toEqual([{ round: 1, x: 1.5 }]);
    expect(variant.totalTrials).toBe(4);
    expect(variant.totalRuntimeHours).toBe(2.5);
    expect(variant.points).toEqual([
      { x: 1, y: 1, improved: true },
      { x: 1.5, y: 2, improved: true },
      { x: 1.75, y: 2, improved: false },
      { x: 2.5, y: 3, improved: true }
    ]);
  });

  it("rejects runtime plots when a real trial has no duration", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "autotune-progress-runtime-missing-"));
    await writeResult(path.join(root, "01_base_optuna", "results.json"), "maximize", [trial(0, 1)]);

    await expect(readVariantProgress(root, { xAxis: "runtime" })).rejects.toThrow(
      "runtime x-axis requires autotune_duration_seconds"
    );
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

  it("writes runtime units in the SVG axis, resets, and legend", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "autotune-progress-runtime-svg-"));
    const output = path.join(root, "plot.svg");
    await writeResult(path.join(root, "02_resets_no_trial_transfer", "results.round_0.json"), "maximize", [
      trial(0, 1, { autotune_duration_seconds: 3600 }),
      trial(1, 2, { autotune_duration_seconds: 1800 })
    ]);
    await writeResult(path.join(root, "02_resets_no_trial_transfer", "results.round_1.json"), "maximize", [
      trial(0, 3, { autotune_duration_seconds: 3600 })
    ]);

    await plotProgress(root, {
      output,
      maxTrials: 3,
      xAxis: "runtime",
      maxRuntimeHours: 3,
      yMin: 0,
      yMax: 4
    });

    const svg = await readFile(output, "utf8");
    expect(svg).toContain("Cumulative trial runtime (hours)");
    expect(svg).toContain("reset @ 1.5h");
    expect(svg).toContain("2.5h · 3 trials");
    expect(svg).not.toMatch(/<text[^>]+fill="#2563eb">3\.000<\/text>/);
  });

  it("merges nearby runtime reset markers from different arms", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "autotune-progress-runtime-resets-"));
    const output = path.join(root, "plot.svg");
    await writeResult(path.join(root, "02_resets_no_trial_transfer", "results.round_0.json"), "maximize", [
      trial(0, 1, { autotune_duration_seconds: 5400 })
    ]);
    await writeResult(path.join(root, "02_resets_no_trial_transfer", "results.round_1.json"), "maximize", [
      trial(0, 2, { autotune_duration_seconds: 3600 })
    ]);
    await writeResult(path.join(root, "03_resets_trial_transfer", "results.round_0.json"), "maximize", [
      trial(0, 1, { autotune_duration_seconds: 5580 })
    ]);
    await writeResult(path.join(root, "03_resets_trial_transfer", "results.round_1.json"), "maximize", [
      trial(0, 2, { autotune_duration_seconds: 3420 })
    ]);

    await plotProgress(root, {
      output,
      maxTrials: 2,
      xAxis: "runtime",
      maxRuntimeHours: 3,
      yMin: 0,
      yMax: 3
    });

    const svg = await readFile(output, "utf8");
    expect(svg.match(/class="reset"/g)).toHaveLength(1);
    expect(svg).toContain("resets ≈ 1.5h");
  });

  it("keeps resets for different round boundaries distinct", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "autotune-progress-runtime-round-resets-"));
    const output = path.join(root, "plot.svg");
    await writeResult(path.join(root, "02_resets_no_trial_transfer", "results.round_0.json"), "maximize", [
      trial(0, 1, { autotune_duration_seconds: 3600 })
    ]);
    await writeResult(path.join(root, "02_resets_no_trial_transfer", "results.round_1.json"), "maximize", [
      trial(0, 2, { autotune_duration_seconds: 3600 })
    ]);
    await writeResult(path.join(root, "03_resets_trial_transfer", "results.round_1.json"), "maximize", [
      trial(0, 1, { autotune_duration_seconds: 5400 })
    ]);
    await writeResult(path.join(root, "03_resets_trial_transfer", "results.round_2.json"), "maximize", [
      trial(0, 2, { autotune_duration_seconds: 1800 })
    ]);

    await plotProgress(root, {
      output,
      maxTrials: 2,
      xAxis: "runtime",
      maxRuntimeHours: 3,
      yMin: 0,
      yMax: 3
    });

    const svg = await readFile(output, "utf8");
    expect(svg.match(/class="reset"/g)).toHaveLength(2);
    expect(svg).toContain("reset @ 1h");
    expect(svg).toContain("reset @ 1.5h");
  });

  it("keeps nearby reset generations from the same arm distinct", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "autotune-progress-runtime-distinct-resets-"));
    const output = path.join(root, "plot.svg");
    await writeResult(path.join(root, "02_resets_no_trial_transfer", "results.round_0.json"), "maximize", [
      trial(0, 1, { autotune_duration_seconds: 3600 })
    ]);
    await writeResult(path.join(root, "02_resets_no_trial_transfer", "results.round_1.json"), "maximize", [
      trial(0, 2, { autotune_duration_seconds: 180 })
    ]);
    await writeResult(path.join(root, "02_resets_no_trial_transfer", "results.round_2.json"), "maximize", [
      trial(0, 3, { autotune_duration_seconds: 3420 })
    ]);

    await plotProgress(root, {
      output,
      maxTrials: 3,
      xAxis: "runtime",
      maxRuntimeHours: 3,
      yMin: 0,
      yMax: 4
    });

    const svg = await readFile(output, "utf8");
    expect(svg.match(/class="reset"/g)).toHaveLength(2);
    expect(svg).toContain("reset @ 1h");
    expect(svg).toContain("reset @ 1.1h");
  });

  it("preserves useful precision for sub-hour runtime plots", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "autotune-progress-runtime-short-"));
    const output = path.join(root, "plot.svg");
    await writeResult(path.join(root, "01_base_optuna", "results.json"), "maximize", [
      trial(0, 1, { autotune_duration_seconds: 50 }),
      trial(1, 2, { autotune_duration_seconds: 50 })
    ]);

    await plotProgress(root, {
      output,
      maxTrials: 2,
      xAxis: "runtime",
      maxRuntimeHours: 0.03,
      yMin: 0,
      yMax: 3
    });

    const svg = await readFile(output, "utf8");
    expect(svg).toContain(">0.0075</text>");
    expect(svg).toContain("0.028h · 2 trials");
  });

  it("anchors late runtime reset labels inside the chart", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "autotune-progress-runtime-late-reset-"));
    const output = path.join(root, "plot.svg");
    await writeResult(path.join(root, "02_resets_no_trial_transfer", "results.round_0.json"), "maximize", [
      trial(0, 1, { autotune_duration_seconds: 10080 })
    ]);
    await writeResult(path.join(root, "02_resets_no_trial_transfer", "results.round_1.json"), "maximize", [
      trial(0, 2, { autotune_duration_seconds: 360 })
    ]);

    await plotProgress(root, {
      output,
      maxTrials: 2,
      xAxis: "runtime",
      maxRuntimeHours: 3,
      yMin: 0,
      yMax: 3
    });

    const svg = await readFile(output, "utf8");
    expect(svg).toMatch(/text-anchor="end"[^>]*>reset @ 2\.8h<\/text>/);
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
