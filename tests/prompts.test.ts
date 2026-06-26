import {
  renderAnalyzePrompt,
  renderModifiedScriptPrompt,
  renderRefineSearchSpacePrompt,
  renderReviseSearchSpacePrompt
} from "../src/prompts.js";
import type { Invocation, SearchSpace } from "../src/types.js";

const invocation: Invocation = {
  language: "python",
  command: ["python3"],
  script: "/tmp/train.py"
};

const searchSpace: SearchSpace = {
  parameters: [{ name: "lr", cli_flag: "--lr", type: "float", low: 0.0001, high: 0.01, log: true }],
  has_arg_parsing: false,
  needs_wrapper: true,
  has_metric_output: false,
  direction: "maximize",
  reasoning: "validation_accuracy is the metric"
};

describe("prompt metric-boundary contract", () => {
  it("tells analysis to exclude objective measurement knobs", () => {
    const prompt = renderAnalyzePrompt({ invocation });

    expect(prompt).toContain("candidate behavior");
    expect(prompt).toContain("Do not include parameters that change objective measurement");
    expect(prompt).toContain("evaluation input set");
    expect(prompt).toContain("scoring formula");
    expect(prompt).toContain("aggregation");
    expect(prompt).toContain("reporting threshold");
    expect(prompt).toContain("random seed used only for measurement");
  });

  it("tells revision to preserve fixed metric and evaluation semantics", () => {
    const prompt = renderReviseSearchSpacePrompt({
      invocation,
      searchSpace,
      feedback: "also tune the validation sample count"
    });

    expect(prompt).toContain("Preserve fixed objective measurement semantics");
    expect(prompt).toContain("If feedback asks to tune");
    expect(prompt).toContain("omit it from parameters");
  });

  it("tells modified-copy generation to preserve metric computation", () => {
    const prompt = renderModifiedScriptPrompt({
      invocation,
      searchSpace,
      outputPath: "/tmp/train_modified.py"
    });

    expect(prompt).toContain("only for the confirmed parameters");
    expect(prompt).toContain("do not add CLI flags for values used only by objective measurement");
    expect(prompt).toContain("preserve the original objective computation exactly");
  });
});

describe("prompt Optuna config contract", () => {
  it("asks analysis to propose safe Optuna settings", () => {
    const prompt = renderAnalyzePrompt({ invocation });

    expect(prompt).toContain("optuna");
    expect(prompt).toContain("sampler");
    expect(prompt).toContain("pruner");
    expect(prompt).toContain("Do not propose storage");
    expect(prompt).toContain("Do not propose n_jobs");
  });

  it("asks revision to preserve the Optuna config contract", () => {
    const prompt = renderReviseSearchSpacePrompt({
      invocation,
      searchSpace,
      feedback: "use hyperband"
    });

    expect(prompt).toContain("optuna");
    expect(prompt).toContain("sampler");
    expect(prompt).toContain("pruner");
    expect(prompt).toContain("Do not add storage");
    expect(prompt).toContain("Do not add n_jobs");
  });
});

describe("prompt trial-result refinement contract", () => {
  it("asks refinement to use trial evidence without changing metric semantics", () => {
    const prompt = renderRefineSearchSpacePrompt({
      invocation,
      searchSpace,
      round: 1,
      trialSummary: {
        direction: "maximize",
        n_trials: 4,
        best_trial: { number: 3, value: 0.9, params: { lr: 0.002 }, state: "COMPLETE" },
        top_trials: [{ number: 3, value: 0.9, params: { lr: 0.002 }, state: "COMPLETE" }],
        parameter_ranges: [{ name: "lr", low: 0.0001, high: 0.01, best_value: 0.002 }]
      }
    });

    expect(prompt).toContain("round 1");
    expect(prompt).toContain("Trial result summary");
    expect(prompt).toContain("narrow");
    expect(prompt).toContain("broaden");
    expect(prompt).toContain("best values sit near bounds");
    expect(prompt).toContain("Preserve fixed objective measurement semantics");
    expect(prompt).toContain("Do not add storage");
    expect(prompt).toContain("Do not add n_jobs");
  });
});
