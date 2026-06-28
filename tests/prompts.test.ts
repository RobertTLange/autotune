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

    expect(prompt).toContain("active and fixed parameters");
    expect(prompt).toContain("do not add CLI flags for values used only by objective measurement");
    expect(prompt).toContain("preserve the original objective computation exactly");
  });
});

describe("prompt Optuna config contract", () => {
  it("asks analysis to propose safe Optuna settings", () => {
    const prompt = renderAnalyzePrompt({ invocation });

    expect(prompt).toContain("optuna");
    expect(prompt).toContain("fixed_parameters");
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
    expect(prompt).toContain("fixed_parameters");
    expect(prompt).toContain("sampler");
    expect(prompt).toContain("pruner");
    expect(prompt).toContain("Do not add storage");
    expect(prompt).toContain("Do not add n_jobs");
  });
});

describe("prompt budget contract", () => {
  it("includes trial budget and timeout during analysis", () => {
    const prompt = renderAnalyzePrompt({
      invocation,
      budget: { trials: 5, timeoutSeconds: 1800 }
    });

    expect(prompt).toContain("initial_trials: 5");
    expect(prompt).toContain("total_planned_trials: 5");
    expect(prompt).toContain("per_trial_timeout_seconds: 1800");
    expect(prompt).toContain("1-3 high-impact");
  });

  it("includes budget when revising from feedback", () => {
    const prompt = renderReviseSearchSpacePrompt({
      invocation,
      searchSpace,
      feedback: "keep it tiny",
      budget: { trials: 5, timeoutSeconds: 900 }
    });

    expect(prompt).toContain("initial_trials: 5");
    expect(prompt).toContain("Scale the search-space breadth");
  });

  it("includes refinement metadata during refinement", () => {
    const prompt = renderRefineSearchSpacePrompt({
      invocation,
      searchSpace,
      round: 1,
      budget: {
        trials: 20,
        timeoutSeconds: 1800,
        refineRounds: 2,
        refineTrials: 10,
        refineMode: "auto",
        currentRefinementRound: 1,
        currentRoundTrials: 10
      },
      trialSummary: {
        direction: "maximize",
        n_trials: 20,
        best_trial: { number: 3, value: 0.9, params: { lr: 0.002 }, state: "COMPLETE" },
        top_trials: [{ number: 3, value: 0.9, params: { lr: 0.002 }, state: "COMPLETE" }],
        parameter_ranges: [{ name: "lr", low: 0.0001, high: 0.01, best_value: 0.002 }]
      }
    });

    expect(prompt).toContain("total_planned_trials: 40");
    expect(prompt).toContain("refinement_rounds: 2");
    expect(prompt).toContain("refinement_trials_per_round: 10");
    expect(prompt).toContain("refinement_mode: auto");
    expect(prompt).toContain("current_refinement_round: 1");
    expect(prompt).toContain("current_round_trials: 10");
  });
});

describe("prompt invocation contract", () => {
  it("does not append the source path for standalone runtime commands", () => {
    const prompt = renderAnalyzePrompt({
      invocation: {
        language: "cpp",
        command: ["/work/.autotune/model"],
        script: "/work/model.cpp",
        scriptArgument: "none"
      }
    });

    expect(prompt).toContain("The script is invoked via: /work/.autotune/model");
    expect(prompt).not.toContain("/work/.autotune/model /work/model.cpp");
  });

  it("does not duplicate explicit script slots", () => {
    const prompt = renderAnalyzePrompt({
      invocation: {
        language: "python",
        command: ["python3", "-u", "/work/train.py"],
        script: "/work/train.py",
        scriptArgument: "included"
      }
    });

    expect(prompt).toContain("The script is invoked via: python3 -u /work/train.py");
    expect(prompt).not.toContain("/work/train.py /work/train.py");
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
