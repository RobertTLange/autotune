import { renderAnalyzePrompt, renderModifiedScriptPrompt, renderReviseSearchSpacePrompt } from "../src/prompts.js";
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
