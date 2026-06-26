import { printSearchSpace } from "../src/confirm.js";

describe("printSearchSpace", () => {
  it("prints numeric and categorical parameters", () => {
    const original = console.log;
    const lines: string[] = [];
    console.log = (value?: unknown) => {
      lines.push(String(value ?? ""));
    };
    try {
      printSearchSpace({
        parameters: [
          { name: "lr", cli_flag: "--lr", type: "float", low: 0.001, high: 0.1, log: true },
          { name: "optimizer", cli_flag: "--optimizer", type: "categorical", choices: ["adam", "sgd"] }
        ],
        has_arg_parsing: true,
        needs_wrapper: false,
        direction: "maximize",
        reasoning: "accuracy"
      });
    } finally {
      console.log = original;
    }
    expect(lines.join("\n")).toContain("[0.001, 0.1] log");
    expect(lines.join("\n")).toContain("[adam, sgd]");
  });
});
