import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { confirmSearchSpace, printSearchSpace } from "../src/confirm.js";

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
    expect(lines.join("\n")).not.toContain("\t");
    expect(lines.join("\n")).toContain("Parameter  Type         Range/Choices     CLI Flag     Current");
    expect(lines.join("\n")).toContain("---------  -----------  ----------------  -----------  -------");
    expect(lines.join("\n")).toContain("lr         float        [0.001, 0.1] log  --lr");
    expect(lines.join("\n")).toContain("optimizer  categorical  [adam, sgd]       --optimizer");
  });
});

describe("confirmSearchSpace", () => {
  it("revises the proposal from feedback, then accepts the revised space", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-confirm-"));
    const filePath = path.join(dir, "search_space.yaml");
    const answers = ["feedback", "make x wider", "Y"];

    const confirmed = await confirmSearchSpace({
      searchSpace: {
        parameters: [{ name: "x", cli_flag: "--x", type: "float", low: 0, high: 1 }],
        has_arg_parsing: true,
        needs_wrapper: false,
        direction: "maximize"
      },
      filePath,
      yes: false,
      ask: async () => answers.shift() ?? "Y",
      revise: async (searchSpace, feedback) => ({
        ...searchSpace,
        parameters: [{ name: "x", cli_flag: "--x", type: "float", low: -1, high: 2 }],
        reasoning: feedback
      })
    });

    expect(confirmed.parameters[0]).toMatchObject({ low: -1, high: 2 });
    expect(await readFile(filePath, "utf8")).toContain("high: 2");
  });
});
