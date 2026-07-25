import { chmod, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { confirmSearchSpace, printSearchSpace } from "../src/confirm.js";
import { validateSearchSpaceParameterLimit } from "../src/search-space.js";

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
        fixed_parameters: [{ name: "batch_size", cli_flag: "--batch-size", value: 128 }],
        has_arg_parsing: true,
        needs_wrapper: false,
        optuna: { sampler: "random", pruner: "median", reasoning: "short iterative trials" },
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
    expect(lines.join("\n")).toContain("Fixed parameters");
    expect(lines.join("\n")).toContain("batch_size  128    --batch-size");
    expect(lines.join("\n")).toContain("Sampler: random");
    expect(lines.join("\n")).toContain("Pruner: median");
    expect(lines.join("\n")).toContain("short iterative trials");
  });
});

describe("confirmSearchSpace", () => {
  it("validates a search space before accepting it", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-confirm-validation-"));
    const validate = vi.fn(() => {
      throw new Error("too many active parameters");
    });

    await expect(confirmSearchSpace({
      searchSpace: {
        parameters: [{ name: "x", cli_flag: "--x", type: "float", low: 0, high: 1 }],
        has_arg_parsing: true,
        needs_wrapper: false,
        direction: "maximize"
      },
      filePath: path.join(dir, "search_space.yaml"),
      yes: true,
      validate
    })).rejects.toThrow(/too many active parameters/i);
    expect(validate).toHaveBeenCalledOnce();
  });

  it("rejects an edited search space that exceeds the active-parameter limit", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-confirm-edit-limit-"));
    const filePath = path.join(dir, "search_space.yaml");
    const editor = path.join(dir, "editor");
    const originalEditor = process.env.EDITOR;
    await writeFile(
      editor,
      `#!/usr/bin/env node
require("node:fs").writeFileSync(process.argv[2], ${JSON.stringify(`
parameters:
  - { name: x, cli_flag: --x, type: float, low: 0, high: 1 }
  - { name: y, cli_flag: --y, type: float, low: 0, high: 1 }
has_arg_parsing: true
needs_wrapper: false
direction: maximize
`)});
`,
      "utf8"
    );
    await chmod(editor, 0o755);
    process.env.EDITOR = editor;

    try {
      await expect(confirmSearchSpace({
        searchSpace: {
          parameters: [{ name: "x", cli_flag: "--x", type: "float", low: 0, high: 1 }],
          has_arg_parsing: true,
          needs_wrapper: false,
          direction: "maximize"
        },
        filePath,
        yes: false,
        ask: async () => "edit",
        validate: (candidate) => validateSearchSpaceParameterLimit(candidate, 1)
      })).rejects.toThrow(/2 active parameters.*--max-parameters 1/i);
    } finally {
      process.env.EDITOR = originalEditor;
    }
  });

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
