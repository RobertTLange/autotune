import { spawn } from "node:child_process";
import readline from "node:readline/promises";
import { stdin, stdout } from "node:process";
import { readSearchSpace, writeSearchSpace } from "./search-space.js";
import type { SearchSpace } from "./types.js";

type Ask = (prompt: string) => Promise<string>;

export async function confirmSearchSpace(input: {
  searchSpace: SearchSpace;
  filePath: string;
  yes: boolean;
  ask?: Ask;
  revise?: (searchSpace: SearchSpace, feedback: string) => Promise<SearchSpace>;
}): Promise<SearchSpace> {
  const request = input;
  let current = request.searchSpace;
  await writeSearchSpace(request.filePath, current);
  if (request.yes) {
    return current;
  }

  const rl = request.ask ? undefined : readline.createInterface({ input: stdin, output: stdout });
  const ask = request.ask ?? ((prompt: string) => rl?.question(prompt) ?? Promise.resolve(""));
  try {
    while (true) {
      printSearchSpace(current);
      const rawAnswer = (await ask("Run search with this space? [Y/feedback/edit/n] ")).trim();
      const answer = rawAnswer.toLowerCase();
      if (answer === "" || answer === "y" || answer === "yes") {
        await writeSearchSpace(request.filePath, current);
        return current;
      }
      if (answer === "edit") {
        await openEditor(request.filePath);
        current = await readSearchSpace(request.filePath);
        continue;
      }
      if (answer === "n" || answer === "no") {
        throw new Error("aborted by user");
      }

      const feedback = answer === "feedback" || answer === "f" ? (await ask("Feedback: ")).trim() : rawAnswer;
      if (!feedback) {
        continue;
      }
      if (!request.revise) {
        throw new Error("feedback revision is unavailable");
      }
      console.error("Revising search space from feedback...");
      current = await request.revise(current, feedback);
      await writeSearchSpace(request.filePath, current);
    }
  } finally {
    rl?.close();
  }
}

export function printSearchSpace(searchSpace: SearchSpace): void {
  console.log("\nAnalysis complete. Proposed search space:\n");
  const rows = searchSpace.parameters.map((parameter) => ({
    parameter: parameter.name,
    type: parameter.type,
    range:
      parameter.type === "categorical"
        ? `[${parameter.choices?.join(", ")}]`
        : `[${parameter.low}, ${parameter.high}]${parameter.log ? " log" : ""}`,
    flag: parameter.cli_flag,
    current: String(parameter.current_value ?? "")
  }));
  const widths = {
    parameter: columnWidth("Parameter", rows.map((row) => row.parameter)),
    type: columnWidth("Type", rows.map((row) => row.type)),
    range: columnWidth("Range/Choices", rows.map((row) => row.range)),
    flag: columnWidth("CLI Flag", rows.map((row) => row.flag))
  };

  console.log(formatSearchSpaceRow("Parameter", "Type", "Range/Choices", "CLI Flag", "Current", widths));
  for (const row of rows) {
    console.log(formatSearchSpaceRow(row.parameter, row.type, row.range, row.flag, row.current, widths));
  }
  console.log(`\nDirection: ${searchSpace.direction}${searchSpace.reasoning ? ` (${searchSpace.reasoning})` : ""}`);
  console.log(`Arg parsing: ${searchSpace.has_arg_parsing ? "yes" : "no"}`);
}

function columnWidth(header: string, values: string[]): number {
  return Math.max(header.length, ...values.map((value) => value.length));
}

function formatSearchSpaceRow(
  parameter: string,
  type: string,
  range: string,
  flag: string,
  current: string,
  widths: { parameter: number; type: number; range: number; flag: number }
): string {
  return [
    parameter.padEnd(widths.parameter),
    type.padEnd(widths.type),
    range.padEnd(widths.range),
    flag.padEnd(widths.flag),
    current
  ].join("  ");
}

async function openEditor(filePath: string): Promise<void> {
  const editor = process.env.EDITOR ?? "vi";
  await new Promise<void>((resolve, reject) => {
    const child = spawn(editor, [filePath], { stdio: "inherit" });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`${editor} exited with ${code}`));
      }
    });
  });
}
