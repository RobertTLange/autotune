import { spawn } from "node:child_process";
import readline from "node:readline/promises";
import { stdin, stdout } from "node:process";
import { readSearchSpace, writeSearchSpace } from "./search-space.js";
import type { SearchSpace } from "./types.js";

export async function confirmSearchSpace(input: {
  searchSpace: SearchSpace;
  filePath: string;
  yes: boolean;
}): Promise<SearchSpace> {
  const request = input;
  await writeSearchSpace(request.filePath, request.searchSpace);
  if (request.yes) {
    return request.searchSpace;
  }

  printSearchSpace(request.searchSpace);
  const rl = readline.createInterface({ input: stdin, output: stdout });
  try {
    const answer = (await rl.question("Confirm search space? [Y/edit/n] ")).trim().toLowerCase();
    if (answer === "" || answer === "y" || answer === "yes") {
      return request.searchSpace;
    }
    if (answer === "edit") {
      await openEditor(request.filePath);
      return readSearchSpace(request.filePath);
    }
    throw new Error("aborted by user");
  } finally {
    rl.close();
  }
}

export function printSearchSpace(searchSpace: SearchSpace): void {
  console.log("\nAnalysis complete. Proposed search space:\n");
  console.log("Parameter\tType\tRange/Choices\tCLI Flag\tCurrent");
  for (const parameter of searchSpace.parameters) {
    const range =
      parameter.type === "categorical"
        ? `[${parameter.choices?.join(", ")}]`
        : `[${parameter.low}, ${parameter.high}]${parameter.log ? " log" : ""}`;
    console.log(`${parameter.name}\t${parameter.type}\t${range}\t${parameter.cli_flag}\t${String(parameter.current_value ?? "")}`);
  }
  console.log(`\nDirection: ${searchSpace.direction}${searchSpace.reasoning ? ` (${searchSpace.reasoning})` : ""}`);
  console.log(`Arg parsing: ${searchSpace.has_arg_parsing ? "yes" : "no"}`);
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
