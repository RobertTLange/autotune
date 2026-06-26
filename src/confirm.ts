import { spawn } from "node:child_process";
import readline from "node:readline/promises";
import { stdin, stdout } from "node:process";
import { readSearchSpace, writeSearchSpace } from "./search-space.js";
import { formatTable, styles, wrapText, writeStatus } from "./terminal.js";
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
      writeStatus("Revising search space from feedback...");
      current = await request.revise(current, feedback);
      await writeSearchSpace(request.filePath, current);
    }
  } finally {
    rl?.close();
  }
}

export function printSearchSpace(searchSpace: SearchSpace): void {
  console.log(`\n${styles.cyan(styles.bold("Analysis complete. Proposed search space"))}\n`);
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

  for (const line of formatTable({
    headers: ["Parameter", "Type", "Range/Choices", "CLI Flag", "Current"],
    rows: rows.map((row) => [row.parameter, row.type, row.range, row.flag, row.current])
  })) {
    console.log(line);
  }
  console.log("");
  console.log(`${styles.bold("Direction:")} ${searchSpace.direction}`);
  if (searchSpace.reasoning) {
    for (const line of wrapText(searchSpace.reasoning, 100)) {
      console.log(`  ${styles.dim(line)}`);
    }
  }
  console.log(`${styles.bold("Arg parsing:")} ${searchSpace.has_arg_parsing ? styles.green("yes") : styles.yellow("no")}`);
  console.log(`${styles.bold("Metric output:")} ${searchSpace.has_metric_output === false ? styles.yellow("no") : styles.green("yes")}`);
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
