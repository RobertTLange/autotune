import { readFile } from "node:fs/promises";
import path from "node:path";
import { formatNumber, formatTable, styles } from "./terminal.js";

export interface TrialResult {
  number: number;
  value: number | null;
  params: Record<string, unknown>;
  state?: string;
}

export interface StudyResult {
  study_name: string;
  direction: "maximize" | "minimize";
  n_trials: number;
  best_trial: TrialResult | null;
  all_trials: TrialResult[];
}

export async function readResults(location = ".autotune"): Promise<StudyResult> {
  const file = location.endsWith(".json") ? location : path.join(location, "results.json");
  return JSON.parse(await readFile(file, "utf8")) as StudyResult;
}

export function renderResults(result: StudyResult, top = 5): string {
  const lines = [
    styles.cyan(styles.bold(`autotune · ${result.n_trials} trials · ${result.direction}`)),
    "",
    result.best_trial
      ? `Best trial: #${result.best_trial.number} | Value: ${styles.green(styles.bold(formatNumber(Number(result.best_trial.value))))}`
      : "No completed trials"
  ];

  if (result.best_trial) {
    lines.push("", "Parameters:");
    lines.push(
      ...formatTable({
        headers: ["Parameter", "Value"],
        rows: Object.entries(result.best_trial.params).map(([key, value]) => [key, formatResultValue(value)])
      }).map((line) => `  ${line}`)
    );
  }

  const completed = result.all_trials
    .filter((trial) => typeof trial.value === "number")
    .sort((left, right) =>
      result.direction === "maximize"
        ? Number(right.value) - Number(left.value)
        : Number(left.value) - Number(right.value)
    )
    .slice(0, top);

  if (completed.length > 0) {
    lines.push("", `Top ${completed.length} trials:`);
    lines.push(
      ...formatTable({
        headers: ["Trial", "Value", "Parameters"],
        rows: completed.map((trial) => [
          `#${trial.number}`,
          formatNumber(Number(trial.value)),
          Object.entries(trial.params)
            .map(([key, value]) => `${key}=${formatResultValue(value)}`)
            .join(" ")
        ])
      }).map((line) => `  ${line}`)
    );
  }

  return lines.join("\n");
}

function formatResultValue(value: unknown): string {
  return typeof value === "number" ? formatNumber(value) : String(value);
}
