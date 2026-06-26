import { readFile } from "node:fs/promises";
import path from "node:path";

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
    `autotune · ${result.n_trials} trials · ${result.direction}`,
    "",
    result.best_trial
      ? `Best trial: #${result.best_trial.number} | Value: ${result.best_trial.value}`
      : "No completed trials"
  ];

  if (result.best_trial) {
    lines.push("", "Parameters:");
    for (const [key, value] of Object.entries(result.best_trial.params)) {
      lines.push(`  ${key}\t${String(value)}`);
    }
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
    for (const trial of completed) {
      const params = Object.entries(trial.params)
        .map(([key, value]) => `${key}=${String(value)}`)
        .join(" ");
      lines.push(`  #${trial.number}\t${trial.value}\t${params}`);
    }
  }

  return lines.join("\n");
}
