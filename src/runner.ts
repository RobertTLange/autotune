import { spawn } from "node:child_process";

export async function runPythonRunner(input: {
  runnerPath: string;
  trials: number;
  direction: "maximize" | "minimize";
  sampler: string;
  pruner: string;
  nJobs: number;
  storage?: string;
  studyName?: string;
  output?: string;
}): Promise<string> {
  const args = [
    input.runnerPath,
    "--trials",
    String(input.trials),
    "--direction",
    input.direction,
    "--sampler",
    input.sampler,
    "--pruner",
    input.pruner,
    "--n-jobs",
    String(input.nJobs)
  ];

  if (input.storage) {
    args.push("--storage", input.storage);
  }
  if (input.studyName) {
    args.push("--study-name", input.studyName);
  }
  if (input.output) {
    args.push("--output", input.output);
  }

  return new Promise((resolve, reject) => {
    const child = spawn("python3", args, { stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";

    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
      process.stderr.write(chunk);
    });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) {
        resolve(stdout);
      } else {
        reject(new Error(`python runner exited with ${code}: ${(stderr || stdout).trim()}`));
      }
    });
  });
}
