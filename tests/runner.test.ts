import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { runPythonRunner } from "../src/runner.js";

describe("runPythonRunner", () => {
  it("passes runner args without shell interpolation", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-runner-"));
    const runner = path.join(dir, "runner.py");
    await writeFile(
      runner,
      [
        "import json",
        "import sys",
        "with open(sys.argv[sys.argv.index('--output') + 1], 'w', encoding='utf-8') as fh:",
        "    json.dump(sys.argv[1:], fh)",
        "print('ok')"
      ].join("\n"),
      "utf8"
    );
    const output = path.join(dir, "args.json");
    await runPythonRunner({
      python: "python3",
      runnerPath: runner,
      trials: 3,
      direction: "maximize",
      sampler: "tpe",
      pruner: "none",
      nJobs: 2,
      studyName: "train_autotune",
      output
    });
    const args = JSON.parse(await readFile(output, "utf8")) as string[];
    expect(args).toEqual(expect.arrayContaining(["--study-name", "train_autotune"]));
  });

  it("forwards runner stderr progress", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-runner-progress-"));
    const runner = path.join(dir, "runner.py");
    await writeFile(
      runner,
      [
        "import json",
        "import sys",
        "print('trial 1/3 complete', file=sys.stderr)",
        "with open(sys.argv[sys.argv.index('--output') + 1], 'w', encoding='utf-8') as fh:",
        "    json.dump({'ok': True}, fh)"
      ].join("\n"),
      "utf8"
    );
    const output = path.join(dir, "results.json");
    let forwarded = "";
    const original = process.stderr.write;
    process.stderr.write = ((chunk: string | Uint8Array) => {
      forwarded += chunk.toString();
      return true;
    }) as typeof process.stderr.write;
    try {
      await runPythonRunner({
        python: "python3",
        runnerPath: runner,
        trials: 3,
        direction: "maximize",
        sampler: "tpe",
        pruner: "none",
        nJobs: 1,
        output
      });
    } finally {
      process.stderr.write = original;
    }
    expect(forwarded).toContain("trial 1/3 complete");
  });

  it("isolates the controller while preserving Python settings for trial commands", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-runner-env-"));
    const runner = path.join(dir, "runner.py");
    const output = path.join(dir, "environment.json");
    await writeFile(
      runner,
      [
        "import json",
        "import os",
        "import sys",
        "output = sys.argv[sys.argv.index('--output') + 1]",
        "with open(output, 'w', encoding='utf-8') as handle:",
        "    json.dump({",
        "        'controllerPythonPath': os.environ.get('PYTHONPATH'),",
        "        'targetPythonEnvironment': json.loads(os.environ.get('AUTOTUNE_TARGET_PYTHON_ENV', '{}'))",
        "    }, handle)"
      ].join("\n"),
      "utf8"
    );

    await runPythonRunner({
      python: "python3",
      runnerPath: runner,
      trials: 1,
      direction: "maximize",
      sampler: "tpe",
      pruner: "none",
      nJobs: 1,
      output,
      env: { ...process.env, PYTHONPATH: "/training/environment" }
    });

    const captured = JSON.parse(await readFile(output, "utf8")) as {
      controllerPythonPath?: string;
      targetPythonEnvironment: Record<string, string>;
    };
    expect(captured.controllerPythonPath).toBeNull();
    expect(captured.targetPythonEnvironment.PYTHONPATH).toBe("/training/environment");
  });

  it("preserves the caller environment for legacy generated runners", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-runner-legacy-env-"));
    const runner = path.join(dir, "legacy.py");
    const output = path.join(dir, "environment.json");
    await writeFile(
      runner,
      [
        "import json",
        "import os",
        "import sys",
        "output = sys.argv[sys.argv.index('--output') + 1]",
        "with open(output, 'w', encoding='utf-8') as handle:",
        "    json.dump({'pythonHome': os.environ.get('PYTHONHOME'), 'pythonPath': os.environ.get('PYTHONPATH')}, handle)"
      ].join("\n"),
      "utf8"
    );

    await runPythonRunner({
      python: "python3",
      runnerPath: runner,
      trials: 1,
      direction: "maximize",
      sampler: "tpe",
      pruner: "none",
      nJobs: 1,
      output,
      env: {
        ...process.env,
        PYTHONHOME: "/missing-training-python-home",
        PYTHONPATH: "/legacy/training/environment"
      }
    });

    const captured = JSON.parse(await readFile(output, "utf8")) as {
      pythonHome?: string;
      pythonPath?: string;
    };
    expect(captured.pythonHome).toBe("/missing-training-python-home");
    expect(captured.pythonPath).toBe("/legacy/training/environment");
  });

  it.skipIf(process.platform === "win32")("kills detached legacy trial processes on cancellation", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-runner-legacy-cancel-"));
    const runner = path.join(dir, "legacy.py");
    const marker = path.join(dir, "trial.pid");
    await writeFile(
      runner,
      [
        "import subprocess",
        "import sys",
        "import time",
        "from pathlib import Path",
        "trial = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)",
        `Path(${JSON.stringify(marker)}).write_text(str(trial.pid))`,
        "time.sleep(60)"
      ].join("\n"),
      "utf8"
    );
    const previousExitCode = process.exitCode;
    const priorHandlers = new Set(process.listeners("SIGTERM"));
    const running = runPythonRunner({
      python: "python3",
      runnerPath: runner,
      trials: 1,
      direction: "maximize",
      sampler: "tpe",
      pruner: "none",
      nJobs: 1
    });
    const trialPid = Number(await waitForFileText(marker));
    const handler = process.listeners("SIGTERM").find((listener) => !priorHandlers.has(listener));
    expect(handler).toBeDefined();
    try {
      (handler as () => void)();
      await expect(running).rejects.toThrow(/interrupted by SIGTERM/);
      expect(processIsAlive(trialPid)).toBe(false);
    } finally {
      process.exitCode = previousExitCode;
      if (processIsAlive(trialPid)) {
        process.kill(-trialPid, "SIGKILL");
      }
    }
  }, 15_000);
});

async function waitForFileText(filePath: string): Promise<string> {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    try {
      return await readFile(filePath, "utf8");
    } catch {
      await new Promise((resolve) => setTimeout(resolve, 50));
    }
  }
  throw new Error(`timed out waiting for ${filePath}`);
}

function processIsAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}
