import { mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { spawn } from "node:child_process";
import { tmpdir } from "node:os";
import path from "node:path";

const VALIDATOR = path.join("examples", "nanochat", "validate_nanochat.py");

describe("nanochat finalist validation", () => {
  it("fills benchmark defaults for parameters dropped by refinement", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-nanochat-refined-finalist-"));
    const discovery = path.join(dir, "results.round_1.json");
    const { weight_decay: _dropped, ...refinedParams } = nanochatParams({ depth: 12 });
    await writeFile(
      discovery,
      JSON.stringify({ direction: "minimize", all_trials: [trial(0, 0.85, refinedParams)] }),
      "utf8"
    );

    const selected = await selectOneFinalist(discovery);

    expect(selected).toMatchObject({
      discovery_value: 0.85,
      params: { depth: 12, weight_decay: 0.2 }
    });
  });

  it("normalizes direct batch parameters introduced by refinement", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-nanochat-refined-batch-"));
    const discovery = path.join(dir, "results.round_1.json");
    const { batch_config: _dropped, ...refinedParams } = nanochatParams({ depth: 12 });
    await writeFile(
      discovery,
      JSON.stringify({
        direction: "minimize",
        all_trials: [trial(0, 0.8, { ...refinedParams, device_batch_size: 64, total_batch_size: 262144 })]
      }),
      "utf8"
    );

    const selected = await selectOneFinalist(discovery);

    expect(selected.params).toMatchObject({ batch_config: "64x262144" });
  });

  it("selects valid unique finalists and resumes completed seed jobs", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-nanochat-validation-"));
    const discovery = path.join(dir, "results.json");
    const discoveryRound = path.join(dir, "results.round_1.json");
    const output = path.join(dir, "validation");
    const benchmark = path.join(dir, "fake_benchmark.py");
    const counter = path.join(dir, "counter.txt");
    const good = nanochatParams({ depth: 10 });
    const better = nanochatParams({ depth: 12 });
    await writeFile(
      discovery,
      JSON.stringify({
        direction: "minimize",
        all_trials: [
          trial(0, 0.8, nanochatParams({ depth: 9 }), { autotune_failure_reason: "timeout" }),
          trial(1, 0.9, good),
          trial(2, 0.91, good),
          trial(3, 100, nanochatParams({ depth: 11 }))
        ]
      }),
      "utf8"
    );
    await writeFile(
      discoveryRound,
      JSON.stringify({
        direction: "minimize",
        all_trials: [
          trial(0, 0.1, nanochatParams({ depth: 20 }), { autotune_transfer: true }),
          trial(1, 0.85, better)
        ]
      }),
      "utf8"
    );
    await writeFile(
      benchmark,
      [
        "import argparse, hashlib, json, os",
        "from pathlib import Path",
        "parser = argparse.ArgumentParser()",
        ...Object.values(paramFlags()).map((flag) => `parser.add_argument('${flag}')`),
        "args = parser.parse_args()",
        "seed = int(os.environ['NANOCHAT_BENCHMARK_SEED'])",
        "counter = Path(os.environ['FAKE_VALIDATION_COUNTER'])",
        "counter.write_text(counter.read_text() + 'x' if counter.exists() else 'x')",
        "batch = args.batch_config.split('x')",
        "config = vars(args)",
        "config.pop('batch_config')",
        "config['device_batch_size'] = int(batch[0])",
        "config['total_batch_size'] = int(batch[1])",
        "for key, value in list(config.items()):",
        "    if key in {'window_pattern'}: continue",
        "    config[key] = float(value) if '.' in str(value) else int(value)",
        "benchmark_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()",
        "protocol = {'time_budget_seconds': 300, 'eval_tokens': 40 * 524288, 'max_seq_len': 2048, 'vocab_size': 8192, 'gpu_count': 1, 'autoresearch_commit': '228791fb499afffb54b46200aca536f79142f117', 'benchmark_sha256': benchmark_sha, 'adapter_sha256': 'c' * 64, 'data_identity_sha256': 'd' * 64, 'source_sha256': {'train.py': 'e' * 64}, 'runtime': {'gpu': 'fake'}}",
        "protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(',', ':')).encode()).hexdigest()",
        "metric = float(os.environ.get('FAKE_METRIC', 0.9 + seed * 0.01))",
        "result = {'schema_version': 1, 'status': 'complete', 'metric': metric, 'seed': seed, 'config': config, 'protocol': protocol, 'protocol_sha256': protocol_hash}",
        "Path(os.environ['NANOCHAT_BENCHMARK_RESULT_JSON']).write_text(json.dumps(result))",
        "print(f'autotune_metric={result[\"metric\"]}')",
        "raise SystemExit(int(os.environ.get('FAKE_EXIT_CODE', '0')))"
      ].join("\n"),
      "utf8"
    );
    const provenanceDir = path.join(dir, "trial_results");
    const discoveryProvenance = path.join(provenanceDir, "discovery.json");
    await mkdir(provenanceDir);
    const discoveryArgv = Object.entries(paramFlags()).flatMap(([name, flag]) => [flag, String(better[name as keyof typeof better])]);
    await runPython([benchmark, ...discoveryArgv], {
      FAKE_VALIDATION_COUNTER: counter,
      FAKE_METRIC: "0.85",
      NANOCHAT_BENCHMARK_SEED: "42",
      NANOCHAT_BENCHMARK_RESULT_JSON: discoveryProvenance
    });
    await writeFile(counter, "", "utf8");

    const args = [
      VALIDATOR,
      "--result",
      `baseline=${discovery}`,
      "--result",
      `baseline=${discoveryRound}`,
      "--output-dir",
      output,
      "--benchmark",
      benchmark,
      "--finalists",
      "1",
      "--seeds",
      "0,1"
    ];
    const validationEnv = { FAKE_VALIDATION_COUNTER: counter, NANOCHAT_DATA_IDENTITY_SHA256: "d".repeat(64) };
    await runPython(args, validationEnv);
    await runPython(args, validationEnv);

    const selection = JSON.parse(await readFile(path.join(output, "selected_finalists.json"), "utf8"));
    const summary = JSON.parse(await readFile(path.join(output, "summary.json"), "utf8"));
    expect(selection.methods.baseline[0]).toMatchObject({ source_trial: 1, discovery_value: 0.85, params: better });
    expect(summary.methods.baseline[0]).toMatchObject({ mean: 0.905, n: 2, metrics_by_seed: { "0": 0.9, "1": 0.91 } });
    expect(await readFile(counter, "utf8")).toBe("xx");

    const candidateId = selection.methods.baseline[0].candidate_id;
    const cachedResult = path.join(output, "jobs", candidateId, "seed_0", "attempt_001", "result.json");
    const tampered = JSON.parse(await readFile(cachedResult, "utf8"));
    tampered.protocol_sha256 = "b".repeat(64);
    await writeFile(cachedResult, JSON.stringify(tampered), "utf8");
    await mkdir(path.join(output, "jobs", candidateId, "seed_0", "attempt_002"));
    await runPython(args, validationEnv);
    expect(await readFile(counter, "utf8")).toBe("xxx");

    const failedOutput = path.join(dir, "failed-validation");
    const failedCounter = path.join(dir, "failed-counter.txt");
    const failedArgs = args.map((value) => (value === output ? failedOutput : value));
    failedArgs.push("--seeds", "0", "--max-attempts", "1");
    const failedEnv = {
      FAKE_VALIDATION_COUNTER: failedCounter,
      FAKE_EXIT_CODE: "1",
      NANOCHAT_DATA_IDENTITY_SHA256: "d".repeat(64)
    };
    const failedAttempt = await runPythonProcess(failedArgs, failedEnv);
    expect(failedAttempt.code).not.toBe(0);
    expect(failedAttempt.stderr).toContain("validation exhausted 1 attempts");
    expect(await readFile(failedCounter, "utf8")).toBe("x");

    const failedResume = await runPythonProcess(failedArgs, {
      FAKE_VALIDATION_COUNTER: failedCounter,
      NANOCHAT_DATA_IDENTITY_SHA256: "d".repeat(64)
    });
    expect(failedResume.code).not.toBe(0);
    expect(failedResume.stderr).toContain("validation exhausted 1 attempts");
    expect(await readFile(failedCounter, "utf8")).toBe("x");

    const failedSelection = JSON.parse(await readFile(path.join(failedOutput, "selected_finalists.json"), "utf8"));
    const failedCandidateId = failedSelection.methods.baseline[0].candidate_id;
    const completion = path.join(failedOutput, "jobs", failedCandidateId, "seed_0", "attempt_001", "completed.json");
    await writeFile(completion, "{}", "utf8");
    const malformedResume = await runPythonProcess(failedArgs, failedEnv);
    expect(malformedResume.code).not.toBe(0);
    expect(malformedResume.stderr).toContain("invalid validation completion marker");
  });

  it("rejects validation seeds that overlap the discovery seed", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-nanochat-validation-"));
    const discovery = path.join(dir, "results.json");
    await mkdir(path.join(dir, "output"));
    await writeFile(discovery, JSON.stringify({ direction: "minimize", all_trials: [trial(0, 0.9, nanochatParams())] }), "utf8");

    const result = await runPythonProcess([
      VALIDATOR,
      "--result",
      `baseline=${discovery}`,
      "--output-dir",
      path.join(dir, "output"),
      "--seeds",
      "1,42"
    ]);

    expect(result.code).not.toBe(0);
    expect(result.stderr).toContain("must not include discovery seed 42");
  });

  it("serializes concurrent validators for the same seed job", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-nanochat-validation-lock-"));
    const validatorPath = path.resolve(VALIDATOR);
    const childPath = path.join(dir, "lock_child.py");
    const jobDir = path.join(dir, "job");
    const readyPath = path.join(dir, "ready");
    const acquiredPath = path.join(dir, "acquired");
    await writeFile(
      childPath,
      [
        "import importlib.util, sys",
        "from pathlib import Path",
        `sys.path.insert(0, ${JSON.stringify(path.dirname(validatorPath))})`,
        `spec = importlib.util.spec_from_file_location('validator', ${JSON.stringify(validatorPath)})`,
        "validator = importlib.util.module_from_spec(spec)",
        "spec.loader.exec_module(validator)",
        `Path(${JSON.stringify(readyPath)}).write_text('ready')`,
        `with validator.locked_job(Path(${JSON.stringify(jobDir)})):`,
        `    Path(${JSON.stringify(acquiredPath)}).write_text('acquired')`
      ].join("\n"),
      "utf8"
    );
    const script = [
      "import importlib.util, subprocess, sys, time",
      "from pathlib import Path",
      `sys.path.insert(0, ${JSON.stringify(path.dirname(validatorPath))})`,
      `spec = importlib.util.spec_from_file_location('validator', ${JSON.stringify(validatorPath)})`,
      "validator = importlib.util.module_from_spec(spec)",
      "spec.loader.exec_module(validator)",
      `ready = Path(${JSON.stringify(readyPath)})`,
      `acquired = Path(${JSON.stringify(acquiredPath)})`,
      `with validator.locked_job(Path(${JSON.stringify(jobDir)})):`,
      `    child = subprocess.Popen([sys.executable, ${JSON.stringify(childPath)}])`,
      "    deadline = time.monotonic() + 5",
      "    while not ready.exists() and time.monotonic() < deadline: time.sleep(0.01)",
      "    assert ready.exists(), 'child did not reach the lock'",
      "    time.sleep(0.1)",
      "    assert not acquired.exists(), 'child acquired a held job lock'",
      "assert child.wait(timeout=5) == 0",
      "assert acquired.read_text() == 'acquired'"
    ].join("\n");

    await runPython(["-c", script], {});
  });
});

function nanochatParams(overrides: Record<string, string | number> = {}) {
  return {
    depth: 8,
    aspect_ratio: 64,
    head_dim: 128,
    batch_config: "128x524288",
    embedding_lr: 0.6,
    unembedding_lr: 0.004,
    matrix_lr: 0.04,
    scalar_lr: 0.5,
    weight_decay: 0.2,
    warmup_ratio: 0,
    warmdown_ratio: 0.5,
    final_lr_frac: 0,
    window_pattern: "SSSL",
    ...overrides
  };
}

function paramFlags() {
  return {
    depth: "--depth",
    aspect_ratio: "--aspect-ratio",
    head_dim: "--head-dim",
    batch_config: "--batch-config",
    embedding_lr: "--embedding-lr",
    unembedding_lr: "--unembedding-lr",
    matrix_lr: "--matrix-lr",
    scalar_lr: "--scalar-lr",
    weight_decay: "--weight-decay",
    warmup_ratio: "--warmup-ratio",
    warmdown_ratio: "--warmdown-ratio",
    final_lr_frac: "--final-lr-frac",
    window_pattern: "--window-pattern"
  };
}

function trial(number: number, value: number, params: Record<string, unknown>, user_attrs = {}) {
  return { number, value, params, state: "COMPLETE", user_attrs };
}

async function runPython(args: string[], env: Record<string, string>) {
  const result = await runPythonProcess(args, env);
  if (result.code !== 0) throw new Error(result.stderr || result.stdout);
  return result.stdout;
}

async function selectOneFinalist(discovery: string) {
  const selected = await runPython([
    "-c",
    [
      "import json, sys",
      "from pathlib import Path",
      "from validate_nanochat import select_finalists",
      "selected, _ = select_finalists([Path(sys.argv[1])], 1)",
      "print(json.dumps(selected[0]))"
    ].join("; "),
    discovery
  ], { PYTHONPATH: path.dirname(VALIDATOR) });
  return JSON.parse(selected);
}

function runPythonProcess(args: string[], env: Record<string, string> = {}) {
  return new Promise<{ code: number | null; stdout: string; stderr: string }>((resolve, reject) => {
    const child = spawn("python3", args, { env: { ...process.env, ...env }, stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => (stdout += chunk));
    child.stderr.on("data", (chunk) => (stderr += chunk));
    child.on("error", reject);
    child.on("close", (code) => resolve({ code, stdout, stderr }));
  });
}
