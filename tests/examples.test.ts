import { chmod, mkdir, mkdtemp, readFile, rename, writeFile } from "node:fs/promises";
import { spawn } from "node:child_process";
import { tmpdir } from "node:os";
import path from "node:path";

const EXAMPLES = {
  cifar10Speedrun: path.join("examples", "cifar10_speedrun", "cifar10_speedrun.py"),
  mnist: path.join("examples", "mnist", "mnist_cnn.py"),
  nanochat: path.join("examples", "nanochat", "nanochat_benchmark.py"),
  nanochatTrain: path.join("examples", "nanochat", "autoresearch_train.py"),
  nanochatCache: path.join("examples", "nanochat", "prepare_nanochat_cache.py")
} as const;

describe("packaged examples", () => {
  it("documents the task-based example layout", async () => {
    const readme = await readFile("README.md", "utf8");
    const examplesReadme = await readFile(path.join("examples", "README.md"), "utf8");
    const packageJson = JSON.parse(await readFile("package.json", "utf8"));

    expect(readme).toContain("examples/mnist/mnist_cnn.py");
    expect(readme).toContain("examples/README.md");
    expect(examplesReadme).toContain("mnist/mnist_cnn.py");
    expect(examplesReadme).toContain("pid_controller/pid_controller.cpp");
    expect(examplesReadme).toContain("cifar10_speedrun/cifar10_speedrun.py");
    expect(examplesReadme).toContain("nanochat/nanochat_benchmark.py");
    expect(examplesReadme).toContain("bbob/benchmark.py");
    expect(examplesReadme).toContain("bbob/run_experiments.sh");
    expect(readme).not.toContain("mnist_cnn_no_cli.py");
    expect(packageJson.files).toContain("examples/*/*.py");
    expect(packageJson.files).toContain("examples/*/*.cpp");
    expect(packageJson.files).toContain("examples/*/*.yaml");
    expect(packageJson.files).toContain("examples/*/*.sh");
    expect(packageJson.files).toContain("examples/*/*.sbatch");
    expect(packageJson.files).toContain("examples/README.md");
  });

  it("documents benchmark setup details in examples README", async () => {
    const readme = await readFile(path.join("examples", "README.md"), "utf8");

    expect(readme).toContain("CIFAR10_SPEEDRUN_DATA_DIR");
    expect(readme).toContain("CIFAR10_SPEEDRUN_RESULTS_DIR");
    expect(readme).toContain("CIFAR10_SPEEDRUN_NUM_RUNS=100");
    expect(readme).toContain("AUTORESEARCH_DIR");
    expect(readme).toContain("uv run --frozen prepare.py");
    expect(readme).toContain("NANOCHAT_BENCHMARK_SEED");
    expect(readme).toContain("20,971,520");
    expect(readme).not.toContain("NANOCHAT_BENCHMARK_TOKENS_PER_SECOND");
  });

  it("keeps the MNIST example intentionally agent-compatible", async () => {
    const mnist = await readFile(EXAMPLES.mnist, "utf8");

    expect(mnist).not.toContain("argparse");
    expect(mnist).not.toContain("autotune_metric");
  });

  it("defines an Autotune-native CIFAR-10 speedrun example", async () => {
    const speedrun = await readFile(EXAMPLES.cifar10Speedrun, "utf8");

    expect(speedrun).toContain("argparse.ArgumentParser");
    expect(speedrun).toContain("autotune_metric=");
    expect(speedrun).toContain("ACCURACY_THRESHOLD = 0.94");
    expect(speedrun).toContain("1e6 ** (20 * accuracy_gap)");
    expect(speedrun).toContain("CIFAR10_SPEEDRUN_NUM_RUNS");
    expect(speedrun).toContain("CIFAR10_SPEEDRUN_RESULTS_DIR");
    expect(speedrun).toContain("config_fingerprint");
    expect(speedrun).toContain("write_optional_results_json(results, config)");
    expect(speedrun).not.toContain("--num-runs");
    expect(speedrun).not.toContain("--tta-uncertain-quantile");
  });

  it("keeps CIFAR-10 speedrun CLI flags limited to training hyperparameters", async () => {
    const speedrun = await readFile(EXAMPLES.cifar10Speedrun, "utf8");
    const flags = [...speedrun.matchAll(/parser\.add_argument\("([^"]+)"/g)].map((match) => match[1]).sort();

    expect(flags).toEqual([
      "--bias-lr",
      "--brightness-range",
      "--contrast-range",
      "--head-lr",
      "--label-smoothing",
      "--muon-lr",
      "--muon-momentum",
      "--sgd-momentum",
      "--train-epochs",
      "--training-batch-size",
      "--translate",
      "--weight-decay-scale",
      "--whiten-bias-epochs"
    ]);
  });

  it("defines an Autotune-native nanochat benchmark wrapper", async () => {
    const nanochat = await readFile(EXAMPLES.nanochat, "utf8");
    const adapter = await readFile(EXAMPLES.nanochatTrain, "utf8");

    expect(nanochat).toContain("argparse.ArgumentParser");
    expect(nanochat).toContain("AUTORESEARCH_DIR");
    expect(nanochat).toContain("228791fb499afffb54b46200aca536f79142f117");
    expect(nanochat).toContain("autotune_metric=");
    expect(nanochat).toContain("OOM_PENALTY = 100.0");
    expect(nanochat).toContain("EVAL_TOKENS = 40 * 524288");
    expect(nanochat).not.toContain("shell=True");
    expect(adapter.indexOf("emit_execution_provenance(materialized, globals_dict)")).toBeGreaterThan(adapter.indexOf("exec(compile("));
  });

  it("launches the comparable nanochat protocol on one GPU without throughput calibration", async () => {
    const local = await readFile(path.join("examples", "nanochat", "run_nanochat_benchmark.sh"), "utf8");
    const slurm = await readFile(path.join("examples", "nanochat", "run_nanobench_ablation.sbatch"), "utf8");

    expect(local).toContain("AUTORESEARCH_DIR");
    expect(local).toContain("--sampler-seed");
    expect(local).toContain("validate_nanochat.py");
    expect(slurm).toContain("#SBATCH --gres=gpu:1");
    expect(slurm).toContain('N_JOBS must be 1');
    expect(slurm).toContain("results.round_$round.json");
    expect(slurm).toContain("uv run --frozen prepare.py");
    expect(`${local}\n${slurm}`).not.toMatch(/export NANOCHAT_DATA_IDENTITY_SHA256="\$\(/);
    expect(`${local}\n${slurm}`).not.toContain("TOKENS_PER_SECOND");
    expect(`${local}\n${slurm}`).not.toContain("NPROC_PER_NODE");
  });

  it("keeps nanochat benchmark CLI flags limited to paper hyperparameters", async () => {
    const nanochat = await readFile(EXAMPLES.nanochat, "utf8");
    const flags = [...nanochat.matchAll(/parser\.add_argument\("([^"]+)"/g)].map((match) => match[1]).sort();

    expect(flags).toEqual([
      "--aspect-ratio",
      "--batch-config",
      "--depth",
      "--device-batch-size",
      "--embedding-lr",
      "--final-lr-frac",
      "--head-dim",
      "--matrix-lr",
      "--scalar-lr",
      "--total-batch-size",
      "--unembedding-lr",
      "--warmdown-ratio",
      "--warmup-ratio",
      "--weight-decay",
      "--window-pattern"
    ]);
  });

  it("ships a parseable nanochat benchmark search space", async () => {
    const { parseSearchSpaceText } = await import("../src/search-space.js");
    const text = await readFile(path.join("examples", "nanochat", "nanochat_search_space.yaml"), "utf8");
    const searchSpace = parseSearchSpaceText(text);

    expect(searchSpace.direction).toBe("minimize");
    expect(searchSpace.failure_value).toBe(100);
    expect(searchSpace.optuna).toMatchObject({ sampler: "tpe", pruner: "none" });
    expect(searchSpace.parameters.map((parameter) => parameter.name).sort()).toEqual([
      "aspect_ratio",
      "batch_config",
      "depth",
      "embedding_lr",
      "final_lr_frac",
      "head_dim",
      "matrix_lr",
      "scalar_lr",
      "unembedding_lr",
      "warmdown_ratio",
      "warmup_ratio",
      "weight_decay",
      "window_pattern"
    ]);
    const batchConfig = searchSpace.parameters.find((parameter) => parameter.name === "batch_config");
    expect(batchConfig).toMatchObject({
      type: "categorical",
      choices: [
        "16x131072",
        "16x262144",
        "16x524288",
        "32x131072",
        "32x262144",
        "32x524288",
        "64x131072",
        "64x262144",
        "64x524288",
        "128x262144",
        "128x524288"
      ]
    });
  });

  it("runs the pinned canonical harness and records comparable protocol metadata", async () => {
    const fixture = await createFakeAutoresearch();
    const resultPath = path.join(fixture.dir, "result.json");
    const configPath = path.join(fixture.root, "config.json");

    const output = await runPythonScript([EXAMPLES.nanochat, "--batch-config", "64x524288"], {
      AUTORESEARCH_DIR: fixture.root,
      AUTORESEARCH_UV: fixture.python,
      NANOCHAT_BENCHMARK_RESULT_JSON: resultPath,
      NANOCHAT_BENCHMARK_SEED: "7",
      NANOCHAT_DATA_IDENTITY_SHA256: fixture.identity,
      HOME: fixture.home,
      PATH: `${fixture.binDir}:${process.env.PATH}`
    });

    const result = JSON.parse(await readFile(resultPath, "utf8"));
    const config = JSON.parse(await readFile(configPath, "utf8"));
    const uvArgv = await readFile(path.join(fixture.root, "uv-argv.txt"), "utf8");
    expect(output).toContain("autotune_metric=0.912345");
    expect(config).toMatchObject({ device_batch_size: 64, total_batch_size: 524288, seed: 7 });
    expect(uvArgv).toContain("--frozen");
    expect(uvArgv).not.toContain("--no-sync");
    expect(result).toMatchObject({
      metric: 0.912345,
      seed: 7,
      protocol: { time_budget_seconds: 300, eval_tokens: 20971520, max_seq_len: 2048, vocab_size: 8192 },
      autoresearch_commit: "228791fb499afffb54b46200aca536f79142f117",
      training: { training_seconds: 300, total_tokens_m: 501.2, num_steps: 956 }
    });
  });

  it("injects only HPO constants and the evaluation seed into canonical train.py", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-autoresearch-adapter-"));
    const source = [
      "import json",
      "ASPECT_RATIO = 64",
      "HEAD_DIM = 128",
      'WINDOW_PATTERN = "SSSL"',
      "TOTAL_BATCH_SIZE = 2**19",
      "EMBEDDING_LR = 0.6",
      "UNEMBEDDING_LR = 0.004",
      "MATRIX_LR = 0.04",
      "SCALAR_LR = 0.5",
      "WEIGHT_DECAY = 0.2",
      "WARMUP_RATIO = 0.0",
      "WARMDOWN_RATIO = 0.5",
      "FINAL_LR_FRAC = 0.0",
      "DEPTH = 8",
      "DEVICE_BATCH_SIZE = 128",
      "class Cuda:",
      "    def manual_seed(self, value): self.seed = value",
      "class Torch:",
      "    cuda = Cuda()",
      "    def manual_seed(self, value): self.seed = value",
      "torch = Torch()",
      "torch.manual_seed(42)",
      "torch.cuda.manual_seed(42)",
      "print(json.dumps({'depth': DEPTH, 'aspect_ratio': ASPECT_RATIO, 'head_dim': HEAD_DIM, 'window_pattern': WINDOW_PATTERN, 'device_batch_size': DEVICE_BATCH_SIZE, 'total_batch_size': TOTAL_BATCH_SIZE, 'embedding_lr': EMBEDDING_LR, 'unembedding_lr': UNEMBEDDING_LR, 'matrix_lr': MATRIX_LR, 'scalar_lr': SCALAR_LR, 'weight_decay': WEIGHT_DECAY, 'warmup_ratio': WARMUP_RATIO, 'warmdown_ratio': WARMDOWN_RATIO, 'final_lr_frac': FINAL_LR_FRAC, 'seed': torch.seed, 'cuda_seed': torch.cuda.seed}))"
    ].join("\n");
    await writeFile(path.join(dir, "train.py"), source, "utf8");
    await writeFile(path.join(dir, "prepare.py"), "# canonical prepare fixture\n", "utf8");
    const config = {
      depth: 12,
      aspect_ratio: 48,
      head_dim: 64,
      window_pattern: "LLLL",
      device_batch_size: 32,
      total_batch_size: 262144,
      embedding_lr: 0.4,
      unembedding_lr: 0.003,
      matrix_lr: 0.03,
      scalar_lr: 0.4,
      weight_decay: 0.1,
      warmup_ratio: 0.1,
      warmdown_ratio: 0.4,
      final_lr_frac: 0.05,
      seed: 9
    };

    const output = await runPythonScript([EXAMPLES.nanochatTrain], {
      AUTORESEARCH_DIR: dir,
      AUTOTUNE_NANOCHAT_CONFIG: JSON.stringify(config)
    });

    expect(JSON.parse(output)).toEqual({ ...config, cuda_seed: 9 });

    const changedAfterVerification = await runPythonProcess([EXAMPLES.nanochatTrain], {
      AUTORESEARCH_DIR: dir,
      AUTOTUNE_NANOCHAT_CONFIG: JSON.stringify(config),
      AUTOTUNE_AUTORESEARCH_TRAIN_SHA256: "0".repeat(64)
    });
    expect(changedAfterVerification.code).not.toBe(0);
    expect(changedAfterVerification.stderr).toContain("train.py changed after checkout verification");

    await writeFile(path.join(dir, "train.py"), source.replace("DEPTH = 8\n", ""), "utf8");
    const drifted = await runPythonProcess([EXAMPLES.nanochatTrain], {
      AUTORESEARCH_DIR: dir,
      AUTOTUNE_NANOCHAT_CONFIG: JSON.stringify(config)
    });
    expect(drifted.code).not.toBe(0);
    expect(drifted.stderr).toContain("expected one DEPTH, found 0");
  });

  it("rejects an unpinned autoresearch checkout", async () => {
    const fixture = await createFakeAutoresearch("not-the-pinned-commit");
    const result = await runPythonProcess([EXAMPLES.nanochat], {
      AUTORESEARCH_DIR: fixture.root,
      AUTORESEARCH_UV: fixture.python,
      NANOCHAT_DATA_IDENTITY_SHA256: fixture.identity,
      HOME: fixture.home,
      PATH: `${fixture.binDir}:${process.env.PATH}`
    });

    expect(result.code).not.toBe(0);
    expect(result.stderr).toContain("expected pinned autoresearch commit");
  });

  it("penalizes batch geometry incompatible with the single-GPU protocol", async () => {
    const fixture = await createFakeAutoresearch();
    const result = await runPythonProcess(
      [EXAMPLES.nanochat, "--device-batch-size", "128", "--total-batch-size", "131072"],
      {
        AUTORESEARCH_DIR: fixture.root,
        AUTORESEARCH_UV: fixture.python,
        NANOCHAT_DATA_IDENTITY_SHA256: fixture.identity,
        HOME: fixture.home,
        PATH: `${fixture.binDir}:${process.env.PATH}`
      }
    );

    expect(result.code).not.toBe(0);
    expect(result.stdout).toContain("autotune_metric=100.0");
  });

  it("rejects non-finite HPO values before launching training", async () => {
    const result = await runPythonProcess([EXAMPLES.nanochat, "--embedding-lr", "NaN"], {});

    expect(result.code).not.toBe(0);
    expect(result.stderr).toContain("embedding_lr must be finite");
    expect(result.stdout).toContain("autotune_metric=100.0");
  });

  it("rejects extra training shards that change the canonical data profile", async () => {
    const fixture = await createFakeAutoresearch();
    await writeFile(path.join(fixture.home, ".cache", "autoresearch", "data", "shard_00010.parquet"), "data", "utf8");
    const result = await runPythonProcess([EXAMPLES.nanochat], {
      AUTORESEARCH_DIR: fixture.root,
      AUTORESEARCH_UV: fixture.python,
      NANOCHAT_DATA_IDENTITY_SHA256: fixture.identity,
      HOME: fixture.home,
      PATH: `${fixture.binDir}:${process.env.PATH}`
    });

    expect(result.code).not.toBe(0);
    expect(result.stderr).toContain("extra=['shard_00010.parquet']");
  });

  it("requires an explicit trusted cache manifest before loading the tokenizer", async () => {
    const fixture = await createFakeAutoresearch();
    const manifest = path.join(fixture.home, ".cache", "autoresearch", "autotune_data_manifest.json");
    await rename(manifest, `${manifest}.untrusted`);
    const result = await runPythonProcess([EXAMPLES.nanochat], {
      AUTORESEARCH_DIR: fixture.root,
      AUTORESEARCH_UV: fixture.python,
      NANOCHAT_DATA_IDENTITY_SHA256: fixture.identity,
      HOME: fixture.home,
      PATH: `${fixture.binDir}:${process.env.PATH}`
    });

    expect(result.code).not.toBe(0);
    expect(result.stderr).toContain("missing trusted cache manifest");
  });

  it("rehashes canonical data before launching a benchmark group", async () => {
    const fixture = await createFakeAutoresearch();
    await writeFile(path.join(fixture.home, ".cache", "autoresearch", "data", "shard_00000.parquet"), "evil", "utf8");
    const result = await runPythonProcess([EXAMPLES.nanochatCache, "verify"], { HOME: fixture.home });

    expect(result.code).not.toBe(0);
    expect(result.stderr).toContain("shard_00000.parquet");
  });

  it("marks canonical harness OOM penalties as failed trials", async () => {
    const fixture = await createFakeAutoresearch();
    await writeFile(fixture.python, ["#!/usr/bin/env bash", "echo 'CUDA out of memory' >&2", "exit 1"].join("\n"), "utf8");
    const result = await runPythonProcess([EXAMPLES.nanochat], {
      AUTORESEARCH_DIR: fixture.root,
      AUTORESEARCH_UV: fixture.python,
      NANOCHAT_DATA_IDENTITY_SHA256: fixture.identity,
      HOME: fixture.home,
      PATH: `${fixture.binDir}:${process.env.PATH}`
    });

    expect(result.code).not.toBe(0);
    expect(result.stdout).toContain("autotune_metric=100.0");
  });
});

async function createFakeAutoresearch(commit = "228791fb499afffb54b46200aca536f79142f117") {
  const dir = await mkdtemp(path.join(tmpdir(), "autotune-autoresearch-"));
  const root = path.join(dir, "autoresearch");
  const home = path.join(dir, "home");
  const binDir = path.join(dir, "bin");
  const python = path.join(dir, "python");
  await mkdir(root, { recursive: true });
  await mkdir(binDir, { recursive: true });
  await mkdir(path.join(home, ".cache", "autoresearch", "data"), { recursive: true });
  await mkdir(path.join(home, ".cache", "autoresearch", "tokenizer"), { recursive: true });
  for (const index of [...Array(10).keys(), 6542]) {
    await writeFile(path.join(home, ".cache", "autoresearch", "data", `shard_${String(index).padStart(5, "0")}.parquet`), "data", "utf8");
  }
  await writeFile(path.join(home, ".cache", "autoresearch", "tokenizer", "tokenizer.pkl"), "tokenizer", "utf8");
  await writeFile(path.join(home, ".cache", "autoresearch", "tokenizer", "token_bytes.pt"), "token bytes", "utf8");
  const identity = (await runPythonScript([EXAMPLES.nanochatCache, "create"], { HOME: home })).trim();
  await writeFile(path.join(root, "train.py"), "# fake canonical train\n", "utf8");
  await writeFile(path.join(root, "prepare.py"), "# fake canonical prepare\n", "utf8");
  await writeFile(path.join(root, "pyproject.toml"), "[project]\nname='fake'\n", "utf8");
  await writeFile(path.join(root, "uv.lock"), "version = 1\n", "utf8");
  await writeFile(path.join(root, ".python-version"), "3.10\n", "utf8");
  await writeFile(path.join(root, "README.md"), "# fake autoresearch\n", "utf8");
  await writeFile(
    path.join(binDir, "git"),
    [
      "#!/usr/bin/env bash",
      "if [[ \"$*\" == *\"rev-parse --show-toplevel\"* ]]; then echo \"$2\"; exit 0; fi",
      `if [[ \"$*\" == *\"rev-parse HEAD\" ]]; then echo '${commit}'; exit 0; fi`,
      "if [[ \"$*\" == *\"cat-file blob 228791fb499afffb54b46200aca536f79142f117:\"* ]]; then name=\"${5#*:}\"; cat \"$2/$name\"; exit 0; fi",
      "exit 0"
    ].join("\n"),
    "utf8"
  );
  await writeFile(
    python,
    [
      "#!/usr/bin/env bash",
      "printf '%s\\n' \"$@\" > \"$AUTORESEARCH_DIR/uv-argv.txt\"",
      "printf '%s' \"$AUTOTUNE_NANOCHAT_CONFIG\" > \"$AUTORESEARCH_DIR/config.json\"",
      "echo 'autotune_materialized_sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'",
      "echo 'autotune_runtime={\"cuda\":\"12.8\",\"gpu_capability\":[9,0],\"gpu_name\":\"Fake H100\",\"kernels_package\":\"0.12.1\",\"loaded_kernels\":[{\"metadata_id\":\"fake\",\"repo_id\":\"varunneal/flash-attention-3\",\"revision\":\"main\",\"snapshot_commit\":\"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\"}],\"python\":\"3.10.12\",\"torch\":\"2.9.1\"}'",
      "echo 'val_bpb:          0.912345'",
      "echo 'training_seconds: 300.0'",
      "echo 'total_tokens_M:   501.2'",
      "echo 'num_steps:        956'"
    ].join("\n"),
    "utf8"
  );
  await chmod(path.join(binDir, "git"), 0o755);
  await chmod(python, 0o755);
  return { dir, root, home, binDir, python, identity };
}

async function runPythonScript(args: string[], env: Record<string, string>): Promise<string> {
  const result = await runPythonProcess(args, env);
  if (result.code === 0) {
    return result.stdout;
  }
  throw new Error(`python3 ${args.join(" ")} failed with ${result.code}: ${result.stderr || result.stdout}`);
}

function runPythonProcess(
  args: string[],
  env: Record<string, string>
): Promise<{ code: number | null; stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    const child = spawn("python3", args, { env: { ...process.env, ...env }, stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
    });
    child.on("error", reject);
    child.on("close", (code) => {
      resolve({ code, stdout, stderr });
    });
  });
}
