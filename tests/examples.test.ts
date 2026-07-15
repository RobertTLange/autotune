import { chmod, mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { spawn } from "node:child_process";
import { tmpdir } from "node:os";
import path from "node:path";

const EXAMPLES = {
  cifar10Speedrun: path.join("examples", "cifar10_speedrun", "cifar10_speedrun.py"),
  mnist: path.join("examples", "mnist", "mnist_cnn.py"),
  nanochat: path.join("examples", "nanochat", "nanochat_benchmark.py")
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
    expect(readme).toContain("NANOCHAT_DIR");
    expect(readme).toContain("python -m nanochat.dataset");
    expect(readme).toContain("NANOCHAT_BENCHMARK_NUM_ITERATIONS");
    expect(readme).toContain("NANOCHAT_BENCHMARK_TARGET_SECONDS");
    expect(readme).toContain("NANOCHAT_BENCHMARK_TOKENS_PER_SECOND");
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

    expect(nanochat).toContain("argparse.ArgumentParser");
    expect(nanochat).toContain("NANOCHAT_DIR");
    expect(nanochat).toContain("autotune_metric=");
    expect(nanochat).toContain("OOM_PENALTY = 100.0");
    expect(nanochat).toContain("Minimum validation bpb");
    expect(nanochat).not.toContain("shell=True");
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
        "8x131072",
        "8x262144",
        "8x524288",
        "8x1048576",
        "8x2097152",
        "16x262144",
        "16x524288",
        "16x1048576",
        "16x2097152",
        "32x524288",
        "32x1048576",
        "32x2097152"
      ]
    });
  });

  it("runs the nanochat benchmark wrapper against a fake nanochat command", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-nanochat-"));
    const nanochatDir = path.join(dir, "nanochat");
    const fakePython = path.join(dir, "python");
    await mkdir(path.join(nanochatDir, "scripts"), { recursive: true });
    await writeFile(path.join(nanochatDir, "scripts", "base_train.py"), "# fake\n", "utf8");
    await writeFile(
      fakePython,
      [
        "#!/usr/bin/env bash",
        "printf '%s\\n' \"$@\" > \"$FAKE_NANOCHAT_ARGV\"",
        "echo 'Step 00020 | Validation bpb: 0.345678'",
        "echo 'Minimum validation bpb: 0.123456'"
      ].join("\n"),
      "utf8"
    );
    await chmod(fakePython, 0o755);
    const argvPath = path.join(dir, "argv.txt");

    const output = await runPythonScript([EXAMPLES.nanochat, "--warmup-ratio", "0.1"], {
      NANOCHAT_DIR: nanochatDir,
      NANOCHAT_PYTHON: fakePython,
      FAKE_NANOCHAT_ARGV: argvPath,
      NANOCHAT_BENCHMARK_NUM_ITERATIONS: "20"
    });

    expect(output).toContain("autotune_metric=0.123456");
    expect(await readFile(argvPath, "utf8")).toContain("--warmup-steps\n2");
  });

  it("derives nanochat iterations from a target token budget", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-nanochat-"));
    const nanochatDir = path.join(dir, "nanochat");
    const fakePython = path.join(dir, "python");
    await mkdir(path.join(nanochatDir, "scripts"), { recursive: true });
    await writeFile(path.join(nanochatDir, "scripts", "base_train.py"), "# fake\n", "utf8");
    await writeFile(
      fakePython,
      [
        "#!/usr/bin/env bash",
        "printf '%s\\n' \"$@\" > \"$FAKE_NANOCHAT_ARGV\"",
        "echo 'Minimum validation bpb: 0.123456'"
      ].join("\n"),
      "utf8"
    );
    await chmod(fakePython, 0o755);
    const argvPath = path.join(dir, "argv.txt");

    const output = await runPythonScript([EXAMPLES.nanochat, "--warmup-ratio", "0.5"], {
      NANOCHAT_DIR: nanochatDir,
      NANOCHAT_PYTHON: fakePython,
      FAKE_NANOCHAT_ARGV: argvPath,
      NANOCHAT_BENCHMARK_TARGET_TOKENS: "1048576",
      NANOCHAT_BENCHMARK_TARGET_SECONDS: "300",
      NANOCHAT_BENCHMARK_TOKENS_PER_SECOND: "10000000"
    });

    const argv = await readFile(argvPath, "utf8");
    expect(output).toContain("autotune_metric=0.123456");
    expect(argv).toContain("--num-iterations\n2");
    expect(argv).toContain("--warmup-steps\n1");
  });

  it("prefers explicit nanochat iterations over a target token budget", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-nanochat-"));
    const nanochatDir = path.join(dir, "nanochat");
    const fakePython = path.join(dir, "python");
    await mkdir(path.join(nanochatDir, "scripts"), { recursive: true });
    await writeFile(path.join(nanochatDir, "scripts", "base_train.py"), "# fake\n", "utf8");
    await writeFile(
      fakePython,
      [
        "#!/usr/bin/env bash",
        "printf '%s\\n' \"$@\" > \"$FAKE_NANOCHAT_ARGV\"",
        "echo 'Minimum validation bpb: 0.123456'"
      ].join("\n"),
      "utf8"
    );
    await chmod(fakePython, 0o755);
    const argvPath = path.join(dir, "argv.txt");

    const output = await runPythonScript([EXAMPLES.nanochat], {
      NANOCHAT_DIR: nanochatDir,
      NANOCHAT_PYTHON: fakePython,
      FAKE_NANOCHAT_ARGV: argvPath,
      NANOCHAT_BENCHMARK_TARGET_TOKENS: "1048576",
      NANOCHAT_BENCHMARK_NUM_ITERATIONS: "20"
    });

    const argv = await readFile(argvPath, "utf8");
    expect(output).toContain("autotune_metric=0.123456");
    expect(argv).toContain("--num-iterations\n20");
  });

  it("derives nanochat iterations from a calibrated time budget", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-nanochat-"));
    const nanochatDir = path.join(dir, "nanochat");
    const fakePython = path.join(dir, "python");
    await mkdir(path.join(nanochatDir, "scripts"), { recursive: true });
    await writeFile(path.join(nanochatDir, "scripts", "base_train.py"), "# fake\n", "utf8");
    await writeFile(
      fakePython,
      [
        "#!/usr/bin/env bash",
        "printf '%s\\n' \"$@\" > \"$FAKE_NANOCHAT_ARGV\"",
        "echo 'Minimum validation bpb: 0.123456'"
      ].join("\n"),
      "utf8"
    );
    await chmod(fakePython, 0o755);
    const argvPath = path.join(dir, "argv.txt");

    const output = await runPythonScript([EXAMPLES.nanochat], {
      NANOCHAT_DIR: nanochatDir,
      NANOCHAT_PYTHON: fakePython,
      FAKE_NANOCHAT_ARGV: argvPath,
      NANOCHAT_BENCHMARK_TARGET_SECONDS: "300",
      NANOCHAT_BENCHMARK_TOKENS_PER_SECOND: "10000000"
    });

    const argv = await readFile(argvPath, "utf8");
    expect(output).toContain("autotune_metric=0.123456");
    expect(argv).toContain("--num-iterations\n5723");
  });

  it("can launch nanochat through torchrun for multi-GPU trials", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-nanochat-"));
    const nanochatDir = path.join(dir, "nanochat");
    const fakePython = path.join(dir, "python");
    await mkdir(path.join(nanochatDir, "scripts"), { recursive: true });
    await writeFile(path.join(nanochatDir, "scripts", "base_train.py"), "# fake\n", "utf8");
    await writeFile(
      fakePython,
      [
        "#!/usr/bin/env bash",
        "printf '%s\\n' \"$@\" > \"$FAKE_NANOCHAT_ARGV\"",
        "echo 'Minimum validation bpb: 0.123456'"
      ].join("\n"),
      "utf8"
    );
    await chmod(fakePython, 0o755);
    const argvPath = path.join(dir, "argv.txt");

    const output = await runPythonScript(
      [
        EXAMPLES.nanochat,
        "--device-batch-size",
        "32",
        "--total-batch-size",
        "524288"
      ],
      {
        NANOCHAT_DIR: nanochatDir,
        NANOCHAT_PYTHON: fakePython,
        FAKE_NANOCHAT_ARGV: argvPath,
        NANOCHAT_BENCHMARK_NPROC_PER_NODE: "8"
      }
    );

    const argv = await readFile(argvPath, "utf8");
    expect(output).toContain("autotune_metric=0.123456");
    expect(argv).toContain("torch.distributed.run\n");
    expect(argv).toContain("--standalone\n");
    expect(argv).toContain("--nproc_per_node=8\n");
    expect(argv).toContain("scripts.base_train\n");
    expect(argv).toContain("scripts.base_train\n--\n--run\n");
  });

  it("maps nanochat paired batch configs to device and total batch flags", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-nanochat-"));
    const nanochatDir = path.join(dir, "nanochat");
    const fakePython = path.join(dir, "python");
    await mkdir(path.join(nanochatDir, "scripts"), { recursive: true });
    await writeFile(path.join(nanochatDir, "scripts", "base_train.py"), "# fake\n", "utf8");
    await writeFile(
      fakePython,
      [
        "#!/usr/bin/env bash",
        "printf '%s\\n' \"$@\" > \"$FAKE_NANOCHAT_ARGV\"",
        "echo 'Minimum validation bpb: 0.123456'"
      ].join("\n"),
      "utf8"
    );
    await chmod(fakePython, 0o755);
    const argvPath = path.join(dir, "argv.txt");

    const output = await runPythonScript([EXAMPLES.nanochat, "--batch-config", "16x1048576"], {
      NANOCHAT_DIR: nanochatDir,
      NANOCHAT_PYTHON: fakePython,
      FAKE_NANOCHAT_ARGV: argvPath,
      NANOCHAT_BENCHMARK_NPROC_PER_NODE: "8",
      NANOCHAT_BENCHMARK_NUM_ITERATIONS: "20"
    });

    const argv = await readFile(argvPath, "utf8");
    expect(output).toContain("autotune_metric=0.123456");
    expect(argv).toContain("--device-batch-size\n16");
    expect(argv).toContain("--total-batch-size\n1048576");
  });

  it("penalizes infeasible nanochat batch geometry", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-nanochat-"));
    const nanochatDir = path.join(dir, "nanochat");
    await mkdir(path.join(nanochatDir, "scripts"), { recursive: true });
    await writeFile(path.join(nanochatDir, "scripts", "base_train.py"), "# fake\n", "utf8");

    const result = await runPythonProcess([EXAMPLES.nanochat], {
      NANOCHAT_DIR: nanochatDir,
      NANOCHAT_BENCHMARK_MAX_SEQ_LEN: "8192"
    });

    expect(result.code).not.toBe(0);
    expect(result.stdout).toContain("autotune_metric=100.0");
  });

  it("marks nanochat OOM penalties as failed trials", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-nanochat-"));
    const nanochatDir = path.join(dir, "nanochat");
    const fakePython = path.join(dir, "python");
    await mkdir(path.join(nanochatDir, "scripts"), { recursive: true });
    await writeFile(path.join(nanochatDir, "scripts", "base_train.py"), "# fake\n", "utf8");
    await writeFile(fakePython, ["#!/usr/bin/env bash", "echo 'CUDA out of memory' >&2", "exit 1"].join("\n"), "utf8");
    await chmod(fakePython, 0o755);

    const result = await runPythonProcess([EXAMPLES.nanochat], {
      NANOCHAT_DIR: nanochatDir,
      NANOCHAT_PYTHON: fakePython,
      NANOCHAT_BENCHMARK_NUM_ITERATIONS: "1"
    });

    expect(result.code).not.toBe(0);
    expect(result.stdout).toContain("autotune_metric=100.0");
  });

  it("accounts for torchrun world size when checking nanochat batch geometry", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-nanochat-"));
    const nanochatDir = path.join(dir, "nanochat");
    await mkdir(path.join(nanochatDir, "scripts"), { recursive: true });
    await writeFile(path.join(nanochatDir, "scripts", "base_train.py"), "# fake\n", "utf8");

    const result = await runPythonProcess(
      [
        EXAMPLES.nanochat,
        "--device-batch-size",
        "32",
        "--total-batch-size",
        "65536"
      ],
      {
        NANOCHAT_DIR: nanochatDir,
        NANOCHAT_BENCHMARK_NPROC_PER_NODE: "8"
      }
    );

    expect(result.code).not.toBe(0);
    expect(result.stdout).toContain("autotune_metric=100.0");
  });
});

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
