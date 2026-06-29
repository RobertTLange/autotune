import { readFile } from "node:fs/promises";
import path from "node:path";

describe("packaged examples", () => {
  it("uses concise example filenames in docs", async () => {
    const readme = await readFile("README.md", "utf8");
    const packageJson = JSON.parse(await readFile("package.json", "utf8"));

    expect(readme).toContain("examples/mnist_cnn.py");
    expect(readme).toContain("examples/cifar10_resnet.py");
    expect(readme).toContain("examples/cifar10_speedrun.py");
    expect(readme).not.toContain("mnist_cnn_no_cli.py");
    expect(packageJson.files).toContain("examples/*.py");
  });

  it("keeps deep-learning examples intentionally agent-compatible", async () => {
    const mnist = await readFile(path.join("examples", "mnist_cnn.py"), "utf8");
    const cifar = await readFile(path.join("examples", "cifar10_resnet.py"), "utf8");

    for (const source of [mnist, cifar]) {
      expect(source).not.toContain("argparse");
      expect(source).not.toContain("autotune_metric");
    }
  });

  it("defines a full CIFAR-10 ResNet training example", async () => {
    const cifar = await readFile(path.join("examples", "cifar10_resnet.py"), "utf8");

    expect(cifar).toContain("datasets.CIFAR10");
    expect(cifar).toContain("class BasicBlock");
    expect(cifar).toContain("class CifarResNet");
    expect(cifar).not.toContain("Subset(");
    expect(cifar).toContain("validation_accuracy");
  });

  it("defines an Autotune-native CIFAR-10 speedrun example", async () => {
    const speedrun = await readFile(path.join("examples", "cifar10_speedrun.py"), "utf8");

    expect(speedrun).toContain("argparse.ArgumentParser");
    expect(speedrun).toContain("autotune_metric=");
    expect(speedrun).toContain("ACCURACY_THRESHOLD = 0.94");
    expect(speedrun).toContain("1e6 ** (20 * accuracy_gap)");
    expect(speedrun).toContain("CIFAR10_SPEEDRUN_NUM_RUNS");
    expect(speedrun).not.toContain("--num-runs");
    expect(speedrun).not.toContain("--tta-uncertain-quantile");
  });

  it("keeps CIFAR-10 speedrun CLI flags limited to training hyperparameters", async () => {
    const speedrun = await readFile(path.join("examples", "cifar10_speedrun.py"), "utf8");
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
});
