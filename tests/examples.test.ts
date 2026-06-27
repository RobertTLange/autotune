import { readFile } from "node:fs/promises";
import path from "node:path";

describe("packaged examples", () => {
  it("uses concise example filenames in docs", async () => {
    const readme = await readFile("README.md", "utf8");

    expect(readme).toContain("examples/mnist_cnn.py");
    expect(readme).toContain("examples/cifar10_resnet.py");
    expect(readme).not.toContain("mnist_cnn_no_cli.py");
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
});
