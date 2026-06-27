import { chmod, mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { detectInvocation, splitCommand } from "../src/detect.js";

describe("splitCommand", () => {
  it("splits quoted command overrides without using a shell", () => {
    expect(splitCommand("julia +nightly --project='my env'")).toEqual([
      "julia",
      "+nightly",
      "--project=my env"
    ]);
  });

  it("rejects unterminated quotes", () => {
    expect(() => splitCommand("python3 'broken")).toThrow(/unterminated/i);
  });
});

describe("detectInvocation", () => {
  it("detects known script runtimes by extension", () => {
    expect(detectInvocation("/work/train.py")).toMatchObject({
      language: "python",
      command: ["python3"],
      script: "/work/train.py"
    });
    expect(detectInvocation("/work/train.R").command).toEqual(["Rscript"]);
    expect(detectInvocation("/work/train.sh").command).toEqual(["bash"]);
  });

  it("uses override command argv", () => {
    expect(detectInvocation("/work/train.jl", "julia +nightly")).toMatchObject({
      command: ["julia", "+nightly"],
      scriptArgument: "append"
    });
  });

  it("treats compiled runtime command overrides as standalone", () => {
    expect(detectInvocation("/work/model.cpp", "/work/.autotune/model")).toMatchObject({
      command: ["/work/.autotune/model"],
      scriptArgument: "none"
    });
  });

  it("detects explicit script slots in command overrides", () => {
    expect(detectInvocation("/work/train.py", "python3 -u /work/train.py")).toMatchObject({
      command: ["python3", "-u", "/work/train.py"],
      scriptArgument: "included"
    });
  });

  it("runs extensionless executables directly", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-detect-"));
    const script = path.join(dir, "train");
    await writeFile(script, "#!/usr/bin/env bash\n");
    await chmod(script, 0o755);
    expect(detectInvocation(script)).toMatchObject({
      language: "executable",
      command: [script]
    });
  });

  it("rejects unknown extensions without command override", () => {
    expect(() => detectInvocation("/work/train.xyz")).toThrow(/Use --command/);
  });
});
