import { execFile } from "node:child_process";
import { chmod, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { promisify } from "node:util";
import { parse } from "yaml";

const execFileAsync = promisify(execFile);
type RegistryMode = "conflict" | "missing" | "propagates";

async function publishScript(): Promise<string> {
  const source = await readFile(".github/workflows/release.yml", "utf8");
  const workflow = parse(source) as {
    jobs: { publish: { steps: Array<{ name?: string; run?: string }> } };
  };
  const step = workflow.jobs.publish.steps.find(({ name }) => name === "Publish package");
  if (!step?.run) throw new Error("Publish package step not found");
  return step.run;
}

async function withFakeRegistry(
  mode: RegistryMode,
  assertion: (dir: string, run: () => Promise<unknown>) => Promise<void>
): Promise<void> {
  const dir = await mkdtemp(path.join(tmpdir(), "autotune-release-workflow-"));
  const fakeNpm = path.join(dir, "npm");
  const fakeSleep = path.join(dir, "sleep");
  const fakeTimeout = path.join(dir, "timeout");

  await writeFile(fakeNpm, `#!/usr/bin/env bash
set -euo pipefail
state="$FAKE_NPM_STATE"
case "$1" in
  publish)
    touch "$state/published"
    ;;
  view)
    if [[ ! -f "$state/published" ]]; then
      if [[ "$FAKE_NPM_MODE" = conflict ]]; then printf '%s\n' 'sha512-conflict'; else exit 1; fi
    elif [[ "$FAKE_NPM_MODE" = missing ]]; then
      attempts=0
      [[ ! -f "$state/attempts" ]] || attempts="$(cat "$state/attempts")"
      printf '%s' "$((attempts + 1))" > "$state/attempts"
      exit 1
    else
      attempts=0
      [[ ! -f "$state/attempts" ]] || attempts="$(cat "$state/attempts")"
      attempts=$((attempts + 1))
      printf '%s' "$attempts" > "$state/attempts"
      if (( attempts < 3 )); then exit 1; fi
      printf '{"version":"%s","dist.integrity":"%s","dist.attestations.provenance.predicateType":"https://slsa.dev/provenance/v1"}\n' \
        "$PACKAGE_VERSION" "$PACKAGE_INTEGRITY"
    fi
    ;;
  *)
    exit 2
    ;;
esac
`);
  await writeFile(fakeSleep, "#!/usr/bin/env bash\nexit 0\n");
  await writeFile(fakeTimeout, "#!/usr/bin/env bash\nshift\nexec \"$@\"\n");
  await Promise.all([fakeNpm, fakeSleep, fakeTimeout].map(file => chmod(file, 0o755)));

  const publishScriptSource = await publishScript();
  const run = () => execFileAsync("bash", ["-c", publishScriptSource], {
    env: {
      ...process.env,
      PATH: `${dir}:${process.env.PATH ?? ""}`,
      FAKE_NPM_MODE: mode,
      FAKE_NPM_STATE: dir,
      PACKAGE_NAME: "@roberttlange/autotune",
      PACKAGE_VERSION: "0.1.0",
      PACKAGE_TARBALL: path.join(dir, "package.tgz"),
      PACKAGE_INTEGRITY: "sha512-test"
    }
  });

  try {
    await assertion(dir, run);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

describe("release workflow", () => {
  it("retries npm metadata verification while a new release propagates", async () => {
    await withFakeRegistry("propagates", async (dir, run) => {
      await run();
      expect(await readFile(path.join(dir, "attempts"), "utf8")).toBe("3");
    });
  });

  it("stops after the bounded number of unavailable metadata responses", async () => {
    await withFakeRegistry("missing", async (dir, run) => {
      await expect(run()).rejects.toThrow("Published package metadata did not propagate before timeout");
      expect(await readFile(path.join(dir, "attempts"), "utf8")).toBe("12");
    });
  }, 15_000);

  it("rejects a pre-existing release with conflicting integrity", async () => {
    await withFakeRegistry("conflict", async (dir, run) => {
      await expect(run()).rejects.toThrow("Registry version exists with different integrity");
      await expect(readFile(path.join(dir, "published"))).rejects.toThrow();
    });
  });
});
