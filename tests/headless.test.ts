import { access, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { npxHeadlessEnvironment } from "../src/headless-environment.js";
import { FALLBACK_HEADLESS_PACKAGE, extractHeadlessJson, extractHeadlessObject, runHeadless } from "../src/headless.js";

describe("headless fallback package", () => {
  it("uses a pinned npx package spec", () => {
    expect(FALLBACK_HEADLESS_PACKAGE).toMatch(/^@roberttlange\/headless@\d+\.\d+\.\d+$/);
  });

  it("rejects an empty explicit executable without falling back", async () => {
    const previous = process.env.AUTOTUNE_HEADLESS_BIN;
    process.env.AUTOTUNE_HEADLESS_BIN = "";
    try {
      await expect(runHeadless([], { cwd: process.cwd() })).rejects.toThrow(/must not be empty/i);
    } finally {
      if (previous === undefined) delete process.env.AUTOTUNE_HEADLESS_BIN;
      else process.env.AUTOTUNE_HEADLESS_BIN = previous;
    }
  });

  it("validates provider selection before resolving npx", async () => {
    const previousPath = process.env.PATH;
    let resolutionCalled = false;
    process.env.PATH = path.resolve("/not-on-path");
    try {
      await expect(runHeadless(["opencode"], {
        cwd: process.cwd(),
        resolveNpm: async () => {
          resolutionCalled = true;
          return { command: process.execPath, args: [] };
        }
      })).rejects.toThrow(/provider-qualified model or AUTOTUNE_HEADLESS_ENV/i);
      expect(resolutionCalled).toBe(false);
    } finally {
      if (previousPath === undefined) delete process.env.PATH;
      else process.env.PATH = previousPath;
    }
  });

  it("runs the locked fallback in a temporary package boundary", async () => {
    const previousPath = process.env.PATH;
    const previousOpenAiKey = process.env.OPENAI_API_KEY;
    const project = await mkdtemp(path.join(tmpdir(), "autotune-headless-project-"));
    await writeFile(path.join(project, "package.json"), JSON.stringify({
      dependencies: { "@roberttlange/headless": "file:./malicious-headless" }
    }), "utf8");
    process.env.PATH = path.resolve("/not-on-path");
    process.env.OPENAI_API_KEY = "selected-key";

    try {
      const installedCli = [
        "const fs=require('fs');",
        "const install=JSON.parse(fs.readFileSync('install.json','utf8'));",
        "console.log(JSON.stringify({cwd:process.cwd(),pkg:JSON.parse(fs.readFileSync('package.json','utf8')),args:process.argv.slice(2),install,executionKey:process.env.OPENAI_API_KEY}));"
      ].join("");
      const installer = [
        "const fs=require('fs');",
        "fs.mkdirSync('node_modules/@roberttlange/headless/dist',{recursive:true});",
        `fs.writeFileSync('node_modules/@roberttlange/headless/dist/cli.js',${JSON.stringify(installedCli)});`,
        "fs.writeFileSync('install.json',JSON.stringify({args:process.argv.slice(1),env:process.env}));"
      ].join("");
      const output = await runHeadless(["codex"], {
        cwd: project,
        resolveNpm: async () => ({
          command: process.execPath,
          args: ["-e", installer, "--"]
        })
      });
      const invocation = JSON.parse(output) as {
        cwd: string;
        pkg: Record<string, unknown>;
        args: string[];
        install: { args: string[]; env: NodeJS.ProcessEnv };
        executionKey?: string;
      };

      expect(invocation.cwd).not.toBe(project);
      expect(invocation.pkg).toMatchObject({
        private: true,
        dependencies: { "@roberttlange/headless": "0.4.0" }
      });
      expect(invocation.args).toEqual(["codex"]);
      expect(invocation.install.args).toEqual(expect.arrayContaining([
        "ci",
        "--ignore-scripts",
        "--no-audit",
        "--no-fund"
      ]));
      expect(invocation.install.env.OPENAI_API_KEY).toBeUndefined();
      expect(invocation.executionKey).toBe("selected-key");
      await expect(access(invocation.cwd)).rejects.toThrow();
      expect(JSON.parse(await readFile(path.join(project, "package.json"), "utf8")))
        .toHaveProperty("dependencies.@roberttlange/headless");
    } finally {
      if (previousPath === undefined) delete process.env.PATH;
      else process.env.PATH = previousPath;
      if (previousOpenAiKey === undefined) delete process.env.OPENAI_API_KEY;
      else process.env.OPENAI_API_KEY = previousOpenAiKey;
      await rm(project, { recursive: true, force: true });
    }
  });

  it("limits the fallback to selected-agent and npm environment variables", async () => {
    const customPath = path.resolve("/custom/bin");
    const environment = npxHeadlessEnvironment({
      agent: "codex",
      model: "gpt-5.5",
      env: {
        PATH: [customPath, ".", "relative-bin"].join(path.delimiter),
        OPENAI_API_KEY: "selected-key",
        AWS_SECRET_ACCESS_KEY: "unrelated-aws-secret",
        AUTOTUNE_TEST_SECRET: "unrelated-secret",
        AUTOTUNE_HEADLESS_ENV: "CUSTOM_PROVIDER_TOKEN",
        CUSTOM_PROVIDER_TOKEN: "explicit-key",
        HTTP_PROXY: "http://proxy.example",
        NPM_CONFIG_USERCONFIG: "/config/npmrc",
        XDG_DATA_HOME: "/data/xdg"
      }
    });

    expect(environment.OPENAI_API_KEY).toBe("selected-key");
    expect(environment.HTTP_PROXY).toBe("http://proxy.example");
    expect(environment.NPM_CONFIG_USERCONFIG).toBe("/config/npmrc");
    expect(environment.XDG_DATA_HOME).toBe("/data/xdg");
    expect(environment.CUSTOM_PROVIDER_TOKEN).toBe("explicit-key");
    expect(environment.AWS_SECRET_ACCESS_KEY).toBeUndefined();
    expect(environment.AUTOTUNE_TEST_SECRET).toBeUndefined();
    expect(environment.PATH?.split(path.delimiter)).toContain(customPath);
    expect(environment.PATH?.split(path.delimiter)).not.toEqual(expect.arrayContaining([".", "relative-bin"]));
  });

  it("selects only the enabled Claude cloud provider credentials", () => {
    const direct = npxHeadlessEnvironment({
      agent: "claude",
      env: { ANTHROPIC_CUSTOM_HEADERS: "X-Gateway-Key: secret" }
    });
    const bedrock = npxHeadlessEnvironment({
      agent: "claude",
      env: {
        CLAUDE_CODE_USE_BEDROCK: "1",
        CLAUDE_CONFIG_DIR: "/config/claude",
        ANTHROPIC_MODEL: "bedrock-model",
        AWS_CONFIG_FILE: "/config/aws",
        AWS_SECRET_ACCESS_KEY: "aws-secret",
        AWS_SHARED_CREDENTIALS_FILE: "/config/aws-credentials",
        GOOGLE_CLOUD_API_KEY: "google-secret",
        ANTHROPIC_API_KEY: "anthropic-secret",
        ANTHROPIC_CUSTOM_HEADERS: "X-Gateway-Key: unrelated"
      }
    });
    const vertex = npxHeadlessEnvironment({
      agent: "claude",
      env: {
        CLAUDE_CODE_USE_VERTEX: "1",
        CLAUDE_CONFIG_DIR: "/config/claude",
        ANTHROPIC_MODEL: "vertex-model",
        GOOGLE_CLOUD_API_KEY: "google-secret",
        AWS_SECRET_ACCESS_KEY: "aws-secret",
        ANTHROPIC_API_KEY: "anthropic-secret",
        ANTHROPIC_CUSTOM_HEADERS: "X-Gateway-Key: unrelated"
      }
    });

    expect(bedrock.CLAUDE_CODE_USE_BEDROCK).toBe("1");
    expect(direct.ANTHROPIC_CUSTOM_HEADERS).toBe("X-Gateway-Key: secret");
    expect(bedrock.CLAUDE_CONFIG_DIR).toBe("/config/claude");
    expect(bedrock.ANTHROPIC_MODEL).toBe("bedrock-model");
    expect(bedrock.AWS_SECRET_ACCESS_KEY).toBe("aws-secret");
    expect(bedrock.AWS_CONFIG_FILE).toBe("/config/aws");
    expect(bedrock.AWS_SHARED_CREDENTIALS_FILE).toBe("/config/aws-credentials");
    expect(bedrock.GOOGLE_CLOUD_API_KEY).toBeUndefined();
    expect(bedrock.ANTHROPIC_API_KEY).toBeUndefined();
    expect(bedrock.ANTHROPIC_CUSTOM_HEADERS).toBeUndefined();
    expect(vertex.CLAUDE_CODE_USE_VERTEX).toBe("1");
    expect(vertex.CLAUDE_CONFIG_DIR).toBe("/config/claude");
    expect(vertex.ANTHROPIC_MODEL).toBe("vertex-model");
    expect(vertex.GOOGLE_CLOUD_API_KEY).toBe("google-secret");
    expect(vertex.AWS_SECRET_ACCESS_KEY).toBeUndefined();
    expect(vertex.ANTHROPIC_API_KEY).toBeUndefined();
    expect(vertex.ANTHROPIC_CUSTOM_HEADERS).toBeUndefined();
  });

  it("keeps Gemini Vertex configuration without exposing unrelated credentials", () => {
    const environment = npxHeadlessEnvironment({
      agent: "gemini",
      env: {
        GOOGLE_GENAI_USE_VERTEXAI: "true",
        GOOGLE_CLOUD_PROJECT: "project-id",
        GOOGLE_CLOUD_LOCATION: "europe-west1",
        GOOGLE_APPLICATION_CREDENTIALS: "/credentials/google.json",
        GOOGLE_API_KEY: "vertex-key",
        GEMINI_API_KEY: "unrelated-gemini-key",
        AWS_SECRET_ACCESS_KEY: "unrelated-aws-secret"
      }
    });

    expect(environment.GOOGLE_GENAI_USE_VERTEXAI).toBe("true");
    expect(environment.GOOGLE_CLOUD_PROJECT).toBe("project-id");
    expect(environment.GOOGLE_CLOUD_LOCATION).toBe("europe-west1");
    expect(environment.GOOGLE_APPLICATION_CREDENTIALS).toBe("/credentials/google.json");
    expect(environment.GOOGLE_API_KEY).toBe("vertex-key");
    expect(environment.GEMINI_API_KEY).toBeUndefined();
    expect(environment.AWS_SECRET_ACCESS_KEY).toBeUndefined();

    const direct = npxHeadlessEnvironment({
      agent: "gemini",
      env: {
        GEMINI_API_KEY: "gemini-key",
        GOOGLE_APPLICATION_CREDENTIALS: "/credentials/unrelated.json",
        GOOGLE_CLOUD_PROJECT: "organization-project"
      }
    });
    expect(direct.GEMINI_API_KEY).toBe("gemini-key");
    expect(direct.GOOGLE_APPLICATION_CREDENTIALS).toBeUndefined();
    expect(direct.GOOGLE_CLOUD_PROJECT).toBe("organization-project");
  });

  it("rejects runtime-injection variables in the explicit allowlist", () => {
    expect(() => npxHeadlessEnvironment({
      agent: "codex",
      env: { AUTOTUNE_HEADLESS_ENV: "NODE_OPTIONS", NODE_OPTIONS: "--require ./payload.cjs" }
    })).toThrow(/credential or config environment variable names/i);
  });

  it("preserves the Centaur-specific explicit-environment alias", () => {
    const environment = npxHeadlessEnvironment({
      agent: "opencode",
      env: {
        AUTOTUNE_CENTAUR_HEADLESS_ENV: "CUSTOM_PROVIDER_TOKEN",
        CUSTOM_PROVIDER_TOKEN: "legacy-key"
      }
    });

    expect(environment.CUSTOM_PROVIDER_TOKEN).toBe("legacy-key");
  });

  it("requires explicit provider selection for multiprovider agents", () => {
    expect(() => npxHeadlessEnvironment({
      agent: "opencode",
      env: { OPENAI_API_KEY: "ambiguous-key" }
    })).toThrow(/provider-qualified model or AUTOTUNE_HEADLESS_ENV/i);

    const selected = npxHeadlessEnvironment({
      agent: "opencode",
      model: "openai/gpt-5",
      env: { OPENAI_API_KEY: "selected-key", AWS_SECRET_ACCESS_KEY: "unrelated-secret" }
    });
    expect(selected.OPENAI_API_KEY).toBe("selected-key");
    expect(selected.AWS_SECRET_ACCESS_KEY).toBeUndefined();
  });
});

describe("extractHeadlessJson", () => {
  it("extracts plain JSON from agent output", () => {
    const result = extractHeadlessJson(`noise\n{"parameters":[],"has_arg_parsing":true,"needs_wrapper":false,"direction":"maximize"}`);
    expect(result.direction).toBe("maximize");
  });

  it("extracts JSON from JSONL headless traces", () => {
    const result = extractHeadlessJson(
      [
        JSON.stringify({ type: "message", content: "starting" }),
        JSON.stringify({
          type: "assistant",
          message: {
            content: [
              {
                type: "text",
                text: '{"parameters":[],"has_arg_parsing":true,"needs_wrapper":false,"direction":"minimize"}'
              }
            ]
          }
        })
      ].join("\n")
    );
    expect(result.direction).toBe("minimize");
  });

  it("extracts JSON from headless item text traces", () => {
    const result = extractHeadlessJson(
      JSON.stringify({
        type: "item.completed",
        item: {
          type: "agent_message",
          text: '{"parameters":[],"has_arg_parsing":true,"needs_wrapper":false,"direction":"maximize"}'
        }
      })
    );
    expect(result.direction).toBe("maximize");
  });

  it("prefers final JSONL text over earlier logged objects", () => {
    const result = extractHeadlessJson(
      [
        JSON.stringify({ type: "tool", output: '{"parameters":[],"has_arg_parsing":true,"needs_wrapper":false,"direction":"minimize"}' }),
        JSON.stringify({
          type: "item.completed",
          item: {
            type: "agent_message",
            text: '{"parameters":[],"has_arg_parsing":true,"needs_wrapper":false,"direction":"maximize"}'
          }
        })
      ].join("\n")
    );

    expect(result.direction).toBe("maximize");
  });
});

describe("extractHeadlessObject", () => {
  it("extracts generic JSON objects from headless traces", () => {
    expect(
      extractHeadlessObject(
        JSON.stringify({
          type: "item.completed",
          item: { type: "agent_message", text: '{"code":"print(1)"}' }
        })
      )
    ).toEqual({ code: "print(1)" });
  });

  it("prefers final code objects over earlier logged objects", () => {
    expect(
      extractHeadlessObject(
        [
          JSON.stringify({ type: "tool", output: '{"code":"print(0)"}' }),
          JSON.stringify({ type: "item.completed", item: { type: "agent_message", text: '{"code":"print(1)"}' } })
        ].join("\n")
      )
    ).toEqual({ code: "print(1)" });
  });
});
