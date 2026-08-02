import {
  redactSdkErrorMessage,
  SDK_PROTOCOL_VERSION,
  renderSdkError,
  renderSdkResult
} from "../src/sdk.js";

describe("SDK protocol", () => {
  it("renders versioned result envelopes", () => {
    expect(JSON.parse(renderSdkResult("results", { n_trials: 2 }))).toEqual({
      protocolVersion: SDK_PROTOCOL_VERSION,
      type: "result",
      command: "results",
      exitCode: 0,
      data: { n_trials: 2 }
    });
  });

  it("renders versioned error envelopes", () => {
    expect(JSON.parse(renderSdkError("bad input", 2, "run"))).toEqual({
      protocolVersion: SDK_PROTOCOL_VERSION,
      type: "error",
      command: "run",
      exitCode: 2,
      error: { message: "bad input" }
    });
  });
});

describe("SDK error redaction", () => {
  it("returns a safe error when sensitive equals-form arguments are present", () => {
    const secret = "postgres://user:secret@db";
    const message = `error: unknown option '--storage=${secret}'`;

    expect(redactSdkErrorMessage(message, ["node", "autotune", "analyze", `--storage=${secret}`]))
      .toBe("autotune SDK command failed");
    expect(redactSdkErrorMessage(`failed: ${secret}`, [
      "node", "autotune", "analyze", `--storage=${secret}`
    ])).toBe("autotune SDK command failed");
  });

  it("returns a safe error when sensitive split-form arguments are present", () => {
    const secret = "top-secret-guidance";

    expect(redactSdkErrorMessage(`failed: ${secret}`, [
      "node", "autotune", "run", "--agent-guidance", secret
    ])).toBe("autotune SDK command failed");
  });

  it("does not expose a secret echoed by a sensitive build command", () => {
    const secret = "review-secret";

    expect(redactSdkErrorMessage(`build failed: ${secret}`, [
      "node", "autotune", "run", "--build-command", `sh -c 'printf ${secret} >&2; exit 1'`
    ])).toBe("autotune SDK command failed");
  });
});
