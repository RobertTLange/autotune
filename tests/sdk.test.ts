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
  it("redacts sensitive equals-form arguments echoed in errors", () => {
    const secret = "postgres://user:secret@db";
    const message = `error: unknown option '--storage=${secret}'`;

    expect(redactSdkErrorMessage(message, ["node", "autotune", "analyze", `--storage=${secret}`]))
      .toBe("error: unknown option '--storage=[REDACTED]'");
    expect(redactSdkErrorMessage(`failed: ${secret}`, [
      "node", "autotune", "analyze", `--storage=${secret}`
    ])).toBe("failed: [REDACTED]");
  });

  it("redacts sensitive split-form argument values echoed in errors", () => {
    const secret = "top-secret-guidance";

    expect(redactSdkErrorMessage(`failed: ${secret}`, [
      "node", "autotune", "run", "--agent-guidance", secret
    ])).toBe("failed: [REDACTED]");
  });
});
