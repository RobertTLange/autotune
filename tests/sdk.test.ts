import { SDK_PROTOCOL_VERSION, renderSdkError, renderSdkResult } from "../src/sdk.js";

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
