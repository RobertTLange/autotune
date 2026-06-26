import { extractHeadlessJson } from "../src/headless.js";

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
});
