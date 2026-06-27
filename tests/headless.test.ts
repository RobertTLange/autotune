import { FALLBACK_HEADLESS_PACKAGE, extractHeadlessJson, extractHeadlessObject } from "../src/headless.js";

describe("headless fallback package", () => {
  it("uses a pinned npx package spec", () => {
    expect(FALLBACK_HEADLESS_PACKAGE).toMatch(/^@roberttlange\/headless@\d+\.\d+\.\d+$/);
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
