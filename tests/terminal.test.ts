import { formatStatus, formatTable, stripAnsi, styles, timestamp } from "../src/terminal.js";

describe("terminal formatting", () => {
  it("aligns tables while ignoring ANSI escape codes", () => {
    const table = formatTable({
      headers: ["Name", "Value"],
      rows: [
        [styles.green("lr"), "0.001"],
        ["dropout", "0.2"]
      ]
    }).join("\n");

    expect(stripAnsi(table)).toContain("Name     Value");
    expect(stripAnsi(table)).toContain("-------  -----");
    expect(stripAnsi(table)).toContain("lr       0.001");
    expect(stripAnsi(table)).toContain("dropout  0.2");
  });

  it("honors color environment controls", () => {
    const previousForceColor = process.env.FORCE_COLOR;
    const previousNoColor = process.env.NO_COLOR;
    try {
      process.env.FORCE_COLOR = "1";
      delete process.env.NO_COLOR;
      expect(styles.green("ok")).toContain("\u001b[");

      process.env.NO_COLOR = "1";
      expect(styles.green("ok")).toBe("ok");
    } finally {
      restoreEnv("FORCE_COLOR", previousForceColor);
      restoreEnv("NO_COLOR", previousNoColor);
    }
  });

  it("formats status messages with timestamps", () => {
    expect(timestamp(new Date("2026-06-26T12:34:56Z"))).toMatch(/^\d{2}:34:56$/);
    expect(stripAnsi(formatStatus("Running trials", "info", new Date("2026-06-26T12:34:56Z")))).toMatch(
      /^\[\d{2}:34:56\] Running trials$/
    );
  });
});

function restoreEnv(name: string, value: string | undefined): void {
  if (value === undefined) {
    delete process.env[name];
  } else {
    process.env[name] = value;
  }
}
