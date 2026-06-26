type Level = "info" | "success" | "warning" | "error";
type Align = "left" | "right";

const ANSI_PATTERN = /\u001b\[[0-9;]*m/g;

const codes = {
  bold: 1,
  dim: 2,
  red: 31,
  green: 32,
  yellow: 33,
  cyan: 36
} as const;

export const styles = {
  bold: (text: string) => color(text, codes.bold),
  dim: (text: string) => color(text, codes.dim),
  red: (text: string) => color(text, codes.red),
  green: (text: string) => color(text, codes.green),
  yellow: (text: string) => color(text, codes.yellow),
  cyan: (text: string) => color(text, codes.cyan)
};

export function stripAnsi(text: string): string {
  return text.replace(ANSI_PATTERN, "");
}

export function visibleWidth(text: string): number {
  return stripAnsi(text).length;
}

export function formatTable(input: {
  headers: string[];
  rows: ReadonlyArray<ReadonlyArray<string>>;
  align?: Align[];
}): string[] {
  const rows = [input.headers, ...input.rows];
  const widths = input.headers.map((_, index) =>
    Math.max(...rows.map((row) => visibleWidth(String(row[index] ?? ""))))
  );
  const separator = widths.map((width) => "-".repeat(width)).join("  ");
  return [
    formatRow(input.headers, widths, input.align),
    separator,
    ...input.rows.map((row) => formatRow(row, widths, input.align))
  ];
}

export function timestamp(date = new Date()): string {
  return date.toTimeString().slice(0, 8);
}

export function formatStatus(message: string, level: Level = "info", date = new Date()): string {
  const prefix = color(`[${timestamp(date)}]`, codes.dim, process.stderr);
  const styled =
    level === "success"
      ? color(message, codes.green, process.stderr)
      : level === "warning"
        ? color(message, codes.yellow, process.stderr)
        : level === "error"
          ? color(message, codes.red, process.stderr)
          : message;
  return `${prefix} ${styled}`;
}

export function writeStatus(message: string, level: Level = "info"): void {
  console.error(formatStatus(message, level));
}

export function color(text: string, code: number, stream: NodeJS.WriteStream = process.stdout): string {
  if (!colorEnabled(stream)) {
    return text;
  }
  return `\u001b[${code}m${text}\u001b[0m`;
}

export function formatNumber(value: number): string {
  if (!Number.isFinite(value)) {
    return String(value);
  }
  if (Number.isInteger(value)) {
    return String(value);
  }
  return Number(value.toPrecision(12)).toString();
}

export function wrapText(text: string, width = 100): string[] {
  const words = text.split(/\s+/).filter(Boolean);
  const lines: string[] = [];
  let line = "";
  for (const word of words) {
    const candidate = line ? `${line} ${word}` : word;
    if (visibleWidth(candidate) <= width) {
      line = candidate;
      continue;
    }
    if (line) {
      lines.push(line);
    }
    line = word;
  }
  if (line) {
    lines.push(line);
  }
  return lines.length > 0 ? lines : [""];
}

function formatRow(row: ReadonlyArray<string>, widths: number[], align: Align[] = []): string {
  return row
    .map((value, index) => pad(String(value ?? ""), widths[index] ?? 0, align[index] ?? "left"))
    .join("  ");
}

function pad(value: string, width: number, align: Align): string {
  const padding = " ".repeat(Math.max(0, width - visibleWidth(value)));
  return align === "right" ? `${padding}${value}` : `${value}${padding}`;
}

function colorEnabled(stream: NodeJS.WriteStream): boolean {
  if (process.env.NO_COLOR !== undefined) {
    return false;
  }
  if (process.env.FORCE_COLOR !== undefined && process.env.FORCE_COLOR !== "0") {
    return true;
  }
  return Boolean(stream.isTTY);
}
