import { mkdir, readdir, readFile, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import type { StudyResult, TrialResult } from "./results.js";
import type { Direction } from "./types.js";

const DEFAULT_WIDTH = 1100;
const DEFAULT_HEIGHT = 650;
const DEFAULT_MAX_TRIALS = 100;
const COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#d97706"];

export type ProgressXAxis = "trials" | "runtime";

export interface PlotProgressOptions {
  output: string;
  title?: string;
  maxTrials?: number;
  xAxis?: ProgressXAxis;
  maxRuntimeHours?: number;
  width?: number;
  height?: number;
  yMin?: number;
  yMax?: number;
  includeFailed?: boolean;
}

interface RoundFile {
  round: number;
  file: string;
}

interface ProgressPoint {
  x: number;
  y: number;
  improved: boolean;
}

interface ResetMarker {
  x: number;
  approximate: boolean;
}

interface ProgressReset {
  round: number;
  x: number;
}

interface VariantProgress {
  label: string;
  direction: Direction;
  points: ProgressPoint[];
  resets: ProgressReset[];
  totalTrials: number;
  totalRuntimeHours: number;
  best?: number;
}

interface VariantDirectory {
  order: number;
  label: string;
  directory: string;
}

export async function plotProgress(runDirectory: string, options: PlotProgressOptions): Promise<void> {
  const variants = await readVariantProgress(path.resolve(runDirectory), options);
  if (variants.length === 0) {
    throw new Error(`no result variants found under ${runDirectory}`);
  }
  const svg = renderProgressSvg(variants, {
    title: options.title ?? path.basename(path.resolve(runDirectory)),
    width: options.width ?? DEFAULT_WIDTH,
    height: options.height ?? DEFAULT_HEIGHT,
    maxTrials: options.maxTrials ?? DEFAULT_MAX_TRIALS,
    xAxis: options.xAxis ?? "trials",
    maxRuntimeHours: options.maxRuntimeHours,
    yMin: options.yMin,
    yMax: options.yMax
  });
  await mkdir(path.dirname(path.resolve(options.output)), { recursive: true });
  await writeFile(options.output, svg, "utf8");
}

export async function readVariantProgress(
  runDirectory: string,
  options: Pick<PlotProgressOptions, "maxTrials" | "includeFailed" | "xAxis"> = {}
): Promise<VariantProgress[]> {
  const maxTrials = options.maxTrials ?? DEFAULT_MAX_TRIALS;
  const xAxis = options.xAxis ?? "trials";
  const variants = await discoverVariantDirectories(runDirectory);
  return Promise.all(
    variants.map(async (variant) => {
      const rounds = await discoverRoundFiles(variant.directory);
      const direction = await readDirection(rounds);
      const { points, resets, totalTrials, totalRuntimeHours, best } = await buildProgress(rounds, {
        direction,
        maxTrials,
        xAxis,
        includeFailed: options.includeFailed ?? false
      });
      return { label: variant.label, direction, points, resets, totalTrials, totalRuntimeHours, best };
    })
  );
}

async function discoverVariantDirectories(runDirectory: string): Promise<VariantDirectory[]> {
  const entries = await readdir(runDirectory, { withFileTypes: true });
  const directories = entries
    .filter((entry) => entry.isDirectory())
    .map((entry) => ({
      directory: path.join(runDirectory, entry.name),
      ...labelForVariant(entry.name)
    }))
    .sort((left, right) => left.order - right.order || left.label.localeCompare(right.label));

  const withResults = [];
  for (const directory of directories) {
    if ((await discoverRoundFiles(directory.directory)).length > 0) {
      withResults.push(directory);
    }
  }
  return withResults;
}

function labelForVariant(name: string): { order: number; label: string } {
  if (/^01_/.test(name) || /base.*optuna|optuna.*baseline/.test(name)) {
    return { order: 1, label: "Base Optuna" };
  }
  if (/^02_/.test(name) || /no.*trial.*transfer/.test(name)) {
    return { order: 2, label: "Resets, no transfer" };
  }
  if (/^03_/.test(name) || /trial.*transfer/.test(name)) {
    return { order: 3, label: "Resets + transfer" };
  }
  if (/centaur/i.test(name)) {
    return { order: 4, label: "Centaur" };
  }
  return { order: Number.POSITIVE_INFINITY, label: name };
}

async function discoverRoundFiles(directory: string): Promise<RoundFile[]> {
  const entries = await readdir(directory);
  const rounds = entries
    .map((entry) => /^results\.round_(\d+)\.json$/.exec(entry))
    .filter((match): match is RegExpExecArray => match !== null)
    .map((match) => ({ round: Number(match[1]), file: path.join(directory, match[0]) }))
    .sort((left, right) => left.round - right.round);

  if (rounds.length > 0) {
    return rounds;
  }
  const results = path.join(directory, "results.json");
  try {
    const info = await stat(results);
    return info.isFile() ? [{ round: 0, file: results }] : [];
  } catch (error) {
    if (errorCode(error) === "ENOENT") {
      return [];
    }
    throw error;
  }
}

async function readDirection(rounds: RoundFile[]): Promise<Direction> {
  for (const round of rounds) {
    const result = await readStudyResult(round.file);
    if (result.direction === "maximize" || result.direction === "minimize") {
      return result.direction;
    }
  }
  throw new Error("could not infer optimization direction");
}

async function buildProgress(
  rounds: RoundFile[],
  options: { direction: Direction; maxTrials: number; xAxis: ProgressXAxis; includeFailed: boolean }
): Promise<{ points: ProgressPoint[]; resets: ProgressReset[]; totalTrials: number; totalRuntimeHours: number; best?: number }> {
  const points: ProgressPoint[] = [];
  const resets: ProgressReset[] = [];
  let totalTrials = 0;
  let totalRuntimeHours = 0;
  let best: number | undefined;

  for (let index = 0; index < rounds.length; index += 1) {
    const result = await readStudyResult(rounds[index].file);
    for (const trial of result.all_trials.sort((left, right) => left.number - right.number)) {
      if (isTransferTrial(trial)) {
        continue;
      }
      if (totalTrials >= options.maxTrials) {
        break;
      }
      totalTrials += 1;
      totalRuntimeHours += trialRuntimeHours(trial, options.xAxis === "runtime");
      const x = options.xAxis === "runtime" ? totalRuntimeHours : totalTrials;
      let improved = false;
      if (isScoredTrial(trial, options.includeFailed)) {
        const previous = best;
        best = bestScore(best, Number(trial.value), options.direction);
        improved = previous !== best;
      }
      if (best !== undefined) {
        points.push({ x, y: best, improved });
      }
    }
    if (index < rounds.length - 1 && totalTrials < options.maxTrials) {
      resets.push({
        round: rounds[index + 1].round,
        x: options.xAxis === "runtime" ? totalRuntimeHours : totalTrials
      });
    }
    if (totalTrials >= options.maxTrials) {
      break;
    }
  }
  return { points, resets, totalTrials, totalRuntimeHours, best };
}

function trialRuntimeHours(trial: TrialResult, required: boolean): number {
  const duration = trial.user_attrs?.autotune_duration_seconds;
  if (typeof duration === "number" && Number.isFinite(duration) && duration > 0) {
    return duration / 3600;
  }
  if (required) {
    throw new Error(`runtime x-axis requires autotune_duration_seconds for trial #${trial.number}`);
  }
  return 0;
}

function isScoredTrial(trial: TrialResult, includeFailed: boolean): boolean {
  return typeof trial.value === "number" && Number.isFinite(trial.value) && (includeFailed || !isFailedTrial(trial));
}

function isFailedTrial(trial: TrialResult): boolean {
  return typeof trial.user_attrs?.autotune_failure_reason === "string";
}

function isTransferTrial(trial: TrialResult): boolean {
  return trial.user_attrs?.autotune_transfer === true;
}

function bestScore(current: number | undefined, value: number, direction: Direction): number {
  if (current === undefined) {
    return value;
  }
  return direction === "maximize" ? Math.max(current, value) : Math.min(current, value);
}

async function readStudyResult(file: string): Promise<StudyResult> {
  return JSON.parse(await readFile(file, "utf8")) as StudyResult;
}

function renderProgressSvg(
  variants: VariantProgress[],
  options: {
    title: string;
    width: number;
    height: number;
    maxTrials: number;
    xAxis: ProgressXAxis;
    maxRuntimeHours?: number;
    yMin?: number;
    yMax?: number;
  }
): string {
  const margin = { top: 72, right: 40, bottom: 72, left: 86 };
  const chart = {
    left: margin.left,
    top: margin.top,
    width: options.width - margin.left - margin.right,
    height: options.height - margin.top - margin.bottom
  };
  const allY = variants.flatMap((variant) => variant.points.map((point) => point.y));
  if (allY.length === 0) {
    throw new Error("no scored trials available to plot");
  }
  const yDomain = resolveYDomain(allY, options);
  const direction = variants[0]?.direction ?? "maximize";
  const subtitle = direction === "maximize" ? "higher is better" : "lower is better";
  const xMax = resolveXMax(variants, options);
  const xToPx = (value: number) => chart.left + (value / xMax) * chart.width;
  const yToPx = (value: number) => chart.top + chart.height - ((value - yDomain.min) / (yDomain.max - yDomain.min)) * chart.height;
  const yTicks = ticks(yDomain.min, yDomain.max, 5);
  const runtimeAxis = options.xAxis === "runtime";
  const xTicks = runtimeAxis ? ticks(0, xMax, 5) : ticks(0, xMax, 5).map(Math.round);
  const resetMarkers = runtimeAxis
    ? mergeCorrespondingResetMarkers(variants)
    : [...new Set(variants.flatMap((variant) => variant.resets.map((reset) => reset.x)))]
        .sort((left, right) => left - right)
        .map((x) => ({ x, approximate: false }));
  const legend = legendBox(chart, variants.length, direction);

  const parts = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${options.width}" height="${options.height}" viewBox="0 0 ${options.width} ${options.height}" role="img" aria-label="${escapeXml(options.title)}">`,
    "<style>",
    "text{font-family:Inter,Arial,sans-serif;fill:#111827}",
    ".muted{fill:#6b7280}",
    ".grid{stroke:#e5e7eb;stroke-width:1}",
    ".axis{stroke:#111827;stroke-width:1.4}",
    ".reset{stroke:#6b7280;stroke-width:1.2;stroke-dasharray:5 5}",
    ".line{fill:none;stroke-width:3;stroke-linecap:round;stroke-linejoin:round}",
    ".star{stroke:#ffffff;stroke-width:1.4;stroke-linejoin:round}",
    ".legend-bg{fill:#ffffff;fill-opacity:0.88;stroke:#d1d5db;stroke-width:1}",
    "</style>",
    `<defs><clipPath id="chart-clip"><rect x="${chart.left}" y="${chart.top}" width="${chart.width}" height="${chart.height}"/></clipPath></defs>`,
    `<rect width="${options.width}" height="${options.height}" fill="#ffffff"/>`,
    `<text x="${chart.left}" y="34" font-size="24" font-weight="700">${escapeXml(options.title)}</text>`,
    `<text x="${chart.left}" y="58" font-size="14" class="muted">Best score so far (${subtitle}); ${runtimeAxis ? "failed trial runtime included" : "failed trials counted"} on x-axis but ignored for best-score updates.</text>`
  ];

  for (const tick of yTicks) {
    const y = yToPx(tick);
    parts.push(`<line x1="${chart.left}" y1="${formatSvgNumber(y)}" x2="${chart.left + chart.width}" y2="${formatSvgNumber(y)}" class="grid"/>`);
    parts.push(`<text x="${chart.left - 12}" y="${formatSvgNumber(y + 4)}" font-size="12" text-anchor="end" class="muted">${formatTick(tick)}</text>`);
  }
  for (const tick of xTicks) {
    const x = xToPx(tick);
    parts.push(`<line x1="${formatSvgNumber(x)}" y1="${chart.top}" x2="${formatSvgNumber(x)}" y2="${chart.top + chart.height}" class="grid"/>`);
    parts.push(`<text x="${formatSvgNumber(x)}" y="${chart.top + chart.height + 24}" font-size="12" text-anchor="middle" class="muted">${runtimeAxis ? formatRuntimeHours(tick) : Math.round(tick)}</text>`);
  }
  for (const reset of resetMarkers) {
    const x = xToPx(reset.x);
    const alignRight = x > chart.left + chart.width - 100;
    const labelX = x + (alignRight ? -6 : 6);
    parts.push(`<line x1="${formatSvgNumber(x)}" y1="${chart.top}" x2="${formatSvgNumber(x)}" y2="${chart.top + chart.height}" class="reset"/>`);
    const resetLabel = reset.approximate
      ? `resets ≈ ${formatRuntimeHours(reset.x)}h`
      : `reset @ ${runtimeAxis ? `${formatRuntimeHours(reset.x)}h` : reset.x}`;
    parts.push(`<text x="${formatSvgNumber(labelX)}" y="${chart.top + 18}" font-size="12" text-anchor="${alignRight ? "end" : "start"}" class="muted">${resetLabel}</text>`);
  }

  parts.push(`<line x1="${chart.left}" y1="${chart.top + chart.height}" x2="${chart.left + chart.width}" y2="${chart.top + chart.height}" class="axis"/>`);
  parts.push(`<line x1="${chart.left}" y1="${chart.top}" x2="${chart.left}" y2="${chart.top + chart.height}" class="axis"/>`);
  parts.push(`<text x="${chart.left + chart.width / 2}" y="${options.height - 24}" font-size="14" text-anchor="middle">${runtimeAxis ? "Cumulative trial runtime (hours)" : "Total evaluated trials"}</text>`);
  parts.push(`<text transform="translate(24 ${chart.top + chart.height / 2}) rotate(-90)" font-size="14" text-anchor="middle">Best score so far</text>`);
  parts.push(`<rect x="${formatSvgNumber(legend.x)}" y="${formatSvgNumber(legend.y)}" width="${legend.width}" height="${legend.height}" rx="6" class="legend-bg"/>`);

  variants.forEach((variant, index) => {
    const color = COLORS[index % COLORS.length];
    if (variant.points.length > 0) {
      parts.push(`<g clip-path="url(#chart-clip)">`);
      parts.push(`<path d="${linePath(variant.points, xToPx, yToPx)}" class="line" stroke="${color}"/>`);
      for (const point of variant.points.filter((candidate) => candidate.improved && yDomain.min <= candidate.y && candidate.y <= yDomain.max)) {
        parts.push(`<polygon points="${starPoints(xToPx(point.x), yToPx(point.y), 7, 3.1)}" class="star" fill="${color}"/>`);
      }
      parts.push(`</g>`);
      const last = variant.points[variant.points.length - 1];
      parts.push(`<circle cx="${formatSvgNumber(xToPx(last.x))}" cy="${formatSvgNumber(yToPx(last.y))}" r="4" fill="${color}"/>`);
      if (!runtimeAxis) {
        const endpoint = endpointLabel(xToPx(last.x), yToPx(last.y), chart);
        parts.push(`<text x="${formatSvgNumber(endpoint.x)}" y="${formatSvgNumber(endpoint.y)}" font-size="12" text-anchor="${endpoint.anchor}" fill="${color}">${formatTick(last.y)}</text>`);
      }
    }
    const legendY = legend.y + 20 + index * 28;
    const legendX = legend.x + 14;
    parts.push(`<line x1="${legendX}" y1="${legendY}" x2="${legendX + 24}" y2="${legendY}" class="line" stroke="${color}"/>`);
    parts.push(`<text x="${legendX + 34}" y="${legendY + 4}" font-size="13">${escapeXml(variant.label)}</text>`);
    const total = runtimeAxis
      ? `${formatRuntimeHours(variant.totalRuntimeHours)}h · ${variant.totalTrials} trials`
      : `${variant.totalTrials} trials`;
    parts.push(`<text x="${legendX + 34}" y="${legendY + 20}" font-size="11" class="muted">best ${variant.best === undefined ? "n/a" : formatTick(variant.best)} · ${total}</text>`);
  });

  parts.push("</svg>");
  return `${parts.join("\n")}\n`;
}

function mergeCorrespondingResetMarkers(variants: VariantProgress[]): ResetMarker[] {
  const groups = new Map<number, number[]>();
  for (const variant of variants) {
    for (const reset of variant.resets) {
      const group = groups.get(reset.round) ?? [];
      group.push(reset.x);
      groups.set(reset.round, group);
    }
  }
  return [...groups.values()]
    .map((group) => ({
      x: group.reduce((sum, value) => sum + value, 0) / group.length,
      approximate: group.length > 1
    }))
    .sort((left, right) => left.x - right.x);
}

function resolveXMax(
  variants: VariantProgress[],
  options: { xAxis: ProgressXAxis; maxTrials: number; maxRuntimeHours?: number }
): number {
  if (options.xAxis === "trials") {
    return options.maxTrials;
  }
  const observed = Math.max(...variants.map((variant) => variant.totalRuntimeHours));
  const maximum = options.maxRuntimeHours ?? observed;
  if (!Number.isFinite(maximum) || maximum <= 0) {
    throw new Error(`expected max runtime hours greater than zero, got ${String(maximum)}`);
  }
  if (maximum < observed) {
    throw new Error(`max runtime hours ${maximum} is less than observed runtime ${formatRuntimeHours(observed)}`);
  }
  return maximum;
}

function legendBox(
  chart: { left: number; top: number; width: number; height: number },
  variantCount: number,
  direction: Direction
): { x: number; y: number; width: number; height: number } {
  const width = 220;
  const height = 20 + variantCount * 28;
  const padding = 14;
  const topClearance = 42;
  return {
    x: chart.left + chart.width - width - padding,
    y: direction === "maximize" ? chart.top + chart.height - height - padding : chart.top + topClearance,
    width,
    height
  };
}

function endpointLabel(
  x: number,
  y: number,
  chart: { left: number; top: number; width: number; height: number }
): { x: number; y: number; anchor: "start" | "end" } {
  const chartRight = chart.left + chart.width;
  const labelY = Math.min(chart.top + chart.height - 8, Math.max(chart.top + 14, y - 8));
  if (x > chartRight - 80) {
    return { x: x - 8, y: labelY, anchor: "end" };
  }
  return { x: x + 8, y: labelY, anchor: "start" };
}

function resolveYDomain(allY: number[], options: { yMin?: number; yMax?: number }): { min: number; max: number } {
  const min = options.yMin ?? Math.min(...allY);
  const max = options.yMax ?? Math.max(...allY);
  if (!Number.isFinite(min) || !Number.isFinite(max) || min >= max) {
    throw new Error(`expected y-axis limits with yMin < yMax, got ${String(options.yMin)} and ${String(options.yMax)}`);
  }
  return options.yMin === undefined && options.yMax === undefined ? paddedDomain(min, max) : { min, max };
}

function paddedDomain(min: number, max: number): { min: number; max: number } {
  if (min === max) {
    const pad = Math.abs(min) > 1 ? Math.abs(min) * 0.05 : 0.05;
    return { min: min - pad, max: max + pad };
  }
  const pad = (max - min) * 0.08;
  return { min: min - pad, max: max + pad };
}

function starPoints(cx: number, cy: number, outerRadius: number, innerRadius: number): string {
  const points = [];
  for (let index = 0; index < 10; index += 1) {
    const radius = index % 2 === 0 ? outerRadius : innerRadius;
    const angle = -Math.PI / 2 + (index * Math.PI) / 5;
    points.push(`${formatSvgNumber(cx + radius * Math.cos(angle))},${formatSvgNumber(cy + radius * Math.sin(angle))}`);
  }
  return points.join(" ");
}

function ticks(min: number, max: number, count: number): number[] {
  if (count <= 1) {
    return [min];
  }
  return Array.from({ length: count }, (_, index) => min + ((max - min) * index) / (count - 1));
}

function linePath(points: ProgressPoint[], xToPx: (x: number) => number, yToPx: (y: number) => number): string {
  return points
    .map((point, index) => `${index === 0 ? "M" : "L"}${formatSvgNumber(xToPx(point.x))},${formatSvgNumber(yToPx(point.y))}`)
    .join(" ");
}

function formatTick(value: number): string {
  if (Math.abs(value) >= 100) {
    return value.toFixed(0);
  }
  if (Math.abs(value) >= 10) {
    return value.toFixed(2);
  }
  if (Math.abs(value) >= 1) {
    return value.toFixed(3);
  }
  return value.toPrecision(3);
}

function formatRuntimeHours(value: number): string {
  const absolute = Math.abs(value);
  if (absolute === 0) {
    return "0";
  }
  const decimals = absolute >= 1 ? 1 : absolute >= 0.1 ? 2 : absolute >= 0.01 ? 3 : absolute >= 0.001 ? 4 : 6;
  return value.toFixed(decimals).replace(/\.?0+$/, "");
}

function formatSvgNumber(value: number): string {
  return value.toFixed(2).replace(/\.?0+$/, "");
}

function escapeXml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function errorCode(error: unknown): string | undefined {
  if (typeof error !== "object" || error === null || !("code" in error)) {
    return undefined;
  }
  const code = (error as { code?: unknown }).code;
  return typeof code === "string" ? code : undefined;
}
