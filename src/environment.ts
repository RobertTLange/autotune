import process from "node:process";

export function environmentValue(
  env: NodeJS.ProcessEnv,
  name: string,
  caseInsensitive = process.platform === "win32"
): string | undefined {
  if (env[name] !== undefined || !caseInsensitive) return env[name];
  const key = Object.keys(env).find((candidate) => candidate.toUpperCase() === name.toUpperCase());
  return key ? env[key] : undefined;
}
