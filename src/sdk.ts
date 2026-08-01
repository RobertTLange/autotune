export const SDK_PROTOCOL_VERSION = 1;

const SENSITIVE_FLAGS = new Set([
  "--agent-guidance",
  "--build-command",
  "--command",
  "--storage"
]);
const SENSITIVE_FLAG_SUFFIXES = [
  "-access-key",
  "-api-key",
  "-auth-token",
  "-authorization",
  "-client-secret",
  "-credential",
  "-credentials",
  "-passphrase",
  "-password",
  "-private-key",
  "-secret",
  "-secret-key",
  "-token"
];

interface SdkResultEnvelope {
  protocolVersion: typeof SDK_PROTOCOL_VERSION;
  type: "result";
  command: string;
  exitCode: number;
  data: unknown;
}

interface SdkErrorEnvelope {
  protocolVersion: typeof SDK_PROTOCOL_VERSION;
  type: "error";
  command: string;
  exitCode: number;
  error: { message: string };
}

export function renderSdkResult(command: string, data: unknown, exitCode = 0): string {
  const envelope: SdkResultEnvelope = {
    protocolVersion: SDK_PROTOCOL_VERSION,
    type: "result",
    command,
    exitCode,
    data
  };
  return JSON.stringify(envelope);
}

export function renderSdkError(message: string, exitCode: number, command = "cli"): string {
  const envelope: SdkErrorEnvelope = {
    protocolVersion: SDK_PROTOCOL_VERSION,
    type: "error",
    command,
    exitCode,
    error: { message }
  };
  return JSON.stringify(envelope);
}

export function redactSdkErrorMessage(message: string, argv: string[]): string {
  let redacted = message;
  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index] ?? "";
    const separator = value.indexOf("=");
    const flag = separator >= 0 ? value.slice(0, separator) : value;
    if (!isSensitiveFlag(flag)) continue;
    if (separator >= 0) {
      redacted = replaceAll(redacted, value, `${flag}=[REDACTED]`);
      redacted = replaceAll(redacted, value.slice(separator + 1), "[REDACTED]");
      continue;
    }
    const secret = argv[index + 1];
    if (secret !== undefined) redacted = replaceAll(redacted, secret, "[REDACTED]");
  }
  return redacted;
}

function isSensitiveFlag(flag: string): boolean {
  const normalized = flag.toLowerCase().replaceAll("_", "-");
  return normalized.startsWith("--") && (
    SENSITIVE_FLAGS.has(normalized) ||
    SENSITIVE_FLAG_SUFFIXES.some((suffix) => normalized.endsWith(suffix))
  );
}

function replaceAll(value: string, search: string, replacement: string): string {
  return search.length === 0 ? value : value.split(search).join(replacement);
}
