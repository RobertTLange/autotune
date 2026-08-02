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
const SAFE_SDK_ERROR_MESSAGE = "autotune SDK command failed";

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
  return containsSensitiveArgument(argv) ? SAFE_SDK_ERROR_MESSAGE : message;
}

function containsSensitiveArgument(argv: string[]): boolean {
  return argv.some((value) => isSensitiveFlag(value.split("=", 1)[0] ?? ""));
}

function isSensitiveFlag(flag: string): boolean {
  const normalized = flag.toLowerCase().replaceAll("_", "-");
  return normalized.startsWith("--") && (
    SENSITIVE_FLAGS.has(normalized) ||
    SENSITIVE_FLAG_SUFFIXES.some((suffix) => normalized.endsWith(suffix))
  );
}
