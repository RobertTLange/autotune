export const SDK_PROTOCOL_VERSION = 1;

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
