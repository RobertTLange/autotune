import path from "node:path";
import process from "node:process";

const BASE_ENVIRONMENT = words(
  "PATH HOME USER LOGNAME SHELL TMPDIR TMP TEMP LANG LC_ALL TERM NO_COLOR "
  + "FORCE_COLOR CI XDG_CONFIG_HOME XDG_CACHE_HOME XDG_DATA_HOME HEADLESS_CONFIG"
);
const NPM_ENVIRONMENT = words(
  "SystemRoot ComSpec PATHEXT USERPROFILE APPDATA LOCALAPPDATA PROGRAMDATA "
  + "HTTP_PROXY HTTPS_PROXY NO_PROXY ALL_PROXY http_proxy https_proxy no_proxy all_proxy "
  + "NODE_EXTRA_CA_CERTS NPM_CONFIG_REGISTRY NPM_CONFIG_PROXY NPM_CONFIG_HTTPS_PROXY "
  + "NPM_CONFIG_NOPROXY NPM_CONFIG_CAFILE NPM_CONFIG_CACHE NPM_CONFIG_STRICT_SSL "
  + "NPM_CONFIG_USERCONFIG NPM_CONFIG_GLOBALCONFIG "
  + "npm_config_registry npm_config_proxy npm_config_https_proxy npm_config_noproxy "
  + "npm_config_cafile npm_config_cache npm_config_strict_ssl "
  + "npm_config_userconfig npm_config_globalconfig"
);
const AGENT_ENVIRONMENT: Record<string, ReadonlySet<string>> = {
  codex: words("CODEX_API_KEY OPENAI_API_KEY OPENAI_BASE_URL CODEX_HOME"),
  cursor: words("CURSOR_API_KEY"),
  pi: words(
    "PI_CODING_AGENT_API_KEY PI_CODING_AGENT_MODEL PI_CODING_AGENT_MODELS "
    + "PI_CODING_AGENT_PROVIDER"
  )
};
const PROVIDER_ENVIRONMENT: Record<string, ReadonlySet<string>> = {
  anthropic: words("ANTHROPIC_API_KEY"),
  aws: words(
    "AWS_ACCESS_KEY_ID AWS_BEARER_TOKEN_BEDROCK AWS_CONTAINER_AUTHORIZATION_TOKEN "
    + "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE AWS_CONTAINER_CREDENTIALS_FULL_URI "
    + "AWS_CONFIG_FILE AWS_CONTAINER_CREDENTIALS_RELATIVE_URI AWS_DEFAULT_REGION AWS_PROFILE AWS_REGION "
    + "AWS_ROLE_ARN AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_SHARED_CREDENTIALS_FILE "
    + "AWS_WEB_IDENTITY_TOKEN_FILE"
  ),
  azure: words(
    "AZURE_OPENAI_API_KEY AZURE_OPENAI_API_VERSION AZURE_OPENAI_BASE_URL "
    + "AZURE_OPENAI_DEPLOYMENT_NAME_MAP AZURE_OPENAI_ENDPOINT "
    + "AZURE_OPENAI_RESOURCE_NAME AZURE_RESOURCE_NAME"
  ),
  google: words("GEMINI_API_KEY GOOGLE_API_KEY"),
  vertex: words(
    "GCLOUD_PROJECT GOOGLE_APPLICATION_CREDENTIALS GOOGLE_CLOUD_API_KEY "
    + "GOOGLE_CLOUD_LOCATION GOOGLE_CLOUD_PROJECT"
  ),
  openai: words("OPENAI_API_KEY OPENAI_BASE_URL"),
  openrouter: words("OPENROUTER_API_KEY")
};
const CLAUDE_COMMON_ENVIRONMENT = words(
  "CLAUDE_CONFIG_DIR ANTHROPIC_MODEL ANTHROPIC_SMALL_FAST_MODEL"
);
const CLAUDE_PROVIDER_ENVIRONMENT: Record<string, ReadonlySet<string>> = {
  anthropic: words(
    "ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN ANTHROPIC_BASE_URL ANTHROPIC_CUSTOM_HEADERS "
    + "CLAUDE_CODE_OAUTH_TOKEN"
  ),
  aws: words(
    "CLAUDE_CODE_USE_BEDROCK CLAUDE_CODE_SKIP_BEDROCK_AUTH ANTHROPIC_BEDROCK_BASE_URL "
    + "ANTHROPIC_SMALL_FAST_MODEL_AWS_REGION"
  ),
  vertex: words(
    "CLAUDE_CODE_USE_VERTEX CLAUDE_CODE_SKIP_VERTEX_AUTH ANTHROPIC_VERTEX_BASE_URL "
    + "ANTHROPIC_VERTEX_PROJECT_ID CLOUD_ML_REGION"
  )
};
const GEMINI_PROVIDER_ENVIRONMENT: Record<string, ReadonlySet<string>> = {
  google: words("GEMINI_API_KEY GOOGLE_API_KEY"),
  vertex: words(
    "GOOGLE_API_KEY GOOGLE_APPLICATION_CREDENTIALS GOOGLE_GENAI_USE_VERTEXAI GOOGLE_CLOUD_LOCATION"
  )
};
const GEMINI_COMMON_ENVIRONMENT = words("GOOGLE_CLOUD_PROJECT GOOGLE_CLOUD_PROJECT_ID");
const PROVIDER_FAMILY: Record<string, string> = {
  "amazon-bedrock": "aws",
  anthropic: "anthropic",
  google: "google",
  "google-vertex": "vertex",
  openai: "openai",
  "openai-codex": "openai",
  openrouter: "openrouter"
};
const AGENT_PROVIDER_FAMILY: Record<string, Record<string, string>> = {
  opencode: { azure: "azure" },
  pi: { aws: "aws", "azure-openai-responses": "azure" }
};
const EXPLICIT_ENVIRONMENT = "AUTOTUNE_HEADLESS_ENV";
const LEGACY_EXPLICIT_ENVIRONMENT = "AUTOTUNE_CENTAUR_HEADLESS_ENV";
const ENVIRONMENT_NAME = /^[A-Za-z_][A-Za-z0-9_]*$/;
const EXPLICIT_CREDENTIAL_OR_CONFIG = /(?:^|_)(?:API_KEY|TOKEN|SECRET|PASSWORD|CREDENTIALS(?:_FILE)?|CONFIG(?:_DIR|_FILE)?|PROFILE|PROJECT|REGION|LOCATION|ENDPOINT|BASE_URL|URL|DEPLOYMENT|RESOURCE|ACCOUNT(?:_ID)?|TENANT(?:_ID)?|CLIENT_ID|ORG(?:ANIZATION)?|MODEL|AUTH)$/;

export function npxHeadlessEnvironment(input: {
  agent: string;
  model?: string;
  env?: NodeJS.ProcessEnv;
}): NodeJS.ProcessEnv {
  const source = input.env ?? process.env;
  const explicitNames = explicitEnvironmentNames(source);
  const environment = npmInstallEnvironment(source);
  const names = new Set<string>();
  const normalizedAgent = input.agent.trim().toLowerCase();
  const provider = selectedProvider(normalizedAgent, input.model, source, explicitNames.length > 0);
  addNames(names, AGENT_ENVIRONMENT[normalizedAgent]);
  addNames(names, PROVIDER_ENVIRONMENT[provider]);
  if (normalizedAgent === "claude") {
    addNames(names, CLAUDE_COMMON_ENVIRONMENT);
    addNames(names, CLAUDE_PROVIDER_ENVIRONMENT[provider]);
  }
  if (normalizedAgent === "gemini") {
    addNames(names, GEMINI_COMMON_ENVIRONMENT);
    addNames(names, GEMINI_PROVIDER_ENVIRONMENT[provider]);
  }
  addNames(names, explicitNames);
  for (const name of names) {
    if (source[name] !== undefined) environment[name] = source[name];
  }
  return environment;
}

export function npmInstallEnvironment(source: NodeJS.ProcessEnv = process.env): NodeJS.ProcessEnv {
  const environment: NodeJS.ProcessEnv = {};
  for (const name of [...BASE_ENVIRONMENT, ...NPM_ENVIRONMENT]) {
    if (source[name] !== undefined) environment[name] = source[name];
  }
  environment.PATH = safeNpxPath(source.PATH);
  return environment;
}

function selectedProvider(
  normalizedAgent: string,
  model: string | undefined,
  env: NodeJS.ProcessEnv,
  hasExplicitEnvironment: boolean
): string {
  if (normalizedAgent === "claude") return selectedClaudeProvider(env);
  if (normalizedAgent === "gemini") return enabled(env.GOOGLE_GENAI_USE_VERTEXAI) ? "vertex" : "google";
  if (normalizedAgent !== "opencode" && normalizedAgent !== "pi") return "";
  const configuredModel = model || (normalizedAgent === "pi" ? env.PI_CODING_AGENT_MODEL : undefined);
  if (!configuredModel) {
    if (hasExplicitEnvironment) return "";
    throw new Error(
      `Headless agent ${normalizedAgent} requires a provider-qualified model or ${EXPLICIT_ENVIRONMENT}`
    );
  }
  const separator = configuredModel.indexOf("/");
  const provider = separator > 0 && separator < configuredModel.length - 1
    ? configuredModel.slice(0, separator).toLowerCase()
    : normalizedAgent === "pi"
    ? (env.PI_CODING_AGENT_PROVIDER ?? "").toLowerCase()
    : "";
  if (!provider && !hasExplicitEnvironment) {
    throw new Error(
      `Headless agent ${normalizedAgent} requires a provider-qualified model or ${EXPLICIT_ENVIRONMENT}`
    );
  }
  return AGENT_PROVIDER_FAMILY[normalizedAgent]?.[provider] ?? PROVIDER_FAMILY[provider] ?? "";
}

function selectedClaudeProvider(env: NodeJS.ProcessEnv): string {
  const bedrock = enabled(env.CLAUDE_CODE_USE_BEDROCK);
  const vertex = enabled(env.CLAUDE_CODE_USE_VERTEX);
  if (bedrock && vertex) {
    throw new Error("CLAUDE_CODE_USE_BEDROCK and CLAUDE_CODE_USE_VERTEX cannot both be enabled");
  }
  return bedrock ? "aws" : vertex ? "vertex" : "anthropic";
}

function explicitEnvironmentNames(env: NodeJS.ProcessEnv): string[] {
  const configured = env[EXPLICIT_ENVIRONMENT];
  const legacy = env[LEGACY_EXPLICIT_ENVIRONMENT];
  if (configured !== undefined && legacy !== undefined && configured !== legacy) {
    throw new Error(`${EXPLICIT_ENVIRONMENT} and ${LEGACY_EXPLICIT_ENVIRONMENT} conflict`);
  }
  const names = (configured ?? legacy ?? "").split(",").map((name) => name.trim()).filter(Boolean);
  if (
    names.length > 32
    || names.some((name) => !ENVIRONMENT_NAME.test(name) || !EXPLICIT_CREDENTIAL_OR_CONFIG.test(name))
  ) {
    throw new Error(
      `${EXPLICIT_ENVIRONMENT} must list at most 32 credential or config environment variable names`
    );
  }
  return names;
}

function safeNpxPath(configured: string | undefined): string {
  const entries = [path.dirname(process.execPath)];
  entries.push(...(configured ?? "").split(path.delimiter).filter((entry) => path.isAbsolute(entry)));
  if (process.platform !== "win32") entries.push("/usr/bin", "/bin");
  return [...new Set(entries.filter(Boolean))].join(path.delimiter);
}

function enabled(value: string | undefined): boolean {
  return value !== undefined && !["", "0", "false", "no"].includes(value.trim().toLowerCase());
}

function addNames(target: Set<string>, names: ReadonlySet<string> | readonly string[] | undefined): void {
  if (!names) return;
  for (const name of names) target.add(name);
}

function words(value: string): Set<string> {
  return new Set(value.split(/\s+/).filter(Boolean));
}
