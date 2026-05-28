import {
  readCodeLabProgress,
  writeCodeLabProgress,
  normalizeCodeLabProgress,
} from './codeLabProgress.js';

export const GITHUB_SYNC_SETTINGS_KEY = 'ml-animations:github-sync-settings:v1';
export const GITHUB_SYNC_EVENT = 'ml-animations:github-sync-settings-updated';
export const GITHUB_SYNC_TARGETS = {
  repo: 'repo',
  gist: 'gist',
};

export const DEFAULT_GITHUB_SYNC_SETTINGS = {
  enabled: false,
  autoSync: false,
  target: GITHUB_SYNC_TARGETS.repo,
  brokerUrl: '',
  owner: '',
  repo: '',
  branch: 'main',
  path: '.ml-animations/progress.v1.json',
  gistId: '',
  gistFilename: 'ml-animations-progress.v1.json',
  lastSyncAt: '',
  lastStatus: '',
  lastError: '',
};

export class GitHubProgressSyncError extends Error {
  constructor(message, details = {}) {
    super(message);
    this.name = 'GitHubProgressSyncError';
    this.status = details.status;
    this.rateLimit = details.rateLimit;
    this.retryAfter = details.retryAfter;
  }
}

function getStorage(storage) {
  if (storage) return storage;
  if (typeof window === 'undefined') return null;
  return window.localStorage;
}

function isPlainObject(value) {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value));
}

function notifySettingsUpdated() {
  if (typeof window === 'undefined') return;
  window.dispatchEvent(new Event(GITHUB_SYNC_EVENT));
}

function parseJson(value, fallback) {
  if (!value) return fallback;
  try {
    return JSON.parse(value);
  } catch {
    return fallback;
  }
}

function cleanString(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function cleanPath(value) {
  return cleanString(value).replace(/^\/+/, '') || DEFAULT_GITHUB_SYNC_SETTINGS.path;
}

function cleanBrokerUrl(value) {
  return cleanString(value).replace(/\/+$/, '');
}

function readHeader(headers, name) {
  if (!headers) return null;
  if (typeof headers.get === 'function') return headers.get(name);
  return headers[name] || headers[name.toLowerCase()] || null;
}

function readRateLimit(headers) {
  return {
    limit: Number(readHeader(headers, 'x-ratelimit-limit') || 0),
    remaining: Number(readHeader(headers, 'x-ratelimit-remaining') || 0),
    reset: Number(readHeader(headers, 'x-ratelimit-reset') || 0),
  };
}

function compareIsoDates(left, right) {
  return Date.parse(left || '') - Date.parse(right || '');
}

function joinBrokerUrl(baseUrl, pathname) {
  const normalizedBase = cleanBrokerUrl(baseUrl);
  if (!normalizedBase) {
    throw new GitHubProgressSyncError('Add a GitHub Storage URL before signing in or syncing.');
  }
  return `${normalizedBase}${pathname.startsWith('/') ? pathname : `/${pathname}`}`;
}

function createSearchParams(entries) {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(entries)) {
    if (value !== undefined && value !== null && String(value).trim() !== '') {
      params.set(key, String(value));
    }
  }
  return params;
}

export function normalizeGitHubSyncSettings(value) {
  const source = isPlainObject(value) ? value : {};
  const target = source.target === GITHUB_SYNC_TARGETS.gist
    ? GITHUB_SYNC_TARGETS.gist
    : GITHUB_SYNC_TARGETS.repo;

  return {
    ...DEFAULT_GITHUB_SYNC_SETTINGS,
    enabled: source.enabled === true,
    autoSync: source.autoSync === true,
    target,
    brokerUrl: cleanBrokerUrl(source.brokerUrl),
    owner: cleanString(source.owner),
    repo: cleanString(source.repo),
    branch: cleanString(source.branch) || DEFAULT_GITHUB_SYNC_SETTINGS.branch,
    path: cleanPath(source.path),
    gistId: cleanString(source.gistId),
    gistFilename: cleanString(source.gistFilename) || DEFAULT_GITHUB_SYNC_SETTINGS.gistFilename,
    lastSyncAt: cleanString(source.lastSyncAt),
    lastStatus: cleanString(source.lastStatus),
    lastError: cleanString(source.lastError),
  };
}

export function readGitHubSyncSettings(storage) {
  const target = getStorage(storage);
  if (!target) return { ...DEFAULT_GITHUB_SYNC_SETTINGS };
  return normalizeGitHubSyncSettings(parseJson(target.getItem(GITHUB_SYNC_SETTINGS_KEY), {}));
}

export function writeGitHubSyncSettings(settings, storage) {
  const target = getStorage(storage);
  const normalized = normalizeGitHubSyncSettings(settings);
  if (!target) return normalized;
  target.setItem(GITHUB_SYNC_SETTINGS_KEY, JSON.stringify(normalized));
  notifySettingsUpdated();
  return normalized;
}

export function disconnectGitHubSyncSettings(storage) {
  const current = readGitHubSyncSettings(storage);
  return writeGitHubSyncSettings({
    ...current,
    enabled: false,
    autoSync: false,
    lastStatus: 'Disconnected',
    lastError: '',
  }, storage);
}

export function buildProgressEnvelope(progress, now = new Date()) {
  return {
    version: 1,
    updatedAt: now.toISOString(),
    source: 'ml-animations',
    progress: normalizeCodeLabProgress(progress),
  };
}

export function normalizeProgressEnvelope(value) {
  if (isPlainObject(value) && value.version === 1 && isPlainObject(value.progress)) {
    return {
      version: 1,
      updatedAt: typeof value.updatedAt === 'string' ? value.updatedAt : '',
      source: typeof value.source === 'string' ? value.source : 'ml-animations',
      progress: normalizeCodeLabProgress(value.progress),
    };
  }

  return buildProgressEnvelope(normalizeCodeLabProgress(value), new Date(0));
}

export function parseProgressEnvelopeJson(json) {
  return normalizeProgressEnvelope(JSON.parse(json));
}

export function mergeCodeLabProgress(localProgress, remoteProgress) {
  const local = normalizeCodeLabProgress(localProgress);
  const remote = normalizeCodeLabProgress(remoteProgress);
  const merged = { ...local };

  for (const [scopeId, scopeProgress] of Object.entries(remote)) {
    merged[scopeId] = { ...(merged[scopeId] || {}) };

    for (const [exerciseId, remoteEntry] of Object.entries(scopeProgress)) {
      const localEntry = merged[scopeId][exerciseId];
      if (!localEntry || compareIsoDates(remoteEntry.lastPassedAt, localEntry.lastPassedAt) >= 0) {
        merged[scopeId][exerciseId] = remoteEntry;
      }
    }
  }

  return normalizeCodeLabProgress(merged);
}

export function summarizeProgressDocument(progress) {
  const normalized = normalizeCodeLabProgress(progress);
  let passedCount = 0;
  for (const scopeProgress of Object.values(normalized)) {
    passedCount += Object.keys(scopeProgress).length;
  }

  return {
    scopeCount: Object.keys(normalized).length,
    passedCount,
  };
}

export function getGitHubSyncAuthUrl(settings, returnTo = '') {
  const normalized = normalizeGitHubSyncSettings(settings);
  const params = createSearchParams({
    returnTo,
    target: normalized.target,
  });
  return `${joinBrokerUrl(normalized.brokerUrl, '/auth/github/start')}?${params.toString()}`;
}

export function startGitHubSignIn(settings, returnTo) {
  if (typeof window === 'undefined') {
    throw new GitHubProgressSyncError('GitHub sign-in requires a browser window.');
  }
  window.location.assign(getGitHubSyncAuthUrl(settings, returnTo || window.location.href));
}

async function parseBrokerResponse(response) {
  const rateLimit = readRateLimit(response.headers);
  const retryAfter = readHeader(response.headers, 'retry-after');
  const text = await response.text();
  const payload = parseJson(text, text ? { error: text } : {});

  if (!response.ok) {
    const message = payload?.error || payload?.message || `GitHub sync failed with HTTP ${response.status}.`;
    throw new GitHubProgressSyncError(message, {
      status: response.status,
      rateLimit,
      retryAfter: retryAfter ? Number(retryAfter) : undefined,
    });
  }

  return {
    payload,
    rateLimit,
  };
}

async function brokerRequest(settings, pathname, options = {}, fetchFn = fetch) {
  const response = await fetchFn(joinBrokerUrl(settings.brokerUrl, pathname), {
    credentials: 'include',
    headers: {
      Accept: 'application/json',
      ...(options.body ? { 'Content-Type': 'application/json' } : {}),
      ...(options.headers || {}),
    },
    ...options,
  });
  return parseBrokerResponse(response);
}

function buildProgressQuery(settings) {
  const normalized = normalizeGitHubSyncSettings(settings);
  return createSearchParams({
    target: normalized.target,
    owner: normalized.owner,
    repo: normalized.repo,
    branch: normalized.branch,
    path: normalized.path,
    gistId: normalized.gistId,
    gistFilename: normalized.gistFilename,
  });
}

export async function getGitHubSyncSession(settings, fetchFn = fetch) {
  const { payload } = await brokerRequest(settings, '/api/github/session', {}, fetchFn);
  return payload;
}

export async function listGitHubSyncRepos(settings, fetchFn = fetch) {
  const { payload } = await brokerRequest(settings, '/api/github/repos', {}, fetchFn);
  return Array.isArray(payload.repos) ? payload.repos : [];
}

export async function disconnectGitHubSession(settings, fetchFn = fetch) {
  const { payload } = await brokerRequest(settings, '/auth/github/logout', {
    method: 'POST',
  }, fetchFn);
  return payload;
}

export async function pullGitHubProgress(settings, fetchFn = fetch) {
  const query = buildProgressQuery(settings);
  const { payload, rateLimit } = await brokerRequest(
    settings,
    `/api/github/progress?${query.toString()}`,
    {},
    fetchFn,
  );

  return {
    exists: payload.exists === true,
    sha: payload.sha || '',
    gistId: payload.gistId || '',
    envelope: normalizeProgressEnvelope(payload.envelope || {}),
    rateLimit,
  };
}

export async function pushGitHubProgress(settings, envelope, remoteMeta = {}, fetchFn = fetch) {
  const normalized = normalizeGitHubSyncSettings(settings);
  const { payload, rateLimit } = await brokerRequest(normalized, '/api/github/progress', {
    method: 'PUT',
    body: JSON.stringify({
      target: normalized.target,
      owner: normalized.owner,
      repo: normalized.repo,
      branch: normalized.branch,
      path: normalized.path,
      gistId: remoteMeta.gistId || normalized.gistId,
      gistFilename: normalized.gistFilename,
      sha: remoteMeta.sha || '',
      envelope: normalizeProgressEnvelope(envelope),
    }),
  }, fetchFn);

  return {
    sha: payload.sha || '',
    gistId: payload.gistId || '',
    rateLimit,
  };
}

export async function syncCodeLabProgressToGitHub({
  settings = readGitHubSyncSettings(),
  storage,
  fetchFn = fetch,
  now = new Date(),
} = {}) {
  const normalized = normalizeGitHubSyncSettings(settings);
  if (!normalized.enabled) {
    throw new GitHubProgressSyncError('Enable GitHub sync before syncing progress.');
  }

  const localProgress = readCodeLabProgress(storage);
  const remote = await pullGitHubProgress(normalized, fetchFn);
  const mergedProgress = mergeCodeLabProgress(localProgress, remote.envelope.progress);
  const envelope = buildProgressEnvelope(mergedProgress, now);

  writeCodeLabProgress(mergedProgress, storage);

  let pushResult;
  try {
    pushResult = await pushGitHubProgress(normalized, envelope, remote, fetchFn);
  } catch (error) {
    if (!(error instanceof GitHubProgressSyncError) || error.status !== 409) throw error;
    const latestRemote = await pullGitHubProgress(normalized, fetchFn);
    const retryProgress = mergeCodeLabProgress(mergedProgress, latestRemote.envelope.progress);
    const retryEnvelope = buildProgressEnvelope(retryProgress, now);
    writeCodeLabProgress(retryProgress, storage);
    pushResult = await pushGitHubProgress(normalized, retryEnvelope, latestRemote, fetchFn);
  }

  const nextSettings = writeGitHubSyncSettings({
    ...normalized,
    enabled: true,
    gistId: pushResult.gistId || normalized.gistId,
    lastSyncAt: now.toISOString(),
    lastStatus: 'Synced',
    lastError: '',
  }, storage);

  return {
    settings: nextSettings,
    progress: readCodeLabProgress(storage),
    summary: summarizeProgressDocument(readCodeLabProgress(storage)),
    remote: pushResult,
  };
}
