export const CODE_LAB_PROGRESS_KEY = 'ml-animations:code-lab-progress:v1';
export const CODE_LAB_PROGRESS_EVENT = 'ml-animations:code-lab-progress-updated';

function getStorage(storage) {
  if (storage) return storage;
  if (typeof window === 'undefined') return null;
  return window.localStorage;
}

function notifyCodeLabProgressUpdated() {
  if (typeof window === 'undefined') return;
  window.dispatchEvent(new Event(CODE_LAB_PROGRESS_EVENT));
}

function parseJson(value, fallback) {
  if (!value) return fallback;
  try {
    return JSON.parse(value);
  } catch {
    return fallback;
  }
}

function isPlainObject(value) {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value));
}

function normalizeEntry(entry) {
  if (!isPlainObject(entry) || entry.passed !== true) return null;

  return {
    passed: true,
    lastPassedAt: typeof entry.lastPassedAt === 'string'
      ? entry.lastPassedAt
      : new Date(0).toISOString(),
    checkCount: Number.isFinite(entry.checkCount) && entry.checkCount >= 0
      ? Math.floor(entry.checkCount)
      : 0,
  };
}

export function normalizeCodeLabProgress(value) {
  if (!isPlainObject(value)) return {};

  return Object.fromEntries(
    Object.entries(value)
      .map(([scopeId, scopeProgress]) => {
        if (!isPlainObject(scopeProgress)) return null;

        const entries = Object.entries(scopeProgress)
          .map(([exerciseId, entry]) => {
            const normalizedEntry = normalizeEntry(entry);
            return normalizedEntry ? [exerciseId, normalizedEntry] : null;
          })
          .filter(Boolean);

        return entries.length > 0 ? [scopeId, Object.fromEntries(entries)] : null;
      })
      .filter(Boolean),
  );
}

export function readCodeLabProgress(storage) {
  const target = getStorage(storage);
  if (!target) return {};
  return normalizeCodeLabProgress(parseJson(target.getItem(CODE_LAB_PROGRESS_KEY), {}));
}

export function writeCodeLabProgress(progress, storage) {
  const target = getStorage(storage);
  if (!target) return {};

  const normalizedProgress = normalizeCodeLabProgress(progress);
  target.setItem(CODE_LAB_PROGRESS_KEY, JSON.stringify(normalizedProgress));
  notifyCodeLabProgressUpdated();
  return normalizedProgress;
}

export function markCodeLabExercisePassed({
  scopeId,
  exerciseId,
  checkCount,
  storage,
  now = new Date(),
}) {
  if (!scopeId || !exerciseId) return readCodeLabProgress(storage);

  const progress = readCodeLabProgress(storage);
  const nextProgress = {
    ...progress,
    [scopeId]: {
      ...(progress[scopeId] || {}),
      [exerciseId]: {
        passed: true,
        lastPassedAt: now.toISOString(),
        checkCount: Number.isFinite(checkCount) && checkCount >= 0 ? Math.floor(checkCount) : 0,
      },
    },
  };

  return writeCodeLabProgress(nextProgress, storage);
}

export function summarizeCodeLabProgress(scopeId, exercises, progress = readCodeLabProgress()) {
  const exerciseList = Array.isArray(exercises) ? exercises : [];
  const scopeProgress = scopeId && isPlainObject(progress?.[scopeId]) ? progress[scopeId] : {};
  const passedIds = new Set(
    exerciseList
      .filter((exercise) => scopeProgress[exercise.id]?.passed === true)
      .map((exercise) => exercise.id),
  );

  return {
    scopeId,
    passedIds,
    passedCount: passedIds.size,
    totalCount: exerciseList.length,
    complete: exerciseList.length > 0 && passedIds.size === exerciseList.length,
  };
}

export function exportCodeLabProgressJson(storage) {
  return JSON.stringify(readCodeLabProgress(storage), null, 2);
}

export function importCodeLabProgressJson(json, storage) {
  const importedProgress = normalizeCodeLabProgress(JSON.parse(json));
  const currentProgress = readCodeLabProgress(storage);
  const mergedProgress = { ...currentProgress };

  for (const [scopeId, scopeProgress] of Object.entries(importedProgress)) {
    mergedProgress[scopeId] = {
      ...(mergedProgress[scopeId] || {}),
      ...scopeProgress,
    };
  }

  return writeCodeLabProgress(mergedProgress, storage);
}
