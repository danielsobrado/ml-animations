import assert from 'node:assert/strict';
import test from 'node:test';

import {
  CODE_LAB_PROGRESS_KEY,
  exportCodeLabProgressJson,
  importCodeLabProgressJson,
  markCodeLabExercisePassed,
  readCodeLabProgress,
  summarizeCodeLabProgress,
} from './codeLabProgress.js';

function createStorage(seed = {}) {
  const values = new Map(Object.entries(seed));
  return {
    getItem(key) {
      return values.has(key) ? values.get(key) : null;
    },
    setItem(key, value) {
      values.set(key, String(value));
    },
  };
}

test('code lab progress reads empty and malformed storage safely', () => {
  assert.deepEqual(readCodeLabProgress(createStorage()), {});
  assert.deepEqual(readCodeLabProgress(createStorage({ [CODE_LAB_PROGRESS_KEY]: '{bad json' })), {});
  assert.deepEqual(readCodeLabProgress(createStorage({ [CODE_LAB_PROGRESS_KEY]: '[]' })), {});
});

test('code lab progress writes and merges passed exercise metadata', () => {
  const storage = createStorage();
  const firstTime = new Date('2026-05-25T10:00:00.000Z');
  const secondTime = new Date('2026-05-25T10:05:00.000Z');

  markCodeLabExercisePassed({
    scopeId: 'matrix-multiplication',
    exerciseId: 'dot-product-first-pair',
    checkCount: 3,
    storage,
    now: firstTime,
  });
  const progress = markCodeLabExercisePassed({
    scopeId: 'matrix-multiplication',
    exerciseId: 'dot-product-two-pairs',
    checkCount: 4,
    storage,
    now: secondTime,
  });

  assert.deepEqual(progress, {
    'matrix-multiplication': {
      'dot-product-first-pair': {
        passed: true,
        lastPassedAt: firstTime.toISOString(),
        checkCount: 3,
      },
      'dot-product-two-pairs': {
        passed: true,
        lastPassedAt: secondTime.toISOString(),
        checkCount: 4,
      },
    },
  });
});

test('code lab progress summaries only count exercises from the selected lesson', () => {
  const progress = {
    'matrix-multiplication': {
      'dot-product-first-pair': { passed: true, lastPassedAt: '2026-05-25T10:00:00.000Z', checkCount: 3 },
      'other-lesson-exercise': { passed: true, lastPassedAt: '2026-05-25T10:01:00.000Z', checkCount: 3 },
    },
  };
  const summary = summarizeCodeLabProgress('matrix-multiplication', [
    { id: 'dot-product-first-pair' },
    { id: 'dot-product-two-pairs' },
  ], progress);

  assert.equal(summary.passedCount, 1);
  assert.equal(summary.totalCount, 2);
  assert.equal(summary.complete, false);
  assert.deepEqual([...summary.passedIds], ['dot-product-first-pair']);
});

test('code lab progress imports and exports sanitized JSON', () => {
  const storage = createStorage({
    [CODE_LAB_PROGRESS_KEY]: JSON.stringify({
      existing: {
        exercise: {
          passed: true,
          lastPassedAt: '2026-05-25T09:00:00.000Z',
          checkCount: 2,
          sourceCode: 'should not persist',
        },
      },
    }),
  });

  const progress = importCodeLabProgressJson(JSON.stringify({
    imported: {
      passed: {
        passed: true,
        lastPassedAt: '2026-05-25T10:00:00.000Z',
        checkCount: 3,
        sourceCode: 'should not import',
      },
      failed: {
        passed: false,
        lastPassedAt: '2026-05-25T10:01:00.000Z',
        checkCount: 3,
      },
    },
  }), storage);

  assert.deepEqual(progress, {
    existing: {
      exercise: {
        passed: true,
        lastPassedAt: '2026-05-25T09:00:00.000Z',
        checkCount: 2,
      },
    },
    imported: {
      passed: {
        passed: true,
        lastPassedAt: '2026-05-25T10:00:00.000Z',
        checkCount: 3,
      },
    },
  });
  assert.equal(exportCodeLabProgressJson(storage).includes('sourceCode'), false);
});
