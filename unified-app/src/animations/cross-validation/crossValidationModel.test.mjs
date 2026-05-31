import test from 'node:test';
import assert from 'node:assert/strict';

import {
  assignFolds,
  duplicateUserLeakage,
  scoreFold,
  summarize,
} from './crossValidationModel.js';

test('grouped folds keep repeated users in one fold', () => {
  const rows = assignFolds(5, 'grouped');
  const foldsByUser = new Map();

  rows.forEach((row) => {
    const folds = foldsByUser.get(row.user) || new Set();
    folds.add(row.fold);
    foldsByUser.set(row.user, folds);
  });

  assert.ok([...foldsByUser.values()].every((folds) => folds.size === 1));
  assert.deepEqual(
    Array.from({ length: 5 }, (_, fold) => duplicateUserLeakage(rows, fold)),
    [[], [], [], [], []],
  );
});

test('random folds surface duplicate-user leakage for repeated entities', () => {
  const rows = assignFolds(5, 'random');
  const leakedUsers = Array.from({ length: 5 }, (_, fold) => duplicateUserLeakage(rows, fold)).flat();

  assert.deepEqual([...new Set(leakedUsers)].sort(), ['u02', 'u04']);
});

test('fold scoring rewards leakage and global preprocessing in the toy diagnostic', () => {
  const rows = assignFolds(5, 'random');
  const cleanScore = scoreFold(rows, 1, true).score;
  const globalPreprocessingScore = scoreFold(rows, 1, false).score;
  const noLeakScore = scoreFold(assignFolds(5, 'grouped'), 1, true).score;

  assert.ok(cleanScore > noLeakScore);
  assert.ok(globalPreprocessingScore > cleanScore);
});

test('summary reports stable aggregate score and leaked fold count', () => {
  const randomSummary = summarize(assignFolds(5, 'random'), true, 5);
  const groupedSummary = summarize(assignFolds(5, 'grouped'), true, 5);

  assert.equal(randomSummary.folds.length, 5);
  assert.equal(randomSummary.leakedFoldCount, 3);
  assert.equal(groupedSummary.leakedFoldCount, 0);
  assert.ok(randomSummary.max >= randomSummary.min);
  assert.ok(randomSummary.std > 0);
});
