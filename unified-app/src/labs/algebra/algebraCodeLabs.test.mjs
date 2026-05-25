import assert from 'node:assert/strict';
import test from 'node:test';

import {
  ALGEBRA_CODE_LABS,
  getAlgebraCodeLabsForLesson,
} from './algebraCodeLabs.js';

const REQUIRED_FIELDS = [
  'id',
  'group',
  'stepLabel',
  'title',
  'concept',
  'objective',
  'starterCode',
  'testCode',
  'hints',
  'solution',
  'explanation',
];

const MATRIX_GROUPS = new Set([
  'Dot product',
  'Matrix cell',
  'Matrix multiplication',
  'Shape compatibility',
]);

test('aggregate algebra code labs keep the Rustlings-style schema', () => {
  const ids = new Set();

  for (const exercise of ALGEBRA_CODE_LABS) {
    for (const field of REQUIRED_FIELDS) {
      assert.ok(exercise[field] !== undefined, `${exercise.id || '<missing id>'} is missing ${field}`);
    }

    assert.equal(typeof exercise.id, 'string');
    assert.equal(ids.has(exercise.id), false, `${exercise.id} should be unique`);
    ids.add(exercise.id);

    assert.match(exercise.starterCode, /TODO/);
    assert.equal(Array.isArray(exercise.hints), true, `${exercise.id} hints should be an array`);
    assert.ok(exercise.hints.length >= 1, `${exercise.id} should have at least one hint`);

    const checkCalls = exercise.testCode.match(/\bcheck\s*\(/g) ?? [];
    assert.ok(checkCalls.length >= 2, `${exercise.id} should have at least two embedded checks`);
    assert.match(exercise.testCode, /return\s+results\s*;/);
  }
});

test('aggregate algebra code lab solutions pass their embedded tests', () => {
  const failures = [];

  for (const exercise of ALGEBRA_CODE_LABS) {
    try {
      const run = new Function(`${exercise.solution}\n${exercise.testCode}`);
      const results = run();
      const failed = Array.isArray(results)
        ? results.filter((result) => !result.passed)
        : [{ name: 'testCode return value', actual: results, expected: 'array', passed: false }];

      if (failed.length > 0) failures.push({ id: exercise.id, failed });
    } catch (error) {
      failures.push({ id: exercise.id, error: error.stack || String(error) });
    }
  }

  assert.deepEqual(failures, []);
});

test('matrix multiplication resolves a focused lesson-scoped code lab', () => {
  const matrixExercises = getAlgebraCodeLabsForLesson('matrix-multiplication');

  assert.ok(matrixExercises.length > 0);
  assert.ok(matrixExercises.length < ALGEBRA_CODE_LABS.length);

  for (const exercise of matrixExercises) {
    assert.ok(MATRIX_GROUPS.has(exercise.group), `${exercise.id} should be matrix-scoped`);
  }
});

test('unknown lessons do not claim a scoped algebra lab', () => {
  assert.equal(getAlgebraCodeLabsForLesson('unknown-lesson'), null);
});
