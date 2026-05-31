import test from 'node:test';
import assert from 'node:assert/strict';

import {
  TRAIN_VALIDATION_ROWS,
  assignByMode,
  driftGap,
  positiveRate,
  splitCounts,
} from './trainValidationTestSplitModel.js';

function sizes(splits) {
  return {
    train: splits.train.length,
    validation: splits.validation.length,
    test: splits.test.length,
  };
}

test('train validation split model assigns exact split sizes for each mode', () => {
  const expected = splitCounts(TRAIN_VALIDATION_ROWS.length, 0.2, 0.2);

  assert.deepEqual(sizes(assignByMode('random', 0.2, 0.2)), expected);
  assert.deepEqual(sizes(assignByMode('stratified', 0.2, 0.2)), expected);
  assert.deepEqual(sizes(assignByMode('time', 0.2, 0.2)), expected);
});

test('time split keeps later rows out of training', () => {
  const splits = assignByMode('time', 0.2, 0.2);
  const maxTrainTime = Math.max(...splits.train.map((row) => row.time));
  const minValidationTime = Math.min(...splits.validation.map((row) => row.time));
  const minTestTime = Math.min(...splits.test.map((row) => row.time));

  assert.ok(maxTrainTime < minValidationTime);
  assert.ok(minValidationTime < minTestTime);
});

test('stratified split keeps each split represented by both classes', () => {
  const stratified = assignByMode('stratified', 0.2, 0.2);
  const rates = ['train', 'validation', 'test'].map((bucket) => positiveRate(stratified[bucket]));

  assert.deepEqual(rates, [5 / 9, 1 / 3, 2 / 3]);
  assert.ok(rates.every((rate) => rate > 0 && rate < 1));
});

test('diagnostic helpers return stable rates and feature drift', () => {
  const splits = assignByMode('time', 0.2, 0.2);

  assert.equal(positiveRate(splits.test), 2 / 3);
  assert.ok(Math.abs(driftGap(splits.train, splits.test) - (85 / 3)) < 1e-12);
});
