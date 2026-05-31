import test from 'node:test';
import assert from 'node:assert/strict';

import {
  BASE_POINTS,
  OUTLIER,
  fitScaler,
  transformPoint,
  transformValue,
} from './featureScalingPreprocessingModel.js';

function closeTo(actual, expected, tolerance = 1e-12) {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} should be within ${tolerance} of ${expected}`);
}

test('fit scaler learns statistics from training rows by default', () => {
  const scaler = fitScaler([...BASE_POINTS, OUTLIER], false);

  closeTo(scaler.age.mean, 37.25);
  closeTo(scaler.age.std, Math.sqrt(128.1875));
  closeTo(scaler.income.mean, 65000);
  closeTo(scaler.income.std, Math.sqrt(465000000));
  assert.equal(scaler.income.max, 94000);
});

test('fit on all data exposes the leakage toggle by changing validation-dependent statistics', () => {
  const safe = fitScaler([...BASE_POINTS, OUTLIER], false);
  const leaky = fitScaler([...BASE_POINTS, OUTLIER], true);

  assert.equal(safe.income.max, 94000);
  assert.equal(leaky.income.max, OUTLIER.income);
  assert.ok(leaky.income.mean > safe.income.mean);
  assert.ok(leaky.income.std > safe.income.std);
});

test('min-max scaling anchors training endpoints while held-out values can exceed the fitted range', () => {
  const scaler = fitScaler([...BASE_POINTS, OUTLIER], false);
  const trainMinimum = BASE_POINTS.find((point) => point.id === 'A');
  const trainMaximum = BASE_POINTS.find((point) => point.id === 'D');

  assert.equal(transformPoint(trainMinimum, scaler, 'minmax').x, 0);
  assert.equal(transformPoint(trainMaximum, scaler, 'minmax').x, 1);
  assert.ok(transformValue(OUTLIER.income, scaler.income, 'minmax') > 1);
});

test('robust scaling uses median and iqr instead of mean and standard deviation', () => {
  const scaler = fitScaler([...BASE_POINTS, OUTLIER], false);
  const analyst = BASE_POINTS.find((point) => point.id === 'F');

  closeTo(transformPoint(analyst, scaler, 'standard').y, (58000 - 65000) / Math.sqrt(465000000));
  closeTo(transformPoint(analyst, scaler, 'robust').y, (58000 - 64000) / 32000);
});
