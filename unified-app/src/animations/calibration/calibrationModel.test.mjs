import assert from 'node:assert/strict';
import test from 'node:test';
import {
  SCENARIOS,
  brierScore,
  expectedCalibrationError,
  project,
  thresholdStats,
  totalCount,
} from './calibrationModel.js';

const closeTo = (actual, expected, tolerance = 1e-10) => {
  assert.ok(Math.abs(actual - expected) <= tolerance, `expected ${actual} to be within ${tolerance} of ${expected}`);
};

test('scenario bins preserve the displayed sample support', () => {
  for (const scenario of Object.values(SCENARIOS)) {
    assert.equal(totalCount(scenario.bins), 110);
    assert.equal(scenario.bins.length, 5);
  }
});

test('expected calibration error weights bucket gaps by bucket size', () => {
  closeTo(expectedCalibrationError(SCENARIOS.calibrated.bins), 1.5 / 110);
  closeTo(expectedCalibrationError(SCENARIOS.overconfident.bins), 9.08 / 110);
  assert.ok(expectedCalibrationError(SCENARIOS.overconfident.bins) > expectedCalibrationError(SCENARIOS.calibrated.bins));
});

test('brier score rewards the calibrated scenario over the overconfident scenario', () => {
  assert.ok(brierScore(SCENARIOS.calibrated.bins) < brierScore(SCENARIOS.overconfident.bins));
});

test('thresholdStats computes expected fractional confusion summaries from buckets', () => {
  const stats = thresholdStats(SCENARIOS.overconfident.bins, 0.5);

  assert.equal(stats.predictedPositive, 68);
  closeTo(stats.precision, 40.76 / 68);
  closeTo(stats.recall, 40.76 / 53.84);
});

test('project maps reliability rates into the displayed svg coordinate range', () => {
  assert.equal(project(0), 260);
  assert.equal(project(1), 40);
  assert.equal(project(0.5), 150);
});
