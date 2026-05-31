import test from 'node:test';
import assert from 'node:assert/strict';

import {
  POINTS,
  PRESETS,
  boundaryLine,
  classifyPoint,
  logit,
  metricPercent,
  safeRatio,
  scorePoint,
  sigmoid,
  summarize,
} from './logisticRegressionModel.js';

function closeTo(actual, expected, tolerance = 1e-12) {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} should be within ${tolerance} of ${expected}`);
}

test('sigmoid and logit are inverse transforms around valid probabilities', () => {
  for (const probability of [0.1, 0.25, 0.5, 0.75, 0.9]) {
    closeTo(sigmoid(logit(probability)), probability);
  }
});

test('balanced preset scores and classifies every displayed point', () => {
  const preset = PRESETS.balanced;
  const scored = POINTS.map((point) => classifyPoint(scorePoint(point, preset.weightRisk, preset.weightEngagement, preset.bias), preset.threshold));
  const counts = summarize(scored);

  assert.equal(scored.length, 16);
  assert.deepEqual(counts, { tp: 8, fp: 0, fn: 2, tn: 6 });
  assert.equal(counts.tp + counts.fp + counts.fn + counts.tn, POINTS.length);
});

test('raising threshold trades false positives for false negatives on the same fitted scores', () => {
  const balanced = PRESETS.balanced;
  const cautious = PRESETS.cautious;
  const balancedCounts = summarize(POINTS.map((point) => classifyPoint(scorePoint(point, balanced.weightRisk, balanced.weightEngagement, balanced.bias), balanced.threshold)));
  const cautiousCounts = summarize(POINTS.map((point) => classifyPoint(scorePoint(point, cautious.weightRisk, cautious.weightEngagement, cautious.bias), cautious.threshold)));

  assert.ok(cautiousCounts.tp + cautiousCounts.fp < balancedCounts.tp + balancedCounts.fp);
  assert.ok(cautiousCounts.fn > balancedCounts.fn);
});

test('safe ratios and percent formatting handle empty denominators', () => {
  assert.equal(safeRatio(3, 0), 0);
  assert.equal(metricPercent(0.625), '63%');
});

test('decision boundary remains finite for regular, vertical, and near-constant models', () => {
  const lines = [
    boundaryLine(1.35, -0.45, 0.1, 0.5),
    boundaryLine(1.35, 0, 0.1, 0.5),
    boundaryLine(0, 0, 0, 0.5),
  ];

  for (const line of lines) {
    for (const value of Object.values(line)) {
      assert.ok(Number.isFinite(value), `boundary coordinate should be finite, got ${value}`);
    }
  }
});
