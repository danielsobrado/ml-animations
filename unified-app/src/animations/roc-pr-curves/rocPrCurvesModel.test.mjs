import assert from 'node:assert/strict';
import test from 'node:test';
import {
  confusionAt,
  curvePoints,
  metricPercent,
  metrics,
  prPrecisionForPlot,
} from './rocPrCurvesModel.js';

test('confusionAt computes the displayed threshold counts', () => {
  assert.deepEqual(confusionAt(0.5), { tp: 5, fp: 3, fn: 2, tn: 6 });
  assert.deepEqual(confusionAt(0.8), { tp: 2, fp: 1, fn: 5, tn: 8 });
});

test('metrics use the correct ROC and PR denominators', () => {
  const summary = metrics({ tp: 5, fp: 3, fn: 2, tn: 6 });

  assert.equal(summary.precision, 5 / 8);
  assert.equal(summary.recall, 5 / 7);
  assert.equal(summary.tpr, summary.recall);
  assert.equal(summary.fpr, 3 / 9);
  assert.equal(summary.specificity, 6 / 9);
});

test('raising threshold reduces recovery on the same ranked scores', () => {
  const strict = metrics(confusionAt(0.8));
  const loose = metrics(confusionAt(0.4));

  assert.ok(loose.recall > strict.recall);
  assert.ok(loose.fpr > strict.fpr);
});

test('precision is undefined when no positives are predicted, while PR plot keeps its anchor', () => {
  const summary = metrics(confusionAt(1));

  assert.equal(summary.predictedPositives, 0);
  assert.equal(summary.precision, null);
  assert.equal(metricPercent(summary.precision), 'N/A');
  assert.equal(prPrecisionForPlot(summary), 1);
});

test('curve points stay sorted by ROC position and include PR plotting values', () => {
  const points = curvePoints();

  assert.equal(points.length, 21);
  for (let index = 1; index < points.length; index += 1) {
    const previous = points[index - 1];
    const current = points[index];
    assert.ok(
      current.fpr > previous.fpr || (current.fpr === previous.fpr && current.recall >= previous.recall),
      `point ${index + 1} should not move backward on the ROC sweep`,
    );
    assert.equal(typeof current.precisionPlot, 'number');
  }
});
