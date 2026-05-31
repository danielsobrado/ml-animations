import assert from 'node:assert/strict';
import test from 'node:test';
import {
  POINTS,
  classStats,
  classifyKnn,
  classifyNaiveBayes,
  classifySvm,
  project,
  svmBoundarySegment,
  svmMarginScore,
} from './knnNaiveBayesSvmModel.js';

test('kNN sorts neighbors by distance and reports vote confidence', () => {
  const result = classifyKnn({ x: -1.8, y: 0.8 }, 3);

  assert.equal(result.neighbors.length, POINTS.length);
  assert.deepEqual(result.neighbors.slice(0, 3).map((point) => point.id), ['B', 'A', 'C']);
  assert.equal(result.prediction, 'blue');
  assert.equal(result.confidence, 1);
});

test('Gaussian Naive Bayes uses class priors and finite feature likelihood scores', () => {
  const blueStats = classStats('blue');
  const orangeStats = classStats('orange');
  const result = classifyNaiveBayes({ x: 1.5, y: -1.0 });

  assert.equal(blueStats.prior, 0.5);
  assert.equal(orangeStats.prior, 0.5);
  assert.ok(Number.isFinite(result.scores.blue));
  assert.ok(Number.isFinite(result.scores.orange));
  assert.equal(result.prediction, 'orange');
  assert.ok(result.confidence > 0.5 && result.confidence <= 1);
});

test('SVM prediction follows the sign of the displayed margin score', () => {
  const blueQuery = { x: -1.2, y: 1.0 };
  const orangeQuery = { x: 1.4, y: -1.0 };

  assert.ok(svmMarginScore(blueQuery) < 0);
  assert.ok(svmMarginScore(orangeQuery) > 0);
  assert.equal(classifySvm(blueQuery).prediction, 'blue');
  assert.equal(classifySvm(orangeQuery).prediction, 'orange');
});

test('SVM boundary segment is derived from the same margin equation as classification', () => {
  const [start, end] = svmBoundarySegment();

  assert.ok(Number.isFinite(start.cx));
  assert.ok(Number.isFinite(start.cy));
  assert.ok(Number.isFinite(end.cx));
  assert.ok(Number.isFinite(end.cy));
  assert.ok(start.cx < end.cx);
  assert.ok(start.cy > end.cy);
  assert.equal(start.cy, project({ x: 0, y: -2.4 }).cy);
  assert.equal(end.cy, project({ x: 0, y: 2.4 }).cy);
});

test('projection keeps lesson points inside the displayed plot bounds', () => {
  for (const point of POINTS) {
    const { cx, cy } = project(point);
    assert.ok(cx >= 36 && cx <= 364, `${point.id} x should be inside chart`);
    assert.ok(cy >= 36 && cy <= 276, `${point.id} y should be inside chart`);
  }
});
