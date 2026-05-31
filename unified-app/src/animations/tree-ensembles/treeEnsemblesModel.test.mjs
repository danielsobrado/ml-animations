import assert from 'node:assert/strict';
import test from 'node:test';
import {
  POINTS,
  accuracy,
  boostedScore,
  forestPrediction,
  predictTree,
  ruleVote,
  toScreen,
} from './treeEnsemblesModel.js';

test('single tree depth increases the displayed training fit on the toy data', () => {
  assert.equal(POINTS.length, 12);
  assert.ok(accuracy(2) >= accuracy(1));
  assert.ok(accuracy(3) >= accuracy(2));
  assert.equal(accuracy(3), 11 / 12);
});

test('single tree split rules match the displayed depth controls', () => {
  const leftLow = { x: 0.25, y: 0.64, label: 0 };
  const leftHigh = { x: 0.31, y: 0.79, label: 1 };
  const rightLow = { x: 0.60, y: 0.34, label: 0 };
  const farRightLow = { x: 0.81, y: 0.28, label: 1 };

  assert.equal(predictTree(leftLow, 2), 0);
  assert.equal(predictTree(leftHigh, 2), 1);
  assert.equal(predictTree(rightLow, 2), 0);
  assert.equal(predictTree(farRightLow, 3), 1);
});

test('forest prediction aggregates only the selected number of rule votes', () => {
  const selectedPoint = POINTS[8];
  const forest = forestPrediction(selectedPoint, 5);

  assert.equal(forest.votes.length, 5);
  assert.equal(forest.positiveVotes, forest.votes.filter(Boolean).length);
  assert.equal(forest.probability, forest.positiveVotes / 5);
  assert.equal(forest.label, forest.positiveVotes >= 3 ? 1 : 0);
});

test('ruleVote honors positive and inverted threshold polarity', () => {
  assert.equal(ruleVote({ x: 0.8 }, { feature: 'x', threshold: 0.74, polarity: 1 }), 1);
  assert.equal(ruleVote({ x: 0.8 }, { feature: 'x', threshold: 0.74, polarity: -1 }), 0);
});

test('boosting score applies only matched correction rounds with learning-rate shrinkage', () => {
  const point = { x: 0.81, y: 0.28, label: 1 };
  const boosted = boostedScore(point, 5, 0.5);
  const matchedDeltaSum = boosted.steps.reduce((sum, step) => sum + step.delta, 0);

  assert.equal(boosted.steps.length, 5);
  assert.equal(boosted.steps.filter((step) => step.matched).length, 3);
  assert.equal(Number((boosted.score - (-0.15)).toFixed(6)), Number(matchedDeltaSum.toFixed(6)));
  assert.ok(boosted.probability > 0.5);
});

test('toScreen projects normalized points into the split-map chart bounds', () => {
  for (const point of POINTS) {
    const [x, y] = toScreen(point);
    assert.ok(x >= 32 && x <= 328);
    assert.ok(y >= 32 && y <= 328);
  }
});
