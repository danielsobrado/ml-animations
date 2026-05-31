import assert from 'node:assert/strict';
import test from 'node:test';

import { gradient, learningRateStatus, loss, nextWeight } from './gradientDescentModel.js';

test('gradient descent model follows the displayed quadratic update', () => {
  assert.equal(loss(4), 16);
  assert.equal(gradient(4), 8);
  assert.equal(nextWeight(4, 0.1), 3.2);
  assert.equal(nextWeight(-4, 0.1), -3.2);
});

test('learning-rate examples match the lesson dynamics', () => {
  assert.equal(learningRateStatus(0.01).text, 'Too slow');
  assert.equal(learningRateStatus(0.1).text, 'Good');
  assert.equal(learningRateStatus(0.5).text, 'Good');
  assert.equal(learningRateStatus(0.95).text, 'Oscillates');

  let stableWeight = 4;
  for (let step = 0; step < 12; step += 1) {
    stableWeight = nextWeight(stableWeight, 0.1);
  }

  const exactStepWeight = nextWeight(4, 0.5);
  let oscillatingWeight = 4;
  const signs = [];
  for (let step = 0; step < 4; step += 1) {
    oscillatingWeight = nextWeight(oscillatingWeight, 0.95);
    signs.push(Math.sign(oscillatingWeight));
  }

  assert.ok(Math.abs(stableWeight) < 0.3, '0.1 should converge toward the bowl minimum');
  assert.equal(exactStepWeight, 0, '0.5 should jump directly to the quadratic minimum');
  assert.deepEqual(signs, [-1, 1, -1, 1], '0.95 should oscillate around the minimum');
  assert.ok(Math.abs(oscillatingWeight) < 4, '0.95 should oscillate with shrinking magnitude on this quadratic');
});
