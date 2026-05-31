import assert from 'node:assert/strict';
import test from 'node:test';
import {
  FEATURES,
  bestLambda,
  diagnosisForState,
  linePath,
  lossProfile,
  regularizationSummary,
  shrinkFeature,
  sweepProfile,
} from './regularizationModel.js';

test('no penalty leaves weights and losses unchanged across lambda values', () => {
  const lowWeights = FEATURES.map((feature) => shrinkFeature(feature, 'none', 0));
  const highWeights = FEATURES.map((feature) => shrinkFeature(feature, 'none', 1));
  const lowLoss = lossProfile(lowWeights, 0, 'none');
  const highLoss = lossProfile(highWeights, 1, 'none');

  assert.deepEqual(highWeights, lowWeights);
  assert.equal(highLoss.penaltyLoss, 0);
  assert.equal(highLoss.train, lowLoss.train);
  assert.equal(highLoss.validation, lowLoss.validation);
});

test('L2 shrinks weights smoothly without sparse feature removal at moderate lambda', () => {
  const weights = FEATURES.map((feature) => shrinkFeature(feature, 'l2', 0.35));
  const summary = regularizationSummary(weights);

  assert.ok(Math.abs(weights.find((feature) => feature.id === 'signalA').weight) < 2.4);
  assert.equal(summary.removedCount, 0);
  assert.equal(summary.noisyActive, 3);
  assert.ok(summary.usefulRetention > 0.55);
});

test('L1 removes weak noisy coefficients before strong useful signal', () => {
  const weights = FEATURES.map((feature) => shrinkFeature(feature, 'l1', 0.8));
  const noiseWeights = weights.filter((feature) => !feature.useful);
  const usefulWeights = weights.filter((feature) => feature.useful);

  assert.ok(noiseWeights.filter((feature) => feature.removed).length >= 2);
  assert.ok(usefulWeights.some((feature) => !feature.removed));
  assert.ok(Math.abs(weights.find((feature) => feature.id === 'signalA').weight) > Math.abs(weights.find((feature) => feature.id === 'noiseA').weight));
});

test('regularized sweep exposes a validation optimum away from the largest lambda', () => {
  const sweep = sweepProfile('elastic');
  const best = bestLambda(sweep);

  assert.equal(sweep.length, 11);
  assert.ok(best.lambda > 0);
  assert.ok(best.lambda < 1);
  assert.ok(sweep.at(-1).validation > best.validation);
});

test('diagnosis copy separates no penalty, weak, strong, and balanced states', () => {
  assert.equal(
    diagnosisForState({ penaltyId: 'none', lambda: 1, noisyActive: 3, usefulRetention: 1 }),
    'No penalty: noisy weights remain active; compare a regularized setting on validation.',
  );
  assert.equal(
    diagnosisForState({ penaltyId: 'l2', lambda: 0.05, noisyActive: 3, usefulRetention: 0.95 }),
    'Too weak: noisy weights remain active and validation can suffer.',
  );
  assert.equal(
    diagnosisForState({ penaltyId: 'l1', lambda: 0.9, noisyActive: 0, usefulRetention: 0.25 }),
    'Too strong: useful signal is being shrunk enough to underfit.',
  );
  assert.equal(
    diagnosisForState({ penaltyId: 'elastic', lambda: 0.35, noisyActive: 1, usefulRetention: 0.75 }),
    'Balanced: noisy weights are controlled while useful signal remains.',
  );
});

test('linePath returns one finite svg command per sweep point', () => {
  const path = linePath(sweepProfile('l2'), 'validation');
  const commands = path.match(/[ML]/g) || [];

  assert.match(path, /^M \d+\.\d -?\d+\.\d/);
  assert.equal(commands.length, 11);
});
