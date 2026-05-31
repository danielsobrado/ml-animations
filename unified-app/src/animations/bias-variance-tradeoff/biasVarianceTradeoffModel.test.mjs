import assert from 'node:assert/strict';
import test from 'node:test';
import {
  curvePath,
  errorProfile,
  makePoints,
  predict,
  project,
  recommendationForProfile,
  truth,
} from './biasVarianceTradeoffModel.js';

test('bias variance sample levels produce deterministic point sets', () => {
  const small = makePoints('small', 0.45);
  const medium = makePoints('medium', 0.45);
  const large = makePoints('large', 0.45);

  assert.equal(small.length, 10);
  assert.equal(medium.length, 22);
  assert.equal(large.length, 42);
  assert.equal(small[0].id, 0);
  assert.equal(small[0].x, 4);
  assert.equal(small.at(-1).x, 96);
  assert.deepEqual(makePoints('small', 0.45), small);
});

test('simple model is diagnosed as high-bias underfitting', () => {
  const profile = errorProfile('simple', 'medium', 0.45);

  assert.ok(profile.bias > profile.variance);
  assert.equal(profile.bias, 34);
  assert.equal(recommendationForProfile(profile), 'The model is underfitting: add useful flexibility or better features.');
});

test('flexible model on scarce noisy data is diagnosed as variance-heavy', () => {
  const profile = errorProfile('flexible', 'small', 0.8);

  assert.ok(profile.variance > profile.bias + 10);
  assert.ok(profile.gap > 35);
  assert.equal(profile.bias, 6);
  assert.equal(recommendationForProfile(profile), 'The model is variance-heavy: add data, regularize, simplify, or use averaging.');
});

test('larger samples reduce the flexible model variance term', () => {
  const small = errorProfile('flexible', 'small', 0.6);
  const large = errorProfile('flexible', 'large', 0.6);

  assert.ok(large.variance < small.variance);
  assert.ok(large.validation < small.validation);
});

test('prediction curves keep simple, balanced, and truth behaviors distinct', () => {
  const x = 30;
  const simpleMiss = Math.abs(predict(x, 'simple', 0.45) - truth(x));
  const balancedMiss = Math.abs(predict(x, 'balanced', 0.45) - truth(x));
  const flexibleMiss = Math.abs(predict(x, 'flexible', 0.45) - truth(x));

  assert.ok(balancedMiss < simpleMiss);
  assert.ok(flexibleMiss < simpleMiss);
});

test('curve and projection helpers return finite chart coordinates', () => {
  const projected = project({ x: 50, y: truth(50) });
  const path = curvePath('balanced', 0.45);
  const commands = path.match(/[ML]/g) || [];

  assert.ok(Number.isFinite(projected.cx));
  assert.ok(Number.isFinite(projected.cy));
  assert.match(path, /^M \d+\.\d -?\d+\.\d/);
  assert.equal(commands.length, 70);
});
