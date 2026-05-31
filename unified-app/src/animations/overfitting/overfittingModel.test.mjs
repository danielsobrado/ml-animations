import test from 'node:test';
import assert from 'node:assert/strict';

import {
  bestEpoch,
  curvePath,
  epochProfile,
  errorPath,
  makePoints,
  predict,
  project,
  pseudoNoise,
} from './overfittingModel.js';

test('point generation is deterministic and marks only noisy-label examples in the noisy dataset', () => {
  const clean = makePoints('clean');
  const noisy = makePoints('noisy');
  const tiny = makePoints('tiny');

  assert.equal(clean.length, 26);
  assert.equal(noisy.length, 26);
  assert.equal(tiny.length, 14);
  assert.deepEqual(noisy.filter((point) => point.noisy).map((point) => point.id), [4, 10, 17, 22]);
  assert.ok(clean.every((point) => !point.noisy));
  assert.deepEqual(makePoints('noisy'), noisy);
});

test('pseudo-noise and projection stay within expected display ranges', () => {
  for (let index = 0; index < 20; index += 1) {
    const noise = pseudoNoise(index);
    assert.ok(noise >= 0 && noise < 1, `noise ${index} should be in [0, 1)`);
  }

  assert.deepEqual(project({ x: 0, y: 12 }), { cx: 34, cy: 262 });
  assert.deepEqual(project({ x: 100, y: 116 }), { cx: 366, cy: 36 });
});

test('no regularization on noisy data creates the displayed overfitting gap after the best epoch', () => {
  const profile = epochProfile('noisy', 'none', 12);
  const best = bestEpoch(profile);
  const final = profile.at(-1);

  assert.ok(best.epoch < final.epoch);
  assert.ok(final.train < best.train);
  assert.ok(final.validation > best.validation);
  assert.ok(final.validation - final.train > 12);
});

test('regularization reduces the late noisy-data generalization gap', () => {
  const none = epochProfile('noisy', 'none', 12).at(-1);
  const mild = epochProfile('noisy', 'mild', 12).at(-1);
  const strong = epochProfile('noisy', 'strong', 12).at(-1);

  assert.ok((none.validation - none.train) > (mild.validation - mild.train));
  assert.ok((mild.validation - mild.train) > (strong.validation - strong.train));
});

test('underfit-style early complexity is visibly smoother than late flexible fits', () => {
  const early = predict(50, 1, 'noisy', 'mild');
  const middle = predict(50, 4, 'noisy', 'mild');
  const late = predict(50, 12, 'noisy', 'mild');

  assert.notEqual(early, middle);
  assert.notEqual(middle, late);
});

test('svg path helpers emit stable point counts for the lesson curves', () => {
  const profile = epochProfile('noisy', 'mild', 7);

  assert.equal(curvePath(7, 'noisy', 'mild').split(' ').filter((token) => token === 'M' || token === 'L').length, 75);
  assert.equal(errorPath(profile, 'validation').split(' ').filter((token) => token === 'M' || token === 'L').length, 12);
});
