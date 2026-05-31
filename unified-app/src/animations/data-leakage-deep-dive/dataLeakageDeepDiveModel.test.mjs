import test from 'node:test';
import assert from 'node:assert/strict';

import {
  LEAKAGE_MODES,
  LEAKAGE_ROWS,
  rowIsLeaked,
  scoreGap,
} from './dataLeakageDeepDiveModel.js';

test('strict pipeline removes displayed optimism for every leakage mode', () => {
  for (const mode of Object.keys(LEAKAGE_MODES)) {
    const suspicious = scoreGap(mode, false);
    const strict = scoreGap(mode, true);

    assert.ok(suspicious.optimism > 0, `${mode} should show inflated score before the fix`);
    assert.equal(strict.suspicious, strict.honest, `${mode} strict pipeline should use the honest score`);
    assert.equal(strict.optimism, 0, `${mode} strict pipeline should remove optimism`);
  }
});

test('row leakage highlights match the selected leakage mechanism', () => {
  const idsByMode = Object.fromEntries(
    Object.keys(LEAKAGE_MODES).map((mode) => [
      mode,
      LEAKAGE_ROWS.filter((row) => rowIsLeaked(row, mode)).map((row) => row.id),
    ]),
  );

  assert.deepEqual(idsByMode.duplicates, ['B', 'D']);
  assert.deepEqual(idsByMode.preprocessing, ['D', 'E', 'F']);
  assert.deepEqual(idsByMode.target, ['B', 'D', 'F']);
  assert.deepEqual(idsByMode.time, ['E', 'F']);
  assert.deepEqual(idsByMode.testTuning, ['F']);
});

test('mode copy names the leak path, blocked item, and safer fix', () => {
  for (const [mode, config] of Object.entries(LEAKAGE_MODES)) {
    assert.ok(config.label.length > 5, `${mode} should have a usable label`);
    assert.ok(config.leak.includes('.') || config.leak.length > 30, `${mode} should describe the leak path`);
    assert.ok(config.fix.length > 30, `${mode} should describe a concrete prevention rule`);
    assert.ok(config.leakedItem.length > 3, `${mode} should name the crossed information`);
  }
});
