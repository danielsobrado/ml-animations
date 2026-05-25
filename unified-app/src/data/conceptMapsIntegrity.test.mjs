import assert from 'node:assert/strict';
import test from 'node:test';

import { allAnimations } from './animations.js';
import {
  CONCEPT_MAPS,
  NODE_TYPES,
  getConceptMap,
  isConceptMap,
} from './conceptMaps.js';

const REQUIRED_BRANCH_IDS = [
  'prerequisites',
  'mechanism',
  'intuitions',
  'formula-code',
  'traps',
  'used-later',
];

test('concept map module exports the required API', () => {
  assert.equal(typeof NODE_TYPES, 'object');
  assert.ok(Object.keys(NODE_TYPES).length > 0);
  assert.equal(typeof CONCEPT_MAPS, 'object');
  assert.equal(typeof getConceptMap, 'function');
  assert.equal(typeof isConceptMap, 'function');
});

test('every live lesson has a structured six-branch concept map', () => {
  const lessonIds = new Set(allAnimations.map((animation) => animation.id));

  for (const animation of allAnimations) {
    const map = getConceptMap(animation.id);

    assert.ok(map, `${animation.id} should have a concept map`);
    assert.equal(isConceptMap(map), true, `${animation.id} should use concept-map structure`);
    assert.ok(map.center?.id, `${animation.id} missing center id`);
    assert.ok(map.center?.label, `${animation.id} missing center label`);
    assert.equal(Array.isArray(map.branches), true, `${animation.id} missing branches`);

    const branchIds = new Set(map.branches.map((branch) => branch.id));
    assert.deepEqual([...branchIds], REQUIRED_BRANCH_IDS, `${animation.id} should use the canonical six branches`);

    for (const branch of map.branches) {
      assert.ok(branch.id, `${animation.id} has a branch without id`);
      assert.ok(branch.label, `${animation.id}/${branch.id} missing label`);
      assert.ok(NODE_TYPES[branch.type], `${animation.id}/${branch.id} has unknown type ${branch.type}`);
      assert.equal(Array.isArray(branch.children), true, `${animation.id}/${branch.id} missing children`);
      assert.ok(branch.children.length > 0, `${animation.id}/${branch.id} should have child concepts`);

      for (const child of branch.children) {
        assert.ok(child.id, `${animation.id}/${branch.id} child missing id`);
        assert.ok(child.label, `${animation.id}/${child.id} missing label`);
        assert.ok(
          child.tooltip?.short || child.tooltip?.intuition,
          `${animation.id}/${child.id} missing useful tooltip text`,
        );

        if (child.lessonId) {
          assert.ok(
            lessonIds.has(child.lessonId),
            `${animation.id}/${child.id} links to missing lesson ${child.lessonId}`,
          );
        }
      }
    }
  }
});
