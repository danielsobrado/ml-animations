import assert from 'node:assert/strict';
import test from 'node:test';

import { allAnimations, categories } from './animations.js';
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

const AUDIT_SENTINELS = {
  'matrix-multiplication': [/shape|dimension/i, /row|column|dot/i, /commut|order matters/i, /linear map|composition/i],
  'self-attention': [/sqrt|scale|scaled/i, /causal|mask/i, /head|concat/i, /residual|norm/i],
  'classification-metrics': [/threshold|precision|recall/i, /calibration|brier|ece/i, /imbalance|class balance|base rate/i],
  'cuped-variance-reduction': [/pre.?treatment|covariate/i, /variance/i, /post.?treatment|bias/i],
  optimizers: [/curvature|valley/i, /momentum|velocity/i, /adam|squared-gradient|second moment/i, /mini.?batch|noise/i],
  'ppo-clipped-policy-gradient': [/ratio/i, /advantage/i, /kl|entropy/i, /trust.?region|monotonic/i],
};

function mapSearchText(map) {
  const parts = [map.center?.label, ...Object.values(map.center?.tooltip || {})];

  for (const branch of map.branches || []) {
    parts.push(branch.label, ...Object.values(branch.tooltip || {}));
    for (const child of branch.children || []) {
      parts.push(child.label, ...Object.values(child.tooltip || {}));
    }
  }

  return parts.filter(Boolean).join('\n');
}

test('concept map module exports the required API', () => {
  assert.equal(typeof NODE_TYPES, 'object');
  assert.ok(Object.keys(NODE_TYPES).length > 0);
  assert.equal(typeof CONCEPT_MAPS, 'object');
  assert.equal(typeof getConceptMap, 'function');
  assert.equal(typeof isConceptMap, 'function');
});

test('concept map registry has no unowned ids', () => {
  const ownedIds = new Set([
    ...allAnimations.map((animation) => animation.id),
    ...categories.map((category) => category.id),
  ]);

  for (const lessonId of Object.keys(CONCEPT_MAPS)) {
    assert.ok(ownedIds.has(lessonId), `${lessonId} has a concept map but no active lesson or category`);
  }
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

test('audit-priority concept maps retain canonical nuance coverage', () => {
  for (const [lessonId, patterns] of Object.entries(AUDIT_SENTINELS)) {
    const map = getConceptMap(lessonId);
    assert.ok(map, `${lessonId} should have a concept map`);

    const text = mapSearchText(map);
    for (const pattern of patterns) {
      assert.match(text, pattern, `${lessonId} concept map should mention ${pattern}`);
    }
  }
});
