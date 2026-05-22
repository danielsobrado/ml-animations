import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { allAnimations, categories } from './animations.js';
import { D_TIER_PLACEHOLDERS, MANUAL_LESSON_QUALITY } from './lessonQualityManifest.js';
import { HUB_LEARNING_PATHS as LEARNING_PATHS } from './learningPaths.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const APP_DIR = path.resolve(__dirname, '..', '..');

function findIndexFile(dir) {
  const candidates = ['index.jsx', 'index.tsx', 'index.js', 'index.ts'];
  return candidates.find((fileName) => fs.existsSync(path.join(dir, fileName)));
}

function inferTierFromSource(id) {
  const dir = path.join(APP_DIR, 'animations', id);
  const indexFile = findIndexFile(dir);

  if (!indexFile) {
    return {
      tier: 'D',
      reason: 'Missing index entry file.',
      nextAction: 'Create a lesson entry module before enabling the route.',
    };
  }

  const indexPath = path.join(dir, indexFile);
  const source = fs.readFileSync(indexPath, 'utf8');
  const { size } = fs.statSync(indexPath);
  const childCount = fs
    .readdirSync(dir, { withFileTypes: true })
    .filter((entry) => entry.isFile())
    .map((entry) => entry.name)
    .filter((fileName) => /\.tsx?$|\.jsx$/.test(fileName) && !fileName.startsWith('index.'))
    .length;
  const sharedLesson = /CoreMlLesson/.test(source) || /core-ml-shared/.test(source);

  if (size <= 260) return { tier: 'C', reason: 'Minimal lesson scaffold.', nextAction: 'Needs lesson-specific mechanics.' };
  if (sharedLesson && size <= 1400 && childCount === 0) {
    return {
      tier: 'C',
      reason: 'Delegates most structure to shared lesson shell.',
      nextAction: 'Add lesson-specific controls and failure-mode checkpoints.',
    };
  }
  if (size >= 13000 || childCount >= 9) {
    return { tier: 'A', reason: 'High-density custom lesson structure.', nextAction: 'Run focused misconception and edge-case checks.' };
  }
  if (size >= 7000 || childCount >= 4) return { tier: 'B', reason: 'Meaningful custom lesson body with reusable mechanics.', nextAction: 'Add 1-2 advanced misconception checks.' };
  return { tier: 'B', reason: 'Adequate custom content and lesson controls.', nextAction: 'Add deeper failure-case examples.' };
}

function getLessonQuality(id) {
  const manual = MANUAL_LESSON_QUALITY[id];
  if (manual) return manual;

  return inferTierFromSource(id);
}

test('critical learning-path lessons carry manual audit entries', () => {
  const startPath = LEARNING_PATHS.find((path) => path.id === 'start-here');
  const reliabilityPath = LEARNING_PATHS.find((path) => path.id === 'model-reliability-path');
  const critical = new Set([...(startPath?.nodes || []), ...(reliabilityPath?.nodes || [])]);

  for (const animationId of critical) {
    assert.ok(
      Object.prototype.hasOwnProperty.call(MANUAL_LESSON_QUALITY, animationId),
      `${animationId} should have explicit manual quality audit entry`,
    );
  }
});

test('every animation has a valid quality tier classification', () => {
  const categoryIds = new Set(categories.map((category) => category.id));
  const knownTiers = new Set(['A', 'B', 'C', 'D']);
  const allIds = new Set(allAnimations.map((animation) => animation.id));

  for (const animationId of allIds) {
    assert.ok(categoryIds.has(allAnimations.find((animation) => animation.id === animationId).categoryId));
    const quality = getLessonQuality(animationId);

    assert.ok(knownTiers.has(quality.tier), `${animationId} has invalid tier ${quality.tier}`);
    assert.ok(quality.reason && /\S/.test(quality.reason), `${animationId} should document rationale`);
    assert.ok(quality.nextAction && /\S/.test(quality.nextAction), `${animationId} should document a concrete next action`);
  }
});

test('no unexpected critical-path Tier D after this sprint', () => {
  const startPath = LEARNING_PATHS.find((path) => path.id === 'start-here');
  const reliabilityPath = LEARNING_PATHS.find((path) => path.id === 'model-reliability-path');
  const critical = new Set([...(startPath?.nodes || []), ...(reliabilityPath?.nodes || [])]);

  for (const animationId of critical) {
    const quality = getLessonQuality(animationId);

    if (quality.tier === 'D') {
      assert.ok(
        D_TIER_PLACEHOLDERS.includes(animationId),
        `${animationId} is D but is not in approved placeholder list`,
      );
    }
  }
});
