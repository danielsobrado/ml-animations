import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { categories, allAnimations, curriculumTracks } from './animations.js';
import { HUB_LEARNING_PATHS } from './learningPaths.js';
import { availableAnimations } from '../animations/index.js';
import { lessonAssessments } from './lessonAssessments.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const animationsDir = path.join(__dirname, '..', 'animations');
const animationIds = new Set(allAnimations.map((animation) => animation.id));
const registryIds = new Set(availableAnimations);

test('every catalog item has a registered route module', () => {
  const categoryIds = categories.flatMap((category) => category.items.map((item) => item.id));
  for (const categoryId of categoryIds) {
    assert.ok(registryIds.has(categoryId), `${categoryId} is missing from animationRegistry`);
  }
});

test('every registered module corresponds to an active catalog lesson', () => {
  for (const registryId of registryIds) {
    assert.ok(animationIds.has(registryId), `Registry includes unknown module ${registryId}`);
  }
});

test('every learning path node and track item exists in catalog', () => {
  for (const track of curriculumTracks) {
    for (const animationId of track.animationIds) {
      assert.ok(animationIds.has(animationId), `Track ${track.id} references unknown animation ${animationId}`);
    }
  }

  for (const pathConfig of HUB_LEARNING_PATHS) {
    for (const animationId of pathConfig.nodes) {
      assert.ok(animationIds.has(animationId), `Learning path ${pathConfig.id} references unknown animation ${animationId}`);
    }
  }
});

test('every prerequisite id references an existing animation', () => {
  for (const animation of allAnimations) {
    for (const prerequisiteId of animation.prerequisites || []) {
      assert.ok(animationIds.has(prerequisiteId), `${animation.id} prerequisite ${prerequisiteId} does not exist`);
    }
  }
});

test('all assessment entries target existing lessons and valid quiz answers', () => {
  const assessmentLessonIds = new Set(Object.keys(lessonAssessments));
  for (const lessonId of assessmentLessonIds) {
    assert.ok(animationIds.has(lessonId), `Assessment targets missing lesson ${lessonId}`);

    const assessment = lessonAssessments[lessonId];
    for (const question of assessment.quiz || []) {
      assert.ok(question.id, `${lessonId} has quiz item without id`);
      assert.ok(question.prompt && /\S/.test(question.prompt), `${lessonId} quiz ${question.id} has empty prompt`);
      assert.ok(Array.isArray(question.choices), `${lessonId} quiz ${question.id} has missing choices`);
      assert.ok(question.choices.length >= 2, `${lessonId} quiz ${question.id} needs at least 2 choices`);
      assert.ok(Number.isInteger(question.answerIndex), `${lessonId} quiz ${question.id} answerIndex must be an integer`);
      assert.ok(question.answerIndex >= 0, `${lessonId} quiz ${question.id} answerIndex must be >= 0`);
      assert.ok(question.answerIndex < question.choices.length, `${lessonId} quiz ${question.id} answerIndex out of range`);
    }

    for (const lab of assessment.labs || []) {
      assert.ok(lab.id, `${lessonId} has lab without id`);
      assert.ok(lab.title && /\S/.test(lab.title), `${lessonId} lab ${lab.id} has empty title`);
      assert.ok(lab.prompt && /\S/.test(lab.prompt), `${lessonId} lab ${lab.id} has empty prompt`);
      assert.ok(
        lab.successCriteria && /\S/.test(lab.successCriteria),
        `${lessonId} lab ${lab.id} has empty successCriteria`,
      );
    }
  }
});

test('every lazy-loaded registry module has a resolvable animation entry file', () => {
  const entryExtensions = ['.js', '.jsx', '.ts', '.tsx'];

  for (const animationId of availableAnimations) {
    const dir = path.join(animationsDir, animationId);
    assert.ok(fs.existsSync(dir), `${animationId} registry directory missing`);
    const hasEntry = entryExtensions.some((extension) => fs.existsSync(path.join(dir, `index${extension}`)));
    assert.ok(hasEntry, `${animationId} should export an index.* file for lazy loading`);
  }
});
