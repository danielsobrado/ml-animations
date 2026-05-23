import test from 'node:test';
import assert from 'node:assert/strict';

import { allAnimations } from './animations.js';
import { getCurriculumDepth, conceptComparisons, failureGalleryItems, paperReadingSignals, caveatBoxes } from './curriculumDepth.js';
import { glossaryTerms } from './glossaryRepository.js';
import { SCENARIO_QUESTIONS_BY_LESSON } from './scenarioQuestions.js';

const lessonIds = new Set(allAnimations.map((animation) => animation.id));
const glossaryIds = new Set(glossaryTerms.map((term) => term.id));

function assertLessonRefs(items, collectionName) {
  for (const item of items) {
    assert.ok(item.id, `${collectionName} item needs an id`);
    assert.ok(item.lessonIds?.length > 0, `${item.id} needs lessonIds`);

    for (const lessonId of item.lessonIds) {
      assert.ok(lessonIds.has(lessonId), `${item.id} references unknown lesson ${lessonId}`);
    }

    for (const termId of item.glossaryTerms || []) {
      assert.ok(glossaryIds.has(termId), `${item.id} references unknown glossary term ${termId}`);
    }
  }
}

test('curriculum depth data references active lessons and glossary terms', () => {
  assertLessonRefs(conceptComparisons, 'comparison');
  assertLessonRefs(failureGalleryItems, 'failure');
  assertLessonRefs(paperReadingSignals, 'paper signal');
  assertLessonRefs(caveatBoxes, 'caveat');
});

test('frontier lessons expose comparison, caveat, failure, or paper-reading depth panels', () => {
  const frontierLessons = allAnimations.filter((animation) => animation.categoryId === 'frontier-llms');
  assert.ok(frontierLessons.length >= 12);

  for (const animation of frontierLessons) {
    const depth = getCurriculumDepth(animation);
    const panelCount = depth.comparisons.length + depth.failures.length + depth.paperSignals.length + depth.caveats.length;
    assert.ok(panelCount > 0, `${animation.id} needs at least one curriculum depth panel`);
  }
});

test('priority depth panels include misconception-resistant fields', () => {
  for (const comparison of conceptComparisons) {
    assert.ok(comparison.commonMistake, `${comparison.id} needs a common mistake`);
    assert.ok(comparison.diagnostic, `${comparison.id} needs a diagnostic question`);
  }

  for (const failure of failureGalleryItems) {
    assert.ok(failure.symptom, `${failure.id} needs a symptom`);
    assert.ok(failure.whyItHappens, `${failure.id} needs whyItHappens`);
    assert.ok(failure.howToDetect, `${failure.id} needs howToDetect`);
    assert.ok(failure.howToFix, `${failure.id} needs howToFix`);
  }

  for (const signal of paperReadingSignals) {
    assert.ok(signal.phrase, `${signal.id} needs a phrase`);
    assert.ok(signal.ask?.length > 0, `${signal.id} needs ask prompts`);
    assert.ok(signal.means, `${signal.id} needs means`);
    assert.ok(signal.doesNotMean, `${signal.id} needs doesNotMean`);
    assert.ok(signal.check?.length > 0, `${signal.id} needs check prompts`);
  }
});

test('curated scenario questions target active lessons and valid answers', () => {
  for (const [lessonId, questions] of Object.entries(SCENARIO_QUESTIONS_BY_LESSON)) {
    assert.ok(lessonIds.has(lessonId), `scenario questions reference unknown lesson ${lessonId}`);
    assert.ok(questions.length > 0, `${lessonId} needs at least one scenario question`);

    for (const question of questions) {
      assert.ok(question.id, `${lessonId} scenario needs an id`);
      assert.ok(question.scenario, `${question.id} needs a scenario`);
      assert.ok(question.prompt, `${question.id} needs a prompt`);
      assert.ok(question.explanation, `${question.id} needs an explanation`);
      assert.ok(question.misconceptionTested, `${question.id} needs a misconceptionTested field`);
      assert.ok(Array.isArray(question.choices) && question.choices.length >= 3, `${question.id} needs choices`);
      assert.ok(Number.isInteger(question.answerIndex), `${question.id} needs answerIndex`);
      assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `${question.id} answerIndex is out of range`);
    }
  }
});
