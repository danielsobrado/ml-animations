import test from 'node:test';
import assert from 'node:assert/strict';

import { allAnimations, categories } from './animations.js';
import { createLearningModel } from './animationLearning.js';
import {
  GLOSSARY_IDS_BY_CATEGORY,
  getGlossaryTermsForCategory,
  glossaryTerms,
} from './glossaryRepository.js';

const MOJIBAKE_SEQUENCES = ['Ã', 'Â', 'â€', 'âˆ', 'â†', 'â–', 'Î', 'Ï', 'ï¿½'];
const DEPTH_REQUIRED_TERMS = [
  'calibration',
  'grouped-query-attention',
  'multi-head-latent-attention',
  'kv-cache',
  'long-context',
  'effective-context',
  'goodput',
  'grpo',
  'best-of-n',
  'prompt-injection',
  'safe-success-rate',
];

test('every catalog category has valid glossary terms', () => {
  const glossaryIds = new Set(glossaryTerms.map((term) => term.id));

  for (const category of categories) {
    const mappedIds = GLOSSARY_IDS_BY_CATEGORY[category.id] || [];
    assert.ok(mappedIds.length > 0, `${category.id} needs glossary terms`);

    for (const termId of mappedIds) {
      assert.ok(glossaryIds.has(termId), `${category.id} references unknown glossary term ${termId}`);
    }

    assert.ok(getGlossaryTermsForCategory(category.id).length > 0, `${category.id} resolves glossary terms`);
  }
});

test('glossary slugs are unique and route-safe', () => {
  const slugs = new Set();

  for (const term of glossaryTerms) {
    assert.match(term.slug, /^[a-z0-9-]+$/, `${term.id} has an invalid slug`);
    assert.equal(slugs.has(term.slug), false, `duplicate glossary slug ${term.slug}`);
    slugs.add(term.slug);
  }
});

test('glossary text stays free of common mojibake sequences', () => {
  const glossaryText = JSON.stringify(glossaryTerms);

  for (const sequence of MOJIBAKE_SEQUENCES) {
    assert.equal(glossaryText.includes(sequence), false, `glossary contains mojibake sequence ${sequence}`);
  }
});

test('glossary graph links resolve to known terms or lessons', () => {
  const glossaryIds = new Set(glossaryTerms.map((term) => term.id));
  const lessonIds = new Set(allAnimations.map((animation) => animation.id));

  for (const term of glossaryTerms) {
    for (const field of ['related', 'usedIn', 'confusedWith', 'prerequisiteFor']) {
      for (const id of term[field] || []) {
        assert.ok(
          glossaryIds.has(id) || lessonIds.has(id),
          `${term.id}.${field} references unknown id ${id}`,
        );
      }
    }
  }
});

test('advanced glossary entries include depth scaffolding', () => {
  for (const termId of DEPTH_REQUIRED_TERMS) {
    const term = glossaryTerms.find((entry) => entry.id === termId);
    assert.ok(term, `${termId} should exist`);
    assert.ok(term.intuitions?.plain, `${termId} needs a plain intuition`);
    assert.ok(term.intuitions?.math, `${termId} needs a math intuition`);
    assert.ok(term.intuitions?.systems, `${termId} needs a systems intuition`);
    assert.ok(term.intuitions?.failure, `${termId} needs a failure intuition`);
    assert.ok(term.minimalExample, `${termId} needs a minimal example`);
    assert.ok(term.boundary, `${termId} needs a boundary of validity`);
    assert.ok(term.comparisonNote, `${termId} needs a comparison note`);
  }
});

test('advanced lessons expose meaningful glossary coverage', () => {
  const advancedCategoryIds = new Set([
    'frontier-llms',
    'advanced-models',
    'model-reliability',
    'experimentation-causal-ml',
  ]);

  const advancedLessons = allAnimations.filter((animation) => advancedCategoryIds.has(animation.categoryId));

  for (const animation of advancedLessons) {
    const model = createLearningModel(animation, allAnimations);
    assert.ok(model.glossary.length >= 5, `${animation.id} needs at least 5 glossary terms`);
  }
});

test('frontier llm modules expose dense lesson glossary coverage', () => {
  const frontierLessons = allAnimations.filter((animation) => animation.categoryId === 'frontier-llms');

  assert.ok(frontierLessons.length >= 12);

  for (const animation of frontierLessons) {
    const model = createLearningModel(animation, allAnimations);
    assert.ok(model.glossary.length >= 20, `${animation.id} needs at least 20 frontier glossary terms`);
    assert.ok(
      model.glossary.filter((term) => term.paperSignals?.length > 0).length >= 5,
      `${animation.id} needs paper-anchor glossary terms`,
    );
  }
});
