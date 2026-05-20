import test from 'node:test';
import assert from 'node:assert/strict';

import { allAnimations, curriculumBacklog, curriculumTracks, getAnimationById } from './animations.js';
import {
  CARD_TYPES,
  MATH_CONTROLS,
  createLearningModel,
} from './animationLearning.js';
import { getGlossaryTerm, glossaryTerms } from './glossaryRepository.js';

test('createLearningModel gives every animation the uniform learning shell contract', () => {
  for (const animation of allAnimations) {
    const model = createLearningModel(animation, allAnimations);

    assert.equal(model.conceptName, animation.name);
    assert.ok(model.headlineEquation.latex);
    assert.ok(model.chips.difficulty);
    assert.ok(model.chips.prereq);

    assert.deepEqual(
      model.learningCards.map((card) => card.type),
      CARD_TYPES.map((cardType) => cardType.id),
    );

    assert.deepEqual(
      model.controls.map((control) => control.sigil),
      MATH_CONTROLS.map((control) => control.sigil),
    );

    assert.equal(model.mindmap.current.id, animation.id);
    assert.ok(model.mindmap.prereqs.length >= 1);
    assert.ok(model.mindmap.next.length >= 1);
    assert.ok(model.glossary.length >= 8);

    for (const term of model.glossary) {
      const canonical = getGlossaryTerm(term.id);
      assert.ok(canonical);
      assert.equal(term.term, canonical.term);
      assert.equal(term.definition, canonical.definition);
      assert.equal(term.href, `/glossary/${canonical.slug}`);
      assert.ok(term.image.src.startsWith('data:image/svg+xml'));
      assert.ok(term.image.alt.includes(canonical.term));
    }
  }
});

test('central glossary repository exposes reusable image-backed term pages', () => {
  assert.ok(glossaryTerms.length >= 30);

  for (const term of glossaryTerms) {
    assert.ok(term.id);
    assert.ok(term.slug);
    assert.equal(term.href, `/glossary/${term.slug}`);
    assert.ok(term.definition);
    assert.ok(term.image.src.startsWith('data:image/svg+xml'));
    assert.ok(term.image.alt);
  }

  assert.equal(getGlossaryTerm('attention').href, '/glossary/attention');
  assert.equal(getGlossaryTerm('query').category, 'Transformers');
});

test('every active animation exposes curriculum metadata', () => {
  const validDifficulties = new Set(['beginner', 'intermediate', 'advanced']);
  const animationIds = new Set(allAnimations.map((animation) => animation.id));

  for (const animation of allAnimations) {
    assert.ok(
      validDifficulties.has(animation.difficulty),
      `${animation.id} has invalid difficulty`,
    );
    assert.ok(Number.isInteger(animation.estimatedMinutes), `${animation.id} needs minutes`);
    assert.ok(animation.estimatedMinutes >= 5, `${animation.id} minutes too low`);
    assert.ok(Array.isArray(animation.learningObjectives), `${animation.id} needs objectives`);
    assert.ok(animation.learningObjectives.length >= 2, `${animation.id} needs 2+ objectives`);
    assert.ok(animation.commonMisconception, `${animation.id} needs a misconception`);
    assert.ok(Array.isArray(animation.trackIds), `${animation.id} needs tracks`);
    assert.ok(animation.trackIds.length >= 1, `${animation.id} needs at least one track`);

    for (const prerequisiteId of animation.prerequisites) {
      assert.ok(animationIds.has(prerequisiteId), `${animation.id} has unknown prereq ${prerequisiteId}`);
    }
  }
});

test('curriculum tracks reference only active animations and backlog topics stay inactive', () => {
  const animationIds = new Set(allAnimations.map((animation) => animation.id));
  const seenTrackIds = new Set();

  assert.equal(curriculumTracks.length, 6);
  for (const track of curriculumTracks) {
    assert.ok(track.id);
    assert.ok(!seenTrackIds.has(track.id), `${track.id} is duplicated`);
    seenTrackIds.add(track.id);
    assert.ok(track.title);
    assert.ok(track.description);
    assert.ok(track.animationIds.length >= 2, `${track.id} needs active animations`);

    for (const animationId of track.animationIds) {
      assert.ok(animationIds.has(animationId), `${track.id} references unknown ${animationId}`);
      assert.ok(
        getAnimationById(animationId).trackIds.includes(track.id),
        `${animationId} should include ${track.id}`,
      );
    }
  }

  for (const topic of curriculumBacklog) {
    assert.ok(topic.id);
    assert.ok(topic.title);
    assert.ok(topic.trackId);
    assert.ok(seenTrackIds.has(topic.trackId), `${topic.id} has unknown backlog track`);
    assert.ok(!animationIds.has(topic.id), `${topic.id} should not be an active route yet`);
  }
});

test('createLearningModel uses curriculum metadata for one-click navigation context', () => {
  const animation = getAnimationById('self-attention');
  const model = createLearningModel(animation, allAnimations);

  assert.equal(model.mindmap.current.label, 'Self-Attention');
  assert.ok(model.mindmap.prereqs.some((node) => node.id === 'matrix-multiplication'));
  assert.ok(model.mindmap.prereqs.some((node) => node.id === 'softmax'));
  assert.ok(model.mindmap.next.some((node) => node.id === 'positional-encoding'));
  assert.equal(model.chips.difficulty, 'intermediate');
  assert.equal(model.chips.prereq, 'Matrix Multiplication, Softmax');
});
