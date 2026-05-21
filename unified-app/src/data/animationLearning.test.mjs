import test from 'node:test';
import assert from 'node:assert/strict';

import { allAnimations, curriculumBacklog, curriculumTracks, getAnimationById } from './animations.js';
import {
  PRIORITY_ASSESSMENT_LESSON_IDS,
  getAssessmentStats,
  lessonAssessments,
} from './lessonAssessments.js';
import { isAssessmentComplete } from './learningProgress.js';
import {
  CARD_TYPES,
  LEARNING_CARD_OVERRIDES,
  MATH_CONTROLS,
  createLearningModel,
} from './animationLearning.js';
import { getGlossaryTerm, glossaryTerms } from './glossaryRepository.js';
import { isAnimationAvailable } from '../animations/index.js';

test('createLearningModel gives every animation the uniform learning shell contract', () => {
  for (const animation of allAnimations) {
    const model = createLearningModel(animation, allAnimations);

    assert.equal(model.conceptName, animation.name);
    assert.ok(model.headlineEquation.latex);
    assert.ok(model.chips.difficulty);
    assert.ok(model.chips.prereq);

    assert.deepEqual(
      model.learningCards.slice(0, CARD_TYPES.length).map((card) => card.type),
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
  assert.equal(getGlossaryTerm('logits').category, 'Neural Networks');
  assert.equal(getGlossaryTerm('temperature').category, 'Neural Networks');
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

test('core ml gap topics are promoted from backlog into active guided lessons', () => {
  const activeCoreMlIds = [
    'train-validation-test-split',
    'cross-validation',
    'feature-scaling-preprocessing',
    'overfitting',
    'logistic-regression',
    'classification-metrics',
    'regularization',
    'knn-naive-bayes-svm',
  ];
  const activeIds = new Set(allAnimations.map((animation) => animation.id));
  const backlogIds = new Set(curriculumBacklog.map((topic) => topic.id));
  const coreMlTrack = curriculumTracks.find((track) => track.id === 'core-ml');

  assert.ok(coreMlTrack, 'core-ml track should exist');

  for (const animationId of activeCoreMlIds) {
    const animation = getAnimationById(animationId);

    assert.ok(activeIds.has(animationId), `${animationId} should be an active route`);
    assert.ok(animation, `${animationId} needs metadata`);
    assert.equal(animation.categoryId, 'core-ml');
    assert.ok(animation.trackIds.includes('core-ml'), `${animationId} should be in core-ml track`);
    assert.ok(coreMlTrack.animationIds.includes(animationId), `${animationId} should appear in core-ml sequence`);
    assert.ok(!backlogIds.has(animationId), `${animationId} should no longer be backlog`);
    assert.ok(animation.learningObjectives.length >= 3, `${animationId} needs course-grade objectives`);
    assert.match(animation.commonMisconception, /\S/);
  }

  assert.ok(
    coreMlTrack.animationIds.indexOf('train-validation-test-split') <
      coreMlTrack.animationIds.indexOf('cross-validation'),
    'basic holdout splitting should come before cross-validation',
  );
  assert.ok(
    coreMlTrack.animationIds.indexOf('cross-validation') <
      coreMlTrack.animationIds.indexOf('logistic-regression'),
    'leakage-aware validation should come before model-specific classification',
  );
  assert.ok(
    coreMlTrack.animationIds.indexOf('classification-metrics') <
      coreMlTrack.animationIds.indexOf('regularization'),
    'evaluation should come before regularization decisions',
  );
});

test('computation graph and backpropagation is an active neural-network bridge lesson', () => {
  const animation = getAnimationById('computation-graph-backprop');
  const backlogIds = new Set(curriculumBacklog.map((topic) => topic.id));
  const neuralTrack = curriculumTracks.find((track) => track.id === 'neural-networks');

  assert.ok(animation, 'backprop lesson should be an active catalog item');
  assert.equal(animation.categoryId, 'neural-networks');
  assert.ok(animation.trackIds.includes('neural-networks'));
  assert.ok(neuralTrack.animationIds.includes('computation-graph-backprop'));
  assert.ok(!backlogIds.has('computation-graph-backprop'));
  assert.ok(isAnimationAvailable('computation-graph-backprop'));
  assert.deepEqual(animation.prerequisites, ['linear-regression', 'gradient-descent', 'relu']);
  assert.ok(animation.learningObjectives.length >= 4);
  assert.match(animation.commonMisconception, /chain rule|local derivative/i);

  assert.ok(
    neuralTrack.animationIds.indexOf('cross-entropy') <
      neuralTrack.animationIds.indexOf('computation-graph-backprop'),
    'losses should come before backpropagation',
  );
  assert.ok(
    neuralTrack.animationIds.indexOf('computation-graph-backprop') <
      neuralTrack.animationIds.indexOf('gradient-problems'),
    'backprop should unlock gradient stability topics',
  );
});

test('logistic regression metadata matches the sigmoid binary-classification lesson', () => {
  const animation = getAnimationById('logistic-regression');

  assert.deepEqual(animation.prerequisites, ['linear-regression']);
  assert.match(animation.learningObjectives.join(' '), /sigmoid/i);
});

test('optimizers are promoted into the neural-network training path', () => {
  const animation = getAnimationById('optimizers');
  const backlogIds = new Set(curriculumBacklog.map((topic) => topic.id));
  const neuralTrack = curriculumTracks.find((track) => track.id === 'neural-networks');

  assert.ok(animation, 'optimizers lesson should be active');
  assert.equal(animation.categoryId, 'neural-networks');
  assert.ok(animation.trackIds.includes('neural-networks'));
  assert.ok(neuralTrack.animationIds.includes('optimizers'));
  assert.ok(!backlogIds.has('optimizers'));
  assert.ok(isAnimationAvailable('optimizers'));
  assert.deepEqual(animation.prerequisites, ['gradient-descent', 'computation-graph-backprop']);
  assert.match(animation.learningObjectives.join(' '), /SGD|momentum|Adam|mini-batch/i);
  assert.match(animation.commonMisconception, /Adam|learning rate|batch/i);

  assert.ok(
    neuralTrack.animationIds.indexOf('computation-graph-backprop') <
      neuralTrack.animationIds.indexOf('optimizers'),
    'backpropagation should explain gradients before optimizer variants',
  );
  assert.ok(
    neuralTrack.animationIds.indexOf('optimizers') <
      neuralTrack.animationIds.indexOf('gradient-problems'),
    'optimizer behavior should come before gradient stability failure modes',
  );
});

test('PCA is promoted into the foundations path as a variance projection lesson', () => {
  const animation = getAnimationById('pca');
  const backlogIds = new Set(curriculumBacklog.map((topic) => topic.id));
  const foundationsTrack = curriculumTracks.find((track) => track.id === 'foundations');

  assert.ok(animation, 'PCA lesson should be active');
  assert.equal(animation.categoryId, 'math-fundamentals');
  assert.ok(animation.trackIds.includes('foundations'));
  assert.ok(foundationsTrack.animationIds.includes('pca'));
  assert.ok(!backlogIds.has('pca'));
  assert.ok(isAnimationAvailable('pca'));
  assert.deepEqual(animation.prerequisites, ['matrix-multiplication', 'expected-value-variance']);
  assert.match(animation.learningObjectives.join(' '), /center|covariance|variance/i);
  assert.match(animation.commonMisconception, /not supervised/i);

  assert.ok(
    foundationsTrack.animationIds.indexOf('expected-value-variance') <
      foundationsTrack.animationIds.indexOf('pca'),
    'variance should come before PCA',
  );
});

test('k-means is promoted into Core ML as an unsupervised clustering lesson', () => {
  const animation = getAnimationById('k-means');
  const backlogIds = new Set(curriculumBacklog.map((topic) => topic.id));
  const coreMlTrack = curriculumTracks.find((track) => track.id === 'core-ml');

  assert.ok(animation, 'k-means lesson should be active');
  assert.equal(animation.categoryId, 'core-ml');
  assert.ok(animation.trackIds.includes('core-ml'));
  assert.ok(coreMlTrack.animationIds.includes('k-means'));
  assert.ok(!backlogIds.has('k-means'));
  assert.ok(isAnimationAvailable('k-means'));
  assert.deepEqual(animation.prerequisites, ['pca', 'feature-scaling-preprocessing']);
  assert.match(animation.learningObjectives.join(' '), /centroid|inertia/i);
  assert.match(animation.commonMisconception, /choosing k|number of groups/i);

  assert.ok(
    coreMlTrack.animationIds.indexOf('feature-scaling-preprocessing') <
      coreMlTrack.animationIds.indexOf('k-means'),
    'scaling should precede distance-based clustering',
  );
});

test('feature scaling and preprocessing is promoted into Core ML before distance-based models', () => {
  const animation = getAnimationById('feature-scaling-preprocessing');
  const backlogIds = new Set(curriculumBacklog.map((topic) => topic.id));
  const coreMlTrack = curriculumTracks.find((track) => track.id === 'core-ml');

  assert.ok(animation, 'feature scaling lesson should be active');
  assert.equal(animation.categoryId, 'core-ml');
  assert.ok(animation.trackIds.includes('core-ml'));
  assert.ok(coreMlTrack.animationIds.includes('feature-scaling-preprocessing'));
  assert.ok(!backlogIds.has('feature-scaling-preprocessing'));
  assert.ok(isAnimationAvailable('feature-scaling-preprocessing'));
  assert.deepEqual(animation.prerequisites, ['cross-validation']);
  assert.match(animation.learningObjectives.join(' '), /standardization|min-max|robust|training data/i);
  assert.match(animation.commonMisconception, /leaks|validation|test/i);

  assert.ok(
    coreMlTrack.animationIds.indexOf('cross-validation') <
      coreMlTrack.animationIds.indexOf('feature-scaling-preprocessing'),
    'leakage-aware validation should precede preprocessing fit-scope decisions',
  );
});

test('ROC and precision-recall curves are promoted into Core ML evaluation', () => {
  const animation = getAnimationById('roc-pr-curves');
  const coreMlTrack = curriculumTracks.find((track) => track.id === 'core-ml');

  assert.ok(animation, 'ROC / PR curves lesson should be active');
  assert.equal(animation.categoryId, 'core-ml');
  assert.ok(animation.trackIds.includes('core-ml'));
  assert.ok(coreMlTrack.animationIds.includes('roc-pr-curves'));
  assert.ok(isAnimationAvailable('roc-pr-curves'));
  assert.deepEqual(animation.prerequisites, ['classification-metrics']);
  assert.match(animation.learningObjectives.join(' '), /threshold|ROC|precision-recall/i);
  assert.match(animation.commonMisconception, /ROC-AUC|threshold/i);

  assert.ok(
    coreMlTrack.animationIds.indexOf('classification-metrics') <
      coreMlTrack.animationIds.indexOf('roc-pr-curves'),
    'confusion-matrix metrics should precede curve-based threshold sweeps',
  );
});

test('tree ensembles are promoted into Core ML as a guided model-family lesson', () => {
  const animation = getAnimationById('tree-ensembles');
  const backlogIds = new Set(curriculumBacklog.map((topic) => topic.id));
  const coreMlTrack = curriculumTracks.find((track) => track.id === 'core-ml');

  assert.ok(animation, 'tree ensembles lesson should be active');
  assert.equal(animation.categoryId, 'core-ml');
  assert.ok(animation.trackIds.includes('core-ml'));
  assert.ok(coreMlTrack.animationIds.includes('tree-ensembles'));
  assert.ok(!backlogIds.has('tree-ensembles'));
  assert.ok(isAnimationAvailable('tree-ensembles'));
  assert.deepEqual(animation.prerequisites, ['overfitting', 'classification-metrics', 'knn-naive-bayes-svm']);
  assert.match(animation.learningObjectives.join(' '), /decision tree|random forests|gradient boosting/i);
  assert.match(animation.commonMisconception, /deeper|overfit|variance/i);

  assert.ok(
    coreMlTrack.animationIds.indexOf('knn-naive-bayes-svm') <
      coreMlTrack.animationIds.indexOf('tree-ensembles'),
    'classical single-model baselines should precede ensemble capacity tradeoffs',
  );
});

test('kNN, Naive Bayes, and SVM are promoted from backlog into Core ML', () => {
  const animation = getAnimationById('knn-naive-bayes-svm');
  const backlogIds = new Set(curriculumBacklog.map((topic) => topic.id));
  const coreMlTrack = curriculumTracks.find((track) => track.id === 'core-ml');

  assert.ok(animation, 'classical classifier comparison lesson should be active');
  assert.equal(animation.categoryId, 'core-ml');
  assert.ok(animation.trackIds.includes('core-ml'));
  assert.ok(coreMlTrack.animationIds.includes('knn-naive-bayes-svm'));
  assert.ok(!backlogIds.has('knn-naive-bayes-svm'));
  assert.ok(isAnimationAvailable('knn-naive-bayes-svm'));
  assert.deepEqual(animation.prerequisites, ['feature-scaling-preprocessing', 'classification-metrics']);
  assert.match(animation.learningObjectives.join(' '), /kNN|Naive Bayes|SVM|margin/i);
  assert.match(animation.commonMisconception, /scale|independence|margin/i);

  assert.ok(
    coreMlTrack.animationIds.indexOf('feature-scaling-preprocessing') <
      coreMlTrack.animationIds.indexOf('knn-naive-bayes-svm'),
    'scaling should precede scale-sensitive classical classifiers',
  );
});

test('transformer token generation is promoted into the NLP transformer path', () => {
  const animation = getAnimationById('transformer-token-generation');
  const backlogIds = new Set(curriculumBacklog.map((topic) => topic.id));
  const transformerTrack = curriculumTracks.find((track) => track.id === 'nlp-transformers');

  assert.ok(animation, 'token generation lesson should be active');
  assert.equal(animation.categoryId, 'transformers');
  assert.ok(animation.trackIds.includes('nlp-transformers'));
  assert.ok(transformerTrack.animationIds.includes('transformer-token-generation'));
  assert.ok(!backlogIds.has('transformer-token-generation'));
  assert.ok(isAnimationAvailable('transformer-token-generation'));
  assert.deepEqual(animation.prerequisites, ['transformer', 'tokenization', 'softmax']);
  assert.match(animation.learningObjectives.join(' '), /temperature|top-k|top-p|KV cache/i);

  assert.ok(
    transformerTrack.animationIds.indexOf('gpt2-comprehensive') <
      transformerTrack.animationIds.indexOf('transformer-token-generation'),
    'GPT-style architecture should come before generation loop practice',
  );
  assert.ok(
    transformerTrack.animationIds.indexOf('transformer-token-generation') <
      transformerTrack.animationIds.indexOf('kv-cache'),
    'generation loop should motivate the dedicated KV cache lesson',
  );
});

test('attention masks are promoted into the transformer path before full architecture', () => {
  const animation = getAnimationById('attention-masks');
  const transformer = getAnimationById('transformer');
  const backlogIds = new Set(curriculumBacklog.map((topic) => topic.id));
  const transformerTrack = curriculumTracks.find((track) => track.id === 'nlp-transformers');

  assert.ok(animation, 'attention masks lesson should be active');
  assert.equal(animation.categoryId, 'transformers');
  assert.ok(animation.trackIds.includes('nlp-transformers'));
  assert.ok(transformerTrack.animationIds.includes('attention-masks'));
  assert.ok(!backlogIds.has('attention-masks'));
  assert.ok(isAnimationAvailable('attention-masks'));
  assert.deepEqual(animation.prerequisites, ['self-attention', 'tokenization']);
  assert.ok(transformer.prerequisites.includes('attention-masks'));
  assert.match(animation.learningObjectives.join(' '), /causal|padding|bidirectional|cross-attention/i);
  assert.match(animation.commonMisconception, /masked-language|visibility/i);

  assert.ok(
    transformerTrack.animationIds.indexOf('self-attention') <
      transformerTrack.animationIds.indexOf('attention-masks'),
    'self-attention mechanics should precede mask rules',
  );
  assert.ok(
    transformerTrack.animationIds.indexOf('attention-masks') <
      transformerTrack.animationIds.indexOf('transformer'),
    'mask rules should precede full transformer architecture',
  );
});

test('RAG retrieval evaluation is promoted into the generative AI path', () => {
  const animation = getAnimationById('rag-retrieval-evaluation');
  const backlogIds = new Set(curriculumBacklog.map((topic) => topic.id));
  const generativeTrack = curriculumTracks.find((track) => track.id === 'generative-ai');

  assert.ok(animation, 'RAG retrieval evaluation lesson should be active');
  assert.equal(animation.categoryId, 'advanced-models');
  assert.ok(animation.trackIds.includes('generative-ai'));
  assert.ok(generativeTrack.animationIds.includes('rag-retrieval-evaluation'));
  assert.ok(!backlogIds.has('rag-retrieval-evaluation'));
  assert.ok(isAnimationAvailable('rag-retrieval-evaluation'));
  assert.deepEqual(animation.prerequisites, ['rag', 'embeddings', 'cosine-similarity']);
  assert.match(animation.learningObjectives.join(' '), /chunk|rerank|recall@k|nDCG/i);
  assert.match(animation.commonMisconception, /missing evidence|right chunks|context/i);

  assert.ok(
    generativeTrack.animationIds.indexOf('rag') <
      generativeTrack.animationIds.indexOf('rag-retrieval-evaluation'),
    'broad RAG introduction should precede retrieval evaluation',
  );
});

test('lesson assessments provide backed quiz and lab counts for priority lessons', () => {
  const animationIds = new Set(allAnimations.map((animation) => animation.id));
  const stats = getAssessmentStats(lessonAssessments);

  assert.equal(stats.totalQuizQuestions, PRIORITY_ASSESSMENT_LESSON_IDS.length * 2);
  assert.equal(stats.totalLabs, PRIORITY_ASSESSMENT_LESSON_IDS.length);

  for (const lessonId of PRIORITY_ASSESSMENT_LESSON_IDS) {
    const assessment = lessonAssessments[lessonId];

    assert.ok(animationIds.has(lessonId), `${lessonId} should be an active lesson`);
    assert.equal(assessment.quiz.length, 2, `${lessonId} needs two seeded quiz questions`);
    assert.equal(assessment.labs.length, 1, `${lessonId} needs one seeded lab`);

    for (const question of assessment.quiz) {
      assert.ok(question.id);
      assert.ok(question.prompt);
      assert.ok(question.choices.length >= 2);
      assert.ok(question.answerIndex >= 0);
      assert.ok(question.answerIndex < question.choices.length);
      assert.ok(question.explanation);
    }

    for (const lab of assessment.labs) {
      assert.ok(lab.id);
      assert.ok(lab.title);
      assert.ok(lab.prompt);
      assert.ok(lab.successCriteria);
    }
  }
});

test('assessment completion requires all seeded quiz and lab items', () => {
  const assessment = lessonAssessments['logistic-regression'];

  assert.equal(isAssessmentComplete(assessment, {}), false);
  assert.equal(
    isAssessmentComplete(assessment, {
      quiz: {
        'sigmoid-role': { correct: true },
        'threshold-tradeoff': { correct: true },
      },
      labs: {
        'threshold-flips': true,
      },
    }),
    true,
  );
});

test('createLearningModel uses curriculum metadata for one-click navigation context', () => {
  const animation = getAnimationById('self-attention');
  const model = createLearningModel(animation, allAnimations);

  assert.equal(model.mindmap.current.label, 'Self-Attention');
  assert.ok(model.mindmap.prereqs.some((node) => node.id === 'matrix-multiplication'));
  assert.ok(model.mindmap.prereqs.some((node) => node.id === 'softmax'));
  assert.ok(model.mindmap.next.some((node) => node.id === 'attention-masks'));
  assert.equal(model.chips.difficulty, 'intermediate');
  assert.equal(model.chips.prereq, 'Matrix Multiplication, Softmax');
});

test('softmax adds formal theorem and marginalia learning card flavours', () => {
  const animation = getAnimationById('softmax');
  const model = createLearningModel(animation, allAnimations);

  assert.deepEqual(
    model.learningCards.map((card) => card.type),
    [...CARD_TYPES.map((cardType) => cardType.id), 'theorem', 'marginalia'],
  );

  const theorem = model.learningCards.find((card) => card.type === 'theorem');
  const marginalia = model.learningCards.find((card) => card.type === 'marginalia');

  assert.match(theorem.title, /Theorem 3\.1/);
  assert.match(theorem.body, /translation invariance/i);
  assert.match(theorem.equation, /softmax/);
  assert.match(marginalia.body, /overflow/i);
  assert.ok(model.glossary.some((term) => term.id === 'logits'));
  assert.ok(model.glossary.some((term) => term.id === 'temperature'));
});

test('priority lessons use lesson-specific learning card overrides', () => {
  const genericFragments = [
    'change one value, token, state, or vector',
    'Read the stage as a flow',
    'headline equation compresses',
  ];

  for (const lessonId of PRIORITY_ASSESSMENT_LESSON_IDS) {
    const overrides = LEARNING_CARD_OVERRIDES[lessonId];
    const model = createLearningModel(getAnimationById(lessonId), allAnimations);

    assert.ok(overrides, `${lessonId} needs card overrides`);

    for (const cardType of CARD_TYPES) {
      const override = overrides[cardType.id];
      const rendered = model.learningCards.find((card) => card.type === cardType.id);

      assert.ok(override?.body, `${lessonId} needs ${cardType.id} override copy`);
      assert.equal(rendered.body, override.body);
      assert.ok(override.body.length > 40, `${lessonId} ${cardType.id} copy is too thin`);

      for (const fragment of genericFragments) {
        assert.ok(!override.body.includes(fragment), `${lessonId} still uses generic ${cardType.id} copy`);
      }
    }
  }
});
