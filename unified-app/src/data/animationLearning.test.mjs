import test from 'node:test';
import assert from 'node:assert/strict';

import { allAnimations, curriculumBacklog, curriculumTracks, getAnimationById } from './animations.js';
import { HUB_LEARNING_PATHS } from './learningPaths.js';
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
    'data-leakage-deep-dive',
    'feature-scaling-preprocessing',
    'overfitting',
    'logistic-regression',
    'classification-metrics',
    'calibration',
    'regularization',
    'bias-variance-tradeoff',
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

test('start-here path is prerequisite-safe and includes core-flow milestones', () => {
  const startPath = HUB_LEARNING_PATHS.find((path) => path.id === 'start-here');
  assert.ok(startPath, 'start-here path should exist');

  const pathOrder = new Map(startPath.nodes.map((id, index) => [id, index]));
  for (const nodeId of startPath.nodes) {
    const animation = getAnimationById(nodeId);
    assert.ok(animation, `${nodeId} in start-here must be active`);
    assert.match(animation.name, /\S/);
  }

  assert.ok(
    startPath.nodes.includes('cross-validation'),
    'start-here should include cross-validation before deeper classification modeling',
  );
  assert.ok(
    startPath.nodes.includes('data-leakage-deep-dive'),
    'start-here should include data leakage before preprocessing or modeling',
  );
  assert.ok(
    startPath.nodes.includes('feature-scaling-preprocessing'),
    'start-here should include preprocessing fit-scope before dependent workflows',
  );
  assert.ok(startPath.nodes.includes('neural-network'), 'start-here should include neural-network before ReLU');

  for (const nodeId of startPath.nodes) {
    const index = pathOrder.get(nodeId);
    const animation = getAnimationById(nodeId);
    for (const prerequisite of animation.prerequisites) {
      const prereqIndex = pathOrder.get(prerequisite);
      if (Number.isInteger(prereqIndex)) {
        assert.ok(
          prereqIndex < index,
          `${nodeId} requires ${prerequisite} before it in start-here`,
        );
      }
    }
  }

  assert.ok(
    pathOrder.get('gradient-descent') < pathOrder.get('neural-network'),
    'gradient descent should precede the neural-network review before backprop',
  );
  assert.ok(
    pathOrder.get('neural-network') < pathOrder.get('relu'),
    'neural-network basics should precede ReLU in start-here',
  );
  assert.ok(
    pathOrder.get('classification-metrics') < pathOrder.get('roc-pr-curves'),
    'ROC/PR should come after classification-metrics in start-here',
  );
  assert.ok(
    pathOrder.get('roc-pr-curves') < pathOrder.get('calibration'),
    'calibration should build on ROC/PR understanding',
  );
  assert.ok(
    pathOrder.get('calibration') < pathOrder.get('overfitting'),
    'overfitting should come after classification-evaluation checkpoints',
  );
  assert.ok(
    pathOrder.get('overfitting') < pathOrder.get('bias-variance-tradeoff'),
    'bias-variance should follow overfitting',
  );
  assert.ok(
    pathOrder.get('bias-variance-tradeoff') < pathOrder.get('regularization'),
    'regularization should come after bias-variance framing',
  );
  assert.ok(
    pathOrder.get('regularization') < pathOrder.get('knn-naive-bayes-svm'),
    'classical classifiers should come after regularization in start-here',
  );
  assert.ok(
    pathOrder.get('knn-naive-bayes-svm') < pathOrder.get('tree-ensembles'),
    'tree ensembles should follow knn-naive-bayes-svm in start-here',
  );
  assert.ok(
    pathOrder.get('relu') < pathOrder.get('computation-graph-backprop'),
    'ReLU should come before computation graph backprop',
  );
  assert.ok(
    pathOrder.get('computation-graph-backprop') < pathOrder.get('optimizers'),
    'optimizers should come after backprop in start-here',
  );
  assert.ok(startPath.nodes.includes('optimizers'), 'start-here should include optimizers');
  assert.ok(
    startPath.nodes.includes('roc-pr-curves'),
    'start-here should include ROC and precision-recall for threshold interpretation',
  );
  assert.ok(
    startPath.nodes.includes('calibration'),
    'start-here should include calibration as part of classification evaluation quality',
  );
  assert.ok(startPath.nodes.includes('overfitting'), 'start-here should include overfitting');
  assert.ok(startPath.nodes.includes('bias-variance-tradeoff'), 'start-here should include bias-variance tradeoff');
  assert.ok(startPath.nodes.includes('regularization'), 'start-here should include regularization');
  assert.ok(startPath.nodes.includes('knn-naive-bayes-svm'), 'start-here should include classical classifiers');
  assert.ok(startPath.nodes.includes('tree-ensembles'), 'start-here should include tree ensembles');
});

test('LLM path includes modern inference sequencing', () => {
  const llmPath = HUB_LEARNING_PATHS.find((path) => path.id === 'llm-path');
  assert.ok(llmPath, 'llm-path should exist');

  const pathOrder = new Map(llmPath.nodes.map((id, index) => [id, index]));

  assert.ok(llmPath.nodes.includes('attention-masks'), 'llm-path should include attention mask visibility rules');
  assert.ok(
    llmPath.nodes.includes('transformer-token-generation'),
    'llm-path should include token generation loop',
  );
  assert.ok(
    llmPath.nodes.includes('sampling-strategies'),
    'llm-path should include sampling strategy lesson',
  );
  assert.ok(
    llmPath.nodes.includes('fine-tuning'),
    'llm-path should include fine-tuning methods after decoding controls',
  );
  assert.ok(
    pathOrder.get('self-attention') < pathOrder.get('attention-masks'),
    'attention masks should follow self-attention',
  );
  assert.ok(
    pathOrder.get('attention-masks') < pathOrder.get('transformer'),
    'transformer lesson should follow attention-mask setup',
  );
  assert.ok(
    pathOrder.get('transformer') < pathOrder.get('transformer-token-generation'),
    'token generation should come after transformer',
  );
  assert.ok(
    pathOrder.get('transformer-token-generation') < pathOrder.get('sampling-strategies'),
    'sampling should come after token generation',
  );
  assert.ok(
    pathOrder.get('sampling-strategies') < pathOrder.get('fine-tuning'),
    'fine-tuning should come after sampling and decoding controls',
  );
  assert.ok(
    pathOrder.get('transformer-architecture-families') < pathOrder.get('transformer-token-generation'),
    'architecture family context should come before token generation details',
  );
});

test('RAG path sequences retrieval, grounding, failures, and evaluation', () => {
  const ragPath = HUB_LEARNING_PATHS.find((path) => path.id === 'rag-path');
  assert.ok(ragPath, 'rag-path should exist');

  const pathOrder = new Map(ragPath.nodes.map((id, index) => [id, index]));
  for (const nodeId of ragPath.nodes) {
    const animation = getAnimationById(nodeId);
    assert.ok(animation, `${nodeId} in rag-path must be active`);
    for (const prerequisite of animation.prerequisites) {
      const prereqIndex = pathOrder.get(prerequisite);
      if (Number.isInteger(prereqIndex)) {
        assert.ok(
          prereqIndex < pathOrder.get(nodeId),
          `${nodeId} requires ${prerequisite} before it in rag-path`,
        );
      }
    }
  }

  assert.ok(pathOrder.get('embeddings') < pathOrder.get('cosine-similarity'));
  assert.ok(pathOrder.get('cosine-similarity') < pathOrder.get('rag'));
  assert.ok(pathOrder.get('rag') < pathOrder.get('rag-chunking-context'));
  assert.ok(pathOrder.get('rag-chunking-context') < pathOrder.get('rag-vector-indexing'));
  assert.ok(pathOrder.get('rag-vector-indexing') < pathOrder.get('rag-reranking-grounding'));
  assert.ok(pathOrder.get('rag-reranking-grounding') < pathOrder.get('rag-failure-modes'));
  assert.ok(pathOrder.get('rag-failure-modes') < pathOrder.get('rag-retrieval-evaluation'));
});

test('vision path includes active diffusion component lessons after the latent bridge', () => {
  const visionPath = HUB_LEARNING_PATHS.find((path) => path.id === 'vision-path');
  assert.ok(visionPath, 'vision-path should exist');

  const pathOrder = new Map(visionPath.nodes.map((id, index) => [id, index]));
  const diffusionComponentIds = [
    'diffusion-basics',
    'self-attention',
    'sd3-overview',
    'flow-matching',
    'clip-encoder',
    't5-encoder',
    'joint-attention',
    'dit',
  ];

  for (const nodeId of diffusionComponentIds) {
    assert.ok(visionPath.nodes.includes(nodeId), `vision-path should include ${nodeId}`);
    assert.ok(getAnimationById(nodeId), `${nodeId} in vision-path must be active`);
  }

  assert.ok(pathOrder.get('vae') < pathOrder.get('diffusion-basics'));
  assert.ok(pathOrder.get('diffusion-basics') < pathOrder.get('diffusion-vae'));
  assert.ok(pathOrder.get('diffusion-vae') < pathOrder.get('self-attention'));
  assert.ok(pathOrder.get('self-attention') < pathOrder.get('sd3-overview'));
  assert.ok(pathOrder.get('sd3-overview') < pathOrder.get('flow-matching'));
  assert.ok(pathOrder.get('flow-matching') < pathOrder.get('clip-encoder'));
  assert.ok(pathOrder.get('clip-encoder') < pathOrder.get('t5-encoder'));
  assert.ok(pathOrder.get('t5-encoder') < pathOrder.get('joint-attention'));
  assert.ok(pathOrder.get('joint-attention') < pathOrder.get('dit'));
});

test('diffusion basics bridges VAE and advanced diffusion components', () => {
  const animation = getAnimationById('diffusion-basics');
  const generativeTrack = curriculumTracks.find((track) => track.id === 'generative-ai');
  const visionPath = HUB_LEARNING_PATHS.find((path) => path.id === 'vision-path');

  assert.ok(animation, 'diffusion basics lesson should be active');
  assert.equal(animation.categoryId, 'diffusion-models');
  assert.ok(animation.trackIds.includes('generative-ai'));
  assert.ok(generativeTrack.animationIds.includes('diffusion-basics'));
  assert.ok(isAnimationAvailable('diffusion-basics'));
  assert.deepEqual(animation.prerequisites, ['vae']);
  assert.match(animation.learningObjectives.join(' '), /noise|timestep|clean/i);
  assert.match(animation.commonMisconception, /one magic step|denoising/i);
  assert.ok(
    visionPath.nodes.indexOf('vae') < visionPath.nodes.indexOf('diffusion-basics') &&
      visionPath.nodes.indexOf('diffusion-basics') < visionPath.nodes.indexOf('diffusion-vae'),
    'diffusion basics should sit between VAE and diffusion VAE',
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
  assert.deepEqual(animation.prerequisites, ['gradient-descent', 'computation-graph-backprop', 'initialization']);
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

test('initialization is promoted before optimizers in the neural-network path', () => {
  const animation = getAnimationById('initialization');
  const neuralTrack = curriculumTracks.find((track) => track.id === 'neural-networks');
  const startPath = HUB_LEARNING_PATHS.find((path) => path.id === 'start-here');

  assert.ok(animation, 'initialization lesson should be active');
  assert.equal(animation.categoryId, 'neural-networks');
  assert.ok(animation.trackIds.includes('neural-networks'));
  assert.ok(neuralTrack.animationIds.includes('initialization'));
  assert.ok(isAnimationAvailable('initialization'));
  assert.deepEqual(animation.prerequisites, ['computation-graph-backprop', 'relu']);
  assert.match(animation.learningObjectives.join(' '), /Xavier|He|variance|gradient/i);
  assert.match(animation.commonMisconception, /random noise|variance/i);

  assert.ok(
    neuralTrack.animationIds.indexOf('computation-graph-backprop') <
      neuralTrack.animationIds.indexOf('initialization'),
    'backprop should come before initialization',
  );
  assert.ok(
    neuralTrack.animationIds.indexOf('initialization') < neuralTrack.animationIds.indexOf('optimizers'),
    'initialization should come before optimizers',
  );
  assert.ok(
    startPath.nodes.indexOf('computation-graph-backprop') < startPath.nodes.indexOf('initialization') &&
      startPath.nodes.indexOf('initialization') < startPath.nodes.indexOf('optimizers'),
    'start-here should sequence initialization between backprop and optimizers',
  );
});

test('dropout and batchnorm are promoted after optimizer basics in the neural-network path', () => {
  const animation = getAnimationById('dropout-batchnorm');
  const neuralTrack = curriculumTracks.find((track) => track.id === 'neural-networks');

  assert.ok(animation, 'dropout and batchnorm lesson should be active');
  assert.equal(animation.categoryId, 'neural-networks');
  assert.ok(animation.trackIds.includes('neural-networks'));
  assert.ok(neuralTrack.animationIds.includes('dropout-batchnorm'));
  assert.ok(isAnimationAvailable('dropout-batchnorm'));
  assert.deepEqual(animation.prerequisites, ['initialization', 'optimizers']);
  assert.match(animation.learningObjectives.join(' '), /BatchNorm|dropout|inference/i);
  assert.match(animation.commonMisconception, /not interchangeable|masks units/i);

  assert.ok(
    neuralTrack.animationIds.indexOf('optimizers') < neuralTrack.animationIds.indexOf('dropout-batchnorm'),
    'optimizer basics should come before regularization layers',
  );
  assert.ok(
    neuralTrack.animationIds.indexOf('dropout-batchnorm') < neuralTrack.animationIds.indexOf('gradient-problems'),
    'dropout and batchnorm should precede later gradient stability lessons',
  );
});

test('training loop dynamics bridge optimizer steps and validation behavior', () => {
  const animation = getAnimationById('training-loop-dynamics');
  const neuralTrack = curriculumTracks.find((track) => track.id === 'neural-networks');
  const startPath = HUB_LEARNING_PATHS.find((path) => path.id === 'start-here');

  assert.ok(animation, 'training loop dynamics lesson should be active');
  assert.equal(animation.categoryId, 'neural-networks');
  assert.ok(animation.trackIds.includes('neural-networks'));
  assert.ok(neuralTrack.animationIds.includes('training-loop-dynamics'));
  assert.ok(isAnimationAvailable('training-loop-dynamics'));
  assert.deepEqual(animation.prerequisites, ['optimizers', 'overfitting']);
  assert.match(animation.learningObjectives.join(' '), /mini-batch|validation|overshooting/i);
  assert.match(animation.commonMisconception, /training loss|validation/i);

  assert.ok(
    neuralTrack.animationIds.indexOf('optimizers') < neuralTrack.animationIds.indexOf('training-loop-dynamics'),
    'optimizer rules should come before training loop diagnosis',
  );
  assert.ok(
    startPath.nodes.indexOf('optimizers') < startPath.nodes.indexOf('training-loop-dynamics'),
    'start-here should place training loop dynamics after optimizers',
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
  assert.deepEqual(animation.prerequisites, ['data-leakage-deep-dive']);
  assert.match(animation.learningObjectives.join(' '), /standardization|min-max|robust|training data/i);
  assert.match(animation.commonMisconception, /leaks|validation|test/i);

  assert.ok(
    coreMlTrack.animationIds.indexOf('data-leakage-deep-dive') <
      coreMlTrack.animationIds.indexOf('feature-scaling-preprocessing'),
    'leakage deep dive should precede preprocessing fit-scope decisions',
  );
});

test('data leakage deep dive is promoted into Core ML before preprocessing', () => {
  const animation = getAnimationById('data-leakage-deep-dive');
  const coreMlTrack = curriculumTracks.find((track) => track.id === 'core-ml');

  assert.ok(animation, 'data leakage deep dive should be active');
  assert.equal(animation.categoryId, 'core-ml');
  assert.ok(animation.trackIds.includes('core-ml'));
  assert.ok(coreMlTrack.animationIds.includes('data-leakage-deep-dive'));
  assert.ok(isAnimationAvailable('data-leakage-deep-dive'));
  assert.deepEqual(animation.prerequisites, ['cross-validation']);
  assert.match(animation.learningObjectives.join(' '), /duplicate|target|time|test/i);
  assert.match(animation.commonMisconception, /random split|deployment/i);

  assert.ok(
    coreMlTrack.animationIds.indexOf('cross-validation') <
      coreMlTrack.animationIds.indexOf('data-leakage-deep-dive'),
    'cross-validation should introduce folds before leakage mode audits',
  );
  assert.ok(
    coreMlTrack.animationIds.indexOf('data-leakage-deep-dive') <
      coreMlTrack.animationIds.indexOf('feature-scaling-preprocessing'),
    'leakage audits should precede train-only preprocessing rules',
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

test('calibration is promoted into Core ML probability-quality evaluation', () => {
  const animation = getAnimationById('calibration');
  const coreMlTrack = curriculumTracks.find((track) => track.id === 'core-ml');

  assert.ok(animation, 'calibration lesson should be active');
  assert.equal(animation.categoryId, 'core-ml');
  assert.ok(animation.trackIds.includes('core-ml'));
  assert.ok(coreMlTrack.animationIds.includes('calibration'));
  assert.ok(isAnimationAvailable('calibration'));
  assert.deepEqual(animation.prerequisites, ['logistic-regression', 'roc-pr-curves']);
  assert.match(animation.learningObjectives.join(' '), /probabilities|reliability|Brier/i);
  assert.match(animation.commonMisconception, /sigmoid|softmax|calibrated/i);

  assert.ok(
    coreMlTrack.animationIds.indexOf('roc-pr-curves') <
      coreMlTrack.animationIds.indexOf('calibration'),
    'threshold-sweep metrics should precede probability-quality evaluation',
  );
  assert.ok(
    coreMlTrack.animationIds.indexOf('calibration') <
      coreMlTrack.animationIds.indexOf('overfitting'),
    'probability quality should be covered before broader generalization diagnostics',
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

test('bias-variance tradeoff is promoted into Core ML before regularization', () => {
  const animation = getAnimationById('bias-variance-tradeoff');
  const backlogIds = new Set(curriculumBacklog.map((topic) => topic.id));
  const coreMlTrack = curriculumTracks.find((track) => track.id === 'core-ml');
  const regularization = getAnimationById('regularization');

  assert.ok(animation, 'bias-variance lesson should be active');
  assert.equal(animation.categoryId, 'core-ml');
  assert.ok(animation.trackIds.includes('core-ml'));
  assert.ok(coreMlTrack.animationIds.includes('bias-variance-tradeoff'));
  assert.ok(!backlogIds.has('bias-variance-tradeoff'));
  assert.ok(isAnimationAvailable('bias-variance-tradeoff'));
  assert.deepEqual(animation.prerequisites, ['overfitting', 'classification-metrics']);
  assert.ok(regularization.prerequisites.includes('bias-variance-tradeoff'));
  assert.match(animation.learningObjectives.join(' '), /bias|variance|sample size/i);
  assert.match(animation.commonMisconception, /overfitting|underfit|bias/i);

  assert.ok(
    coreMlTrack.animationIds.indexOf('overfitting') <
      coreMlTrack.animationIds.indexOf('bias-variance-tradeoff'),
    'overfitting should introduce generalization failure before the decomposition lesson',
  );
  assert.ok(
    coreMlTrack.animationIds.indexOf('bias-variance-tradeoff') <
      coreMlTrack.animationIds.indexOf('regularization'),
    'bias-variance diagnosis should precede regularization as one treatment',
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
  assert.deepEqual(animation.prerequisites, ['llm-training-objectives', 'tokenization', 'softmax']);
  assert.match(animation.learningObjectives.join(' '), /temperature|top-k|top-p|KV cache/i);

  assert.ok(
    transformerTrack.animationIds.indexOf('llm-training-objectives') <
      transformerTrack.animationIds.indexOf('transformer-token-generation'),
    'training objectives should come before generation loop practice',
  );
  assert.ok(
    transformerTrack.animationIds.indexOf('transformer-token-generation') <
      transformerTrack.animationIds.indexOf('kv-cache'),
    'generation loop should motivate the dedicated KV cache lesson',
  );
});

test('sampling strategies are promoted after the token generation loop', () => {
  const animation = getAnimationById('sampling-strategies');
  const transformerTrack = curriculumTracks.find((track) => track.id === 'nlp-transformers');
  const backlogIds = new Set(curriculumBacklog.map((topic) => topic.id));

  assert.ok(animation, 'sampling strategies lesson should be active');
  assert.equal(animation.categoryId, 'transformers');
  assert.ok(animation.trackIds.includes('nlp-transformers'));
  assert.ok(transformerTrack.animationIds.includes('sampling-strategies'));
  assert.ok(!backlogIds.has('sampling-strategies'));
  assert.ok(isAnimationAvailable('sampling-strategies'));
  assert.deepEqual(animation.prerequisites, ['transformer-token-generation', 'softmax']);
  assert.match(animation.learningObjectives.join(' '), /greedy|beam|temperature|top-k|top-p/i);
  assert.match(animation.commonMisconception, /weights|probability distribution|inference/i);

  assert.ok(
    transformerTrack.animationIds.indexOf('transformer-token-generation') <
      transformerTrack.animationIds.indexOf('sampling-strategies'),
    'token generation should introduce the loop before decoding strategy tradeoffs',
  );
  assert.ok(
    transformerTrack.animationIds.indexOf('sampling-strategies') <
      transformerTrack.animationIds.indexOf('kv-cache'),
    'sampling strategy should precede lower-level cache optimization',
  );
});

test('fine-tuning methods are guided after training objectives and decoding', () => {
  const animation = getAnimationById('fine-tuning');
  const transformerTrack = curriculumTracks.find((track) => track.id === 'nlp-transformers');

  assert.ok(animation, 'fine-tuning lesson should be active');
  assert.equal(animation.categoryId, 'transformers');
  assert.ok(animation.trackIds.includes('nlp-transformers'));
  assert.ok(transformerTrack.animationIds.includes('fine-tuning'));
  assert.ok(isAnimationAvailable('fine-tuning'));
  assert.deepEqual(animation.prerequisites, ['llm-training-objectives', 'sampling-strategies']);
  assert.match(animation.learningObjectives.join(' '), /LoRA|QLoRA|SFT|DPO|RLHF|preference/i);
  assert.match(animation.commonMisconception, /retrieval|gradients|preference/i);

  assert.ok(
    transformerTrack.animationIds.indexOf('sampling-strategies') <
      transformerTrack.animationIds.indexOf('fine-tuning'),
    'decoding controls should precede method selection for behavior tuning',
  );
});

test('transformer architecture families are promoted before model-specific transformer lessons', () => {
  const animation = getAnimationById('transformer-architecture-families');
  const transformerTrack = curriculumTracks.find((track) => track.id === 'nlp-transformers');
  const tokenGeneration = getAnimationById('transformer-token-generation');

  assert.ok(animation, 'transformer architecture families lesson should be active');
  assert.equal(animation.categoryId, 'transformers');
  assert.ok(animation.trackIds.includes('nlp-transformers'));
  assert.ok(transformerTrack.animationIds.includes('transformer-architecture-families'));
  assert.ok(isAnimationAvailable('transformer-architecture-families'));
  assert.deepEqual(animation.prerequisites, ['transformer', 'attention-masks']);
  assert.match(animation.learningObjectives.join(' '), /encoder-only|decoder-only|encoder-decoder|BERT|GPT|T5/i);
  assert.match(animation.commonMisconception, /BERT|GPT|T5|interchangeable/i);

  assert.ok(
    transformerTrack.animationIds.indexOf('transformer') <
      transformerTrack.animationIds.indexOf('transformer-architecture-families'),
    'the full transformer block should precede architecture family comparison',
  );
  assert.ok(
    transformerTrack.animationIds.indexOf('transformer-architecture-families') <
      transformerTrack.animationIds.indexOf('bert'),
    'family comparison should precede BERT-specific detail',
  );
});

test('LLM training objectives are promoted before token generation and fine-tuning', () => {
  const animation = getAnimationById('llm-training-objectives');
  const transformerTrack = curriculumTracks.find((track) => track.id === 'nlp-transformers');
  const tokenGeneration = getAnimationById('transformer-token-generation');

  assert.ok(animation, 'LLM training objectives lesson should be active');
  assert.equal(animation.categoryId, 'transformers');
  assert.ok(animation.trackIds.includes('nlp-transformers'));
  assert.ok(transformerTrack.animationIds.includes('llm-training-objectives'));
  assert.ok(isAnimationAvailable('llm-training-objectives'));
  assert.deepEqual(animation.prerequisites, ['transformer-architecture-families', 'tokenization', 'softmax']);
  assert.ok(tokenGeneration.prerequisites.includes('llm-training-objectives'));
  assert.match(animation.learningObjectives.join(' '), /next-token|masked|preference|fine-tuning/i);
  assert.match(animation.commonMisconception, /Instruction tuning|preference|pretraining/i);

  assert.ok(
    transformerTrack.animationIds.indexOf('transformer-architecture-families') <
      transformerTrack.animationIds.indexOf('llm-training-objectives'),
    'architecture families should precede objective comparison',
  );
  assert.ok(
    transformerTrack.animationIds.indexOf('llm-training-objectives') <
      transformerTrack.animationIds.indexOf('transformer-token-generation'),
    'training objectives should precede inference-time token generation',
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
  assert.deepEqual(animation.prerequisites, ['rag-reranking-grounding', 'embeddings', 'cosine-similarity']);
  assert.match(animation.learningObjectives.join(' '), /chunk|rerank|recall@k|nDCG/i);
  assert.match(animation.commonMisconception, /missing evidence|right chunks|context/i);

  assert.ok(
    generativeTrack.animationIds.indexOf('rag') <
      generativeTrack.animationIds.indexOf('rag-retrieval-evaluation'),
    'broad RAG introduction should precede retrieval evaluation',
  );
});

test('RAG chunking and context packing are promoted before retrieval evaluation', () => {
  const animation = getAnimationById('rag-chunking-context');
  const evaluation = getAnimationById('rag-retrieval-evaluation');
  const generativeTrack = curriculumTracks.find((track) => track.id === 'generative-ai');

  assert.ok(animation, 'RAG chunking lesson should be active');
  assert.equal(animation.categoryId, 'advanced-models');
  assert.ok(animation.trackIds.includes('generative-ai'));
  assert.ok(generativeTrack.animationIds.includes('rag-chunking-context'));
  assert.ok(isAnimationAvailable('rag-chunking-context'));
  assert.deepEqual(animation.prerequisites, ['rag', 'embeddings']);
  assert.ok(generativeTrack.animationIds.indexOf('rag-chunking-context') < generativeTrack.animationIds.indexOf('rag-retrieval-evaluation'));
  assert.match(animation.learningObjectives.join(' '), /chunk size|overlap|top-k|context budget/i);
  assert.match(animation.commonMisconception, /Bigger chunks|overlap|top-k/i);

  assert.ok(
    generativeTrack.animationIds.indexOf('rag') <
      generativeTrack.animationIds.indexOf('rag-chunking-context'),
    'basic RAG pipeline should precede chunking and packing tradeoffs',
  );
  assert.ok(
    generativeTrack.animationIds.indexOf('rag-chunking-context') <
      generativeTrack.animationIds.indexOf('rag-retrieval-evaluation'),
    'chunking and packing should precede retrieval metrics',
  );
});

test('RAG vector indexing is promoted between chunking and retrieval evaluation', () => {
  const animation = getAnimationById('rag-vector-indexing');
  const evaluation = getAnimationById('rag-retrieval-evaluation');
  const generativeTrack = curriculumTracks.find((track) => track.id === 'generative-ai');

  assert.ok(animation, 'RAG vector indexing lesson should be active');
  assert.equal(animation.categoryId, 'advanced-models');
  assert.ok(animation.trackIds.includes('generative-ai'));
  assert.ok(generativeTrack.animationIds.includes('rag-vector-indexing'));
  assert.ok(isAnimationAvailable('rag-vector-indexing'));
  assert.deepEqual(animation.prerequisites, ['rag-chunking-context', 'embeddings', 'cosine-similarity']);
  assert.ok(evaluation.prerequisites.includes('rag-reranking-grounding'));
  assert.match(animation.learningObjectives.join(' '), /exact|approximate|IVF|HNSW|latency|recall/i);
  assert.match(animation.commonMisconception, /Approximate|exact search|reranking/i);

  assert.ok(
    generativeTrack.animationIds.indexOf('rag-chunking-context') <
      generativeTrack.animationIds.indexOf('rag-vector-indexing'),
    'chunking should define indexed units before vector index tradeoffs',
  );
  assert.ok(
    generativeTrack.animationIds.indexOf('rag-vector-indexing') <
      generativeTrack.animationIds.indexOf('rag-retrieval-evaluation'),
    'vector indexing should precede downstream reranking and evaluation',
  );
});

test('RAG reranking and grounding is promoted before retrieval evaluation', () => {
  const animation = getAnimationById('rag-reranking-grounding');
  const evaluation = getAnimationById('rag-retrieval-evaluation');
  const generativeTrack = curriculumTracks.find((track) => track.id === 'generative-ai');
  const backlogIds = new Set(curriculumBacklog.map((topic) => topic.id));

  assert.ok(animation, 'RAG reranking and grounding lesson should be active');
  assert.equal(animation.categoryId, 'advanced-models');
  assert.ok(animation.trackIds.includes('generative-ai'));
  assert.ok(generativeTrack.animationIds.includes('rag-reranking-grounding'));
  assert.ok(isAnimationAvailable('rag-reranking-grounding'));
  assert.ok(!backlogIds.has('rag-reranking-grounding'));
  assert.deepEqual(animation.prerequisites, ['rag-vector-indexing', 'rag-chunking-context', 'embeddings', 'cosine-similarity']);
  assert.match(animation.learningObjectives.join(' '), /first-pass|rerank|grounding|conflicting|stale/i);
  assert.match(animation.commonMisconception, /cannot fix|stale|conflicting/i);
  assert.ok(evaluation.prerequisites.includes('rag-reranking-grounding'));

  assert.ok(
    generativeTrack.animationIds.indexOf('rag-vector-indexing') <
      generativeTrack.animationIds.indexOf('rag-reranking-grounding'),
    'vector indexing should enable reranking and grounding',
  );
  assert.ok(
    generativeTrack.animationIds.indexOf('rag-reranking-grounding') <
      generativeTrack.animationIds.indexOf('rag-retrieval-evaluation'),
    'reranking and grounding should be before retrieval metrics',
  );
});

test('RAG failure modes lesson is active between reranking and retrieval evaluation', () => {
  const animation = getAnimationById('rag-failure-modes');
  const generativeTrack = curriculumTracks.find((track) => track.id === 'generative-ai');

  assert.ok(animation, 'RAG failure modes lesson should be active');
  assert.equal(animation.categoryId, 'advanced-models');
  assert.ok(animation.trackIds.includes('generative-ai'));
  assert.ok(generativeTrack.animationIds.includes('rag-failure-modes'));
  assert.ok(isAnimationAvailable('rag-failure-modes'));
  assert.deepEqual(animation.prerequisites, ['rag-reranking-grounding']);
  assert.match(animation.learningObjectives.join(' '), /missing|stale|conflict|irrelevant/i);
  assert.match(animation.commonMisconception, /grounded|fluent|evidence/i);

  assert.ok(
    generativeTrack.animationIds.indexOf('rag-reranking-grounding') <
      generativeTrack.animationIds.indexOf('rag-failure-modes'),
    'reranking should precede failure-mode diagnostics',
  );
  assert.ok(
    generativeTrack.animationIds.indexOf('rag-failure-modes') <
      generativeTrack.animationIds.indexOf('rag-retrieval-evaluation'),
    'failure-mode diagnostics should prepare retrieval evaluation',
  );
});

test('MDP formalism bridges RL foundations and Q-learning', () => {
  const animation = getAnimationById('mdp-formalism');
  const valueIteration = getAnimationById('value-iteration');
  const rlTrack = curriculumTracks.find((track) => track.id === 'rl-algorithms');
  const rlPath = HUB_LEARNING_PATHS.find((path) => path.id === 'rl-path');

  assert.ok(animation, 'MDP formalism lesson should be active');
  assert.equal(animation.categoryId, 'reinforcement-learning');
  assert.ok(animation.trackIds.includes('rl-algorithms'));
  assert.ok(rlTrack.animationIds.includes('mdp-formalism'));
  assert.ok(isAnimationAvailable('mdp-formalism'));
  assert.deepEqual(animation.prerequisites, ['rl-foundations']);
  assert.deepEqual(valueIteration.prerequisites, ['mdp-formalism', 'expected-value-variance']);
  assert.match(animation.learningObjectives.join(' '), /states|actions|transition|rewards|discount/i);
  assert.match(animation.commonMisconception, /state diagram|probability distribution/i);

  assert.ok(
    rlPath.nodes.indexOf('rl-foundations') < rlPath.nodes.indexOf('mdp-formalism'),
    'RL foundations should introduce MDP vocabulary first',
  );
  assert.ok(
    rlPath.nodes.indexOf('mdp-formalism') < rlPath.nodes.indexOf('value-iteration'),
    'MDP formalism should precede value iteration planning',
  );
});

test('value iteration bridges MDP planning and Q-learning', () => {
  const animation = getAnimationById('value-iteration');
  const policyIteration = getAnimationById('policy-iteration');
  const rlTrack = curriculumTracks.find((track) => track.id === 'rl-algorithms');
  const rlPath = HUB_LEARNING_PATHS.find((path) => path.id === 'rl-path');

  assert.ok(animation, 'Value iteration lesson should be active');
  assert.equal(animation.categoryId, 'reinforcement-learning');
  assert.ok(animation.trackIds.includes('rl-algorithms'));
  assert.ok(rlTrack.animationIds.includes('value-iteration'));
  assert.ok(isAnimationAvailable('value-iteration'));
  assert.deepEqual(animation.prerequisites, ['mdp-formalism', 'expected-value-variance']);
  assert.deepEqual(policyIteration.prerequisites, ['value-iteration']);
  assert.match(animation.learningObjectives.join(' '), /Bellman|sweeps|policy/i);
  assert.match(animation.commonMisconception, /known transition model|sampled experience/i);

  assert.ok(
    rlPath.nodes.indexOf('value-iteration') < rlPath.nodes.indexOf('policy-iteration'),
    'Value iteration should precede policy iteration comparison',
  );
});

test('policy iteration is active before model-free Q-learning', () => {
  const animation = getAnimationById('policy-iteration');
  const qLearning = getAnimationById('q-learning');
  const rlTrack = curriculumTracks.find((track) => track.id === 'rl-algorithms');
  const rlPath = HUB_LEARNING_PATHS.find((path) => path.id === 'rl-path');

  assert.ok(animation, 'Policy iteration lesson should be active');
  assert.equal(animation.categoryId, 'reinforcement-learning');
  assert.ok(animation.trackIds.includes('rl-algorithms'));
  assert.ok(rlTrack.animationIds.includes('policy-iteration'));
  assert.ok(isAnimationAvailable('policy-iteration'));
  assert.deepEqual(animation.prerequisites, ['value-iteration']);
  assert.deepEqual(qLearning.prerequisites, ['value-iteration', 'expected-value-variance']);
  assert.match(animation.learningObjectives.join(' '), /policy evaluation|policy improvement|stable/i);
  assert.match(animation.commonMisconception, /evaluates the current policy|greedily improves/i);

  assert.ok(
    rlPath.nodes.indexOf('policy-iteration') < rlPath.nodes.indexOf('q-learning'),
    'Policy iteration should be seen before model-free Q-learning',
  );
});

test('policy gradients are active after exploration in the RL path', () => {
  const animation = getAnimationById('policy-gradients');
  const rlTrack = curriculumTracks.find((track) => track.id === 'rl-algorithms');
  const rlPath = HUB_LEARNING_PATHS.find((path) => path.id === 'rl-path');

  assert.ok(animation, 'Policy gradients lesson should be active');
  assert.equal(animation.categoryId, 'reinforcement-learning');
  assert.ok(animation.trackIds.includes('rl-algorithms'));
  assert.ok(rlTrack.animationIds.includes('policy-gradients'));
  assert.ok(isAnimationAvailable('policy-gradients'));
  assert.deepEqual(animation.prerequisites, ['rl-exploration', 'expected-value-variance']);
  assert.match(animation.learningObjectives.join(' '), /stochastic policy|returns|expected return/i);
  assert.match(animation.commonMisconception, /action probabilities|sampled returns/i);

  assert.ok(
    rlPath.nodes.indexOf('rl-exploration') < rlPath.nodes.indexOf('policy-gradients'),
    'Exploration should precede policy-gradient probability updates',
  );
});

test('actor-critic follows policy gradients with a value-baseline lesson', () => {
  const animation = getAnimationById('actor-critic');
  const rlTrack = curriculumTracks.find((track) => track.id === 'rl-algorithms');
  const rlPath = HUB_LEARNING_PATHS.find((path) => path.id === 'rl-path');

  assert.ok(animation, 'Actor-critic lesson should be active');
  assert.equal(animation.categoryId, 'reinforcement-learning');
  assert.ok(animation.trackIds.includes('rl-algorithms'));
  assert.ok(rlTrack.animationIds.includes('actor-critic'));
  assert.ok(isAnimationAvailable('actor-critic'));
  assert.deepEqual(animation.prerequisites, ['policy-gradients']);
  assert.match(animation.learningObjectives.join(' '), /actor|critic|advantage/i);
  assert.match(animation.commonMisconception, /critic does not choose|cleaner learning signal/i);

  assert.ok(
    rlPath.nodes.indexOf('policy-gradients') < rlPath.nodes.indexOf('actor-critic'),
    'Policy gradients should precede actor-critic',
  );
});

test('reward shaping follows actor-critic with sparse-reward guidance', () => {
  const animation = getAnimationById('reward-shaping');
  const rlTrack = curriculumTracks.find((track) => track.id === 'rl-algorithms');
  const rlPath = HUB_LEARNING_PATHS.find((path) => path.id === 'rl-path');

  assert.ok(animation, 'Reward shaping lesson should be active');
  assert.equal(animation.categoryId, 'reinforcement-learning');
  assert.ok(animation.trackIds.includes('rl-algorithms'));
  assert.ok(rlTrack.animationIds.includes('reward-shaping'));
  assert.ok(isAnimationAvailable('reward-shaping'));
  assert.deepEqual(animation.prerequisites, ['actor-critic']);
  assert.match(animation.learningObjectives.join(' '), /sparse|potential|optimal/i);
  assert.match(animation.commonMisconception, /wrong objective|reward hacking/i);

  assert.ok(
    rlPath.nodes.indexOf('actor-critic') < rlPath.nodes.indexOf('reward-shaping'),
    'Actor-critic should precede reward shaping',
  );
});

test('lesson assessments provide backed quiz and lab counts for priority lessons', () => {
  const animationIds = new Set(allAnimations.map((animation) => animation.id));
  const stats = getAssessmentStats(lessonAssessments);
  const expectedStats = Object.values(lessonAssessments).reduce(
    (totals, assessment) => ({
      totalQuizQuestions: totals.totalQuizQuestions + assessment.quiz.length,
      totalLabs: totals.totalLabs + assessment.labs.length,
    }),
    { totalQuizQuestions: 0, totalLabs: 0 },
  );

  assert.deepEqual(stats, expectedStats);

  for (const lessonId of Object.keys(lessonAssessments)) {
    assert.ok(animationIds.has(lessonId), `${lessonId} should be an active lesson`);
  }

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
