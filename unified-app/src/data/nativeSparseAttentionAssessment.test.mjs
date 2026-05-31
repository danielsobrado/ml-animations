import assert from 'node:assert/strict';
import test from 'node:test';
import { getLessonAssessment } from './lessonAssessments.js';

const LEVEL_ORDER = {
  Foundation: 0,
  Mechanism: 1,
  Application: 2,
  Tricky: 3,
  Interview: 4,
};

function normalized(value) {
  return String(value || '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
}

function correctAnswer(question) {
  return question.choices[question.answerIndex];
}

test('native sparse attention has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('native-sparse-attention');

  assert.equal(quiz.length, 100);
  assert.equal(labs.length, 3);
  assert.deepEqual(labs.map((lab) => lab.id), [
    'mini-nsa-block-window',
    'mini-nsa-score-select',
    'mini-nsa-gates-hardware',
  ]);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.match(question.id, /^nsa-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use the nsa id format`);
    assert.equal(Number(question.id.slice(4, 7)), index + 1, `question ${index + 1} id number should match its position`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.equal(new Set(question.choices.map((choice) => normalized(choice))).size, 3, `question ${index + 1} should have distinct choices`);
    assert.ok(Number.isInteger(question.answerIndex), `question ${index + 1} answer index should be an integer`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(Object.hasOwn(LEVEL_ORDER, question.level), `question ${index + 1} should have a recognized level`);
  }
});

test('native sparse attention assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('native-sparse-attention');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('native sparse attention assessment progresses from branch basics to interview readiness', () => {
  const { quiz } = getLessonAssessment('native-sparse-attention');
  const expectedBands = [
    ['Foundation', 0, 20],
    ['Mechanism', 20, 50],
    ['Application', 50, 75],
    ['Tricky', 75, 90],
    ['Interview', 90, 100],
  ];

  for (const [level, start, end] of expectedBands) {
    assert.deepEqual([...new Set(quiz.slice(start, end).map((question) => question.level))], [level]);
  }

  for (let index = 1; index < quiz.length; index += 1) {
    assert.ok(LEVEL_ORDER[quiz[index].level] >= LEVEL_ORDER[quiz[index - 1].level]);
  }
});

test('native sparse attention assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('native-sparse-attention');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const milestones = [
    ['purpose', ['long context each query must read']],
    ['three branches', ['compression selection and sliding window']],
    ['native training', ['trained with the sparse reading pattern']],
    ['block ranges', ['half open contiguous ranges']],
    ['compressed scores', ['dot the query with each compressed key']],
    ['gated merge', ['weighted sum of branch output vectors']],
    ['gqa shared selection', ['average block scores across query heads']],
    ['budget estimate', ['compressed summaries plus selected block tokens plus window tokens']],
    ['tricky false claims', ['nsa claim is false']],
    ['interview readiness', ['production ready nsa takeaway']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of milestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('native sparse attention assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('native-sparse-attention');
  const unsafePatterns = [
    /random attention-score deletion/i,
    /tokenizer, embedding, and logits/i,
    /guarantee every important detail/i,
    /always be random/i,
    /only for far-away tokens/i,
    /sparse attention is automatically fast/i,
    /post-hoc pruning is always equivalent/i,
    /unrelated kv blocks/i,
    /do not need causal or padding masks/i,
    /removes all attention-style weighting/i,
    /kv-cache memory and validity irrelevant/i,
    /same as fetching external documents/i,
    /lowest loaded-token count is always/i,
    /unlimited context length at no cost/i,
    /random, free, maskless/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /false|unsafe|wrong|trap|reject|claim|belief|misconception/i.test(question.prompt);
    assert.ok(!unsafeAnswer || explicitTrapPrompt, `question ${index + 1} keys a false claim outside a trap prompt`);
  }
});

test('native sparse attention misconception traps are placed after concept scaffolding', () => {
  const { quiz } = getLessonAssessment('native-sparse-attention');
  const trapIds = [
    'nsa-076-false-random',
    'nsa-077-false-branches',
    'nsa-078-false-compression',
    'nsa-079-false-selection',
    'nsa-080-false-window',
    'nsa-081-false-speed',
    'nsa-082-false-native',
    'nsa-083-false-gqa',
    'nsa-084-false-mask',
    'nsa-085-false-softmax',
    'nsa-086-false-cache',
    'nsa-087-false-rag',
    'nsa-088-false-budget',
    'nsa-089-false-context',
    'nsa-090-tricky-summary',
  ];

  assert.deepEqual(quiz.slice(75, 90).map((question) => question.id), trapIds);

  for (const question of quiz.slice(0, 75)) {
    assert.doesNotMatch(question.prompt, /^Which .* claim is false\?/);
  }
});

test('native sparse attention assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('native-sparse-attention');

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const page = quiz.slice(pageStart, pageStart + 10);
    const answers = page.map((question) => normalized(correctAnswer(question)));

    assert.equal(new Set(answers).size, answers.length, `page starting at question ${pageStart + 1} should not repeat exact answers`);

    for (const [answerIndex, answer] of answers.entries()) {
      for (const [promptIndex, question] of page.entries()) {
        if (answerIndex === promptIndex || answer.length < 8) continue;
        assert.ok(!normalized(question.prompt).includes(answer));
      }
    }
  }
});

test('native sparse attention assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('native-sparse-attention');
  const globalPositionCounts = [0, 1, 2].map((slot) => quiz.filter((question) => question.answerIndex === slot).length);

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const positions = quiz.slice(pageStart, pageStart + 10).map((question) => question.answerIndex);
    const maxSameSlot = Math.max(...[0, 1, 2].map((slot) => positions.filter((position) => position === slot).length));

    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
    assert.ok(maxSameSlot <= 6, `page starting at question ${pageStart + 1} should not overuse one answer position`);
  }

  assert.ok(
    Math.max(...globalPositionCounts) - Math.min(...globalPositionCounts) <= 1,
    `global answer positions should be balanced, got ${globalPositionCounts.join(', ')}`,
  );
});
