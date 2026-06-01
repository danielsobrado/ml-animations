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

test('transformer has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('transformer');

  assert.equal(quiz.length, 100);
  assert.equal(labs.length, 3);
  assert.deepEqual(labs.map((lab) => lab.id), [
    'trace-token',
    'compare-mixing-and-mlp',
    'debug-mask-and-shape-contracts',
  ]);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.match(question.id, /^transformer-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use the ordered transformer id format`);
    assert.equal(Number(question.id.slice(12, 15)), index + 1, `question ${index + 1} should have a sequential id`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.equal(new Set(question.choices.map((choice) => normalized(choice))).size, 3, `question ${index + 1} should have distinct choices`);
    assert.ok(Number.isInteger(question.answerIndex), `question ${index + 1} answer index should be an integer`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(Object.hasOwn(LEVEL_ORDER, question.level), `question ${index + 1} should have a recognized level`);
  }
});

test('transformer assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('transformer');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('transformer assessment progresses from block basics to interview readiness', () => {
  const { quiz } = getLessonAssessment('transformer');
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

test('transformer assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('transformer');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const milestones = [
    ['purpose', ['stacking attention feed forward transforms residual paths and normalization']],
    ['not only attention', ['not only attention']],
    ['token mixing', ['move information across token positions']],
    ['per token transform', ['feed forward network']],
    ['mechanism flow', ['q k and v projections']],
    ['norm variants', ['pre norm transformer block']],
    ['application mask bug', ['causal mask and input target shifting']],
    ['production tests', ['shape mask residual add']],
    ['tricky false claims', ['transformer claim is false']],
    ['interview readiness', ['lesson ready transformer takeaway']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of milestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('transformer assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('transformer');
  const unsafePatterns = [
    /only an attention layer/i,
    /optional decoration with no modeling role/i,
    /replace the need for weights/i,
    /only changes display formatting/i,
    /never needs position information/i,
    /causal masks are unnecessary/i,
    /padding tokens should be treated as normal semantic tokens/i,
    /stores multiple full datasets/i,
    /main cross-token mixing mechanism/i,
    /without any head or readout/i,
    /any checkpoint can load into any transformer block order/i,
    /cost is unrelated to sequence length/i,
    /complete proof of model reasoning/i,
    /fixes one universal training objective/i,
    /attention-only, order-free, mask-free, headless stacks/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /false|unsafe|wrong|trap|reject|claim|belief|misconception/i.test(question.prompt);
    assert.ok(!unsafeAnswer || explicitTrapPrompt, `question ${index + 1} keys a false claim outside a trap prompt`);
  }
});

test('transformer misconception traps are placed after concept scaffolding', () => {
  const { quiz } = getLessonAssessment('transformer');
  const trapIds = [
    'transformer-076-false-only-attention',
    'transformer-077-false-mlp',
    'transformer-078-false-residual',
    'transformer-079-false-norm',
    'transformer-080-false-order',
    'transformer-081-false-mask',
    'transformer-082-false-padding',
    'transformer-083-false-heads',
    'transformer-084-false-mlp-mixing',
    'transformer-085-false-output',
    'transformer-086-false-config',
    'transformer-087-false-cost',
    'transformer-088-false-interpretation',
    'transformer-089-false-objective',
    'transformer-090-tricky-summary',
  ];

  assert.deepEqual(quiz.slice(75, 90).map((question) => question.id), trapIds);

  for (const question of quiz.slice(0, 75)) {
    assert.doesNotMatch(question.prompt, /^Which .* claim is false\?/);
  }
});

test('transformer assessment stays within visible lesson scope', () => {
  const { quiz } = getLessonAssessment('transformer');
  const lessonScopeLeaks = [
    /\bKV[- ]?cache\b/i,
    /\bcache data\b/i,
    /\bcached\b/i,
    /\bkernel\b/i,
    /\bmixed precision\b/i,
    /\bdtype\b/i,
    /\bserving\b/i,
    /\bdeployment\b/i,
    /\bprobe hidden states\b/i,
    /\bproduction\b/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const visibleText = `${question.prompt} ${question.choices.join(' ')} ${question.explanation}`;
    assert.ok(!lessonScopeLeaks.some((pattern) => pattern.test(visibleText)), `question ${index + 1} leaks later or non-visible transformer scope`);
  }
});

test('transformer assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('transformer');

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const page = quiz.slice(pageStart, pageStart + 10);
    const answers = page.map((question) => normalized(correctAnswer(question)));

    assert.equal(new Set(answers).size, answers.length, `page starting at question ${pageStart + 1} should not repeat exact answers`);

    for (const [answerIndex, answer] of answers.entries()) {
      for (const [promptIndex, question] of page.entries()) {
        if (answerIndex === promptIndex || answer.length < 8) continue;
        const visibleQuestionText = normalized(`${question.prompt} ${question.choices.join(' ')}`);
        assert.ok(!visibleQuestionText.includes(answer));
      }
    }
  }
});

test('transformer assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('transformer');
  const allPositions = quiz.map((question) => question.answerIndex);
  const globalCounts = [0, 1, 2].map((slot) => allPositions.filter((position) => position === slot).length);

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const positions = quiz.slice(pageStart, pageStart + 10).map((question) => question.answerIndex);
    const maxSameSlot = Math.max(...[0, 1, 2].map((slot) => positions.filter((position) => position === slot).length));

    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
    assert.ok(maxSameSlot <= 6, `page starting at question ${pageStart + 1} should not overuse one answer position`);
  }

  assert.ok(Math.max(...globalCounts) - Math.min(...globalCounts) <= 1, `global answer positions should stay balanced: ${globalCounts.join(', ')}`);
});
