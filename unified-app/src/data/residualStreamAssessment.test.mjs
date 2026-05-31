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

test('residual stream has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('residual-stream');

  assert.equal(quiz.length, 100);
  assert.equal(labs.length, 3);
  assert.deepEqual(labs.map((lab) => lab.id), [
    'trace-component-write',
    'compare-scale-control',
    'separate-stream-from-cache',
  ]);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.match(question.id, /^resstream-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use the ordered resstream id format`);
    assert.equal(Number(question.id.slice(10, 13)), index + 1, `question ${index + 1} should have a sequential id`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.equal(new Set(question.choices.map((choice) => normalized(choice))).size, 3, `question ${index + 1} should have distinct choices`);
    assert.ok(Number.isInteger(question.answerIndex), `question ${index + 1} answer index should be an integer`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(Object.hasOwn(LEVEL_ORDER, question.level), `question ${index + 1} should have a recognized level`);
  }
});

test('residual stream assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('residual-stream');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('residual stream assessment progresses from additive basics to interview readiness', () => {
  const { quiz } = getLessonAssessment('residual-stream');
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

test('residual stream assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('residual-stream');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const milestones = [
    ['purpose', ['carries token information through many components']],
    ['additive update', ['adds attention and mlp outputs']],
    ['not memory', ['not a separate memory bank or kv cache']],
    ['formula', ['x l 1']],
    ['normalization', ['normalize before a sublayer']],
    ['probing', ['activation patching']],
    ['application scale bug', ['residual write scale']],
    ['production tests', ['tiny tensor case']],
    ['tricky false claims', ['residual stream claim is false']],
    ['interview readiness', ['production ready residual stream takeaway']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of milestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('residual stream assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('residual-stream');
  const unsafePatterns = [
    /separate memory bank outside the forward pass/i,
    /fully replaces the previous token representation/i,
    /only the final layer has a residual stream/i,
    /write scale never matters/i,
    /guarantees every earlier feature is perfectly preserved/i,
    /any width and still be added/i,
    /attention never reads the residual stream/i,
    /external databases attached to the residual stream/i,
    /residual stream and kv cache are the same object/i,
    /only a display feature/i,
    /remove the need for backpropagation/i,
    /unrelated to the final residual stream/i,
    /cannot affect model behavior/i,
    /compatible with any checkpoint/i,
    /external memory, full replacement, shape-free addition, and scale-proof storage/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /false|unsafe|wrong|trap|reject|claim|belief|misconception/i.test(question.prompt);
    assert.ok(!unsafeAnswer || explicitTrapPrompt, `question ${index + 1} keys a false claim outside a trap prompt`);
  }
});

test('residual stream misconception traps are placed after concept scaffolding', () => {
  const { quiz } = getLessonAssessment('residual-stream');
  const trapIds = [
    'resstream-076-false-memory',
    'resstream-077-false-replace',
    'resstream-078-false-final-only',
    'resstream-079-false-scale',
    'resstream-080-false-no-interference',
    'resstream-081-false-shape',
    'resstream-082-false-attention',
    'resstream-083-false-mlp',
    'resstream-084-false-cache',
    'resstream-085-false-norm',
    'resstream-086-false-gradients',
    'resstream-087-false-logits',
    'resstream-088-false-probing',
    'resstream-089-false-parallel',
    'resstream-090-tricky-summary',
  ];

  assert.deepEqual(quiz.slice(75, 90).map((question) => question.id), trapIds);

  for (const question of quiz.slice(0, 75)) {
    assert.doesNotMatch(question.prompt, /^Which .* claim is false\?/);
  }
});

test('residual stream assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('residual-stream');

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

test('residual stream assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('residual-stream');
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
