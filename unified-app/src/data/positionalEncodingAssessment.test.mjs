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

test('positional encoding has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('positional-encoding');

  assert.equal(quiz.length, 100);
  assert.equal(labs.length, 3);
  assert.deepEqual(labs.map((lab) => lab.id), [
    'order-sensitive-sentence',
    'trace-sinusoidal-channels',
    'debug-position-alignment',
  ]);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.match(question.id, /^posenc-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use the ordered posenc id format`);
    assert.equal(Number(question.id.slice(7, 10)), index + 1, `question ${index + 1} should have a sequential id`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.equal(new Set(question.choices.map((choice) => normalized(choice))).size, 3, `question ${index + 1} should have distinct choices`);
    assert.ok(Number.isInteger(question.answerIndex), `question ${index + 1} answer index should be an integer`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(Object.hasOwn(LEVEL_ORDER, question.level), `question ${index + 1} should have a recognized level`);
  }
});

test('positional encoding assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('positional-encoding');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('positional encoding assessment progresses from order basics to interview readiness', () => {
  const { quiz } = getLessonAssessment('positional-encoding');
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

test('positional encoding assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('positional-encoding');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const milestones = [
    ['purpose', ['self attention lacks built in token order']],
    ['order sensitive sentence', ['dog bites man']],
    ['sinusoidal', ['sine and cosine waves']],
    ['not replacing meaning', ['do positional encodings replace token meaning']],
    ['formula', ['pe pos 2i']],
    ['learned table', ['learned position embedding table']],
    ['extrapolation caveat', ['learned absolute table']],
    ['application bug', ['kv cache', 'positions need careful alignment']],
    ['tricky false claims', ['self attention order claim is false']],
    ['interview readiness', ['production ready positional encoding takeaway']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of milestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('positional encoding assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('positional-encoding');
  const unsafePatterns = [
    /inherently knows token order without any position signal/i,
    /replace token embeddings and remove word meaning/i,
    /random learned rows updated by gradient descent/i,
    /automatically extrapolate to every longer prompt/i,
    /identical final input vector at every sequence position/i,
    /fully replaces the need for positional information/i,
    /alone decides the next token/i,
    /necessarily destroys token identity/i,
    /one identical constant vector/i,
    /sorting tokens alphabetically/i,
    /need no care once positional encodings exist/i,
    /always be reused after prompt edits/i,
    /off-by-one position id is harmless/i,
    /long-context quality is guaranteed/i,
    /optional decoration, token replacements, and automatic long-context guarantees/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /false|unsafe|wrong|trap|reject|claim|belief|misconception/i.test(question.prompt);
    assert.ok(!unsafeAnswer || explicitTrapPrompt, `question ${index + 1} keys a false claim outside a trap prompt`);
  }
});

test('positional encoding misconception traps are placed after concept scaffolding', () => {
  const { quiz } = getLessonAssessment('positional-encoding');
  const trapIds = [
    'posenc-076-false-self-attention',
    'posenc-077-false-replace-token',
    'posenc-078-false-sinusoidal-random',
    'posenc-079-false-learned-extrapolate',
    'posenc-080-false-same-vector',
    'posenc-081-false-mask-replaces',
    'posenc-082-false-position-alone',
    'posenc-083-false-destroy-identity',
    'posenc-084-false-constant',
    'posenc-085-false-alphabetical',
    'posenc-086-false-padding',
    'posenc-087-false-cache',
    'posenc-088-false-off-by-one',
    'posenc-089-false-long-context',
    'posenc-090-tricky-summary',
  ];

  assert.deepEqual(quiz.slice(75, 90).map((question) => question.id), trapIds);

  for (const question of quiz.slice(0, 75)) {
    assert.doesNotMatch(question.prompt, /^Which .* claim is false\?/);
  }
});

test('positional encoding assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('positional-encoding');

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

test('positional encoding assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('positional-encoding');
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
