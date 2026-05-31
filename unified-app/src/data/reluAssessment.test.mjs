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

test('relu has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('relu');

  assert.equal(quiz.length, 100);
  assert.deepEqual(
    labs.map((lab) => lab.id),
    ['cross-zero', 'trace-dot-bias-gate', 'diagnose-dead-gate-risk'],
  );
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.match(question.id, /^relu-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use the ordered relu id format`);
    assert.equal(Number(question.id.slice(5, 8)), index + 1, `question ${index + 1} should keep ordered ids`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.equal(new Set(question.choices.map(normalized)).size, 3, `question ${index + 1} choices should be distinct`);
    assert.equal(Number.isInteger(question.answerIndex), true, `question ${index + 1} answer index should be an integer`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(Object.hasOwn(LEVEL_ORDER, question.level), `question ${index + 1} should have a recognized level`);
  }

  const positionCounts = [0, 1, 2].map((answerIndex) => (
    quiz.filter((question) => question.answerIndex === answerIndex).length
  ));
  assert.ok(Math.max(...positionCounts) - Math.min(...positionCounts) <= 1, `answer positions should be balanced: ${positionCounts.join(', ')}`);
});

test('relu assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('relu');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const correctAnswers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(correctAnswers).size, correctAnswers.length);
});

test('relu assessment progresses from definitions to interview readiness', () => {
  const { quiz } = getLessonAssessment('relu');
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

test('relu assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('relu');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const milestones = [
    ['formula', ['f z max 0 z']],
    ['negative clipping', ['negative input', 'zero']],
    ['positive branch', ['positive input', 'identity']],
    ['local slope positive', ['slope', 'positive side', 'one']],
    ['local slope negative', ['slope zero on the blocked branch']],
    ['dead unit', ['dead relu unit']],
    ['backward mask', ['mask', 'positive']],
    ['initialization', ['initialization', 'relu']],
    ['application diagnosis', ['dead unit', 'activation frequency']],
    ['tricky false claims', ['claim is false']],
    ['interview readiness', ['production ready relu takeaway']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of milestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('relu assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('relu');
  const unsafePatterns = [
    /turns any hidden value into a calibrated probability/i,
    /preserves negative inputs unchanged/i,
    /negative branch has local derivative one/i,
    /always harmless because zeros are sparse/i,
    /guarantees deep networks will avoid all gradient problems/i,
    /bounded above by one/i,
    /always the correct final activation for classification/i,
    /all examples in a batch must share the same relu mask/i,
    /every initialization scale works equally well/i,
    /in-place relu is always safe/i,
    /more zero activations are always better/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /false|unsafe|wrong|trap|claim/i.test(question.prompt);
    assert.ok(!unsafeAnswer || explicitTrapPrompt, `question ${index + 1} keys a false claim outside a trap prompt`);
  }
});

test('relu assessment keeps misconception traps after setup', () => {
  const { quiz } = getLessonAssessment('relu');
  const expectedTrapIds = [
    'relu-076-false-probability-trap',
    'relu-077-false-negative-trap',
    'relu-078-false-gradient-trap',
    'relu-079-false-dead-trap',
    'relu-080-false-magic-trap',
    'relu-081-false-bounded-trap',
    'relu-082-false-smooth-trap',
    'relu-083-false-final-layer-trap',
    'relu-084-false-zero-trap',
    'relu-085-false-batch-trap',
    'relu-086-false-bias-trap',
    'relu-087-false-init-trap',
    'relu-088-false-inplace-trap',
    'relu-089-false-sparsity-trap',
    'relu-090-tricky-summary',
  ];
  const misconceptionPatterns = [
    /turns any hidden value into a calibrated probability/i,
    /preserves negative inputs unchanged/i,
    /negative branch has local derivative one/i,
    /always harmless because zeros are sparse/i,
    /guarantees deep networks will avoid all gradient problems/i,
    /bounded above by one/i,
    /differentiable with one smooth derivative at zero/i,
    /always the correct final activation for classification/i,
    /every zero relu output proves a data pipeline bug/i,
    /all examples in a batch must share the same relu mask/i,
    /bias has no effect on whether a relu gate opens/i,
    /every initialization scale works equally well/i,
    /in-place relu is always safe/i,
    /more zero activations are always better/i,
    /cannot be the source of training bugs/i,
  ];
  const trapPrompt = /false|unsafe|wrong|trap|claim/i;

  assert.deepEqual(quiz.slice(75, 90).map((question) => question.id), expectedTrapIds);

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const containsMisconception = misconceptionPatterns.some((pattern) => pattern.test(answer));
    if (!containsMisconception) continue;
    assert.ok(index >= 75, `${question.id} introduces misconception too early`);
    assert.match(question.prompt, trapPrompt, `${question.id} should mark misconception as a trap`);
  }
});

test('relu assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('relu');

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const page = quiz.slice(pageStart, pageStart + 10);
    const answers = page.map((question) => normalized(correctAnswer(question)));

    assert.equal(new Set(answers).size, answers.length, `page starting at question ${pageStart + 1} should not repeat exact answers`);

    for (const [answerIndex, answer] of answers.entries()) {
      for (const [promptIndex, question] of page.entries()) {
        if (answerIndex === promptIndex) continue;
        assert.ok(!normalized(question.prompt).includes(answer));
      }
    }
  }
});

test('relu assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('relu');

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const positions = quiz.slice(pageStart, pageStart + 10).map((question) => question.answerIndex);
    const maxSameSlot = Math.max(...[0, 1, 2].map((slot) => positions.filter((position) => position === slot).length));

    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
    assert.ok(maxSameSlot <= 6, `page starting at question ${pageStart + 1} should not overuse one answer position`);
  }
});
