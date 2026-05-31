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

test('max pooling has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('max-pooling');

  assert.equal(quiz.length, 100);
  assert.deepEqual(
    labs.map((lab) => lab.id),
    ['trace-window-argmax', 'audit-stride-coverage', 'compare-max-average-gap'],
  );
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.match(question.id, /^mp-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use the ordered mp id format`);
    assert.equal(Number(question.id.slice(3, 6)), index + 1, `question ${index + 1} should keep ordered ids`);
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

test('max pooling assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('max-pooling');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('max pooling assessment progresses from definitions to interview readiness', () => {
  const { quiz } = getLessonAssessment('max-pooling');
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

test('max pooling assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('max-pooling');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const milestones = [
    ['purpose', ['downsamples while keeping']],
    ['local window', ['small local region']],
    ['max rule', ['largest activation']],
    ['information loss', ['non maximum values']],
    ['argmax', ['coordinate of the value']],
    ['shape formula', ['floor h 2p k s 1']],
    ['gradient route', ['input position that won']],
    ['application tracing', ['window starts winners']],
    ['tricky false claims', ['claim is false']],
    ['interview readiness', ['production ready max pooling takeaway']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of milestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('max pooling assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('max-pooling');
  const unsafePatterns = [
    /learns a weighted filter/i,
    /averages all activations/i,
    /stride has no effect/i,
    /collapses all channels/i,
    /non-winning value receives the same gradient/i,
    /all-negative windows into zero/i,
    /guarantees full translation invariance/i,
    /preserves exact positions of all values/i,
    /automatically a calibrated probability/i,
    /padding can never change/i,
    /cannot share any input/i,
    /always increases downstream/i,
    /order never matters/i,
    /only the final accuracy/i,
    /never loses useful information/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /false|unsafe|wrong|trap|reject|claim|belief/i.test(question.prompt);
    assert.ok(!unsafeAnswer || explicitTrapPrompt, `question ${index + 1} keys a false claim outside a trap prompt`);
  }
});

test('max pooling assessment keeps misconception traps after setup', () => {
  const { quiz } = getLessonAssessment('max-pooling');
  const misconceptionPatterns = [
    /learns a weighted filter/i,
    /averages all activations/i,
    /stride has no effect/i,
    /collapses all channels/i,
    /non-winning value receives the same gradient/i,
    /all-negative windows into zero/i,
    /guarantees full translation invariance/i,
    /preserves exact positions of all values/i,
    /automatically a calibrated probability/i,
    /padding can never change/i,
    /cannot share any input/i,
    /always increases downstream/i,
    /order never matters/i,
    /only the final accuracy/i,
    /never loses useful information/i,
  ];
  const trapPrompt = /false|unsafe|wrong|trap|reject|claim|belief/i;

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const containsMisconception = misconceptionPatterns.some((pattern) => pattern.test(answer));
    if (!containsMisconception) continue;
    assert.ok(index >= 75, `${question.id} introduces misconception too early`);
    assert.match(question.prompt, trapPrompt, `${question.id} should mark misconception as a trap`);
  }
});

test('max pooling assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('max-pooling');

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

test('max pooling assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('max-pooling');

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const positions = quiz.slice(pageStart, pageStart + 10).map((question) => question.answerIndex);
    const maxSameSlot = Math.max(...[0, 1, 2].map((slot) => positions.filter((position) => position === slot).length));

    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
    assert.ok(maxSameSlot <= 6, `page starting at question ${pageStart + 1} should not overuse one answer position`);
  }
});
