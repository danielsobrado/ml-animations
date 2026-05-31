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
    .trim()
    .replace(/\s+/g, ' ');
}

function correctAnswer(question) {
  return question.choices[question.answerIndex];
}

test('cross entropy has a complete curated 100-question assessment', () => {
  const { quiz } = getLessonAssessment('cross-entropy');

  assert.equal(quiz.length, 100);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.ok(question.id.startsWith('ce-'), `question ${index + 1} should use the ce id prefix`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(Object.hasOwn(LEVEL_ORDER, question.level), `question ${index + 1} should have a recognized level`);
  }
});

test('cross entropy assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('cross-entropy');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('cross entropy assessment progresses from loss basics to interview readiness', () => {
  const { quiz } = getLessonAssessment('cross-entropy');
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

test('cross entropy assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('cross-entropy');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const milestones = [
    ['classification loss', ['classification loss']],
    ['predicted distribution', ['predicted distribution']],
    ['one hot target', ['one hot target']],
    ['true class probability', ['true class probability']],
    ['negative log', ['negative log']],
    ['softmax prerequisite', ['softmax']],
    ['bits unit', ['bits']],
    ['formula', ['h p q']],
    ['one hot reduction', ['one hot reduction']],
    ['wrong classes', ['wrong class']],
    ['log base', ['preserving the same optimum']],
    ['kl divergence', ['measures extra coding cost']],
    ['label smoothing', ['label smoothing']],
    ['calibration', ['calibration']],
    ['tricky false claims', ['tricky false claims']],
    ['interview readiness', ['interview readiness']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of milestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('cross entropy assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('cross-entropy');
  const unsafePatterns = [
    /same as accuracy/i,
    /only checks the top predicted label/i,
    /wrong classes never matter/i,
    /kl divergence is symmetric/i,
    /log base changes the best model/i,
    /clipping makes the model correct/i,
    /low cross-entropy guarantees calibration/i,
    /tuning hyperparameters on test/i,
    /one low-loss batch.*prove generalization/i,
    /uniform prediction.*automatically good/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /false|unsafe|wrong|trap|reject|claim|shortcut/i.test(question.prompt);
    assert.ok(!unsafeAnswer || explicitTrapPrompt, `question ${index + 1} keys a false claim outside a trap prompt`);
  }
});

test('cross entropy assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('cross-entropy');

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

test('cross entropy assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('cross-entropy');

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const positions = quiz.slice(pageStart, pageStart + 10).map((question) => question.answerIndex);
    const maxSameSlot = Math.max(...[0, 1, 2].map((slot) => positions.filter((position) => position === slot).length));

    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
    assert.ok(maxSameSlot <= 6, `page starting at question ${pageStart + 1} should not overuse one answer position`);
  }
});
