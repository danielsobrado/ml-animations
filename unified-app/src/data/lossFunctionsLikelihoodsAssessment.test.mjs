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
    .replace(/\s+/g, ' ')
    .trim();
}

function correctAnswer(question) {
  return question.choices[question.answerIndex];
}

test('loss functions likelihoods has a complete curated 100-question assessment', () => {
  const { quiz } = getLessonAssessment('loss-functions-likelihoods');

  assert.equal(quiz.length, 100);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.ok(question.id.startsWith('losslik-'), `question ${index + 1} should use the losslik id prefix`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(Object.hasOwn(LEVEL_ORDER, question.level), `question ${index + 1} should have a recognized level`);
  }
});

test('loss functions likelihoods assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('loss-functions-likelihoods');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('loss functions likelihoods assessment progresses from recall to interview readiness', () => {
  const { quiz } = getLessonAssessment('loss-functions-likelihoods');
  const expectedBands = [
    ['Foundation', 0, 20],
    ['Mechanism', 20, 50],
    ['Application', 50, 75],
    ['Tricky', 75, 90],
    ['Interview', 90, 100],
  ];

  for (const [level, start, end] of expectedBands) {
    assert.deepEqual(
      [...new Set(quiz.slice(start, end).map((question) => question.level))],
      [level],
      `${level} questions should occupy positions ${start + 1}-${end}`,
    );
  }

  for (let index = 1; index < quiz.length; index += 1) {
    assert.ok(
      LEVEL_ORDER[quiz[index].level] >= LEVEL_ORDER[quiz[index - 1].level],
      `question ${index + 1} should not regress in difficulty`,
    );
  }
});

test('loss functions likelihoods assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('loss-functions-likelihoods');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const orderedMilestones = [
    ['loss purpose', ['prediction', 'observed target']],
    ['negative log-likelihood', ['negative log-likelihood']],
    ['loss assumption', ['target type', 'noise']],
    ['squared error target', ['continuous numeric targets']],
    ['cross-entropy target', ['categorical targets']],
    ['Gaussian residuals', ['gaussian residual noise']],
    ['independent examples', ['joint likelihood', 'logs turn products']],
    ['outliers', ['outliers']],
    ['class weights', ['class weights']],
    ['validation', ['validation']],
    ['scenario application', ['house prices']],
    ['tricky false claims', ['claim is false']],
    ['interview readiness', ['interview']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of orderedMilestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('loss functions likelihoods assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('loss-functions-likelihoods');
  const unsafePatterns = [
    /one universal loss is best/i,
    /unrelated to maximizing likelihood/i,
    /always best for classification labels/i,
    /ignores the probability assigned to the true class/i,
    /changes the optimum/i,
    /unaffected by huge residuals/i,
    /never affects loss design/i,
    /guarantees better production performance/i,
    /automatically means calibrated probabilities/i,
    /always exactly optimizes the final business metric/i,
    /no longer represents class likelihood/i,
    /can never affect optimization dynamics/i,
    /remove the need to evaluate/i,
    /without looking at target type/i,
    /without assumptions or metrics/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /false|misleading|unsafe|wrong|trap|claim/i.test(question.prompt);

    assert.ok(
      !unsafeAnswer || explicitTrapPrompt,
      `question ${index + 1} keys a false claim outside an explicit trap prompt`,
    );
  }
});

test('loss functions likelihoods assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('loss-functions-likelihoods');
  const pageSize = 10;

  for (let pageStart = 0; pageStart < quiz.length; pageStart += pageSize) {
    const page = quiz.slice(pageStart, pageStart + pageSize);
    const answers = page.map((question) => normalized(correctAnswer(question)));

    assert.equal(new Set(answers).size, answers.length, `page starting at question ${pageStart + 1} should not repeat exact answers`);

    for (const [answerIndex, answer] of answers.entries()) {
      for (const [promptIndex, question] of page.entries()) {
        if (answerIndex === promptIndex) continue;
        assert.ok(
          !normalized(question.prompt).includes(answer),
          `question ${pageStart + promptIndex + 1} prompt should not reveal answer from question ${pageStart + answerIndex + 1}`,
        );
      }
    }
  }
});

test('loss functions likelihoods assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('loss-functions-likelihoods');

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const page = quiz.slice(pageStart, pageStart + 10);
    const positions = page.map((question) => question.answerIndex);

    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
  }
});
