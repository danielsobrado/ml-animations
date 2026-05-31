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

test('time series forecasting has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('time-series-forecasting-track');

  assert.equal(quiz.length, 100);
  assert.equal(labs.length, 3);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.match(question.id, /^tsf-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use a stable ordered tsf id`);
    assert.equal(Number(question.id.slice(4, 7)), index + 1, `question ${index + 1} id should match its position`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.equal(
      new Set(question.choices.map((choice) => normalized(choice))).size,
      question.choices.length,
      `question ${index + 1} choices should be distinct`,
    );
    assert.ok(Number.isInteger(question.answerIndex), `question ${index + 1} answer index should be an integer`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(Object.hasOwn(LEVEL_ORDER, question.level), `question ${index + 1} should have a recognized level`);
  }
});

test('time series forecasting assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('time-series-forecasting-track');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('time series forecasting assessment progresses from temporal basics to interview readiness', () => {
  const { quiz } = getLessonAssessment('time-series-forecasting-track');
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

test('time series forecasting assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('time-series-forecasting-track');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const orderedMilestones = [
    ['purpose', ['future values', 'prediction time']],
    ['horizon', ['how far ahead']],
    ['lag feature', ['past value']],
    ['seasonality', ['repeating pattern']],
    ['rolling split', ['trains on earlier windows', 'evaluates on later windows']],
    ['leakage boundary', ['available at prediction time']],
    ['shifted rolling features', ['shifted correctly']],
    ['multi-step strategy', ['direct multi step forecasting']],
    ['metric caution', ['actual values are zero or near zero']],
    ['application workflow', ['clear cutoff rules', 'safe features', 'rolling validation']],
    ['tricky false claims', ['split claim is false']],
    ['interview readiness', ['define the temporal contract']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of orderedMilestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('time series forecasting assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('time-series-forecasting-track');
  const unsafePatterns = [
    /random row split is always honest/i,
    /after the prediction timestamp if accuracy improves/i,
    /including the target period.*harmless/i,
    /automatically proven for every longer horizon/i,
    /mape is always safe/i,
    /complex forecast is useful even if it cannot beat/i,
    /any observed future covariate can be used/i,
    /one lucky cutoff proves/i,
    /automatically gives reliable prediction intervals/i,
    /never needs drift or freshness monitoring/i,
    /repeatedly checking the final test/i,
    /calendar features always leak/i,
    /always safe for every deployed series/i,
    /aggregates computed after the prediction time are safe/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /false|unsafe|wrong|trap|claim|optimistic|dangerous/i.test(question.prompt);

    assert.ok(
      !unsafeAnswer || explicitTrapPrompt,
      `question ${index + 1} keys a false claim outside an explicit trap prompt`,
    );
  }
});

test('time series forecasting assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('time-series-forecasting-track');
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

test('time series forecasting assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('time-series-forecasting-track');
  const totals = [0, 0, 0];

  for (const question of quiz) {
    totals[question.answerIndex] += 1;
  }

  assert.ok(Math.max(...totals) - Math.min(...totals) <= 1, `answer positions should be globally balanced: ${totals.join(', ')}`);

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const page = quiz.slice(pageStart, pageStart + 10);
    const positions = page.map((question) => question.answerIndex);

    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
  }
});
