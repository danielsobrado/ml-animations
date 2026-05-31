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

test('data engineering for ML has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('data-engineering-for-ml-track');

  assert.equal(quiz.length, 100);
  assert.equal(labs.length, 3);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.match(question.id, /^deml-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use an ordered deml id`);
    assert.equal(Number(question.id.slice(5, 8)), index + 1, `question ${index + 1} id should match its position`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.equal(new Set(question.choices.map((choice) => normalized(choice))).size, 3, `question ${index + 1} choices should be distinct`);
    assert.equal(Number.isInteger(question.answerIndex), true, `question ${index + 1} answer index should be an integer`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(Object.hasOwn(LEVEL_ORDER, question.level), `question ${index + 1} should have a recognized level`);
  }
});

test('data engineering for ML assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('data-engineering-for-ml-track');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('data engineering for ML assessment progresses from data boundaries to interview readiness', () => {
  const { quiz } = getLessonAssessment('data-engineering-for-ml-track');
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

test('data engineering for ML assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('data-engineering-for-ml-track');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const orderedMilestones = [
    ['purpose', ['reproducible time correct and production safe']],
    ['prediction time', ['model must make a decision']],
    ['label window', ['time interval after prediction']],
    ['point-in-time correctness', ['at or before that row prediction time']],
    ['target encoding', ['statistics of the label']],
    ['train/serve skew', ['training features differ']],
    ['data contract', ['schema freshness meaning quality']],
    ['availability time', ['could actually be used by the model']],
    ['point-in-time join', ['before that example prediction time']],
    ['target encoding mechanism', ['training labels available within the relevant fold']],
    ['parity checks', ['same entity and cutoff']],
    ['application workflow', ['point in time features valid labels contracts']],
    ['tricky false claims', ['claim is false']],
    ['interview readiness', ['prediction cutoffs and labels']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of orderedMilestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('data engineering for ML assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('data-engineering-for-ml-track');
  const unsafePatterns = [
    /safe if its event time is before prediction, even when it arrived later/i,
    /latest warehouse value is always point-in-time correct/i,
    /labels can be defined using information available only before prediction/i,
    /target encodings on the full dataset before splitting is always safe/i,
    /feature definitions may differ as long as offline accuracy is high/i,
    /contracts are documentation only and should not block/i,
    /null-rate spike can be ignored/i,
    /backfills can recompute historical features using all data currently available/i,
    /stale features are harmless when the schema is unchanged/i,
    /saved model artifact is enough to reproduce/i,
    /user ids are always harmless/i,
    /random row splits are always honest/i,
    /data quality monitoring is unnecessary/i,
    /any private field can be used as a feature/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /false|unsafe|wrong|trap|claim/i.test(question.prompt);

    assert.ok(
      !unsafeAnswer || explicitTrapPrompt,
      `question ${index + 1} keys a false claim outside an explicit trap prompt`,
    );
  }
});

test('data engineering for ML assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('data-engineering-for-ml-track');
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

test('data engineering for ML assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('data-engineering-for-ml-track');
  const totals = [0, 0, 0];

  for (const question of quiz) {
    totals[question.answerIndex] += 1;
  }

  assert.ok(Math.max(...totals) - Math.min(...totals) <= 1, `answer positions should be globally balanced, saw ${totals.join('/')}`);

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const page = quiz.slice(pageStart, pageStart + 10);
    const positions = page.map((question) => question.answerIndex);

    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
  }
});
