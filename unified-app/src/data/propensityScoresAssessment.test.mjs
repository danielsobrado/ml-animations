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

test('propensity scores has a complete curated 100-question assessment', () => {
  const { quiz } = getLessonAssessment('propensity-scores');

  assert.equal(quiz.length, 100);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.ok(question.id.startsWith('ps-'), `question ${index + 1} should use the ps id prefix`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(Object.hasOwn(LEVEL_ORDER, question.level), `question ${index + 1} should have a recognized level`);
  }
});

test('propensity scores assessment progresses from recall to interview readiness', () => {
  const { quiz } = getLessonAssessment('propensity-scores');
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

test('propensity scores assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('propensity-scores');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const orderedMilestones = [
    ['purpose', ['observational', 'covariates']],
    ['propensity definition', ['probability', 'treatment', 'covariates']],
    ['matching', ['matching']],
    ['weighting', ['weighting']],
    ['balance', ['covariate balance']],
    ['overlap', ['overlap']],
    ['hidden bias', ['hidden bias']],
    ['design tool', ['design tool']],
    ['positivity', ['positivity']],
    ['pre-treatment timing', ['post-treatment', 'bias']],
    ['sensitivity', ['sensitivity']],
    ['protocol', ['protocol']],
    ['scenario application', ['sicker patients']],
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

test('propensity scores assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('propensity-scores');
  const unsafePatterns = [
    /fix unmeasured confounding/i,
    /extrapolate perfectly/i,
    /balance no longer needs checking/i,
    /highest treatment-prediction AUC always/i,
    /post-treatment variables are always safe/i,
    /largest favorable outcome effect/i,
    /probability of a positive outcome/i,
    /always harmless/i,
    /original population estimand unchanged/i,
    /guarantees identical counterfactual outcomes/i,
    /small adjusted p-value proves/i,
    /proves no hidden bias/i,
    /need no comparable controls/i,
    /only the adjusted effect is needed/i,
    /without assumptions/i,
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

test('propensity scores assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('propensity-scores');
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

test('propensity scores assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('propensity-scores');

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const page = quiz.slice(pageStart, pageStart + 10);
    const positions = page.map((question) => question.answerIndex);

    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
  }
});
