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
const LEVELS = Object.keys(LEVEL_ORDER);

function normalized(value) {
  return String(value || '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
}

function correctAnswer(question) {
  return question.choices[question.answerIndex];
}

test('propensity scores has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('propensity-scores');

  assert.equal(quiz.length, 100);
  assert.equal(labs.length, 3);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    const expectedNumber = String(index + 1).padStart(3, '0');

    assert.match(question.id, /^ps-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use the ps id format`);
    assert.equal(question.id.slice(3, 6), expectedNumber, `question ${index + 1} should preserve numeric order`);
    assert.ok(!question.id.includes('generated-'), `question ${index + 1} should not use a generated id`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.equal(new Set(question.choices.map(normalized)).size, 3, `question ${index + 1} should have unique choices`);
    assert.ok(Number.isInteger(question.answerIndex), `question ${index + 1} answer index should be an integer`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(LEVELS.includes(question.level), `question ${index + 1} should have a recognized level`);
  }
});

test('propensity scores assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('propensity-scores');
  const prompts = new Map();
  const correctAnswers = new Map();

  for (const question of quiz) {
    const prompt = normalized(question.prompt);
    const answer = normalized(correctAnswer(question));

    assert.ok(!prompts.has(prompt), `${question.id} duplicates prompt from ${prompts.get(prompt)}`);
    prompts.set(prompt, question.id);

    assert.ok(!correctAnswers.has(answer), `${question.id} duplicates correct answer from ${correctAnswers.get(answer)}`);
    correctAnswers.set(answer, question.id);
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
      `${level} band should occupy questions ${start + 1}-${end}`,
    );
  }

  for (let index = 1; index < quiz.length; index += 1) {
    assert.ok(
      LEVEL_ORDER[quiz[index].level] >= LEVEL_ORDER[quiz[index - 1].level],
      `question ${index + 1} should not move backward in difficulty`,
    );
  }
});

test('propensity scores assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('propensity-scores');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));

  const orderedMilestones = [
    [/what problem do propensity scores address/, 0, 6],
    [/when are propensity scores usually considered/, 1, 7],
    [/what is the treatment indicator/, 2, 8],
    [/what role does the outcome play after propensity design/, 3, 9],
    [/covariates belong in a basic propensity model/, 4, 10],
    [/what is a propensity score e x/, 5, 12],
    [/why summarize covariates into a propensity score/, 6, 13],
    [/propensity score matching try to do/, 7, 14],
    [/inverse propensity weighting try to create/, 8, 15],
    [/simple ipw what is the treated unit weight/, 9, 16],
    [/simple ipw what is the control unit weight/, 10, 17],
    [/what does covariate balance mean/, 11, 18],
    [/what is propensity score overlap/, 12, 19],
    [/why is poor overlap dangerous/, 13, 20],
    [/what is hidden bias in this lesson/, 14, 20],
    [/not assume about propensity scores/, 15, 20],
    [/why call propensity scoring a design tool/, 16, 20],
    [/why define the estimand before matching or weighting/, 17, 20],
    [/what is initial imbalance/, 18, 20],
    [/first propensity score readout check/, 19, 22],
    [/what does the propensity model predict/, 20, 28],
    [/different from outcome prediction/, 21, 29],
    [/balancing intuition behind a propensity score/, 22, 30],
    [/why use a caliper/, 23, 31],
    [/nearest neighbor propensity matching/, 24, 32],
    [/why are extreme ipw weights a warning sign/, 25, 33],
    [/trimming common support violations/, 26, 34],
    [/standardized mean difference diagnose/, 27, 35],
    [/compare balance before and after adjustment/, 28, 36],
    [/what is the positivity requirement/, 29, 37],
    [/no unmeasured confounding require/, 30, 38],
    [/propensity covariates be pre treatment/, 31, 39],
    [/including a mediator in the propensity model/, 32, 40],
    [/what does an overlap plot reveal/, 33, 41],
    [/restricting to common support/, 34, 42],
    [/what does att focus on/, 35, 43],
    [/what does ate weighting aim to represent/, 36, 44],
    [/stabilized or clipped weights/, 37, 45],
    [/model quality be judged/, 38, 46],
    [/delay outcome comparison until after design checks/, 39, 47],
    [/hidden bias sensitivity analysis/, 40, 48],
    [/negative control outcome help/, 41, 49],
    [/pre treatment placebo check/, 42, 50],
    [/propensity analysis report/, 48, 50],
    [/propensity score protocol/, 49, 52],
    [/sicker patients are more likely to receive a drug/, 50, 58],
    [/age affects treatment choice and outcome/, 51, 59],
    [/no comparable controls/, 52, 60],
    [/e x 0 02/, 53, 61],
    [/control unit has e x 0 80/, 54, 62],
    [/treated unit has e x 0 80/, 55, 63],
    [/smds fall near zero/, 56, 64],
    [/income imbalance gets worse/, 57, 65],
    [/doctor preference affects treatment and recovery/, 58, 66],
    [/adherence measured after prescription/, 59, 67],
    [/many matching rules and keeps the one with best outcome lift/, 60, 68],
    [/active users are more likely to see a campaign/, 61, 69],
    [/only power users ever receive an offer/, 62, 70],
    [/trims non overlap regions/, 63, 71],
    [/high treatment prediction auc/, 64, 72],
    [/love plot shows many covariates still imbalanced/, 66, 74],
    [/effect on a pre treatment outcome/, 67, 75],
    [/affect an unrelated outcome/, 68, 75],
    [/few weights are 60 or 80/, 70, 75],
    [/low overlap and high hidden bias/, 74, 75],
    [/which propensity score claim is false/, 75, 83],
    [/which overlap claim is wrong/, 76, 84],
    [/which balance claim is unsafe/, 77, 85],
    [/which model selection claim is misleading/, 78, 86],
    [/which covariate claim is false/, 79, 87],
    [/which workflow claim is unsafe/, 80, 88],
    [/which score definition claim is wrong/, 81, 89],
    [/which weighting claim is false/, 82, 90],
    [/which support claim is false/, 87, 90],
    [/which causal proof claim is false/, 89, 90],
    [/define propensity score in an interview/, 90, 96],
    [/workflow should you describe for propensity methods/, 91, 97],
    [/diagnostics should you name first/, 92, 98],
    [/explain poor overlap/, 93, 99],
    [/hidden confounding limitation/, 94, 100],
    [/compare matching and weighting/, 95, 100],
    [/discuss extreme weights/, 96, 100],
    [/production ready propensity report/, 97, 100],
    [/interview ready mastery of propensity scores/, 98, 100],
  ];

  for (const [pattern, minIndex, maxIndex] of orderedMilestones) {
    const index = textByQuestion.findIndex((text) => pattern.test(text));
    assert.notEqual(index, -1, `missing milestone: ${pattern}`);
    assert.ok(
      index >= minIndex && index < maxIndex,
      `${pattern} should appear in questions ${minIndex + 1}-${maxIndex}, found question ${index + 1}`,
    );
  }
});

test('propensity scores assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('propensity-scores');
  const unsafePatterns = [
    /propensity scores fix unmeasured confounding/i,
    /no overlap is fine because weighting can extrapolate perfectly/i,
    /using a propensity model means balance no longer needs checking/i,
    /highest treatment-prediction auc always gives the best causal design/i,
    /post-treatment variables are always safe in the propensity model/i,
    /choose the matching rule that gives the largest favorable outcome effect/i,
    /propensity score is the probability of a positive outcome/i,
    /extreme weights are always harmless/i,
    /original population estimand unchanged/i,
    /matched pair guarantees identical counterfactual outcomes/i,
    /small adjusted p-value proves the propensity design was valid/i,
    /good observed balance proves no hidden bias remains/i,
    /treated units need no comparable controls after matching/i,
    /only the adjusted effect is needed in the report/i,
    /propensity-adjusted estimate proves the exact causal effect without assumptions/i,
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
  const totals = [0, 0, 0];

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const page = quiz.slice(pageStart, pageStart + 10);
    const positions = page.map((question) => question.answerIndex);

    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
    for (const position of positions) {
      totals[position] += 1;
    }
  }

  assert.ok(Math.max(...totals) - Math.min(...totals) <= 1, `answer positions should be balanced, found ${totals.join('/')}`);
});
