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

test('confounding simpsons paradox has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('confounding-simpsons-paradox');

  assert.equal(quiz.length, 100);
  assert.equal(labs.length, 3);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    const expectedNumber = String(index + 1).padStart(3, '0');

    assert.match(question.id, /^conf-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use the conf id format`);
    assert.equal(question.id.slice(5, 8), expectedNumber, `question ${index + 1} should preserve numeric order`);
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

test('confounding simpsons paradox assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('confounding-simpsons-paradox');
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

test('confounding simpsons paradox assessment progresses from recall to interview readiness', () => {
  const { quiz } = getLessonAssessment('confounding-simpsons-paradox');
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

test('confounding simpsons paradox assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('confounding-simpsons-paradox');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));

  const orderedMilestones = [
    [/observational comparison/, 0, 7],
    [/what is a confounder/, 0, 8],
    [/exposure mean/, 1, 9],
    [/outcome in a causal comparison/, 2, 10],
    [/association not automatically causation/, 3, 11],
    [/segment in simpson examples/, 4, 12],
    [/segment mix/, 5, 13],
    [/high risk share/, 6, 14],
    [/aggregate effect/, 7, 15],
    [/within segment effect/, 8, 16],
    [/simpson s paradox/, 9, 17],
    [/simpson reversal signal/, 10, 18],
    [/compare like with like/, 11, 19],
    [/baseline gap/, 12, 20],
    [/randomization help/, 13, 20],
    [/nonrandom exposures/, 14, 20],
    [/adjustment for a confounder/, 15, 20],
    [/stratification do/, 16, 20],
    [/common mistake with simpson examples/, 17, 20],
    [/first check/, 18, 22],
    [/aggregate differ from each segment effect/, 20, 28],
    [/lesson formula aggregate sum segment rate/, 20, 30],
    [/condition on a confounder/, 21, 31],
    [/measured before exposure/, 22, 32],
    [/mediator when estimating total effect/, 23, 33],
    [/adjusting for a collider/, 24, 34],
    [/what is a backdoor path/, 25, 35],
    [/blocking a confounding path/, 26, 36],
    [/overadjustment/, 27, 37],
    [/effect modification different from confounding/, 28, 38],
    [/average treatment effect/, 29, 39],
    [/target population/, 30, 40],
    [/standardization in this context/, 31, 41],
    [/matching try to do/, 32, 42],
    [/regression adjustment help/, 33, 43],
    [/overlap matter/, 34, 44],
    [/positivity require/, 35, 45],
    [/unmeasured confounding/, 36, 46],
    [/proxy help with confounding/, 37, 47],
    [/balance check inspect/, 38, 48],
    [/sensitivity analysis/, 39, 49],
    [/observational data more defensible/, 40, 50],
    [/clean randomized experiment/, 41, 50],
    [/small segment samples/, 42, 50],
    [/simpson reversal report include/, 43, 50],
    [/non collapsible/, 44, 50],
    [/risk adjustment/, 45, 50],
    [/selection into treatment/, 46, 50],
    [/causal graphs help/, 47, 50],
    [/confounding analysis protocol/, 48, 52],
    [/many more high risk users/, 50, 57],
    [/wins in every risk segment but loses overall/, 50, 58],
    [/hospital a treats sicker patients/, 51, 59],
    [/coupon is sent mostly/, 52, 60],
    [/support program appears to reduce outcomes/, 53, 61],
    [/university admits higher percentages/, 54, 62],
    [/opted in voluntarily/, 55, 63],
    [/effect flips direction/, 56, 64],
    [/no comparable high risk controls/, 57, 65],
    [/severity is unmeasured/, 58, 66],
    [/engagement measured after feature launch/, 59, 67],
    [/opened a complaint ticket/, 60, 68],
    [/different region mixes/, 61, 69],
    [/unmeasured motivation/, 62, 70],
    [/randomized a b test/, 63, 71],
    [/segment effects all point positive/, 64, 72],
    [/aggregate says launch/, 65, 73],
    [/rising demand/, 66, 74],
    [/winter and control mostly in summer/, 67, 75],
    [/severity is unavailable/, 68, 75],
    [/modest hidden confounder/, 69, 75],
    [/dag shows risk/, 70, 75],
    [/good overlap balance/, 71, 75],
    [/simpson s paradox appears/, 72, 75],
    [/served harder cases/, 73, 75],
    [/aggregate claim is false/, 75, 82],
    [/confounder claim is wrong/, 75, 83],
    [/simpson claim is misleading/, 76, 84],
    [/randomization claim is unsafe/, 77, 85],
    [/adjustment claim is false/, 78, 86],
    [/mediator claim is unsafe/, 79, 87],
    [/collider claim is wrong/, 80, 88],
    [/overlap claim is false/, 81, 89],
    [/unmeasured confounding claim is unsafe/, 82, 90],
    [/segment claim is false/, 83, 90],
    [/standardization claim is misleading/, 84, 90],
    [/proxy claim is wrong/, 85, 90],
    [/reporting claim is unsafe/, 86, 90],
    [/causal claim is false/, 87, 90],
    [/proof claim is false/, 88, 90],
    [/define confounding in an interview/, 90, 96],
    [/explain simpson s paradox/, 90, 97],
    [/clean example to give/, 91, 98],
    [/diagnose a suspected simpson reversal/, 92, 99],
    [/describe adjustment/, 93, 100],
    [/traps should you name/, 94, 100],
    [/contrast with a b testing/, 95, 100],
    [/complete readout include/, 96, 100],
    [/decision from a confounded comparison/, 97, 100],
    [/interview ready mastery/, 98, 100],
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

test('confounding simpsons paradox assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('confounding-simpsons-paradox');
  const unsafePatterns = [
    /pooled aggregate is always the best causal estimate/i,
    /confounder is caused by the treatment after exposure/i,
    /simpson reversal automatically proves which treatment is best/i,
    /randomization is unnecessary if you have many observational rows/i,
    /adding every available variable always improves causal validity/i,
    /adjusting for a treatment mediator is always correct for total effects/i,
    /conditioning on a collider always removes bias/i,
    /no overlap is fine because regression can extrapolate perfectly/i,
    /measured balance proves no unmeasured confounding exists/i,
    /if every segment points one way uncertainty and target weights no longer matter/i,
    /standardization creates causal identification even with unmeasured confounders/i,
    /any proxy fully removes confounding from an unmeasured cause/i,
    /report only the adjusted result and hide the reversal table/i,
    /large association is enough to claim causality in observational data/i,
    /regression-adjusted observational estimate proves the exact causal effect/i,
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

test('confounding simpsons paradox assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('confounding-simpsons-paradox');
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

test('confounding simpsons paradox assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('confounding-simpsons-paradox');
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
