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

test('power sample size has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('power-sample-size');

  assert.equal(quiz.length, 100);
  assert.equal(labs.length, 3);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    const expectedNumber = String(index + 1).padStart(3, '0');

    assert.match(question.id, /^power-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use the power id format`);
    assert.equal(question.id.slice(6, 9), expectedNumber, `question ${index + 1} should preserve numeric order`);
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

test('power sample size assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('power-sample-size');
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

test('power sample size assessment progresses from recall to interview readiness', () => {
  const { quiz } = getLessonAssessment('power-sample-size');
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

test('power sample size assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('power-sample-size');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));

  const orderedMilestones = [
    [/purpose of power and sample size planning/, 0, 8],
    [/minimum detectable effect/, 0, 8],
    [/sample size mainly reduce/, 1, 10],
    [/alpha control/, 2, 12],
    [/statistical power describe/, 3, 12],
    [/type i error/, 4, 14],
    [/type ii error/, 5, 14],
    [/false negative risk related to power/, 6, 15],
    [/baseline rate matter/, 7, 16],
    [/outcome variance matter/, 8, 18],
    [/standard error as sample size grows/, 10, 18],
    [/sample change is needed to halve standard error/, 11, 20],
    [/relative lift from absolute effect/, 12, 20],
    [/underpowered experiment/, 13, 20],
    [/non significant underpowered result/, 14, 20],
    [/first power check/, 18, 22],
    [/effect size divided by standard error/, 20, 28],
    [/lower alpha require/, 20, 30],
    [/increasing target power/, 21, 31],
    [/mde is made smaller/, 22, 32],
    [/halving the mde require about four times/, 23, 34],
    [/binary conversion metric/, 24, 35],
    [/per group sample/, 26, 36],
    [/imbalanced split affect precision/, 27, 38],
    [/confidence interval width/, 29, 40],
    [/post hoc power based on the observed effect/, 32, 42],
    [/lowered without increasing sample/, 33, 43],
    [/clustered users reduce effective sample/, 37, 46],
    [/missing outcomes matter for power/, 40, 48],
    [/traffic is split across many variants/, 43, 50],
    [/guardrails need their own sensitivity checks/, 45, 50],
    [/fixed horizon before launch/, 47, 50],
    [/power planning protocol/, 48, 52],
    [/team halves the mde/, 50, 57],
    [/35 percent power is non significant/, 51, 59],
    [/alpha 1 percent instead of 5 percent/, 52, 60],
    [/95 percent power instead of 80 percent/, 53, 61],
    [/baseline conversion is 2 percent/, 54, 62],
    [/halve the detectable absolute effect/, 55, 64],
    [/95 5 traffic split/, 56, 65],
    [/counts users as independent/, 57, 66],
    [/10 sessions per user/, 58, 67],
    [/planned split is 50 50/, 59, 68],
    [/sample will not be reached/, 62, 70],
    [/valid pre treatment covariate/, 64, 72],
    [/churn guardrail is rare and noisy/, 65, 73],
    [/primary metric after seeing/, 67, 74],
    [/planned n is below required n/, 72, 75],
    [/non significant underpowered test is false/, 75, 82],
    [/alpha claim is wrong/, 75, 83],
    [/mde claim is unsafe/, 76, 84],
    [/sample size claim is false/, 77, 85],
    [/large sample claim is unsafe/, 80, 88],
    [/post hoc power claim is misleading/, 83, 90],
    [/stopping claim is wrong/, 87, 90],
    [/define power in an interview/, 90, 96],
    [/complete power readout/, 95, 100],
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

test('power sample size assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('power-sample-size');
  const unsafePatterns = [
    /it proves the treatment has no useful effect/i,
    /lower alpha always increases power at the same sample size/i,
    /mde should be chosen after reading the observed lift/i,
    /doubling sample size halves standard error exactly/i,
    /a 5 percent relative lift always means a 5 percentage point absolute change/i,
    /variance is irrelevant once sample size is planned/i,
    /a large sample makes every statistically significant result worth launching/i,
    /every repeated session can always be counted as an independent user/i,
    /a 99\/1 split is just as powerful as 50\/50 for a fixed total sample/i,
    /observed-effect post-hoc power is the best proof that the null is true/i,
    /a powered primary metric means every guardrail is powered too/i,
    /adding variants never changes power if total traffic is fixed/i,
    /reaching sample during an abnormal event always represents normal behavior/i,
    /stopping when the p-value first looks good preserves the fixed-horizon design/i,
    /a powered test proves the exact treatment effect/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /trap|false|misleading|unsafe|too strong|wrong|claim/i.test(question.prompt);

    assert.ok(
      !unsafeAnswer || explicitTrapPrompt,
      `question ${index + 1} keys a false claim outside an explicit trap prompt`,
    );
  }
});

test('power sample size assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('power-sample-size');
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

test('power sample size assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('power-sample-size');
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
