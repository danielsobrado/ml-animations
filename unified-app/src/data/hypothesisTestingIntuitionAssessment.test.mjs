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

test('hypothesis testing intuition has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('hypothesis-testing-intuition');

  assert.equal(quiz.length, 100);
  assert.equal(labs.length, 3);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    const expectedNumber = String(index + 1).padStart(3, '0');

    assert.match(question.id, /^ht-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use the ht id format`);
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

test('hypothesis testing intuition assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('hypothesis-testing-intuition');
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

test('hypothesis testing intuition assessment progresses from recall to interview readiness', () => {
  const { quiz } = getLessonAssessment('hypothesis-testing-intuition');
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

test('hypothesis testing intuition assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('hypothesis-testing-intuition');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));

  const orderedMilestones = [
    [/hypothesis testing start/, 0, 8],
    [/null hypothesis/, 0, 8],
    [/alternative hypothesis/, 0, 8],
    [/what is an observed effect/, 2, 10],
    [/variability matter/, 4, 12],
    [/standard error play/, 5, 12],
    [/test statistic often compare/, 6, 14],
    [/null distribution/, 7, 15],
    [/p value measure/, 8, 16],
    [/alpha in a hypothesis test/, 9, 17],
    [/what does rejecting the null mean/, 10, 18],
    [/statistical significance/, 12, 20],
    [/practical significance/, 13, 20],
    [/type i error/, 18, 22],
    [/power related to/, 18, 22],
    [/divide observed effect by standard error/, 20, 28],
    [/tail area as extreme as the observed statistic/, 21, 30],
    [/common alpha decision rule/, 22, 30],
    [/minimum detectable effect/, 26, 34],
    [/confidence intervals connect to two sided tests/, 28, 36],
    [/p value alone does not show magnitude/, 29, 38],
    [/risk in testing a nonrandomized comparison/, 32, 40],
    [/permutation test/, 34, 42],
    [/many hypotheses are tested/, 36, 44],
    [/repeated unplanned p value checks/, 38, 46],
    [/predeclaring a test plan/, 39, 48],
    [/low power test vulnerable/, 43, 50],
    [/a b test shows a small conversion lift/, 50, 58],
    [/p 0 04 means a 4 percent chance/, 54, 62],
    [/tests 100 metrics/, 58, 66],
    [/checks every hour and stops/, 60, 68],
    [/one sided test is chosen after seeing/, 61, 70],
    [/fails to reject and says the feature has no effect/, 70, 75],
    [/p value claim is false/, 75, 83],
    [/significance claim is false/, 76, 84],
    [/multiple testing claim is unsafe/, 82, 89],
    [/define hypothesis testing in an interview/, 90, 100],
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

test('hypothesis testing intuition assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('hypothesis-testing-intuition');
  const unsafePatterns = [
    /the p-value is the probability that the null hypothesis is true/i,
    /statistical significance proves the alternative is true with certainty/i,
    /a significant result is automatically large enough to matter/i,
    /failing to reject proves the null hypothesis is true/i,
    /alpha is the probability that this rejected null is false/i,
    /power is the probability that the null is true/i,
    /more data always makes an effect practically important/i,
    /a tiny p-value from observational data proves the treatment caused the outcome/i,
    /testing many metrics needs no adjustment or exploratory label/i,
    /peeking until p < 0\.05 preserves a fixed 5 percent false-positive rate/i,
    /choose a one-sided test after seeing which direction won/i,
    /a confidence interval excluding zero proves the effect is useful/i,
    /software p-values are assumption-free because they came from a package/i,
    /p = 0\.049 and p = 0\.051 are opposite kinds of evidence/i,
    /report only significant findings after trying many tests/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /trap|false|misleading|unsafe|too strong|wrong|failure mode|correct|claim|practice/i.test(question.prompt);

    assert.ok(
      !unsafeAnswer || explicitTrapPrompt,
      `question ${index + 1} keys a false claim outside an explicit trap prompt`,
    );
  }
});

test('hypothesis testing intuition assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('hypothesis-testing-intuition');
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

test('hypothesis testing intuition assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('hypothesis-testing-intuition');
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
