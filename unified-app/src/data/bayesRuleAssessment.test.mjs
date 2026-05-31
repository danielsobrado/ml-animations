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

test('bayes rule ml has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('bayes-rule-ml');

  assert.equal(quiz.length, 100);
  assert.equal(labs.length, 3);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    const expectedNumber = String(index + 1).padStart(3, '0');

    assert.match(question.id, /^bayes-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use the bayes id format`);
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

test('bayes rule ml assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('bayes-rule-ml');
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

test('bayes rule ml assessment progresses from recall to interview readiness', () => {
  const { quiz } = getLessonAssessment('bayes-rule-ml');
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

test('bayes rule ml assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('bayes-rule-ml');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));

  const orderedMilestones = [
    [/main purpose of bayes rule/, 0, 8],
    [/prior probability/, 0, 8],
    [/posterior probability/, 0, 8],
    [/likelihood term/, 0, 10],
    [/bayes denominator represent/, 4, 12],
    [/base rate in a classification setting/, 6, 14],
    [/sensitivity or true positive rate/, 6, 16],
    [/false positive rate/, 6, 16],
    [/rare class/, 8, 18],
    [/distinguish p a b from p b a/, 10, 18],
    [/numerator of bayes rule/, 20, 28],
    [/false positive mass/, 20, 32],
    [/posterior odds equal prior odds times the likelihood ratio/, 22, 34],
    [/precision and recall/, 28, 36],
    [/naive assumption in naive bayes/, 34, 42],
    [/compute bayes products in log space/, 36, 44],
    [/smoothing in naive bayes/, 36, 45],
    [/prior probability shift/, 38, 48],
    [/check probability calibration/, 40, 50],
    [/rare disease/, 50, 60],
    [/repeated synonyms/, 58, 68],
    [/recall and calls it the probability an alert is correct/, 62, 70],
    [/content filter/, 70, 75],
    [/sensitivity claim is false/, 75, 83],
    [/conditioning statement is wrong/, 76, 84],
    [/define bayes rule in an interview/, 90, 100],
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

test('bayes rule ml assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('bayes-rule-ml');
  const unsafePatterns = [
    /high sensitivity alone guarantees a high posterior after a positive signal/i,
    /the prior can be ignored once evidence sounds strong/i,
    /p\(class \| alert\) is always the same as p\(alert \| class\)/i,
    /use only the true-positive path because the alert was positive/i,
    /rare classes cannot have useful alerts/i,
    /the independence assumption is always exactly true in real text/i,
    /any model score between 0 and 1 is automatically a calibrated posterior/i,
    /the best action threshold is always 0\.5/i,
    /two correlated positive signals should always be treated as independent confirmations/i,
    /a posterior calibrated under one prevalence is guaranteed calibrated after prevalence changes/i,
    /false positive rate is irrelevant when sensitivity is high/i,
    /smoothing proves the model assumptions are correct/i,
    /a negative signal can never be informative/i,
    /report a posterior without stating the prior or error rates/i,
    /bayes rule turns any nonzero evidence into certainty/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /trap|false|misleading|unsafe|too strong|wrong|challenge|claim|shortcut|practice|statement/i.test(question.prompt);

    assert.ok(
      !unsafeAnswer || explicitTrapPrompt,
      `question ${index + 1} keys a false claim outside an explicit trap prompt`,
    );
  }
});

test('bayes rule ml assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('bayes-rule-ml');
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

test('bayes rule ml assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('bayes-rule-ml');
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
