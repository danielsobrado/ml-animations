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

test('logistic regression has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('logistic-regression');

  assert.equal(quiz.length, 100);
  assert.equal(labs.length, 3);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    const expectedNumber = String(index + 1).padStart(3, '0');

    assert.match(question.id, /^logreg-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use the logreg id scheme`);
    assert.equal(question.id.slice(7, 10), expectedNumber, `question ${index + 1} should be numerically ordered`);
    assert.ok(!question.id.includes('generated'), `question ${index + 1} should not use a generated id`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.equal(new Set(question.choices.map(normalized)).size, 3, `question ${index + 1} should have unique choices`);
    assert.equal(Number.isInteger(question.answerIndex), true, `question ${index + 1} answer index should be an integer`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(LEVELS.includes(question.level), `question ${index + 1} should have a recognized level`);
  }
});

test('logistic regression assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('logistic-regression');
  const promptOwners = new Map();
  const answerOwners = new Map();

  for (const [index, question] of quiz.entries()) {
    const prompt = normalized(question.prompt);
    const answer = normalized(correctAnswer(question));

    assert.ok(!promptOwners.has(prompt), `question ${index + 1} duplicates prompt from question ${promptOwners.get(prompt)}`);
    assert.ok(!answerOwners.has(answer), `question ${index + 1} duplicates correct answer from question ${answerOwners.get(answer)}`);
    promptOwners.set(prompt, index + 1);
    answerOwners.set(answer, index + 1);
  }
});

test('logistic regression assessment progresses from recall to interview readiness', () => {
  const { quiz } = getLessonAssessment('logistic-regression');
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

test('logistic regression assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('logistic-regression');
  const promptByQuestion = quiz.map((question) => normalized(question.prompt));
  const orderedMilestones = [
    [/what problem does binary logistic regression solve/, 0, 6],
    [/before applying sigmoid/, 1, 7],
    [/what does the sigmoid function do/, 2, 8],
    [/sigmoid output represent/, 3, 9],
    [/classification threshold do/, 4, 10],
    [/0 5 threshold mean/, 5, 11],
    [/threshold rises from 0 5 to 0 7/, 6, 12],
    [/threshold falls from 0 7 to 0 3/, 7, 13],
    [/decision boundary when threshold is 0 5/, 8, 14],
    [/what do logistic regression weights control/, 9, 15],
    [/bias term do/, 10, 16],
    [/positive weight usually imply/, 11, 17],
    [/negative weight usually imply/, 12, 18],
    [/loss is commonly used/, 13, 19],
    [/for a positive example/, 14, 20],
    [/for a negative example/, 15, 20],
    [/not just ordinary linear regression/, 16, 20],
    [/common mistake with a 0 61 score/, 17, 20],
    [/caveat applies to sigmoid outputs/, 18, 20],
    [/reviewing a logistic regression demo/, 19, 22],
    [/usual logit formula/, 20, 28],
    [/sigmoid formula in words/, 21, 29],
    [/sigmoid of zero/, 22, 30],
    [/logit is very positive/, 23, 31],
    [/logit is very negative/, 24, 32],
    [/increasing the logit increase/, 25, 33],
    [/changing the bias affect the boundary/, 26, 34],
    [/decision boundary in original feature space/, 27, 35],
    [/logit represent statistically/, 28, 36],
    [/coefficient be interpreted through odds/, 29, 37],
    [/for y 1/, 30, 38],
    [/for y 0/, 31, 39],
    [/confident wrong predictions punished strongly/, 32, 40],
    [/positive example gets too low a score/, 33, 41],
    [/perfectly separable data/, 34, 42],
    [/why add regularization/, 35, 43],
    [/feature scaling matter/, 36, 44],
    [/move only the threshold after training/, 37, 45],
    [/threshold differ from 0 5/, 38, 46],
    [/class imbalance affect logistic regression decisions/, 39, 47],
    [/changing the threshold not change/, 40, 48],
    [/calibration checked separately/, 41, 49],
    [/correlated features do to coefficient interpretation/, 42, 50],
    [/plain linear logit have/, 43, 50],
    [/avoid when interpreting coefficients/, 44, 50],
    [/monitor validation loss/, 45, 50],
    [/threshold decision report include/, 46, 50],
    [/diagnostics fit this lesson/, 47, 50],
    [/train from logits directly/, 48, 50],
    [/logistic regression fitting protocol/, 49, 52],
    [/point has logit 0/, 50, 58],
    [/large positive logit/, 51, 59],
    [/large negative logit/, 52, 60],
    [/score 0 62 and threshold 0 7/, 53, 61],
    [/score 0 42 and threshold 0 3/, 54, 62],
    [/raise the threshold after training/, 55, 63],
    [/lower the threshold after training/, 56, 64],
    [/missing a positive case is very expensive/, 57, 65],
    [/false alarms are very costly/, 58, 66],
    [/feature weight is positive/, 59, 67],
    [/feature weight is negative/, 60, 68],
    [/bias is increased/, 61, 69],
    [/weights explode on separable data/, 62, 70],
    [/measured in dollars and another in millions/, 63, 71],
    [/features are almost duplicates/, 64, 72],
    [/true boundary is curved/, 65, 73],
    [/scores look overconfident/, 66, 74],
    [/positive label receives predicted probability 0 01/, 67, 75],
    [/negative label receives predicted probability 0 99/, 68, 75],
    [/custom implementation overflows/, 69, 75],
    [/deployment base rate differs from training/, 70, 75],
    [/only coefficients and no threshold/, 71, 75],
    [/score above 0 8/, 72, 75],
    [/0 6 score guarantees/, 73, 75],
    [/before shipping logistic regression/, 74, 75],
    [/which logistic regression claim is false/, 75, 83],
    [/which sigmoid claim is wrong/, 76, 84],
    [/which threshold claim is unsafe/, 77, 85],
    [/which boundary claim is false/, 78, 86],
    [/which calibration claim is misleading/, 79, 87],
    [/which probability claim is wrong/, 80, 88],
    [/which coefficient claim is unsafe/, 81, 89],
    [/which reporting claim is false/, 89, 90],
    [/define logistic regression in an interview/, 90, 96],
    [/explain sigmoid/, 91, 97],
    [/training objective/, 92, 98],
    [/threshold choice/, 93, 99],
    [/interpret coefficients carefully/, 94, 100],
    [/explain the decision boundary/, 95, 100],
    [/discuss regularization/, 96, 100],
    [/calibration caveat should you mention/, 97, 100],
    [/limitation should accompany a logistic regression answer/, 98, 100],
    [/interview ready mastery of logistic regression/, 99, 100],
  ];

  for (const [pattern, minInclusive, maxExclusive] of orderedMilestones) {
    const index = promptByQuestion.findIndex((text) => pattern.test(text));

    assert.notEqual(index, -1, `missing milestone: ${pattern}`);
    assert.ok(
      index >= minInclusive && index < maxExclusive,
      `${pattern} should appear in questions ${minInclusive + 1}-${maxExclusive}, found ${index + 1}`,
    );
  }
});

test('logistic regression assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('logistic-regression');
  const unsafePatterns = [
    /it predicts an unbounded continuous value as the final output/i,
    /sigmoid maps every input to only 0 or 1 exactly/i,
    /changing the threshold retrains all model weights automatically/i,
    /basic logistic regression always learns a nonlinear boundary/i,
    /any sigmoid output is automatically calibrated because it is between 0 and 1/i,
    /a 0\.61 score guarantees the true class is positive/i,
    /predictive coefficients automatically prove causal effects/i,
    /regularization removes the need for validation/i,
    /feature scaling never matters for regularized logistic regression/i,
    /perfect separability always gives small stable unregularized weights/i,
    /the best threshold is always exactly 0\.5/i,
    /binary cross-entropy ignores the observed label/i,
    /a logit is already a valid probability before sigmoid/i,
    /class imbalance never affects threshold or calibration decisions/i,
    /only the final hard labels are needed, without scores or threshold/i,
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

test('logistic regression assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('logistic-regression');
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

test('logistic regression assessment distributes correct-answer positions across pages and total', () => {
  const { quiz } = getLessonAssessment('logistic-regression');
  const totals = [0, 0, 0];

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const page = quiz.slice(pageStart, pageStart + 10);
    const positions = page.map((question) => question.answerIndex);

    for (const position of positions) totals[position] += 1;

    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
  }

  assert.ok(Math.max(...totals) - Math.min(...totals) <= 1, `answer positions should be balanced, got ${totals.join('/')}`);
});
