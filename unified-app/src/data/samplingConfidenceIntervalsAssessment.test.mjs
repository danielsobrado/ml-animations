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

test('sampling confidence intervals has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('sampling-confidence-intervals');

  assert.equal(quiz.length, 100);
  assert.equal(labs.length, 3);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    const expectedNumber = String(index + 1).padStart(3, '0');

    assert.match(question.id, /^ci-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use the ci id format`);
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

test('sampling confidence intervals assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('sampling-confidence-intervals');
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

test('sampling confidence intervals assessment progresses from recall to interview readiness', () => {
  const { quiz } = getLessonAssessment('sampling-confidence-intervals');
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

test('sampling confidence intervals assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('sampling-confidence-intervals');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));

  const orderedMilestones = [
    [/confidence interval solve/, 0, 8],
    [/what is a point estimate/, 0, 8],
    [/population parameter/, 0, 8],
    [/repeated samples produce different estimates/, 2, 10],
    [/sampling distribution/, 3, 11],
    [/standard error measure/, 4, 12],
    [/margin of error/, 6, 14],
    [/95 percent confidence level describe/, 8, 16],
    [/one interval is computed/, 8, 16],
    [/larger sample/, 10, 18],
    [/quadrupling sample size/, 10, 18],
    [/99 percent interval usually wider/, 12, 20],
    [/sampling design matter/, 12, 20],
    [/central limit theorem/, 14, 22],
    [/usual standard error estimate/, 20, 28],
    [/critical value times standard error/, 22, 30],
    [/t critical values larger/, 24, 32],
    [/normal approximation for a proportion/, 24, 34],
    [/wilson interval/, 26, 36],
    [/bootstrap sample drawn/, 28, 38],
    [/coverage simulation/, 32, 42],
    [/confidence interval different from a prediction interval/, 34, 42],
    [/clustering do to a naive interval/, 38, 46],
    [/many intervals are reported/, 44, 50],
    [/practical threshold/, 46, 52],
    [/poll estimates support/, 50, 58],
    [/sample size change is needed/, 52, 60],
    [/95 percent refers to long run procedure coverage/, 54, 62],
    [/study samples 1000 rows from only 10 households/, 58, 66],
    [/before and after study measures the same users twice/, 60, 68],
    [/dashboard shows 50 subgroup intervals/, 64, 72],
    [/recomputes intervals after each new user/, 66, 74],
    [/confidence interval claim is false/, 75, 83],
    [/bootstrap claim is too strong/, 80, 88],
    [/define a confidence interval in an interview/, 90, 100],
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

test('sampling confidence intervals assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('sampling-confidence-intervals');
  const unsafePatterns = [
    /a 95 percent interval guarantees the parameter is inside this one range/i,
    /a 95 percent confidence interval contains exactly 95 percent of sample rows/i,
    /doubling sample size cuts margin of error in half/i,
    /raising confidence always narrows the interval/i,
    /a narrow interval proves the sample was unbiased/i,
    /bootstrapping automatically fixes biased data collection/i,
    /correlated rows always provide the same information as independent rows/i,
    /the simple wald interval is always reliable near zero or one/i,
    /overlapping 95 percent intervals always prove two groups are not different/i,
    /any statistically nonzero interval is automatically worth acting on/i,
    /a printed interval is assumption-free because the package computed it/i,
    /coverage is unaffected no matter how often you peek and stop selectively/i,
    /a confidence interval for the mean always contains the next individual observation/i,
    /it is fine for a probability interval to imply impossible negative probabilities without review/i,
    /report only the narrowest interval found after trying many methods/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /trap|false|misleading|unsafe|too strong|wrong|misconception|correct|shortcut|claim|practice/i.test(question.prompt);

    assert.ok(
      !unsafeAnswer || explicitTrapPrompt,
      `question ${index + 1} keys a false claim outside an explicit trap prompt`,
    );
  }
});

test('sampling confidence intervals assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('sampling-confidence-intervals');
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

test('sampling confidence intervals assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('sampling-confidence-intervals');
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
