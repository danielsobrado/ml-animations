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

test('maximum likelihood estimation has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('maximum-likelihood-estimation');

  assert.equal(quiz.length, 100);
  assert.equal(labs.length, 3);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    const expectedNumber = String(index + 1).padStart(3, '0');

    assert.match(question.id, /^mle-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use the mle id scheme`);
    assert.equal(question.id.slice(4, 7), expectedNumber, `question ${index + 1} should be numerically ordered`);
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

test('maximum likelihood estimation assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('maximum-likelihood-estimation');
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

test('maximum likelihood estimation assessment progresses from recall to interview readiness', () => {
  const { quiz } = getLessonAssessment('maximum-likelihood-estimation');
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

test('maximum likelihood estimation assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('maximum-likelihood-estimation');
  const promptByQuestion = quiz.map((question) => normalized(question.prompt));
  const orderedMilestones = [
    [/what problem does maximum likelihood estimation solve/, 0, 6],
    [/what does likelihood measure/, 1, 7],
    [/what is a parameter in the mle lesson/, 2, 8],
    [/what is a candidate parameter/, 3, 9],
    [/why does observed data matter in likelihood/, 4, 10],
    [/what does argmax mean/, 5, 11],
    [/what does a bernoulli model describe/, 6, 12],
    [/where does the mle for p land/, 7, 13],
    [/how do failures affect bernoulli likelihood/, 8, 14],
    [/6 of 10 trials are successes/, 9, 15],
    [/2 of 12 trials are successes/, 10, 16],
    [/9 of 10 trials are successes/, 11, 17],
    [/what does the gaussian mean example fit/, 12, 18],
    [/fixed variance gaussian data/, 13, 19],
    [/what are residuals/, 14, 20],
    [/why use log likelihood/, 15, 20],
    [/what is negative log likelihood/, 16, 20],
    [/what does relative likelihood compare/, 17, 20],
    [/common likelihood interpretation mistake/, 18, 20],
    [/first in the mle workbench/, 19, 22],
    [/bernoulli likelihood for s successes and f failures/, 20, 28],
    [/bernoulli log likelihood form/, 21, 29],
    [/why clamp p away from 0 and 1/, 22, 30],
    [/likelihood peak near the observed rate/, 23, 31],
    [/more data punish wrong candidate probabilities/, 24, 32],
    [/sharper likelihood curve/, 25, 33],
    [/flatter likelihood curve/, 26, 34],
    [/gaussian log likelihood penalize/, 27, 35],
    [/what role does sigma play/, 28, 36],
    [/sample mean the fixed variance gaussian mle/, 29, 37],
    [/why do training loops often minimize nll/, 30, 38],
    [/raw likelihood products numerically fragile/, 31, 39],
    [/multiplying observation likelihoods/, 32, 40],
    [/what does iid mean/, 33, 41],
    [/why does the model family matter/, 34, 42],
    [/what is model misspecification/, 35, 43],
    [/how can mle overfit/, 36, 44],
    [/regularization interact with mle/, 37, 45],
    [/map different from pure mle/, 38, 46],
    [/why is relative likelihood useful/, 39, 47],
    [/likelihood shape relate to uncertainty/, 40, 48],
    [/likelihood ratio compare/, 41, 49],
    [/bernoulli p what summary is sufficient/, 42, 50],
    [/zero probability to observed data/, 44, 50],
    [/careful statement about comparing different model families/, 45, 50],
    [/why use validation after fitting by mle/, 46, 50],
    [/diagnostics belong with mle/, 47, 50],
    [/mle report include/, 48, 50],
    [/mle fitting protocol/, 49, 52],
    [/80 of 100 bernoulli trials/, 50, 58],
    [/2 successes and 10 failures/, 51, 59],
    [/9 successes and 1 failure/, 52, 60],
    [/candidate p 0 2/, 53, 61],
    [/one success and one failure/, 54, 62],
    [/trials increase from 10 to 1000/, 55, 63],
    [/measurements cluster around 5 0/, 56, 64],
    [/measurements shift upward around 6 7/, 57, 65],
    [/gaussian dataset becomes noisier/, 58, 66],
    [/candidate mean is far from every gaussian observation/, 59, 67],
    [/log likelihood is 12 5/, 60, 68],
    [/nll 4 0 and 9 0/, 61, 69],
    [/relative likelihood 0 05/, 62, 70],
    [/candidate p 0 predicts impossible failures/, 63, 71],
    [/likelihood is the chance theta is true/, 64, 72],
    [/prior strongly favors smaller parameters/, 65, 73],
    [/flexible model assigns extreme likelihood/, 66, 74],
    [/bernoulli likelihood is used for continuous measurements/, 67, 75],
    [/product of thousands of probabilities underflows/, 68, 75],
    [/minimizes negative log likelihood/, 69, 75],
    [/slider is left of the likelihood peak/, 70, 75],
    [/same successes and failures in different orders/, 71, 75],
    [/constant term is added/, 72, 75],
    [/after finding the mle/, 73, 75],
    [/candidate far from the mle/, 74, 75],
    [/which mle claim is false/, 75, 83],
    [/which likelihood claim is wrong/, 76, 84],
    [/which nll claim is unsafe/, 77, 85],
    [/which bernoulli mle claim is false/, 78, 86],
    [/which gaussian mean claim is wrong/, 79, 87],
    [/which log likelihood claim is misleading/, 80, 88],
    [/which zero probability claim is false/, 81, 89],
    [/which reporting claim is unsafe/, 89, 90],
    [/define mle in an interview/, 90, 96],
    [/distinguish likelihood from probability of a parameter/, 91, 97],
    [/derive the bernoulli mle intuitively/, 92, 98],
    [/explain gaussian mean mle/, 93, 99],
    [/why do ml systems minimize negative log likelihood/, 94, 100],
    [/more data sharpening likelihood/, 95, 100],
    [/caveat should accompany every mle answer/, 96, 100],
    [/diagnostics should accompany an mle fit/, 97, 100],
    [/compare mle and map/, 98, 100],
    [/interview ready mastery of mle/, 99, 100],
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

test('maximum likelihood estimation assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('maximum-likelihood-estimation');
  const unsafePatterns = [
    /mle proves the model family is true/i,
    /likelihood is the prior probability that theta is true/i,
    /minimizing nll is unrelated to maximizing likelihood/i,
    /mle for p is always 0\.5 regardless of data/i,
    /fixed-variance gaussian mle ignores residuals/i,
    /taking logs changes which candidate is best/i,
    /candidate assigning zero probability to observed data is safest/i,
    /maximum training likelihood guarantees best generalization/i,
    /more data makes all candidate parameters equally plausible/i,
    /relative likelihood is the posterior probability of the candidate/i,
    /failures can be ignored when estimating p/i,
    /mle automatically chooses the correct model family/i,
    /mle and map always include the same prior information/i,
    /high likelihood automatically gives a valid p-value/i,
    /only the final mle is needed, without data or assumptions/i,
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

test('maximum likelihood estimation assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('maximum-likelihood-estimation');
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

test('maximum likelihood estimation assessment distributes correct-answer positions across pages and total', () => {
  const { quiz } = getLessonAssessment('maximum-likelihood-estimation');
  const totals = [0, 0, 0];

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const page = quiz.slice(pageStart, pageStart + 10);
    const positions = page.map((question) => question.answerIndex);

    for (const position of positions) totals[position] += 1;

    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
  }

  assert.ok(Math.max(...totals) - Math.min(...totals) <= 1, `answer positions should be balanced, got ${totals.join('/')}`);
});
