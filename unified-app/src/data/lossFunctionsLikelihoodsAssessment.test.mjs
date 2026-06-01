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
const EXPECTED_LEVEL_COUNTS = {
  Foundation: 20,
  Mechanism: 30,
  Application: 25,
  Tricky: 15,
  Interview: 10,
};

function normalized(value) {
  return String(value || '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim()
    .replace(/\s+/g, ' ');
}

function correctAnswer(question) {
  return question.choices[question.answerIndex];
}

test('loss functions likelihoods has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('loss-functions-likelihoods');

  assert.equal(quiz.length, 100);
  assert.deepEqual(labs.map((lab) => lab.id), [
    'match-loss-to-noise',
    'outlier-and-confidence-diagnostic',
    'loss-selection-report',
  ]);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);
  assert.deepEqual(
    quiz.reduce((counts, question) => {
      counts[question.level] = (counts[question.level] || 0) + 1;
      return counts;
    }, {}),
    EXPECTED_LEVEL_COUNTS,
  );

  for (const [index, question] of quiz.entries()) {
    const expectedNumber = String(index + 1).padStart(3, '0');

    assert.match(question.id, /^losslik-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use the losslik id scheme`);
    assert.equal(question.id.slice(8, 11), expectedNumber, `question ${index + 1} should be numerically ordered`);
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

test('loss functions likelihoods assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('loss-functions-likelihoods');
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

test('loss functions likelihoods assessment progresses from recall to interview readiness', () => {
  const { quiz } = getLessonAssessment('loss-functions-likelihoods');
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

test('loss functions likelihoods assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('loss-functions-likelihoods');
  const promptByQuestion = quiz.map((question) => normalized(question.prompt));
  const orderedMilestones = [
    [/what does a loss function tell a training algorithm/, 0, 6],
    [/why does the observed target matter/, 1, 7],
    [/what does a lower loss mean/, 2, 8],
    [/core link between many ml losses and probability models/, 3, 9],
    [/what does likelihood contribute to a loss function/, 4, 10],
    [/why is the negative log used/, 5, 11],
    [/near zero probability to the true class/, 6, 12],
    [/what does choosing a loss usually encode/, 7, 13],
    [/ordinary squared error usually meant for/, 8, 14],
    [/cross entropy usually meant for/, 9, 15],
    [/what is a residual in regression loss/, 10, 16],
    [/what does squared error do to a residual/, 11, 17],
    [/basic idea of absolute error/, 12, 18],
    [/what does cross entropy penalize/, 13, 19],
    [/binary cross entropy used for/, 14, 20],
    [/multiclass cross entropy need/, 15, 20],
    [/assumption connects squared error to likelihood/, 16, 20],
    [/assumption connects cross entropy to likelihood/, 17, 20],
    [/constants in an nll derivation/, 18, 20],
    [/before choosing a loss/, 19, 22],
    [/independent examples produce a summed nll/, 20, 28],
    [/losses in log space/, 21, 29],
    [/fixed variance gaussian nll reduce to squared error/, 22, 30],
    [/role does gaussian variance play/, 23, 31],
    [/noise story is commonly associated with absolute error/, 24, 32],
    [/squared error sensitive to outliers/, 25, 33],
    [/binary cross entropy combine/, 26, 34],
    [/softmax plus cross entropy score/, 27, 35],
    [/punish confident wrong predictions strongly/, 28, 36],
    [/probability loss different from a pure margin/, 29, 37],
    [/nll style probability losses useful for calibrated predictions/, 30, 38],
    [/argmin mean for a loss function/, 31, 39],
    [/loss shape matter for gradient based training/, 32, 40],
    [/loss scale matter/, 33, 41],
    [/class weights or example weights/, 34, 42],
    [/class imbalance require loss adjustments/, 35, 43],
    [/label smoothing change/, 36, 44],
    [/intent of focal loss/, 37, 45],
    [/why use huber loss/, 38, 46],
    [/quantile loss estimate/, 39, 47],
    [/ranking losses different/, 40, 48],
    [/what is a surrogate loss/, 41, 49],
    [/optimizing a loss fail to improve the metric/, 42, 50],
    [/classification loss match the output representation/, 43, 50],
    [/combine logits and cross entropy/, 44, 50],
    [/mean versus sum reduction/, 45, 50],
    [/regularization different from the data fit loss/, 46, 50],
    [/why still use validation/, 47, 50],
    [/diagnostics help evaluate a chosen loss/, 48, 50],
    [/loss selection protocol/, 49, 52],
    [/house prices as continuous dollars/, 50, 58],
    [/one of five product categories/, 51, 59],
    [/predicted probability 0 001/, 52, 60],
    [/predicted probability 0 9/, 53, 61],
    [/huge recording error/, 54, 62],
    [/occasional extreme outliers/, 55, 63],
    [/heavy tailed and median accuracy/, 56, 64],
    [/fraud classifier has one positive/, 57, 65],
    [/false negatives are much more expensive/, 58, 66],
    [/overconfident on noisy labels/, 59, 67],
    [/search system cares most about the order/, 60, 68],
    [/product metric is top k recall/, 61, 69],
    [/logits rather than softmax probabilities/, 62, 70],
    [/mean to sum/, 63, 71],
    [/drop a constant from a gaussian nll/, 64, 72],
    [/multiplying a loss by 100/, 65, 73],
    [/accurate but poorly calibrated/, 66, 74],
    [/count target has rare huge spikes/, 67, 75],
    [/unbounded scores for classes/, 68, 75],
    [/p 0 8 for a positive example/, 69, 75],
    [/p 0 8 for a negative example/, 70, 75],
    [/90th percentile rather than the mean/, 71, 75],
    [/validation loss rises while training loss falls/, 72, 75],
    [/report says only loss went down/, 73, 75],
    [/before shipping a model trained with a new loss/, 74, 75],
    [/which loss function claim is false/, 75, 83],
    [/which nll claim is wrong/, 76, 84],
    [/which squared error claim is unsafe/, 77, 85],
    [/which cross entropy claim is false/, 78, 86],
    [/which constant term claim is misleading/, 79, 87],
    [/which outlier claim is wrong/, 80, 88],
    [/which imbalance claim is unsafe/, 81, 89],
    [/which reporting claim is false/, 89, 90],
    [/define a loss function in an interview/, 90, 96],
    [/negative log likelihood as a loss/, 91, 97],
    [/connect squared error to likelihood/, 92, 98],
    [/connect cross entropy to likelihood/, 93, 99],
    [/loss choice under outliers/, 94, 100],
    [/loss versus metric/, 95, 100],
    [/class imbalance in loss selection/, 96, 100],
    [/probability calibration/, 97, 100],
    [/caveat should accompany any loss recommendation/, 98, 100],
    [/interview ready mastery of losses and likelihoods/, 99, 100],
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

test('loss functions likelihoods assessment keeps misconception traps after setup', () => {
  const { quiz } = getLessonAssessment('loss-functions-likelihoods');
  const misconceptionPatterns = [
    /one universal loss is best for every ml task/i,
    /minimizing nll is unrelated to maximizing likelihood/i,
    /squared error is always best for classification labels/i,
    /cross-entropy ignores the probability assigned to the true class/i,
    /dropping parameter-independent constants changes the optimum/i,
    /squared error is unaffected by huge residuals/i,
    /class imbalance never affects loss design/i,
    /lower training loss guarantees better production performance/i,
    /high accuracy automatically means calibrated probabilities/i,
    /a surrogate loss always exactly optimizes the final business metric/i,
    /using logits means the loss no longer represents class likelihood/i,
    /multiplying a loss can never affect optimization dynamics/i,
    /class weights remove the need to evaluate class-specific metrics/i,
    /the loss can be chosen without looking at target type/i,
    /only the final loss number is needed, without assumptions or metrics/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const misconceptionAnswer = misconceptionPatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /false|misleading|unsafe|wrong|trap|claim/i.test(question.prompt);

    assert.ok(
      !misconceptionAnswer || index >= 75,
      `question ${index + 1} keys a misconception before the tricky band`,
    );
    assert.ok(
      !misconceptionAnswer || explicitTrapPrompt,
      `question ${index + 1} keys a false claim outside an explicit trap prompt`,
    );
  }
});

test('loss functions likelihoods assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('loss-functions-likelihoods');
  const pageSize = 10;

  for (let pageStart = 0; pageStart < quiz.length; pageStart += pageSize) {
    const page = quiz.slice(pageStart, pageStart + pageSize);
    const answers = page.map((question) => normalized(correctAnswer(question)));

    assert.equal(new Set(answers).size, answers.length, `page starting at question ${pageStart + 1} should not repeat exact answers`);

    for (const [answerIndex, answer] of answers.entries()) {
      for (const [promptIndex, question] of page.entries()) {
        if (answerIndex === promptIndex) continue;
        assert.ok(
          !normalized([question.prompt, ...question.choices].join(' ')).includes(answer),
          `question ${pageStart + promptIndex + 1} visible text should not reveal answer from question ${pageStart + answerIndex + 1}`,
        );
      }
    }
  }
});

test('loss functions likelihoods assessment stays within visible lesson scope', () => {
  const { quiz } = getLessonAssessment('loss-functions-likelihoods');
  const outOfScopePatterns = [
    /\bui\b/i,
    /button/i,
    /rows should be hidden/i,
    /validation set was deleted/i,
    /chart theme/i,
    /feature columns/i,
    /row id/i,
    /row number/i,
    /sorted alphabetically/i,
    /hidden validation label/i,
    /image pixels/i,
    /rounded final loss/i,
    /file imported/i,
    /hidden units/i,
    /chart title color/i,
    /gpu temperature/i,
    /chart only/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const visibleText = [question.prompt, ...question.choices].join(' ');
    for (const pattern of outOfScopePatterns) {
      assert.ok(!pattern.test(visibleText), `question ${index + 1} has out-of-scope visible text: ${pattern}`);
    }
  }
});

test('loss functions likelihoods assessment distributes correct-answer positions across pages and total', () => {
  const { quiz } = getLessonAssessment('loss-functions-likelihoods');
  const totals = [0, 0, 0];

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const page = quiz.slice(pageStart, pageStart + 10);
    const pageCounts = [0, 0, 0];

    for (const question of page) {
      totals[question.answerIndex] += 1;
      pageCounts[question.answerIndex] += 1;
    }

    assert.ok(
      Math.max(...pageCounts) - Math.min(...pageCounts) <= 1,
      `page starting at question ${pageStart + 1} should balance answer positions, got ${pageCounts.join('/')}`,
    );
  }

  assert.ok(Math.max(...totals) - Math.min(...totals) <= 1, `answer positions should be balanced, got ${totals.join('/')}`);
});
