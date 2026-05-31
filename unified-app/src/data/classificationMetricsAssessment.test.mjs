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

test('classification metrics has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('classification-metrics');

  assert.equal(quiz.length, 100);
  assert.deepEqual(labs.map((lab) => lab.id), [
    'threshold-counts',
    'metric-denominator-audit',
    'cost-and-slice-report',
  ]);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    const expectedNumber = String(index + 1).padStart(3, '0');

    assert.match(question.id, /^clfmet-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use the clfmet id scheme`);
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

test('classification metrics assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('classification-metrics');
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

test('classification metrics assessment progresses from recall to interview readiness', () => {
  const { quiz } = getLessonAssessment('classification-metrics');
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

test('classification metrics assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('classification-metrics');
  const promptByQuestion = quiz.map((question) => normalized(question.prompt));
  const orderedMilestones = [
    [/main purpose of classification metrics/, 0, 6],
    [/what does a confusion matrix count/, 1, 7],
    [/what is a true positive/, 2, 8],
    [/what is a false positive/, 3, 9],
    [/what is a false negative/, 4, 10],
    [/what is a true negative/, 5, 11],
    [/what does accuracy measure/, 6, 12],
    [/what does precision measure/, 7, 13],
    [/what does recall measure/, 8, 14],
    [/what does specificity measure/, 9, 15],
    [/what is the false positive rate/, 10, 16],
    [/what is the false negative rate/, 11, 17],
    [/what does f1 combine/, 12, 18],
    [/moving a classification threshold change/, 13, 19],
    [/positive threshold is raised/, 14, 20],
    [/positive threshold is lowered/, 15, 20],
    [/accuracy be misleading with class imbalance/, 16, 20],
    [/mistake costs matter/, 17, 20],
    [/positive class be defined clearly/, 18, 20],
    [/reading metric tiles/, 19, 22],
    [/accuracy formula/, 20, 28],
    [/precision formula/, 21, 29],
    [/recall formula/, 22, 30],
    [/specificity formula/, 23, 31],
    [/false positive rate related to specificity/, 24, 32],
    [/false negative rate related to recall/, 25, 33],
    [/f1 formula in terms of precision and recall/, 26, 34],
    [/f1 drop when either precision or recall/, 27, 35],
    [/raising a threshold improve precision/, 28, 36],
    [/recall when the threshold is raised/, 29, 37],
    [/lowering a threshold improve recall/, 30, 38],
    [/precision when the threshold is lowered/, 31, 39],
    [/majority class baseline/, 32, 40],
    [/balanced accuracy average/, 33, 41],
    [/negative predictive value measure/, 34, 42],
    [/prevalence in a binary classification dataset/, 35, 43],
    [/predicted positive rate or selection rate/, 36, 44],
    [/check metric denominators/, 37, 45],
    [/precision be undefined/, 38, 46],
    [/recall be undefined or uninformative/, 39, 47],
    [/micro averaging do/, 40, 48],
    [/macro averaging do/, 41, 49],
    [/support weighted average do/, 42, 50],
    [/metrics by slice or subgroup/, 43, 50],
    [/state the threshold with metric results/, 44, 50],
    [/distinguish scores from labels/, 45, 50],
    [/validation or test data/, 46, 50],
    [/classification metrics too optimistic/, 47, 50],
    [/metric report include/, 48, 50],
    [/classification metric protocol/, 49, 52],
    [/tp 30 and fp 10/, 50, 58],
    [/tp 30 and fn 20/, 51, 59],
    [/tp 30 tn 50 fp 10 and fn 10/, 52, 60],
    [/tn 90 and fp 10/, 53, 61],
    [/fp 5 and tn 95/, 54, 62],
    [/precision is 0 8 and recall is 0 8/, 55, 63],
    [/precision is 0 95 but recall is 0 10/, 56, 64],
    [/recall is 0 95 but precision is 0 10/, 57, 65],
    [/99 percent negative/, 58, 66],
    [/predicts every example positive/, 59, 67],
    [/predicts no positives/, 60, 68],
    [/missing disease is very costly/, 61, 69],
    [/blocking legitimate mail is very costly/, 62, 70],
    [/fraud team can review only a small queue/, 63, 71],
    [/threshold is raised and predicted positives drop/, 64, 72],
    [/threshold is lowered and predicted positives rise/, 65, 73],
    [/false negatives become more expensive/, 66, 74],
    [/recall is poor for rural users/, 67, 75],
    [/rare class/, 68, 75],
    [/overall example weighted performance/, 69, 75],
    [/does not define the positive class/, 70, 75],
    [/final test set/, 71, 75],
    [/92 percent accuracy/, 72, 75],
    [/precision and recall both matter/, 73, 75],
    [/before shipping a classifier/, 74, 75],
    [/which accuracy claim is false/, 75, 83],
    [/which precision claim is wrong/, 76, 84],
    [/which recall claim is false/, 77, 85],
    [/which f1 claim is misleading/, 78, 86],
    [/which threshold claim is unsafe/, 79, 87],
    [/which imbalance claim is false/, 80, 88],
    [/which cost claim is wrong/, 81, 89],
    [/which reporting claim is false/, 89, 90],
    [/confusion matrix in an interview/, 90, 96],
    [/distinguish precision from recall/, 91, 97],
    [/caveat accuracy/, 92, 98],
    [/explain f1/, 93, 99],
    [/threshold movement/, 94, 100],
    [/imbalanced classification metrics/, 95, 100],
    [/metrics for a real application/, 96, 100],
    [/micro versus macro averaging/, 97, 100],
    [/evaluation caveat/, 98, 100],
    [/interview ready mastery of classification metrics/, 99, 100],
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

test('classification metrics assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('classification-metrics');
  const unsafePatterns = [
    /high accuracy always proves the classifier is useful/i,
    /precision measures how many actual positives were recovered/i,
    /recall measures how many predicted positives were correct/i,
    /f1 includes true negatives directly/i,
    /changing threshold cannot change precision or recall/i,
    /class imbalance never affects metric interpretation/i,
    /false positives and false negatives always have equal cost/i,
    /precision and recall use the same denominator/i,
    /specificity is the same as recall for the positive class/i,
    /false positive rate is computed over predicted positives/i,
    /precision is automatically perfect when no positives are predicted/i,
    /aggregate metrics always reveal subgroup failures/i,
    /safe to tune thresholds repeatedly on the final test set/i,
    /precision and recall mean the same thing no matter which class is positive/i,
    /only one rounded metric is enough for production review/i,
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

test('classification metrics assessment keeps misconception traps after setup', () => {
  const { quiz } = getLessonAssessment('classification-metrics');
  const misconceptionPatterns = [
    /high accuracy always proves/i,
    /precision measures how many actual positives were recovered/i,
    /recall measures how many predicted positives were correct/i,
    /f1 includes true negatives directly/i,
    /changing threshold cannot change precision or recall/i,
    /class imbalance never affects metric interpretation/i,
    /false positives and false negatives always have equal cost/i,
    /precision and recall use the same denominator/i,
    /specificity is the same as recall for the positive class/i,
    /false positive rate is computed over predicted positives/i,
    /precision is automatically perfect/i,
    /aggregate metrics always reveal subgroup failures/i,
    /safe to tune thresholds repeatedly on the final test set/i,
    /precision and recall mean the same thing/i,
    /only one rounded metric is enough/i,
  ];
  const trapPrompt = /trap|false|misleading|unsafe|wrong|claim/i;

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const containsMisconception = misconceptionPatterns.some((pattern) => pattern.test(answer));
    if (!containsMisconception) continue;

    assert.ok(index >= 75, `${question.id} introduces misconception too early`);
    assert.match(question.prompt, trapPrompt, `${question.id} should mark misconception as a trap`);
  }
});

test('classification metrics assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('classification-metrics');
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

test('classification metrics assessment distributes correct-answer positions across pages and total', () => {
  const { quiz } = getLessonAssessment('classification-metrics');
  const totals = [0, 0, 0];

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const page = quiz.slice(pageStart, pageStart + 10);
    const positions = page.map((question) => question.answerIndex);

    for (const position of positions) totals[position] += 1;

    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
  }

  assert.ok(Math.max(...totals) - Math.min(...totals) <= 1, `answer positions should be balanced, got ${totals.join('/')}`);
});
