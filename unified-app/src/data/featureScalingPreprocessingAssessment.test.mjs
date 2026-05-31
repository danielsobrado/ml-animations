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

test('feature scaling preprocessing has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('feature-scaling-preprocessing');

  assert.equal(quiz.length, 100);
  assert.deepEqual(labs.map((lab) => lab.id), [
    'compare-scaler-statistics',
    'outlier-scaler-comparison',
    'audit-train-only-pipeline',
  ]);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    const expectedNumber = String(index + 1).padStart(3, '0');

    assert.match(question.id, /^scale-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use the scale id format`);
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

test('feature scaling preprocessing assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('feature-scaling-preprocessing');
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

test('feature scaling preprocessing assessment progresses from recall to interview readiness', () => {
  const { quiz } = getLessonAssessment('feature-scaling-preprocessing');
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

test('feature scaling preprocessing assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('feature-scaling-preprocessing');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));

  const orderedMilestones = [
    [/main purpose of feature scaling/, 0, 8],
    [/standardization usually compute/, 0, 8],
    [/min max scaling usually do/, 0, 8],
    [/robust scaling usually use/, 0, 10],
    [/scaler be fitted/, 4, 12],
    [/validation and test be handled/, 4, 12],
    [/preprocessing before splitting unsafe/, 6, 14],
    [/distance based models/, 8, 16],
    [/gradient descent/, 8, 18],
    [/l1 or l2 penalties/, 10, 20],
    [/outlier affect min max scaling/, 10, 20],
    [/median imputer be fitted/, 28, 38],
    [/cv pipeline/, 32, 42],
    [/label free preprocessing leak/, 34, 42],
    [/scaling affect knn or k means/, 36, 45],
    [/train serving preprocessing parity/, 32, 42],
    [/knn model uses age in years and income in dollars/, 50, 60],
    [/standardscaler before cross val score/, 50, 60],
    [/scaler is fit on all data because it does not use labels/, 62, 70],
    [/preprocessing claim is false/, 75, 90],
    [/define feature scaling in an interview/, 90, 100],
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

test('feature scaling preprocessing assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('feature-scaling-preprocessing');
  const unsafePatterns = [
    /scaling is harmless bookkeeping that can be fit on all rows/i,
    /if a transform ignores labels it cannot leak anything/i,
    /every model family benefits equally from standardization/i,
    /min max scaling guarantees future values stay between 0 and 1/i,
    /robust scaling automatically removes every outlier problem/i,
    /fit the scaler once before making folds/i,
    /compare regularized coefficients from unscaled features as if units matched/i,
    /choose clipping thresholds after inspecting final test errors/i,
    /refit the scaler independently in serving for each request/i,
    /one hot encoding never needs an unknown category policy/i,
    /all binary indicators must always be standardized/i,
    /a fitted scaler from training remains representative forever/i,
    /pick the scaler after repeatedly checking the final test score/i,
    /guarantee that every model improves/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /trap|false|misleading|unsafe|too absolute|too strong|challenge|claim|practice|statement/i.test(question.prompt);

    assert.ok(
      !unsafeAnswer || explicitTrapPrompt,
      `question ${index + 1} keys a false claim outside an explicit trap prompt`,
    );
  }
});

test('feature scaling preprocessing assessment keeps misconception traps after setup', () => {
  const { quiz } = getLessonAssessment('feature-scaling-preprocessing');
  const expectedTrapIds = [
    'scale-076-trap-harmless',
    'scale-077-trap-label-free',
    'scale-078-trap-all-models',
    'scale-079-trap-minmax',
    'scale-080-trap-robust',
    'scale-081-trap-cv',
    'scale-082-trap-coefficients',
    'scale-083-trap-outlier',
    'scale-084-trap-serving',
    'scale-085-trap-one-hot',
    'scale-086-trap-binary',
    'scale-087-trap-target',
    'scale-088-trap-drift',
    'scale-089-trap-schema',
    'scale-090-trap-best-score',
  ];
  const misconceptionPatterns = [
    /scaling is harmless bookkeeping/i,
    /if a transform ignores labels/i,
    /every model family benefits equally/i,
    /min max scaling guarantees future values stay between 0 and 1/i,
    /robust scaling automatically removes every outlier problem/i,
    /fit the scaler once before making folds/i,
    /compare regularized coefficients from unscaled features as if units matched/i,
    /choose clipping thresholds after inspecting final test errors/i,
    /refit the scaler independently in serving/i,
    /one hot encoding never needs an unknown category policy/i,
    /all binary indicators must always be standardized/i,
    /fitted scaler from training remains representative forever/i,
    /pick the scaler after repeatedly checking the final test score/i,
  ];
  const trapPrompt = /trap|false|misleading|unsafe|too absolute|too strong|claim|practice|statement/i;

  assert.deepEqual(quiz.slice(75, 90).map((question) => question.id), expectedTrapIds);

  for (const [index, question] of quiz.entries()) {
    const text = `${question.prompt} ${question.choices.join(' ')} ${question.explanation}`;
    const containsMisconception = misconceptionPatterns.some((pattern) => pattern.test(text));
    if (!containsMisconception) continue;

    assert.ok(index >= 75, `${question.id} introduces misconception too early`);
    assert.match(question.prompt, trapPrompt, `${question.id} should mark misconception as a trap`);
  }
});

test('feature scaling preprocessing assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('feature-scaling-preprocessing');
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

test('feature scaling preprocessing assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('feature-scaling-preprocessing');
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
