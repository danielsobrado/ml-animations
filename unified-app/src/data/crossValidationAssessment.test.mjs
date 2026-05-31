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

const LEVELS = new Set(Object.keys(LEVEL_ORDER));

function normalized(value) {
  return String(value || '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
}

function correctAnswer(question) {
  return question.choices[question.answerIndex];
}

test('cross-validation has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('cross-validation');
  const ids = new Set(quiz.map((question) => question.id));
  const globalCounts = [0, 0, 0];

  assert.equal(quiz.length, 100);
  assert.deepEqual(labs.map((lab) => lab.id), [
    'trace-fold-rotation',
    'choose-fold-design',
    'audit-fold-pipeline',
  ]);
  assert.equal(ids.size, 100);
  assert.ok(quiz.every((question) => !question.id.startsWith('generated-')));

  for (const [index, question] of quiz.entries()) {
    assert.match(question.id, /^cv-\d{3}-[a-z0-9-]+$/, `${question.id} should use the curated id format`);
    assert.equal(Number(question.id.slice(3, 6)), index + 1, `${question.id} should stay in numeric order`);
    assert.ok(LEVELS.has(question.level), `${question.id} should use a known level`);
    assert.ok(question.prompt && question.prompt.length > 20, `${question.id} should have a substantial prompt`);
    assert.equal(question.choices.length, 3, `${question.id} should have three choices`);
    assert.equal(new Set(question.choices.map(normalized)).size, 3, `${question.id} should not repeat a choice`);
    assert.ok(Number.isInteger(question.answerIndex), `${question.id} should have an integer answer index`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `${question.id} has invalid answer index`);
    assert.ok(question.explanation && question.explanation.length > 30, `${question.id} should explain the answer`);
    globalCounts[question.answerIndex] += 1;
  }

  assert.ok(
    Math.max(...globalCounts) - Math.min(...globalCounts) <= 1,
    `answer positions should be globally balanced, got ${globalCounts.join(', ')}`,
  );
});

test('cross-validation assessment avoids duplicate prompts and exact correct answers', () => {
  const { quiz } = getLessonAssessment('cross-validation');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length, 'prompts should be unique');
  assert.equal(new Set(answers).size, answers.length, 'exact correct answers should be unique');
});

test('cross-validation assessment progresses from recall to interview readiness', () => {
  const { quiz } = getLessonAssessment('cross-validation');
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

test('cross-validation assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('cross-validation');
  const milestones = [
    [/main purpose of cross validation/, 0, 8],
    [/what does k mean in k fold cross validation/, 0, 8],
    [/held out fold/, 0, 10],
    [/average scores across folds/, 0, 10],
    [/learned preprocessing/, 5, 14],
    [/stratified folds/, 5, 16],
    [/grouped folds/, 8, 18],
    [/ordinary shuffled cv risky for forecasting/, 8, 18],
    [/not a replacement for a final test boundary/, 5, 14],
    [/nested cv address/, 20, 35],
    [/clone the pipeline for each fold/, 25, 36],
    [/fitting a scaler before cv leak/, 28, 38],
    [/target encoding/, 30, 42],
    [/rolling origin cv/, 35, 45],
    [/out of fold predictions/, 40, 52],
    [/standardscaler before gridsearchcv/, 50, 60],
    [/nested cross validation with inner tuning/, 55, 65],
    [/claim about preprocessing before cv is false/, 75, 90],
    [/define cross validation in a technical interview/, 90, 100],
  ];

  for (const [pattern, minIndex, maxIndex] of milestones) {
    const matchIndex = quiz.findIndex((question) => pattern.test(normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`)));
    assert.notEqual(matchIndex, -1, `missing learning point ${pattern}`);
    assert.ok(
      matchIndex >= minIndex && matchIndex < maxIndex,
      `${pattern} appears at question ${matchIndex + 1}, outside expected range ${minIndex + 1}-${maxIndex}`,
    );
  }
});

test('cross-validation assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('cross-validation');
  const unsafePatterns = [
    /train on the test set safely/,
    /guarantee that hyperparameter search is unbiased/,
    /cross validation removes any need/,
    /always the most reliable cv choice/,
    /selecting the best fold score/,
    /letting the outer held out fold influence/,
    /a prediction from a model trained on the same row it predicts/,
    /choosing the metric after inspecting/,
    /changing hyperparameters after seeing the final test result/,
  ];

  for (const [index, question] of quiz.entries()) {
    const prompt = normalized(question.prompt);
    const answer = normalized(correctAnswer(question));
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /trap|false|misleading|overclaims|too absolute|wrong|invalid|unsafe|challenge|mistake|leaks/.test(prompt);

    assert.ok(
      !unsafeAnswer || explicitTrapPrompt,
      `question ${index + 1} keys a false claim outside an explicit trap prompt`,
    );
  }
});

test('cross-validation assessment keeps misconception traps after setup', () => {
  const { quiz } = getLessonAssessment('cross-validation');
  const misconceptionPatterns = [
    /preprocessing before CV is false/i,
    /best fold score as the headline estimate/i,
    /removes any need for an untouched final evaluation/i,
    /random KFold the wrong default/i,
    /nested-CV mistake/i,
    /repeating CV not fix/i,
    /leave-one-out is always the most reliable/i,
    /out-of-fold prediction is invalid/i,
    /metric practice leaks judgment/i,
    /group and stratification requirements conflict/i,
    /shuffled CV look excellent on time series/i,
    /final-refit action is unsafe/i,
    /low fold variance is unsafe/i,
    /cached artifact is dangerous/i,
    /production risk of a clean CV score/i,
  ];
  const trapPrompt = /trap|false|misleading|overclaims|wrong|mistake|not fix|too absolute|invalid|leaks judgment|conflict|unsafe|dangerous|risk|why can/i;

  for (const [index, question] of quiz.entries()) {
    const text = `${question.prompt} ${question.choices.join(' ')} ${question.explanation}`;
    const containsMisconception = misconceptionPatterns.some((pattern) => pattern.test(text));
    if (!containsMisconception) continue;

    assert.ok(index >= 75, `${question.id} introduces misconception too early`);
    assert.match(question.prompt, trapPrompt, `${question.id} should mark misconception as a trap`);
  }
});

test('cross-validation assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('cross-validation');
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

test('cross-validation assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('cross-validation');

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const page = quiz.slice(pageStart, pageStart + 10);
    const counts = [0, 1, 2].map((slot) => page.filter((question) => question.answerIndex === slot).length);
    const maxSameSlot = Math.max(...counts);
    const minSameSlot = Math.min(...counts);

    assert.ok(
      maxSameSlot - minSameSlot <= 1,
      `page starting at question ${pageStart + 1} should balance correct option slots, got ${counts.join('/')}`,
    );
  }
});
