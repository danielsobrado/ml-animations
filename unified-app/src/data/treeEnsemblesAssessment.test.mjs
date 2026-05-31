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

function normalized(value) {
  return String(value || '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
}

function correctAnswer(question) {
  return question.choices[question.answerIndex];
}

test('tree ensembles has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('tree-ensembles');

  assert.equal(quiz.length, 100);
  assert.deepEqual(
    labs.map((lab) => lab.id),
    ['depth-vs-ensemble', 'forest-vote-diagnosis', 'boosting-correction-plan'],
  );
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.match(question.id, /^treeens-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use an ordered treeens id`);
    assert.equal(Number(question.id.slice(8, 11)), index + 1, `question ${index + 1} id should match its position`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.equal(new Set(question.choices.map(normalized)).size, question.choices.length, `question ${index + 1} choices should be distinct`);
    assert.ok(Number.isInteger(question.answerIndex), `question ${index + 1} answer index should be an integer`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(Object.hasOwn(LEVEL_ORDER, question.level), `question ${index + 1} should have a recognized level`);
  }
});

test('tree ensembles assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('tree-ensembles');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('tree ensembles assessment progresses from tree basics to interview readiness', () => {
  const { quiz } = getLessonAssessment('tree-ensembles');
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

test('tree ensembles assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('tree-ensembles');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const orderedMilestones = [
    ['purpose', ['bagging', 'random forests', 'gradient boosting']],
    ['single tree behavior', ['feature threshold splits']],
    ['single tree risk', ['high variance']],
    ['bagging', ['bootstrap samples', 'averaging']],
    ['random forest randomness', ['random subset of features']],
    ['decorrelation', ['diverse errors cancel']],
    ['boosting sequence', ['add weak trees sequentially']],
    ['learning rate', ['shrinks each new tree contribution']],
    ['mechanism summary', ['depth', 'tree count', 'feature randomness', 'learning rate', 'rounds']],
    ['application tuning', ['early stopping or the validated round count']],
    ['tricky false claims', ['false forest claim']],
    ['interview readiness', ['leakage safe splits', 'tune complexity on validation']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of orderedMilestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('tree ensembles assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('tree-ensembles');
  const unsafePatterns = [
    /always eliminates overfitting completely/i,
    /always improve deployment performance with no cost/i,
    /all preprocessing concerns disappear/i,
    /guaranteed replacement for every final evaluation/i,
    /cannot overfit because each tree is weak/i,
    /automatically removes the need to tune rounds/i,
    /always better because they fit training data/i,
    /proves a causal effect/i,
    /immune to leaked future information/i,
    /automatically calibrated/i,
    /repeatedly retrying the final test/i,
    /always extrapolate smoothly/i,
    /production-ready for every dataset/i,
    /as transparent as a single small tree/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /false|unsafe|wrong|trap|claim|dangerous|incorrect/i.test(question.prompt);

    assert.ok(
      !unsafeAnswer || explicitTrapPrompt,
      `question ${index + 1} keys a false claim outside an explicit trap prompt`,
    );
  }
});

test('tree ensembles assessment keeps misconception traps after setup', () => {
  const { quiz } = getLessonAssessment('tree-ensembles');
  const expectedTrapIds = [
    'treeens-076-trap-forest-overfit',
    'treeens-077-trap-more-trees',
    'treeens-078-trap-feature-scale',
    'treeens-079-trap-oob',
    'treeens-080-trap-boosting',
    'treeens-081-trap-learning-rate',
    'treeens-082-trap-depth',
    'treeens-083-trap-importance',
    'treeens-084-trap-leakage',
    'treeens-085-trap-calibration',
    'treeens-086-trap-metric',
    'treeens-087-trap-extrapolation',
    'treeens-088-trap-defaults',
    'treeens-089-trap-interpretability',
    'treeens-090-tricky-summary',
  ];
  const misconceptionPatterns = [
    /always eliminates overfitting completely/i,
    /always improve deployment performance with no cost/i,
    /all preprocessing concerns disappear/i,
    /guaranteed replacement for every final evaluation/i,
    /cannot overfit because each tree is weak/i,
    /automatically removes the need to tune rounds/i,
    /always better because they fit training data/i,
    /proves a causal effect/i,
    /immune to leaked future information/i,
    /automatically calibrated/i,
    /repeatedly retrying the final test/i,
    /always extrapolate smoothly/i,
    /production-ready for every dataset/i,
    /as transparent as a single small tree/i,
  ];
  const trapPrompt = /false|unsafe|wrong|trap|claim|dangerous|incorrect/i;

  assert.deepEqual(quiz.slice(75, 90).map((question) => question.id), expectedTrapIds);

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const containsMisconception = misconceptionPatterns.some((pattern) => pattern.test(answer));
    if (!containsMisconception) continue;
    assert.ok(index >= 75, `${question.id} introduces misconception too early`);
    assert.match(question.prompt, trapPrompt, `${question.id} should mark misconception as a trap`);
  }
});

test('tree ensembles assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('tree-ensembles');
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

test('tree ensembles assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('tree-ensembles');
  const totals = [0, 0, 0];

  for (const question of quiz) {
    totals[question.answerIndex] += 1;
  }

  assert.ok(Math.max(...totals) - Math.min(...totals) <= 1, `answer positions should be balanced: ${totals.join(', ')}`);

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const page = quiz.slice(pageStart, pageStart + 10);
    const positions = page.map((question) => question.answerIndex);

    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
  }
});
