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
    .replace(/\s+/g, ' ')
    .trim();
}

function correctAnswer(question) {
  return question.choices[question.answerIndex];
}

test('tree ensembles has a complete curated 100-question assessment', () => {
  const { quiz } = getLessonAssessment('tree-ensembles');

  assert.equal(quiz.length, 100);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.ok(question.id.startsWith('treeens-'), `question ${index + 1} should use the treeens id prefix`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
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
    ['single tree behavior', ['feature-threshold splits']],
    ['single tree risk', ['high variance']],
    ['bagging', ['bootstrap samples', 'averaging']],
    ['random forest randomness', ['random subset of features']],
    ['decorrelation', ['diverse errors cancel']],
    ['boosting sequence', ['add weak trees sequentially']],
    ['learning rate', ['shrinks each new tree contribution']],
    ['mechanism summary', ['depth', 'tree count', 'feature randomness', 'learning rate', 'rounds']],
    ['application tuning', ['early stopping or the validated round count']],
    ['tricky false claims', ['false forest claim']],
    ['interview readiness', ['leakage-safe splits', 'tune complexity on validation']],
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

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const page = quiz.slice(pageStart, pageStart + 10);
    const positions = page.map((question) => question.answerIndex);

    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
  }
});
