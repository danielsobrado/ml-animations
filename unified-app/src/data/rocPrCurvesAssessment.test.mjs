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

test('roc pr curves has a complete curated 100-question assessment', () => {
  const { quiz } = getLessonAssessment('roc-pr-curves');

  assert.equal(quiz.length, 100);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.ok(question.id.startsWith('rocpr-'), `question ${index + 1} should use the rocpr id prefix`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(Object.hasOwn(LEVEL_ORDER, question.level), `question ${index + 1} should have a recognized level`);
  }
});

test('roc pr curves assessment progresses from recall to interview readiness', () => {
  const { quiz } = getLessonAssessment('roc-pr-curves');
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

test('roc pr curves assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('roc-pr-curves');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const orderedMilestones = [
    ['purpose', ['many possible thresholds']],
    ['threshold sweep', ['threshold sweep']],
    ['ROC axes', ['true positive rate', 'false positive rate']],
    ['PR axes', ['precision versus recall']],
    ['operating point', ['one chosen threshold']],
    ['AUC', ['area under a curve']],
    ['rare positives', ['positives are rare']],
    ['PR baseline', ['positive-class prevalence']],
    ['ranking mechanics', ['ordered scores']],
    ['random ROC', ['diagonal']],
    ['validation', ['held-out data']],
    ['scenario application', ['tpr and x-axis fpr']],
    ['tricky false claims', ['claim is false']],
    ['interview readiness', ['interview']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of orderedMilestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('roc pr curves assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('roc-pr-curves');
  const unsafePatterns = [
    /automatically gives the best deployment threshold/i,
    /plots precision against recall/i,
    /plot true negative rate against false positive rate/i,
    /always tells the full alert-quality story/i,
    /always means few false positives/i,
    /independent of prevalence/i,
    /proves the scores are calibrated probabilities/i,
    /always changes ROC AUC/i,
    /changes the true labels/i,
    /best for every threshold region automatically/i,
    /safe to pick the best threshold after repeatedly inspecting the final test curve/i,
    /need only one fixed set of hard labels with no scores/i,
    /ignores false positives/i,
    /AUC alone is enough/i,
    /does not depend on which class is positive/i,
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

test('roc pr curves assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('roc-pr-curves');
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

test('roc pr curves assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('roc-pr-curves');

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const page = quiz.slice(pageStart, pageStart + 10);
    const positions = page.map((question) => question.answerIndex);

    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
  }
});
