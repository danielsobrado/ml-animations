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

test('gradient problems has a complete curated 100-question assessment', () => {
  const { quiz } = getLessonAssessment('gradient-problems');

  assert.equal(quiz.length, 100);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.ok(question.id.startsWith('grp-'), `question ${index + 1} should use the grp id prefix`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(Object.hasOwn(LEVEL_ORDER, question.level), `question ${index + 1} should have a recognized level`);
  }
});

test('gradient problems assessment progresses from chain-rule basics to interview readiness', () => {
  const { quiz } = getLessonAssessment('gradient-problems');
  const expectedBands = [
    ['Foundation', 0, 20],
    ['Mechanism', 20, 50],
    ['Application', 50, 75],
    ['Tricky', 75, 90],
    ['Interview', 90, 100],
  ];

  for (const [level, start, end] of expectedBands) {
    assert.deepEqual([...new Set(quiz.slice(start, end).map((question) => question.level))], [level]);
  }

  for (let index = 1; index < quiz.length; index += 1) {
    assert.ok(LEVEL_ORDER[quiz[index].level] >= LEVEL_ORDER[quiz[index - 1].level]);
  }
});

test('gradient problems assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('gradient-problems');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const milestones = [
    ['purpose', ['too small or too large']],
    ['chain rule', ['multiplies many local derivatives']],
    ['vanishing', ['shrinks so much']],
    ['exploding', ['grows so large']],
    ['residual shortcut', ['shortcut path']],
    ['clipping limit', ['does not restore vanished signal']],
    ['product math', ['exponential shrinkage']],
    ['layer ledger', ['gradient magnitude changes sharply']],
    ['application diagnosis', ['early layers barely change', 'blocked backward paths']],
    ['clipping application', ['gradient clipping with a monitored threshold']],
    ['unsafe trap', ['claim is unsafe']],
    ['interview readiness', ['trace the chain product']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of milestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('gradient problems assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('gradient-problems');
  const unsafePatterns = [
    /fixes vanishing gradients by making tiny gradients large/i,
    /loss alone is enough/i,
    /depth cannot affect gradient scale/i,
    /guarantee no vanishing or exploding/i,
    /monitoring unnecessary/i,
    /always means the model is finished/i,
    /always useful because it means strong learning/i,
    /always recover automatically/i,
    /cannot affect gradients after the first batch/i,
    /any threshold works/i,
    /cannot have vanishing gradients/i,
    /one global gradient norm always reveals/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /false|unsafe|wrong|trap|claim|reject|overconfident|overbroad|too weak|needs caution/i.test(question.prompt);
    assert.ok(!unsafeAnswer || explicitTrapPrompt, `question ${index + 1} keys a false claim outside a trap prompt`);
  }
});

test('gradient problems assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('gradient-problems');

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const page = quiz.slice(pageStart, pageStart + 10);
    const answers = page.map((question) => normalized(correctAnswer(question)));

    assert.equal(new Set(answers).size, answers.length, `page starting at question ${pageStart + 1} should not repeat exact answers`);

    for (const [answerIndex, answer] of answers.entries()) {
      for (const [promptIndex, question] of page.entries()) {
        if (answerIndex === promptIndex) continue;
        assert.ok(!normalized(question.prompt).includes(answer));
      }
    }
  }
});

test('gradient problems assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('gradient-problems');

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const positions = quiz.slice(pageStart, pageStart + 10).map((question) => question.answerIndex);
    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
  }
});
