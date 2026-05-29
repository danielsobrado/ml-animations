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

test('leaky relu has a complete curated 100-question assessment', () => {
  const { quiz } = getLessonAssessment('leaky-relu');

  assert.equal(quiz.length, 100);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.ok(question.id.startsWith('lrelu-'), `question ${index + 1} should use the lrelu id prefix`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(Object.hasOwn(LEVEL_ORDER, question.level), `question ${index + 1} should have a recognized level`);
  }
});

test('leaky relu assessment progresses from definitions to interview readiness', () => {
  const { quiz } = getLessonAssessment('leaky-relu');
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

test('leaky relu assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('leaky-relu');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const milestones = [
    ['relu prerequisite', ['plain relu has zero slope']],
    ['formula', ['f(z) = z for z >= 0 and alpha z']],
    ['alpha meaning', ['slope of the negative branch']],
    ['negative forward compute', ['alpha = 0.1', '-0.2']],
    ['negative derivative', ['alpha on the negative branch']],
    ['dead-zone reduction', ['dead-unit behavior']],
    ['backward mask', ['local derivative is one or alpha']],
    ['alpha tradeoff', ['activation becomes close to linear']],
    ['application debugging', ['dead units', 'activation experiment']],
    ['tricky false claims', ['claim is false']],
    ['interview readiness', ['production-ready takeaway']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of milestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('leaky relu assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('leaky-relu');
  const unsafePatterns = [
    /multiplies positive inputs by alpha/i,
    /outputs exactly zero for every negative input/i,
    /turns negative evidence positive/i,
    /negative branch always blocks all gradient/i,
    /larger alpha always improves the model/i,
    /eliminates the need for good initialization/i,
    /as exactly sparse as relu by default/i,
    /converts activations into probabilities between zero and one/i,
    /alpha = 1 gives plain relu/i,
    /always learns alpha automatically/i,
    /always the correct final activation/i,
    /diagnostics are unnecessary/i,
    /every framework uses the same default negative slope/i,
    /fixes dead relus without any tradeoff/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /false|unsafe|wrong|trap|claim/i.test(question.prompt);
    assert.ok(!unsafeAnswer || explicitTrapPrompt, `question ${index + 1} keys a false claim outside a trap prompt`);
  }
});

test('leaky relu assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('leaky-relu');

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

test('leaky relu assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('leaky-relu');

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const positions = quiz.slice(pageStart, pageStart + 10).map((question) => question.answerIndex);
    const maxSameSlot = Math.max(...[0, 1, 2].map((slot) => positions.filter((position) => position === slot).length));

    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
    assert.ok(maxSameSlot <= 6, `page starting at question ${pageStart + 1} should not overuse one answer position`);
  }
});
