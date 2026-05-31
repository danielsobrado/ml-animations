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

test('layer normalization has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('layer-normalization');

  assert.equal(quiz.length, 100);
  assert.equal(labs.length, 3);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.match(question.id, /^ln-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use the ordered ln id format`);
    assert.equal(Number(question.id.slice(3, 6)), index + 1, `question ${index + 1} should keep ordered ids`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.equal(new Set(question.choices.map(normalized)).size, 3, `question ${index + 1} choices should be distinct`);
    assert.equal(Number.isInteger(question.answerIndex), true, `question ${index + 1} answer index should be an integer`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(Object.hasOwn(LEVEL_ORDER, question.level), `question ${index + 1} should have a recognized level`);
  }

  const positionCounts = [0, 1, 2].map((answerIndex) => (
    quiz.filter((question) => question.answerIndex === answerIndex).length
  ));
  assert.ok(Math.max(...positionCounts) - Math.min(...positionCounts) <= 1, `answer positions should be balanced: ${positionCounts.join(', ')}`);
});

test('layer normalization assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('layer-normalization');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('layer normalization assessment progresses from formula basics to interview readiness', () => {
  const { quiz } = getLessonAssessment('layer-normalization');
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

test('layer normalization assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('layer-normalization');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const milestones = [
    ['purpose', ['feature scale drift']],
    ['axis', ['feature dimension']],
    ['gamma', ['what is gamma', 'per feature scale']],
    ['batch contrast', ['layernorm uses per example features']],
    ['pre norm', ['applying layernorm before']],
    ['formula', ['compute mean and variance']],
    ['residual stream', ['residual additions']],
    ['rmsnorm', ['without subtracting the mean']],
    ['application axis', ['wrong axis for transformer tokens']],
    ['production checks', ['hidden state shapes']],
    ['unsafe trap', ['claim is unsafe']],
    ['interview readiness', ['trace the per token formula']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of milestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('layer normalization assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('layer-normalization');
  const unsafePatterns = [
    /needs batch running statistics just like batchnorm/i,
    /normalize across all tokens in the sequence by default/i,
    /make normalization unnecessary/i,
    /switch to running statistics during evaluation/i,
    /must use the same layernorm placement/i,
    /removes the need for sensible initialization/i,
    /makes learning-rate tuning unnecessary/i,
    /exactly the same formula/i,
    /just dropout because both improve training/i,
    /proven best/i,
    /any normalized_shape works/i,
    /cannot affect other normalized features/i,
    /fails by definition when batch size is one/i,
    /no state worth checking/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /false|unsafe|wrong|trap|claim|reject|absolute/i.test(question.prompt);
    assert.ok(!unsafeAnswer || explicitTrapPrompt, `question ${index + 1} keys a false claim outside a trap prompt`);
  }
});

test('layer normalization assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('layer-normalization');

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

test('layer normalization assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('layer-normalization');

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const positions = quiz.slice(pageStart, pageStart + 10).map((question) => question.answerIndex);
    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
  }
});
