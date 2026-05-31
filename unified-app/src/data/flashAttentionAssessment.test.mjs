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

test('flash attention has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('flash-attention');

  assert.equal(quiz.length, 100);
  assert.equal(labs.length, 3);
  assert.deepEqual(labs.map((lab) => lab.id), [
    'compare-full-vs-tile-memory',
    'trace-online-softmax-state',
    'tune-tile-and-dtype',
  ]);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.match(question.id, /^flash-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use the flash id format`);
    assert.equal(Number(question.id.slice(6, 9)), index + 1, `question ${index + 1} id number should match its position`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.equal(new Set(question.choices.map((choice) => normalized(choice))).size, 3, `question ${index + 1} should have distinct choices`);
    assert.ok(Number.isInteger(question.answerIndex), `question ${index + 1} answer index should be an integer`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(Object.hasOwn(LEVEL_ORDER, question.level), `question ${index + 1} should have a recognized level`);
  }
});

test('flash attention assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('flash-attention');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('flash attention assessment progresses from exact-attention basics to interview readiness', () => {
  const { quiz } = getLessonAssessment('flash-attention');
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

test('flash attention assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('flash-attention');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const milestones = [
    ['purpose', ['too much score matrix data']],
    ['avoid full matrix', ['full n by n attention']],
    ['exactness', ['same attention result']],
    ['online softmax', ['normalize rows correctly']],
    ['running state', ['running row maximum']],
    ['mask semantics', ['causal mask']],
    ['debug invariants', ['masking scaling']],
    ['application validation', ['compare outputs and gradients']],
    ['tricky false claims', ['flash attention claim is false']],
    ['interview readiness', ['production ready flash attention takeaway']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of milestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('flash attention assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('flash-attention');
  const unsafePatterns = [
    /approximates dense attention/i,
    /same thing as sparse attention/i,
    /removes softmax normalization/i,
    /stores the full n by n probability matrix/i,
    /eliminating all query-key dot products/i,
    /causal masks are unnecessary/i,
    /makes kv caching obsolete/i,
    /unlimited context length at no cost/i,
    /low-rank factorization/i,
    /always drops distant tokens/i,
    /must be bit-identical/i,
    /forward parity proves the backward pass/i,
    /any tile size gives the same performance/i,
    /return the exposed n by n score matrix/i,
    /sparse, approximate, cache-replacing/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /false|unsafe|wrong|trap|reject|claim|belief|misconception/i.test(question.prompt);
    assert.ok(!unsafeAnswer || explicitTrapPrompt, `question ${index + 1} keys a false claim outside a trap prompt`);
  }
});

test('flash attention misconception traps are placed after concept scaffolding', () => {
  const { quiz } = getLessonAssessment('flash-attention');
  const trapIds = [
    'flash-076-false-approx',
    'flash-077-false-sparse',
    'flash-078-false-softmax',
    'flash-079-false-memory',
    'flash-080-false-flops',
    'flash-081-false-mask',
    'flash-082-false-cache',
    'flash-083-false-context',
    'flash-084-false-low-rank',
    'flash-085-false-window',
    'flash-086-false-numerics',
    'flash-087-false-backward',
    'flash-088-false-tile',
    'flash-089-false-output',
    'flash-090-tricky-summary',
  ];

  assert.deepEqual(quiz.slice(75, 90).map((question) => question.id), trapIds);

  for (const question of quiz.slice(0, 75)) {
    assert.doesNotMatch(question.prompt, /^Which .* claim is false\?/);
  }
});

test('flash attention assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('flash-attention');

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const page = quiz.slice(pageStart, pageStart + 10);
    const answers = page.map((question) => normalized(correctAnswer(question)));

    assert.equal(new Set(answers).size, answers.length, `page starting at question ${pageStart + 1} should not repeat exact answers`);

    for (const [answerIndex, answer] of answers.entries()) {
      for (const [promptIndex, question] of page.entries()) {
        if (answerIndex === promptIndex || answer.length < 8) continue;
        assert.ok(!normalized(question.prompt).includes(answer));
      }
    }
  }
});

test('flash attention assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('flash-attention');
  const globalPositionCounts = [0, 1, 2].map((slot) => quiz.filter((question) => question.answerIndex === slot).length);

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const positions = quiz.slice(pageStart, pageStart + 10).map((question) => question.answerIndex);
    const maxSameSlot = Math.max(...[0, 1, 2].map((slot) => positions.filter((position) => position === slot).length));

    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
    assert.ok(maxSameSlot <= 6, `page starting at question ${pageStart + 1} should not overuse one answer position`);
  }

  assert.ok(
    Math.max(...globalPositionCounts) - Math.min(...globalPositionCounts) <= 1,
    `global answer positions should be balanced, got ${globalPositionCounts.join(', ')}`,
  );
});
