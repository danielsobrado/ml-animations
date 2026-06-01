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

test('attention masks has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('attention-masks');

  assert.equal(quiz.length, 100);
  assert.equal(labs.length, 3);
  assert.deepEqual(labs.map((lab) => lab.id), [
    'trace-visible-keys',
    'combine-causal-and-padding',
    'debug-mask-conventions',
  ]);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.match(question.id, /^amask-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use the amask id format`);
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

test('attention masks assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('attention-masks');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('attention masks assessment progresses from visibility basics to interview readiness', () => {
  const { quiz } = getLessonAssessment('attention-masks');
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

test('attention masks assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('attention-masks');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const milestones = [
    ['purpose', ['legally visible before softmax']],
    ['not mlm', ['masked language model input corruption']],
    ['causal', ['position reading future tokens']],
    ['padding', ['padding positions that are not real content']],
    ['formula', ['mask m is added to scores']],
    ['combined masks', ['time visibility and real token validity']],
    ['application leakage', ['causal mask or input target shift']],
    ['debug grids', ['visible key cases']],
    ['tricky false claims', ['attention mask claim is false']],
    ['interview readiness', ['lesson ready attention mask takeaway']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of milestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('attention masks assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('attention-masks');
  const unsafePatterns = [
    /same as replacing input tokens with \[mask\]/i,
    /after values are already mixed/i,
    /read future target tokens/i,
    /maximum attention probability/i,
    /hide all right-context tokens/i,
    /future decoder tokens as encoder memory/i,
    /larger probabilities after softmax/i,
    /can never change visible keys/i,
    /always control the same thing/i,
    /mask axes never matter/i,
    /always safe without special handling/i,
    /mode label proves every cell/i,
    /same boolean mask polarity/i,
    /cannot affect which context a query reads/i,
    /input token corruption, post-softmax cleanup/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /false|unsafe|wrong|trap|reject|claim|belief|misconception/i.test(question.prompt);
    assert.ok(!unsafeAnswer || explicitTrapPrompt, `question ${index + 1} keys a false claim outside a trap prompt`);
  }
});

test('attention masks misconception traps are placed after concept scaffolding', () => {
  const { quiz } = getLessonAssessment('attention-masks');
  const trapIds = [
    'amask-076-false-mlm',
    'amask-077-false-after-softmax',
    'amask-078-false-causal',
    'amask-079-false-padding',
    'amask-080-false-bidir',
    'amask-081-false-cross',
    'amask-082-false-softmax',
    'amask-083-false-toggle',
    'amask-084-false-loss-mask',
    'amask-085-false-axis',
    'amask-086-false-all-masked',
    'amask-087-false-grid',
    'amask-088-false-api',
    'amask-089-false-contract',
    'amask-090-tricky-summary',
  ];

  assert.deepEqual(quiz.slice(75, 90).map((question) => question.id), trapIds);

  for (const question of quiz.slice(0, 75)) {
    assert.doesNotMatch(question.prompt, /^Which .* claim is false\?/);
  }
});

test('attention masks assessment stays within visible lesson scope', () => {
  const { quiz } = getLessonAssessment('attention-masks');
  const lessonScopeLeaks = [
    /\bKV[- ]?cache\b/i,
    /\bcache slots?\b/i,
    /\bprefix[- ]?LM\b/i,
    /\bsparse\b/i,
    /\bblock attention\b/i,
    /\bfused\b/i,
    /\bkernel\b/i,
    /\bprivacy\b/i,
    /\bsecurity\b/i,
    /\bproduction\b/i,
    /\bmonitoring\b/i,
    /\bFP16\b/i,
    /\bdtype\b/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const visibleText = `${question.prompt} ${question.choices.join(' ')} ${question.explanation}`;
    assert.ok(!lessonScopeLeaks.some((pattern) => pattern.test(visibleText)), `question ${index + 1} leaks later or non-visible attention-mask scope`);
  }
});

test('attention masks assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('attention-masks');

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const page = quiz.slice(pageStart, pageStart + 10);
    const answers = page.map((question) => normalized(correctAnswer(question)));

    assert.equal(new Set(answers).size, answers.length, `page starting at question ${pageStart + 1} should not repeat exact answers`);

    for (const [answerIndex, answer] of answers.entries()) {
      for (const [promptIndex, question] of page.entries()) {
        if (answerIndex === promptIndex || answer.length < 8) continue;
        const visibleQuestionText = normalized(`${question.prompt} ${question.choices.join(' ')}`);
        assert.ok(!visibleQuestionText.includes(answer));
      }
    }
  }
});

test('attention masks assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('attention-masks');
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
