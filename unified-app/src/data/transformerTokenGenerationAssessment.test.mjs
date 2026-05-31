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

test('transformer token generation has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('transformer-token-generation');

  assert.equal(quiz.length, 100);
  assert.equal(labs.length, 3);
  assert.deepEqual(labs.map((lab) => lab.id), [
    'sampling-contrast',
    'trace-token-step',
    'verify-serving-behavior',
  ]);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.match(question.id, /^ttg-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use a strict ttg id`);
    assert.equal(Number(question.id.slice(4, 7)), index + 1, `question ${index + 1} id should be sequential`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.equal(new Set(question.choices.map(normalized)).size, 3, `question ${index + 1} should have distinct choices`);
    assert.equal(Number.isInteger(question.answerIndex), true, `question ${index + 1} answer index should be an integer`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(Object.hasOwn(LEVEL_ORDER, question.level), `question ${index + 1} should have a recognized level`);
  }
});

test('transformer token generation assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('transformer-token-generation');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('transformer token generation assessment progresses from loop basics to interview readiness', () => {
  const { quiz } = getLessonAssessment('transformer-token-generation');
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

test('transformer token generation assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('transformer-token-generation');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const milestones = [
    ['purpose', ['turning a trained transformer into text one next token decision at a time']],
    ['loop', ['read context score next tokens choose one append it and repeat']],
    ['logits', ['raw vocabulary scores before probability normalization']],
    ['controls', ['top k limits the candidate set']],
    ['cache basics', ['reuse prior key and value rows']],
    ['mechanism loop', ['prefill context compute next token logits']],
    ['application cache bug', ['enabling kv cache changes outputs']],
    ['application production', ['tune decoding by task']],
    ['tricky false claims', ['generation claim is false']],
    ['interview readiness', ['production ready token generation takeaway']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of milestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('transformer token generation assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('transformer-token-generation');
  const unsafePatterns = [
    /entire final answer as one fixed vector/i,
    /supposed to change next-token probabilities/i,
    /read future unsampled tokens/i,
    /permanently changes the model weights/i,
    /always keeps every token/i,
    /always keeps exactly p tokens/i,
    /explores many possible continuations/i,
    /guarantees the highest-quality answer/i,
    /discarded before the next decision/i,
    /trains new model facts/i,
    /always stops correctly/i,
    /guarantees factual correctness/i,
    /whole answer was known before decoding began/i,
    /replace the need for task evaluation/i,
    /single prediction and ignore context updates/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /false|unsafe|wrong|trap|reject|claim|belief|misconception/i.test(question.prompt);
    assert.ok(!unsafeAnswer || explicitTrapPrompt, `question ${index + 1} keys a false claim outside a trap prompt`);
  }
});

test('transformer token generation misconception traps are placed after concept scaffolding', () => {
  const { quiz } = getLessonAssessment('transformer-token-generation');
  const trapIds = [
    'ttg-076-false-whole-answer',
    'ttg-077-false-cache-prob',
    'ttg-078-false-future',
    'ttg-079-false-temperature',
    'ttg-080-false-topk',
    'ttg-081-false-topp',
    'ttg-082-false-greedy',
    'ttg-083-false-sampling',
    'ttg-084-false-append',
    'ttg-085-false-softmax',
    'ttg-086-false-stop',
    'ttg-087-false-seed',
    'ttg-088-false-streaming',
    'ttg-089-false-controls',
    'ttg-090-tricky-summary',
  ];

  assert.deepEqual(quiz.slice(75, 90).map((question) => question.id), trapIds);

  for (const question of quiz.slice(0, 75)) {
    assert.doesNotMatch(question.prompt, /^Which .* (false|wrong|unsafe|reject|trap|misconception)/i);
  }
});

test('transformer token generation assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('transformer-token-generation');

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

test('transformer token generation assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('transformer-token-generation');
  const allPositions = quiz.map((question) => question.answerIndex);
  const globalCounts = [0, 1, 2].map((slot) => allPositions.filter((position) => position === slot).length);

  assert.ok(
    Math.max(...globalCounts) - Math.min(...globalCounts) <= 1,
    `global answer positions are imbalanced: ${globalCounts.join(', ')}`,
  );

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const positions = quiz.slice(pageStart, pageStart + 10).map((question) => question.answerIndex);
    const maxSameSlot = Math.max(...[0, 1, 2].map((slot) => positions.filter((position) => position === slot).length));

    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
    assert.ok(maxSameSlot <= 6, `page starting at question ${pageStart + 1} should not overuse one answer position`);
  }
});
