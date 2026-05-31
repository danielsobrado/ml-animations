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

test('sampling strategies has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('sampling-strategies');

  assert.equal(quiz.length, 100);
  assert.equal(labs.length, 3);
  assert.deepEqual(labs.map((lab) => lab.id), [
    'decode-for-task',
    'inspect-candidate-set',
    'compare-production-defaults',
  ]);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.match(question.id, /^samp-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use a strict samp id`);
    assert.equal(Number(question.id.slice(5, 8)), index + 1, `question ${index + 1} id should be sequential`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.equal(new Set(question.choices.map(normalized)).size, 3, `question ${index + 1} should have distinct choices`);
    assert.equal(Number.isInteger(question.answerIndex), true, `question ${index + 1} answer index should be an integer`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(Object.hasOwn(LEVEL_ORDER, question.level), `question ${index + 1} should have a recognized level`);
  }
});

test('sampling strategies assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('sampling-strategies');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('sampling strategies assessment progresses from decoding basics to interview readiness', () => {
  const { quiz } = getLessonAssessment('sampling-strategies');
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

test('sampling strategies assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('sampling-strategies');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const milestones = [
    ['purpose', ['choosing one useful continuation from a probability distribution']],
    ['greedy', ['choose the highest probability next token']],
    ['beam', ['several high scoring partial sequences']],
    ['temperature', ['sharp or flat the next token probability distribution']],
    ['top-p misconception', ['top p or nucleus sampling']],
    ['mechanism pipeline', ['rescale logits filter candidates renormalize']],
    ['task settings', ['factual qa product']],
    ['production default', ['measured task tradeoffs']],
    ['tricky false claims', ['sampling strategy claim is false']],
    ['interview readiness', ['production ready sampling strategy takeaway']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of milestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('sampling strategies assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('sampling-strategies');
  const unsafePatterns = [
    /retrains the model weights/i,
    /exactly p tokens or p percent/i,
    /until a probability-mass threshold/i,
    /always makes answers more correct/i,
    /searches several future paths/i,
    /samples randomly from the retained candidates/i,
    /automatically better for every task/i,
    /always the best creative choices/i,
    /no latency or memory cost/i,
    /guarantees that generated content is true/i,
    /guarantees valid json/i,
    /never need task-specific evaluation/i,
    /can only be fixed by retraining/i,
    /cannot change the candidate set/i,
    /fashionable setting without checking candidate sets/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /false|unsafe|wrong|trap|reject|claim|belief|misconception/i.test(question.prompt);
    assert.ok(!unsafeAnswer || explicitTrapPrompt, `question ${index + 1} keys a false claim outside a trap prompt`);
  }
});

test('sampling strategies misconception traps are placed after concept scaffolding', () => {
  const { quiz } = getLessonAssessment('sampling-strategies');
  const trapIds = [
    'samp-076-false-retrain',
    'samp-077-false-topp-count',
    'samp-078-false-topk-mass',
    'samp-079-false-temperature',
    'samp-080-false-greedy',
    'samp-081-false-beam',
    'samp-082-false-diversity',
    'samp-083-false-tail',
    'samp-084-false-beam-cost',
    'samp-085-false-seed',
    'samp-086-false-json',
    'samp-087-false-eval',
    'samp-088-false-generic',
    'samp-089-false-order',
    'samp-090-tricky-summary',
  ];

  assert.deepEqual(quiz.slice(75, 90).map((question) => question.id), trapIds);

  for (const question of quiz.slice(0, 75)) {
    assert.doesNotMatch(question.prompt, /^Which .* (false|wrong|unsafe|reject|trap|misconception)/i);
  }
});

test('sampling strategies assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('sampling-strategies');

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

test('sampling strategies assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('sampling-strategies');
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
