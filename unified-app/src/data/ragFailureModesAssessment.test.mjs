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

test('rag failure modes has a complete curated 100-question assessment', () => {
  const { quiz } = getLessonAssessment('rag-failure-modes');

  assert.equal(quiz.length, 100);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.ok(question.id.startsWith('ragfail-'), `question ${index + 1} should use the ragfail id prefix`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(Object.hasOwn(LEVEL_ORDER, question.level), `question ${index + 1} should have a recognized level`);
  }
});

test('rag failure modes assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('rag-failure-modes');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('rag failure modes assessment progresses from labels to interview readiness', () => {
  const { quiz } = getLessonAssessment('rag-failure-modes');
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

test('rag failure modes assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('rag-failure-modes');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const milestones = [
    ['purpose', ['fluent answer can still be unreliable']],
    ['claim-level review', ['individual claim against the retrieved evidence']],
    ['failure buckets', ['missing, stale, conflicting, and irrelevant evidence']],
    ['diagnostic before decoding', ['check whether valid evidence exists']],
    ['mechanism summary', ['test each claim against retrieved evidence']],
    ['stale application', ['last year policy']],
    ['production standard', ['valid, current, non-conflicting support']],
    ['tricky false claims', ['smooth answer proves the evidence path is healthy']],
    ['interview debugging', ['query rewrite, candidate retrieval, reranker order, and grounding decisions']],
    ['interview readiness', ['claim-level evidence classification, targeted controls, and regression monitoring']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of milestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('rag failure modes assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('rag-failure-modes');
  const unsafePatterns = [
    /smooth answer proves the evidence path is healthy/i,
    /recover a source that first-pass retrieval never returned/i,
    /always improves grounded answer quality/i,
    /citation marker means the exact claim is supported/i,
    /guarantees every accepted answer is complete/i,
    /remains valid forever/i,
    /hide contradictions/i,
    /off-topic chunks are safe/i,
    /abstaining is always a product failure/i,
    /authorized and unauthorized sources should be treated the same/i,
    /aggregate grounded coverage proves every topic slice is safe/i,
    /tune decoding first when evidence is missing/i,
    /chunk boundaries cannot affect grounding/i,
    /evaluate only final answer pleasantness/i,
    /answers look confident and cited/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /false|unsafe|wrong|trap|reject|claim|belief|misconception/i.test(question.prompt);
    assert.ok(!unsafeAnswer || explicitTrapPrompt, `question ${index + 1} keys a false claim outside a trap prompt`);
  }
});

test('rag failure modes assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('rag-failure-modes');

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

test('rag failure modes assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('rag-failure-modes');

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const positions = quiz.slice(pageStart, pageStart + 10).map((question) => question.answerIndex);
    const maxSameSlot = Math.max(...[0, 1, 2].map((slot) => positions.filter((position) => position === slot).length));

    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
    assert.ok(maxSameSlot <= 6, `page starting at question ${pageStart + 1} should not overuse one answer position`);
  }
});
