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

test('rag vector indexing has a complete curated 100-question assessment with focused labs', () => {
  const { quiz, labs } = getLessonAssessment('rag-vector-indexing');

  assert.equal(quiz.length, 100);
  assert.equal(labs.length, 3);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.match(question.id, /^ragindex-\d{3}-[a-z0-9-]+$/);
    assert.equal(Number(question.id.slice(9, 12)), index + 1);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.equal(new Set(question.choices.map(normalized)).size, 3);
    assert.ok(Number.isInteger(question.answerIndex));
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(Object.hasOwn(LEVEL_ORDER, question.level), `question ${index + 1} should have a recognized level`);
  }

  const allPositions = quiz.map((question) => question.answerIndex);
  const globalCounts = [0, 1, 2].map((slot) => allPositions.filter((position) => position === slot).length);
  assert.ok(
    Math.max(...globalCounts) - Math.min(...globalCounts) <= 1,
    `answer positions should be globally balanced, got ${globalCounts.join(', ')}`,
  );
});

test('rag vector indexing assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('rag-vector-indexing');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('rag vector indexing assessment progresses from index basics to interview readiness', () => {
  const { quiz } = getLessonAssessment('rag-vector-indexing');
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

test('rag vector indexing assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('rag-vector-indexing');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const milestones = [
    ['purpose', ['finding similar chunks quickly when exact comparison across the whole corpus is too slow']],
    ['exact search', ['compare the query vector with every indexed chunk vector']],
    ['ann tradeoff', ['lower latency in exchange for possible recall loss']],
    ['ivf hnsw', ['exact search ivf and hnsw']],
    ['mechanism summary', ['exact scans all vectors ivf probes buckets hnsw explores a neighbor graph']],
    ['small corpus application', ['2 000 chunks and strict recall needs']],
    ['production review', ['corpus scale recall latency memory freshness and filters']],
    ['tricky false claims', ['ann claim is false']],
    ['interview evaluation', ['compare ann against exact baselines']],
    ['interview readiness', ['production ready vector indexing takeaway']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of milestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('rag vector indexing assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('rag-vector-indexing');
  const unsafePatterns = [
    /automatically equivalent to exact search/i,
    /reranker can fix relevant chunks/i,
    /always lowers latency and raises recall/i,
    /always too slow for every corpus size/i,
    /always probes the bucket/i,
    /guaranteed to visit every vector/i,
    /always produce identical rankings/i,
    /authorization filters are optional/i,
    /one fast query proves/i,
    /never need freshness or permission checks/i,
    /keyword search is useless/i,
    /need no monitoring/i,
    /guarantees the relevant chunk was searched/i,
    /corpus scale has no effect/i,
    /fastest ann setting without measuring/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /false|unsafe|wrong|trap|reject|claim|belief|misconception/i.test(question.prompt);
    assert.ok(!unsafeAnswer || explicitTrapPrompt, `question ${index + 1} keys a false claim outside a trap prompt`);
  }
});

test('rag vector indexing assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('rag-vector-indexing');

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

test('rag vector indexing assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('rag-vector-indexing');

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const positions = quiz.slice(pageStart, pageStart + 10).map((question) => question.answerIndex);
    const maxSameSlot = Math.max(...[0, 1, 2].map((slot) => positions.filter((position) => position === slot).length));

    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
    assert.ok(maxSameSlot <= 6, `page starting at question ${pageStart + 1} should not overuse one answer position`);
  }
});
