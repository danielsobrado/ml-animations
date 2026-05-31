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

test('rag retrieval evaluation has a complete curated 100-question assessment with focused labs', () => {
  const { quiz, labs } = getLessonAssessment('rag-retrieval-evaluation');

  assert.equal(quiz.length, 100);
  assert.equal(labs.length, 3);
  assert.deepEqual(labs.map((lab) => lab.id), [
    'chunking-rerank-audit',
    'metric-tradeoff-table',
    'slice-regression-check',
  ]);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.match(question.id, /^rageval-\d{3}-[a-z0-9-]+$/);
    assert.equal(Number(question.id.slice(8, 11)), index + 1);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.equal(new Set(question.choices.map(normalized)).size, 3, `question ${index + 1} choices should be distinct`);
    assert.ok(Number.isInteger(question.answerIndex), `question ${index + 1} answer index should be an integer`);
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

test('rag retrieval evaluation assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('rag-retrieval-evaluation');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('rag retrieval evaluation assessment progresses from retrieval basics to interview readiness', () => {
  const { quiz } = getLessonAssessment('rag-retrieval-evaluation');
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

test('rag retrieval evaluation assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('rag-retrieval-evaluation');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const milestones = [
    ['purpose', ['knowing whether answer evidence actually reached the model']],
    ['chunking role', ['what evidence can be retrieved as candidate chunks']],
    ['metric intro', ['rank sensitive relevance']],
    ['reranker limit', ['reranker cannot rescue a relevant chunk']],
    ['mechanism summary', ['recall k for coverage', 'mrr for first useful rank', 'ndcg for rank sensitive relevance']],
    ['application missing evidence', ['refund answer is wrong']],
    ['production review', ['chunking top k reranking and metrics']],
    ['tricky false claims', ['generation claim is false']],
    ['interview debugging', ['retrieved reranked high packed fresh and cited correctly']],
    ['interview readiness', ['production ready retrieval evaluation takeaway']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of milestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('rag retrieval evaluation assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('rag-retrieval-evaluation');
  const unsafePatterns = [
    /fluent answer proves retrieval/i,
    /reranker can fix evidence/i,
    /recall@k tells you the first useful rank/i,
    /mrr measures every relevant chunk/i,
    /ndcg ignores rank order/i,
    /always better with no cost/i,
    /always improves distinct evidence coverage/i,
    /need no relevance judgments/i,
    /one successful demo query proves/i,
    /guaranteed to be visible to the generator/i,
    /stale chunk is safe/i,
    /contradictory retrieved chunks automatically/i,
    /guarantees every query slice works well/i,
    /thresholds cannot hurt recall/i,
    /ship based on answer fluency/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /false|unsafe|wrong|trap|reject|claim|belief|misconception/i.test(question.prompt);
    assert.ok(!unsafeAnswer || explicitTrapPrompt, `question ${index + 1} keys a false claim outside a trap prompt`);
  }
});

test('rag retrieval evaluation misconception traps are placed after concept scaffolding', () => {
  const { quiz } = getLessonAssessment('rag-retrieval-evaluation');
  const trapIds = [
    'rageval-076-false-fluency',
    'rageval-077-false-reranker',
    'rageval-078-false-recall',
    'rageval-079-false-mrr',
    'rageval-080-false-ndcg',
    'rageval-081-false-topk',
    'rageval-082-false-overlap',
    'rageval-083-false-labels',
    'rageval-084-false-single',
    'rageval-085-false-packed',
    'rageval-086-false-stale',
    'rageval-087-false-conflict',
    'rageval-088-false-aggregate',
    'rageval-089-false-threshold',
    'rageval-090-trap-summary',
  ];
  assert.deepEqual(quiz.slice(75, 90).map((question) => question.id), trapIds);
  for (const question of quiz.slice(0, 75)) {
    assert.doesNotMatch(question.prompt, /^Which .* (false|wrong|unsafe|reject|trap|misconception)/i);
  }
});

test('rag retrieval evaluation assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('rag-retrieval-evaluation');

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

test('rag retrieval evaluation assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('rag-retrieval-evaluation');

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const positions = quiz.slice(pageStart, pageStart + 10).map((question) => question.answerIndex);
    const maxSameSlot = Math.max(...[0, 1, 2].map((slot) => positions.filter((position) => position === slot).length));

    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
    assert.ok(maxSameSlot <= 6, `page starting at question ${pageStart + 1} should not overuse one answer position`);
  }
});
