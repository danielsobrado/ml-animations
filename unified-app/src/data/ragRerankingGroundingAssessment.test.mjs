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

test('rag reranking grounding has a complete curated 100-question assessment with focused labs', () => {
  const { quiz, labs } = getLessonAssessment('rag-reranking-grounding');

  assert.equal(quiz.length, 100);
  assert.equal(labs.length, 3);
  assert.deepEqual(labs.map((lab) => lab.id), [
    'grounding-audit',
    'isolate-reranker-effect',
    'audit-citation-validity',
  ]);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.match(question.id, /^ragrank-\d{3}-[a-z0-9-]+$/);
    assert.equal(Number(question.id.slice(8, 11)), index + 1);
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

test('rag reranking grounding assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('rag-reranking-grounding');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('rag reranking grounding assessment progresses from roles to interview readiness', () => {
  const { quiz } = getLessonAssessment('rag-reranking-grounding');
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

test('rag reranking grounding assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('rag-reranking-grounding');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const milestones = [
    ['purpose', ['choosing which retrieved chunks surface first']],
    ['reranking role', ['ordering of already retrieved candidates']],
    ['grounding role', ['trusted as supported citations']],
    ['high-rank trap', ['high ranked chunk is not automatically a valid citation source']],
    ['mechanism summary', ['retrieve candidates rerank them then apply a grounding validity gate']],
    ['application stale', ['last year policy']],
    ['production review', ['valid current non conflicting support']],
    ['tricky false claims', ['reranking claim is false']],
    ['interview debugging', ['retrieval reranker order packed evidence source freshness']],
    ['interview readiness', ['production ready reranking and grounding takeaway']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of milestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('rag reranking grounding assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('rag-reranking-grounding');
  const unsafePatterns = [
    /recover relevant chunks that first-pass retrieval never returned/i,
    /^A high-ranked chunk is automatically a valid citation source$/i,
    /only checks whether any chunk is retrieved/i,
    /always improves every production metric/i,
    /makes weak evidence safe/i,
    /valid if it has the highest reranker score/i,
    /automatically ground both sides/i,
    /always directly supports/i,
    /any retrieved source can be cited/i,
    /cite a plausible source/i,
    /reliability is irrelevant/i,
    /unauthorized chunks can ground answers/i,
    /credit whichever one you prefer/i,
    /cannot drift once the rules are shipped/i,
    /trust citations because retrieval returned/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /false|unsafe|wrong|trap|reject|claim|belief|misconception/i.test(question.prompt);
    assert.ok(!unsafeAnswer || explicitTrapPrompt, `question ${index + 1} keys a false claim outside a trap prompt`);
  }
});

test('rag reranking grounding misconception traps are placed after concept scaffolding', () => {
  const { quiz } = getLessonAssessment('rag-reranking-grounding');
  const trapIds = [
    'ragrank-076-false-reranker',
    'ragrank-077-false-rank',
    'ragrank-078-false-grounding',
    'ragrank-079-false-strict',
    'ragrank-080-false-lenient',
    'ragrank-081-false-stale',
    'ragrank-082-false-conflict',
    'ragrank-083-false-support',
    'ragrank-084-false-citation',
    'ragrank-085-false-absence',
    'ragrank-086-false-trust',
    'ragrank-087-false-auth',
    'ragrank-088-false-ablation',
    'ragrank-089-false-monitor',
    'ragrank-090-trap-summary',
  ];
  assert.deepEqual(quiz.slice(75, 90).map((question) => question.id), trapIds);
  for (const question of quiz.slice(0, 75)) {
    assert.doesNotMatch(question.prompt, /^Which .* (false|wrong|unsafe|reject|trap|misconception)/i);
  }
});

test('rag reranking grounding assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('rag-reranking-grounding');

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

test('rag reranking grounding assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('rag-reranking-grounding');

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const positions = quiz.slice(pageStart, pageStart + 10).map((question) => question.answerIndex);
    const maxSameSlot = Math.max(...[0, 1, 2].map((slot) => positions.filter((position) => position === slot).length));

    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
    assert.ok(maxSameSlot <= 6, `page starting at question ${pageStart + 1} should not overuse one answer position`);
  }
});
