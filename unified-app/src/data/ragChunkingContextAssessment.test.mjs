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

test('rag chunking context has a complete curated 100-question assessment with focused labs', () => {
  const { quiz, labs } = getLessonAssessment('rag-chunking-context');

  assert.equal(quiz.length, 100);
  assert.equal(labs.length, 3);
  assert.deepEqual(labs.map((lab) => lab.id), [
    'tune-chunk-budget',
    'trace-retrieved-vs-packed',
    'measure-context-tradeoffs',
  ]);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.match(question.id, /^ragchunk-\d{3}-[a-z0-9-]+$/);
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

test('rag chunking context assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('rag-chunking-context');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('rag chunking context assessment progresses from chunk basics to interview readiness', () => {
  const { quiz } = getLessonAssessment('rag-chunking-context');
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

test('rag chunking context assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('rag-chunking-context');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const milestones = [
    ['purpose', ['turning long documents into evidence the model can actually use']],
    ['retrieval units', ['retrieval units cut from longer source documents']],
    ['overlap basics', ['repeated boundary text']],
    ['top k and budget', ['survive retrieval ranking top k and token budget constraints']],
    ['mechanism summary', ['split retrieve rank and pack evidence under a context budget']],
    ['boundary application', ['refund policy spans two paragraphs']],
    ['production review', ['retrieval recall packed evidence coverage duplicate tokens latency and answer quality']],
    ['tricky false claims', ['chunking claim is false']],
    ['interview debug', ['chunked retrieved ranked high enough packed and visible']],
    ['interview readiness', ['production ready takeaway']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of milestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('rag chunking context assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('rag-chunking-context');
  const unsafePatterns = [
    /always better/i,
    /removes all need for retrieval evaluation/i,
    /guarantees every returned chunk reaches the generator/i,
    /budget is irrelevant/i,
    /smallest possible chunks are always optimal/i,
    /one huge chunk per document always gives best/i,
    /retrieved evidence and packed evidence are always the same/i,
    /repetition proves importance/i,
    /eliminates ranking and packing decisions/i,
    /recall at k alone proves/i,
    /generated citations are always reliable/i,
    /regardless of user permissions/i,
    /one successful demo query is enough/i,
    /update model weights like lora/i,
    /one knob upward/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /false|unsafe|wrong|trap|reject|claim|belief|misconception/i.test(question.prompt);
    assert.ok(!unsafeAnswer || explicitTrapPrompt, `question ${index + 1} keys a false claim outside a trap prompt`);
  }
});

test('rag chunking context misconception traps are placed after concept scaffolding', () => {
  const { quiz } = getLessonAssessment('rag-chunking-context');
  const trapIds = [
    'ragchunk-076-false-bigger',
    'ragchunk-077-false-overlap',
    'ragchunk-078-false-topk',
    'ragchunk-079-false-budget',
    'ragchunk-080-false-small',
    'ragchunk-081-false-large',
    'ragchunk-082-false-retrieved',
    'ragchunk-083-false-dedup',
    'ragchunk-084-false-long-context',
    'ragchunk-085-false-metric',
    'ragchunk-086-false-source',
    'ragchunk-087-false-auth',
    'ragchunk-088-false-eval',
    'ragchunk-089-false-finetune',
    'ragchunk-090-trap-summary',
  ];
  assert.deepEqual(quiz.slice(75, 90).map((question) => question.id), trapIds);
  for (const question of quiz.slice(0, 75)) {
    assert.doesNotMatch(question.prompt, /^Which .* (false|wrong|unsafe|reject|trap|misconception)/i);
  }
});

test('rag chunking context assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('rag-chunking-context');

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

test('rag chunking context assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('rag-chunking-context');

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const positions = quiz.slice(pageStart, pageStart + 10).map((question) => question.answerIndex);
    const maxSameSlot = Math.max(...[0, 1, 2].map((slot) => positions.filter((position) => position === slot).length));

    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
    assert.ok(maxSameSlot <= 6, `page starting at question ${pageStart + 1} should not overuse one answer position`);
  }
});
