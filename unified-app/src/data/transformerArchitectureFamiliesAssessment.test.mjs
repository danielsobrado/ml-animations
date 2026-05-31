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

test('transformer architecture families has a complete curated 100-question assessment', () => {
  const { quiz } = getLessonAssessment('transformer-architecture-families');

  assert.equal(quiz.length, 100);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.ok(question.id.startsWith('tfam-'), `question ${index + 1} should use the tfam id prefix`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(Object.hasOwn(LEVEL_ORDER, question.level), `question ${index + 1} should have a recognized level`);
  }
});

test('transformer architecture families assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('transformer-architecture-families');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('transformer architecture families assessment progresses from basics to interview readiness', () => {
  const { quiz } = getLessonAssessment('transformer-architecture-families');
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

test('transformer architecture families assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('transformer-architecture-families');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const milestones = [
    ['purpose', ['matching a transformer stack to the way a task consumes and produces text']],
    ['three families', ['encoder-only, decoder-only, and encoder-decoder']],
    ['bert style', ['bert-style systems encode available input tokens']],
    ['gpt style', ['causal self-attention and next-token prediction']],
    ['t5 style', ['t5-style models encode an input text and decode an output text']],
    ['mechanism masks', ['bidirectional source attention, causal target attention, and source-memory attention']],
    ['lab justification', ['attention visibility, training objective, and expected output type']],
    ['application tasks', ['source sentences into target-language sentences']],
    ['tricky false claims', ['architecture-family claim is false']],
    ['interview readiness', ['production-ready architecture-family takeaway']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of milestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('transformer architecture families assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('transformer-architecture-families');
  const unsafePatterns = [
    /identical once a block contains attention/i,
    /reading future answer tokens/i,
    /future target tokens during normal generation/i,
    /freely read later target tokens/i,
    /future target labels instead of source memory/i,
    /objectives are irrelevant/i,
    /only change visualization/i,
    /decoder-only next-token generators by default/i,
    /encoder-only bidirectional readers by default/i,
    /only encoders with no target decoder/i,
    /must always use decoder-only generation/i,
    /bidirectional encoder alone is the natural default/i,
    /no source-target separation/i,
    /only changes labels/i,
    /ignoring input, output, mask, and objective/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /false|unsafe|wrong|trap|reject|claim|belief|misconception/i.test(question.prompt);
    assert.ok(!unsafeAnswer || explicitTrapPrompt, `question ${index + 1} keys a false claim outside a trap prompt`);
  }
});

test('transformer architecture families assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('transformer-architecture-families');

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

test('transformer architecture families assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('transformer-architecture-families');

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const positions = quiz.slice(pageStart, pageStart + 10).map((question) => question.answerIndex);
    const maxSameSlot = Math.max(...[0, 1, 2].map((slot) => positions.filter((position) => position === slot).length));

    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
    assert.ok(maxSameSlot <= 6, `page starting at question ${pageStart + 1} should not overuse one answer position`);
  }
});
