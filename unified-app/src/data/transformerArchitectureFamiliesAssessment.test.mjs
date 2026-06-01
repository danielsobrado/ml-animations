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

test('transformer architecture families has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('transformer-architecture-families');

  assert.equal(quiz.length, 100);
  assert.equal(labs.length, 3);
  assert.deepEqual(labs.map((lab) => lab.id), [
    'choose-family',
    'compare-visibility-masks',
    'debug-family-mismatch',
  ]);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.match(question.id, /^tfam-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use a strict tfam id`);
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
    ['three families', ['encoder only decoder only and encoder decoder']],
    ['bert style', ['bert style systems encode available input tokens']],
    ['gpt style', ['causal self attention and next token prediction']],
    ['t5 style', ['t5 style models encode an input text and decode an output text']],
    ['mechanism masks', ['bidirectional source attention causal target attention and source memory attention']],
    ['lab justification', ['attention visibility training objective and expected output type']],
    ['application tasks', ['source sentences into target language sentences']],
    ['tricky false claims', ['architecture family claim is false']],
    ['interview readiness', ['lesson ready architecture family takeaway']],
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

test('transformer architecture families misconception traps are placed after concept scaffolding', () => {
  const { quiz } = getLessonAssessment('transformer-architecture-families');
  const trapIds = [
    'tfam-076-false-same',
    'tfam-077-false-encoder-generation',
    'tfam-078-false-decoder-future',
    'tfam-079-false-encdec-mask',
    'tfam-080-false-cross',
    'tfam-081-false-objective',
    'tfam-082-false-masks',
    'tfam-083-false-bert',
    'tfam-084-false-gpt',
    'tfam-085-false-t5',
    'tfam-086-false-classifier',
    'tfam-087-false-chat',
    'tfam-088-false-translation',
    'tfam-089-false-selector',
    'tfam-090-tricky-summary',
  ];

  assert.deepEqual(quiz.slice(75, 90).map((question) => question.id), trapIds);

  for (const question of quiz.slice(0, 75)) {
    assert.doesNotMatch(question.prompt, /^Which .* (claim|belief|misconception) is (false|wrong|unsafe)\?/);
  }
});

test('transformer architecture families assessment stays within visible lesson scope', () => {
  const { quiz } = getLessonAssessment('transformer-architecture-families');
  const lessonScopeLeaks = [
    /\bKV[- ]?cache\b/i,
    /\bcache\b/i,
    /\bserving\b/i,
    /\bproduction\b/i,
    /\bdeployment\b/i,
    /\bRAG\b/i,
    /\bretrieved\b/i,
    /\bretriever\b/i,
    /\bfine[- ]?tuning\b/i,
    /\bhybrid\b/i,
    /\bmodern variants\b/i,
    /\bcalibration\b/i,
    /\bmigration\b/i,
    /\bkernel\b/i,
    /\bdtype\b/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const visibleText = `${question.prompt} ${question.choices.join(' ')} ${question.explanation}`;
    assert.ok(!lessonScopeLeaks.some((pattern) => pattern.test(visibleText)), `question ${index + 1} leaks later or non-visible family scope`);
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
        const visibleQuestionText = normalized(`${question.prompt} ${question.choices.join(' ')}`);
        assert.ok(!visibleQuestionText.includes(answer));
      }
    }
  }
});

test('transformer architecture families assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('transformer-architecture-families');
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
