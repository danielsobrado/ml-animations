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

test('classifier-free guidance has a complete curated 100-question assessment', () => {
  const { quiz } = getLessonAssessment('classifier-free-guidance');

  assert.equal(quiz.length, 100);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.ok(question.id.startsWith('cfg-'), `question ${index + 1} should use the cfg id prefix`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(Object.hasOwn(LEVEL_ORDER, question.level), `question ${index + 1} should have a recognized level`);
  }
});

test('classifier-free guidance assessment progresses from branches to interview readiness', () => {
  const { quiz } = getLessonAssessment('classifier-free-guidance');
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

test('classifier-free guidance assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('classifier-free-guidance');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const milestones = [
    ['purpose', ['without a separate classifier']],
    ['two predictions', ['prompt-conditioned prediction and an unconditional prediction']],
    ['scale tradeoff', ['prompt match can improve while diversity and quality may suffer']],
    ['not max scale', ['maximum guidance always gives the best sample']],
    ['mechanism summary', ['steer each sampling update with a tunable scale']],
    ['low prompt match', ['increase guidance scale gradually']],
    ['production review', ['prompt match, diversity, artifacts, and branch differences']],
    ['tricky false claims', ['requires a separate external classifier during sampling']],
    ['interview debugging', ['sweep scale with fixed seed, inspect both predictions']],
    ['interview readiness', ['conditional-minus-unconditional denoising direction']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of milestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('classifier-free guidance assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('classifier-free-guidance');
  const unsafePatterns = [
    /requires a separate external classifier/i,
    /largest guidance scale is always best/i,
    /maximum guidance always gives the best sample/i,
    /unconditional prediction is irrelevant/i,
    /conditional prediction is a final image classifier score/i,
    /zero guidance strongly amplifies/i,
    /still creates a large prompt direction/i,
    /never changes inference cost or memory/i,
    /behaves identically under every sampler/i,
    /always increases output diversity/i,
    /over-saturation proves the guidance scale is optimal/i,
    /cannot produce an unconditional prediction/i,
    /guarantee removal of unwanted features/i,
    /single lucky seed and one prompt/i,
    /safety filters are unnecessary/i,
    /guaranteed prompt-quality booster with no tradeoffs/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /false|unsafe|wrong|trap|reject|claim|belief|misconception/i.test(question.prompt);
    assert.ok(!unsafeAnswer || explicitTrapPrompt, `question ${index + 1} keys a false claim outside a trap prompt`);
  }
});

test('classifier-free guidance assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('classifier-free-guidance');

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

test('classifier-free guidance assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('classifier-free-guidance');

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const positions = quiz.slice(pageStart, pageStart + 10).map((question) => question.answerIndex);
    const maxSameSlot = Math.max(...[0, 1, 2].map((slot) => positions.filter((position) => position === slot).length));

    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
    assert.ok(maxSameSlot <= 6, `page starting at question ${pageStart + 1} should not overuse one answer position`);
  }
});
