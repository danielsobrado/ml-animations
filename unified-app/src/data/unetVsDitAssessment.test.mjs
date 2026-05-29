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

test('unet vs dit has a complete curated 100-question assessment', () => {
  const { quiz } = getLessonAssessment('unet-vs-dit');

  assert.equal(quiz.length, 100);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.ok(question.id.startsWith('unetdit-'), `question ${index + 1} should use the unetdit id prefix`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(Object.hasOwn(LEVEL_ORDER, question.level), `question ${index + 1} should have a recognized level`);
  }
});

test('unet vs dit assessment progresses from architecture basics to interview readiness', () => {
  const { quiz } = getLessonAssessment('unet-vs-dit');
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

test('unet vs dit assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('unet-vs-dit');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const milestones = [
    ['purpose', ['different diffusion denoiser architectures']],
    ['unet bias', ['local convolutional structure with downsample-upsample skip connections']],
    ['dit patches', ['processed as a transformer token sequence']],
    ['not bigger unet', ['not just a larger u-net']],
    ['mechanism summary', ['local multiscale bias']],
    ['patch sweep', ['smaller patches while measuring token cost and quality']],
    ['production review', ['learner can predict how resolution, patch size, depth, and backbone']],
    ['tricky false claims', ['same representation and mixing operation']],
    ['interview memory debugging', ['token count from resolution and patch size']],
    ['interview readiness', ['real data and compute constraints']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of milestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('unet vs dit assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('unet-vs-dit');
  const unsafePatterns = [
    /just a bigger u-net/i,
    /universally best/i,
    /attention cost is unrelated to token count/i,
    /always improve quality with no cost/i,
    /free at any resolution/i,
    /cannot model multiscale image structure/i,
    /cannot exchange global information/i,
    /need no positional information/i,
    /only decorative/i,
    /removes the need for diffusion timestep/i,
    /never needs to return a prediction compatible/i,
    /never depend on training data scale/i,
    /one cherry-picked image/i,
    /prove u-net and dit are identical/i,
    /only an acronym preference/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /false|unsafe|wrong|trap|reject|claim|belief|misconception/i.test(question.prompt);
    assert.ok(!unsafeAnswer || explicitTrapPrompt, `question ${index + 1} keys a false claim outside a trap prompt`);
  }
});

test('unet vs dit assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('unet-vs-dit');

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

test('unet vs dit assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('unet-vs-dit');

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const positions = quiz.slice(pageStart, pageStart + 10).map((question) => question.answerIndex);
    const maxSameSlot = Math.max(...[0, 1, 2].map((slot) => positions.filter((position) => position === slot).length));

    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
    assert.ok(maxSameSlot <= 6, `page starting at question ${pageStart + 1} should not overuse one answer position`);
  }
});
