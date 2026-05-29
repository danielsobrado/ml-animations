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

test('diffusion basics has a complete curated 100-question assessment', () => {
  const { quiz } = getLessonAssessment('diffusion-basics');

  assert.equal(quiz.length, 100);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.ok(question.id.startsWith('diffbas-'), `question ${index + 1} should use the diffbas id prefix`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(Object.hasOwn(LEVEL_ORDER, question.level), `question ${index + 1} should have a recognized level`);
  }
});

test('diffusion basics assessment progresses from forward process to interview readiness', () => {
  const { quiz } = getLessonAssessment('diffusion-basics');
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

test('diffusion basics assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('diffusion-basics');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const milestones = [
    ['purpose', ['recover structure by learning how noise was added']],
    ['forward process', ['clean data is gradually mixed with controlled noise']],
    ['noise prediction', ['noise added to the clean sample at a timestep']],
    ['not one step', ['not asked to invent the whole image in one step']],
    ['mechanism summary', ['known noise process to train a timestep-aware denoiser']],
    ['lab high timestep', ['timestep moves higher']],
    ['production review', ['timestep and noise-prediction error change the recovered sample']],
    ['tricky false claims', ['invents the whole image in one denoising call']],
    ['interview debugging', ['noise-prediction error, timestep schedule, step count, and data scaling']],
    ['interview readiness', ['known noising process trains a timestep-aware denoiser']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of milestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('diffusion basics assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('diffusion-basics');
  const unsafePatterns = [
    /invents the whole image in one denoising call/i,
    /forward process is learned by the model during generation/i,
    /never needs to know which timestep/i,
    /epsilon is the final clean image/i,
    /without knowing the noise that was added/i,
    /always easier because they contain more clean detail/i,
    /cannot damage the clean signal/i,
    /no longer needs any decoder/i,
    /replaces the denoising objective entirely/i,
    /noise is unrelated to timestep/i,
    /access to the original clean sample/i,
    /cost is independent of the number of denoising steps/i,
    /clean training examples do not matter/i,
    /discard arbitrary detail with no generation impact/i,
    /just random noise with no learned reverse structure/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /false|unsafe|wrong|trap|reject|claim|belief|misconception/i.test(question.prompt);
    assert.ok(!unsafeAnswer || explicitTrapPrompt, `question ${index + 1} keys a false claim outside a trap prompt`);
  }
});

test('diffusion basics assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('diffusion-basics');

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

test('diffusion basics assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('diffusion-basics');

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const positions = quiz.slice(pageStart, pageStart + 10).map((question) => question.answerIndex);
    const maxSameSlot = Math.max(...[0, 1, 2].map((slot) => positions.filter((position) => position === slot).length));

    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
    assert.ok(maxSameSlot <= 6, `page starting at question ${pageStart + 1} should not overuse one answer position`);
  }
});
