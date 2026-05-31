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

test('diffusion sampling has a complete curated 100-question assessment', () => {
  const { quiz } = getLessonAssessment('diffusion-sampling');

  assert.equal(quiz.length, 100);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.ok(question.id.startsWith('diffsamp-'), `question ${index + 1} should use the diffsamp id prefix`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(Object.hasOwn(LEVEL_ORDER, question.level), `question ${index + 1} should have a recognized level`);
  }
});

test('diffusion sampling assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('diffusion-sampling');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('diffusion sampling assessment progresses from sampler basics to interview readiness', () => {
  const { quiz } = getLessonAssessment('diffusion-sampling');
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

test('diffusion sampling assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('diffusion-sampling');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const milestones = [
    ['purpose', ['trained denoiser into a generation procedure']],
    ['ddpm', ['retaining stochastic noise']],
    ['ddim', ['deterministic reverse path']],
    ['not new model', ['sampler choices using a trained denoiser']],
    ['mechanism summary', ['discrete or continuous paths from noise to data']],
    ['few-step tradeoff', ['each denoising jump harder']],
    ['production review', ['sampler type, step count, and denoiser quality affect']],
    ['tricky false claims', ['unrelated image models rather than sampling procedures']],
    ['interview debugging', ['step schedules, solver accuracy, denoiser error']],
    ['interview readiness', ['trade speed, reproducibility, diversity, and quality']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of milestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('diffusion sampling assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('diffusion-sampling');
  const unsafePatterns = [
    /unrelated image models rather than sampling procedures/i,
    /always higher quality than stochastic sampling/i,
    /always improve quality because they are faster/i,
    /always improves diversity and quality together/i,
    /no relationship to inference cost/i,
    /makes denoiser prediction quality irrelevant/i,
    /removes the need for a trained denoising model/i,
    /denoiser output is ignored/i,
    /has no path from noise to data/i,
    /cannot affect the final sample/i,
    /cannot affect ode-style sampling output/i,
    /can never affect visible detail/i,
    /one lucky seed only/i,
    /automatically production-ready/i,
    /harmless knobs with no quality tradeoffs/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /false|unsafe|wrong|trap|reject|claim|belief|misconception/i.test(question.prompt);
    assert.ok(!unsafeAnswer || explicitTrapPrompt, `question ${index + 1} keys a false claim outside a trap prompt`);
  }
});

test('diffusion sampling assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('diffusion-sampling');

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

test('diffusion sampling assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('diffusion-sampling');

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const positions = quiz.slice(pageStart, pageStart + 10).map((question) => question.answerIndex);
    const maxSameSlot = Math.max(...[0, 1, 2].map((slot) => positions.filter((position) => position === slot).length));

    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
    assert.ok(maxSameSlot <= 6, `page starting at question ${pageStart + 1} should not overuse one answer position`);
  }
});
