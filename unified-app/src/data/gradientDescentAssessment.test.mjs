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

test('gradient descent has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('gradient-descent');

  assert.equal(quiz.length, 100);
  assert.deepEqual(
    labs.map((lab) => lab.id),
    ['tune-step-size', 'trace-one-update', 'diagnose-loss-curve'],
  );
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.match(question.id, /^gd-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use an ordered gd id`);
    assert.equal(Number(question.id.slice(3, 6)), index + 1, `question ${index + 1} id should match its position`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.equal(new Set(question.choices.map((choice) => normalized(choice))).size, 3, `question ${index + 1} choices should be distinct`);
    assert.equal(Number.isInteger(question.answerIndex), true, `question ${index + 1} answer index should be an integer`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(Object.hasOwn(LEVEL_ORDER, question.level), `question ${index + 1} should have a recognized level`);
  }
});

test('gradient descent assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('gradient-descent');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('gradient descent assessment progresses from update basics to interview readiness', () => {
  const { quiz } = getLessonAssessment('gradient-descent');
  const expectedBands = [
    ['Foundation', 0, 20],
    ['Mechanism', 20, 50],
    ['Application', 50, 75],
    ['Tricky', 75, 90],
    ['Interview', 90, 100],
  ];

  for (const [level, start, end] of expectedBands) {
    assert.deepEqual(
      [...new Set(quiz.slice(start, end).map((question) => question.level))],
      [level],
      `${level} questions should occupy positions ${start + 1}-${end}`,
    );
  }

  for (let index = 1; index < quiz.length; index += 1) {
    assert.ok(
      LEVEL_ORDER[quiz[index].level] >= LEVEL_ORDER[quiz[index - 1].level],
      `question ${index + 1} should not regress in difficulty`,
    );
  }
});

test('gradient descent assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('gradient-descent');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const orderedMilestones = [
    ['purpose', ['opposite the loss gradient']],
    ['loss', ['scalar objective']],
    ['gradient', ['steepest local increase direction']],
    ['negative gradient', ['local downhill step']],
    ['learning rate', ['step size']],
    ['mini-batch', ['subset of training examples']],
    ['validation loss', ['estimates generalization']],
    ['forward pass', ['predictions and loss']],
    ['backward pass', ['gradients of the loss']],
    ['curvature', ['large steps overshoot']],
    ['batch noise', ['less exact than full batch gradients']],
    ['zero_grad', ['clear accumulated gradients']],
    ['application workflow', ['stable training', 'validation based tuning']],
    ['tricky false claims', ['claim is false']],
    ['interview readiness', ['theta minus eta times gradient']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of orderedMilestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('gradient descent assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('gradient-descent');
  const unsafePatterns = [
    /always reaches the global minimum/i,
    /gradient points downhill, so gradient descent adds it/i,
    /larger learning rate is always better/i,
    /tiny learning rate always fixes bad training/i,
    /zero gradient always proves the best possible model/i,
    /lower training loss always means better deployed/i,
    /tune learning rate repeatedly on the test set/i,
    /validation batches should supply gradients/i,
    /feature scaling cannot affect gradient descent/i,
    /single noisy mini-batch loss proves/i,
    /adam automatically removes the need/i,
    /gradient clipping fixes every cause/i,
    /harmless to omit zero_grad/i,
    /all neural-network losses are convex/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /false|unsafe|wrong|trap|claim/i.test(question.prompt);

    assert.ok(
      !unsafeAnswer || explicitTrapPrompt,
      `question ${index + 1} keys a false claim outside an explicit trap prompt`,
    );
  }
});

test('gradient descent assessment keeps misconception traps after setup', () => {
  const { quiz } = getLessonAssessment('gradient-descent');
  const trapIds = [
    'gd-076-trap-global',
    'gd-077-trap-gradient-sign',
    'gd-078-trap-learning-rate',
    'gd-079-trap-small-lr',
    'gd-080-trap-zero-gradient',
    'gd-081-trap-train-loss',
    'gd-082-trap-test-tuning',
    'gd-083-trap-validation-gradient',
    'gd-084-trap-feature-scale',
    'gd-085-trap-batch',
    'gd-086-trap-adam',
    'gd-087-trap-clipping',
    'gd-088-trap-zero-grad',
    'gd-089-trap-convex',
    'gd-090-tricky-summary',
  ];
  const misconceptionPatterns = [
    /always reaches the global minimum/i,
    /gradient points downhill, so gradient descent adds it/i,
    /larger learning rate is always better/i,
    /tiny learning rate always fixes bad training/i,
    /zero gradient always proves the best possible model/i,
    /lower training loss always means better deployed/i,
    /tune learning rate repeatedly on the test set/i,
    /validation batches should supply gradients/i,
    /feature scaling cannot affect gradient descent/i,
    /single noisy mini-batch loss proves/i,
    /adam automatically removes the need/i,
    /gradient clipping fixes every cause/i,
    /harmless to omit zero_grad/i,
    /all neural-network losses are convex/i,
  ];
  const trapPrompt = /false|unsafe|wrong|trap|claim/i;
  const trapQuestions = quiz.slice(75, 90);

  assert.deepEqual(trapQuestions.map((question) => question.id), trapIds);
  assert.ok(trapQuestions.every((question) => question.level === 'Tricky'));

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const containsMisconception = misconceptionPatterns.some((pattern) => pattern.test(answer));
    if (!containsMisconception) continue;
    assert.ok(index >= 75, `${question.id} introduces misconception too early`);
    assert.match(question.prompt, trapPrompt, `${question.id} should mark misconception as a trap`);
  }
});

test('gradient descent assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('gradient-descent');
  const pageSize = 10;

  for (let pageStart = 0; pageStart < quiz.length; pageStart += pageSize) {
    const page = quiz.slice(pageStart, pageStart + pageSize);
    const answers = page.map((question) => normalized(correctAnswer(question)));

    assert.equal(new Set(answers).size, answers.length, `page starting at question ${pageStart + 1} should not repeat exact answers`);

    for (const [answerIndex, answer] of answers.entries()) {
      for (const [questionIndex, question] of page.entries()) {
        if (answerIndex === questionIndex) continue;
        const visibleText = normalized([question.prompt, ...question.choices].join(' '));
        assert.ok(
          !visibleText.includes(answer),
          `question ${pageStart + questionIndex + 1} visible text should not reveal answer from question ${pageStart + answerIndex + 1}`,
        );
      }
    }
  }
});

test('gradient descent assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('gradient-descent');
  const totals = [0, 0, 0];

  for (const question of quiz) {
    totals[question.answerIndex] += 1;
  }

  assert.ok(Math.max(...totals) - Math.min(...totals) <= 1, `answer positions should be globally balanced, saw ${totals.join('/')}`);

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const page = quiz.slice(pageStart, pageStart + 10);
    const positions = page.map((question) => question.answerIndex);

    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
  }
});
