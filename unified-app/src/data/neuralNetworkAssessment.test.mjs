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

test('neural network has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('neural-network');

  assert.equal(quiz.length, 100);
  assert.deepEqual(
    labs.map((lab) => lab.id),
    ['trace-forward-backward', 'map-layer-shapes', 'review-generalization-evidence'],
  );
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.match(question.id, /^nn-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use an ordered nn id`);
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

test('neural network assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('neural-network');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('neural network assessment progresses from layer basics to interview readiness', () => {
  const { quiz } = getLessonAssessment('neural-network');
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

test('neural network assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('neural-network');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const orderedMilestones = [
    ['purpose', ['flexible mapping']],
    ['weights', ['scales how strongly']],
    ['bias', ['shifts the pre activation']],
    ['activation', ['introduce nonlinearity']],
    ['hidden layer', ['intermediate representation']],
    ['loss', ['training target']],
    ['forward pass', ['transformed layer by layer']],
    ['linear stack warning', ['one linear transformation']],
    ['data split boundary', ['validation checks tuning choices']],
    ['dense equation', ['multiplied by weights']],
    ['gradient flow', ['parameter changes would affect the loss']],
    ['capacity', ['rich a set of functions']],
    ['overfitting', ['performs worse on new data']],
    ['production readiness', ['clean data boundaries']],
    ['tricky false claims', ['claim is unsafe']],
    ['interview readiness', ['layer equation']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of orderedMilestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('neural network assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('neural-network');
  const unsafePatterns = [
    /adding more layers always improves/i,
    /nonlinear activations are optional/i,
    /more neurons guarantee better generalization/i,
    /zero training loss proves/i,
    /validation examples should update weights/i,
    /redesigning the network after each test-set result/i,
    /0\.99 is always a calibrated probability/i,
    /every hidden neuron always corresponds/i,
    /backpropagation is magic unrelated/i,
    /stores each training example exactly/i,
    /starting weights never affect training/i,
    /always finds the global best solution/i,
    /accuracy alone is always enough/i,
    /cannot be tested or debugged/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /false|unsafe|wrong|trap|claim|too narrow|unhelpful/i.test(question.prompt);

    assert.ok(
      !unsafeAnswer || explicitTrapPrompt,
      `question ${index + 1} keys a false claim outside an explicit trap prompt`,
    );
  }
});

test('neural network assessment keeps misconception traps after setup', () => {
  const { quiz } = getLessonAssessment('neural-network');
  const misconceptionPatterns = [
    /adding more layers always improves/i,
    /nonlinear activations are optional/i,
    /more neurons guarantee better generalization/i,
    /zero training loss proves/i,
    /validation examples should update weights/i,
    /redesigning the network after each test-set result/i,
    /0\.99 is always a calibrated probability/i,
    /every hidden neuron always corresponds/i,
    /backpropagation is magic unrelated/i,
    /stores each training example exactly/i,
    /starting weights never affect training/i,
    /always finds the global best solution/i,
    /accuracy alone is always enough/i,
    /cannot be tested or debugged/i,
  ];
  const trapPrompt = /false|unsafe|wrong|trap|claim|too narrow|unhelpful/i;

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const containsMisconception = misconceptionPatterns.some((pattern) => pattern.test(answer));
    if (!containsMisconception) continue;
    assert.ok(index >= 75, `${question.id} introduces misconception too early`);
    assert.match(question.prompt, trapPrompt, `${question.id} should mark misconception as a trap`);
  }
});

test('neural network assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('neural-network');
  const pageSize = 10;

  for (let pageStart = 0; pageStart < quiz.length; pageStart += pageSize) {
    const page = quiz.slice(pageStart, pageStart + pageSize);
    const answers = page.map((question) => normalized(correctAnswer(question)));

    assert.equal(new Set(answers).size, answers.length, `page starting at question ${pageStart + 1} should not repeat exact answers`);

    for (const [answerIndex, answer] of answers.entries()) {
      for (const [promptIndex, question] of page.entries()) {
        if (answerIndex === promptIndex) continue;
        assert.ok(
          !normalized(question.prompt).includes(answer),
          `question ${pageStart + promptIndex + 1} prompt should not reveal answer from question ${pageStart + answerIndex + 1}`,
        );
      }
    }
  }
});

test('neural network assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('neural-network');
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
