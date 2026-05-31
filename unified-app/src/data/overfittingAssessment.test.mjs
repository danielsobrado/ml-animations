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

test('overfitting has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('overfitting');

  assert.equal(quiz.length, 100);
  assert.deepEqual(labs.map((lab) => lab.id), [
    'find-sweet-spot',
    'generalization-gap-diagnosis',
    'remedy-and-report-plan',
  ]);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.match(question.id, /^overfit-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use an ordered overfit id`);
    assert.equal(question.id.slice(8, 11), String(index + 1).padStart(3, '0'), `question ${index + 1} id should match its position`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.equal(new Set(question.choices.map(normalized)).size, 3, `question ${index + 1} choices should be distinct`);
    assert.ok(Number.isInteger(question.answerIndex), `question ${index + 1} answer index should be an integer`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(Object.hasOwn(LEVEL_ORDER, question.level), `question ${index + 1} should have a recognized level`);
  }
});

test('overfitting assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('overfitting');
  const prompts = new Map();
  const correctAnswers = new Map();

  for (const [index, question] of quiz.entries()) {
    const questionNumber = index + 1;
    const prompt = normalized(question.prompt);
    const answer = normalized(correctAnswer(question));

    assert.ok(!prompts.has(prompt), `questions ${prompts.get(prompt)} and ${questionNumber} repeat the same prompt`);
    assert.ok(!correctAnswers.has(answer), `questions ${correctAnswers.get(answer)} and ${questionNumber} repeat the same correct answer`);

    prompts.set(prompt, questionNumber);
    correctAnswers.set(answer, questionNumber);
  }
});

test('overfitting assessment progresses from diagnosis to interview readiness', () => {
  const { quiz } = getLessonAssessment('overfitting');
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

test('overfitting assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('overfitting');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const orderedMilestones = [
    ['classic signal', ['training error falls', 'validation error rises']],
    ['generalization gap', ['difference between training performance and held out performance']],
    ['early stopping', ['best validation point before later memorization']],
    ['model capacity', ['range of patterns a model family can represent']],
    ['validation overfit', ['repeated choices adapt to validation noise']],
    ['regularization strength', ['reduces flexibility but can eventually underfit']],
    ['selection risk', ['winning score may be overfit to validation noise']],
    ['data augmentation', ['label preserving augmentation']],
    ['production monitoring', ['fresh outcome performance and drift']],
    ['tricky false claims', ['claim is false']],
    ['interview readiness', ['production ready answer']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of orderedMilestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('overfitting assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('overfitting');
  const unsafePatterns = [
    /always means a better deployed model/i,
    /largest model is always/i,
    /any gap between train and validation proves/i,
    /classic overfitting is the only possible explanation/i,
    /reused endlessly with no selection bias/i,
    /guarantees no overfitting to model selection/i,
    /more regularization is always better/i,
    /more rows always fix overfitting/i,
    /any random transformation improves generalization/i,
    /always prove the model has generalized perfectly/i,
    /safe to tune repeatedly on the final test set/i,
    /good average validation guarantees every segment is safe/i,
    /single failure proves the model is overfit/i,
    /deploy the checkpoint with the lowest training loss/i,
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

test('overfitting assessment keeps misconception traps after setup', () => {
  const { quiz } = getLessonAssessment('overfitting');
  const expectedTrapIds = [
    'overfit-076-trap-train-loss',
    'overfit-077-trap-big-model',
    'overfit-078-trap-gap',
    'overfit-079-trap-underfit',
    'overfit-080-trap-validation',
    'overfit-081-trap-crossval',
    'overfit-082-trap-regularization',
    'overfit-083-trap-more-data',
    'overfit-084-trap-augmentation',
    'overfit-085-trap-leakage',
    'overfit-086-trap-final-test',
    'overfit-087-trap-slices',
    'overfit-088-trap-complexity',
    'overfit-089-trap-one-point',
    'overfit-090-tricky-summary',
  ];
  const misconceptionPatterns = [
    /always means a better deployed model/i,
    /largest model is always/i,
    /any gap between train and validation proves/i,
    /classic overfitting is the only possible explanation/i,
    /reused endlessly with no selection bias/i,
    /guarantees no overfitting to model selection/i,
    /more regularization is always better/i,
    /more rows always fix overfitting/i,
    /any random transformation improves generalization/i,
    /always prove the model has generalized perfectly/i,
    /safe to tune repeatedly on the final test set/i,
    /good average validation guarantees every segment is safe/i,
    /only matter for neural networks/i,
    /single failure proves the model is overfit/i,
  ];
  const trapPrompt = /false|unsafe|wrong|trap|claim/i;

  assert.deepEqual(quiz.slice(75, 90).map((question) => question.id), expectedTrapIds);

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const containsMisconception = misconceptionPatterns.some((pattern) => pattern.test(answer));
    if (!containsMisconception) continue;

    assert.ok(index >= 75, `${question.id} introduces misconception too early`);
    assert.match(question.prompt, trapPrompt, `${question.id} should mark misconception as a trap`);
  }
});

test('overfitting assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('overfitting');
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

test('overfitting assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('overfitting');
  const totals = [0, 0, 0];

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const page = quiz.slice(pageStart, pageStart + 10);
    const positions = page.map((question) => question.answerIndex);
    positions.forEach((position) => {
      totals[position] += 1;
    });

    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
  }

  assert.ok(
    Math.max(...totals) - Math.min(...totals) <= 1,
    `correct answers should be balanced across positions, got ${totals.join(', ')}`,
  );
});
