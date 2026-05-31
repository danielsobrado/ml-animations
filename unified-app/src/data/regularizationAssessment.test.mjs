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

test('regularization has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('regularization');

  assert.equal(quiz.length, 100);
  assert.deepEqual(
    labs.map((lab) => lab.id),
    ['lambda-sweep', 'penalty-family-comparison', 'validation-sweep-report'],
  );
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.match(question.id, /^reg-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use an ordered reg id`);
    assert.equal(Number(question.id.slice(4, 7)), index + 1, `question ${index + 1} id should match its position`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.equal(new Set(question.choices.map(normalized)).size, question.choices.length, `question ${index + 1} choices should be distinct`);
    assert.ok(Number.isInteger(question.answerIndex), `question ${index + 1} answer index should be an integer`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(Object.hasOwn(LEVEL_ORDER, question.level), `question ${index + 1} should have a recognized level`);
  }
});

test('regularization assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('regularization');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('regularization assessment progresses from penalty intuition to interview readiness', () => {
  const { quiz } = getLessonAssessment('regularization');
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

test('regularization assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('regularization');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const orderedMilestones = [
    ['purpose', ['discourage unnecessary complexity']],
    ['objective shape', ['data loss plus lambda times a penalty term']],
    ['lambda', ['strength of the regularization penalty']],
    ['L1', ['sum of absolute weight magnitudes']],
    ['L2', ['sum of squared weight magnitudes']],
    ['feature scaling', ['penalty size depends on coefficient scale']],
    ['dropout', ['randomly disables activations']],
    ['early stopping', ['limits training before the model fully memorizes']],
    ['validation curve', ['fall first and then rise when underfitting begins']],
    ['application tuning', ['select that setting within the validation protocol']],
    ['tricky false claims', ['claim is false']],
    ['interview readiness', ['diagnosis method choice tuning discipline and verification']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of orderedMilestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('regularization assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('regularization');
  const unsafePatterns = [
    /always improves test performance/i,
    /chosen safely without validation/i,
    /largest lambda is always best/i,
    /always keeps every feature/i,
    /hard feature selection by zeroing many coefficients/i,
    /scaling never matters/i,
    /retry lambda choices based on final test scores/i,
    /normally applied only at final evaluation time/i,
    /any random input change is a safe regularizer/i,
    /removes the need for validation behavior/i,
    /likelihood no longer matters/i,
    /automatically causally interpretable/i,
    /smaller gap always means the model is better/i,
    /one regularizer is best for every model/i,
    /strongest penalty is production-ready by default/i,
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

test('regularization assessment keeps misconception traps after setup', () => {
  const { quiz } = getLessonAssessment('regularization');
  const expectedTrapIds = [
    'reg-076-trap-always-helps',
    'reg-077-trap-no-validation',
    'reg-078-trap-largest',
    'reg-079-trap-l1',
    'reg-080-trap-l2',
    'reg-081-trap-scaling',
    'reg-082-trap-test-tuning',
    'reg-083-trap-dropout',
    'reg-084-trap-augmentation',
    'reg-085-trap-early-stopping',
    'reg-086-trap-prior',
    'reg-087-trap-interpretability',
    'reg-088-trap-gap',
    'reg-089-trap-one-method',
    'reg-090-tricky-summary',
  ];
  const misconceptionPatterns = [
    /always improves test performance/i,
    /chosen safely without validation/i,
    /largest lambda is always best/i,
    /always keeps every feature/i,
    /hard feature selection by zeroing many coefficients/i,
    /scaling never matters/i,
    /retry lambda choices based on final test scores/i,
    /normally applied only at final evaluation time/i,
    /any random input change is a safe regularizer/i,
    /removes the need for validation behavior/i,
    /likelihood no longer matters/i,
    /automatically causally interpretable/i,
    /smaller gap always means the model is better/i,
    /one regularizer is best for every model/i,
  ];
  const trapPrompt = /false|unsafe|wrong|trap|claim/i;
  const trapQuestions = quiz.slice(75, 90);

  assert.deepEqual(trapQuestions.map((question) => question.id), expectedTrapIds);
  assert.ok(trapQuestions.every((question) => question.level === 'Tricky'));

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const containsMisconception = misconceptionPatterns.some((pattern) => pattern.test(answer));
    if (!containsMisconception) continue;
    assert.ok(index >= 75, `${question.id} introduces misconception too early`);
    assert.match(question.prompt, trapPrompt, `${question.id} should mark misconception as a trap`);
  }
});

test('regularization assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('regularization');
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

test('regularization assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('regularization');
  const totals = [0, 0, 0];

  for (const question of quiz) {
    totals[question.answerIndex] += 1;
  }

  assert.ok(Math.max(...totals) - Math.min(...totals) <= 1, `answer positions should be balanced: ${totals.join(', ')}`);

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const page = quiz.slice(pageStart, pageStart + 10);
    const positions = page.map((question) => question.answerIndex);

    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
  }
});
