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

test('knn naive bayes svm has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('knn-naive-bayes-svm');

  assert.equal(quiz.length, 100);
  assert.deepEqual(
    labs.map((lab) => lab.id),
    ['boundary-comparison', 'preprocessing-and-scale-check', 'hyperparameter-selection-report'],
  );
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.match(question.id, /^knnnbsvm-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use an ordered knnnbsvm id`);
    assert.equal(Number(question.id.slice(9, 12)), index + 1, `question ${index + 1} id should match its position`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.equal(new Set(question.choices.map(normalized)).size, question.choices.length, `question ${index + 1} choices should be distinct`);
    assert.ok(Number.isInteger(question.answerIndex), `question ${index + 1} answer index should be an integer`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(Object.hasOwn(LEVEL_ORDER, question.level), `question ${index + 1} should have a recognized level`);
  }
});

test('knn naive bayes svm assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('knn-naive-bayes-svm');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('knn naive bayes svm assessment progresses from model ideas to interview readiness', () => {
  const { quiz } = getLessonAssessment('knn-naive-bayes-svm');
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

test('knn naive bayes svm assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('knn-naive-bayes-svm');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const orderedMilestones = [
    ['purpose', ['neighbor voting', 'probabilistic likelihoods', 'margin based classification']],
    ['kNN idea', ['nearby training examples']],
    ['Naive Bayes idea', ['class priors', 'per feature likelihoods']],
    ['SVM idea', ['large margin']],
    ['kNN scaling', ['largest unit feature']],
    ['Naive Bayes independence', ['conditionally independent given the class']],
    ['SVM support vectors', ['determine the svm boundary']],
    ['mechanics comparison', ['locality', 'distributional assumptions', 'margin geometry']],
    ['application comparison', ['assumptions match the data geometry']],
    ['tricky false claims', ['claim is false']],
    ['interview readiness', ['model intuition with disciplined evaluation']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of orderedMilestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('knn naive bayes svm assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('knn-naive-bayes-svm');
  const unsafePatterns = [
    /always best for every dataset/i,
    /scale never affects kNN/i,
    /compact parametric boundary/i,
    /k = 1 is always/i,
    /always exactly true/i,
    /zero likelihoods are never/i,
    /always perfectly calibrated/i,
    /scale is irrelevant for SVM/i,
    /every future prediction is correct/i,
    /larger C always improves/i,
    /automatically prevents overfitting/i,
    /automatically calibrated probabilities/i,
    /repeated final-test retries/i,
    /accuracy alone always chooses/i,
    /interchangeable/i,
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

test('knn naive bayes svm assessment keeps misconception traps after setup', () => {
  const { quiz } = getLessonAssessment('knn-naive-bayes-svm');
  const expectedTrapIds = [
    'knnnbsvm-076-trap-best',
    'knnnbsvm-077-trap-knn-scale',
    'knnnbsvm-078-trap-knn-training',
    'knnnbsvm-079-trap-k',
    'knnnbsvm-080-trap-nb-independence',
    'knnnbsvm-081-trap-nb-zero',
    'knnnbsvm-082-trap-nb-calibration',
    'knnnbsvm-083-trap-svm-scale',
    'knnnbsvm-084-trap-margin',
    'knnnbsvm-085-trap-c',
    'knnnbsvm-086-trap-kernel',
    'knnnbsvm-087-trap-probabilities',
    'knnnbsvm-088-trap-final-test',
    'knnnbsvm-089-trap-accuracy',
    'knnnbsvm-090-tricky-summary',
  ];
  const misconceptionPatterns = [
    /always best for every dataset/i,
    /scale never affects kNN/i,
    /compact parametric boundary/i,
    /k = 1 is always/i,
    /always exactly true/i,
    /zero likelihoods are never/i,
    /always perfectly calibrated/i,
    /scale is irrelevant for SVM/i,
    /every future prediction is correct/i,
    /larger C always improves/i,
    /automatically prevents overfitting/i,
    /automatically calibrated probabilities/i,
    /repeated final-test retries/i,
    /accuracy alone always chooses/i,
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

test('knn naive bayes svm assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('knn-naive-bayes-svm');
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

test('knn naive bayes svm assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('knn-naive-bayes-svm');
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
