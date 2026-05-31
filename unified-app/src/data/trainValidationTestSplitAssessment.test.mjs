import test from 'node:test';
import assert from 'node:assert/strict';

import { getLessonAssessment } from './lessonAssessments.js';

const LEVEL_ORDER = {
  Foundation: 0,
  Mechanism: 1,
  Application: 2,
  Tricky: 3,
  Interview: 4,
};

const LEVELS = new Set(Object.keys(LEVEL_ORDER));

function normalized(value) {
  return String(value || '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
}

function correctAnswer(question) {
  return question.choices[question.answerIndex];
}

test('train validation test split has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('train-validation-test-split');
  const ids = new Set(quiz.map((question) => question.id));

  assert.equal(quiz.length, 100);
  assert.equal(labs.length, 3);
  assert.equal(ids.size, 100);
  assert.ok(quiz.every((question) => !question.id.startsWith('generated-')));

  for (const [index, question] of quiz.entries()) {
    assert.match(question.id, /^tvt-\d{3}-[a-z0-9-]+$/, `${question.id} should use the curated id format`);
    assert.equal(Number(question.id.slice(4, 7)), index + 1, `${question.id} should stay in numeric order`);
    assert.ok(LEVELS.has(question.level), `${question.id} should use a known level`);
    assert.ok(question.prompt && question.prompt.length > 20, `${question.id} should have a substantial prompt`);
    assert.equal(question.choices.length, 3, `${question.id} should have three choices`);
    assert.equal(new Set(question.choices.map(normalized)).size, 3, `${question.id} should not repeat a choice`);
    assert.ok(Number.isInteger(question.answerIndex), `${question.id} should have an integer answer index`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `${question.id} has invalid answer index`);
    assert.ok(question.explanation && question.explanation.length > 30, `${question.id} should explain the answer`);
  }
});

test('train validation test split assessment avoids duplicate prompts and exact correct answers', () => {
  const { quiz } = getLessonAssessment('train-validation-test-split');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length, 'prompts should be unique');
  assert.equal(new Set(answers).size, answers.length, 'exact correct answers should be unique');
});

test('train validation test split assessment progresses from recall to interview readiness', () => {
  const { quiz } = getLessonAssessment('train-validation-test-split');

  const expectedBands = [
    ['Foundation', 0, 20],
    ['Mechanism', 20, 50],
    ['Application', 50, 75],
    ['Tricky', 75, 90],
    ['Interview', 90, 100],
  ];

  for (const [level, start, end] of expectedBands) {
    const band = quiz.slice(start, end);
    assert.ok(band.every((question) => question.level === level), `${level} band should occupy questions ${start + 1}-${end}`);
  }

  for (let index = 1; index < quiz.length; index += 1) {
    assert.ok(
      LEVEL_ORDER[quiz[index].level] >= LEVEL_ORDER[quiz[index - 1].level],
      `question ${index + 1} should not regress from ${quiz[index - 1].level} to ${quiz[index].level}`,
    );
  }
});

test('train validation test split assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('train-validation-test-split');
  const milestones = [
    [/main purpose of splitting data/, 0, 8],
    [/what is the training set for/, 0, 8],
    [/what is the validation set for/, 0, 10],
    [/what is the test set for/, 0, 10],
    [/stratified split/, 5, 18],
    [/grouped split/, 8, 20],
    [/time based split/, 8, 20],
    [/learned preprocessing/, 10, 22],
    [/duplicate users break a random row split/, 20, 35],
    [/target leakage/, 20, 38],
    [/nested cross validation/, 25, 42],
    [/before opening the test set/, 45, 55],
    [/standardscaler before train test split/, 50, 60],
    [/test score is low/, 55, 65],
    [/trap/, 75, 90],
    [/summarize train validation test splitting/, 90, 100],
  ];

  for (const [pattern, minIndex, maxIndex] of milestones) {
    const matchIndex = quiz.findIndex((question) => pattern.test(normalized(`${question.prompt} ${question.explanation}`)));
    assert.notEqual(matchIndex, -1, `missing learning point ${pattern}`);
    assert.ok(
      matchIndex >= minIndex && matchIndex < maxIndex,
      `${pattern} appears at question ${matchIndex + 1}, outside expected range ${minIndex + 1}-${maxIndex}`,
    );
  }
});

test('train validation test split assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('train-validation-test-split');
  const falseClaimPatterns = [
    /tuning based on test performance/,
    /random row splits are always safe/,
    /validation and test are the same thing/,
    /preprocessing before split is always harmless/,
    /passed test set as permanent proof/,
    /training on future/,
    /repeated submissions adapt/,
    /reporting the best test score/,
  ];

  for (const [index, question] of quiz.entries()) {
    const prompt = normalized(question.prompt);
    const answer = normalized(correctAnswer(question));
    const explicitTrapPrompt = /trap|misconception|what is wrong|risky|interview|respond|define|why can|why is/.test(prompt);
    const falseClaimKeyed = falseClaimPatterns.some((pattern) => pattern.test(answer));

    assert.ok(
      !falseClaimKeyed || explicitTrapPrompt,
      `question ${index + 1} keys a false claim outside an explicit trap prompt`,
    );
  }
});

test('train validation test split assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('train-validation-test-split');
  const pageSize = 10;

  for (let pageStart = 0; pageStart < quiz.length; pageStart += pageSize) {
    const page = quiz.slice(pageStart, pageStart + pageSize);
    const answers = page.map((question) => normalized(correctAnswer(question)));

    assert.equal(new Set(answers).size, answers.length, `page starting at question ${pageStart + 1} should not repeat exact answers`);

    for (const [questionIndex, question] of page.entries()) {
      const prompt = normalized(question.prompt);

      for (const [answerIndex, answer] of answers.entries()) {
        if (questionIndex === answerIndex) continue;

        assert.ok(
          !prompt.includes(answer),
          `question ${pageStart + questionIndex + 1} prompt should not reveal answer from question ${pageStart + answerIndex + 1}`,
        );
      }
    }
  }
});

test('train validation test split assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('train-validation-test-split');
  const pageSize = 10;

  assert.ok(new Set(quiz.map((question) => question.answerIndex)).size > 1);

  for (let pageStart = 0; pageStart < quiz.length; pageStart += pageSize) {
    const page = quiz.slice(pageStart, pageStart + pageSize);
    const counts = [0, 1, 2].map((slot) => page.filter((question) => question.answerIndex === slot).length);
    const maxSameSlot = Math.max(...counts);
    const minSameSlot = Math.min(...counts);

    assert.ok(
      maxSameSlot - minSameSlot <= 1,
      `page starting at question ${pageStart + 1} should balance correct option slots, got ${counts.join('/')}`,
    );
  }
});
