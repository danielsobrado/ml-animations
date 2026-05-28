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

function normalized(value) {
  return String(value || '')
    .toLowerCase()
    .replace(/\s+/g, ' ')
    .trim();
}

function correctAnswer(question) {
  return question.choices[question.answerIndex];
}

test('train validation test split has a complete curated 100-question assessment', () => {
  const { quiz } = getLessonAssessment('train-validation-test-split');
  const ids = new Set(quiz.map((question) => question.id));

  assert.equal(quiz.length, 100);
  assert.equal(ids.size, 100);
  assert.ok(quiz.every((question) => !question.id.startsWith('generated-')));

  for (const question of quiz) {
    assert.ok(question.prompt && /\S/.test(question.prompt), `${question.id} should have a prompt`);
    assert.equal(question.choices.length, 3, `${question.id} should have three choices`);
    assert.ok(Number.isInteger(question.answerIndex), `${question.id} should have an integer answer index`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `${question.id} has invalid answer index`);
    assert.ok(question.explanation && /\S/.test(question.explanation), `${question.id} should explain the answer`);
  }
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
  const sections = [
    quiz.slice(0, 20).map((question) => normalized(`${question.prompt} ${question.explanation}`)).join(' '),
    quiz.slice(20, 50).map((question) => normalized(`${question.prompt} ${question.explanation}`)).join(' '),
    quiz.slice(50, 75).map((question) => normalized(`${question.prompt} ${question.explanation}`)).join(' '),
    quiz.slice(75).map((question) => normalized(`${question.prompt} ${question.explanation}`)).join(' '),
  ];

  assert.match(sections[0], /training|validation|test|stratified|grouped|time|preprocessing/);
  assert.match(sections[1], /leakage|duplicate|future|scaler|feature selection|random seed|freeze/);
  assert.match(sections[1], /unit|distribution shift|leaderboard|augmentation|release protocol/);
  assert.match(sections[2], /scenario|threshold|users|time|rare|calibration|monitoring/);
  assert.match(sections[3], /trap|interview|misconception|test|leakage|protocol/);
});

test('train validation test split assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('train-validation-test-split');
  const falseClaimPatterns = [
    /tuning based on test performance/,
    /random row splits are always safe/,
    /validation and test are the same thing/,
    /preprocessing before split is always harmless/,
    /passed test set as permanent proof/,
  ];

  for (const [index, question] of quiz.entries()) {
    const prompt = normalized(question.prompt);
    const answer = normalized(correctAnswer(question));
    const explicitTrapPrompt = /trap|misconception|what is wrong|risky|interview|respond|define/.test(prompt);
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
        if (questionIndex === answerIndex || answer.length < 8) continue;

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
    const maxSameSlot = Math.max(
      ...[0, 1, 2].map((slot) => page.filter((question) => question.answerIndex === slot).length),
    );

    assert.ok(
      maxSameSlot <= Math.ceil(page.length * 0.6),
      `page starting at question ${pageStart + 1} should not overuse one correct option slot`,
    );
  }
});

