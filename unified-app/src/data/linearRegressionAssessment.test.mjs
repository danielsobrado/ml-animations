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

test('linear regression has a complete curated 100-question assessment', () => {
  const { quiz } = getLessonAssessment('linear-regression');
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

test('linear regression assessment progresses from recall to interview readiness', () => {
  const { quiz } = getLessonAssessment('linear-regression');

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

test('linear regression assessment avoids prior generated-question defects', () => {
  const { quiz } = getLessonAssessment('linear-regression');
  const joined = normalized(quiz.flatMap((question) => [
    question.id,
    question.prompt,
    question.explanation,
    ...question.choices,
  ]).join(' '));

  assert.ok(!joined.includes('generated-'), 'assessment should not include generated question ids');
  assert.ok(!joined.includes('when use the animation'), 'assessment should not contain malformed generated prompt text');
  assert.ok(!joined.includes('answer letter appeared'), 'assessment should not include generic strategy-review distractors');
  assert.ok(!joined.includes('core linear algebra operation'), 'assessment should not inherit unrelated generic filler');
  assert.ok(!joined.includes('fitting linear models versus the shortcut'), 'assessment should not key safe fallback caveats as misconception traps');
});

test('linear regression assessment keys false claims only for explicit trap prompts', () => {
  const { quiz } = getLessonAssessment('linear-regression');
  const falseClaimPatterns = [
    /proven to cause/,
    /will generalize/,
    /delete every inconvenient point/,
    /bigger raw coefficients/,
    /causal effect/,
    /high-scoring line is production-ready/,
  ];

  for (const [index, question] of quiz.entries()) {
    const prompt = normalized(question.prompt);
    const answer = normalized(correctAnswer(question));
    const explicitTrapPrompt = /unsafe|not conclude|risky|trap|misunderstands|hard to interpret/.test(prompt);
    const falseClaimKeyed = falseClaimPatterns.some((pattern) => pattern.test(answer));

    assert.ok(
      !falseClaimKeyed || explicitTrapPrompt,
      `question ${index + 1} keys a false claim outside an explicit trap prompt`,
    );
  }
});

test('linear regression assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('linear-regression');
  const pageSize = 10;

  for (let pageStart = 0; pageStart < quiz.length; pageStart += pageSize) {
    const page = quiz.slice(pageStart, pageStart + pageSize);
    const answers = page.map((question) => normalized(correctAnswer(question)));

    assert.equal(new Set(answers).size, answers.length, `page starting at question ${pageStart + 1} should not repeat exact correct answers`);

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
