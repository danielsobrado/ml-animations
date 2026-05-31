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

test('maximum likelihood estimation has a complete curated 100-question assessment', () => {
  const { quiz } = getLessonAssessment('maximum-likelihood-estimation');

  assert.equal(quiz.length, 100);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.ok(question.id.startsWith('mle-'), `question ${index + 1} should use the mle id prefix`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(Object.hasOwn(LEVEL_ORDER, question.level), `question ${index + 1} should have a recognized level`);
  }
});

test('maximum likelihood estimation assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('maximum-likelihood-estimation');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('maximum likelihood estimation assessment progresses from recall to interview readiness', () => {
  const { quiz } = getLessonAssessment('maximum-likelihood-estimation');
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

test('maximum likelihood estimation assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('maximum-likelihood-estimation');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const orderedMilestones = [
    ['purpose', ['observed data', 'probable']],
    ['likelihood meaning', ['likelihood', 'candidate']],
    ['Bernoulli model', ['bernoulli']],
    ['Bernoulli MLE', ['observed success fraction']],
    ['Gaussian mean', ['sample average']],
    ['log-likelihood', ['log-likelihood']],
    ['negative log-likelihood', ['negative log-likelihood']],
    ['not prior', ['prior probability']],
    ['more data', ['more data', 'wrong candidate']],
    ['model family', ['model family']],
    ['validation', ['validation']],
    ['scenario application', ['80 of 100']],
    ['tricky false claims', ['claim is false']],
    ['interview readiness', ['interview']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of orderedMilestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('maximum likelihood estimation assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('maximum-likelihood-estimation');
  const unsafePatterns = [
    /proves the model family is true/i,
    /prior probability that theta is true/i,
    /unrelated to maximizing likelihood/i,
    /always 0\.5 regardless of data/i,
    /ignores residuals/i,
    /changes which candidate is best/i,
    /assigning zero probability.*safest/i,
    /guarantees best generalization/i,
    /equally plausible/i,
    /posterior probability of the candidate/i,
    /failures can be ignored/i,
    /automatically chooses the correct model family/i,
    /always include the same prior/i,
    /automatically gives a valid p-value/i,
    /without data or assumptions/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /false|misleading|unsafe|wrong|trap|claim/i.test(question.prompt);

    assert.ok(
      !unsafeAnswer || explicitTrapPrompt,
      `question ${index + 1} keys a false claim outside an explicit trap prompt`,
    );
  }
});

test('maximum likelihood estimation assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('maximum-likelihood-estimation');
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

test('maximum likelihood estimation assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('maximum-likelihood-estimation');

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const page = quiz.slice(pageStart, pageStart + 10);
    const positions = page.map((question) => question.answerIndex);

    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
  }
});
