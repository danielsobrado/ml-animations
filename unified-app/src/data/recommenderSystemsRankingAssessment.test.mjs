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

test('recommender systems ranking has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('recommender-systems-ranking-track');

  assert.equal(quiz.length, 100);
  assert.deepEqual(
    labs.map((lab) => lab.id),
    ['recommender-systems-ranking-track-scenario-lab', 'ranking-metric-audit', 'feedback-loop-guardrails'],
  );
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.match(question.id, /^rec-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use a stable ordered rec id`);
    assert.equal(Number(question.id.slice(4, 7)), index + 1, `question ${index + 1} id should match its position`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.equal(
      new Set(question.choices.map((choice) => normalized(choice))).size,
      question.choices.length,
      `question ${index + 1} choices should be distinct`,
    );
    assert.ok(Number.isInteger(question.answerIndex), `question ${index + 1} answer index should be an integer`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(Object.hasOwn(LEVEL_ORDER, question.level), `question ${index + 1} should have a recognized level`);
  }
});

test('recommender systems ranking assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('recommender-systems-ranking-track');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('recommender systems ranking assessment progresses from signal basics to interview readiness', () => {
  const { quiz } = getLessonAssessment('recommender-systems-ranking-track');
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

test('recommender systems ranking assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('recommender-systems-ranking-track');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const orderedMilestones = [
    ['purpose', ['rank or select items', 'relevance']],
    ['exposure', ['chance to be seen']],
    ['candidate generation', ['plausible items before final ranking']],
    ['collaborative filtering', ['patterns of user item interactions']],
    ['cold start', ['little interaction history']],
    ['feedback loop', ['recommendations affect future interactions']],
    ['ranking objectives', ['pointwise ranking objective']],
    ['position bias', ['more interactions because of placement']],
    ['ranking metrics', ['precision k']],
    ['application workflow', ['split design', 'candidate policy', 'metrics k']],
    ['tricky false claims', ['claim is false']],
    ['interview readiness', ['define the exposure problem']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of orderedMilestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('recommender systems ranking assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('recommender-systems-ranking-track');
  const unsafePatterns = [
    /only predicts labels and does not affect future data/i,
    /every unobserved user-item pair is a true negative/i,
    /every click is an unbiased measure/i,
    /ranking position never changes/i,
    /most popular items are always/i,
    /always guarantees a better online/i,
    /cold start is solved automatically/i,
    /exploration is always useless/i,
    /accuracy on isolated pairs is enough/i,
    /ndcg ignores/i,
    /coverage and diversity never matter/i,
    /interactions that happen after the recommendation timestamp/i,
    /only one metric and no guardrails/i,
    /no user or item group can be harmed/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /false|unsafe|wrong|trap|claim|leakage|incorrect/i.test(question.prompt);

    assert.ok(
      !unsafeAnswer || explicitTrapPrompt,
      `question ${index + 1} keys a false claim outside an explicit trap prompt`,
    );
  }
});

test('recommender systems ranking assessment keeps misconception traps after setup', () => {
  const { quiz } = getLessonAssessment('recommender-systems-ranking-track');
  const misconceptionPatterns = [
    /only predicts labels and does not affect future data/i,
    /every unobserved user-item pair is a true negative/i,
    /every click is an unbiased measure/i,
    /ranking position never changes/i,
    /most popular items are always/i,
    /always guarantees a better online/i,
    /cold start is solved automatically/i,
    /exploration is always useless/i,
    /accuracy on isolated pairs is enough/i,
    /ndcg ignores/i,
    /coverage and diversity never matter/i,
    /interactions that happen after the recommendation timestamp/i,
    /only one metric and no guardrails/i,
    /no user or item group can be harmed/i,
  ];
  const trapPrompt = /false|unsafe|wrong|trap|claim|leakage|incorrect/i;

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const containsMisconception = misconceptionPatterns.some((pattern) => pattern.test(answer));
    if (!containsMisconception) continue;
    assert.ok(index >= 75, `${question.id} introduces misconception too early`);
    assert.match(question.prompt, trapPrompt, `${question.id} should mark misconception as a trap`);
  }
});

test('recommender systems ranking assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('recommender-systems-ranking-track');
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

test('recommender systems ranking assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('recommender-systems-ranking-track');
  const totals = [0, 0, 0];

  for (const question of quiz) {
    totals[question.answerIndex] += 1;
  }

  assert.ok(Math.max(...totals) - Math.min(...totals) <= 1, `answer positions should be globally balanced: ${totals.join(', ')}`);

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const page = quiz.slice(pageStart, pageStart + 10);
    const positions = page.map((question) => question.answerIndex);

    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
  }
});
