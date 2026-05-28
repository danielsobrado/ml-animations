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

test('cuped variance reduction has a complete curated 100-question assessment', () => {
  const { quiz } = getLessonAssessment('cuped-variance-reduction');

  assert.equal(quiz.length, 100);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.ok(question.id.startsWith('cuped-'), `question ${index + 1} should use the cuped id prefix`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(Object.hasOwn(LEVEL_ORDER, question.level), `question ${index + 1} should have a recognized level`);
  }
});

test('cuped variance reduction assessment progresses from recall to interview readiness', () => {
  const { quiz } = getLessonAssessment('cuped-variance-reduction');
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
      `${level} band should occupy questions ${start + 1}-${end}`,
    );
  }

  for (let index = 1; index < quiz.length; index += 1) {
    assert.ok(
      LEVEL_ORDER[quiz[index].level] >= LEVEL_ORDER[quiz[index - 1].level],
      `question ${index + 1} should not move backward in difficulty`,
    );
  }
});

test('cuped variance reduction assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('cuped-variance-reduction');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));

  const orderedMilestones = [
    ['purpose', ['reduce outcome noise']],
    ['pre-treatment covariate', ['before treatment assignment']],
    ['post-treatment trap', ['post-treatment covariates']],
    ['preserve estimand', ['same causal treatment effect target']],
    ['correlation lever', ['pre/post correlation']],
    ['rho squared', ['1 - rho squared']],
    ['theta adjustment', ['adjustment coefficient']],
    ['first validity check', ['first cuped validity check']],
    ['regression view', ['regression adjustment']],
    ['standard error scaling', ['raw standard error times the square root']],
    ['post-treatment bias mechanism', ['post-treatment covariate bias']],
    ['protocol', ['cuped protocol']],
    ['scenario application', ['pre-period metric has 80 percent']],
    ['tricky false claims', ['cuped claim is false']],
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

test('cuped variance reduction assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('cuped-variance-reduction');
  const unsafePatterns = [
    /guarantee/i,
    /guarantees/i,
    /replaces the need/i,
    /always safe/i,
    /changes the true treatment effect/i,
    /any nonzero correlation guarantees/i,
    /new independent users were created/i,
    /try many covariates/i,
    /fix broken logging/i,
    /always improves total-effect/i,
    /most flexible adjustment model is always/i,
    /silently dropped/i,
    /automatically makes any effect worth/i,
    /only the adjusted p-value/i,
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

test('cuped variance reduction assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('cuped-variance-reduction');
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

test('cuped variance reduction assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('cuped-variance-reduction');

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const page = quiz.slice(pageStart, pageStart + 10);
    const positions = page.map((question) => question.answerIndex);

    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
  }
});
