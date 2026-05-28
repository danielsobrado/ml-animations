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

test('causal graphs dags has a complete curated 100-question assessment', () => {
  const { quiz } = getLessonAssessment('causal-graphs-dags');

  assert.equal(quiz.length, 100);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    assert.ok(question.id.startsWith('dag-'), `question ${index + 1} should use the dag id prefix`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(Object.hasOwn(LEVEL_ORDER, question.level), `question ${index + 1} should have a recognized level`);
  }
});

test('causal graphs dags assessment progresses from recall to interview readiness', () => {
  const { quiz } = getLessonAssessment('causal-graphs-dags');
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

test('causal graphs dags assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('causal-graphs-dags');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));

  const orderedMilestones = [
    ['dag purpose', ['causal dag']],
    ['directed arrow', ['directed arrow']],
    ['backdoor path', ['backdoor path']],
    ['confounder', ['confounder']],
    ['collider', ['collider']],
    ['mediator', ['mediator']],
    ['adjustment set', ['adjustment set']],
    ['first check', ['first dag adjustment check']],
    ['d-separation', ['d-separation']],
    ['collider differs from fork', ['collider differ from a fork']],
    ['minimal sufficient set', ['minimal sufficient adjustment set']],
    ['protocol', ['dag adjustment protocol']],
    ['scenario application', ['age affects treatment choice']],
    ['tricky false claims', ['dag claim is false']],
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

test('causal graphs dags assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('causal-graphs-dags');
  const unsafePatterns = [
    /guarantee/i,
    /guarantees/i,
    /proves causality/i,
    /adjust for every available/i,
    /always removes bias/i,
    /caused by treatment/i,
    /always correct for total effects/i,
    /never valid/i,
    /fully solves/i,
    /removes the need for overlap/i,
    /always harmless/i,
    /small adjusted p-value proves/i,
    /data alone always determines/i,
    /without assumptions/i,
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

test('causal graphs dags assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('causal-graphs-dags');
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

test('causal graphs dags assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('causal-graphs-dags');

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const page = quiz.slice(pageStart, pageStart + 10);
    const positions = page.map((question) => question.answerIndex);

    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
  }
});
