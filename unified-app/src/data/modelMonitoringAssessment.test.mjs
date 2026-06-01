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

const EXPECTED_LEVEL_COUNTS = {
  Foundation: 20,
  Mechanism: 30,
  Application: 25,
  Tricky: 15,
  Interview: 10,
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

test('model monitoring has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('model-monitoring');

  assert.equal(quiz.length, 100);
  assert.deepEqual(labs.map((lab) => lab.id), ['configure-playbook']);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);
  assert.deepEqual(
    Object.fromEntries(Object.keys(EXPECTED_LEVEL_COUNTS).map((level) => [
      level,
      quiz.filter((question) => question.level === level).length,
    ])),
    EXPECTED_LEVEL_COUNTS,
  );

  for (const [index, question] of quiz.entries()) {
    assert.match(question.id, /^mon-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use an ordered mon id`);
    assert.equal(Number(question.id.slice(4, 7)), index + 1, `question ${index + 1} id should match its position`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.equal(new Set(question.choices.map((choice) => normalized(choice))).size, 3, `question ${index + 1} choices should be distinct`);
    assert.equal(Number.isInteger(question.answerIndex), true, `question ${index + 1} answer index should be an integer`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(Object.hasOwn(LEVEL_ORDER, question.level), `question ${index + 1} should have a recognized level`);
  }
});

test('model monitoring assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('model-monitoring');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('model monitoring assessment progresses from signal basics to interview readiness', () => {
  const { quiz } = getLessonAssessment('model-monitoring');
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

test('model monitoring assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('model-monitoring');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const orderedMilestones = [
    ['purpose', ['live model behavior stays within expected bounds']],
    ['signal set', ['input drift performance calibration latency and throughput']],
    ['alert strictness', ['how easily monitored signals cross alert thresholds']],
    ['playbooks', ['investigate retrain or rollback']],
    ['time series mechanism', ['drift and quality signals evolve']],
    ['threshold tradeoff', ['false alerts and intervention noise']],
    ['data contract signals', ['data contracts and schemas']],
    ['application covariate shift', ['input drift rises while precision and recall']],
    ['playbook application', ['rollback to the last known good']],
    ['tricky false claims', ['claim is false']],
    ['interview readiness', ['production practice for tracking live data']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of orderedMilestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('model monitoring assessment avoids unsafe misconception keying before explicit traps', () => {
  const { quiz } = getLessonAssessment('model-monitoring');
  const unsafePatterns = [
    /service is up.*model is healthy/i,
    /drift can always be ignored/i,
    /every monitoring alert should trigger immediate retraining/i,
    /always improve operations with no downside/i,
    /stable accuracy proves probability calibration/i,
    /latency is irrelevant/i,
    /turning off a monitor removes/i,
    /single health score fully diagnoses/i,
    /one baseline stays valid/i,
    /playbook replaces the need/i,
    /no monitoring is possible/i,
    /stable global metric proves every subgroup/i,
    /rollback success is proven once the command finishes/i,
    /thresholds should never be revisited/i,
    /uptime.*automatic retraining are enough/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /false|unsafe|wrong|trap|claim|belief|reject/i.test(question.prompt);

    assert.ok(
      !unsafeAnswer || (index >= 75 && explicitTrapPrompt),
      `question ${index + 1} keys a false claim outside an explicit trap prompt`,
    );
  }
});

test('model monitoring misconception traps are placed after concept scaffolding', () => {
  const { quiz } = getLessonAssessment('model-monitoring');
  const trapIds = [
    'mon-076-trap-uptime-only',
    'mon-077-trap-drift-harmless',
    'mon-078-trap-retrain-all',
    'mon-079-trap-strictness',
    'mon-080-trap-calibration',
    'mon-081-trap-latency',
    'mon-082-trap-toggle',
    'mon-083-trap-health-score',
    'mon-084-trap-baseline',
    'mon-085-trap-playbook',
    'mon-086-trap-delayed-labels',
    'mon-087-trap-slice',
    'mon-088-trap-rollback',
    'mon-089-trap-thresholds',
    'mon-090-tricky-summary',
  ];
  const misconceptionPatterns = [
    /service is up.*healthy by definition/i,
    /input drift can always be ignored/i,
    /every monitoring alert should trigger/i,
    /always improve operations with no downside/i,
    /accuracy proves probability calibration/i,
    /latency is irrelevant/i,
    /turning off a monitor removes/i,
    /single health score fully diagnoses/i,
    /one baseline stays valid/i,
    /playbook replaces the need/i,
    /no monitoring is possible/i,
    /global metric proves every subgroup/i,
    /rollback success is proven/i,
    /thresholds should never be revisited/i,
    /automatic retraining are enough/i,
  ];
  const trapPrompt = /false|unsafe|wrong|trap|claim|belief|reject/i;

  assert.deepEqual(quiz.slice(75, 90).map((question) => question.id), trapIds);

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const containsMisconception = misconceptionPatterns.some((pattern) => pattern.test(answer));
    if (!containsMisconception) continue;
    assert.ok(index >= 75, `${question.id} introduces misconception too early`);
    assert.match(question.prompt, trapPrompt, `${question.id} should mark misconception as a trap`);
  }
});

test('model monitoring assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('model-monitoring');

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const page = quiz.slice(pageStart, pageStart + 10);
    const answers = page.map((question) => normalized(correctAnswer(question)));

    assert.equal(new Set(answers).size, answers.length, `page starting at question ${pageStart + 1} should not repeat exact answers`);

    for (const [answerIndex, answer] of answers.entries()) {
      for (const [promptIndex, question] of page.entries()) {
        if (answerIndex === promptIndex) continue;
        const visibleText = normalized([question.prompt, ...question.choices].join(' '));
        assert.ok(
          !visibleText.includes(answer),
          `question ${pageStart + promptIndex + 1} visible text should not reveal answer from question ${pageStart + answerIndex + 1}`,
        );
      }
    }
  }
});

test('model monitoring assessment keeps visible wording scoped to monitoring mechanics', () => {
  const { quiz } = getLessonAssessment('model-monitoring');
  const genericLeakagePatterns = [
    /model-owner/i,
    /model card being renamed/i,
    /metric name changes in a report/i,
    /final incident report title/i,
    /documentation issue/i,
    /feature labels/i,
    /chart/i,
  ];

  for (const question of quiz) {
    const visibleText = [question.prompt, ...question.choices, question.explanation].join(' ');
    for (const pattern of genericLeakagePatterns) {
      assert.doesNotMatch(visibleText, pattern, `${question.id} contains generic or off-scope visible wording`);
    }
  }
});

test('model monitoring assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('model-monitoring');
  const totals = [0, 0, 0];

  for (const question of quiz) {
    totals[question.answerIndex] += 1;
  }

  assert.ok(Math.max(...totals) - Math.min(...totals) <= 1, `answer positions should be globally balanced, saw ${totals.join('/')}`);

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const page = quiz.slice(pageStart, pageStart + 10);
    const pageTotals = [0, 0, 0];
    for (const question of page) {
      pageTotals[question.answerIndex] += 1;
    }

    assert.ok(
      Math.max(...pageTotals) - Math.min(...pageTotals) <= 1,
      `page starting at question ${pageStart + 1} should balance answer positions, saw ${pageTotals.join('/')}`,
    );
  }
});
