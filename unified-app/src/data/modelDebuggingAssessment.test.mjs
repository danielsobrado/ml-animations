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

test('model debugging has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('model-debugging');

  assert.equal(quiz.length, 100);
  assert.deepEqual(labs.map((lab) => lab.id), ['debug-loop']);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);
  assert.deepEqual(
    Object.fromEntries(Object.keys(EXPECTED_LEVEL_COUNTS).map((level) => [
      level,
      quiz.filter((question) => question.level === level).length,
    ])),
    EXPECTED_LEVEL_COUNTS,
  );

  for (const [index, question] of quiz.entries()) {
    assert.match(question.id, /^dbg-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use an ordered dbg id`);
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

test('model debugging assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('model-debugging');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('model debugging assessment progresses from incident basics to interview readiness', () => {
  const { quiz } = getLessonAssessment('model-debugging');
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

test('model debugging assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('model-debugging');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const orderedMilestones = [
    ['purpose', ['where expected model behavior first diverged']],
    ['pipeline stages', ['data training evaluation and serving stages']],
    ['slice analysis', ['subgroup failures can be hidden']],
    ['targeted intervention', ['one supported root cause hypothesis']],
    ['segment regression', ['recent traffic mix shifted']],
    ['leakage scenario', ['target derived external identifier']],
    ['serving contract', ['feature ordering and preprocessing version']],
    ['before after verification', ['remeasurement tests']],
    ['application triage', ['new region has much higher error']],
    ['tricky false claims', ['claim is false']],
    ['interview readiness', ['disciplined process for localizing ml failures']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of orderedMilestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('model debugging assessment avoids unsafe misconception keying before explicit traps', () => {
  const { quiz } = getLessonAssessment('model-debugging');
  const unsafePatterns = [
    /start with random hyperparameter tuning/i,
    /stable aggregate score proves no subgroup/i,
    /global threshold tweak is always/i,
    /high validation always proves/i,
    /frozen model weights guarantee/i,
    /tiny support should be treated exactly/i,
    /retraining is always the correct response/i,
    /recalibration fixes shuffled feature columns/i,
    /ablations are useless/i,
    /logs and diagnostics are optional/i,
    /holdout should be reused repeatedly/i,
    /changing many unrelated parts at once/i,
    /root cause should never change/i,
    /metric is sufficient even when it ignores/i,
    /stable averages.*remove the need/i,
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

test('model debugging misconception traps are placed after concept scaffolding', () => {
  const { quiz } = getLessonAssessment('model-debugging');
  const trapIds = [
    'dbg-076-trap-hyperparameter-first',
    'dbg-077-trap-aggregate',
    'dbg-078-trap-threshold',
    'dbg-079-trap-leakage',
    'dbg-080-trap-serving',
    'dbg-081-trap-support',
    'dbg-082-trap-retrain',
    'dbg-083-trap-calibration',
    'dbg-084-trap-ablation',
    'dbg-085-trap-observability',
    'dbg-086-trap-holdout',
    'dbg-087-trap-one-change',
    'dbg-088-trap-root-cause',
    'dbg-089-trap-evaluation',
    'dbg-090-tricky-summary',
  ];
  const misconceptionPatterns = [
    /start with random hyperparameter tuning/i,
    /stable aggregate score proves no subgroup/i,
    /global threshold tweak is always/i,
    /high validation always proves/i,
    /frozen model weights guarantee/i,
    /tiny support should be treated exactly/i,
    /retraining is always/i,
    /recalibration fixes shuffled feature columns/i,
    /ablations are useless/i,
    /diagnostics are optional/i,
    /holdout should be reused repeatedly/i,
    /changing many unrelated parts at once/i,
    /selected root cause should never change/i,
    /metric is sufficient/i,
    /remove the need for staged diagnosis/i,
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

test('model debugging assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('model-debugging');

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

test('model debugging assessment keeps visible wording scoped to debugging mechanics', () => {
  const { quiz } = getLessonAssessment('model-debugging');
  const genericLeakagePatterns = [
    /option letters/i,
    /route url/i,
    /decorative/i,
    /page title/i,
    /button label/i,
    /source file/i,
    /quiz answer position/i,
    /chart colors/i,
    /scenario buttons/i,
    /displayed scenario order/i,
    /alphabetized/i,
    /visual order/i,
    /dashboard theme/i,
  ];

  for (const question of quiz) {
    const visibleText = [question.prompt, ...question.choices, question.explanation].join(' ');
    for (const pattern of genericLeakagePatterns) {
      assert.doesNotMatch(visibleText, pattern, `${question.id} contains generic or off-scope visible wording`);
    }
  }
});

test('model debugging assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('model-debugging');
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
