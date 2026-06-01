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

test('model interpretability has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('model-interpretability');

  assert.equal(quiz.length, 100);
  assert.deepEqual(labs.map((lab) => lab.id), ['compare-modes']);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);
  assert.deepEqual(
    Object.fromEntries(Object.keys(EXPECTED_LEVEL_COUNTS).map((level) => [
      level,
      quiz.filter((question) => question.level === level).length,
    ])),
    EXPECTED_LEVEL_COUNTS,
  );

  for (const [index, question] of quiz.entries()) {
    assert.match(question.id, /^interp-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use an ordered interp id`);
    assert.equal(Number(question.id.slice(7, 10)), index + 1, `question ${index + 1} id should match its position`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.equal(new Set(question.choices.map((choice) => normalized(choice))).size, 3, `question ${index + 1} choices should be distinct`);
    assert.equal(Number.isInteger(question.answerIndex), true, `question ${index + 1} answer index should be an integer`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(Object.hasOwn(LEVEL_ORDER, question.level), `question ${index + 1} should have a recognized level`);
  }
});

test('model interpretability assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('model-interpretability');
  const prompts = quiz.map((question) => normalized(question.prompt));
  const answers = quiz.map((question) => normalized(correctAnswer(question)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('model interpretability assessment progresses from explanation basics to interview readiness', () => {
  const { quiz } = getLessonAssessment('model-interpretability');
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

test('model interpretability assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('model-interpretability');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));
  const firstIndexContaining = (terms) => textByQuestion.findIndex((text) => terms.every((term) => text.includes(term)));
  const orderedMilestones = [
    ['purpose', ['which inputs or patterns are influencing']],
    ['global local counterfactual views', ['global importance local attribution and counterfactual']],
    ['ablation baseline', ['prediction degradation when a feature is replaced']],
    ['correlation caveat', ['correlated features can make']],
    ['global mechanism', ['replace each feature by a baseline']],
    ['local contribution mechanism', ['weight multiplies the selected feature difference']],
    ['counterfactual mechanism', ['one selected feature value']],
    ['multiple lenses', ['agreement and disagreement between lenses']],
    ['application method fit', ['stakeholder asks what the model generally relies on']],
    ['tricky false claims', ['claim is false']],
    ['interview readiness', ['feature influence examples and sensitivity']],
  ];

  let previousIndex = -1;
  for (const [label, terms] of orderedMilestones) {
    const index = firstIndexContaining(terms);
    assert.notEqual(index, -1, `missing milestone: ${label}`);
    assert.ok(index > previousIndex, `${label} should appear after the previous milestone`);
    previousIndex = index;
  }
});

test('model interpretability assessment avoids unsafe misconception keying before explicit traps', () => {
  const { quiz } = getLessonAssessment('model-interpretability');
  const unsafePatterns = [
    /proves the feature caused/i,
    /top global feature must be the top driver/i,
    /any feature perturbation is actionable/i,
    /make attribution perfectly stable/i,
    /automatically fair/i,
    /independent of the reference baseline/i,
    /accuracy guarantees trustworthy explanations/i,
    /exact causal mechanism/i,
    /confident prediction explains itself/i,
    /complete for every model/i,
    /scaling is irrelevant/i,
    /remove the need to evaluate/i,
    /positive contribution always means/i,
    /safest action is to ignore assumptions/i,
    /enough to prove causality/i,
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

test('model interpretability misconception traps are placed after concept scaffolding', () => {
  const { quiz } = getLessonAssessment('model-interpretability');
  const trapIds = [
    'interp-076-trap-causal-proof',
    'interp-077-trap-global-local',
    'interp-078-trap-counterfactual',
    'interp-079-trap-correlation',
    'interp-080-trap-sparse',
    'interp-081-trap-baseline',
    'interp-082-trap-accuracy',
    'interp-083-trap-ablation',
    'interp-084-trap-confidence',
    'interp-085-trap-method-universal',
    'interp-086-trap-normalization',
    'interp-087-trap-validation',
    'interp-088-trap-sign',
    'interp-089-trap-stability',
    'interp-090-tricky-summary',
  ];
  const misconceptionPatterns = [
    /proves the feature caused/i,
    /top global feature must be/i,
    /any feature perturbation is actionable/i,
    /perfectly stable/i,
    /automatically fair/i,
    /independent of the reference baseline/i,
    /accuracy guarantees/i,
    /exact causal mechanism/i,
    /confident prediction explains itself/i,
    /complete for every model/i,
    /scaling is irrelevant/i,
    /remove the need to evaluate/i,
    /positive contribution always means/i,
    /ignore assumptions/i,
    /prove causality fairness and deployment/i,
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

test('model interpretability assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('model-interpretability');

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

test('model interpretability assessment keeps visible wording scoped to explanation mechanics', () => {
  const { quiz } = getLessonAssessment('model-interpretability');
  const genericLeakagePatterns = [
    /page title/i,
    /display label/i,
    /sample id has the longest/i,
    /feature labels are capitalized/i,
    /technical terms/i,
    /colorful attribution/i,
    /display formatting/i,
    /reporting tools/i,
    /feature name has enough characters/i,
    /sorted alphabetically/i,
  ];

  for (const question of quiz) {
    const visibleText = [question.prompt, ...question.choices, question.explanation].join(' ');
    for (const pattern of genericLeakagePatterns) {
      assert.doesNotMatch(visibleText, pattern, `${question.id} contains generic or off-scope visible wording`);
    }
  }
});

test('model interpretability assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('model-interpretability');
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
