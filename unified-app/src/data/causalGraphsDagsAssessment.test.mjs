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
const LEVELS = Object.keys(LEVEL_ORDER);

function normalized(value) {
  return String(value || '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
}

function correctAnswer(question) {
  return question.choices[question.answerIndex];
}

test('causal graphs dags has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('causal-graphs-dags');

  assert.equal(quiz.length, 100);
  assert.equal(labs.length, 3);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    const expectedNumber = String(index + 1).padStart(3, '0');

    assert.match(question.id, /^dag-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use the dag id format`);
    assert.equal(question.id.slice(4, 7), expectedNumber, `question ${index + 1} should preserve numeric order`);
    assert.ok(!question.id.includes('generated-'), `question ${index + 1} should not use a generated id`);
    assert.ok(question.prompt.length > 20, `question ${index + 1} prompt should be substantive`);
    assert.equal(question.choices.length, 3, `question ${index + 1} should have three choices`);
    assert.equal(new Set(question.choices.map(normalized)).size, 3, `question ${index + 1} should have unique choices`);
    assert.ok(Number.isInteger(question.answerIndex), `question ${index + 1} answer index should be an integer`);
    assert.ok(question.answerIndex >= 0 && question.answerIndex < question.choices.length, `question ${index + 1} answer index should be valid`);
    assert.ok(question.explanation.length > 30, `question ${index + 1} explanation should teach the point`);
    assert.ok(LEVELS.includes(question.level), `question ${index + 1} should have a recognized level`);
  }
});

test('causal graphs dags assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('causal-graphs-dags');
  const prompts = new Map();
  const correctAnswers = new Map();

  for (const question of quiz) {
    const prompt = normalized(question.prompt);
    const answer = normalized(correctAnswer(question));

    assert.ok(!prompts.has(prompt), `${question.id} duplicates prompt from ${prompts.get(prompt)}`);
    prompts.set(prompt, question.id);

    assert.ok(!correctAnswers.has(answer), `${question.id} duplicates correct answer from ${correctAnswers.get(answer)}`);
    correctAnswers.set(answer, question.id);
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

  const orderedMilestones = [
    [/main purpose of a causal dag/, 0, 7],
    [/directed arrow/, 0, 8],
    [/acyclic/, 1, 9],
    [/treatment node/, 2, 10],
    [/outcome node/, 3, 11],
    [/causal path from treatment to outcome/, 4, 12],
    [/backdoor path/, 5, 13],
    [/confounder in dag language/, 6, 14],
    [/valid confounder adjustment/, 7, 15],
    [/collider node/, 8, 16],
    [/condition on a collider/, 9, 17],
    [/mediator/, 10, 18],
    [/estimating total effect/, 11, 19],
    [/direct effect/, 12, 20],
    [/adjustment set/, 13, 20],
    [/domain assumptions/, 14, 20],
    [/drawing it proves/, 15, 20],
    [/what is an open path/, 16, 20],
    [/what is a closed path/, 17, 20],
    [/first dag adjustment check/, 18, 22],
    [/c t and c y/, 20, 28],
    [/true pre treatment confounder/, 20, 30],
    [/t s u/, 21, 31],
    [/conditioning on s/, 22, 32],
    [/descendant of a collider/, 23, 33],
    [/t m y/, 24, 34],
    [/adjust for m/, 25, 35],
    [/frontdoor paths/, 26, 36],
    [/d separation/, 27, 37],
    [/fork structure/, 28, 38],
    [/chain structure/, 29, 39],
    [/collider differ from a fork/, 30, 40],
    [/good control/, 31, 41],
    [/bad control/, 32, 42],
    [/minimal sufficient adjustment set/, 33, 43],
    [/empty adjustment set/, 34, 44],
    [/m bias/, 35, 45],
    [/selection into the sample/, 36, 46],
    [/measurement timing/, 37, 47],
    [/unmeasured common cause/, 38, 48],
    [/proxy variable/, 39, 49],
    [/positivity still matter/, 40, 50],
    [/overadjustment in dag terms/, 41, 50],
    [/descendant of a confounder/, 42, 50],
    [/effect target/, 43, 50],
    [/lesson ui emphasize/, 44, 50],
    [/qualitative bias score/, 45, 50],
    [/audit before trusting a dag/, 46, 50],
    [/dag based analysis report/, 47, 50],
    [/dag adjustment protocol/, 48, 52],
    [/age affects treatment choice/, 50, 57],
    [/hospital admission/, 50, 58],
    [/engagement increases retention/, 51, 59],
    [/post feature engagement/, 52, 60],
    [/direct effect not through engagement/, 53, 61],
    [/prior purchase intent/, 54, 62],
    [/clicked a notification/, 55, 63],
    [/hidden health score/, 56, 64],
    [/diagnosis codes proxy/, 57, 65],
    [/all high severity patients/, 58, 66],
    [/customer satisfaction measured after treatment/, 59, 67],
    [/pre treatment risk score/, 60, 68],
    [/fraud review flag/, 61, 69],
    [/randomized experiment/, 62, 70],
    [/pre treatment covariate/, 63, 71],
    [/survey response variable/, 64, 72],
    [/experts disagree/, 65, 73],
    [/total ate/, 66, 74],
    [/regression includes all variables/, 67, 75],
    [/adjusted effect but no graph/, 68, 75],
    [/frontdoor identification/, 69, 75],
    [/hidden confounder might exist/, 70, 75],
    [/policy affects screening/, 71, 75],
    [/strong backdoor left/, 72, 75],
    [/high collider conditioning/, 73, 75],
    [/dag claim is false/, 75, 82],
    [/control claim is unsafe/, 75, 83],
    [/collider claim is wrong/, 76, 84],
    [/confounder claim is false/, 77, 85],
    [/mediator claim is unsafe/, 78, 86],
    [/empty set claim is false/, 79, 87],
    [/proxy claim is misleading/, 80, 88],
    [/support claim is wrong/, 81, 89],
    [/descendant claim is unsafe/, 82, 90],
    [/p value claim is false/, 83, 90],
    [/domain knowledge claim is wrong/, 84, 90],
    [/direct effect claim is unsafe/, 85, 90],
    [/selection claim is false/, 86, 90],
    [/reporting claim is unsafe/, 87, 90],
    [/proof claim is false/, 88, 90],
    [/define a causal dag in an interview/, 90, 96],
    [/explain backdoor adjustment/, 90, 97],
    [/explain collider bias/, 91, 98],
    [/explain mediator adjustment/, 92, 99],
    [/choose an adjustment set/, 93, 100],
    [/caveat should you give/, 94, 100],
    [/diagnostics should accompany dag adjustment/, 95, 100],
    [/dag based readout include/, 96, 100],
    [/compact example demonstrates dag mastery/, 97, 100],
    [/interview ready mastery of causal graphs/, 98, 100],
  ];

  for (const [pattern, minIndex, maxIndex] of orderedMilestones) {
    const index = textByQuestion.findIndex((text) => pattern.test(text));
    assert.notEqual(index, -1, `missing milestone: ${pattern}`);
    assert.ok(
      index >= minIndex && index < maxIndex,
      `${pattern} should appear in questions ${minIndex + 1}-${maxIndex}, found question ${index + 1}`,
    );
  }
});

test('causal graphs dags assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('causal-graphs-dags');
  const unsafePatterns = [
    /dag proves causality because the arrows were drawn/i,
    /adjust for every available variable to be safe/i,
    /conditioning on a collider always removes bias/i,
    /confounder is caused by treatment/i,
    /adjusting for a mediator is always correct for total effects/i,
    /empty adjustment set is never valid/i,
    /any proxy fully solves unmeasured confounding/i,
    /correct dag removes the need for overlap in the data/i,
    /conditioning on descendants of colliders is always harmless/i,
    /small adjusted p-value proves the adjustment set was valid/i,
    /data alone always determines the correct dag uniquely/i,
    /direct and total effects always use the same adjustment set/i,
    /analyzing only selected units can never create bias/i,
    /only the regression coefficient is needed/i,
    /dag-adjusted estimate proves the exact causal effect without assumptions/i,
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
  const totals = [0, 0, 0];

  for (let pageStart = 0; pageStart < quiz.length; pageStart += 10) {
    const page = quiz.slice(pageStart, pageStart + 10);
    const positions = page.map((question) => question.answerIndex);

    assert.ok(new Set(positions).size >= 2, `page starting at question ${pageStart + 1} should vary answer positions`);
    for (const position of positions) {
      totals[position] += 1;
    }
  }

  assert.ok(Math.max(...totals) - Math.min(...totals) <= 1, `answer positions should be balanced, found ${totals.join('/')}`);
});
