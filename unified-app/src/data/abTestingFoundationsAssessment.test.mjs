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

test('ab testing foundations has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('ab-testing-foundations');

  assert.equal(quiz.length, 100);
  assert.equal(labs.length, 3);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    const expectedNumber = String(index + 1).padStart(3, '0');

    assert.match(question.id, /^ab-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use the ab id format`);
    assert.equal(question.id.slice(3, 6), expectedNumber, `question ${index + 1} should preserve numeric order`);
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

test('ab testing foundations assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('ab-testing-foundations');
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

test('ab testing foundations assessment progresses from recall to interview readiness', () => {
  const { quiz } = getLessonAssessment('ab-testing-foundations');
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

test('ab testing foundations assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('ab-testing-foundations');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));

  const orderedMilestones = [
    [/main purpose of an a b test/, 0, 8],
    [/control group/, 0, 8],
    [/treatment group/, 1, 8],
    [/randomization central/, 2, 10],
    [/assignment unit/, 3, 12],
    [/analysis unit/, 4, 12],
    [/primary metric/, 5, 14],
    [/guardrail metric/, 6, 15],
    [/treatment effect estimate/, 7, 16],
    [/confidence interval/, 11, 20],
    [/practical significance/, 12, 20],
    [/minimum detectable effect/, 13, 22],
    [/power mean/, 14, 22],
    [/alpha control/, 15, 23],
    [/first pass a b decision/, 17, 24],
    [/what counterfactual/, 20, 28],
    [/balance check/, 20, 30],
    [/sample ratio mismatch/, 21, 31],
    [/intent to treat/, 23, 33],
    [/unit mismatch dangerous/, 25, 35],
    [/interference between units/, 26, 36],
    [/cluster randomization/, 27, 38],
    [/p value only reporting weak/, 30, 40],
    [/choose mde before running/, 33, 42],
    [/unplanned peeking/, 38, 46],
    [/many tested outcomes/, 39, 48],
    [/subgroup findings carefully/, 40, 49],
    [/experiment logging/, 44, 50],
    [/a b test protocol/, 48, 52],
    [/dashboard shows treated users/, 50, 56],
    [/planned 50 50 allocation/, 50, 58],
    [/primary metric improves significantly/, 51, 60],
    [/huge test finds a 0 02 percent/, 52, 61],
    [/checked p values daily/, 55, 63],
    [/one of 80 secondary metrics/, 56, 65],
    [/users are randomized but analysis treats every page view/, 58, 67],
    [/latency guardrails degrade/, 66, 73],
    [/outcome logging fails more often/, 67, 74],
    [/positive lift interval above mde/, 69, 75],
    [/a b testing claim is false/, 75, 82],
    [/randomization claim is false/, 75, 83],
    [/monitoring claim is false/, 79, 87],
    [/metric shopping claim is unsafe/, 80, 88],
    [/sample ratio claim is unsafe/, 83, 90],
    [/define an a b test in an interview/, 90, 96],
    [/debug a suspicious a b result/, 95, 100],
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

test('ab testing foundations assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('ab-testing-foundations');
  const unsafePatterns = [
    /randomization guarantees treatment and control are identical in every realized sample/i,
    /a statistically significant lift is automatically worth launching/i,
    /a non-significant underpowered test proves the feature has no effect/i,
    /guardrails can be ignored when the primary metric wins/i,
    /stopping the first time p < 0\.05 preserves the original fixed-horizon alpha/i,
    /after many metrics, report only the significant winner as confirmatory/i,
    /session-level rows are always independent when users were randomized/i,
    /filtering to exposed users is always unbiased no matter how exposure occurs/i,
    /severe sample ratio mismatch can be ignored if treatment looks better/i,
    /a simple user-randomized test is always valid when users affect each other/i,
    /mde should be chosen after results to match the observed lift/i,
    /running only until significance appears is a valid fixed-duration plan/i,
    /remove inconvenient users after seeing treatment lose among them/i,
    /a significant observational rollout comparison has the same causal strength as randomized assignment/i,
  ];

  for (const [index, question] of quiz.entries()) {
    const answer = correctAnswer(question);
    const unsafeAnswer = unsafePatterns.some((pattern) => pattern.test(answer));
    const explicitTrapPrompt = /trap|false|misleading|unsafe|too strong|wrong|claim/i.test(question.prompt);

    assert.ok(
      !unsafeAnswer || explicitTrapPrompt,
      `question ${index + 1} keys a false claim outside an explicit trap prompt`,
    );
  }
});

test('ab testing foundations assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('ab-testing-foundations');
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

test('ab testing foundations assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('ab-testing-foundations');
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
