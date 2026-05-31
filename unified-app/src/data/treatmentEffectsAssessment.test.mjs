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

test('treatment effects has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('treatment-effects');

  assert.equal(quiz.length, 100);
  assert.equal(labs.length, 3);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    const expectedNumber = String(index + 1).padStart(3, '0');

    assert.match(question.id, /^te-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use the te id format`);
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

test('treatment effects assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('treatment-effects');
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

test('treatment effects assessment progresses from recall to interview readiness', () => {
  const { quiz } = getLessonAssessment('treatment-effects');
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

test('treatment effects assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('treatment-effects');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));

  const orderedMilestones = [
    [/what problem does treatment effect analysis address/, 0, 6],
    [/what is a unit in a treatment effect comparison/, 1, 7],
    [/what does treatment mean in this lesson/, 2, 8],
    [/what does the control condition represent/, 3, 9],
    [/what is the outcome in a treatment effect lesson/, 4, 10],
    [/what does y 1 refer to for a unit/, 5, 11],
    [/individual causal effect usually not directly observed/, 7, 13],
    [/what does ate summarize/, 8, 14],
    [/what does cate describe/, 9, 15],
    [/what does uplift emphasize/, 10, 16],
    [/what is treatment effect heterogeneity/, 11, 17],
    [/what does segment share affect/, 13, 19],
    [/positive ate fail to guarantee/, 14, 20],
    [/why consider targeting instead of treating everyone/, 15, 21],
    [/why define the estimand before analysis/, 18, 24],
    [/first treatment effect readout compare/, 19, 25],
    [/how do segment effects combine into an ate/, 20, 28],
    [/high response segment share increases/, 21, 29],
    [/segment effects have opposite signs/, 22, 30],
    [/large cate spread indicate/, 23, 31],
    [/targeting beat treating everyone/, 24, 32],
    [/subgroup estimates often noisier than ate/, 28, 36],
    [/why predefine important segments/, 30, 38],
    [/checking many segments require caution/, 31, 39],
    [/segment confidence intervals communicate/, 32, 40],
    [/what should an uplift model predict/, 34, 42],
    [/what is policy value in targeted treatment/, 35, 43],
    [/why include treatment cost in rollout decisions/, 36, 44],
    [/why does overlap matter within a segment/, 40, 48],
    [/what is an effect modifier/, 41, 49],
    [/why specify the target population/, 44, 50],
    [/why can treatment effects fail to transport/, 45, 50],
    [/why inspect guardrails by segment/, 46, 50],
    [/treatment effect protocol/, 49, 52],
    [/ate is positive but returning users are harmed/, 50, 58],
    [/high response users gain 18 low response users lose 4/, 51, 59],
    [/discount raises purchases but reduces margin/, 54, 62],
    [/tiny segment has huge observed lift/, 56, 64],
    [/slicing 200 ways/, 57, 65],
    [/planned segment cates and heterogeneity diagnostics/, 59, 67],
    [/segment mix differs between treated and control/, 60, 68],
    [/highest expected incremental benefit after costs/, 62, 70],
    [/high baseline users with low uplift/, 63, 71],
    [/protected group has negative cate/, 64, 72],
    [/almost no controls/, 68, 75],
    [/post treatment mediator rather than a pre treatment modifier/, 69, 75],
    [/targeted policy value against treat all and treat none/, 70, 75],
    [/positive ate and high heterogeneity/, 74, 75],
    [/which ate claim is false/, 75, 83],
    [/which uplift claim is wrong/, 76, 84],
    [/which subgroup claim is unsafe/, 77, 85],
    [/which analysis claim is misleading/, 78, 86],
    [/which rollout claim is false/, 79, 87],
    [/which targeting claim is wrong/, 80, 88],
    [/which segment claim is false/, 81, 89],
    [/which cate claim is unsafe/, 82, 90],
    [/which average effect claim is wrong/, 83, 90],
    [/which support claim is false/, 84, 90],
    [/which guardrail claim is false/, 87, 90],
    [/how should you explain ate versus cate in an interview/, 90, 96],
    [/how should you explain uplift targeting/, 91, 97],
    [/positive ate with subgroup harm/, 92, 98],
    [/noisy subgroup discoveries/, 93, 99],
    [/costs enter a treatment effect decision/, 94, 100],
    [/evaluate an uplift model/, 95, 100],
    [/assumptions should you mention for segment effects/, 96, 100],
    [/production ready treatment effect report/, 97, 100],
    [/interview ready mastery of treatment effects/, 98, 100],
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

test('treatment effects assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('treatment-effects');
  const unsafePatterns = [
    /positive ate means every segment benefits/i,
    /uplift is just the treated response rate/i,
    /always target the subgroup with the largest observed lift/i,
    /randomization makes unlimited subgroup mining safe/i,
    /any positive effect should be shipped regardless of cost/i,
    /target users with highest baseline outcome because they have highest uplift/i,
    /segments chosen after results are automatically confirmatory/i,
    /cate proves the best policy without uncertainty/i,
    /ate is always enough for rollout decisions/i,
    /reliable with no comparable controls/i,
    /post-treatment behavior is always a safe targeting segment/i,
    /experiment ate always transfers unchanged to any population/i,
    /average benefit lets you ignore harmed important groups/i,
    /policy value is just the largest cate/i,
    /one noisy subgroup win proves stable treatment-effect heterogeneity/i,
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

test('treatment effects assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('treatment-effects');
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

test('treatment effects assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('treatment-effects');
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
