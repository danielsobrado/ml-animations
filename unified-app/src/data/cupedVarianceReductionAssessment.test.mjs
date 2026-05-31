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

test('cuped variance reduction has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('cuped-variance-reduction');

  assert.equal(quiz.length, 100);
  assert.equal(labs.length, 3);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    const expectedNumber = String(index + 1).padStart(3, '0');

    assert.match(question.id, /^cuped-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use the cuped id format`);
    assert.equal(question.id.slice(6, 9), expectedNumber, `question ${index + 1} should preserve numeric order`);
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

test('cuped variance reduction assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('cuped-variance-reduction');
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

  const orderedMilestones = [
    [/main purpose of cuped/, 0, 7],
    [/what does cuped help reduce/, 0, 8],
    [/kind of covariate is safest/, 1, 9],
    [/pre period behavior useful/, 2, 10],
    [/post treatment covariates risky/, 3, 11],
    [/preserve when used correctly/, 4, 12],
    [/when cuped reduces variance/, 5, 13],
    [/confidence intervals/, 6, 14],
    [/pre post correlation matter/, 7, 15],
    [/1 rho squared/, 8, 16],
    [/adjustment coefficient/, 9, 17],
    [/adjusted outcome in cuped/, 10, 18],
    [/center the covariate/, 11, 19],
    [/sample equivalent precision/, 12, 20],
    [/affect experiment power/, 13, 20],
    [/replace randomization/, 14, 20],
    [/metric property makes cuped more useful/, 15, 20],
    [/little benefit/, 16, 20],
    [/planned for a primary decision/, 17, 20],
    [/first cuped validity check/, 18, 22],
    [/noise does cuped try to remove/, 20, 28],
    [/regression adjustment/, 20, 30],
    [/optimal adjustment coefficient/, 21, 31],
    [/rho rises/, 22, 32],
    [/adjusted standard error scale/, 23, 33],
    [/z score increase/, 24, 34],
    [/reduction in interval width/, 25, 35],
    [/preserve unbiasedness/, 26, 36],
    [/random imbalance/, 27, 37],
    [/subtracting the covariate mean/, 28, 38],
    [/treatment induced variation/, 29, 39],
    [/experiment population/, 30, 40],
    [/missing pre period covariates/, 31, 42],
    [/new users limit cuped gains/, 32, 43],
    [/beyond continuous metrics/, 33, 44],
    [/linear cuped adjustment assume/, 34, 45],
    [/multiple pre treatment covariates help/, 35, 46],
    [/overfitting the adjustment model/, 36, 47],
    [/cross fitting/, 37, 48],
    [/post treatment covariate bias cuped/, 38, 49],
    [/mediator dangerous/, 39, 50],
    [/collider like adjustment/, 40, 50],
    [/a a tests help a cuped rollout/, 41, 50],
    [/include cuped in the analysis plan/, 42, 50],
    [/guardrails automatically/, 43, 50],
    [/sample ratio mismatch/, 44, 50],
    [/cuped readout report/, 47, 50],
    [/cuped protocol/, 48, 52],
    [/80 percent correlation/, 50, 56],
    [/near zero correlation/, 50, 57],
    [/number of clicks after treatment exposure/, 51, 58],
    [/clicks in the week before assignment/, 52, 59],
    [/mostly in treatment/, 54, 62],
    [/raw analysis is not significant but cuped is significant/, 55, 63],
    [/tries 40 covariates/, 56, 64],
    [/sample ratio mismatch but cuped narrows intervals/, 59, 67],
    [/pre post correlation was high historically/, 60, 68],
    [/signups that happen after treatment/, 62, 70],
    [/2x sample equivalent gain/, 63, 71],
    [/effect is below the mde/, 64, 72],
    [/adjusted interval is narrow/, 65, 73],
    [/negatively correlated/, 66, 74],
    [/machine learning model for adjustment/, 67, 75],
    [/last year s users/, 68, 75],
    [/cuped experiment readout/, 69, 75],
    [/adjusted se is larger than raw se/, 70, 75],
    [/analysis plan names cuped/, 71, 75],
    [/pipeline fails before analysis/, 72, 75],
    [/guardrails are clean/, 73, 75],
    [/cuped claim is false/, 75, 82],
    [/covariate claim is unsafe/, 75, 83],
    [/effect claim is wrong/, 76, 84],
    [/correlation claim is misleading/, 77, 85],
    [/sample equivalent claim is false/, 78, 86],
    [/analysis claim is unsafe/, 79, 87],
    [/guardrail claim is wrong/, 80, 88],
    [/validity claim is false/, 81, 89],
    [/mediator claim is unsafe/, 82, 90],
    [/modeling claim is wrong/, 83, 90],
    [/a a claim is false/, 84, 90],
    [/missing data claim is unsafe/, 85, 90],
    [/launch claim is misleading/, 86, 90],
    [/causal claim is false/, 87, 90],
    [/reporting claim is unsafe/, 88, 90],
    [/define cuped in an interview/, 90, 96],
    [/formula intuition/, 90, 97],
    [/correlation to variance reduction/, 91, 98],
    [/how should you explain adjusted standard error/, 92, 99],
    [/cuped covariate valid/, 93, 100],
    [/biggest cuped trap/, 94, 100],
    [/diagnostics should accompany cuped/, 95, 100],
    [/complete cuped readout/, 96, 100],
    [/connect cuped to power planning/, 97, 100],
    [/interview ready cuped mastery/, 98, 100],
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

test('cuped variance reduction assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('cuped-variance-reduction');
  const unsafePatterns = [
    /cuped replaces the need for random assignment/i,
    /post-treatment covariates are always safe if they predict the outcome/i,
    /changes the true treatment effect/i,
    /any nonzero correlation guarantees/i,
    /new independent users were created/i,
    /try many covariates after launch and report only the most significant adjusted result/i,
    /cuped-powered primary metric means every guardrail is equally powered/i,
    /narrow adjusted intervals fix broken logging or assignment/i,
    /always improves total-effect/i,
    /most flexible adjustment model is always/i,
    /a\/a test with many false positives proves cuped is working/i,
    /missing covariates can always be silently dropped without consequences/i,
    /narrower interval automatically makes any effect worth shipping/i,
    /predictive covariate is valid for cuped even if treatment caused it/i,
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
