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

test('spearman correlation has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('spearman-correlation');

  assert.equal(quiz.length, 100);
  assert.equal(labs.length, 3);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    const expectedNumber = String(index + 1).padStart(3, '0');

    assert.match(question.id, /^sp-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use the sp id format`);
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

test('spearman correlation assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('spearman-correlation');
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

test('spearman correlation assessment progresses from recall to interview readiness', () => {
  const { quiz } = getLessonAssessment('spearman-correlation');
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

test('spearman correlation assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('spearman-correlation');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));

  const orderedMilestones = [
    [/what does spearman correlation measure/, 0, 6],
    [/what does spearman compare before computing association/, 1, 7],
    [/rank 1 usually mean/, 2, 8],
    [/what symbol is commonly used/, 3, 9],
    [/what kind of relationship can spearman capture well/, 4, 10],
    [/positive spearman value suggest/, 5, 11],
    [/negative spearman value suggest/, 6, 12],
    [/spearman value near zero/, 7, 13],
    [/how does pearson differ from spearman/, 8, 14],
    [/raw numeric gaps matter less/, 9, 15],
    [/less sensitive to an extreme outlier/, 10, 16],
    [/simple no ties formula use/, 11, 17],
    [/what is d in the calculation lab/, 12, 18],
    [/why square each rank difference/, 13, 19],
    [/sum d squared summarize/, 14, 20],
    [/exactly 1 in the simple setting/, 15, 20],
    [/exactly negative one in the simple setting/, 16, 20],
    [/what are ties in spearman correlation/, 17, 20],
    [/not be treated as/, 18, 20],
    [/inspect before trusting spearman/, 19, 22],
    [/first calculation step for spearman/, 20, 28],
    [/why sort by x/, 21, 29],
    [/why sort by y/, 22, 30],
    [/why return to each item/, 23, 31],
    [/single swapped rank/, 24, 32],
    [/curved increasing relationship/, 25, 33],
    [/u shaped pattern/, 26, 34],
    [/unit changes/, 27, 35],
    [/outlier remains largest/, 28, 36],
    [/outlier still hurt spearman strongly/, 29, 37],
    [/ties commonly handled/, 30, 38],
    [/many ties require caution/, 31, 39],
    [/ordinal data/, 32, 40],
    [/sample size matter/, 33, 41],
    [/spearman p value test/, 34, 42],
    [/report uncertainty with spearman/, 35, 43],
    [/missing values be handled/, 36, 44],
    [/duplicate observations matter/, 37, 45],
    [/why still draw a scatterplot/, 38, 46],
    [/compare pearson and spearman/, 39, 47],
    [/strictly increasing transform/, 40, 48],
    [/strictly decreasing transform/, 41, 49],
    [/what does spearman not communicate well/, 42, 50],
    [/simple 1 minus formula most direct/, 44, 50],
    [/equivalent way to compute spearman/, 45, 50],
    [/diagnostics belong with spearman/, 48, 50],
    [/spearman analysis protocol/, 49, 52],
    [/x increases and y equals x cubed/, 50, 58],
    [/u shaped dependence/, 51, 59],
    [/largest x point gets an enormous y/, 52, 60],
    [/largest y to smallest y/, 53, 61],
    [/five star rating variables/, 54, 62],
    [/two orderings move together/, 55, 63],
    [/negative rank correlation/, 56, 64],
    [/monotonic transform changes feature scale/, 57, 65],
    [/spearman is near one on a curved rise/, 58, 66],
    [/plot is circular/, 59, 67],
    [/only five points are available/, 60, 68],
    [/duplicate customer rows/, 61, 69],
    [/average of the tied rank positions/, 62, 70],
    [/outlier slider changes raw y/, 63, 71],
    [/ice cream rank and swimming rank/, 64, 72],
    [/missing y/, 65, 73],
    [/human relevance order to model order/, 66, 74],
    [/jumps from low to high/, 67, 75],
    [/x ranks 1 2 3 4 while y ranks 4 3 2 1/, 68, 75],
    [/rank x 4 and rank y 2/, 73, 75],
    [/high spearman and moderate pearson/, 74, 75],
    [/which spearman claim is false/, 75, 83],
    [/which interpretation claim is wrong/, 76, 84],
    [/which robustness claim is unsafe/, 77, 85],
    [/which near zero claim is misleading/, 78, 86],
    [/which ties claim is false/, 79, 87],
    [/which raw value claim is wrong/, 80, 88],
    [/which transform claim is false/, 81, 89],
    [/which formula claim is unsafe/, 82, 90],
    [/which reporting claim is unsafe/, 88, 90],
    [/which perfect correlation claim is false/, 89, 90],
    [/define spearman in an interview/, 90, 96],
    [/contrast spearman with pearson/, 91, 97],
    [/when would you choose spearman/, 92, 98],
    [/calculation steps/, 93, 99],
    [/outlier robustness/, 94, 100],
    [/what should you say about tied values/, 95, 100],
    [/limitations should you name/, 96, 100],
    [/diagnostics should accompany spearman/, 97, 100],
    [/example shows spearman beating pearson/, 98, 100],
    [/interview ready mastery of spearman correlation/, 99, 100],
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

test('spearman correlation assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('spearman-correlation');
  const unsafePatterns = [
    /spearman measures only straight-line association/i,
    /high spearman proves one variable causes the other/i,
    /spearman is completely immune to outliers/i,
    /near-zero spearman proves no relationship of any kind/i,
    /ties can always be ignored without affecting interpretation/i,
    /spearman preserves the size of raw numeric gaps/i,
    /any transformation leaves spearman unchanged/i,
    /simple d-squared formula handles ties with no caveats/i,
    /significant spearman p-value proves practical importance/i,
    /sort x once and reuse that order as rank y/i,
    /ordinal ranks guarantee equal distances between categories/i,
    /spearman value from tiny n is automatically stable/i,
    /pearson and spearman must always give the same result/i,
    /only rho is needed/i,
    /spearman of one proves the raw relationship is a straight line/i,
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

test('spearman correlation assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('spearman-correlation');
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

test('spearman correlation assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('spearman-correlation');
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
