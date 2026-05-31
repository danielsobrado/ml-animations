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

test('sequential testing peeking has a complete curated 100-question assessment', () => {
  const { quiz, labs } = getLessonAssessment('sequential-testing-peeking');

  assert.equal(quiz.length, 100);
  assert.equal(labs.length, 3);
  assert.equal(new Set(quiz.map((question) => question.id)).size, 100);

  for (const [index, question] of quiz.entries()) {
    const expectedNumber = String(index + 1).padStart(3, '0');

    assert.match(question.id, /^seq-\d{3}-[a-z0-9-]+$/, `question ${index + 1} should use the seq id format`);
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

test('sequential testing peeking assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('sequential-testing-peeking');
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

test('sequential testing peeking assessment progresses from recall to interview readiness', () => {
  const { quiz } = getLessonAssessment('sequential-testing-peeking');
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

test('sequential testing peeking assessment covers learning points in the right order', () => {
  const { quiz } = getLessonAssessment('sequential-testing-peeking');
  const textByQuestion = quiz.map((question) => normalized(`${question.prompt} ${correctAnswer(question)} ${question.explanation}`));

  const orderedMilestones = [
    [/problem does sequential testing solve/, 0, 8],
    [/fixed horizon experiment/, 0, 8],
    [/peeking in an experiment/, 1, 9],
    [/naive peeking risky/, 2, 10],
    [/alpha mean in a fixed horizon test/, 3, 11],
    [/any look false positive risk/, 4, 12],
    [/what is a stopping rule/, 5, 13],
    [/what is an interim look/, 6, 14],
    [/alpha spending/, 7, 15],
    [/sequential stopping boundary/, 8, 16],
    [/futility stopping/, 10, 18],
    [/information time/, 11, 19],
    [/monitoring plan/, 12, 20],
    [/primary metric for monitoring/, 13, 20],
    [/fixed horizon and sequential testing/, 15, 20],
    [/first check for peeking risk/, 18, 22],
    [/repeated looks find a small p value under the null/, 20, 28],
    [/1 1 alpha looks approximate/, 20, 30],
    [/many naive looks/, 21, 31],
    [/early sequential boundaries usually stricter/, 23, 33],
    [/total alpha budget/, 25, 35],
    [/alpha spending function/, 26, 36],
    [/o brien fleming style boundaries/, 27, 38],
    [/pocock style boundaries/, 28, 39],
    [/information fraction/, 29, 40],
    [/sequential testing affect power/, 30, 42],
    [/maximum sample size rise/, 32, 43],
    [/conditional power/, 34, 45],
    [/early stopping affect effect estimates/, 35, 46],
    [/reports include after sequential stopping/, 36, 47],
    [/ordinary confidence intervals/, 37, 48],
    [/monitoring many metrics/, 38, 49],
    [/operational safety alerts/, 41, 50],
    [/preregister sequential rules/, 46, 50],
    [/sequential testing protocol/, 48, 52],
    [/checks p 0 05 every day/, 50, 57],
    [/dashboard is visible/, 50, 58],
    [/assignment is 70 30/, 51, 59],
    [/severe latency harm/, 52, 60],
    [/p 0 04 but the sequential boundary/, 53, 61],
    [/30 metrics daily/, 55, 64],
    [/subgroup crosses p 0 05/, 56, 65],
    [/futility rule says stop/, 57, 66],
    [/very early look shows a huge lift/, 59, 68],
    [/adds 5 more after seeing noisy results/, 60, 69],
    [/same stricter threshold at every look/, 62, 70],
    [/weekly looks with the same maximum n/, 63, 71],
    [/p was below 0 05 at some point/, 65, 73],
    [/looked at interim charts but no one could see treatment labels/, 71, 75],
    [/live monitoring for every launch/, 72, 75],
    [/alpha claim is false/, 75, 82],
    [/peeking claim is unsafe/, 75, 83],
    [/boundary claim is wrong/, 76, 84],
    [/dashboard claim is misleading/, 77, 85],
    [/final look claim is unsafe/, 79, 87],
    [/small p claim is misleading/, 87, 90],
    [/define peeking in an interview/, 90, 96],
    [/what makes a sequential design valid/, 91, 98],
    [/reported after a sequential stop/, 94, 100],
    [/interview ready sequential testing mastery/, 97, 100],
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

test('sequential testing peeking assessment avoids unsafe misconception keying', () => {
  const { quiz } = getLessonAssessment('sequential-testing-peeking');
  const unsafePatterns = [
    /using p < 0\.05 at every interim look preserves a 5 percent total false positive rate/i,
    /peeking is harmless whenever the final sample size is unchanged/i,
    /interim boundaries can be chosen after seeing the interim p-values/i,
    /a live dashboard automatically gives valid sequential testing/i,
    /sequential testing always increases power for free/i,
    /the final look can always be judged by ordinary p < 0\.05 regardless of earlier looks/i,
    /a futility stop proves the treatment has exactly zero effect/i,
    /stopping early for benefit makes the observed effect estimate unbiased by default/i,
    /monitoring many metrics does not change false positive risk if each uses 0\.05/i,
    /any interim subgroup p < 0\.05 is confirmatory launch evidence/i,
    /checking assignment logs is the same as repeatedly testing treatment efficacy/i,
    /teams should ignore severe treatment harm until the final efficacy look/i,
    /adding unplanned looks after launch keeps the original sequential guarantee intact/i,
    /the smallest p-value seen during monitoring is enough to report as final/i,
    /a valid sequential stop proves the exact future product impact/i,
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

test('sequential testing peeking assessment does not leak exact answers within a visible page', () => {
  const { quiz } = getLessonAssessment('sequential-testing-peeking');
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

test('sequential testing peeking assessment distributes correct-answer positions across every page', () => {
  const { quiz } = getLessonAssessment('sequential-testing-peeking');
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
