import assert from 'node:assert/strict';
import test from 'node:test';
import { getLessonAssessment } from './lessonAssessments.js';

const LEVELS = new Set(['Foundation', 'Mechanism', 'Application', 'Tricky', 'Interview']);

function normalize(value) {
  return String(value).toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
}

function correctAnswer(item) {
  return item.choices[item.answerIndex];
}

test('frontier evaluation safety assessment has 100 production-ready questions with focused labs', () => {
  const { quiz, labs } = getLessonAssessment('frontier-evaluation-safety');

  assert.equal(labs.length, 4);
  assert.deepEqual(labs.map((lab) => lab.id), [
    'capability-product-layer',
    'prompt-injection-defense',
    'reward-hacking-simulator',
    'deployment-gate-lab',
  ]);

  assert.equal(quiz.length, 100);
  const ids = new Set();
  const globalCounts = [0, 0, 0];

  for (const [index, item] of quiz.entries()) {
    assert.match(item.id, /^frontier-\d{3}$/);
    assert.equal(ids.has(item.id), false, `duplicate id ${item.id}`);
    ids.add(item.id);
    assert.equal(item.id, `frontier-${String(index + 1).padStart(3, '0')}`);
    assert.equal(LEVELS.has(item.level), true, `${item.id} has unexpected level ${item.level}`);
    assert.equal(item.choices.length, 3, `${item.id} should have three choices`);
    assert.equal(Number.isInteger(item.answerIndex), true, `${item.id} answerIndex should be an integer`);
    assert.ok(item.answerIndex >= 0 && item.answerIndex < 3, `${item.id} answerIndex out of range`);
    assert.ok(item.prompt.length > 20, `${item.id} prompt too short`);
    assert.ok(item.explanation.length > 30, `${item.id} explanation too short`);
    assert.equal(new Set(item.choices.map(normalize)).size, 3, `${item.id} has duplicate choices`);
    globalCounts[item.answerIndex] += 1;
  }

  assert.ok(
    Math.max(...globalCounts) - Math.min(...globalCounts) <= 1,
    `answer positions should be globally balanced, got ${globalCounts.join(', ')}`,
  );
});

test('frontier evaluation safety assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('frontier-evaluation-safety');
  const prompts = quiz.map((item) => normalize(item.prompt));
  const correctAnswers = quiz.map((item) => normalize(correctAnswer(item)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(correctAnswers).size, correctAnswers.length);
});

test('frontier evaluation safety assessment progresses through layered safety evidence', () => {
  const { quiz } = getLessonAssessment('frontier-evaluation-safety');
  const ranges = [
    ['Foundation', 0, 20],
    ['Mechanism', 20, 50],
    ['Application', 50, 75],
    ['Tricky', 75, 90],
    ['Interview', 90, 100],
  ];

  for (const [level, start, end] of ranges) {
    assert.equal(quiz.slice(start, end).every((item) => item.level === level), true, `${level} range mismatch`);
  }

  const milestones = [
    ['evaluation layers', ['capability evaluation', 'perform a task']],
    ['agent benchmarks', ['SWE-bench-style', 'real repository issues']],
    ['prompt injection safety', ['source trust labels', 'instruction hierarchy']],
    ['deployment gates', ['deployment scorecard', 'rollback readiness']],
    ['scenario practice', ['support workflow', 'tools']],
    ['production takeaway', ['production-ready takeaway', 'staged gates']],
  ];

  let previous = -1;
  for (const [name, terms] of milestones) {
    const index = quiz.findIndex((item) => (
      terms.every((term) => normalize(`${item.prompt} ${item.choices.join(' ')} ${item.explanation}`).includes(normalize(term)))
    ));
    assert.notEqual(index, -1, `missing milestone: ${name}`);
    assert.ok(index > previous, `${name} appears out of order`);
    previous = index;
  }
});

test('frontier evaluation safety assessment marks unsafe misconceptions as traps after setup', () => {
  const { quiz } = getLessonAssessment('frontier-evaluation-safety');
  const misconceptionTerms = [
    /benchmark score is by itself a deployment decision/i,
    /automatically test product workflows/i,
    /policy violations do not matter/i,
    /Maximizing blocking alone/i,
    /webpage instructions should override/i,
    /same approval policy/i,
    /one needle proves/i,
    /without describing tools/i,
    /proxy metric cannot be gamed/i,
    /explanation is enough/i,
    /Oversight-sensitive behavior is irrelevant/i,
    /refuses every request/i,
    /single fixed score/i,
    /hide limitations/i,
    /preparedness unnecessary/i,
  ];
  const trapPrompt = /false|wrong|unsafe|trap|reject|dangerous/i;

  for (const [index, item] of quiz.entries()) {
    const text = `${item.prompt} ${item.choices.join(' ')}`;
    const containsMisconception = misconceptionTerms.some((pattern) => pattern.test(text));
    if (!containsMisconception) continue;

    assert.ok(index >= 75, `${item.id} introduces misconception too early`);
    assert.match(item.prompt, trapPrompt, `${item.id} should mark misconception as a trap`);
  }
});

test('frontier evaluation safety assessment stays within visible lesson scope', () => {
  const { quiz } = getLessonAssessment('frontier-evaluation-safety');
  const unrelatedScopeLeaks = [
    /\btokenizer vocabulary\b/i,
    /\bmodel-weight compression\b/i,
    /\braw arithmetic exam\b/i,
    /\bUI color contrast\b/i,
    /\bspelling quality\b/i,
    /\bmodel parameters per layer\b/i,
    /\bbenchmark examples memorized\b/i,
    /\bhidden layers\b/i,
    /\bUI panel\b/i,
    /\btokenizer switch\b/i,
    /\bhidden training batch\b/i,
    /\bbrowser cache\b/i,
    /\bfont size\b/i,
    /\bclassify images\b/i,
    /\blongest passage\b/i,
    /\btokenizer compression\b/i,
    /\bimage captioning\b/i,
    /\bfirst token after a prompt\b/i,
    /\bfinal UI styling\b/i,
    /\baverage response length\b/i,
    /\bnumber of UI panels\b/i,
    /\bfont rendering\b/i,
    /\brhyme quality\b/i,
    /\bprompt is short\b/i,
    /\bmodel-size setting\b/i,
    /\bspelling mistakes\b/i,
    /\bexact punctuation\b/i,
    /\bcapability trivia\b/i,
    /\blonger model name\b/i,
    /\bUI screenshot\b/i,
    /\bhero chart\b/i,
    /\bmodel nickname\b/i,
    /\bvocabulary issue\b/i,
    /\bmodel temperature\b/i,
    /\bUI tab name\b/i,
    /\bprompt-writing style guide\b/i,
    /\btrivia answers\b/i,
    /\bmodel size\b/i,
    /\bone demo looks good\b/i,
  ];

  for (const [index, item] of quiz.entries()) {
    const visibleText = normalize(`${item.prompt} ${item.choices.join(' ')} ${item.explanation}`);
    for (const pattern of unrelatedScopeLeaks) {
      assert.doesNotMatch(visibleText, pattern, `question ${index + 1} drifts outside the frontier-eval lesson scope`);
    }
  }
});

test('frontier evaluation safety assessment avoids visible-page answer leakage', () => {
  const { quiz } = getLessonAssessment('frontier-evaluation-safety');
  const pageSize = 10;
  for (let pageStart = 0; pageStart < quiz.length; pageStart += pageSize) {
    const page = quiz.slice(pageStart, pageStart + pageSize);
    const correctAnswers = page.map((item) => normalize(item.choices[item.answerIndex]));

    assert.equal(
      new Set(correctAnswers).size,
      correctAnswers.length,
      `page starting at question ${pageStart + 1} should not repeat exact answers`,
    );

    for (const [offset, item] of page.entries()) {
      const surroundingVisibleText = page
        .filter((_, otherOffset) => otherOffset !== offset)
        .map((other) => normalize(`${other.prompt} ${other.choices.join(' ')}`));
      const leaked = surroundingVisibleText.some((text) => text.includes(correctAnswers[offset]));
      assert.equal(leaked, false, `${item.id} answer appears in another visible item on same page`);
    }
  }
});

test('frontier evaluation safety assessment distributes correct answer positions per page', () => {
  const { quiz } = getLessonAssessment('frontier-evaluation-safety');
  const pageSize = 10;
  for (let pageStart = 0; pageStart < quiz.length; pageStart += pageSize) {
    const page = quiz.slice(pageStart, pageStart + pageSize);
    const counts = [0, 0, 0];
    for (const item of page) counts[item.answerIndex] += 1;
    assert.ok(Math.max(...counts) - Math.min(...counts) <= 1, `imbalanced page at ${pageStart + 1}: ${counts.join(',')}`);
  }
});
