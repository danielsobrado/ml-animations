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

test('agentic coding systems assessment has 100 production-ready questions with focused labs', () => {
  const { quiz, labs } = getLessonAssessment('agentic-coding-systems');

  assert.equal(labs.length, 3);
  assert.deepEqual(labs.map((lab) => lab.id), [
    'swe-bench-loop-trace',
    'patch-review-lab',
    'approval-boundary-lab',
  ]);

  assert.equal(quiz.length, 100);
  const ids = new Set();
  const globalCounts = [0, 0, 0];

  for (const [index, item] of quiz.entries()) {
    assert.match(item.id, /^agentcode-\d{3}$/);
    assert.equal(ids.has(item.id), false, `duplicate id ${item.id}`);
    ids.add(item.id);
    assert.equal(item.id, `agentcode-${String(index + 1).padStart(3, '0')}`);
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

test('agentic coding systems assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('agentic-coding-systems');
  const prompts = quiz.map((item) => normalize(item.prompt));
  const correctAnswers = quiz.map((item) => normalize(correctAnswer(item)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(correctAnswers).size, correctAnswers.length);
});

test('agentic coding systems assessment progresses through workflow evidence', () => {
  const { quiz } = getLessonAssessment('agentic-coding-systems');
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
    ['loop basics', ['permissioned loop', 'reads issues']],
    ['swe-bench evidence', ['FAIL_TO_PASS evidence', 'target test fails']],
    ['repo navigation', ['map frames', 'likely files']],
    ['approval boundaries', ['shell command approval-required', 'external state']],
    ['scenario practice', ['parser crash', 'stack trace']],
    ['production takeaway', ['production-ready takeaway', 'permission gates']],
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

test('agentic coding systems assessment marks unsafe misconceptions as traps after setup', () => {
  const { quiz } = getLessonAssessment('agentic-coding-systems');
  const misconceptionTerms = [
    /just asking an LLM for code/i,
    /Passing only FAIL_TO_PASS proves/i,
    /edit confidently without reading/i,
    /larger diff is usually safer/i,
    /Editing test expectations is enough/i,
    /Sandboxing proves the code/i,
    /AGENTS\.md is always current/i,
    /Passing tests authorizes any destructive/i,
    /compiling patch is automatically reviewable/i,
    /keep all bad edits/i,
    /More reviewer agents always improve/i,
    /narrow target test passing is proof/i,
    /all tests pass when only one focused test/i,
    /Extra unrelated improvements are harmless/i,
    /deploy automatically by default/i,
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

test('agentic coding systems assessment stays within visible lesson scope', () => {
  const { quiz } = getLessonAssessment('agentic-coding-systems');
  const unrelatedScopeLeaks = [
    /\bcompiler optimization\b/i,
    /\bgenerated PR title\b/i,
    /\bmodel tokens\b/i,
    /\bformatting in random files\b/i,
    /\bmarketing description\b/i,
    /\bnumber of files in the repo\b/i,
    /\bfirst noun\b/i,
    /\bauthor profile\b/i,
    /\bPR titles\b/i,
    /\bpackage download speed\b/i,
    /\bmaximum number of tools\b/i,
    /\bpackage logo\b/i,
    /\btoken probability\b/i,
    /\bfashionable\b/i,
    /\bPR title is too short\b/i,
    /\bstatic autocomplete model\b/i,
    /\bbiggest file\b/i,
    /\bgenerated tokens\b/i,
  ];

  for (const [index, item] of quiz.entries()) {
    const visibleText = normalize(`${item.prompt} ${item.choices.join(' ')} ${item.explanation}`);
    for (const pattern of unrelatedScopeLeaks) {
      assert.doesNotMatch(visibleText, pattern, `question ${index + 1} drifts outside the coding-agent lesson scope`);
    }
  }
});

test('agentic coding systems assessment avoids visible-page answer leakage', () => {
  const { quiz } = getLessonAssessment('agentic-coding-systems');
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

test('agentic coding systems assessment distributes correct answer positions per page', () => {
  const { quiz } = getLessonAssessment('agentic-coding-systems');
  const pageSize = 10;
  for (let pageStart = 0; pageStart < quiz.length; pageStart += pageSize) {
    const page = quiz.slice(pageStart, pageStart + pageSize);
    const counts = [0, 0, 0];
    for (const item of page) counts[item.answerIndex] += 1;
    assert.ok(Math.max(...counts) - Math.min(...counts) <= 1, `imbalanced page at ${pageStart + 1}: ${counts.join(',')}`);
  }
});
