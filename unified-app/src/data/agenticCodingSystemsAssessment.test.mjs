import assert from 'node:assert/strict';
import test from 'node:test';
import { AGENTIC_CODING_SYSTEMS_QUIZ } from './agenticCodingSystemsAssessment.js';

const LEVELS = new Set(['Foundation', 'Mechanism', 'Application', 'Tricky', 'Interview']);

function normalize(value) {
  return String(value).toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
}

test('agentic coding systems assessment has 100 production-ready questions', () => {
  assert.equal(AGENTIC_CODING_SYSTEMS_QUIZ.length, 100);
  const ids = new Set();
  const prompts = new Set();
  const correctAnswers = new Set();

  for (const [index, item] of AGENTIC_CODING_SYSTEMS_QUIZ.entries()) {
    assert.match(item.id, /^agentcode-\d{3}$/);
    assert.equal(ids.has(item.id), false, `duplicate id ${item.id}`);
    ids.add(item.id);
    assert.equal(item.id, `agentcode-${String(index + 1).padStart(3, '0')}`);
    assert.equal(LEVELS.has(item.level), true, `${item.id} has unexpected level ${item.level}`);
    assert.equal(item.choices.length, 3, `${item.id} should have three choices`);
    assert.ok(item.answerIndex >= 0 && item.answerIndex < 3, `${item.id} answerIndex out of range`);
    assert.ok(item.prompt.length > 20, `${item.id} prompt too short`);
    assert.ok(item.explanation.length > 30, `${item.id} explanation too short`);
    assert.equal(new Set(item.choices.map(normalize)).size, 3, `${item.id} has duplicate choices`);

    const normalizedPrompt = normalize(item.prompt);
    const normalizedCorrect = normalize(item.choices[item.answerIndex]);
    assert.equal(prompts.has(normalizedPrompt), false, `${item.id} repeats a prompt`);
    assert.equal(correctAnswers.has(normalizedCorrect), false, `${item.id} repeats a correct answer`);
    prompts.add(normalizedPrompt);
    correctAnswers.add(normalizedCorrect);
  }
});

test('agentic coding systems assessment progresses through workflow evidence', () => {
  const ranges = [
    ['Foundation', 0, 20],
    ['Mechanism', 20, 50],
    ['Application', 50, 75],
    ['Tricky', 75, 90],
    ['Interview', 90, 100],
  ];

  for (const [level, start, end] of ranges) {
    assert.equal(AGENTIC_CODING_SYSTEMS_QUIZ.slice(start, end).every((item) => item.level === level), true, `${level} range mismatch`);
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
    const index = AGENTIC_CODING_SYSTEMS_QUIZ.findIndex((item) => (
      terms.every((term) => normalize(`${item.prompt} ${item.choices.join(' ')} ${item.explanation}`).includes(normalize(term)))
    ));
    assert.notEqual(index, -1, `missing milestone: ${name}`);
    assert.ok(index > previous, `${name} appears out of order`);
    previous = index;
  }
});

test('agentic coding systems assessment marks unsafe misconceptions as traps after setup', () => {
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

  for (const [index, item] of AGENTIC_CODING_SYSTEMS_QUIZ.entries()) {
    const text = `${item.prompt} ${item.choices.join(' ')}`;
    const containsMisconception = misconceptionTerms.some((pattern) => pattern.test(text));
    if (!containsMisconception) continue;

    assert.ok(index >= 75, `${item.id} introduces misconception too early`);
    assert.match(item.prompt, trapPrompt, `${item.id} should mark misconception as a trap`);
  }
});

test('agentic coding systems assessment avoids visible-page answer leakage', () => {
  const pageSize = 10;
  for (let pageStart = 0; pageStart < AGENTIC_CODING_SYSTEMS_QUIZ.length; pageStart += pageSize) {
    const page = AGENTIC_CODING_SYSTEMS_QUIZ.slice(pageStart, pageStart + pageSize);
    const correctAnswers = page.map((item) => normalize(item.choices[item.answerIndex]));

    for (const [offset, item] of page.entries()) {
      const surroundingPrompts = page
        .filter((_, otherOffset) => otherOffset !== offset)
        .map((other) => normalize(other.prompt));
      const leaked = surroundingPrompts.some((prompt) => prompt.includes(correctAnswers[offset]));
      assert.equal(leaked, false, `${item.id} answer appears in another prompt on same page`);
    }
  }
});

test('agentic coding systems assessment distributes correct answer positions per page', () => {
  const pageSize = 10;
  for (let pageStart = 0; pageStart < AGENTIC_CODING_SYSTEMS_QUIZ.length; pageStart += pageSize) {
    const page = AGENTIC_CODING_SYSTEMS_QUIZ.slice(pageStart, pageStart + pageSize);
    const counts = [0, 0, 0];
    for (const item of page) counts[item.answerIndex] += 1;
    assert.ok(Math.max(...counts) - Math.min(...counts) <= 1, `imbalanced page at ${pageStart + 1}: ${counts.join(',')}`);
  }
});
