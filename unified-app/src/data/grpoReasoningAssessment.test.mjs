import assert from 'node:assert/strict';
import test from 'node:test';
import { GRPO_REASONING_QUIZ } from './grpoReasoningAssessment.js';

const LEVELS = new Set(['Foundation', 'Mechanism', 'Application', 'Tricky', 'Interview']);

function normalize(value) {
  return String(value).toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
}

test('grpo reasoning assessment has 100 production-ready questions', () => {
  assert.equal(GRPO_REASONING_QUIZ.length, 100);
  const ids = new Set();

  for (const [index, item] of GRPO_REASONING_QUIZ.entries()) {
    assert.match(item.id, /^grpo-\d{3}$/);
    assert.equal(ids.has(item.id), false, `duplicate id ${item.id}`);
    ids.add(item.id);
    assert.equal(item.id, `grpo-${String(index + 1).padStart(3, '0')}`);
    assert.equal(LEVELS.has(item.level), true, `${item.id} has unexpected level ${item.level}`);
    assert.equal(item.choices.length, 3, `${item.id} should have three choices`);
    assert.ok(item.answerIndex >= 0 && item.answerIndex < 3, `${item.id} answerIndex out of range`);
    assert.ok(item.prompt.length > 20, `${item.id} prompt too short`);
    assert.ok(item.explanation.length > 30, `${item.id} explanation too short`);
  }
});

test('grpo reasoning assessment progresses from group basics to production audit', () => {
  const ranges = [
    ['Foundation', 0, 20],
    ['Mechanism', 20, 50],
    ['Application', 50, 75],
    ['Tricky', 75, 90],
    ['Interview', 90, 100],
  ];

  for (const [level, start, end] of ranges) {
    assert.equal(GRPO_REASONING_QUIZ.slice(start, end).every((item) => item.level === level), true, `${level} range mismatch`);
  }

  const milestones = [
    ['group sampling', ['group of candidate completions']],
    ['advantage formula', ['reward relative to the group average']],
    ['guardrails', ['clipping', 'KL reduce']],
    ['R1 pipeline', ['cold-start data', 'further reinforcement learning']],
    ['distillation filters', ['distillation dataset filter']],
    ['production readiness', ['rewards', 'contrast', 'guardrails', 'distilled data']],
  ];

  let previous = -1;
  for (const [name, terms] of milestones) {
    const index = GRPO_REASONING_QUIZ.findIndex((item) => (
      terms.every((term) => normalize(`${item.prompt} ${item.choices.join(' ')} ${item.explanation}`).includes(normalize(term)))
    ));
    assert.notEqual(index, -1, `missing milestone: ${name}`);
    assert.ok(index > previous, `${name} appears out of order`);
    previous = index;
  }
});

test('grpo reasoning assessment marks unsafe misconceptions as traps after setup', () => {
  const misconceptionTerms = [
    /hand-labels every reasoning step/i,
    /always a separately trained critic/i,
    /always provide the strongest/i,
    /always safe to reinforce/i,
    /necessarily correct reasoning/i,
    /guarantee there is no reward hacking/i,
    /high reward curve alone proves/i,
  ];
  const trapPrompt = /false|wrong|unsafe|trap|reject|dangerous|misconception/i;

  for (const [index, item] of GRPO_REASONING_QUIZ.entries()) {
    const text = `${item.prompt} ${item.choices.join(' ')}`;
    const containsMisconception = misconceptionTerms.some((pattern) => pattern.test(text));
    if (!containsMisconception) continue;

    assert.ok(index >= 75, `${item.id} introduces misconception too early`);
    assert.match(item.prompt, trapPrompt, `${item.id} should mark misconception as a trap`);
  }
});

test('grpo reasoning assessment avoids visible-page answer leakage', () => {
  const pageSize = 10;
  for (let pageStart = 0; pageStart < GRPO_REASONING_QUIZ.length; pageStart += pageSize) {
    const page = GRPO_REASONING_QUIZ.slice(pageStart, pageStart + pageSize);
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

test('grpo reasoning assessment distributes correct answer positions per page', () => {
  const pageSize = 10;
  for (let pageStart = 0; pageStart < GRPO_REASONING_QUIZ.length; pageStart += pageSize) {
    const page = GRPO_REASONING_QUIZ.slice(pageStart, pageStart + pageSize);
    const counts = [0, 0, 0];
    for (const item of page) counts[item.answerIndex] += 1;
    assert.ok(Math.max(...counts) - Math.min(...counts) <= 2, `imbalanced page at ${pageStart + 1}: ${counts.join(',')}`);
  }
});
