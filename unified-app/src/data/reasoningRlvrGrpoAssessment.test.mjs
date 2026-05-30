import assert from 'node:assert/strict';
import test from 'node:test';
import { REASONING_RLVR_GRPO_QUIZ } from './reasoningRlvrGrpoAssessment.js';

const LEVELS = new Set(['Foundation', 'Mechanism', 'Application', 'Tricky', 'Interview']);

function normalize(value) {
  return String(value).toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
}

test('reasoning RLVR GRPO assessment has 100 production-ready questions', () => {
  assert.equal(REASONING_RLVR_GRPO_QUIZ.length, 100);
  const ids = new Set();
  const prompts = new Set();
  const correctAnswers = new Set();

  for (const [index, item] of REASONING_RLVR_GRPO_QUIZ.entries()) {
    assert.match(item.id, /^rlvr-grpo-\d{3}$/);
    assert.equal(ids.has(item.id), false, `duplicate id ${item.id}`);
    ids.add(item.id);
    assert.equal(item.id, `rlvr-grpo-${String(index + 1).padStart(3, '0')}`);
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

test('reasoning RLVR GRPO assessment progresses from pipeline basics to production audit', () => {
  const ranges = [
    ['Foundation', 0, 20],
    ['Mechanism', 20, 50],
    ['Application', 50, 75],
    ['Tricky', 75, 90],
    ['Interview', 90, 100],
  ];

  for (const [level, start, end] of ranges) {
    assert.equal(REASONING_RLVR_GRPO_QUIZ.slice(start, end).every((item) => item.level === level), true, `${level} range mismatch`);
  }

  const milestones = [
    ['cold-start setup', ['cold start', 'early rl']],
    ['GRPO baseline', ['group', 'baseline']],
    ['ORM PRM contrast', ['ORM', 'whole chain']],
    ['reward shaping', ['correctness', 'format', 'language coherence', 'length penalty']],
    ['distillation', ['distillation curation', 'verbose loops']],
    ['production readiness', ['verifier aligned rewards', 'hidden tasks']],
  ];

  let previous = -1;
  for (const [name, terms] of milestones) {
    const index = REASONING_RLVR_GRPO_QUIZ.findIndex((item) => (
      terms.every((term) => normalize(`${item.prompt} ${item.choices.join(' ')} ${item.explanation}`).includes(normalize(term)))
    ));
    assert.notEqual(index, -1, `missing milestone: ${name}`);
    assert.ok(index > previous, `${name} appears out of order`);
    previous = index;
  }
});

test('reasoning RLVR GRPO assessment marks unsafe misconceptions as traps after setup', () => {
  const misconceptionTerms = [
    /chain-of-thought examples alone/i,
    /guarantee there is no reward hacking/i,
    /verify every open-ended task/i,
    /always safe to reinforce/i,
    /readable tags necessarily mean correct reasoning/i,
    /larger length penalty always improves/i,
    /automatically has the teacher model capability/i,
    /necessarily correct in an absolute sense/i,
    /always deterministic, free, and available/i,
    /zero KL pressure is always best/i,
    /longer traces are always better/i,
    /all-correct groups always provide/i,
    /same as online RL/i,
    /proves every intermediate reasoning step/i,
    /rising training reward alone proves/i,
  ];
  const trapPrompt = /false|wrong|unsafe|trap|reject|misconception/i;

  for (const [index, item] of REASONING_RLVR_GRPO_QUIZ.entries()) {
    const text = `${item.prompt} ${item.choices.join(' ')}`;
    const containsMisconception = misconceptionTerms.some((pattern) => pattern.test(text));
    if (!containsMisconception) continue;

    assert.ok(index >= 75, `${item.id} introduces misconception too early`);
    assert.match(item.prompt, trapPrompt, `${item.id} should mark misconception as a trap`);
  }
});

test('reasoning RLVR GRPO assessment avoids visible-page answer leakage', () => {
  const pageSize = 10;
  for (let pageStart = 0; pageStart < REASONING_RLVR_GRPO_QUIZ.length; pageStart += pageSize) {
    const page = REASONING_RLVR_GRPO_QUIZ.slice(pageStart, pageStart + pageSize);
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

test('reasoning RLVR GRPO assessment distributes correct answer positions per page', () => {
  const pageSize = 10;
  for (let pageStart = 0; pageStart < REASONING_RLVR_GRPO_QUIZ.length; pageStart += pageSize) {
    const page = REASONING_RLVR_GRPO_QUIZ.slice(pageStart, pageStart + pageSize);
    const counts = [0, 0, 0];
    for (const item of page) counts[item.answerIndex] += 1;
    assert.ok(Math.max(...counts) - Math.min(...counts) <= 1, `imbalanced page at ${pageStart + 1}: ${counts.join(',')}`);
  }
});
