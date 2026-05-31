import assert from 'node:assert/strict';
import test from 'node:test';
import { TOOL_USING_REASONING_MODELS_QUIZ } from './toolUsingReasoningModelsAssessment.js';

const LEVELS = new Set(['Foundation', 'Mechanism', 'Application', 'Tricky', 'Interview']);

function normalize(value) {
  return String(value).toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
}

function correctAnswer(item) {
  return item.choices[item.answerIndex];
}

test('tool-using reasoning models assessment has 100 production-ready questions', () => {
  assert.equal(TOOL_USING_REASONING_MODELS_QUIZ.length, 100);
  const ids = new Set();

  for (const [index, item] of TOOL_USING_REASONING_MODELS_QUIZ.entries()) {
    assert.match(item.id, /^tool-\d{3}$/);
    assert.equal(ids.has(item.id), false, `duplicate id ${item.id}`);
    ids.add(item.id);
    assert.equal(item.id, `tool-${String(index + 1).padStart(3, '0')}`);
    assert.equal(LEVELS.has(item.level), true, `${item.id} has unexpected level ${item.level}`);
    assert.equal(item.choices.length, 3, `${item.id} should have three choices`);
    assert.ok(item.answerIndex >= 0 && item.answerIndex < 3, `${item.id} answerIndex out of range`);
    assert.ok(item.prompt.length > 20, `${item.id} prompt too short`);
    assert.ok(item.explanation.length > 30, `${item.id} explanation too short`);
    assert.equal(new Set(item.choices.map(normalize)).size, 3, `${item.id} has duplicate choices`);
  }
});

test('tool-using reasoning models assessment avoids duplicate prompts and correct answers', () => {
  const prompts = TOOL_USING_REASONING_MODELS_QUIZ.map((item) => normalize(item.prompt));
  const correctAnswers = TOOL_USING_REASONING_MODELS_QUIZ.map((item) => normalize(correctAnswer(item)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(correctAnswers).size, correctAnswers.length);
});

test('tool-using reasoning models assessment progresses through the lesson objectives', () => {
  const ranges = [
    ['Foundation', 0, 20],
    ['Mechanism', 20, 50],
    ['Application', 50, 75],
    ['Tricky', 75, 90],
    ['Interview', 90, 100],
  ];

  for (const [level, start, end] of ranges) {
    assert.equal(TOOL_USING_REASONING_MODELS_QUIZ.slice(start, end).every((item) => item.level === level), true, `${level} range mismatch`);
  }

  const milestones = [
    ['loop basics', ['interleaves internal reasoning', 'external actions']],
    ['search policy', ['multi-turn query', 'specific follow-up queries']],
    ['tool-result masking', ['tool-result masking', 'model-generation loss']],
    ['permission gates', ['side-effecting tools', 'stricter controls']],
    ['scenario practice', ['today s stock price', 'time-sensitive']],
    ['production takeaway', ['production-ready takeaway', 'least privilege']],
  ];

  let previous = -1;
  for (const [name, terms] of milestones) {
    const index = TOOL_USING_REASONING_MODELS_QUIZ.findIndex((item) => (
      terms.every((term) => normalize(`${item.prompt} ${item.choices.join(' ')} ${item.explanation}`).includes(normalize(term)))
    ));
    assert.notEqual(index, -1, `missing milestone: ${name}`);
    assert.ok(index > previous, `${name} appears out of order`);
    previous = index;
  }
});

test('tool-using reasoning models assessment marks unsafe misconceptions as traps after setup', () => {
  const misconceptionTerms = [
    /automatically makes final answers correct and safe/i,
    /first search result should be treated as current truth/i,
    /Python output is always correct/i,
    /document contains an instruction.*follow it as policy/i,
    /valid JSON tool call proves/i,
    /never revising the first action sequence/i,
    /prevents the model from using observations as context/i,
    /Rewarding more tool calls always improves/i,
    /Read\/write tool access should be equally unrestricted/i,
    /report that result even when the tool did not return it/i,
    /click any destructive control without approval/i,
    /Persistent memory is always current/i,
    /High final-answer accuracy alone proves/i,
    /Repeating the same failed search indefinitely/i,
    /static QA set is sufficient/i,
  ];
  const trapPrompt = /false|wrong|unsafe|trap|reject|dangerous/i;

  for (const [index, item] of TOOL_USING_REASONING_MODELS_QUIZ.entries()) {
    const text = `${item.prompt} ${item.choices.join(' ')}`;
    const containsMisconception = misconceptionTerms.some((pattern) => pattern.test(text));
    if (!containsMisconception) continue;

    assert.ok(index >= 75, `${item.id} introduces misconception too early`);
    assert.match(item.prompt, trapPrompt, `${item.id} should mark misconception as a trap`);
  }
});

test('tool-using reasoning models assessment avoids visible-page answer leakage', () => {
  const pageSize = 10;
  for (let pageStart = 0; pageStart < TOOL_USING_REASONING_MODELS_QUIZ.length; pageStart += pageSize) {
    const page = TOOL_USING_REASONING_MODELS_QUIZ.slice(pageStart, pageStart + pageSize);
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

test('tool-using reasoning models assessment distributes correct answer positions per page', () => {
  const pageSize = 10;
  for (let pageStart = 0; pageStart < TOOL_USING_REASONING_MODELS_QUIZ.length; pageStart += pageSize) {
    const page = TOOL_USING_REASONING_MODELS_QUIZ.slice(pageStart, pageStart + pageSize);
    const counts = [0, 0, 0];
    for (const item of page) counts[item.answerIndex] += 1;
    assert.ok(Math.max(...counts) - Math.min(...counts) <= 1, `imbalanced page at ${pageStart + 1}: ${counts.join(',')}`);
  }
});
