import assert from 'node:assert/strict';
import test from 'node:test';
import { LONG_CONTEXT_FRONTIER_MODELS_QUIZ } from './longContextFrontierModelsAssessment.js';

const LEVELS = new Set(['Foundation', 'Mechanism', 'Application', 'Tricky', 'Interview']);

function normalize(value) {
  return String(value).toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
}

test('long-context frontier models assessment has 100 production-ready questions', () => {
  assert.equal(LONG_CONTEXT_FRONTIER_MODELS_QUIZ.length, 100);
  const ids = new Set();
  const prompts = new Set();
  const correctAnswers = new Set();

  for (const [index, item] of LONG_CONTEXT_FRONTIER_MODELS_QUIZ.entries()) {
    assert.match(item.id, /^longctx-\d{3}$/);
    assert.equal(ids.has(item.id), false, `duplicate id ${item.id}`);
    ids.add(item.id);
    assert.equal(item.id, `longctx-${String(index + 1).padStart(3, '0')}`);
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

test('long-context frontier models assessment progresses from basics to production audit', () => {
  const ranges = [
    ['Foundation', 0, 20],
    ['Mechanism', 20, 50],
    ['Application', 50, 75],
    ['Tricky', 75, 90],
    ['Interview', 90, 100],
  ];

  for (const [level, start, end] of ranges) {
    assert.equal(LONG_CONTEXT_FRONTIER_MODELS_QUIZ.slice(start, end).every((item) => item.level === level), true, `${level} range mismatch`);
  }

  const milestones = [
    ['effective context', ['effective context', 'given task']],
    ['hybrid RAG', ['hybrid RAG', 'selected evidence']],
    ['KV cache cost', ['KV cache', 'head dimension']],
    ['semantic needle', ['semantic needle', 'wording differs']],
    ['legal hybrid scenario', ['200 legal documents', 'hybrid retrieval']],
    ['production readiness', ['effective systems', 'serving cost']],
  ];

  let previous = -1;
  for (const [name, terms] of milestones) {
    const index = LONG_CONTEXT_FRONTIER_MODELS_QUIZ.findIndex((item) => (
      terms.every((term) => normalize(`${item.prompt} ${item.choices.join(' ')} ${item.explanation}`).includes(normalize(term)))
    ));
    assert.notEqual(index, -1, `missing milestone: ${name}`);
    assert.ok(index > previous, `${name} appears out of order`);
    previous = index;
  }
});

test('long-context frontier models assessment marks unsafe misconceptions as traps after setup', () => {
  const misconceptionTerms = [
    /bigger context window is a substitute/i,
    /reliably uses all 1M tokens/i,
    /Passing needle-in-haystack proves/i,
    /RoPE scaling guarantees/i,
    /no serving cost once/i,
    /RAG always contains all required evidence/i,
    /Full context removes distractor confusion/i,
    /compressed memory is perfect context/i,
    /removes the need to evaluate retrieval quality/i,
    /position cannot affect use/i,
    /just finding one passage/i,
    /fluent answer means/i,
    /Increasing top-k always improves/i,
    /Prompt caching makes every long-context request cheap/i,
    /One long-context benchmark score is enough/i,
  ];
  const trapPrompt = /false|wrong|unsafe|trap|reject|dangerous/i;

  for (const [index, item] of LONG_CONTEXT_FRONTIER_MODELS_QUIZ.entries()) {
    const text = `${item.prompt} ${item.choices.join(' ')}`;
    const containsMisconception = misconceptionTerms.some((pattern) => pattern.test(text));
    if (!containsMisconception) continue;

    assert.ok(index >= 75, `${item.id} introduces misconception too early`);
    assert.match(item.prompt, trapPrompt, `${item.id} should mark misconception as a trap`);
  }
});

test('long-context frontier models assessment avoids visible-page answer leakage', () => {
  const pageSize = 10;
  for (let pageStart = 0; pageStart < LONG_CONTEXT_FRONTIER_MODELS_QUIZ.length; pageStart += pageSize) {
    const page = LONG_CONTEXT_FRONTIER_MODELS_QUIZ.slice(pageStart, pageStart + pageSize);
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

test('long-context frontier models assessment distributes correct answer positions per page', () => {
  const pageSize = 10;
  for (let pageStart = 0; pageStart < LONG_CONTEXT_FRONTIER_MODELS_QUIZ.length; pageStart += pageSize) {
    const page = LONG_CONTEXT_FRONTIER_MODELS_QUIZ.slice(pageStart, pageStart + pageSize);
    const counts = [0, 0, 0];
    for (const item of page) counts[item.answerIndex] += 1;
    assert.ok(Math.max(...counts) - Math.min(...counts) <= 1, `imbalanced page at ${pageStart + 1}: ${counts.join(',')}`);
  }
});
