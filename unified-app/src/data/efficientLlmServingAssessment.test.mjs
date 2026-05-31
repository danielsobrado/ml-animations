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

test('efficient LLM serving assessment has 100 production-ready questions with focused labs', () => {
  const { quiz, labs } = getLessonAssessment('efficient-llm-serving');

  assert.equal(labs.length, 4);
  assert.deepEqual(labs.map((lab) => lab.id), [
    'prefill-decode-diagnosis',
    'paged-kv-allocation',
    'speculation-acceptance',
    'goodput-slo',
  ]);

  assert.equal(quiz.length, 100);
  const ids = new Set();
  const globalCounts = [0, 0, 0];

  for (const [index, item] of quiz.entries()) {
    assert.match(item.id, /^serve-\d{3}$/);
    assert.equal(ids.has(item.id), false, `duplicate id ${item.id}`);
    ids.add(item.id);
    assert.equal(item.id, `serve-${String(index + 1).padStart(3, '0')}`);
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

test('efficient LLM serving assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('efficient-llm-serving');
  const prompts = quiz.map((item) => normalize(item.prompt));
  const correctAnswers = quiz.map((item) => normalize(correctAnswer(item)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(correctAnswers).size, correctAnswers.length);
});

test('efficient LLM serving assessment progresses from serving basics to production tuning', () => {
  const { quiz } = getLessonAssessment('efficient-llm-serving');
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
    ['prefill decode', ['Processing the prompt tokens', 'KV cache']],
    ['paged KV', ['PagedAttention', 'fixed-size blocks']],
    ['speculation economics', ['speculative decoding', 'acceptance rate']],
    ['goodput SLO', ['Goodput', 'latency targets']],
    ['incident diagnosis', ['KV fragmentation', 'rejected speculation']],
    ['production takeaway', ['production-ready takeaway', 'low-latency completions']],
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

test('efficient LLM serving assessment marks unsafe misconceptions as traps after setup', () => {
  const { quiz } = getLessonAssessment('efficient-llm-serving');
  const misconceptionTerms = [
    /Maximizing raw throughput always improves/i,
    /Continuous batching removes all tail-latency risk/i,
    /eliminates the need to manage KV memory/i,
    /semantically similar but textually different prefixes/i,
    /Smaller prefill chunks are always better/i,
    /Speculative decoding always speeds/i,
    /accepts all future-token heads without verification/i,
    /removes the target model/i,
    /Lower bit width always improves/i,
    /KV quantization is free/i,
    /same bottleneck/i,
    /Average latency is enough/i,
    /counts all generated tokens even when the SLO is missed/i,
    /Adding more devices always reduces latency/i,
    /reported 6.5x speedup guarantees/i,
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

test('efficient LLM serving assessment avoids visible-page answer leakage', () => {
  const { quiz } = getLessonAssessment('efficient-llm-serving');
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
      const surroundingPrompts = page
        .filter((_, otherOffset) => otherOffset !== offset)
        .map((other) => normalize(other.prompt));
      const leaked = surroundingPrompts.some((prompt) => prompt.includes(correctAnswers[offset]));
      assert.equal(leaked, false, `${item.id} answer appears in another prompt on same page`);
    }
  }
});

test('efficient LLM serving assessment distributes correct answer positions per page', () => {
  const { quiz } = getLessonAssessment('efficient-llm-serving');
  const pageSize = 10;
  for (let pageStart = 0; pageStart < quiz.length; pageStart += pageSize) {
    const page = quiz.slice(pageStart, pageStart + pageSize);
    const counts = [0, 0, 0];
    for (const item of page) counts[item.answerIndex] += 1;
    assert.ok(Math.max(...counts) - Math.min(...counts) <= 1, `imbalanced page at ${pageStart + 1}: ${counts.join(',')}`);
  }
});
