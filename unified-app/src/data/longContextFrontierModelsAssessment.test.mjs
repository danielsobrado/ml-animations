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

test('long-context frontier models assessment has 100 production-ready questions with focused labs', () => {
  const { quiz, labs } = getLessonAssessment('long-context-frontier-models');

  assert.equal(labs.length, 3);
  assert.deepEqual(labs.map((lab) => lab.id), [
    'strategy-selection',
    'lost-middle-mitigation',
    'hybrid-packing',
  ]);

  assert.equal(quiz.length, 100);
  const ids = new Set();
  const globalCounts = [0, 0, 0];

  for (const [index, item] of quiz.entries()) {
    assert.match(item.id, /^longctx-\d{3}$/);
    assert.equal(ids.has(item.id), false, `duplicate id ${item.id}`);
    ids.add(item.id);
    assert.equal(item.id, `longctx-${String(index + 1).padStart(3, '0')}`);
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

test('long-context frontier models assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('long-context-frontier-models');
  const prompts = quiz.map((item) => normalize(item.prompt));
  const correctAnswers = quiz.map((item) => normalize(correctAnswer(item)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(correctAnswers).size, correctAnswers.length);
});

test('long-context frontier models assessment progresses from basics to production audit', () => {
  const { quiz } = getLessonAssessment('long-context-frontier-models');
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
    ['effective context', ['effective context', 'given task']],
    ['hybrid RAG', ['hybrid RAG', 'selected evidence']],
    ['KV cache cost', ['KV cache', 'head dimension']],
    ['semantic needle', ['semantic needle', 'wording differs']],
    ['legal hybrid scenario', ['200 legal documents', 'hybrid retrieval']],
    ['production readiness', ['effective systems', 'serving cost']],
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

test('long-context frontier models assessment marks unsafe misconceptions as traps after setup', () => {
  const { quiz } = getLessonAssessment('long-context-frontier-models');
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

  for (const [index, item] of quiz.entries()) {
    const text = `${item.prompt} ${item.choices.join(' ')}`;
    const containsMisconception = misconceptionTerms.some((pattern) => pattern.test(text));
    if (!containsMisconception) continue;

    assert.ok(index >= 75, `${item.id} introduces misconception too early`);
    assert.match(item.prompt, trapPrompt, `${item.id} should mark misconception as a trap`);
  }
});

test('long-context frontier models assessment stays within visible lesson scope', () => {
  const { quiz } = getLessonAssessment('long-context-frontier-models');
  const unrelatedScopeLeaks = [
    /\btokenizer\b/i,
    /\btokenization\b/i,
    /\bimage encoders\b/i,
    /\bimages\b/i,
    /\bmodel layers\b/i,
    /\boutput vocabulary\b/i,
    /\btraining labels\b/i,
    /\bcss\b/i,
    /\bfont size\b/i,
    /\bvocabulary-size\b/i,
    /\bui\b/i,
    /\bactive tab\b/i,
    /\broute icon\b/i,
    /\bmodel vocabulary\b/i,
  ];

  for (const [index, item] of quiz.entries()) {
    const visibleText = normalize(`${item.prompt} ${item.choices.join(' ')} ${item.explanation}`);
    for (const pattern of unrelatedScopeLeaks) {
      assert.doesNotMatch(visibleText, pattern, `question ${index + 1} drifts outside the long-context lesson scope`);
    }
  }
});

test('long-context frontier models assessment avoids visible-page answer leakage', () => {
  const { quiz } = getLessonAssessment('long-context-frontier-models');
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
      assert.equal(leaked, false, `${item.id} answer appears in another prompt on same page`);
    }
  }
});

test('long-context frontier models assessment distributes correct answer positions per page', () => {
  const { quiz } = getLessonAssessment('long-context-frontier-models');
  const pageSize = 10;
  for (let pageStart = 0; pageStart < quiz.length; pageStart += pageSize) {
    const page = quiz.slice(pageStart, pageStart + pageSize);
    const counts = [0, 0, 0];
    for (const item of page) counts[item.answerIndex] += 1;
    assert.ok(Math.max(...counts) - Math.min(...counts) <= 1, `imbalanced page at ${pageStart + 1}: ${counts.join(',')}`);
  }
});
