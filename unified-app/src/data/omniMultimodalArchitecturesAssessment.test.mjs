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

test('omni multimodal architectures assessment has 100 production-ready questions with focused labs', () => {
  const { quiz, labs } = getLessonAssessment('omni-multimodal-architectures');

  assert.equal(labs.length, 4);
  assert.deepEqual(labs.map((lab) => lab.id), [
    'multimodal-token-stream',
    'fusion-strategy',
    'grounding-audit',
    'audio-codec-streaming',
  ]);

  assert.equal(quiz.length, 100);
  const ids = new Set();
  const globalCounts = [0, 0, 0];

  for (const [index, item] of quiz.entries()) {
    assert.match(item.id, /^omni-\d{3}$/);
    assert.equal(ids.has(item.id), false, `duplicate id ${item.id}`);
    ids.add(item.id);
    assert.equal(item.id, `omni-${String(index + 1).padStart(3, '0')}`);
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

test('omni multimodal architectures assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('omni-multimodal-architectures');
  const prompts = quiz.map((item) => normalize(item.prompt));
  const correctAnswers = quiz.map((item) => normalize(correctAnswer(item)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(correctAnswers).size, correctAnswers.length);
});

test('omni multimodal architectures assessment progresses from streams to production audit', () => {
  const { quiz } = getLessonAssessment('omni-multimodal-architectures');
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
    ['modality token stream', ['modality token stream']],
    ['projector bridge', ['projector bridge', 'hidden space']],
    ['early fusion cost', ['early fusion', 'shared attention']],
    ['temporal alignment', ['temporal drift', 'wrong moments']],
    ['voice assistant', ['real-time voice chat', 'Thinker-Talker']],
    ['production ready', ['production-ready takeaway', 'latency controls']],
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

test('omni multimodal architectures assessment marks unsafe misconceptions as traps after setup', () => {
  const { quiz } = getLessonAssessment('omni-multimodal-architectures');
  const misconceptionTerms = [
    /Adding an image input automatically/i,
    /Early fusion is always cheaper/i,
    /Late fusion always preserves/i,
    /projector is decorative/i,
    /Video is just image understanding repeated/i,
    /transcript contains every useful speech signal/i,
    /fluent description proves/i,
    /First-packet latency is only/i,
    /Codec autoregression always has better/i,
    /Diffusion audio can never be useful/i,
    /text alone answers something plausible/i,
    /product card guarantees/i,
    /Audio and video tokens are negligible/i,
    /cross-attention bridge always passes/i,
    /One static image benchmark is enough/i,
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

test('omni multimodal architectures assessment stays within visible lesson scope', () => {
  const { quiz } = getLessonAssessment('omni-multimodal-architectures');
  const unrelatedScopeLeaks = [
    /\bcss\b/i,
    /\bfont size\b/i,
    /\bsql\b/i,
    /\bbrowser window\b/i,
    /\bcached in git\b/i,
    /\bapp ui\b/i,
    /\bpackage size\b/i,
    /\bcamelcase\b/i,
    /\bpoetic\b/i,
    /\btext capitalization\b/i,
    /\bsource code\b/i,
    /\blongest words\b/i,
    /\bui button\b/i,
    /\bbayes\b/i,
    /\bk-means\b/i,
    /\balphabetically\b/i,
    /\bfilename\b/i,
  ];

  for (const [index, item] of quiz.entries()) {
    const visibleText = normalize(`${item.prompt} ${item.choices.join(' ')} ${item.explanation}`);
    for (const pattern of unrelatedScopeLeaks) {
      assert.doesNotMatch(visibleText, pattern, `question ${index + 1} drifts outside the omni lesson scope`);
    }
  }
});

test('omni multimodal architectures assessment avoids visible-page answer leakage', () => {
  const { quiz } = getLessonAssessment('omni-multimodal-architectures');
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

test('omni multimodal architectures assessment distributes correct answer positions per page', () => {
  const { quiz } = getLessonAssessment('omni-multimodal-architectures');
  const pageSize = 10;
  for (let pageStart = 0; pageStart < quiz.length; pageStart += pageSize) {
    const page = quiz.slice(pageStart, pageStart + pageSize);
    const counts = [0, 0, 0];
    for (const item of page) counts[item.answerIndex] += 1;
    assert.ok(Math.max(...counts) - Math.min(...counts) <= 1, `imbalanced page at ${pageStart + 1}: ${counts.join(',')}`);
  }
});
