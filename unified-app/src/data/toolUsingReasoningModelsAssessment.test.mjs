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

test('tool-using reasoning models assessment has 100 production-ready questions with focused labs', () => {
  const { quiz, labs } = getLessonAssessment('tool-using-reasoning-models');

  assert.equal(labs.length, 7);
  assert.deepEqual(labs.map((lab) => lab.id), [
    'choose-the-right-tool',
    'query-refinement',
    'python-verifier',
    'file-grounding-audit',
    'function-vs-plan',
    'tool-result-masking',
    'failure-injection',
  ]);

  assert.equal(quiz.length, 100);
  const ids = new Set();
  const globalCounts = [0, 0, 0];

  for (const [index, item] of quiz.entries()) {
    assert.match(item.id, /^tool-\d{3}$/);
    assert.equal(ids.has(item.id), false, `duplicate id ${item.id}`);
    ids.add(item.id);
    assert.equal(item.id, `tool-${String(index + 1).padStart(3, '0')}`);
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

test('tool-using reasoning models assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('tool-using-reasoning-models');
  const prompts = quiz.map((item) => normalize(item.prompt));
  const correctAnswers = quiz.map((item) => normalize(correctAnswer(item)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(correctAnswers).size, correctAnswers.length);
});

test('tool-using reasoning models assessment progresses through the lesson objectives', () => {
  const { quiz } = getLessonAssessment('tool-using-reasoning-models');
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
    ['loop basics', ['interleaves internal reasoning', 'external actions']],
    ['search policy', ['multi-turn query', 'specific follow-up queries']],
    ['tool-result masking', ['tool-result masking', 'model-generation loss']],
    ['permission gates', ['side-effecting tools', 'stricter controls']],
    ['scenario practice', ['today s stock price', 'time-sensitive']],
    ['production takeaway', ['production-ready takeaway', 'least privilege']],
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

test('tool-using reasoning models assessment marks unsafe misconceptions as traps after setup', () => {
  const { quiz } = getLessonAssessment('tool-using-reasoning-models');
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

  for (const [index, item] of quiz.entries()) {
    const text = `${item.prompt} ${item.choices.join(' ')}`;
    const containsMisconception = misconceptionTerms.some((pattern) => pattern.test(text));
    if (!containsMisconception) continue;

    assert.ok(index >= 75, `${item.id} introduces misconception too early`);
    assert.match(item.prompt, trapPrompt, `${item.id} should mark misconception as a trap`);
  }
});

test('tool-using reasoning models assessment avoids visible-page answer leakage', () => {
  const { quiz } = getLessonAssessment('tool-using-reasoning-models');
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

test('tool-using reasoning models assessment distributes correct answer positions per page', () => {
  const { quiz } = getLessonAssessment('tool-using-reasoning-models');
  const pageSize = 10;
  for (let pageStart = 0; pageStart < quiz.length; pageStart += pageSize) {
    const page = quiz.slice(pageStart, pageStart + pageSize);
    const counts = [0, 0, 0];
    for (const item of page) counts[item.answerIndex] += 1;
    assert.ok(Math.max(...counts) - Math.min(...counts) <= 1, `imbalanced page at ${pageStart + 1}: ${counts.join(',')}`);
  }
});
