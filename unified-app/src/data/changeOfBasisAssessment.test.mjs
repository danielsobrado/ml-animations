import assert from 'node:assert/strict';
import test from 'node:test';
import { getLessonAssessment } from './lessonAssessments.js';

const LEVELS = new Set(['Foundation', 'Mechanism', 'Application', 'Tricky', 'Interview']);

function normalize(value) {
  return String(value || '').toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
}

function correctAnswer(item) {
  return item.choices[item.answerIndex];
}

test('change of basis assessment has 100 curated questions', () => {
  const { quiz, labs } = getLessonAssessment('change-of-basis');

  assert.equal(quiz.length, 100);
  assert.equal(labs.length, 3);
  assert.equal(new Set(quiz.map((item) => item.id)).size, 100);

  for (const [index, item] of quiz.entries()) {
    assert.match(item.id, /^cob-\d{3}-[a-z0-9-]+$/);
    assert.equal(item.id.startsWith(`cob-${String(index + 1).padStart(3, '0')}`), true, `${item.id} is out of order`);
    assert.equal(LEVELS.has(item.level), true, `${item.id} has unexpected level ${item.level}`);
    assert.equal(item.choices.length, 3, `${item.id} should have three choices`);
    assert.ok(item.answerIndex >= 0 && item.answerIndex < 3, `${item.id} answerIndex out of range`);
    assert.ok(item.prompt.length > 20, `${item.id} prompt too short`);
    assert.ok(item.explanation.length > 30, `${item.id} explanation too short`);
    assert.equal(new Set(item.choices.map(normalize)).size, 3, `${item.id} has duplicate choices`);
  }
});

test('change of basis assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('change-of-basis');
  const prompts = quiz.map((item) => normalize(item.prompt));
  const answers = quiz.map((item) => normalize(correctAnswer(item)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('change of basis assessment progresses through lesson learning points', () => {
  const { quiz } = getLessonAssessment('change-of-basis');
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
    ['purpose', ['main purpose of change of basis']],
    ['basis columns', ['basis matrix B']],
    ['coordinate map', ['x = B [x]_B']],
    ['operator map', ['A_B = B^-1 A B']],
    ['practice lab', ['Practice Lab ask']],
    ['mechanism', ['How does B [x]_B reconstruct x']],
    ['application', ['tilted coordinates [2.5, -0.5]']],
    ['misconception traps', ['Which statement is false about coordinates and vectors']],
    ['interview synthesis', ['strongest final takeaway']],
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

test('change of basis keeps misconception traps after setup', () => {
  const { quiz } = getLessonAssessment('change-of-basis');
  const trapIds = [
    'cob-076-false-coordinate-object',
    'cob-077-wrong-arrow-moves',
    'cob-078-false-row-storage',
    'cob-079-wrong-operator-order',
    'cob-080-dangerous-no-inverse',
    'cob-081-false-dependent-basis',
    'cob-082-false-standard-only',
    'cob-083-wrong-same-tuple',
    'cob-084-false-all-presets',
    'cob-085-wrong-map-vector',
    'cob-086-false-multiply-side',
    'cob-087-wrong-negative-invalid',
    'cob-088-false-map-changes-object',
    'cob-089-dangerous-label-only',
    'cob-090-wrong-practice-spoiler',
  ];
  const misconceptionTerms = [
    /coordinate list is the vector object itself/i,
    /arrow must move/i,
    /stores b1 and b2 as rows/i,
    /A_B = B A B\^-1/i,
    /Ignore B\^-1/i,
    /dependent vectors are always/i,
    /Only the standard basis/i,
    /same numeric tuple always/i,
    /Every preset shown/i,
    /just another coordinate list/i,
    /x = \[x\]_B B/i,
    /negative coordinate means/i,
    /automatically changes the underlying object/i,
    /basis name without tracing/i,
    /Memorize the reveal text/i,
  ];
  const trapPrompt = /false|misleading|wrong|dangerous|rejected|unsafe|overreads|contradicts/i;

  assert.deepEqual(quiz.slice(75, 90).map((question) => question.id), trapIds);

  for (const [index, item] of quiz.entries()) {
    const text = `${item.prompt} ${item.choices.join(' ')}`;
    const containsMisconception = misconceptionTerms.some((pattern) => pattern.test(text));
    if (!containsMisconception) continue;

    assert.ok(index >= 75, `${item.id} introduces misconception too early`);
    assert.match(item.prompt, trapPrompt, `${item.id} should mark misconception as a trap`);
  }
});

test('change of basis assessment avoids visible-page answer leakage', () => {
  const { quiz } = getLessonAssessment('change-of-basis');
  const pageSize = 10;

  for (let pageStart = 0; pageStart < quiz.length; pageStart += pageSize) {
    const page = quiz.slice(pageStart, pageStart + pageSize);
    const answers = page.map((item) => normalize(correctAnswer(item)));

    for (const [offset, item] of page.entries()) {
      const surroundingPrompts = page
        .filter((_, otherOffset) => otherOffset !== offset)
        .map((other) => normalize(other.prompt));
      const leaked = surroundingPrompts.some((prompt) => prompt.includes(answers[offset]));
      assert.equal(leaked, false, `${item.id} answer appears in another prompt on same page`);
    }
  }
});

test('change of basis assessment distributes correct answer positions per page', () => {
  const { quiz } = getLessonAssessment('change-of-basis');
  const pageSize = 10;

  for (let pageStart = 0; pageStart < quiz.length; pageStart += pageSize) {
    const page = quiz.slice(pageStart, pageStart + pageSize);
    const counts = [0, 0, 0];
    for (const item of page) counts[item.answerIndex] += 1;
    assert.ok(Math.max(...counts) - Math.min(...counts) <= 1, `imbalanced page at ${pageStart + 1}: ${counts.join(',')}`);
  }
});
