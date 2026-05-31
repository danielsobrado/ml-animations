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

test('determinant volume assessment has 100 curated questions', () => {
  const { quiz, labs } = getLessonAssessment('determinant-volume');

  assert.equal(quiz.length, 100);
  assert.equal(labs.length, 3);
  assert.equal(new Set(quiz.map((item) => item.id)).size, 100);

  for (const [index, item] of quiz.entries()) {
    assert.match(item.id, /^detv-\d{3}-[a-z0-9-]+$/);
    assert.equal(item.id.startsWith(`detv-${String(index + 1).padStart(3, '0')}`), true, `${item.id} is out of order`);
    assert.equal(LEVELS.has(item.level), true, `${item.id} has unexpected level ${item.level}`);
    assert.equal(item.choices.length, 3, `${item.id} should have three choices`);
    assert.ok(item.answerIndex >= 0 && item.answerIndex < 3, `${item.id} answerIndex out of range`);
    assert.ok(item.prompt.length > 20, `${item.id} prompt too short`);
    assert.ok(item.explanation.length > 30, `${item.id} explanation too short`);
    assert.equal(new Set(item.choices.map(normalize)).size, 3, `${item.id} has duplicate choices`);
  }
});

test('determinant volume assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('determinant-volume');
  const prompts = quiz.map((item) => normalize(item.prompt));
  const answers = quiz.map((item) => normalize(correctAnswer(item)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('determinant volume assessment progresses through lesson learning points', () => {
  const { quiz } = getLessonAssessment('determinant-volume');
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
    ['purpose', ['main purpose of the determinant-as-volume lesson']],
    ['columns', ['columns of A']],
    ['absolute area', ['|det A|']],
    ['zero practice', ['Practice Lab say when det(A) = 0']],
    ['code formula', ['2x2 determinant code lab']],
    ['mechanism', ['Why do the columns determine the parallelogram']],
    ['application', ['det2([[1,0],[0,1]])']],
    ['misconception traps', ['Which determinant claim is false']],
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

test('determinant volume keeps misconception traps after setup', () => {
  const { quiz } = getLessonAssessment('determinant-volume');
  const trapIds = [
    'detv-076-false-magnitude-only',
    'detv-077-wrong-non-square',
    'detv-078-dangerous-length-only',
    'detv-079-false-zero-invertible',
    'detv-080-wrong-sign-collapse',
    'detv-081-dangerous-near-zero',
    'detv-082-false-formula-plus',
    'detv-083-wrong-sign-absolute',
    'detv-084-false-shear',
    'detv-085-wrong-composition',
    'detv-086-false-basis-change',
    'detv-087-dangerous-global-nonlinear',
    'detv-088-false-det-size',
    'detv-089-wrong-collapse-direction',
    'detv-090-dangerous-title-only',
  ];
  const misconceptionTerms = [
    /magnitude alone tells orientation/i,
    /Every rectangular matrix has the same ordinary determinant/i,
    /Use only column lengths/i,
    /zero determinant square matrix can still have/i,
    /Negative determinant means the area collapsed/i,
    /Divide by a near-zero determinant/i,
    /equals a\*d \+ b\*c/i,
    /\|det\| tells whether orientation/i,
    /Any shear must change area magnitude/i,
    /det\(AB\) always equals det\(A\) plus det\(B\)/i,
    /similarity transform changes the intrinsic determinant/i,
    /One constant determinant describes every nonlinear map globally/i,
    /simply the norm or size/i,
    /Collapse to a line still preserves full 2D area/i,
    /lesson title without checking/i,
  ];
  const trapPrompt = /false|misleading|wrong|dangerous|unsafe/i;

  assert.deepEqual(quiz.slice(75, 90).map((question) => question.id), trapIds);

  for (const [index, item] of quiz.entries()) {
    const text = `${item.prompt} ${item.choices.join(' ')}`;
    const containsMisconception = misconceptionTerms.some((pattern) => pattern.test(text));
    if (!containsMisconception) continue;

    assert.ok(index >= 75, `${item.id} introduces misconception too early`);
    assert.match(item.prompt, trapPrompt, `${item.id} should mark misconception as a trap`);
  }
});

test('determinant volume assessment avoids visible-page answer leakage', () => {
  const { quiz } = getLessonAssessment('determinant-volume');
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

test('determinant volume assessment distributes correct answer positions per page', () => {
  const { quiz } = getLessonAssessment('determinant-volume');
  const pageSize = 10;

  for (let pageStart = 0; pageStart < quiz.length; pageStart += pageSize) {
    const page = quiz.slice(pageStart, pageStart + pageSize);
    const counts = [0, 0, 0];
    for (const item of page) counts[item.answerIndex] += 1;
    assert.ok(Math.max(...counts) - Math.min(...counts) <= 1, `imbalanced page at ${pageStart + 1}: ${counts.join(',')}`);
  }
});
