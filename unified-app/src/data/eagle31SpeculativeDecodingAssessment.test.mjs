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

test('eagle 3.1 speculative decoding assessment has 100 curated questions', () => {
  const { quiz, labs } = getLessonAssessment('eagle-3-1-speculative-decoding');

  assert.equal(quiz.length, 100);
  assert.equal(labs.length, 3);
  assert.equal(new Set(quiz.map((item) => item.id)).size, 100);

  for (const [index, item] of quiz.entries()) {
    assert.match(item.id, /^eagle31-\d{3}-[a-z0-9-]+$/);
    assert.equal(item.id.startsWith(`eagle31-${String(index + 1).padStart(3, '0')}`), true, `${item.id} is out of order`);
    assert.equal(LEVELS.has(item.level), true, `${item.id} has unexpected level ${item.level}`);
    assert.equal(item.choices.length, 3, `${item.id} should have three choices`);
    assert.ok(item.answerIndex >= 0 && item.answerIndex < 3, `${item.id} answerIndex out of range`);
    assert.ok(item.prompt.length > 20, `${item.id} prompt too short`);
    assert.ok(item.explanation.length > 30, `${item.id} explanation too short`);
    assert.equal(new Set(item.choices.map(normalize)).size, 3, `${item.id} has duplicate choices`);
  }
});

test('eagle 3.1 speculative decoding assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('eagle-3-1-speculative-decoding');
  const prompts = quiz.map((item) => normalize(item.prompt));
  const answers = quiz.map((item) => normalize(correctAnswer(item)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('eagle 3.1 speculative decoding assessment progresses through lesson learning points', () => {
  const { quiz } = getLessonAssessment('eagle-3-1-speculative-decoding');
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
    ['autoregressive', ['normal autoregressive decoding sequential']],
    ['draft verify', ['basic idea of speculative decoding']],
    ['accepted prefix', ['longest draft-token prefix']],
    ['target hidden states', ['target-model hidden states']],
    ['attention drift', ['What is attention drift']],
    ['FC normalization', ['What does FC normalization stabilize']],
    ['post-norm', ['What does post-norm feedback stabilize']],
    ['mechanism summary', ['Draft verification, hidden-state fusion, drift, normalization, and acceptance payoff']],
    ['mini-eagle labs', ['accepted-prefix lab']],
    ['misconception traps', ['Which statement is false']],
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

test('eagle 3.1 speculative decoding keeps misconception traps after setup', () => {
  const { quiz } = getLessonAssessment('eagle-3-1-speculative-decoding');
  const misconceptionTerms = [
    /accepted blindly without target verification/i,
    /always equals the requested draft depth/i,
    /kept as final output because it was cheap/i,
    /removes the need for target-model verification/i,
    /removes every sequential dependency/i,
    /replaces the target verifier/i,
    /makes every drafted token accepted automatically/i,
    /lower layers never help/i,
    /always a sign of better grounding/i,
    /cannot affect a trained drafter/i,
    /guarantee the same numbers/i,
    /always increases end-to-end speed/i,
    /must be larger than the target model/i,
    /measurements are unnecessary/i,
    /proves exact paper benchmark numbers/i,
  ];
  const trapPrompt = /false|misleading|wrong|dangerous|rejected/i;

  for (const [index, item] of quiz.entries()) {
    const text = `${item.prompt} ${item.choices.join(' ')}`;
    const containsMisconception = misconceptionTerms.some((pattern) => pattern.test(text));
    if (!containsMisconception) continue;

    assert.ok(index >= 75, `${item.id} introduces misconception too early`);
    assert.match(item.prompt, trapPrompt, `${item.id} should mark misconception as a trap`);
  }
});

test('eagle 3.1 speculative decoding assessment avoids visible-page answer leakage', () => {
  const { quiz } = getLessonAssessment('eagle-3-1-speculative-decoding');
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

test('eagle 3.1 speculative decoding assessment distributes correct answer positions per page', () => {
  const { quiz } = getLessonAssessment('eagle-3-1-speculative-decoding');
  const pageSize = 10;

  for (let pageStart = 0; pageStart < quiz.length; pageStart += pageSize) {
    const page = quiz.slice(pageStart, pageStart + pageSize);
    const counts = [0, 0, 0];
    for (const item of page) counts[item.answerIndex] += 1;
    assert.ok(Math.max(...counts) - Math.min(...counts) <= 1, `imbalanced page at ${pageStart + 1}: ${counts.join(',')}`);
  }
});
