import assert from 'node:assert/strict';
import test from 'node:test';
import { getLessonAssessment } from './lessonAssessments.js';

const LEVELS = new Set(['Foundation', 'Mechanism', 'Application', 'Tricky', 'Interview']);
const EXPECTED_LEVEL_COUNTS = {
  Foundation: 20,
  Mechanism: 30,
  Application: 25,
  Tricky: 15,
  Interview: 10,
};
const EXPECTED_LAB_IDS = [
  'mini-eagle-accepted-prefix',
  'mini-eagle-rms-norm',
  'mini-eagle-drift-sim',
];

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
  assert.deepEqual(labs.map((lab) => lab.id), EXPECTED_LAB_IDS);
  assert.equal(new Set(quiz.map((item) => item.id)).size, 100);

  const levelCounts = Object.fromEntries([...LEVELS].map((level) => [level, 0]));
  const answerPositionCounts = [0, 0, 0];

  for (const [index, item] of quiz.entries()) {
    assert.match(item.id, /^eagle31-\d{3}-[a-z0-9-]+$/);
    assert.equal(item.id.startsWith(`eagle31-${String(index + 1).padStart(3, '0')}`), true, `${item.id} is out of order`);
    assert.equal(LEVELS.has(item.level), true, `${item.id} has unexpected level ${item.level}`);
    levelCounts[item.level] += 1;
    answerPositionCounts[item.answerIndex] += 1;
    assert.equal(item.choices.length, 3, `${item.id} should have three choices`);
    assert.ok(item.answerIndex >= 0 && item.answerIndex < 3, `${item.id} answerIndex out of range`);
    assert.ok(item.prompt.length > 20, `${item.id} prompt too short`);
    assert.ok(item.explanation.length > 30, `${item.id} explanation too short`);
    assert.equal(new Set(item.choices.map(normalize)).size, 3, `${item.id} has duplicate choices`);
  }

  assert.deepEqual(levelCounts, EXPECTED_LEVEL_COUNTS);
  assert.ok(Math.max(...answerPositionCounts) - Math.min(...answerPositionCounts) <= 1, `imbalanced answers: ${answerPositionCounts.join(',')}`);
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
      const surroundingVisibleText = page
        .filter((_, otherOffset) => otherOffset !== offset)
        .map((other) => normalize(`${other.prompt} ${other.choices.join(' ')}`));
      const leaked = surroundingVisibleText.some((text) => text.includes(answers[offset]));
      assert.equal(leaked, false, `${item.id} answer appears elsewhere on same page`);
    }
  }
});

test('eagle 3.1 speculative decoding assessment stays within visible lesson scope', () => {
  const { quiz } = getLessonAssessment('eagle-3-1-speculative-decoding');
  const outOfScopePatterns = [
    /tokenizer for multilingual prompts/i,
    /compressing model weights/i,
    /sorted alphabetically/i,
    /ui animation state machine/i,
    /restarts pretraining/i,
    /gpu colors/i,
    /tokenizer vocabulary/i,
    /optimizer gradients/i,
    /raw character counts/i,
    /second tokenizer/i,
    /unrelated benchmark categories/i,
    /optimizer state/i,
    /changes its tokenizer/i,
    /page border color/i,
    /final answer letter distribution/i,
    /browser route/i,
    /number of prompt templates/i,
    /new set of model weights/i,
    /attention keys into optimizer states/i,
    /random UI label/i,
    /parameter to zero/i,
    /character frequency tables/i,
    /greedy UI labels/i,
    /browser after every click/i,
    /chart decoration/i,
    /benchmark numbers change color/i,
    /labs have been completed/i,
    /model weights were quantized/i,
    /tokenizer single-threaded/i,
    /prevents all matrix multiplication/i,
    /every benchmark on every GPU/i,
    /training loss with no inference measurement/i,
    /source links from the page/i,
    /answer choice order/i,
    /weight quantization/i,
    /color palettes/i,
    /vocabulary sizes across languages/i,
    /training epochs/i,
    /source link titles/i,
    /first answer choice/i,
    /ui reset button/i,
    /source links/i,
    /prompt type selector is broken/i,
    /answer ordering policy/i,
    /title length of each source/i,
    /number of answer choices/i,
    /tokenizer merge rule/i,
    /model weight tensor/i,
    /algorithm name and publication date/i,
    /fastest single prompt/i,
    /layer count is the only metric/i,
    /chart color theme/i,
    /reset button/i,
    /post-norm was spelled/i,
    /total number of learning objectives/i,
    /number of animation panels/i,
    /average prompt character count/i,
    /page title/i,
  ];

  for (const item of quiz) {
    const visibleText = `${item.prompt} ${item.choices.join(' ')}`;
    for (const pattern of outOfScopePatterns) {
      assert.equal(pattern.test(visibleText), false, `${item.id} contains out-of-scope clue ${pattern}`);
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
