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

test('multimodal LLM assessment has 100 curated questions', () => {
  const { quiz, labs } = getLessonAssessment('multimodal-llm');

  assert.equal(quiz.length, 100);
  assert.equal(labs.length, 3);
  assert.deepEqual(labs.map((lab) => lab.id), [
    'trace-multimodal-flow',
    'compare-fusion-designs',
    'ground-scenario-answer',
  ]);
  assert.equal(new Set(quiz.map((item) => item.id)).size, 100);

  for (const [index, item] of quiz.entries()) {
    assert.match(item.id, /^mmlm-\d{3}-[a-z0-9-]+$/);
    assert.equal(item.id.startsWith(`mmlm-${String(index + 1).padStart(3, '0')}`), true, `${item.id} is out of order`);
    assert.equal(LEVELS.has(item.level), true, `${item.id} has unexpected level ${item.level}`);
    assert.equal(item.choices.length, 3, `${item.id} should have three choices`);
    assert.ok(item.answerIndex >= 0 && item.answerIndex < 3, `${item.id} answerIndex out of range`);
    assert.ok(item.prompt.length > 20, `${item.id} prompt too short`);
    assert.ok(item.explanation.length > 30, `${item.id} explanation too short`);
    assert.equal(new Set(item.choices.map(normalize)).size, 3, `${item.id} has duplicate choices`);
  }
});

test('multimodal LLM assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('multimodal-llm');
  const prompts = quiz.map((item) => normalize(item.prompt));
  const answers = quiz.map((item) => normalize(correctAnswer(item)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('multimodal LLM assessment progresses through lesson learning points', () => {
  const { quiz } = getLessonAssessment('multimodal-llm');
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
    ['purpose', ['main purpose of a multimodal LLM']],
    ['modalities', ['text, images, audio, and video']],
    ['shared space', ['shared-representation idea']],
    ['alignment', ['What does alignment mean']],
    ['architecture choices', ['early fusion, late fusion, and cross-attention']],
    ['flow order', ['data-flow order']],
    ['ViT patches', ['ViT-style vision step']],
    ['attention', ['animal attend']],
    ['application grounding', ['model answers "dog"']],
    ['misconception traps', ['Which statement is false about multimodal LLMs']],
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

test('multimodal LLM keeps misconception traps after setup', () => {
  const { quiz } = getLessonAssessment('multimodal-llm');
  const trapIds = [
    'mmlm-076-false-caption-only',
    'mmlm-077-wrong-alignment-optional',
    'mmlm-078-dangerous-projection',
    'mmlm-079-false-late-rich',
    'mmlm-080-wrong-early-cheapest',
    'mmlm-081-false-cross-no-encoder',
    'mmlm-082-wrong-vit-storage',
    'mmlm-083-false-text-pixels',
    'mmlm-084-dangerous-seeing',
    'mmlm-085-false-identical-format',
    'mmlm-086-wrong-more-modalities',
    'mmlm-087-false-model-name',
    'mmlm-088-wrong-fluent-visual',
    'mmlm-089-misleading-architecture-names',
    'mmlm-090-false-single-demo',
  ];
  const misconceptionTerms = [
    /must always convert every image into a caption/i,
    /Alignment is optional/i,
    /Projection is decorative/i,
    /richest early cross-modal interactions/i,
    /always the cheapest/i,
    /removes the need to encode images/i,
    /mainly stores images/i,
    /converted into pixels/i,
    /automatically grounded/i,
    /exact same raw format/i,
    /automatically improves every task/i,
    /model name alone determines/i,
    /fluent generated sentence proves/i,
    /interchangeable labels/i,
    /One successful practice scenario proves/i,
  ];
  const trapPrompt = /false|misleading|wrong|dangerous|rejected|unsafe/i;

  assert.deepEqual(quiz.slice(75, 90).map((item) => item.id), trapIds);

  for (const [index, item] of quiz.entries()) {
    const text = `${item.prompt} ${item.choices.join(' ')}`;
    const containsMisconception = misconceptionTerms.some((pattern) => pattern.test(text));
    if (!containsMisconception) continue;

    assert.ok(index >= 75, `${item.id} introduces misconception too early`);
    assert.match(item.prompt, trapPrompt, `${item.id} should mark misconception as a trap`);
  }
});

test('multimodal LLM assessment avoids visible-page answer leakage', () => {
  const { quiz } = getLessonAssessment('multimodal-llm');
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

test('multimodal LLM assessment distributes correct answer positions per page', () => {
  const { quiz } = getLessonAssessment('multimodal-llm');
  const pageSize = 10;

  for (let pageStart = 0; pageStart < quiz.length; pageStart += pageSize) {
    const page = quiz.slice(pageStart, pageStart + pageSize);
    const counts = [0, 0, 0];
    for (const item of page) counts[item.answerIndex] += 1;
    assert.ok(Math.max(...counts) - Math.min(...counts) <= 1, `imbalanced page at ${pageStart + 1}: ${counts.join(',')}`);
  }
});
