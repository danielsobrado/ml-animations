import { createCategoryLessonLabs } from '../lessonLabFactory.js';

export const ADVANCED_MODEL_LESSON_LABS = createCategoryLessonLabs('advanced-models', {
  kind: 'advanced-model pipeline',
  signalName: 'retrieval or multimodal score',
  stages: ['encode', 'retrieve', 'ground'],
  stageExplanation: 'Advanced model systems often encode inputs, retrieve or combine evidence, and check grounding.',
});
