import { createCategoryLessonLabs } from '../lessonLabFactory.js';

export const CORE_ML_LESSON_LABS = createCategoryLessonLabs('core-ml', {
  kind: 'machine-learning workflow',
  signalName: 'validation metric',
  stages: ['split', 'train', 'evaluate'],
  stageExplanation: 'Core ML workflows depend on clean data splits, training logic, and honest evaluation.',
});
