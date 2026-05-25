import { createCategoryLessonLabs } from '../lessonLabFactory.js';

export const NLP_LESSON_LABS = createCategoryLessonLabs('nlp', {
  kind: 'text representation',
  signalName: 'text relevance',
  stages: ['tokenize', 'vectorize', 'compare'],
  stageExplanation: 'NLP code usually has to tokenize text before it can build or compare representations.',
});
