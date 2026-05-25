import { LINEAR_ALGEBRA_CODE_LABS } from './linearAlgebraCodeLabs.js';

export const ALGEBRA_CODE_LAB_GROUPS_BY_LESSON = {
  'matrix-multiplication': [
    'Dot product',
    'Matrix cell',
    'Matrix multiplication',
    'Shape compatibility',
  ],
};

export function getAlgebraCodeLabsForLesson(lessonId) {
  const groups = ALGEBRA_CODE_LAB_GROUPS_BY_LESSON[lessonId];

  if (!groups) return null;

  const groupSet = new Set(groups);
  return LINEAR_ALGEBRA_CODE_LABS.filter((exercise) => groupSet.has(exercise.group));
}
