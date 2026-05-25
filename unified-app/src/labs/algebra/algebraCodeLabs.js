import { LINEAR_ALGEBRA_CODE_LABS } from './linearAlgebraCodeLabs.js';
import { NEURAL_NETWORK_CODE_LABS } from '../neural-networks/neuralNetworkCodeLabs.js';
import { TRANSFORMER_CODE_LABS } from '../transformers/transformerCodeLabs.js';
import { LANGUAGE_MODEL_CODE_LABS } from '../language-models/languageModelCodeLabs.js';
import { RAG_CODE_LABS } from '../rag/ragCodeLabs.js';
import { EVALUATION_CODE_LABS } from '../evaluation/evaluationCodeLabs.js';
import { EXPERIMENTATION_CODE_LABS } from '../experimentation/experimentationCodeLabs.js';
export {
  ALGEBRA_CODE_LAB_GROUPS_BY_LESSON,
  getAlgebraCodeLabsForLesson,
} from './algebraLessonCodeLabs.js';

export const ALGEBRA_CODE_LABS = [
  ...LINEAR_ALGEBRA_CODE_LABS,
  ...NEURAL_NETWORK_CODE_LABS,
  ...TRANSFORMER_CODE_LABS,
  ...LANGUAGE_MODEL_CODE_LABS,
  ...RAG_CODE_LABS,
  ...EVALUATION_CODE_LABS,
  ...EXPERIMENTATION_CODE_LABS,
];
