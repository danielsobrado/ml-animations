import React from 'react';
import CodeFixLab from '../code-fix/CodeFixLab';
import { ALGEBRA_CODE_LABS } from './algebraCodeLabs';

export default function AlgebraCodeLab() {
  return <CodeFixLab exercises={ALGEBRA_CODE_LABS} />;
}
