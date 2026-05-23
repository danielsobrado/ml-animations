export const GENERATION_MODES = {
  autoregressive: {
    label: 'Autoregressive',
    shortLabel: 'AR',
    description: 'Generate one token at a time from left to right.',
    advantage: 'Excellent fluency and mature serving stack.',
    risk: 'Sequential decoding and limited native revision.',
    parallelism: 18,
    fluency: 90,
    editability: 34,
  },
  fullDiffusion: {
    label: 'Full-Sequence Diffusion',
    shortLabel: 'Full diffusion',
    description: 'Start with a masked sequence and iteratively refine all positions.',
    advantage: 'Parallel refinement and natural infilling.',
    risk: 'Fixed-length and schedule sensitivity.',
    parallelism: 88,
    fluency: 72,
    editability: 90,
  },
  blockDiffusion: {
    label: 'Block Diffusion',
    shortLabel: 'Block diffusion',
    description: 'Generate sequence block by block while denoising tokens inside each block.',
    advantage: 'Hybrid of flexible length and parallel token sampling.',
    risk: 'Boundary coherence and block schedule design.',
    parallelism: 68,
    fluency: 78,
    editability: 76,
  },
};

export const DIFFUSION_LM_FAILURES = [
  {
    id: 'too-few-steps',
    label: 'Too few denoising steps',
    symptom: 'Output remains incoherent or under-refined.',
    mitigation: 'Increase steps or use confidence-based remasking.',
  },
  {
    id: 'premature-locking',
    label: 'Premature token locking',
    symptom: 'Wrong tokens become fixed too early.',
    mitigation: 'Raise confidence threshold or allow remasking.',
  },
  {
    id: 'revision-instability',
    label: 'Revision instability',
    symptom: 'Tokens oscillate or contradict each other across steps.',
    mitigation: 'Use better schedules and consistency checks.',
  },
  {
    id: 'fixed-length-friction',
    label: 'Fixed-length friction',
    symptom: 'The model must choose output length before knowing the answer shape.',
    mitigation: 'Use block diffusion or adaptive length control.',
  },
  {
    id: 'fluency-lag',
    label: 'Fluency lag',
    symptom: 'Output is diverse but less fluent than AR decoding.',
    mitigation: 'Tune schedule, increase steps, or use AR-converted checkpoints.',
  },
];

export const DIFFUSION_TABS = [
  { id: 'ar-vs-diffusion', label: 'AR vs Diffusion' },
  { id: 'discrete-diffusion', label: 'Discrete Token Diffusion' },
  { id: 'reverse-denoising', label: 'Reverse Denoising' },
  { id: 'parallel-decoding', label: 'Parallel Decoding' },
  { id: 'block-diffusion', label: 'Block Diffusion' },
  { id: 'conversion', label: 'AR-to-Diffusion Conversion' },
  { id: 'alignment', label: 'SFT / DPO Alignment' },
  { id: 'strengths', label: 'Strengths vs AR' },
  { id: 'failures', label: 'Weaknesses and Failure Modes' },
  { id: 'editing', label: 'Editing and Infilling' },
  { id: 'papers', label: 'Paper Decoder' },
];

export const PAPER_CARDS = [
  {
    id: 'llada',
    label: 'LLaDA',
    signals: ['Forward data masking', 'Reverse masked-token prediction', 'Transformer backbone', 'Pretraining + SFT', 'Instruction following after SFT'],
    interpretation: 'A diffusion LM can be trained as a language model, not only as an image generator.',
  },
  {
    id: 'block-diffusion',
    label: 'Block Diffusion',
    signals: ['Interpolates AR and diffusion', 'Block-wise generation', 'Flexible length', 'KV caching', 'Parallel token sampling'],
    interpretation: 'Block diffusion bridges AR length flexibility and diffusion parallelism.',
  },
  {
    id: 'llada2',
    label: 'LLaDA2.0',
    signals: ['100B diffusion LLM', 'Conversion from AR model', 'Three-phase block-level training', 'SFT + DPO alignment', 'MoE variants'],
    interpretation: 'Frontier diffusion LMs can inherit knowledge from AR models and adapt it to denoising.',
  },
  {
    id: 'llada-v',
    label: 'LLaDA-V',
    signals: ['Vision encoder', 'MLP connector', 'Diffusion language model backbone', 'Visual instruction tuning'],
    interpretation: 'Diffusion language models can extend into multimodal instruction-following systems.',
  },
  {
    id: 'llada-tts',
    label: 'LLaDA-TTS',
    signals: ['Masked diffusion for speech tokens', 'Fixed parallel steps', 'AR checkpoint transfer', 'Zero-shot editing'],
    interpretation: 'Masked diffusion also matters for audio token generation where parallel decoding and editing are valuable.',
  },
];

export const SAMPLE_TOKENS = ['Diffusion', 'language', 'models', 'generate', 'text', 'by', 'refining', 'masked', 'tokens', 'in', 'parallel', 'steps'];

