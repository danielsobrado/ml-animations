export const MODALITY_PIPELINES = {
  text: {
    label: 'Text',
    raw: 'characters / words',
    encoder: 'tokenizer + embedding',
    tokens: 'text tokens',
    rate: 1,
    risks: ['ambiguity', 'long-context pressure'],
  },
  image: {
    label: 'Image',
    raw: 'pixels',
    encoder: 'vision encoder',
    tokens: 'image patch tokens',
    rate: 9,
    risks: ['spatial loss', 'object hallucination', 'token explosion'],
  },
  video: {
    label: 'Video',
    raw: 'frames',
    encoder: 'frame encoder + temporal position',
    tokens: 'video frame tokens',
    rate: 28,
    risks: ['missed events', 'temporal drift', 'high token cost'],
  },
  audio: {
    label: 'Audio',
    raw: 'waveform',
    encoder: 'audio encoder / codec tokenizer',
    tokens: 'audio tokens or codec units',
    rate: 16,
    risks: ['noise', 'speaker overlap', 'latency'],
  },
};

export const FUSION_MODES = {
  early: {
    label: 'Early Fusion',
    description: 'Tokens from multiple modalities enter a shared backbone early.',
    advantage: 'Deep cross-modal interaction.',
    risk: 'High token and compute cost.',
    interaction: 88,
    latency: 58,
  },
  late: {
    label: 'Late Fusion',
    description: 'Modality-specific encoders produce representations that are combined later.',
    advantage: 'Efficient and modular.',
    risk: 'Weaker deep cross-modal reasoning.',
    interaction: 52,
    latency: 38,
  },
  crossAttention: {
    label: 'Cross-Attention Fusion',
    description: 'One stream queries another modality stream through attention.',
    advantage: 'Controlled modality injection.',
    risk: 'Bottleneck if attention bridge is weak.',
    interaction: 70,
    latency: 48,
  },
  thinkerTalker: {
    label: 'Thinker-Talker',
    description: 'Thinker reasons over inputs; Talker generates speech/audio tokens.',
    advantage: 'Text and speech output with real-time control.',
    risk: 'Synchronization and latency complexity.',
    interaction: 78,
    latency: 65,
  },
};

export const FAILURE_MODES = [
  {
    id: 'modality-neglect',
    label: 'Modality neglect',
    symptom: 'Model answers from text while ignoring image or audio evidence.',
    mitigation: 'Require evidence links and cross-modal attention checks.',
  },
  {
    id: 'hallucinated-grounding',
    label: 'Hallucinated grounding',
    symptom: 'Model refers to an object or region not present in the image.',
    mitigation: 'Ground claims to bounding boxes, heatmaps, or detected regions.',
  },
  {
    id: 'temporal-drift',
    label: 'Temporal drift',
    symptom: 'Model aligns an audio event to the wrong video frame.',
    mitigation: 'Use time-aligned positions and event-level checks.',
  },
  {
    id: 'audio-latency',
    label: 'Audio latency',
    symptom: 'Speech output starts too late for real-time interaction.',
    mitigation: 'Use streaming codec generation and smaller chunks.',
  },
];

export const OMNI_TABS = [
  { id: 'omni-map', label: 'Omni Map' },
  { id: 'vision-projector', label: 'Vision Encoder + Projector' },
  { id: 'fusion', label: 'Early vs Late Fusion' },
  { id: 'grounding', label: 'Image Tokens + Grounding' },
  { id: 'video', label: 'Video Frames + Time' },
  { id: 'audio', label: 'Audio + Codec Tokens' },
  { id: 'thinker-talker', label: 'Thinker-Talker' },
  { id: 'speech', label: 'Speech-to-Speech' },
  { id: 'audio-generation', label: 'Diffusion Audio vs Codec AR' },
  { id: 'reasoning', label: 'Multimodal Reasoning' },
  { id: 'latency', label: 'Real-Time Latency' },
  { id: 'papers', label: 'Paper/Product Decoder' },
];

export const PAPER_CARDS = [
  {
    id: 'llama-4',
    label: 'Llama 4',
    signals: ['Native multimodality', 'Early fusion', 'Text and vision tokens in one backbone', 'Image/video training data', 'Image grounding'],
    interpretation: 'Modern VLMs are moving toward shared model backbones where vision and text interact deeply.',
  },
  {
    id: 'qwen3-omni',
    label: 'Qwen3-Omni',
    signals: ['Text, image, audio, and video', 'Thinker-Talker MoE', 'Streaming speech', 'Discrete speech codec prediction', '234 ms theoretical first-packet latency'],
    interpretation: 'Omni systems combine multimodal reasoning with low-latency speech generation.',
  },
  {
    id: 'qwen25-omni',
    label: 'Qwen2.5-Omni precursor',
    signals: ['Thinker-Talker architecture', 'Block-wise audio/visual processing', 'Interleaved audio/video', 'Time-aligned positions', 'Streaming Talker'],
    interpretation: 'Temporal alignment and streaming are central design issues for omni systems.',
  },
  {
    id: 'llava',
    label: 'LLaVA',
    signals: ['Vision encoder + LLM', 'Projector bridge', 'Visual instruction tuning'],
    interpretation: 'The classic bridge pattern is still the simplest way to teach image-to-language alignment.',
  },
  {
    id: 'qwen35-omni',
    label: 'Qwen3.5-Omni frontier card',
    signals: ['Scaled omni family', 'Hybrid Attention MoE', '256K context', 'Long audio/video understanding', 'Audio-visual grounding'],
    interpretation: 'The newest omni systems are scaling both modality coverage and temporal context.',
  },
];
