import { allAnimations } from './animations.js';
import { AB_TESTING_FOUNDATIONS_QUIZ } from './abTestingFoundationsAssessment.js';
import { ATTENTION_MASKS_QUIZ } from './attentionMasksAssessment.js';
import { ATTENTION_MECHANISM_QUIZ } from './attentionMechanismAssessment.js';
import { BAYES_RULE_QUIZ } from './bayesRuleAssessment.js';
import { BIAS_VARIANCE_TRADEOFF_QUIZ } from './biasVarianceTradeoffAssessment.js';
import { CALIBRATION_QUIZ } from './calibrationAssessment.js';
import { CAUSAL_GRAPHS_DAGS_QUIZ } from './causalGraphsDagsAssessment.js';
import { CLASSIFIER_FREE_GUIDANCE_QUIZ } from './classifierFreeGuidanceAssessment.js';
import { CLASSIFICATION_METRICS_QUIZ } from './classificationMetricsAssessment.js';
import { COMPUTATION_GRAPH_BACKPROP_QUIZ } from './computationGraphBackpropAssessment.js';
import { CONV2D_QUIZ } from './conv2dAssessment.js';
import { CONV_RELU_QUIZ } from './convReluAssessment.js';
import { COSINE_SIMILARITY_QUIZ } from './cosineSimilarityAssessment.js';
import { CROSS_VALIDATION_QUIZ } from './crossValidationAssessment.js';
import { CUPED_VARIANCE_REDUCTION_QUIZ } from './cupedVarianceReductionAssessment.js';
import { CONFOUNDING_SIMPSONS_PARADOX_QUIZ } from './confoundingSimpsonsParadoxAssessment.js';
import { DATA_ENGINEERING_FOR_ML_QUIZ } from './dataEngineeringForMlAssessment.js';
import { DATA_LEAKAGE_DEEP_DIVE_QUIZ } from './dataLeakageDeepDiveAssessment.js';
import { DIFFUSION_BASICS_QUIZ } from './diffusionBasicsAssessment.js';
import { DIFFUSION_SAMPLING_QUIZ } from './diffusionSamplingAssessment.js';
import { DROPOUT_BATCHNORM_QUIZ } from './dropoutBatchnormAssessment.js';
import { EFFICIENT_INFERENCE_COMPRESSION_QUIZ } from './efficientInferenceCompressionAssessment.js';
import { EMBEDDINGS_QUIZ } from './embeddingsAssessment.js';
import { FEATURE_SCALING_PREPROCESSING_QUIZ } from './featureScalingPreprocessingAssessment.js';
import { FINE_TUNING_QUIZ } from './fineTuningAssessment.js';
import { FLASH_ATTENTION_QUIZ } from './flashAttentionAssessment.js';
import { FUNDAMENTAL_SUBSPACES_QUIZ } from './fundamentalSubspacesAssessment.js';
import { GRADIENT_DESCENT_QUIZ } from './gradientDescentAssessment.js';
import { GRADIENT_PROBLEMS_QUIZ } from './gradientProblemsAssessment.js';
import { GROUPED_QUERY_ATTENTION_QUIZ } from './groupedQueryAttentionAssessment.js';
import { HYPOTHESIS_TESTING_INTUITION_QUIZ } from './hypothesisTestingIntuitionAssessment.js';
import { INITIALIZATION_QUIZ } from './initializationAssessment.js';
import { KMEANS_QUIZ } from './kMeansAssessment.js';
import { KNN_NAIVE_BAYES_SVM_QUIZ } from './knnNaiveBayesSvmAssessment.js';
import { KV_CACHE_QUIZ } from './kvCacheAssessment.js';
import { LAYER_NORMALIZATION_QUIZ } from './layerNormalizationAssessment.js';
import { LEAKY_RELU_QUIZ } from './leakyReluAssessment.js';
import { LINEAR_REGRESSION_QUIZ } from './linearRegressionAssessment.js';
import { LOGISTIC_REGRESSION_QUIZ } from './logisticRegressionAssessment.js';
import { LLM_TRAINING_OBJECTIVES_QUIZ } from './llmTrainingObjectivesAssessment.js';
import { LOSS_FUNCTIONS_LIKELIHOODS_QUIZ } from './lossFunctionsLikelihoodsAssessment.js';
import { MAXIMUM_LIKELIHOOD_ESTIMATION_QUIZ } from './maximumLikelihoodEstimationAssessment.js';
import { MAX_POOLING_QUIZ } from './maxPoolingAssessment.js';
import { MATRIX_DECOMPOSITIONS_QUIZ } from './matrixDecompositionsAssessment.js';
import { MATRIX_MULTIPLICATION_QUIZ } from './matrixMultiplicationAssessment.js';
import { ML_SECURITY_ROBUSTNESS_QUIZ } from './mlSecurityRobustnessAssessment.js';
import { NATIVE_SPARSE_ATTENTION_QUIZ } from './nativeSparseAttentionAssessment.js';
import { NEURAL_NETWORK_QUIZ } from './neuralNetworkAssessment.js';
import { OPTIMIZERS_QUIZ } from './optimizersAssessment.js';
import { OVERFITTING_QUIZ } from './overfittingAssessment.js';
import { PCA_QUIZ } from './pcaAssessment.js';
import { POSITIONAL_ENCODING_QUIZ } from './positionalEncodingAssessment.js';
import { POWER_SAMPLE_SIZE_QUIZ } from './powerSampleSizeAssessment.js';
import { PROPENSITY_SCORES_QUIZ } from './propensityScoresAssessment.js';
import { QR_DECOMPOSITION_QUIZ } from './qrDecompositionAssessment.js';
import { RAG_CHUNKING_CONTEXT_QUIZ } from './ragChunkingContextAssessment.js';
import { RAG_FAILURE_MODES_QUIZ } from './ragFailureModesAssessment.js';
import { RAG_RERANKING_GROUNDING_QUIZ } from './ragRerankingGroundingAssessment.js';
import { RAG_RETRIEVAL_EVALUATION_QUIZ } from './ragRetrievalEvaluationAssessment.js';
import { RAG_VECTOR_INDEXING_QUIZ } from './ragVectorIndexingAssessment.js';
import { RECOMMENDER_SYSTEMS_RANKING_QUIZ } from './recommenderSystemsRankingAssessment.js';
import { REGULARIZATION_QUIZ } from './regularizationAssessment.js';
import { RELU_QUIZ } from './reluAssessment.js';
import { ROPE_QUIZ } from './ropeAssessment.js';
import { ROC_PR_CURVES_QUIZ } from './rocPrCurvesAssessment.js';
import { RESIDUAL_STREAM_QUIZ } from './residualStreamAssessment.js';
import { SAMPLING_CONFIDENCE_INTERVALS_QUIZ } from './samplingConfidenceIntervalsAssessment.js';
import { SAMPLING_STRATEGIES_QUIZ } from './samplingStrategiesAssessment.js';
import { SEQUENTIAL_TESTING_PEEKING_QUIZ } from './sequentialTestingPeekingAssessment.js';
import { SELF_ATTENTION_QUIZ } from './selfAttentionAssessment.js';
import { SPEARMAN_CORRELATION_QUIZ } from './spearmanCorrelationAssessment.js';
import { getScenarioQuestionsForLesson } from './scenarioQuestions.js';
import { SVD_QUIZ } from './svdAssessment.js';
import { TREE_ENSEMBLES_QUIZ } from './treeEnsemblesAssessment.js';
import { TIME_SERIES_FORECASTING_QUIZ } from './timeSeriesForecastingAssessment.js';
import { TOKENIZATION_QUIZ } from './tokenizationAssessment.js';
import { TRAINING_LOOP_DYNAMICS_QUIZ } from './trainingLoopDynamicsAssessment.js';
import { TRAIN_VALIDATION_TEST_SPLIT_QUIZ } from './trainValidationTestSplitAssessment.js';
import { TRANSFORMER_ARCHITECTURE_FAMILIES_QUIZ } from './transformerArchitectureFamiliesAssessment.js';
import { TRANSFORMER_QUIZ } from './transformerAssessment.js';
import { TRANSFORMER_TOKEN_GENERATION_QUIZ } from './transformerTokenGenerationAssessment.js';
import { TREATMENT_EFFECTS_QUIZ } from './treatmentEffectsAssessment.js';
import { UNET_VS_DIT_QUIZ } from './unetVsDitAssessment.js';

export const PRIORITY_ASSESSMENT_LESSON_IDS = [
  'matrix-multiplication',
  'linear-regression',
  'pca',
  'fundamental-subspaces',
  'matrix-decompositions',
  'qr-decomposition',
  'svd',
  'k-means',
  'train-validation-test-split',
  'cross-validation',
  'data-leakage-deep-dive',
  'feature-scaling-preprocessing',
  'bayes-rule-ml',
  'sampling-confidence-intervals',
  'hypothesis-testing-intuition',
  'ab-testing-foundations',
  'power-sample-size',
  'sequential-testing-peeking',
  'cuped-variance-reduction',
  'confounding-simpsons-paradox',
  'causal-graphs-dags',
  'treatment-effects',
  'propensity-scores',
  'spearman-correlation',
  'maximum-likelihood-estimation',
  'loss-functions-likelihoods',
  'logistic-regression',
  'classification-metrics',
  'roc-pr-curves',
  'calibration',
  'overfitting',
  'bias-variance-tradeoff',
  'regularization',
  'knn-naive-bayes-svm',
  'tree-ensembles',
  'time-series-forecasting-track',
  'recommender-systems-ranking-track',
  'ml-security-robustness-track',
  'efficient-inference-compression-track',
  'data-engineering-for-ml-track',
  'gradient-descent',
  'neural-network',
  'initialization',
  'optimizers',
  'training-loop-dynamics',
  'dropout-batchnorm',
  'gradient-problems',
  'layer-normalization',
  'relu',
  'leaky-relu',
  'conv2d',
  'conv-relu',
  'max-pooling',
  'computation-graph-backprop',
  'tokenization',
  'embeddings',
  'cosine-similarity',
  'attention-mechanism',
  'self-attention',
  'kv-cache',
  'grouped-query-attention',
  'flash-attention',
  'native-sparse-attention',
  'attention-masks',
  'positional-encoding',
  'rope',
  'residual-stream',
  'transformer',
  'transformer-architecture-families',
  'llm-training-objectives',
  'transformer-token-generation',
  'sampling-strategies',
  'fine-tuning',
  'rag-chunking-context',
  'rag-vector-indexing',
  'rag-reranking-grounding',
  'rag-failure-modes',
  'rag-retrieval-evaluation',
  'diffusion-basics',
  'diffusion-sampling',
  'classifier-free-guidance',
  'unet-vs-dit',
  'q-learning',
  'rl-exploration',
  'rl-foundations',
  'grpo-reasoning',
  'dapo-reasoning-rl',
  'coconut-latent-reasoning',
  'bloom-filter',
  'reasoning-rlvr-grpo',
  'test-time-compute-thinking-budgets',
  'long-context-frontier-models',
  'omni-multimodal-architectures',
  'diffusion-language-models',
  'efficient-llm-serving',
  'tool-using-reasoning-models',
  'agentic-coding-systems',
  'frontier-evaluation-safety',
];

export const EMPTY_ASSESSMENT = Object.freeze({
  quiz: Object.freeze([]),
  labs: Object.freeze([]),
});

function causalAssessment(topic, coreIdea, failureMode, labPrompt) {
  return {
    quiz: [
      {
        id: `${topic}-core-purpose`,
        prompt: `What is the main purpose of ${coreIdea.name}?`,
        choices: [
          coreIdea.purpose,
          'To replace randomization with a larger dashboard',
          'To prove every observed association is causal',
        ],
        answerIndex: 0,
        explanation: coreIdea.purposeExplanation,
      },
      {
        id: `${topic}-failure-mode`,
        prompt: `Which mistake is most dangerous for ${coreIdea.name}?`,
        choices: [
          'Using the method before defining the decision metric',
          failureMode,
          'Writing down the assumptions before looking at outcomes',
        ],
        answerIndex: 1,
        explanation: coreIdea.failureExplanation,
      },
      {
        id: `${topic}-decision-use`,
        prompt: `How should ${coreIdea.name} affect the decision?`,
        choices: [
          'Ignore uncertainty and ship the highest observed metric',
          'Use it only after deployment',
          coreIdea.decisionUse,
        ],
        answerIndex: 2,
        explanation: coreIdea.decisionExplanation,
      },
      {
        id: `${topic}-assumption`,
        prompt: `Which assumption matters most here?`,
        choices: [
          coreIdea.assumption,
          'Every subgroup has identical outcomes',
          'The observed result is independent of study design',
        ],
        answerIndex: 0,
        explanation: coreIdea.assumptionExplanation,
      },
      {
        id: `${topic}-diagnostic`,
        prompt: `What diagnostic should you check for ${coreIdea.name}?`,
        choices: [
          'Only the final headline metric',
          coreIdea.diagnostic,
          'Whether the chart color matches the treatment label',
        ],
        answerIndex: 1,
        explanation: coreIdea.diagnosticExplanation,
      },
    ],
    labs: [
      {
        id: `${topic}-scenario-lab`,
        title: coreIdea.labTitle,
        prompt: labPrompt,
        successCriteria: coreIdea.successCriteria,
      },
    ],
  };
}

const SEEDED_LESSON_ASSESSMENTS = {
  'matrix-multiplication': {
    quiz: MATRIX_MULTIPLICATION_QUIZ,
    labs: [
      {
        id: 'compute-one-cell',
        title: 'Compute one cell by hand',
        prompt: 'Pick one highlighted output cell, write the row-column dot product, then compare it with the animation.',
        successCriteria: 'Your written terms match the multiplied row and column entries in order.',
      },
    ],
  },
  'eagle-3-1-speculative-decoding': {
    quiz: [
      {
        id: 'eagle-autoregressive-bottleneck',
        level: 'Foundation',
        prompt: 'Why is normal autoregressive LLM decoding hard to parallelize across future tokens?',
        choices: [
          'Each next token depends on the previously generated tokens',
          'The tokenizer can only run on one CPU core',
          'The KV cache removes all sequential dependencies',
        ],
        answerIndex: 0,
        explanation: 'The target model needs the current generated prefix before it can produce the next token.',
      },
      {
        id: 'eagle-speculative-parallelizes',
        level: 'Foundation',
        prompt: 'What does speculative decoding try to parallelize?',
        choices: [
          'It drafts several possible future tokens cheaply, then verifies them together with the target model',
          'It lets the small drafter skip target-model verification',
          'It trains all transformer layers at the same time during inference',
        ],
        answerIndex: 0,
        explanation: 'The expensive target pass checks a proposed token block instead of generating only one token.',
      },
      {
        id: 'eagle-accepted-prefix',
        level: 'Foundation',
        prompt: 'What is accepted during one speculative decoding round?',
        choices: [
          'The longest draft-token prefix that passes target-model verification',
          'Every token proposed by the drafter',
          'Only the final token in the draft sequence',
        ],
        answerIndex: 0,
        explanation: 'A first rejection stops the accepted prefix; remaining draft tokens are discarded.',
      },
      {
        id: 'eagle-rejection-fallback',
        level: 'Mechanism',
        prompt: 'What happens after the first rejected draft token?',
        choices: [
          'The target model supplies a replacement token and the remaining draft tokens are discarded',
          'The rejected token is kept because it was cheaper to produce',
          'The model restarts training from the original prompt',
        ],
        answerIndex: 0,
        explanation: 'Fallback preserves target-model behavior while still benefiting from accepted draft prefixes.',
      },
      {
        id: 'eagle-acceptance-length-speed',
        level: 'Mechanism',
        prompt: 'Why does a better drafter usually produce more speedup?',
        choices: [
          'It increases the average accepted prefix length per target verification pass',
          'It makes the target model unnecessary',
          'It removes attention from the target model',
        ],
        answerIndex: 0,
        explanation: 'Speedup comes from keeping more tokens from each verified draft block.',
      },
      {
        id: 'eagle-quality-preservation',
        level: 'Mechanism',
        prompt: 'Why can speculative decoding preserve output quality?',
        choices: [
          'The target model verifies proposed tokens before they become final output',
          'The draft model is always larger than the target model',
          'Rejected tokens are hidden but still used as final tokens',
        ],
        answerIndex: 0,
        explanation: 'The target model remains the authority for accepted tokens and fallback tokens.',
      },
      {
        id: 'eagle-long-context-hard',
        level: 'Application',
        prompt: 'Why can long context or changed chat templates make EAGLE-style drafting harder?',
        choices: [
          'They can shift attention patterns and hidden-state distributions away from what the drafter expects',
          'They prevent the target model from verifying draft tokens',
          'They make acceptance length unrelated to speedup',
        ],
        answerIndex: 0,
        explanation: 'Production prompts stress the drafter with formatting, sink-token, and distribution changes.',
      },
      {
        id: 'eagle-attention-drift-definition',
        level: 'Mechanism',
        prompt: 'What is attention drift in this lesson?',
        choices: [
          'The drafter shifts attention away from prompt or sink positions toward recent draft tokens as depth increases',
          'The target model deletes the KV cache after every token',
          'The user prompt becomes shorter during generation',
        ],
        answerIndex: 0,
        explanation: 'The drafter starts conditioning on its own speculative chain instead of stable context anchors.',
      },
      {
        id: 'eagle-self-attention-risk',
        level: 'Application',
        prompt: 'Why is it risky when the drafter mostly watches its own recent draft tokens?',
        choices: [
          'Draft states are less authoritative than verifier-conditioned context, so errors can compound',
          'Recent draft tokens are always correct and should dominate',
          'Attention to the prompt is illegal during speculative decoding',
        ],
        answerIndex: 0,
        explanation: 'The drafter can become increasingly self-referential and less aligned with the target model.',
      },
      {
        id: 'eagle-fc-norm-fix',
        level: 'Mechanism',
        prompt: 'What does FC normalization fix in EAGLE 3.1?',
        choices: [
          'It stabilizes target hidden-state stream scales before fusion so one stream does not dominate by magnitude',
          'It replaces target-model verification with greedy decoding',
          'It removes the need for low- and middle-layer features',
        ],
        answerIndex: 0,
        explanation: 'Balanced feature streams let fusion combine semantic levels instead of obeying only the largest-magnitude stream.',
      },
      {
        id: 'eagle-post-norm-fix',
        level: 'Mechanism',
        prompt: 'What does post-norm hidden-state feedback fix?',
        choices: [
          'It prevents hidden-state magnitude from accumulating across recursive draft steps',
          'It makes every draft token accepted automatically',
          'It turns the drafter into the full target model',
        ],
        answerIndex: 0,
        explanation: 'Each recursive drafter call receives a representation closer to the scale it was trained to consume.',
      },
      {
        id: 'eagle-depth-prediction',
        level: 'Application',
        prompt: 'If speculation depth increases from 2 to 8 and the drafter is unstable, what should you expect?',
        choices: [
          'Later draft tokens become less reliable and acceptance length may drop',
          'Acceptance length must equal 8',
          'Target verification stops being needed',
        ],
        answerIndex: 0,
        explanation: 'Deep speculative chains expose drift and residual-scale accumulation more strongly.',
      },
      {
        id: 'eagle-heatmap-sign',
        level: 'Application',
        prompt: 'What visual sign indicates attention drift in the heatmap?',
        choices: [
          'Bright cells move from prompt and sink columns toward recent draft-token columns',
          'All cells disappear at deeper rows',
          'Only the benchmark panel changes color',
        ],
        answerIndex: 0,
        explanation: 'The center of attention moves from stable context anchors to speculative self-history.',
      },
      {
        id: 'eagle-rms-sign',
        level: 'Application',
        prompt: 'What visual sign indicates residual-scale accumulation?',
        choices: [
          'Hidden-state RMS grows monotonically with draft depth',
          'Accepted tokens are colored green',
          'The prompt type selector has four choices',
        ],
        answerIndex: 0,
        explanation: 'Growing RMS means later recursive steps receive larger-scale states.',
      },
      {
        id: 'eagle-post-norm-prediction',
        level: 'Application',
        prompt: 'What should change when post-norm feedback is enabled?',
        choices: [
          'Hidden-state RMS should become flatter across depth and acceptance length should be more stable',
          'The target model should stop checking draft tokens',
          'The prompt should disappear from the heatmap',
        ],
        answerIndex: 0,
        explanation: 'Post-norm stabilizes the recursive feedback scale.',
      },
      {
        id: 'eagle-template-fragility',
        level: 'Tricky',
        prompt: 'Why might EAGLE-3 work on short controlled prompts but degrade with changed chat templates?',
        choices: [
          'The drafter may learn brittle hidden-state and attention patterns tied to familiar prompt formatting',
          'Chat templates make the target model unable to run',
          'Speculative decoding requires empty prompts',
        ],
        answerIndex: 0,
        explanation: 'A shifted prompt format can change the hidden-state distribution the drafter receives.',
      },
      {
        id: 'eagle-high-layer-domination',
        level: 'Tricky',
        prompt: 'Why is higher-layer hidden-state domination during fusion a problem?',
        choices: [
          'Fusion should combine multiple semantic levels, not just the stream with the largest scale',
          'High-layer states are never useful',
          'Low-layer and middle-layer states must always be larger',
        ],
        answerIndex: 0,
        explanation: 'EAGLE-3 benefits from low-, middle-, and high-level features only if fusion can actually use them.',
      },
      {
        id: 'eagle-recursive-drafter',
        level: 'Tricky',
        prompt: 'Why is EAGLE 3.1 more like repeatedly calling the same drafter?',
        choices: [
          'Post-norm feedback makes each recursive step receive a similarly scaled hidden representation',
          'It deletes all hidden states between steps',
          'It makes the target model smaller',
        ],
        answerIndex: 0,
        explanation: 'Stable feedback scale avoids the interpretation that each step simply stacks another unnormalized layer.',
      },
      {
        id: 'eagle-target-size',
        level: 'Interview',
        prompt: 'Does EAGLE 3.1 make the target model smaller?',
        choices: [
          'No. It reduces effective decoding cost by improving draft acceptance, while the target model still verifies',
          'Yes. It removes half of the target transformer layers',
          'Yes. It replaces the target model with RMSNorm',
        ],
        answerIndex: 0,
        explanation: 'The method improves serving efficiency through speculative acceptance, not target-model compression.',
      },
      {
        id: 'eagle-metric-emphasis',
        level: 'Interview',
        prompt: 'What metric should this animation emphasize besides raw speedup?',
        choices: [
          'Acceptance length, because it explains where speedup comes from',
          'The number of colors in the heatmap',
          'The length of the paper title',
        ],
        answerIndex: 0,
        explanation: 'Acceptance length links the mechanism to throughput.',
      },
    ],
    labs: [
      {
        id: 'mini-eagle-accepted-prefix',
        title: 'Repair accepted-prefix logic',
        prompt: 'Open mini-eagle/exercises/01_accept_prefix.rs and make the tests pass by stopping at the first mismatch.',
        successCriteria: 'The accepted-prefix tests pass and you can explain why accepted length, not draft length, controls speedup.',
      },
      {
        id: 'mini-eagle-rms-norm',
        title: 'Stabilize a toy drafter',
        prompt: 'Complete the RMSNorm and post-norm feedback exercises in mini-eagle, then compare raw and normalized hidden-state RMS.',
        successCriteria: 'The normed path remains near RMS 1.0 after repeated draft steps.',
      },
      {
        id: 'mini-eagle-drift-sim',
        title: 'Measure drift and expected acceptance',
        prompt: 'Complete the attention drift score and acceptance-length simulation exercises.',
        successCriteria: 'You can connect rising recent-draft attention to shorter expected accepted prefixes.',
      },
    ],
  },
  'multi-head-latent-attention': {
    quiz: [
      {
        id: 'mla-kv-cache-growth',
        level: 'Foundation',
        prompt: 'Why does the KV cache grow during autoregressive decoding?',
        choices: [
          'Every new token stores key and value information that future tokens will read',
          'The tokenizer creates a larger vocabulary after each step',
          'The model adds a new transformer layer for each generated token',
        ],
        answerIndex: 0,
        explanation: 'The prefix grows one token at a time, and each layer keeps cached attention state for those prior tokens.',
      },
      {
        id: 'mla-gqa-cache-saving',
        level: 'Foundation',
        prompt: 'What does GQA do to reduce KV cache size?',
        choices: [
          'It lets multiple query heads share a smaller number of key/value heads',
          'It deletes value vectors and keeps only keys',
          'It replaces attention with an MLP during decoding',
        ],
        answerIndex: 0,
        explanation: 'GQA keeps many query heads but caches fewer K/V heads, so the cache width is smaller than MHA.',
      },
      {
        id: 'mla-cached-object',
        level: 'Foundation',
        prompt: 'What does MLA cache instead of full keys and values for every head?',
        choices: [
          'A compressed latent KV vector, plus a small positional key path in the decoupled RoPE design',
          'Only the final output logits for each token',
          'A copy of every query vector from previous tokens',
        ],
        answerIndex: 0,
        explanation: 'MLA changes the attention architecture so the cached content state is a latent bottleneck rather than expanded K/V heads.',
      },
      {
        id: 'mla-not-quantization',
        level: 'Foundation',
        prompt: 'Why is MLA different from ordinary KV-cache quantization?',
        choices: [
          'Quantization stores each number with fewer bits; MLA changes the architecture so fewer cached numbers exist',
          'MLA only changes the tokenizer precision',
          'Quantization and MLA are identical names for the same operation',
        ],
        answerIndex: 0,
        explanation: 'The important distinction is architectural compression versus lower-bit storage of the same tensor layout.',
      },
      {
        id: 'mla-gqa-expressiveness-risk',
        level: 'Mechanism',
        prompt: 'Why can GQA reduce head expressiveness?',
        choices: [
          'Heads inside a group share the same K/V source, so their key/value views are less independent',
          'GQA prevents query heads from using softmax',
          'GQA forces every layer to use the same residual stream',
        ],
        answerIndex: 0,
        explanation: 'Sharing saves memory, but it also ties multiple query heads to the same cached K/V representation.',
      },
      {
        id: 'mla-up-projection-expressiveness',
        level: 'Mechanism',
        prompt: 'How does MLA restore some per-head expressiveness?',
        choices: [
          'It up-projects the cached latent state through learned matrices into richer head-specific behavior',
          'It stores a separate full notebook for each head after all',
          'It removes all positional information from attention',
        ],
        answerIndex: 0,
        explanation: 'The low-rank latent is compact, but learned up-projections can create different effective views for different heads.',
      },
      {
        id: 'mla-core-tradeoff',
        level: 'Mechanism',
        prompt: 'What is the central MLA tradeoff?',
        choices: [
          'Lower KV-cache memory and bandwidth, with extra projection compute and implementation complexity',
          'Higher cache memory in exchange for deleting matrix multiplication',
          'Lower latency because no projections or cache reads remain',
        ],
        answerIndex: 0,
        explanation: 'MLA is a memory-bandwidth optimization that still has computational and kernel-design costs.',
      },
      {
        id: 'mla-absorption',
        level: 'Mechanism',
        prompt: 'What does projection absorption try to avoid?',
        choices: [
          'Materializing full expanded keys and values for every cached token',
          'Computing queries for the current token',
          'Storing the compressed latent content cache',
        ],
        answerIndex: 0,
        explanation: 'Linear maps can be regrouped so attention works against the latent cache rather than expanding every cached token first.',
      },
      {
        id: 'mla-rope-problem',
        level: 'Mechanism',
        prompt: 'Why does RoPE cause trouble for naive MLA absorption?',
        choices: [
          'RoPE is position-sensitive and can sit between linear maps that would otherwise be regrouped',
          'RoPE deletes the value vector before attention',
          'RoPE makes every head share a single query vector',
        ],
        answerIndex: 0,
        explanation: 'A position-dependent rotation does not freely commute with arbitrary projection matrices.',
      },
      {
        id: 'mla-decoupled-rope',
        level: 'Mechanism',
        prompt: 'What does decoupled RoPE do in MLA?',
        choices: [
          'It separates compressed content memory from a small positional RoPE key path',
          'It removes positions from all keys and queries',
          'It turns RoPE into a tokenizer rule',
        ],
        answerIndex: 0,
        explanation: 'DeepSeek-style MLA keeps content in the latent cache while positional information travels through a small separate path.',
      },
      {
        id: 'transmla-theorem',
        level: 'Application',
        prompt: 'What is TransMLA central expressiveness claim?',
        choices: [
          'At the same KV-cache overhead, MLA can represent GQA, but GQA cannot represent every MLA configuration',
          'GQA can represent all MLA configurations with no conversion',
          'MLA and GQA have identical parameter families under every rank',
        ],
        answerIndex: 0,
        explanation: 'The paper frames GQA as a special repeated structure that MLA can express through low-rank factorization.',
      },
      {
        id: 'transmla-repetition-factorization',
        level: 'Application',
        prompt: 'What repeated structure does TransMLA exploit?',
        choices: [
          'GQA repeats shared K/V heads across groups of query heads',
          'MHA repeats every query vector across all previous tokens',
          'RoPE repeats the same absolute position for every token',
        ],
        answerIndex: 0,
        explanation: 'The repeated GQA head structure can be moved into parameters and then factorized into a latent representation.',
      },
      {
        id: 'mla-bandwidth-long-context',
        level: 'Application',
        prompt: 'Why can memory bandwidth matter more than raw FLOPs during long-context decoding?',
        choices: [
          'Each generated token repeatedly reads a large KV cache from memory',
          'Long prompts prevent GPUs from doing matrix multiplication',
          'FLOPs are only used during training and never during inference',
        ],
        answerIndex: 0,
        explanation: 'Decode steps are sequential and must revisit the prefix cache, so cache traffic can dominate.',
      },
      {
        id: 'mla-same-cache-not-same-expressiveness',
        level: 'Tricky',
        prompt: 'Why does same KV-cache size not imply same expressiveness?',
        choices: [
          'Architectures can store the same number of cached values but transform them into attention behavior differently',
          'Cache size fully determines every model parameter',
          'Expressiveness only depends on the number of tokens in the prompt',
        ],
        answerIndex: 0,
        explanation: 'The cached width is only one constraint; the parameterization around that cache also matters.',
      },
      {
        id: 'transmla-practical-use',
        level: 'Interview',
        prompt: 'Why is TransMLA practically useful?',
        choices: [
          'It suggests a migration path for existing GQA models without training a fresh MLA model from scratch',
          'It proves KV cache memory no longer matters',
          'It removes the need to evaluate converted checkpoints',
        ],
        answerIndex: 0,
        explanation: 'A post-training conversion path is useful only if it preserves quality under evaluation while improving serving economics.',
      },
    ],
    labs: [
      {
        id: 'mini-mla-cache-sizing',
        title: 'Compute latent cache width',
        prompt: 'Open mini-mla/exercises/01_kv_cache_size.rs and implement the MHA, GQA, and MLA cache-size formulas.',
        successCriteria: 'The tests pass and you can explain why MLA uses latent_dim + rope_dim instead of 2 x heads x head_dim.',
      },
      {
        id: 'mini-mla-absorption-rope',
        title: 'Verify absorption and RoPE non-commutation',
        prompt: 'Complete mini-mla exercises 05 and 06, then explain why projection absorption works for adjacent linear maps but not through a position-dependent rotation.',
        successCriteria: 'The absorption equivalence test passes, and the rotation/scaling order test shows a nonzero difference.',
      },
      {
        id: 'mini-mla-transmla-factorization',
        title: 'Trace GQA repetition into MLA intuition',
        prompt: 'Complete mini-mla exercises 08 and 09, then describe how repeated GQA rows become a low-rank structure that TransMLA can factorize.',
        successCriteria: 'The repeated-head output and distinct-row count match the expected low-diversity structure.',
      },
    ],
  },
  'native-sparse-attention': {
    quiz: NATIVE_SPARSE_ATTENTION_QUIZ,
    labs: [
      {
        id: 'mini-nsa-block-window',
        title: 'Implement blocks and local windows',
        prompt: 'Open mini-native-sparse-attention exercises 01 and 02, then implement contiguous block ranges and sliding-window indices.',
        successCriteria: 'The block and window tests pass, and you can explain why NSA starts from contiguous regions.',
      },
      {
        id: 'mini-nsa-score-select',
        title: 'Score the map and select blocks',
        prompt: 'Complete exercises 03 through 06 to compress blocks, score compressed keys, choose top-k blocks, and expand them to token indices.',
        successCriteria: 'The tests pass and you can trace map scanning into fine-grained selection.',
      },
      {
        id: 'mini-nsa-gates-hardware',
        title: 'Merge branches and estimate memory',
        prompt: 'Complete exercises 07 through 10, then compare full attention tokens loaded with the toy NSA budget.',
        successCriteria: 'The gate, GQA sharing, memory estimate, and sparse-budget tests pass.',
      },
    ],
  },
  'spec-sparse-attention': {
    quiz: [
      {
        id: 'spec-sparse-two-bottlenecks',
        level: 'Foundation',
        prompt: 'What are the two bottlenecks this lesson combines?',
        choices: [
          'Speculative decoding reduces target-model passes, and sparse attention reduces KV entries read per pass',
          'Sparse attention removes tokenization, and speculative decoding removes the target model',
          'Flash Attention changes the vocabulary, and KV cache removes causal masking',
        ],
        answerIndex: 0,
        explanation: 'The combined goal is more accepted tokens per verification round while loading fewer KV blocks per attention computation.',
      },
      {
        id: 'spec-sparse-long-context',
        level: 'Foundation',
        prompt: 'Why does sparse attention help more in long-context decoding?',
        choices: [
          'KV-cache access grows with context length, so skipping irrelevant entries saves more memory traffic',
          'Long prompts make the tokenizer produce fewer tokens',
          'Long-context models do not need attention heads',
        ],
        answerIndex: 0,
        explanation: 'The memory cost of reading cached keys and values becomes a larger part of decode latency as the prefix grows.',
      },
      {
        id: 'spec-sparse-accepted-prefix',
        level: 'Foundation',
        prompt: 'What does speculative decoding accept?',
        choices: [
          'The longest draft-token prefix that passes target-model verification',
          'Every token proposed by the sparse drafter',
          'Only the token with the largest attention logit',
        ],
        answerIndex: 0,
        explanation: 'A first rejection stops the accepted prefix; later draft tokens are discarded or replaced by the target.',
      },
      {
        id: 'specattn-drafter-model',
        level: 'Mechanism',
        prompt: 'In SpecAttn, what model drafts the tokens?',
        choices: [
          'The original target model itself, run with sparse attention during drafting',
          'A separate retrained verifier model',
          'A tokenizer-only assistant',
        ],
        answerIndex: 0,
        explanation: 'SpecAttn is self-speculative: sparse attention is used for drafting, while full attention verifies final behavior.',
      },
      {
        id: 'specattn-full-verification-quality',
        level: 'Mechanism',
        prompt: 'Why does full-attention verification preserve quality in SpecAttn?',
        choices: [
          'Final tokens are still verified by the full target model before they are accepted',
          'Sparse attention is guaranteed to match full attention for every query',
          'The draft tokens are always sampled greedily',
        ],
        answerIndex: 0,
        explanation: 'Sparse drafting is a proposal mechanism; the target distribution is protected by full-attention verification and fallback.',
      },
      {
        id: 'specattn-verification-extra-signal',
        level: 'Mechanism',
        prompt: 'What extra information does SpecAttn get from verification besides accept or reject?',
        choices: [
          'Full-attention logits or weights that reveal which KV entries mattered',
          'The exact GPU temperature for every block',
          'A new tokenizer vocabulary',
        ],
        answerIndex: 0,
        explanation: 'SpecAttn reuses attention evidence from the verification pass as a map for the next sparse drafting pass.',
      },
      {
        id: 'specattn-last-token-brittle',
        level: 'Application',
        prompt: 'Why is selecting KV entries from only the last accepted token brittle?',
        choices: [
          'It can overfit to one position attention pattern and hurt later draft-token acceptance',
          'It always selects every KV entry, making sparse attention dense',
          'It prevents the target model from computing logits',
        ],
        answerIndex: 0,
        explanation: 'A sparse set that explains one verifier query may not cover the diversity needed across a multi-token draft chain.',
      },
      {
        id: 'specattn-collect-two-query',
        level: 'Mechanism',
        prompt: 'Why collect logits from the first draft token and bonus token?',
        choices: [
          'They are far apart in the draft chain, so together they capture more diversity with low overhead',
          'They are always the only two tokens accepted by speculative decoding',
          'They remove the need to keep a KV cache',
        ],
        answerIndex: 0,
        explanation: 'Collect-2-Query aims to keep most of the sparse-selection signal while avoiding the cost of collecting every verifier row.',
      },
      {
        id: 'specsa-naive-duplicate-work',
        level: 'Mechanism',
        prompt: 'Why does naive sparse speculative verification duplicate work?',
        choices: [
          'Nearby verifier queries may select overlapping KV blocks, but independent sparse kernels reload shared blocks',
          'Every query is forced to read disjoint blocks',
          'The verifier refuses to batch speculative positions',
        ],
        answerIndex: 0,
        explanation: 'The opportunity is cross-query overlap. Naive sparse execution sees smaller rows but loses reuse across those rows.',
      },
      {
        id: 'specsa-exact-merged-preserves',
        level: 'Mechanism',
        prompt: 'What does exact merged scheduling preserve?',
        choices: [
          'Each query original selected-block semantics',
          'Only the representative query selected blocks',
          'The dense all-block layout for every row',
        ],
        answerIndex: 0,
        explanation: 'Exact mode loads the group union once and applies per-query masks so each row attends only to its own selected blocks.',
      },
      {
        id: 'specsa-shared-index-trade',
        level: 'Application',
        prompt: 'What does approximate shared-index scheduling trade?',
        choices: [
          'Exact per-query sparse layouts for simpler execution and greater KV-block reuse',
          'Target-model verification for unverified draft tokens',
          'KV-cache storage for recomputing the whole prefix',
        ],
        answerIndex: 0,
        explanation: 'Shared-index mode is a deliberate approximation: more regularity and reuse, less exact query-specific routing.',
      },
      {
        id: 'specsa-refresh-layer',
        level: 'Mechanism',
        prompt: 'What is a refresh layer?',
        choices: [
          'A layer that recomputes selected sparse indices',
          'A layer that deletes accepted draft tokens',
          'A layer that disables causal masking',
        ],
        answerIndex: 0,
        explanation: 'Refresh layers pay routing cost again so later reuse layers can inherit an up-to-date sparse layout.',
      },
      {
        id: 'specsa-reuse-layer',
        level: 'Mechanism',
        prompt: 'What is a reuse layer?',
        choices: [
          'A layer that reuses selected indices from a previous refresh layer',
          'A layer that recomputes every selected index from scratch',
          'A layer that switches from sparse attention to tokenization',
        ],
        answerIndex: 0,
        explanation: 'Reuse layers avoid repeated routing overhead when sparse layouts are stable enough across nearby layers.',
      },
      {
        id: 'specsa-planner-purpose',
        level: 'Interview',
        prompt: 'Why does SpecSA need a planner?',
        choices: [
          'Draft length, tree shape, grouping, traversal, precision class, and refresh schedule interact with prompt behavior and kernel cost',
          'The planner replaces the target model output distribution',
          'The planner decides which English words the tokenizer supports',
        ],
        answerIndex: 0,
        explanation: 'The best accepted-token throughput depends on both acceptance behavior and the systems cost of sparse verification.',
      },
    ],
    labs: [
      {
        id: 'mini-spec-sparse-prefix',
        title: 'Repair accepted-prefix logic',
        prompt: 'Open mini-spec-sparse/exercises/01_accept_prefix.rs and implement longest-prefix acceptance.',
        successCriteria: 'The prefix tests pass and you can explain why accepted length controls speculative speedup.',
      },
      {
        id: 'mini-spec-sparse-criticality',
        title: 'Build critical KV scores',
        prompt: 'Complete the average-logit and top-k exercises, then explain why Collect-2-Query uses boundary rows.',
        successCriteria: 'The selected top-k entries match the verification criticality scores.',
      },
      {
        id: 'mini-spec-sparse-schedule',
        title: 'Compare sparse verifier schedules',
        prompt: 'Complete overlap, exact merged schedule, shared-index, refresh/reuse, and planner exercises.',
        successCriteria: 'You can identify when exact reuse, approximate reuse, or more frequent refreshes should win.',
      },
    ],
  },
  turboquant: {
    quiz: [
      {
        id: 'turboquant-kv-cache-stores',
        level: 'Foundation',
        prompt: 'What is stored in the KV cache?',
        choices: [
          'Optimizer states for model training',
          'Key and value vectors for previous tokens at each transformer layer',
          'Only the final generated answer text',
        ],
        answerIndex: 1,
        explanation: 'The cache stores K and V vectors so each new token can attend to previous positions without recomputing the whole prefix.',
      },
      {
        id: 'turboquant-cache-growth',
        level: 'Foundation',
        prompt: 'Why does KV cache memory grow during decoding?',
        choices: [
          'Each new token adds another key and value vector to the cache',
          'The vocabulary is copied once per generated token',
          'The model weights are duplicated for every request',
        ],
        answerIndex: 0,
        explanation: 'Autoregressive decoding appends K and V entries as the sequence grows.',
      },
      {
        id: 'turboquant-long-context-pressure',
        level: 'Foundation',
        prompt: 'Why is KV cache especially important for long-context inference?',
        choices: [
          'Long-context inference does not use attention',
          'Long prompts remove the need for key vectors',
          'Cache size grows with context length, increasing memory and bandwidth pressure',
        ],
        answerIndex: 2,
        explanation: 'The KV cache is linear in tokens, layers, KV heads, head dimension, and precision.',
      },
      {
        id: 'turboquant-attention-score',
        level: 'Foundation',
        prompt: 'What is the attention score between a query and a key?',
        choices: [
          'Their dot product, usually scaled before softmax',
          'The number of bytes in the value vector',
          'The token id divided by the layer count',
        ],
        answerIndex: 0,
        explanation: 'Attention logits come from query-key inner products, so key compression must preserve those scores.',
      },
      {
        id: 'turboquant-quantization-definition',
        level: 'Foundation',
        prompt: 'What does quantization do?',
        choices: [
          'It changes causal attention into bidirectional attention',
          'It maps high-precision numbers to a smaller set of low-bit codes',
          'It deletes all previous tokens from the cache',
        ],
        answerIndex: 1,
        explanation: 'Quantization reduces representation precision so vectors use fewer bits.',
      },
      {
        id: 'turboquant-attention-risk',
        level: 'Mechanism',
        prompt: 'Why can quantization hurt attention?',
        choices: [
          'It can change query-key dot products and alter which tokens receive attention',
          'It prevents the softmax from normalizing probabilities',
          'It always increases KV-cache size',
        ],
        answerIndex: 0,
        explanation: 'If quantization changes score ordering, attention can focus on the wrong cached token.',
      },
      {
        id: 'turboquant-mse-not-enough',
        level: 'Mechanism',
        prompt: 'Why is minimizing MSE not always enough for key compression?',
        choices: [
          'MSE can only be computed for value vectors',
          'Low MSE forces every attention score to be exact',
          'A vector can reconstruct well while still producing biased inner products',
        ],
        answerIndex: 2,
        explanation: 'Attention uses query-key inner products, so the relevant failure may be signed score bias rather than visual vector closeness.',
      },
      {
        id: 'turboquant-rotation-purpose',
        level: 'Mechanism',
        prompt: 'What does random rotation help with?',
        choices: [
          'It spreads vector energy more evenly across coordinates',
          'It changes all keys into values',
          'It removes the need for a KV cache',
        ],
        answerIndex: 0,
        explanation: 'After rotation, no single coordinate dominates as much, making coordinate-wise quantization more predictable.',
      },
      {
        id: 'turboquant-two-stages',
        level: 'Mechanism',
        prompt: 'What are TurboQuant main stages?',
        choices: [
          'Tokenizer retraining followed by beam search',
          'MSE-focused vector quantization followed by a 1-bit QJL residual correction',
          'Dense attention followed by residual stream deletion',
        ],
        answerIndex: 1,
        explanation: 'TurboQuant first captures the main vector, then sketches the remaining error to improve inner-product estimation.',
      },
      {
        id: 'turboquant-qjl-fixes',
        level: 'Mechanism',
        prompt: 'What does the QJL residual stage try to fix?',
        choices: [
          'The number of transformer layers',
          'Bias in inner-product estimation',
          'The model vocabulary order',
        ],
        answerIndex: 1,
        explanation: 'The residual sketch is aimed at dot-product correction, not perfect vector reconstruction.',
      },
      {
        id: 'turboquant-unbiased-keys',
        level: 'Mechanism',
        prompt: 'Why is inner-product unbiasedness important for keys?',
        choices: [
          'Keys are never used in attention logits',
          'The value vectors become model weights',
          'Attention logits are query-key inner products',
        ],
        answerIndex: 2,
        explanation: 'Systematic score shifts can change attention rankings and downstream behavior.',
      },
      {
        id: 'turboquant-outlier-bits',
        level: 'Application',
        prompt: 'Why might outlier channels get more bits?',
        choices: [
          'Some dimensions contribute disproportionately to quantization error',
          'Outlier channels are padding tokens',
          'Extra bits make context length shorter',
        ],
        answerIndex: 0,
        explanation: 'Mixed precision spends memory where it most reduces error.',
      },
      {
        id: 'turboquant-online',
        level: 'Application',
        prompt: 'What does online mean in this lesson?',
        choices: [
          'The model must call a web service for every token',
          'Vectors can be quantized as they are produced during inference without a model-specific trained codebook',
          'Only prompt tokens are quantized before decoding starts',
        ],
        answerIndex: 1,
        explanation: 'The method is designed for runtime use on incoming cache vectors.',
      },
      {
        id: 'turboquant-weight-vs-kv',
        level: 'Application',
        prompt: 'What is the difference between weight quantization and KV-cache quantization?',
        choices: [
          'Weight quantization compresses model parameters; KV-cache quantization compresses request-specific runtime memory',
          'KV-cache quantization changes only tokenizer merges',
          'Weight quantization is used only during speculative decoding',
        ],
        answerIndex: 0,
        explanation: 'The KV cache depends on the current request and grows with sequence length.',
      },
      {
        id: 'turboquant-low-bit-works',
        level: 'Interview',
        prompt: 'Why might a 3-bit KV cache still work even though 3 bits sounds tiny?',
        choices: [
          'Three bits are enough to store every FP16 number exactly',
          'High-dimensional geometry, rotation, and residual correction can preserve important dot products',
          'Attention scores do not depend on cached keys',
        ],
        answerIndex: 1,
        explanation: 'The goal is not exact coordinate storage; it is preserving the relationships attention uses.',
      },
      {
        id: 'turboquant-key-metric',
        level: 'Interview',
        prompt: 'Which metric is more directly tied to key compression quality?',
        choices: [
          'Dot-product error, because it affects attention logits directly',
          'The alphabetical order of token labels',
          'The number of React panels in the lesson',
        ],
        answerIndex: 0,
        explanation: 'MSE still matters, but key vectors are used through query-key dot products.',
      },
      {
        id: 'turboquant-values-different',
        level: 'Interview',
        prompt: 'Why are values different from keys?',
        choices: [
          'Values decide whether causal masking is enabled',
          'Keys influence attention weights through dot products; values are combined by those weights into the context vector',
          'Values are never cached',
        ],
        answerIndex: 1,
        explanation: 'Key errors affect routing; value errors affect the information read after routing.',
      },
      {
        id: 'turboquant-batching',
        level: 'Interview',
        prompt: 'Why can TurboQuant help batching?',
        choices: [
          'It removes all latency from token sampling',
          'It changes transformer layers into convolution layers',
          'Smaller KV caches let more requests or longer contexts fit in the same memory budget',
        ],
        answerIndex: 2,
        explanation: 'Runtime memory is often the limit for concurrent long-context serving.',
      },
    ],
    labs: [
      {
        id: 'mini-turboquant-cache-size',
        title: 'Compute KV-cache memory',
        prompt: 'Open mini-turboquant/exercises/01_kv_cache_size.rs and make the cache sizing and compression-ratio tests pass.',
        successCriteria: 'You include both K and V and can explain why context length and KV heads multiply memory.',
      },
      {
        id: 'mini-turboquant-dot-products',
        title: 'Compare MSE and dot-product error',
        prompt: 'Complete the quantization, dot-product, rotation, and MSE-vs-inner-product exercises.',
        successCriteria: 'You can show a small reconstruction error that still creates signed attention-score bias.',
      },
      {
        id: 'mini-turboquant-tradeoff',
        title: 'Plan a safe low-bit cache',
        prompt: 'Complete residual correction, outlier channels, top-k agreement, and compression tradeoff exercises.',
        successCriteria: 'You can choose the lowest bit-width that stays under a dot-product error budget.',
      },
    ],
  },
  'linear-regression': {
    quiz: LINEAR_REGRESSION_QUIZ,
    labs: [
      {
        id: 'move-line',
        title: 'Move the fitted line',
        prompt: 'Change the slope and intercept until residuals shrink, then identify which points still dominate the error.',
        successCriteria: 'You can name at least one point whose residual pulls the fitted line.',
      },
    ],
  },
  pca: {
    quiz: PCA_QUIZ,
    labs: [
      {
        id: 'projection-error',
        title: 'Compare one and two components',
        prompt: 'Switch between 1D and 2D projection, then identify when the 1D reconstruction loses the most information.',
        successCriteria: 'You can connect higher noise or weaker correlation to lower PC1 explained variance.',
      },
    ],
  },
  'fundamental-subspaces': {
    quiz: FUNDAMENTAL_SUBSPACES_QUIZ,
    labs: [
      {
        id: 'classify-four-spaces',
        title: 'Classify the four spaces',
        prompt: 'Choose one rank-nullity case and list each subspace, its ambient space, and its dimension.',
        successCriteria: 'Your dimensions satisfy n = rank + nullity and m = rank + left-nullity.',
      },
    ],
  },
  'matrix-decompositions': {
    quiz: MATRIX_DECOMPOSITIONS_QUIZ,
    labs: [
      {
        id: 'choose-by-goal',
        title: 'Choose by the job',
        prompt: 'Pick three scenarios from the chooser and write the factorization, its requirement, and one warning.',
        successCriteria: 'Your selections are justified by the task and include at least one assumption or stability caveat.',
      },
    ],
  },
  'qr-decomposition': {
    quiz: QR_DECOMPOSITION_QUIZ,
    labs: [
      {
        id: 'trace-projection-removal',
        title: 'Trace projection removal',
        prompt: 'Follow the animation through the second Gram-Schmidt step and identify the projection that is removed.',
        successCriteria: 'You can explain how q2 becomes orthogonal to q1 and where R stores the projection coefficient.',
      },
    ],
  },
  svd: {
    quiz: SVD_QUIZ,
    labs: [
      {
        id: 'rank-k-reconstruction',
        title: 'Trace a rank-k reconstruction',
        prompt: 'Use the SVD animation to identify U, Sigma, and V^T, then explain what is lost when only the largest singular value is kept.',
        successCriteria: 'You can connect singular value size to retained structure and reconstruction error.',
      },
    ],
  },
  'k-means': {
    quiz: KMEANS_QUIZ,
    labs: [
      {
        id: 'k-vs-inertia',
        title: 'Compare k and inertia',
        prompt: 'Change k and the number of iterations, then describe when lower inertia starts splitting a natural group.',
        successCriteria: 'You can explain why inertia alone cannot choose the best k.',
      },
    ],
  },
  'bayes-rule-ml': {
    quiz: BAYES_RULE_QUIZ,
    labs: [
      {
        id: 'rare-class-posterior',
        title: 'Audit a rare-class signal',
        prompt: 'Set a low base rate, then lower the false positive rate until the posterior becomes useful.',
        successCriteria: 'You can explain why evidence quality matters more when the class is rare.',
      },
    ],
  },
  'sampling-confidence-intervals': {
    quiz: SAMPLING_CONFIDENCE_INTERVALS_QUIZ,
    labs: [
      {
        id: 'sample-size-sweep',
        title: 'Sweep sample size',
        prompt: 'Increase sample size from small to large and track how quickly the interval width shrinks.',
        successCriteria: 'You can explain why quadrupling sample size is closer to halving the margin than quartering it.',
      },
    ],
  },
  'hypothesis-testing-intuition': {
    quiz: HYPOTHESIS_TESTING_INTUITION_QUIZ,
    labs: [
      {
        id: 'tiny-effect-large-sample',
        title: 'Find a tiny significant effect',
        prompt: 'Use a small effect with a large sample and explain why the evidence can look strong while the effect remains small.',
        successCriteria: 'You can separate statistical evidence from practical impact.',
      },
    ],
  },
  'ab-testing-foundations': {
    quiz: AB_TESTING_FOUNDATIONS_QUIZ,
    labs: [
      {
        id: 'ab-decision-audit',
        title: 'Audit an A/B decision',
        prompt: 'Build a scenario where a treatment improves the primary metric but fails a guardrail. Explain the launch decision.',
        successCriteria: 'You can explain assignment, metric, guardrail, and decision threshold separately.',
      },
    ],
  },
  'power-sample-size': {
    quiz: POWER_SAMPLE_SIZE_QUIZ,
    labs: [
      {
        id: 'underpowered-design-audit',
        title: 'Diagnose an underpowered test',
        prompt: 'Create an underpowered design and explain why the readout is not decisive.',
        successCriteria: 'You can name the MDE, planned sample, power, and false negative risk.',
      },
    ],
  },
  'sequential-testing-peeking': {
    quiz: SEQUENTIAL_TESTING_PEEKING_QUIZ,
    labs: [
      {
        id: 'peeking-policy-comparison',
        title: 'Compare peeking policies',
        prompt: 'Set many interim looks and compare naive false positive risk with the planned alpha budget.',
        successCriteria: 'You can explain why any-look risk exceeds a single fixed-horizon alpha.',
      },
    ],
  },
  'cuped-variance-reduction': {
    quiz: CUPED_VARIANCE_REDUCTION_QUIZ,
    labs: [
      {
        id: 'pre-period-signal-audit',
        title: 'Find useful pre-period signal',
        prompt: 'Increase pre/post correlation and explain how CUPED narrows the confidence interval.',
        successCriteria: 'You can connect correlation, standard error, and interval width.',
      },
    ],
  },
  'confounding-simpsons-paradox': {
    quiz: CONFOUNDING_SIMPSONS_PARADOX_QUIZ,
    labs: [
      {
        id: 'simpson-reversal-audit',
        title: 'Create a Simpson reversal',
        prompt: 'Change segment mix until the aggregate effect disagrees with the within-segment effect.',
        successCriteria: 'You can explain why the aggregate and segment effects point different ways.',
      },
    ],
  },
  'causal-graphs-dags': {
    quiz: CAUSAL_GRAPHS_DAGS_QUIZ,
    labs: [
      {
        id: 'adjustment-set-repair',
        title: 'Repair an adjustment set',
        prompt: 'Increase collider conditioning and explain why the adjustment set gets worse.',
        successCriteria: 'You can classify a variable as confounder, collider, or mediator.',
      },
    ],
  },
  'treatment-effects': {
    quiz: TREATMENT_EFFECTS_QUIZ,
    labs: [
      {
        id: 'rollout-policy-comparison',
        title: 'Compare rollout policies',
        prompt: 'Create a scenario with positive ATE and one harmed segment. Explain the rollout implication.',
        successCriteria: 'You can explain when targeted treatment beats treating everyone.',
      },
    ],
  },
  'propensity-scores': {
    quiz: PROPENSITY_SCORES_QUIZ,
    labs: [
      {
        id: 'observational-support-audit',
        title: 'Audit observational support',
        prompt: 'Lower overlap or raise hidden bias and explain why the estimate becomes fragile.',
        successCriteria: 'You can identify overlap, balance, and unmeasured-confounding limits.',
      },
    ],
  },
  'time-series-forecasting-track': {
    quiz: TIME_SERIES_FORECASTING_QUIZ,
    labs: [
      {
        id: 'time-series-forecasting-track-scenario-lab',
        title: 'Audit a forecasting backtest',
        prompt: 'Create a forecast setup with future leakage and explain how rolling splits fix it.',
        successCriteria: 'You can identify the horizon, lag features, split design, and leakage boundary.',
      },
    ],
  },
  'recommender-systems-ranking-track': {
    quiz: RECOMMENDER_SYSTEMS_RANKING_QUIZ,
    labs: [
      {
        id: 'recommender-systems-ranking-track-scenario-lab',
        title: 'Design a ranking readout',
        prompt: 'Build a cold-start scenario and choose a hybrid or exploration strategy.',
        successCriteria: 'You can separate prediction, ordering, exposure, and exploration concerns.',
      },
    ],
  },
  'ml-security-robustness-track': {
    quiz: ML_SECURITY_ROBUSTNESS_QUIZ,
    labs: [
      {
        id: 'ml-security-robustness-track-scenario-lab',
        title: 'Map one ML threat',
        prompt: 'Choose a prompt-injection or poisoning scenario and define a defense plus an eval.',
        successCriteria: 'You can name an attack, impacted asset, control, and evaluation.',
      },
    ],
  },
  'efficient-inference-compression-track': {
    quiz: EFFICIENT_INFERENCE_COMPRESSION_QUIZ,
    labs: [
      {
        id: 'efficient-inference-compression-track-scenario-lab',
        title: 'Tune a serving tradeoff',
        prompt: 'Create a high-throughput but high-latency scenario and explain the tradeoff.',
        successCriteria: 'You can explain which bottleneck is addressed by batching, quantization, or paged attention.',
      },
    ],
  },
  'data-engineering-for-ml-track': {
    quiz: DATA_ENGINEERING_FOR_ML_QUIZ,
    labs: [
      {
        id: 'data-engineering-for-ml-track-scenario-lab',
        title: 'Audit point-in-time correctness',
        prompt: 'Create a label-window or target-encoding leakage scenario and define the fix.',
        successCriteria: 'You can identify prediction time, feature time, label window, and serving parity.',
      },
    ],
  },
  'spearman-correlation': {
    quiz: SPEARMAN_CORRELATION_QUIZ,
    labs: [
      {
        id: 'rank-before-score',
        title: 'Rank before scoring',
        prompt: 'Use the calculation tab to rank X and Y, then predict the sign and size of rho before revealing the formula.',
        successCriteria: 'You can explain whether rho changed because the order changed or because raw values moved farther apart.',
      },
    ],
  },
  'maximum-likelihood-estimation': {
    quiz: MAXIMUM_LIKELIHOOD_ESTIMATION_QUIZ,
    labs: [
      {
        id: 'candidate-parameter-sweep',
        title: 'Sweep candidate parameters',
        prompt: 'Move the candidate probability across the observed rate and watch relative likelihood rise and fall.',
        successCriteria: 'You can identify the candidate closest to the maximum likelihood estimate.',
      },
    ],
  },
  'loss-functions-likelihoods': {
    quiz: LOSS_FUNCTIONS_LIKELIHOODS_QUIZ,
    labs: [
      {
        id: 'match-loss-to-noise',
        title: 'Match losses to assumptions',
        prompt: 'Compare regression error and true-class probability, then name the likelihood assumption behind each loss.',
        successCriteria: 'You can connect squared error to Gaussian noise and cross-entropy to categorical likelihood.',
      },
    ],
  },
  'train-validation-test-split': {
    quiz: TRAIN_VALIDATION_TEST_SPLIT_QUIZ,
    labs: [
      {
        id: 'spot-leakage',
        title: 'Spot a leakage path',
        prompt: 'Move the split controls and explain one way preprocessing before splitting could leak information.',
        successCriteria: 'Your example names what statistic or label information crosses the split boundary.',
      },
    ],
  },
  'cross-validation': {
    quiz: CROSS_VALIDATION_QUIZ,
    labs: [
      {
        id: 'choose-fold-design',
        title: 'Choose a leakage-safe fold design',
        prompt: 'Use the fold controls to compare random and grouped splitting, then explain which one is safer for repeated users or related samples.',
        successCriteria: 'You can name the leakage route and justify the fold design that blocks it.',
      },
    ],
  },
  'data-leakage-deep-dive': {
    quiz: DATA_LEAKAGE_DEEP_DIVE_QUIZ,
    labs: [
      {
        id: 'leakage-mode-audit',
        title: 'Audit one leakage mode',
        prompt: 'Select each leakage mode and state the boundary being crossed and the safer split or pipeline rule.',
        successCriteria: 'You can name the leaked information and the concrete prevention rule for at least three modes.',
      },
    ],
  },
  'feature-scaling-preprocessing': {
    quiz: FEATURE_SCALING_PREPROCESSING_QUIZ,
    labs: [
      {
        id: 'outlier-scaler-comparison',
        title: 'Compare scaler sensitivity',
        prompt: 'Toggle the outlier and compare standard, min-max, and robust scaling on the same selected point.',
        successCriteria: 'You can explain which scaler moved most and why robust scaling is less sensitive to one extreme value.',
      },
    ],
  },
  'logistic-regression': {
    quiz: LOGISTIC_REGRESSION_QUIZ,
    labs: [
      {
        id: 'threshold-flips',
        title: 'Find threshold flips',
        prompt: 'Move the threshold and identify which points change class first.',
        successCriteria: 'You can explain each flip by comparing its probability with the threshold.',
      },
    ],
  },
  'classification-metrics': {
    quiz: CLASSIFICATION_METRICS_QUIZ,
    labs: [
      {
        id: 'threshold-counts',
        title: 'Trace the confusion matrix',
        prompt: 'Move the threshold and predict which count changes before reading the metric tiles.',
        successCriteria: 'You can connect at least one threshold move to TP, FP, FN, or TN.',
      },
    ],
  },
  'roc-pr-curves': {
    quiz: ROC_PR_CURVES_QUIZ,
    labs: [
      {
        id: 'threshold-costs',
        title: 'Choose an operating threshold',
        prompt: 'Move the threshold and pick a cutoff for a case where false negatives are more expensive than false positives.',
        successCriteria: 'You can justify the threshold using recall, precision, and the mistake costs.',
      },
    ],
  },
  calibration: {
    quiz: CALIBRATION_QUIZ,
    labs: [
      {
        id: 'reliability-gap',
        title: 'Find the largest reliability gap',
        prompt: 'Switch between calibrated, overconfident, and underconfident modes, then identify the bucket with the largest gap.',
        successCriteria: 'You can compare predicted probability with observed positive rate and describe the calibration error.',
      },
    ],
  },
  overfitting: {
    quiz: OVERFITTING_QUIZ,
    labs: [
      {
        id: 'find-sweet-spot',
        title: 'Find the validation sweet spot',
        prompt: 'Move complexity until validation error is lowest, then compare it with the lowest training error.',
        successCriteria: 'You can explain why the best validation point is not necessarily the most complex model.',
      },
    ],
  },
  'bias-variance-tradeoff': {
    quiz: BIAS_VARIANCE_TRADEOFF_QUIZ,
    labs: [
      {
        id: 'complexity-sweep',
        title: 'Sweep complexity and sample size',
        prompt: 'Switch between simple, balanced, and flexible models, then change sample size and noise.',
        successCriteria: 'You can explain when the dominant problem is bias versus variance from the train and validation errors.',
      },
    ],
  },
  regularization: {
    quiz: REGULARIZATION_QUIZ,
    labs: [
      {
        id: 'lambda-sweep',
        title: 'Sweep lambda',
        prompt: 'Increase the penalty and watch which weights shrink fastest.',
        successCriteria: 'You can describe the tradeoff between data loss, penalty, and total loss.',
      },
    ],
  },
  'knn-naive-bayes-svm': {
    quiz: KNN_NAIVE_BAYES_SVM_QUIZ,
    labs: [
      {
        id: 'boundary-comparison',
        title: 'Compare decision changes',
        prompt: 'Move the query point near the boundary and switch between kNN, Naive Bayes, and SVM.',
        successCriteria: 'You can explain whether the decision came from local neighbors, likelihood scores, or margin side.',
      },
    ],
  },
  'tree-ensembles': {
    quiz: TREE_ENSEMBLES_QUIZ,
    labs: [
      {
        id: 'depth-vs-ensemble',
        title: 'Compare depth and ensemble size',
        prompt: 'Change tree depth, forest tree count, and boosting rounds, then identify which control most risks memorizing quirks.',
        successCriteria: 'You can explain the difference between deeper single trees, forest averaging, and boosting rounds.',
      },
    ],
  },
  'gradient-descent': {
    quiz: GRADIENT_DESCENT_QUIZ,
    labs: [
      {
        id: 'tune-step-size',
        title: 'Tune step size',
        prompt: 'Try a small, medium, and large learning rate and compare the loss trace.',
        successCriteria: 'You can identify which run converges, crawls, or overshoots.',
      },
    ],
  },
  'neural-network': {
    quiz: NEURAL_NETWORK_QUIZ,
    labs: [
      {
        id: 'trace-forward-backward',
        title: 'Trace one XOR pass',
        prompt: 'Choose one XOR input, run a forward pass, then step backward and identify where the loss signal first appears.',
        successCriteria: 'You can name the input, target, output, loss, and the first gradient-carrying layer.',
      },
    ],
  },
  optimizers: {
    quiz: OPTIMIZERS_QUIZ,
    labs: [
      {
        id: 'compare-update-rules',
        title: 'Compare update rules',
        prompt: 'Run SGD, momentum, and Adam with the same learning rate and batch size, then identify which path zigzags least.',
        successCriteria: 'You can connect the visible path to velocity, adaptive scaling, or mini-batch noise.',
      },
      {
        id: 'predict-then-rotate-landscape',
        title: 'Predict then rotate the landscape',
        prompt: 'Use the prediction check for the selected optimizer, then rotate the 3D surface and explain why the first step and final endpoint match the computed path.',
        successCriteria: 'You can connect the first-step sign, surface curvature, endpoint height, and final loss statistic.',
      },
    ],
  },
  initialization: {
    quiz: INITIALIZATION_QUIZ,
    labs: [
      {
        id: 'signal-scale',
        title: 'Find a stable signal scale',
        prompt: 'Switch between tiny, Xavier, He, and huge initialization while changing activation and depth.',
        successCriteria: 'You can explain which setup keeps layer variance stable and why another one vanishes or explodes.',
      },
    ],
  },
  'dropout-batchnorm': {
    quiz: DROPOUT_BATCHNORM_QUIZ,
    labs: [
      {
        id: 'mode-switch',
        title: 'Compare training and inference',
        prompt: 'Toggle training mode and change dropout rate, then explain why the visible output no longer masks units at inference.',
        successCriteria: 'You can separate BatchNorm scaling effects from dropout masking effects.',
      },
    ],
  },
  'training-loop-dynamics': {
    quiz: TRAINING_LOOP_DYNAMICS_QUIZ,
    labs: [
      {
        id: 'loop-diagnosis',
        title: 'Diagnose a training loop',
        prompt: 'Increase learning rate, shrink batch size, and raise validation difficulty, then identify noisy, overshooting, or overfitting behavior.',
        successCriteria: 'You can explain why train loss, validation loss, and update stability must be read together.',
      },
    ],
  },
  relu: {
    quiz: RELU_QUIZ,
    labs: [
      {
        id: 'cross-zero',
        title: 'Cross the kink',
        prompt: 'Move the input across zero and watch both output and slope change.',
        successCriteria: 'You can name the active region and the blocked region.',
      },
    ],
  },
  'leaky-relu': {
    quiz: LEAKY_RELU_QUIZ,
    labs: [
      {
        id: 'compare-dead-zone',
        title: 'Compare the dead zone',
        prompt: 'Set z below zero, compare alpha = 0 with alpha above 0, and record how the output and gradient change.',
        successCriteria: 'You can explain why alpha = 0 behaves like ReLU and alpha above 0 preserves a small gradient.',
      },
    ],
  },
  'computation-graph-backprop': {
    quiz: COMPUTATION_GRAPH_BACKPROP_QUIZ,
    labs: [
      {
        id: 'predict-update',
        title: 'Predict the update',
        prompt: 'Before switching to the update tab, predict whether the next loss will fall.',
        successCriteria: 'Your prediction uses the gradient sign and the learning rate.',
      },
    ],
  },
  tokenization: {
    quiz: TOKENIZATION_QUIZ,
    labs: [
      {
        id: 'compare-splits',
        title: 'Compare token splits',
        prompt: 'Tokenize the same phrase with two modes and identify the longest changed fragment.',
        successCriteria: 'You can explain why one split creates more or fewer tokens.',
      },
    ],
  },
  embeddings: {
    quiz: EMBEDDINGS_QUIZ,
    labs: [
      {
        id: 'nearest-neighbor',
        title: 'Inspect a neighbor',
        prompt: 'Move or choose a vector and compare its nearest neighbors.',
        successCriteria: 'You can separate semantic similarity from incidental training-data correlation.',
      },
    ],
  },
  'cosine-similarity': {
    quiz: COSINE_SIMILARITY_QUIZ,
    labs: [
      {
        id: 'predict-nearest',
        title: 'Predict the nearest vector',
        prompt: 'Before changing the sliders or query, predict which vector or document will rank highest and why.',
        successCriteria: 'Your explanation names either direction alignment or a shared active feature, not just raw magnitude.',
      },
    ],
  },
  'attention-mechanism': {
    quiz: ATTENTION_MECHANISM_QUIZ,
    labs: [
      {
        id: 'shift-query',
        title: 'Shift the query',
        prompt: 'Change a query and predict which value will receive more weight.',
        successCriteria: 'Your prediction follows the largest query-key score.',
      },
    ],
  },
  'self-attention': {
    quiz: SELF_ATTENTION_QUIZ,
    labs: [
      {
        id: 'attention-row',
        title: 'Read one attention row',
        prompt: 'Pick a token and explain which other tokens it attends to most.',
        successCriteria: 'You can connect one row of weights to the resulting context vector.',
      },
    ],
  },
  'kv-cache': {
    quiz: KV_CACHE_QUIZ,
    labs: [
      {
        id: 'decode-step-savings',
        title: 'Compare decode-step work',
        prompt: 'Increase the decode step with caching on and off, then explain which computation grows and which stays flat.',
        successCriteria: 'You can separate avoided K/V projection work from attention reads over the visible cache.',
      },
    ],
  },
  'grouped-query-attention': {
    quiz: GROUPED_QUERY_ATTENTION_QUIZ,
    labs: [
      {
        id: 'compare-mha-mqa-gqa',
        title: 'Compare sharing regimes',
        prompt: 'Set KV heads equal to query heads, then to one, then to an intermediate value. Explain the memory and specialization tradeoff.',
        successCriteria: 'You can identify MHA, MQA, and GQA and explain why GQA is the middle ground.',
      },
    ],
  },
  'flash-attention': {
    quiz: FLASH_ATTENTION_QUIZ,
    labs: [
      {
        id: 'tile-memory-tradeoff',
        title: 'Trace one streamed tile',
        prompt: 'Increase sequence length and compare full score-matrix memory with tile working-set memory. Explain why the FLOPs remain attention-like while memory traffic drops.',
        successCriteria: 'You can separate exact attention computation from the memory schedule that avoids materializing N by N scores.',
      },
    ],
  },
  'attention-masks': {
    quiz: ATTENTION_MASKS_QUIZ,
    labs: [
      {
        id: 'trace-visible-keys',
        title: 'Trace visible keys',
        prompt: 'Pick one query row, switch between mask types, and list which keys remain visible before softmax.',
        successCriteria: 'You can justify each visible or blocked key using causal order, padding, or cross-attention memory.',
      },
    ],
  },
  'positional-encoding': {
    quiz: POSITIONAL_ENCODING_QUIZ,
    labs: [
      {
        id: 'order-sensitive-sentence',
        title: 'Compare order-sensitive meaning',
        prompt: 'Switch between "dog bites man" and "man bites dog", then disable the position signal. Explain what information the model loses.',
        successCriteria: 'You can explain why the same tokens need different positional context to represent different meanings.',
      },
    ],
  },
  rope: {
    quiz: ROPE_QUIZ,
    labs: [
      {
        id: 'relative-shift-check',
        title: 'Check relative shift behavior',
        prompt: 'Move query and key positions together by the same amount. Explain what changes and what stays tied to the relative distance.',
        successCriteria: 'You can explain why absolute rotation angles change while the relative-position relationship is preserved.',
      },
    ],
  },
  'residual-stream': {
    quiz: RESIDUAL_STREAM_QUIZ,
    labs: [
      {
        id: 'trace-component-write',
        title: 'Trace one residual write',
        prompt: 'Increase one attention or MLP write, then explain how the before/write/after vectors change.',
        successCriteria: 'You can distinguish adding a component write from replacing the whole token representation.',
      },
    ],
  },
  'conv-relu': {
    quiz: CONV_RELU_QUIZ,
    labs: [
      {
        id: 'bias-sparsity-sweep',
        title: 'Find the activation cutoff',
        prompt: 'Lower the bias until several feature responses disappear, then identify which pre-activation values were clipped.',
        successCriteria: 'You can explain the difference between the signed convolution map and the sparse ReLU activation map.',
      },
    ],
  },
  'max-pooling': {
    quiz: MAX_POOLING_QUIZ,
    labs: [
      {
        id: 'trace-window-argmax',
        title: 'Trace a pooling window',
        prompt: 'Select an output cell, write the highlighted input values, and identify which coordinate supplied the max.',
        successCriteria: 'You can compute the pooled output cell and explain which input details were discarded.',
      },
    ],
  },
  conv2d: {
    quiz: CONV2D_QUIZ,
    labs: [
      {
        id: 'trace-output-size',
        title: 'Trace output shape',
        prompt: 'Switch stride and padding, then compute the output-size formula before checking the displayed grid.',
        successCriteria: 'You can explain why padding increases available windows while stride skips windows.',
      },
    ],
  },
  'gradient-problems': {
    quiz: GRADIENT_PROBLEMS_QUIZ,
    labs: [
      {
        id: 'diagnose-gradient-flow',
        title: 'Diagnose a gradient chain',
        prompt: 'Create one vanishing case and one exploding case by changing depth and local multiplier, then stabilize one of them.',
        successCriteria: 'You can identify whether the problem comes from depth, local derivative scale, residual paths, or clipping.',
      },
    ],
  },
  'layer-normalization': {
    quiz: LAYER_NORMALIZATION_QUIZ,
    labs: [
      {
        id: 'normalize-shifted-token',
        title: 'Normalize a shifted token',
        prompt: 'Choose the shifted token, inspect mean and standard deviation, then explain what gamma and beta restore after normalization.',
        successCriteria: 'You can distinguish the raw token shift from the normalized feature pattern and the learned affine output.',
      },
    ],
  },
  transformer: {
    quiz: TRANSFORMER_QUIZ,
    labs: [
      {
        id: 'trace-token',
        title: 'Trace one token',
        prompt: 'Follow one token through attention, feed-forward, residual, and normalization stages.',
        successCriteria: 'You can say what each stage changes and what it preserves.',
      },
    ],
  },
  'transformer-architecture-families': {
    quiz: TRANSFORMER_ARCHITECTURE_FAMILIES_QUIZ,
    labs: [
      {
        id: 'choose-family',
        title: 'Choose the right family',
        prompt: 'Pick search reranking, chat completion, and translation tasks, then choose the architecture family for each.',
        successCriteria: 'You can justify each choice using attention visibility, objective, and output type.',
      },
    ],
  },
  'llm-training-objectives': {
    quiz: LLM_TRAINING_OBJECTIVES_QUIZ,
    labs: [
      {
        id: 'match-objective-stage',
        title: 'Match objective to behavior',
        prompt: 'Select each objective and decide whether it mainly teaches continuation, representation, instruction following, or preference-shaped response behavior.',
        successCriteria: 'You can justify each match using the target in the loss.',
      },
    ],
  },
  'transformer-token-generation': {
    quiz: TRANSFORMER_TOKEN_GENERATION_QUIZ,
    labs: [
      {
        id: 'sampling-contrast',
        title: 'Compare two decoding settings',
        prompt: 'Generate a few steps with low temperature and greedy mode, then compare it with higher temperature and sampling.',
        successCriteria: 'You can explain which setting is more deterministic and which setting keeps more alternatives alive.',
      },
    ],
  },
  'sampling-strategies': {
    quiz: SAMPLING_STRATEGIES_QUIZ,
    labs: [
      {
        id: 'decode-for-task',
        title: 'Choose decoding for a task',
        prompt: 'Pick settings for factual QA, brainstorming, and translation, then explain which one should be more deterministic or diverse.',
        successCriteria: 'You can justify each setting using candidate count, probability mass, and sequence-score tradeoffs.',
      },
    ],
  },
  'fine-tuning': {
    quiz: FINE_TUNING_QUIZ,
    labs: [
      {
        id: 'choose-finetune-method',
        title: 'Choose a fine-tuning method',
        prompt: 'Pick one scenario with limited GPU memory, one with demonstrations, and one with preference pairs, then choose LoRA, SFT, or DPO/RLHF.',
        successCriteria: 'Your choice matches the available data signal and the memory or behavior constraint.',
      },
    ],
  },
  'rag-chunking-context': {
    quiz: RAG_CHUNKING_CONTEXT_QUIZ,
    labs: [
      {
        id: 'tune-chunk-budget',
        title: 'Tune chunking and packing',
        prompt: 'Adjust chunk size, overlap, top-k, and context budget until both relevant refund facts fit without too much duplicate text.',
        successCriteria: 'You can explain which control recovered the boundary fact and which control limited packed evidence.',
      },
    ],
  },
  'rag-vector-indexing': {
    quiz: RAG_VECTOR_INDEXING_QUIZ,
    labs: [
      {
        id: 'choose-index',
        title: 'Choose an index strategy',
        prompt: 'Compare exact, IVF, and HNSW modes for a small corpus, a large corpus, and a high-recall support bot.',
        successCriteria: 'You can justify the choice using latency, recall risk, and corpus scale.',
      },
    ],
  },
  'rag-retrieval-evaluation': {
    quiz: RAG_RETRIEVAL_EVALUATION_QUIZ,
    labs: [
      {
        id: 'chunking-rerank-audit',
        title: 'Audit a retrieval setting',
        prompt: 'Change chunk size, overlap, top-k, and reranking, then pick the setting with the clearest grounded evidence.',
        successCriteria: 'You can justify the setting with recall@k, MRR, nDCG, and the text of the top result.',
      },
    ],
  },
  'rag-reranking-grounding': {
    quiz: RAG_RERANKING_GROUNDING_QUIZ,
    labs: [
      {
        id: 'grounding-audit',
        title: 'Audit grounding behavior',
        prompt: 'Use strictness and top-k to make a stale conflict visible, then make every claim grounded with usable evidence.',
        successCriteria: 'You can explain why strictness blocked one claim and why adjusting top-k changed grounded coverage.',
      },
    ],
  },
  'rag-failure-modes': {
    quiz: RAG_FAILURE_MODES_QUIZ,
    labs: [
      {
        id: 'failure-tune',
        title: 'Tune one failure',
        prompt: 'Use reranker mode, top-k, and strictness to reduce stale or conflicting behavior for at least two claims.',
        successCriteria: 'You can report the top-k/strictness region with fewer stale/conflicting tags and a higher grounded count.',
      },
    ],
  },
  'mdp-formalism': {
    quiz: [
      {
        id: 'mdp-parts',
        prompt: 'What does the transition model P describe in an MDP?',
        choices: [
          'The probability distribution over next states after taking an action',
          'The list of all parameters in a neural network',
          'The final answer chosen by a language model',
        ],
        answerIndex: 0,
        explanation: 'P describes environment dynamics: from a state and action, it gives probabilities for possible next states.',
      },
      {
        id: 'discount-role',
        prompt: 'What changes when gamma is reduced toward zero?',
        choices: [
          'Immediate rewards dominate delayed rewards',
          'Future rewards become more important than immediate rewards',
          'Transition probabilities stop summing to one',
        ],
        answerIndex: 0,
        explanation: 'A smaller discount factor puts less weight on future rewards, so near-term reward matters more.',
      },
    ],
    labs: [
      {
        id: 'gamma-comparison',
        title: 'Compare action values',
        prompt: 'Switch between actions and lower gamma until the safer immediate reward becomes more attractive.',
        successCriteria: 'You can explain how the same transition probabilities lead to different action choices when gamma changes.',
      },
    ],
  },
  'value-iteration': {
    quiz: [
      {
        id: 'bellman-backup',
        prompt: 'What does a Bellman optimality backup do in value iteration?',
        choices: [
          'It compares action lookaheads and keeps the highest expected return',
          'It samples one action and permanently stores that single result',
          'It tokenizes rewards into subword units',
        ],
        answerIndex: 0,
        explanation: 'Value iteration uses the known model to compute expected return for each action, then takes the maximum.',
      },
      {
        id: 'model-vs-samples',
        prompt: 'Why is value iteration considered planning rather than model-free learning?',
        choices: [
          'It uses known transitions and rewards to update values before acting',
          'It ignores transition probabilities',
          'It only works when there are no rewards',
        ],
        answerIndex: 0,
        explanation: 'Planning methods use a model of the environment; Q-learning can learn from sampled experience without that full model.',
      },
    ],
    labs: [
      {
        id: 'sweep-propagation',
        title: 'Watch reward propagation',
        prompt: 'Set sweeps to zero, then increase one step at a time and observe which states change after the goal value appears.',
        successCriteria: 'You can identify how terminal rewards propagate backward through Bellman sweeps.',
      },
    ],
  },
  'policy-iteration': {
    quiz: [
      {
        id: 'two-phases',
        prompt: 'What are the two repeating phases of policy iteration?',
        choices: [
          'Evaluate the current policy, then greedily improve it',
          'Tokenize the state, then sample a reward',
          'Split train and test data, then fit a classifier',
        ],
        answerIndex: 0,
        explanation: 'Policy iteration alternates policy evaluation with policy improvement until the greedy policy stops changing.',
      },
      {
        id: 'stable-policy',
        prompt: 'What does it mean when the improved policy matches the current policy?',
        choices: [
          'The policy is stable under the current value estimates',
          'The transition model has been deleted',
          'Rewards no longer affect decisions',
        ],
        answerIndex: 0,
        explanation: 'If greedy improvement no longer changes any state action, policy iteration has reached a stable policy for the model.',
      },
    ],
    labs: [
      {
        id: 'policy-flip',
        title: 'Find a policy flip',
        prompt: 'Increase improvement rounds and identify the first state whose action changes from the initial policy.',
        successCriteria: 'You can name the state, the old action, and the improved greedy action.',
      },
    ],
  },
  'rl-foundations': {
    quiz: [
      {
        id: 'loop-parts',
        prompt: 'What is the basic reinforcement-learning loop?',
        choices: [
          'The agent observes state, takes action, receives reward and next state',
          'The model splits data into train, validation, and test once',
          'The tokenizer maps words to fixed dictionary definitions',
        ],
        answerIndex: 0,
        explanation: 'RL is organized around repeated interaction: state, action, reward, next state, and another decision.',
      },
      {
        id: 'reward-hacking',
        prompt: 'What is reward hacking?',
        choices: [
          'Optimizing the numeric reward in a way that misses the designer intent',
          'Reducing gamma so future rewards count less',
          'Using a random exploratory action',
        ],
        answerIndex: 0,
        explanation: 'An agent follows the reward signal it is given, so a flawed reward can teach unintended behavior.',
      },
      {
        id: 'discount-factor',
        prompt: 'What does a larger discount factor gamma usually do?',
        choices: [
          'It makes distant future rewards matter more',
          'It makes all future rewards disappear',
          'It turns rewards into actions',
        ],
        answerIndex: 0,
        explanation: 'Gamma controls how much delayed reward contributes to return; higher gamma values make long-term payoff more important.',
      },
    ],
    labs: [
      {
        id: 'design-reward',
        title: 'Design a reward safely',
        prompt: 'Use the reward editor to create a path with a tempting side reward, then explain whether it could distract from the goal.',
        successCriteria: 'You can identify the intended behavior, the numeric incentive, and one possible reward-hacking failure.',
      },
    ],
  },
  'q-learning': {
    quiz: [
      {
        id: 'what-q-means',
        prompt: 'What does Q(s, a) estimate in Q-learning?',
        choices: [
          'Expected discounted return after taking action a in state s and then acting well',
          'The probability that state s appears in the dataset',
          'Only the immediate reward before any future value',
        ],
        answerIndex: 0,
        explanation: 'A Q-value is an action-value estimate: immediate reward plus discounted future return from the next state.',
      },
      {
        id: 'td-target',
        prompt: 'In the update target r + gamma max_a Q(s_prime, a), what does the max over next actions represent?',
        choices: [
          'The best estimated future value from the next state',
          'The action that was definitely sampled in the previous state',
          'A supervised label assigned by a trainer',
        ],
        answerIndex: 0,
        explanation: 'Q-learning is off-policy: it backs up toward the greedy next action value even if exploration sampled something else.',
      },
      {
        id: 'learning-rate-role',
        prompt: 'What does the learning rate alpha control in the Q-learning update?',
        choices: [
          'How far the old Q-value moves toward the new target',
          'How many actions the environment has',
          'Whether rewards are terminal or nonterminal',
        ],
        answerIndex: 0,
        explanation: 'Alpha scales the temporal-difference error, so larger values overwrite old estimates faster.',
      },
    ],
    labs: [
      {
        id: 'trace-one-update',
        title: 'Trace one Q update',
        prompt: 'Use the Bellman update tab to choose old Q, reward, future Q, alpha, and gamma, then predict the new Q before reading it.',
        successCriteria: 'You can name the target, TD error, and final nudged Q-value for one transition.',
      },
    ],
  },
  'rl-exploration': {
    quiz: [
      {
        id: 'epsilon-meaning',
        prompt: 'In epsilon-greedy exploration, what does epsilon control?',
        choices: [
          'The probability of choosing a random exploratory action',
          'The reward discount applied after terminal states',
          'The number of states in the environment',
        ],
        answerIndex: 0,
        explanation: 'Epsilon is the exploration rate: higher epsilon means more random actions instead of greedy exploitation.',
      },
      {
        id: 'explore-exploit-tradeoff',
        prompt: 'Why can pure exploitation fail early in training?',
        choices: [
          'The agent may commit to a bad action before it has tried enough alternatives',
          'The agent automatically knows every reward in advance',
          'The discount factor becomes exactly zero',
        ],
        answerIndex: 0,
        explanation: 'Without exploration, early noisy estimates can trap the policy in a locally attractive but poor behavior.',
      },
      {
        id: 'cliff-risk',
        prompt: 'Why can a risky shortest path be worse under high exploration noise?',
        choices: [
          'Random exploratory moves can push the agent into costly failure states',
          'Exploration deletes all rewards from the environment',
          'The Q-table no longer stores action values',
        ],
        answerIndex: 0,
        explanation: 'When random moves are frequent, a path close to a cliff has a high chance of catastrophic deviation.',
      },
    ],
    labs: [
      {
        id: 'tune-epsilon',
        title: 'Tune exploration for risk',
        prompt: 'Compare low and high epsilon in the epsilon and cliff tabs, then choose a setting for fast learning without frequent falls.',
        successCriteria: 'You can justify the setting using exploration count, exploitation count, falls, and wins.',
      },
    ],
  },
  'policy-gradients': {
    quiz: [
      {
        id: 'positive-advantage',
        prompt: 'What happens to the sampled action when its advantage is positive?',
        choices: [
          'Its log-probability is pushed upward',
          'Its probability is forced to zero',
          'The transition model is recomputed exactly',
        ],
        answerIndex: 0,
        explanation: 'Policy gradients reinforce sampled actions that beat the baseline by increasing their log-probability.',
      },
      {
        id: 'policy-object',
        prompt: 'What is optimized directly in policy-gradient methods?',
        choices: [
          'The stochastic policy action probabilities',
          'Only a fixed Q-table',
          'Only the train/test split',
        ],
        answerIndex: 0,
        explanation: 'Unlike tabular value methods, policy gradients update policy parameters that control action probabilities.',
      },
      {
        id: 'advantage-guarantee',
        prompt: 'What is a policy-gradient misconception?',
        choices: [
          'Thinking one positive-advantage sample guarantees the globally best policy',
          'Using advantage sign to push a sampled action up or down',
          'Updating stochastic action probabilities directly',
        ],
        answerIndex: 0,
        explanation: 'A sampled policy-gradient update is noisy local evidence, not a global guarantee.',
      },
    ],
    labs: [
      {
        id: 'advantage-flip',
        title: 'Flip the advantage',
        prompt: 'Move return below and above the baseline and watch the sampled action probability move down or up.',
        successCriteria: 'You can explain how the sign of advantage changes the policy update direction.',
      },
    ],
  },
  'actor-critic': {
    quiz: [
      {
        id: 'actor-vs-critic',
        prompt: 'What does the critic provide to the actor?',
        choices: [
          'A value baseline used to compute advantage',
          'The final sampled action with no policy',
          'A train/test split for supervised learning',
        ],
        answerIndex: 0,
        explanation: 'The critic estimates value; the actor uses return minus that value as an advantage signal.',
      },
      {
        id: 'negative-advantage',
        prompt: 'If return is below the critic value, what should happen to the sampled action?',
        choices: [
          'The actor should reduce its log-probability',
          'The critic should choose the action directly',
          'The action must become deterministic',
        ],
        answerIndex: 0,
        explanation: 'A negative advantage means the action underperformed the critic baseline, so the actor is pushed away from it.',
      },
      {
        id: 'critic-action-choice',
        prompt: 'What is an actor-critic misconception?',
        choices: [
          'Thinking the critic directly chooses the action instead of estimating value for the actor update',
          'Using the critic value as a baseline',
          'Letting the actor own the stochastic policy',
        ],
        answerIndex: 0,
        explanation: 'The actor selects actions; the critic estimates value so the update has a lower-variance advantage signal.',
      },
    ],
    labs: [
      {
        id: 'critic-baseline',
        title: 'Find the sign flip',
        prompt: 'Move critic value above and below return and watch the actor update switch from reinforcing to discouraging.',
        successCriteria: 'You can explain why the critic changes update sign without choosing actions itself.',
      },
    ],
  },
  'ppo-clipped-policy-gradient': {
    quiz: [
      {
        id: 'positive-advantage-upper-clip',
        prompt: 'With a positive advantage, which PPO ratio side can be clipped?',
        choices: [
          'A ratio above 1 + epsilon because extra probability increase stops getting extra objective gain',
          'Only a ratio below 1 - epsilon because positive advantages always punish actions',
          'No ratio can be clipped when the advantage is positive',
        ],
        answerIndex: 0,
        explanation: 'For positive advantage, the unclipped objective keeps growing with ratio, so PPO caps the useful gain at the upper clipping bound.',
      },
      {
        id: 'negative-advantage-lower-clip',
        prompt: 'With a negative advantage, which ratio side is dangerous enough to activate PPO clipping?',
        choices: [
          'A ratio below 1 - epsilon because it would reduce the bad action too aggressively',
          'A ratio above 1 + epsilon because it always improves the policy',
          'Only a ratio exactly equal to 1',
        ],
        answerIndex: 0,
        explanation: 'For negative advantage, the minimum selects the more conservative objective when the ratio falls too far below the lower bound.',
      },
      {
        id: 'ratio-definition',
        prompt: 'What does the PPO policy ratio r_t compare?',
        choices: [
          'The new policy probability for the sampled action divided by the old collection-policy probability',
          'The critic value divided by the immediate reward',
          'The training loss divided by the validation loss',
        ],
        answerIndex: 0,
        explanation: 'PPO is an on-policy-style update that reuses sampled actions by comparing new action probability with the behavior policy probability.',
      },
      {
        id: 'ppo-not-proof',
        prompt: 'What is a PPO clipping misconception?',
        choices: [
          'Thinking clipping proves monotonic policy improvement without KL, entropy, or value-function checks',
          'Computing both unclipped and clipped surrogate terms',
          'Using advantages from an actor-critic baseline',
        ],
        answerIndex: 0,
        explanation: 'PPO clipping is a practical guardrail, not the full trust-region guarantee; diagnostics still matter.',
      },
      {
        id: 'kl-monitoring',
        prompt: 'Why monitor approximate KL during PPO training?',
        choices: [
          'To detect policy drift that can accumulate even when individual clipped terms look acceptable',
          'To replace the advantage calculation entirely',
          'To make the old policy probability always equal one',
        ],
        answerIndex: 0,
        explanation: 'The clipped objective only constrains sampled ratio gains locally, while the full policy distribution can still drift across updates.',
      },
    ],
    labs: [
      {
        id: 'clip-sign-audit',
        title: 'Audit clipping by advantage sign',
        prompt: 'Set a positive advantage and push the ratio above 1 + epsilon, then set a negative advantage and push the ratio below 1 - epsilon.',
        successCriteria: 'You can state which side clipped in each case and compute both surrogate candidates for one minibatch row.',
      },
    ],
  },
  'reward-shaping': {
    quiz: [
      {
        id: 'sparse-vs-shaped',
        prompt: 'Why add a shaping reward to a sparse-reward task?',
        choices: [
          'To provide denser learning feedback while the real goal remains sparse',
          'To replace the environment objective with an easier one',
          'To remove the need for discounting or value estimates',
        ],
        answerIndex: 0,
        explanation: 'Shaping gives the agent intermediate feedback, but the task reward should still define success.',
      },
      {
        id: 'potential-based-safety',
        prompt: 'What is the main reason to prefer potential-based shaping?',
        choices: [
          'It can guide progress without changing which policy is optimal',
          'It always makes rewards larger than zero',
          'It turns policy gradients into supervised learning',
        ],
        answerIndex: 0,
        explanation: 'Potential-based shaping rewards progress between states and is designed to preserve the intended optimum.',
      },
    ],
    labs: [
      {
        id: 'sparse-to-dense',
        title: 'Turn sparse reward into a hint',
        prompt: 'Move the next state toward and away from the goal and compare task reward, shaping bonus, and total signal.',
        successCriteria: 'You can identify when shaping helps exploration and when it would risk changing the objective.',
      },
    ],
  },
  'diffusion-basics': {
    quiz: DIFFUSION_BASICS_QUIZ,
    labs: [
      {
        id: 'noise-prediction-error',
        title: 'Tune the noise prediction',
        prompt: 'Move timestep and prediction error to see how noisy sample and denoised estimate diverge from the clean signal.',
        successCriteria: 'You can explain why a better noise estimate produces a cleaner recovered sample.',
      },
    ],
  },
  'diffusion-sampling': {
    quiz: DIFFUSION_SAMPLING_QUIZ,
    labs: [
      {
        id: 'compare-samplers',
        title: 'Compare sampler paths',
        prompt: 'Switch between DDPM, DDIM, and flow/ODE while changing step count and prediction quality.',
        successCriteria: 'You can describe how stochasticity, step count, and prediction quality change the path from noise to sample.',
      },
    ],
  },
  'classifier-free-guidance': {
    quiz: CLASSIFIER_FREE_GUIDANCE_QUIZ,
    labs: [
      {
        id: 'guidance-scale-sweep',
        title: 'Sweep guidance scale',
        prompt: 'Move guidance scale from low to high and compare prompt match, diversity, and artifact risk.',
        successCriteria: 'You can explain why a moderate scale can be better than the maximum scale.',
      },
    ],
  },
  'unet-vs-dit': {
    quiz: UNET_VS_DIT_QUIZ,
    labs: [
      {
        id: 'patch-cost',
        title: 'Inspect patch-token cost',
        prompt: 'Change resolution and patch size and watch token count and attention-pair cost move.',
        successCriteria: 'You can explain why smaller patches improve detail but increase transformer attention cost.',
      },
    ],
  },
  'model-debugging': {
    quiz: [
      {
        id: 'check-order',
        prompt: 'What should you verify first when an incident appears only in production?',
        choices: [
          'The data and serving stages that changed from validation conditions',
          'Only the last trained checkpoint',
          "Only the final confusion matrix on today's batch",
        ],
        answerIndex: 0,
        explanation: 'A disciplined pipeline check is needed to localize whether drift comes from data, serving, or training.',
      },
      {
        id: 'slice-target',
        prompt: 'If one subgroup has much higher error, the first interpretation is usually:',
        choices: [
          'A local failure mode that may be hidden in aggregate metrics',
          'An unrelated random artifact with no operational impact',
          'A model architecture mismatch that always requires bigger capacity',
        ],
        answerIndex: 0,
        explanation: 'Slicing often reveals failures that global summaries smooth over.',
      },
      {
        id: 'intervention-safety',
        prompt: 'Why is a single global threshold tweak risky as a first fix?',
        choices: [
          'It can hide a subgroup error and increase inequality across traffic slices',
          'It is equivalent to changing the data split strategy',
          'It never affects precision or recall',
        ],
        answerIndex: 0,
        explanation: 'Threshold tuning redistributes errors, so it can fix one slice while worsening others.',
      },
    ],
    labs: [
      {
        id: 'debug-loop',
        title: 'Run a constrained debugging loop',
        prompt: 'Choose a scenario, run checks by stage, pick the highest-support root-cause, and select one targeted intervention.',
        successCriteria: 'You should produce one slice with improved recall and describe why the root-cause hypothesis aligned with a later signal.',
      },
    ],
  },
  'model-interpretability': {
    quiz: [
      {
        id: 'why-ablation',
        prompt: 'What does a feature ablation-style attribution primarily measure in this lesson?',
        choices: [
          'How much predictions degrade when a feature is replaced by a reference value',
          'The exact causal effect of changing one feature',
          'How to remove all uncertainty from the model',
        ],
        answerIndex: 0,
        explanation: 'Ablation-style checks are directional and help rank features; they are not proof of causality.',
      },
      {
        id: 'local-meaning',
        prompt: 'In a local explanation panel, the largest signed contribution is best interpreted as:',
        choices: [
          'The strongest directional driver for this example under the toy model',
          'A guarantee of causal influence in every domain',
          'Proof that the model has no bias',
        ],
        answerIndex: 0,
        explanation: 'Large local contribution means strong influence in the surrogate, with model-dependent caveats.',
      },
      {
        id: 'counterfactual',
        prompt: 'What is the purpose of the counterfactual perturbation in this lesson?',
        choices: [
          'To test decision stability under a controlled input change',
          'To tune weights for all future samples',
          'To prove the explanation is causal',
        ],
        answerIndex: 0,
        explanation: 'Counterfactual checks test whether small input changes materially flip decisions.',
      },
    ],
    labs: [
      {
        id: 'compare-modes',
        title: 'Find an unstable attribution mode',
        prompt: 'Enable correlation mode and compare top attributions before and after a counterfactual perturbation.',
        successCriteria: 'You can explain one case where explanation confidence drops when correlation increases.',
      },
    ],
  },
  'model-monitoring': {
    quiz: [
      {
        id: 'monitor-priority',
        prompt: 'Which signal in this lesson most directly distinguishes input drift from label drift behavior?',
        choices: [
          'Tracking both input drift and precision/recall together',
          'Precision alone',
          'Recall alone',
        ],
        answerIndex: 0,
        explanation: 'Input drift and label-quality issues can both hurt scores, so comparing multiple signals avoids false attribution.',
      },
      {
        id: 'alert-meaning',
        prompt: 'Why can a stricter alert threshold help and hurt at the same time?',
        choices: [
          'It reduces late misses but can increase false alerts and intervention noise',
          'It always improves model quality',
          'It removes the need to monitor serving metrics',
        ],
        answerIndex: 0,
        explanation: 'Strict alerts catch issues earlier but can fire during normal variance.',
      },
      {
        id: 'playbook-choice',
        prompt: 'When throughput drops and recall drops together, a first response is usually:',
        choices: [
          'Pause rollout, inspect serving contract and upstream sampling, then isolate monitoring signals',
          'Increase threshold complexity immediately',
          'Ignore alerts until the trend continues for 3 weeks',
        ],
        answerIndex: 0,
        explanation: 'Parallel degradation in serving and metrics usually needs triage of contract and data before model updates.',
      },
    ],
    labs: [
      {
        id: 'configure-playbook',
        title: 'Tune monitoring and choose a playbook',
        prompt: 'Set a strictness profile and scenario to create 1-2 active alerts, then choose the best response playbook.',
        successCriteria: 'You should describe which metric triggered first and why the selected playbook matches that risk.',
      },
    ],
  },
  'uncertainty-estimation': {
    quiz: [
      {
        id: 'calibration-vs-confidence',
        prompt: 'What is the distinction between confidence and calibration in this context?',
        choices: [
          'Confidence is uncertainty width; calibration is long-run reliability of probabilities or intervals',
          'They are always equivalent',
          'Calibration is only for classification, confidence only for regression',
        ],
        answerIndex: 0,
        explanation: 'A model can be highly confident and still poorly calibrated.',
      },
      {
        id: 'wide-interval',
        prompt: 'A very wide interval should usually be interpreted as:',
        choices: [
          'A warning about higher uncertainty or distribution mismatch',
          'A guaranteed prediction failure',
          'Proof that the model is overfitting',
        ],
        answerIndex: 0,
        explanation: 'Width is a caution signal; it is useful when interpreted with coverage and abstain policies.',
      },
      {
        id: 'abstain-policy',
        prompt: 'What is the operational role of abstain or defer in uncertainty-aware systems?',
        choices: [
          'Send low-confidence cases to fallback review or broader context',
          'Drop half the data for free speed gains',
          'Replace training labels with a prior mean',
        ],
        answerIndex: 0,
        explanation: 'Abstain policies convert uncertain predictions into explicit review cases.',
      },
    ],
    labs: [
      {
        id: 'coverage-vs-width',
        title: 'Balance width and coverage',
        prompt: 'Increase target coverage and adjust abstain threshold while watching coverage and mean interval width.',
        successCriteria: 'You can keep coverage stable while preventing too much unnecessary abstention.',
      },
    ],
  },
  'model-fairness': {
    quiz: [
      {
        id: 'parity-difference',
        prompt: 'Which objective best focuses on equal selection proportions across groups?',
        choices: [
          'Selection-rate parity',
          'FPR parity',
          'TPR parity',
        ],
        answerIndex: 0,
        explanation: 'Selection parity compares the share of positive predictions across groups.',
      },
      {
        id: 'conflict-rule',
        prompt: 'A practical implication of fairness trade-offs is:',
        choices: [
          'Satisfying one constraint may worsen another',
          'Every fairness metric can be optimized simultaneously at full strength',
          'No threshold changes are needed if one metric is equal',
        ],
        answerIndex: 0,
        explanation: 'Metrics can conflict; governance choices must set priorities.',
      },
      {
        id: 'counterfactual-use',
        prompt: 'When should group-specific thresholds be considered?',
        choices: [
          'When global thresholds cannot meet the chosen fairness and utility trade-off',
          'Never, because they always invalidate comparisons',
          'Only when model training accuracy is below 50%',
        ],
        answerIndex: 0,
        explanation: 'Group thresholds are a controlled tool to rebalance operational parity, but they should be policy-driven.',
      },
    ],
    labs: [
      {
        id: 'objective-target',
        title: 'Match one fairness objective',
        prompt: 'Pick an objective and tune thresholds/shift until your selected objective gap is materially reduced.',
        successCriteria: 'You can explain which other metric moved and why this reflects a trade-off.',
      },
    ],
  },
  'bloom-filter': {
    quiz: [
      {
        id: 'false-positive',
        prompt: 'What kind of mistake can a Bloom filter make?',
        choices: [
          'It can say probably present for an item that was never inserted',
          'It can say definitely absent for an item that was inserted',
          'It can return the stored item value instead of a membership answer',
        ],
        answerIndex: 0,
        explanation: 'A Bloom filter has false positives because different items can set the same bits, but it should not have false negatives if inserts and queries use the same hashes.',
      },
      {
        id: 'definitely-not',
        prompt: 'Why does a zero bit prove an item is definitely absent?',
        choices: [
          'Every inserted item would have set all of its hash positions to 1',
          'Zero bits store the original inserted keys',
          'The filter checks a backing database before answering',
        ],
        answerIndex: 0,
        explanation: 'If any queried hash position is 0, that item could not have been inserted because insertion sets every queried position.',
      },
      {
        id: 'parameter-tradeoff',
        prompt: 'What usually lowers the false-positive probability for a fixed number of inserted items?',
        choices: [
          'More bits with a near-optimal number of hash functions',
          'Fewer bits and as many hash functions as possible',
          'Removing hash functions so only one bit is checked',
        ],
        answerIndex: 0,
        explanation: 'The false-positive rate falls when the bit array is less saturated and k is tuned near (m/n) ln 2.',
      },
    ],
    labs: [
      {
        id: 'force-collision',
        title: 'Force and explain a collision',
        prompt: 'Use the false positive lab to add several words, then query a word that was not added until the filter reports probably present.',
        successCriteria: 'You can list the queried hash positions and explain which inserted words already set those bits.',
      },
    ],
  },
  'grpo-reasoning': {
    quiz: [
      {
        id: 'grpo-samples-group',
        prompt: 'What does GRPO sample for each prompt?',
        choices: [
          'A group of candidate completions from the current or old policy.',
          'One hand-written gold solution from a human annotator.',
          'One critic value for every transformer layer.',
        ],
        answerIndex: 0,
        explanation: 'GRPO creates a group of candidate answers for the same prompt, scores them, and compares each answer against its siblings.',
      },
      {
        id: 'grpo-group-baseline',
        prompt: 'What replaces the learned PPO critic baseline in GRPO?',
        choices: [
          'The average reward of the sampled answers for the same prompt.',
          'The largest token logit in the final answer.',
          'The length of the shortest response in the batch.',
        ],
        answerIndex: 0,
        explanation: 'The group reward statistics become the baseline, avoiding a separate value-function model.',
      },
      {
        id: 'grpo-positive-advantage',
        prompt: 'What should happen to a completion with above-average reward inside its group?',
        choices: [
          'Its solution trace should become more likely under the policy update.',
          'It should be deleted because RL only trains on wrong answers.',
          'It should be assigned to the reference model instead of the policy model.',
        ],
        answerIndex: 0,
        explanation: 'Above-average reward gives positive advantage, so the update reinforces the sampled trace.',
      },
      {
        id: 'grpo-negative-advantage',
        prompt: 'What should happen to a completion with below-average reward inside its group?',
        choices: [
          'Its trace should be suppressed by the policy update.',
          'Its reward should be copied to all other samples.',
          'It should become the new group baseline.',
        ],
        answerIndex: 0,
        explanation: 'Below-average reward gives negative advantage, lowering the probability of that sampled behavior.',
      },
      {
        id: 'grpo-no-contrast',
        prompt: 'Why are all-correct or all-wrong GRPO groups weak training examples?',
        choices: [
          'There is little contrast, so relative advantage gives little or risky direction.',
          'They always make the KL term exactly zero.',
          'They prevent the model from sampling any future answers.',
        ],
        answerIndex: 0,
        explanation: 'GRPO learns from within-group differences. If all rewards match, the normalized advantage is near zero; if all are wrong with tiny differences, the least bad wrong answer may be reinforced.',
      },
      {
        id: 'grpo-kl-purpose',
        prompt: 'Why does GRPO use PPO-style clipping and a KL guardrail?',
        choices: [
          'To keep the policy update bounded and close to a reference model while optimizing reward.',
          'To force every answer to have exactly the same token length.',
          'To remove the need for reward functions.',
        ],
        answerIndex: 0,
        explanation: 'Clipping limits large probability-ratio moves, and KL discourages the policy from drifting too far toward reward-hacking behavior.',
      },
      {
        id: 'r1-zero-purpose',
        prompt: 'What did DeepSeek-R1-Zero test?',
        choices: [
          'Whether reasoning behaviors could emerge from large-scale RL applied directly to a base model.',
          'Whether sparse attention can remove all KV-cache memory.',
          'Whether a critic model can replace a language model.',
        ],
        answerIndex: 0,
        explanation: 'R1-Zero is the pure-RL experiment: apply GRPO-style RL without an SFT cold-start stage and observe emergent reasoning behaviors.',
      },
      {
        id: 'r1-cold-start',
        prompt: 'Why did DeepSeek-R1 add cold-start data after the R1-Zero experiment?',
        choices: [
          'To improve readability, formatting, and user-facing behavior before further RL.',
          'To make the model smaller than every distilled student.',
          'To remove rule-based rewards from math and code tasks.',
        ],
        answerIndex: 0,
        explanation: 'R1-Zero showed strong reasoning but had readability and language-mixing issues. Cold-start data helped shape usable response format.',
      },
      {
        id: 'grpo-reward-design',
        prompt: 'Which reward type is especially reliable for math and code reasoning tasks?',
        choices: [
          'Rule-based correctness checks such as exact answers or compiler tests.',
          'A reward for using as many tags as possible.',
          'A reward based only on response length.',
        ],
        answerIndex: 0,
        explanation: 'Math and code often have checkable outcomes, making exact answer checkers and test execution useful reward sources.',
      },
      {
        id: 'grpo-distillation',
        prompt: 'What is the role of distillation after training a large reasoning model?',
        choices: [
          'Use the large model to generate reasoning data that smaller models can imitate.',
          'Run GRPO without any rewards.',
          'Convert all policy-gradient updates into KV-cache compression.',
        ],
        answerIndex: 0,
        explanation: 'DeepSeek-R1 generated data for smaller Qwen and Llama based dense models, transferring useful reasoning patterns through supervised fine-tuning.',
      },
    ],
    labs: [
      {
        id: 'mini-grpo-advantages',
        title: 'Compute group advantages',
        prompt: 'Open mini-grpo exercises 01 and 02, then implement mean reward, population standard deviation, and group-relative advantages.',
        successCriteria: 'The tests show positive advantages for above-average rewards, negative advantages for below-average rewards, and zero signal when variance is zero.',
      },
      {
        id: 'mini-grpo-clipping-kl',
        title: 'Bound the policy update',
        prompt: 'Open mini-grpo exercises 05, 06, and 07, then implement policy ratio, clipped surrogate, and KL-penalized objective.',
        successCriteria: 'The clipped objective handles positive and negative advantages correctly, and KL reduces the reward objective.',
      },
      {
        id: 'mini-grpo-distillation',
        title: 'Filter teacher outputs',
        prompt: 'Open mini-grpo exercises 09 and 10, then detect useful groups and keep high-reward samples for distillation.',
        successCriteria: 'Only contrastive groups count as useful, and only samples above the reward threshold remain in the student dataset.',
      },
    ],
  },
  'dapo-reasoning-rl': {
    quiz: [
      {
        id: 'dapo-builds-on-grpo',
        prompt: 'What training setup does DAPO build on?',
        choices: [
          'GRPO-style reasoning RL that samples multiple answers for the same prompt and compares them within the group.',
          'A pure supervised fine-tuning loop with one gold answer per prompt.',
          'A retrieval-only system that never updates the language model policy.',
        ],
        answerIndex: 0,
        explanation: 'DAPO keeps the group-relative idea but adds practical training fixes for large-scale long-CoT RL.',
      },
      {
        id: 'dapo-four-techniques',
        prompt: 'Which list contains the four DAPO techniques emphasized in the lesson?',
        choices: [
          'Beam search, KV-cache quantization, LoRA rank search, and speculative decoding.',
          'Clip-Higher, Dynamic Sampling, Token-Level Policy Gradient Loss, and Overlong Reward Shaping.',
          'RAG chunking, reranking, calibration, and abstention thresholds.',
        ],
        answerIndex: 1,
        explanation: 'The DAPO recipe combines batch filtering, asymmetric clipping, token-level loss aggregation, and length reward shaping.',
      },
      {
        id: 'dapo-dynamic-sampling',
        prompt: 'Why does Dynamic Sampling drop all-correct and all-wrong groups?',
        choices: [
          'Those groups are too short to fit in the context window.',
          'Those groups always violate the KL penalty.',
          'Those groups have little or no within-group reward contrast, so the group-relative update carries weak signal.',
        ],
        answerIndex: 2,
        explanation: 'Group-relative learning needs siblings that disagree. Mixed groups produce positive and negative advantages.',
      },
      {
        id: 'dapo-clip-higher',
        prompt: 'What does Clip-Higher change relative to symmetric PPO-style clipping?',
        choices: [
          'It raises the upper ratio bound while keeping the lower bound controlled.',
          'It removes clipping from every negative-advantage token.',
          'It clips reward values instead of policy ratios.',
        ],
        answerIndex: 0,
        explanation: 'Clip-Higher decouples the lower and upper clip epsilons so successful traces have more room to grow.',
      },
      {
        id: 'dapo-entropy-collapse',
        prompt: 'What failure mode is Clip-Higher meant to help mitigate?',
        choices: [
          'Tokenizer vocabulary overflow during BPE merges.',
          'Entropy collapse, where exploration and trace diversity shrink too much during long-CoT RL.',
          'A missing final answer regex in supervised data.',
        ],
        answerIndex: 1,
        explanation: 'If positive updates are blocked too aggressively, diverse successful reasoning behaviors can disappear.',
      },
      {
        id: 'dapo-token-level-loss',
        prompt: 'Why is token-level policy-gradient loss useful for long reasoning traces?',
        choices: [
          'It makes every token receive a different human label.',
          'It prevents the model from generating final answers.',
          'A long completion is many token decisions, so token-level aggregation gives denser credit than one sequence-sized block.',
        ],
        answerIndex: 2,
        explanation: 'Long-CoT RL benefits from policy-gradient terms across all valid response tokens.',
      },
      {
        id: 'dapo-overlong-shaping',
        prompt: 'What problem does Overlong Reward Shaping address?',
        choices: [
          'A hard reward cliff around the max response length can be noisy for slightly overlong but otherwise useful traces.',
          'Correct answers always need to be shorter than one token.',
          'Sampling groups cannot contain any long answers.',
        ],
        answerIndex: 0,
        explanation: 'A smoother length penalty gives the model a more informative concision signal near the boundary.',
      },
      {
        id: 'dapo-frontier-prompts',
        prompt: 'Which prompt difficulty usually gives the strongest group-relative signal?',
        choices: [
          'Trivial prompts where every sample is correct.',
          'Frontier prompts where some samples succeed and some fail.',
          'Impossible prompts where every sample is wrong.',
        ],
        answerIndex: 1,
        explanation: 'The best groups are near the current model frontier because they produce mixed outcomes.',
      },
      {
        id: 'dapo-recipe-not-formula',
        prompt: 'Why is DAPO best understood as a recipe rather than a single equation?',
        choices: [
          'It contains no mathematical objective at all.',
          'It only applies at inference time after training is complete.',
          'Sampling, reward shaping, clipping, and loss aggregation all interact to determine training behavior.',
        ],
        answerIndex: 2,
        explanation: 'DAPO depends on how batches are constructed, how rewards are shaped, how ratios are clipped, and how token losses are aggregated.',
      },
      {
        id: 'dapo-health-dashboard',
        prompt: 'Which dashboard signals are useful for diagnosing DAPO-style training?',
        choices: [
          'Effective groups, entropy, clip fraction, overlong rate, and token-gradient count.',
          'Only the number of model parameters.',
          'Only the static prompt template length.',
        ],
        answerIndex: 0,
        explanation: 'Practical RL training needs live diagnostics for signal quality, exploration, clipping pressure, length behavior, and gradient density.',
      },
    ],
    labs: [
      {
        id: 'mini-dapo-group-signal',
        title: 'Filter useful rollout groups',
        prompt: 'Open mini-dapo exercises 01, 02, 03, and 08, then implement group accuracy, dynamic sampling, normalized advantages, and effective batch counts.',
        successCriteria: 'All-correct and all-wrong groups are rejected, mixed groups remain, and normalized advantages become zero when reward variance is zero.',
      },
      {
        id: 'mini-dapo-objective',
        title: 'Implement the DAPO update pieces',
        prompt: 'Open mini-dapo exercises 04, 05, and 09, then implement Clip-Higher ranges, token-level loss, and the toy DAPO sample objective.',
        successCriteria: 'The objective combines shaped reward, group advantage, token ratios, and asymmetric clipping with the expected sign.',
      },
      {
        id: 'mini-dapo-training-health',
        title: 'Diagnose long-CoT training health',
        prompt: 'Open mini-dapo exercises 06, 07, and 10, then implement overlong reward shaping, entropy collapse detection, and training health labels.',
        successCriteria: 'The dashboard flags low signal, entropy collapse, too-overlong batches, and healthy batches according to the specified thresholds.',
      },
    ],
  },
  'coconut-latent-reasoning': {
    quiz: [
      {
        id: 'coconut-intermediate-state',
        prompt: 'What does Coconut feed back during latent mode?',
        choices: [
          'The previous position last hidden state.',
          'Only the final answer token.',
          'A fixed pause token embedding.',
        ],
        answerIndex: 0,
        explanation: 'Coconut bypasses token decoding in latent mode and uses the last hidden state as the next input embedding.',
      },
      {
        id: 'coconut-mode-markers',
        prompt: 'What do <bot> and <eot> mark in Coconut-style inference?',
        choices: [
          'The start and stop of answer grading.',
          'The boundary of the latent reasoning region.',
          'A switch from GPU to CPU execution.',
        ],
        answerIndex: 1,
        explanation: 'Special tokens delimit where continuous thoughts are inserted before text generation resumes.',
      },
      {
        id: 'coconut-lm-head',
        prompt: 'What is skipped when the model is in latent mode?',
        choices: [
          'The transformer stack.',
          'All future answer tokens.',
          'Decoding the hidden state through the LM head into a word token.',
        ],
        answerIndex: 2,
        explanation: 'The transformer still runs, but the hidden state is fed back directly instead of first becoming a token.',
      },
      {
        id: 'coconut-pause-token',
        prompt: 'Why is a continuous thought different from a pause token?',
        choices: [
          'A continuous thought is context-dependent, while a pause token uses a learned token embedding.',
          'Pause tokens can only appear in encoder-only models.',
          'Continuous thoughts are visible natural language words.',
        ],
        answerIndex: 0,
        explanation: 'Pause tokens add extra positions with token embeddings; Coconut uses the current hidden state as the next input.',
      },
      {
        id: 'coconut-curriculum',
        prompt: 'Why does Coconut train with a curriculum that replaces early CoT steps?',
        choices: [
          'To remove all language modeling loss.',
          'To gradually teach latent states to support future reasoning and answers.',
          'To prevent the model from seeing chain-of-thought data.',
        ],
        answerIndex: 1,
        explanation: 'The curriculum starts from text reasoning data and progressively moves early reasoning steps into latent space.',
      },
      {
        id: 'coconut-loss-mask',
        prompt: 'Where is the Coconut training loss applied in the curriculum view?',
        choices: [
          'Only to question tokens.',
          'To latent thought positions directly as if they were words.',
          'To the remaining text targets after the latent thoughts.',
        ],
        answerIndex: 2,
        explanation: 'Question and latent positions are masked; the latent states are optimized by their usefulness for future text prediction.',
      },
      {
        id: 'coconut-delayed-commitment',
        prompt: 'What does delayed commitment mean in the Coconut lesson?',
        choices: [
          'Keeping several branches softly active before committing to one answer path.',
          'Waiting to load the model weights until after decoding.',
          'Always choosing the shortest visible answer.',
        ],
        answerIndex: 0,
        explanation: 'The lesson visualizes latent states as broad branch distributions that sharpen as uncertainty drops.',
      },
      {
        id: 'coconut-visible-vs-compute',
        prompt: 'How should we account for latent thoughts at inference time?',
        choices: [
          'They cost no computation because they are not visible text.',
          'They reduce visible text tokens, but each latent thought still requires a forward computation step.',
          'They are counted as training examples, not inference steps.',
        ],
        answerIndex: 1,
        explanation: 'Latent thoughts may reduce generated text, but they are still positions processed by the model.',
      },
      {
        id: 'coconut-faithfulness-risk',
        prompt: 'What is the main faithfulness caveat for latent thoughts?',
        choices: [
          'They always decode to exactly one human-readable token.',
          'They make reward hacking impossible.',
          'They may be opaque or shortcut-like rather than faithful reasoning states.',
        ],
        answerIndex: 2,
        explanation: 'A latent vector can help prediction without being an interpretable or causally faithful reasoning step.',
      },
      {
        id: 'coconut-faithfulness-test',
        prompt: 'Which test best checks whether a latent thought is reasoning-critical?',
        choices: [
          'Perturb or remove it and see whether branch distributions or answers change.',
          'Count how many characters are in the final answer.',
          'Ignore probes because vectors can never be studied.',
        ],
        answerIndex: 0,
        explanation: 'Causal interventions such as perturbations and ablations are stronger evidence than just decoding a nearest-token probe.',
      },
    ],
    labs: [
      {
        id: 'trace-latent-feedback',
        title: 'Trace language mode versus latent mode',
        prompt: 'Use the Coconut lesson controls to compare CoT and Coconut, then explain when the LM head is used and when h_t becomes x_t+1.',
        successCriteria: 'You can identify the hidden-state feedback edge and explain why the visible token count differs from the compute-step count.',
      },
      {
        id: 'measure-delayed-commitment',
        title: 'Measure delayed commitment',
        prompt: 'Switch to the planning task, vary latent thoughts from 1 to 8, and watch branch entropy and commitment step change.',
        successCriteria: 'You can explain why high early entropy and low final entropy mean the model delayed a hard branch choice.',
      },
      {
        id: 'run-mini-coconut',
        title: 'Implement the mini-coconut mechanics',
        prompt: 'Open mini-coconut exercises 01 through 10 and implement mode switching, latent feedback, masked loss, entropy, perturbation, token budgets, and nearest-token probes.',
        successCriteria: 'The exercise tests pass and you can connect each function to the lesson panel it supports.',
      },
    ],
  },
  'reasoning-rlvr-grpo': {
    quiz: [
      {
        id: 'reasoning-vs-rlvr-grpo',
        prompt: 'How does Group Relative Policy Optimization (GRPO) reduce GPU memory compared to standard Proximal Policy Optimization (PPO)?',
        choices: [
          'By removing the critic network entirely and estimating baselines from the average reward of a group of outputs.',
          'By running attention weights in 4-bit precision instead of 16-bit.',
          'By generating all tokens sequentially in a single batch thread.',
        ],
        answerIndex: 0,
        explanation: 'GRPO eliminates the critic network (which is typically as large as the actor) by using the average reward of a group of samples for the same prompt as the baseline.',
      },
      {
        id: 'grpo-advantage-baseline',
        prompt: 'In the GRPO advantage formula $A_i = \\frac{r_i - \\mu}{\\sigma + \\epsilon}$, what represents the dynamic baseline?',
        choices: [
          'The standard deviation of the policy outputs.',
          'The mean reward of all candidate solutions in the current group.',
          'The clipping parameter used to prevent policy collapse.',
        ],
        answerIndex: 1,
        explanation: 'The mean reward of the group acts as a dynamic baseline. Outputs that perform better than the group average get a positive advantage, while those below average get a negative advantage.',
      },
      {
        id: 'prm-vs-orm',
        prompt: 'What is the primary operational advantage of Process Reward Models (PRMs) over Outcome Reward Models (ORMs)?',
        choices: [
          'PRMs can be computed deterministically using standard regex patterns.',
          'PRMs require no training data since they use compiler verification.',
          'PRMs mitigate alignment bugs by assigning credit to individual reasoning steps, reducing reward hacking.',
        ],
        answerIndex: 2,
        explanation: 'PRMs evaluate intermediate steps of reasoning, making it harder for a model to receive a high reward for a wrong trace that happens to output the correct final answer.',
      },
      {
        id: 'verifiable-rewards',
        prompt: 'Why are verifiable rewards (like compilers or math parsers) preferred over neural reward models for scaling reasoning?',
        choices: [
          'They are deterministic, free from reward hacking (no judge exploitation), and cheap to compute.',
          'They always generalize to creative writing and open-ended chatbot queries.',
          'They automatically double the context window of the model during training.',
        ],
        answerIndex: 0,
        explanation: 'Verifiable rewards are rule-based checks that cannot be hacked or exploited by style tricks, unlike neural reward models which are vulnerable to adversarial formatting.',
      },
      {
        id: 'cold-start-role',
        prompt: 'What is the function of cold-start SFT data in the DeepSeek-R1 training pipeline?',
        choices: [
          'To compress the KV cache dimension using low-rank projection matrices.',
          'To seed the model with basic formatting (like using thinking tags) and prevent early RL divergence.',
          'To optimize the expert routing weights and balance active compute load.',
        ],
        answerIndex: 1,
        explanation: 'Cold-start SFT data provides a template of structured thinking traces so the model does not diverge or produce unreadable streams during early reinforcement learning.',
      },
      {
        id: 'format-reward-hacking',
        prompt: 'What occurs if the format reward weight is set too high relative to correctness rewards during RL?',
        choices: [
          'The model generalizes better to unseen domains.',
          'The model outputs only Chinese and English code blocks.',
          'The model learns to generate empty or repetitive thinking tags to exploit the format scorer without solving the task.',
        ],
        answerIndex: 2,
        explanation: 'When formatting rewards dominate, the model exploits the parser by producing long, useless thinking sequences without actually thinking, a classic case of reward hacking.',
      },
      {
        id: 'all-negative-group-issue',
        prompt: 'What is the training risk when all generated solutions in a GRPO group fail to solve the task (all-negative group)?',
        choices: [
          'Normalization will still assign positive advantages to the "least incorrect" solutions, reinforcing flawed reasoning.',
          'The standard deviation falls to zero, causing the optimizer to divide by zero and crash the training run.',
          'The actor policy is automatically reset to the reference SFT model.',
        ],
        answerIndex: 0,
        explanation: 'Since GRPO normalizes advantages relative to the group, the "least bad" incorrect solution will receive a positive advantage and its incorrect steps will be reinforced.',
      },
      {
        id: 'length-penalty-tradeoff',
        prompt: 'What trade-off is introduced by adding a token length penalty to the RL reward function?',
        choices: [
          'It reduces model size but increases GPU communication latency.',
          'It controls overthinking and generation costs but can truncate the long reasoning paths needed for hard problems.',
          'It increases accuracy on creative tasks but lowers performance on programming tasks.',
        ],
        answerIndex: 1,
        explanation: 'A length penalty keeps traces concise and lowers serving latency, but if it is too strong, it prevents the model from doing the depth of thinking required to solve complex tasks.',
      },
      {
        id: 'overthinking-emergence',
        prompt: 'Why does "overthinking" (excessively long or circular traces) emerge in models trained with pure RL without length constraints?',
        choices: [
          'The tokenizer is biased toward long subword merges.',
          'The learning rate is too high, causing gradient updates to step past local minima.',
          'The model learns that longer reasoning paths correlate with a higher probability of stumbling upon a correct answer.',
        ],
        answerIndex: 2,
        explanation: 'Without penalties for verbosity, the policy exploits test-time compute by writing out extensive backtracking and verification loops, which increases its odds of finding the correct answer.',
      },
      {
        id: 'rejection-sampling-ft',
        prompt: 'How does rejection sampling fine-tuning (RFT) differ from standard reinforcement learning?',
        choices: [
          'RFT is offline: it generates multiple outputs, filters the correct ones, and trains on them using standard cross-entropy.',
          'RFT uses an online critic network to update policy parameters on every token generation.',
          'RFT uses dynamic learning rates based on the KL divergence penalty.',
        ],
        answerIndex: 0,
        explanation: 'RFT generates multiple completions off-policy, selects the correct ones, and applies supervised learning. RL updates the policy online based on immediate rewards or advantages.',
      },
      {
        id: 'distillation-tradeoffs',
        prompt: 'When distilling reasoning traces from a large teacher (like R1) into a small student, what is a key limitation?',
        choices: [
          'The student parameter count must be identical to the teacher parameter count.',
          'The student learns the formatting style and tricks but may lack the raw capacity to generalize to harder, unseen reasoning tasks.',
          'The SFT loss function must use reinforcement learning advantages instead of token predictions.',
        ],
        answerIndex: 1,
        explanation: 'Distillation is highly effective for transfer of formatting style and basic logic, but smaller models often suffer from a "generalization gap" when confronted with hard math beyond their capacity.',
      },
      {
        id: 'kl-divergence-penalty',
        prompt: 'What is the purpose of adding a KL divergence penalty between the active RL policy and the reference SFT model?',
        choices: [
          'To calculate the accuracy difference between training and test sets.',
          'To encourage the model to explore new languages and formatting tags.',
          'To prevent the policy from shifting too far from the base model, avoiding language collapse or trace corruption.',
        ],
        answerIndex: 2,
        explanation: 'The KL penalty acts as a regularization constraint, preventing the active policy from drifting too far from the reference model, which keeps the reasoning trace structured and natural.',
      },
    ],
    labs: [
      {
        id: 'grpo-group-advantage-lab',
        title: 'Compute GRPO group advantages',
        prompt: 'Implement total_reward, group_advantages, and update_direction in the browser Python lab.',
        successCriteria: [
          'Rewards match the reference formula',
          'Advantages are group-normalized',
          'Update directions follow the advantage threshold',
          'Learner can explain which candidate is reinforced',
        ],
      },
      {
        id: 'tune-grpo-rewards',
        title: 'Balance reasoning rewards',
        prompt: 'Configure correctness, format, and language weights to achieve 90% accuracy without triggering overthinking or formatting hacks.',
        successCriteria: 'You can find a reward configuration where accuracy is high, formatting is correct, and trace length remains under 500 tokens.',
      },
      {
        id: 'solve-all-negatives',
        title: 'Mitigate all-negative groups',
        prompt: 'Identify a prompt where all initial samples are incorrect, and adjust temperature or group size to discover at least one correct trace.',
        successCriteria: 'You can explain how increasing group size or temperature helps recover from all-negative training groups.',
      },
      {
        id: 'process-credit',
        title: 'Step-level credit assignment',
        prompt: 'Compare outcome-based vs step-based rewards on a multi-step proof and identify which step receives incorrect credit under an ORM.',
        successCriteria: 'You can pinpoint the exact line where the proof failed but still received a positive score under ORM evaluation.',
      },
      {
        id: 'detect-reward-hacking',
        title: 'Detect formatting exploitation',
        prompt: 'Trigger a formatting hack by setting format reward above 1.0, and capture a generated trace that contains empty thinking tags.',
        successCriteria: 'You can capture a trace where the model outputs `<think></think>` multiple times to collect rewards without reasoning.',
      },
      {
        id: 'distill-student',
        title: 'Verify student distillation',
        prompt: 'Train a 1.5B student on teacher traces and evaluate its accuracy vs trace length on a test set of mathematical word problems.',
        successCriteria: 'You can explain where the student mimics the teacher\'s format but fails to solve the underlying math.',
      },
      {
        id: 'prevent-kl-drift',
        title: 'Control KL divergence',
        prompt: 'Tune the KL coefficient during online policy updates and find the threshold where the model\'s language consistency collapses.',
        successCriteria: 'You can report the KL penalty value that preserves natural language structure while allowing the model to optimize rewards.',
      },
    ],
  },
  'test-time-compute-thinking-budgets': {
    quiz: [
      {
        id: 'ttc-vs-training',
        prompt: 'What is the key operational difference between training-time and inference-time compute scaling?',
        choices: [
          'Training-time scaling permanently improves the model via retraining; inference-time scaling spends more compute per query at generation time.',
          'Inference-time scaling permanently changes model weights; training-time scaling only affects sampling temperature.',
          'Training-time and inference-time scaling are identical in cost structure.',
        ],
        answerIndex: 0,
        explanation: 'Training-time scaling changes model parameters and requires retraining from scratch. Inference-time (test-time) scaling spends additional compute at query time through strategies like more samples or longer reasoning traces.',
      },
      {
        id: 'bon-oracle-bound',
        prompt: 'In Best-of-N sampling, what is the oracle bound and why is it rarely achieved in practice?',
        choices: [
          'The oracle bound is the accuracy achievable when the best sample is always identified; it requires a perfect verifier, which is typically unavailable.',
          'The oracle bound is the accuracy of the smallest sample N=1, used as a baseline.',
          'The oracle bound equals training accuracy and is always achieved when N exceeds the number of model parameters.',
        ],
        answerIndex: 0,
        explanation: 'The oracle bound is the theoretical maximum accuracy when the true best answer is always selected from N samples. In practice, verifiers (reward models) are imperfect, so the actual gain falls below the oracle bound.',
      },
      {
        id: 'bon-cost-scaling',
        prompt: 'How does the cost of Best-of-N sampling scale with the number of samples N?',
        choices: [
          'Cost scales linearly as O(N × L) where L is the average completion length.',
          'Cost scales logarithmically because GPU batch parallelism compresses overhead.',
          'Cost is constant because all N samples share the same forward pass.',
        ],
        answerIndex: 0,
        explanation: 'Each of the N samples requires an independent forward pass generating approximately L tokens, so total cost is O(N × L). Parallelism can reduce wall-clock time but not total compute.',
      },
      {
        id: 'beam-search-prm',
        prompt: 'Why does tree search (beam search) require a Process Reward Model (PRM) rather than an Outcome Reward Model (ORM)?',
        choices: [
          'Beam search needs to score partial reasoning paths at intermediate nodes, which ORMs cannot do since they only evaluate final answers.',
          'PRMs are cheaper to train than ORMs and reduce memory requirements for the search tree.',
          'ORMs require beam search to function correctly since they cannot process token-by-token outputs.',
        ],
        answerIndex: 0,
        explanation: 'Beam search prunes branches at intermediate steps. A PRM can score each reasoning step independently, enabling early pruning of bad paths. An ORM only scores complete answers, which makes it useless for mid-search evaluation.',
      },
      {
        id: 'budget-forcing-tradeoff',
        prompt: 'What happens when a fixed thinking-token budget is set too low for a complex reasoning task?',
        choices: [
          'The reasoning chain is truncated before completion, causing a drop in accuracy on hard problems.',
          'The model automatically extends the context window to fit the reasoning trace.',
          'Low budgets always improve accuracy by forcing the model to be more concise.',
        ],
        answerIndex: 0,
        explanation: 'A hard budget cap truncates the reasoning trace mid-chain. For complex multi-step problems that require long thinking, this leads to incomplete reasoning and degraded accuracy.',
      },
      {
        id: 'adaptive-budget-benefit',
        prompt: 'What is the primary benefit of adaptive thinking budgets over fixed caps?',
        choices: [
          'Adaptive budgets allocate short thinking traces to easy queries and long ones to hard queries, reducing total serving cost without sacrificing accuracy.',
          'Adaptive budgets always use the maximum available context window for every query.',
          'Adaptive budgets eliminate the need for any verifier or reward model.',
        ],
        answerIndex: 0,
        explanation: 'A fixed cap either wastes tokens on simple queries or truncates complex ones. Adaptive budgets use a difficulty signal to right-size thinking compute per query, improving cost-efficiency.',
      },
      {
        id: 'react-tool-use',
        prompt: 'In the ReAct (Reason + Act) pattern, how does tool use change the compute profile of a reasoning task?',
        choices: [
          'The model externalizes sub-tasks to deterministic tools, saving reasoning tokens at the cost of additional tool-call latency.',
          'ReAct always increases total token count by adding Observation tokens that duplicate model knowledge.',
          'Tool calls replace the model entirely; the model generates no tokens in ReAct pipelines.',
        ],
        answerIndex: 0,
        explanation: 'ReAct lets the model delegate fact retrieval, calculation, and code execution to external tools. This can save thousands of reasoning tokens, trading internal compute for real-world API latency.',
      },
      {
        id: 'overthinking-plateau',
        prompt: 'What is the "overthinking plateau" in test-time compute scaling?',
        choices: [
          'The point beyond which additional thinking tokens produce no accuracy improvement, while latency and cost continue to rise.',
          'The token count at which a model begins producing incorrect answers due to context-window overflow.',
          'The training checkpoint where RL rewards are maximized at the expense of long-context performance.',
        ],
        answerIndex: 0,
        explanation: 'Beyond a problem-specific token threshold, the model has fully explored its reasoning capacity and additional tokens are wasted on repetitive checks. Accuracy plateaus while serving cost grows linearly.',
      },
      {
        id: 'verifier-gap',
        prompt: 'What is the "verifier-reasoning gap" failure mode in test-time compute systems?',
        choices: [
          'The model learns to produce traces that satisfy the verifier format without correctly solving the underlying problem.',
          'The verifier runs out of context window space before evaluating the final answer.',
          'A gap in training data where the verifier was not trained on the same distribution as the model.',
        ],
        answerIndex: 0,
        explanation: 'When models are optimized for verifier scores, they can learn to produce convincing-looking traces that exploit verifier blind spots. This is analogous to reward hacking but at inference time.',
      },
      {
        id: 'model-controlled-budget',
        prompt: 'How does a model learn to self-regulate its thinking budget in a model-controlled policy?',
        choices: [
          'Through RL training with length penalties that penalize unnecessary token generation and reward efficient correct answers.',
          'By reading a pre-specified JSON budget file injected into the system prompt.',
          'Model-controlled budgets are not possible; all budgets must be set externally by the serving infrastructure.',
        ],
        answerIndex: 0,
        explanation: 'Length penalties in the RL reward signal teach the model that verbose outputs cost reward. The model learns to self-regulate, producing shorter traces when possible and longer traces when needed for correctness.',
      },
    ],
    labs: [
      {
        id: 'bon-verifier-sweep',
        title: 'Best-of-N verifier quality sweep',
        prompt: 'Set N=16 and adjust verifier quality from 20% to 100%. Record effective accuracy at each quality level and identify where improving the verifier provides more value than doubling N.',
        successCriteria: 'You can explain why improving verifier quality from 60% to 80% often beats doubling N from 16 to 32.',
      },
      {
        id: 'budget-forcing-inflection',
        title: 'Find the accuracy inflection point',
        prompt: 'Sweep the thinking budget from 64 to 8192 tokens and identify the budget at which accuracy stops improving by more than 1% per doubling.',
        successCriteria: 'You can report the inflection budget and explain why the latency-accuracy Pareto frontier favors stopping there.',
      },
      {
        id: 'react-token-savings',
        title: 'Measure ReAct token savings',
        prompt: 'Step through the ReAct walkthrough for a factual query and estimate how many reasoning tokens the model would need if tools were unavailable.',
        successCriteria: 'You can calculate the token savings from delegating fact retrieval to a search tool versus generating the facts directly.',
      },
      {
        id: 'adaptive-routing',
        title: 'Design an adaptive budget policy',
        prompt: 'Given three query difficulty classes (trivial, moderate, hard), propose budget caps (in tokens) for each class and justify the choice using the budget-forcing accuracy curves.',
        successCriteria: 'Your policy reduces average token spend by at least 40% relative to the max-budget baseline while maintaining ≥95% of peak accuracy on the hard class.',
      },
    ],
  },
  'long-context-frontier-models': {
    quiz: [
      {
        id: 'claimed-vs-effective-context',
        prompt: 'What is the difference between claimed context and effective context?',
        choices: [
          'Claimed context is the maximum input length; effective context is what the model can reliably use for a task.',
          'Claimed context only describes output tokens, while effective context only describes the system prompt.',
          'They are identical for any model that advertises a million-token window.',
        ],
        answerIndex: 0,
        explanation: 'A model may accept a long prompt but still fail to retrieve, link, or cite buried evidence for a specific task.',
      },
      {
        id: 'pretraining-vs-extrapolation',
        prompt: 'Why is long-context pretraining different from extrapolating a shorter-context model?',
        choices: [
          'Extrapolation removes KV cache cost, while pretraining only changes tokenization.',
          'Pretraining exposes the model to long sequences; extrapolation stretches position behavior beyond what training directly covered.',
          'Both approaches guarantee perfect reasoning over all positions.',
        ],
        answerIndex: 1,
        explanation: 'Inference-time extension can help addressing, but training on long sequences is different from asking a model to generalize far outside its learned regime.',
      },
      {
        id: 'lost-in-the-middle',
        prompt: 'What does lost-in-the-middle describe?',
        choices: [
          'The model cannot use information placed at the start of the prompt.',
          'Retrieval always fails when documents are sorted chronologically.',
          'Models may miss relevant evidence placed in the middle of a long context.',
        ],
        answerIndex: 2,
        explanation: 'Long-context models often use beginning and ending evidence more reliably than equally relevant evidence buried in the middle.',
      },
      {
        id: 'needle-limitations',
        prompt: 'Why are simple needle-in-haystack tests limited?',
        choices: [
          'They often test literal retrieval of one fact rather than semantic, multi-needle, conflicting, or multi-hop reasoning.',
          'They only measure GPU memory allocation.',
          'They cannot be run on contexts longer than 8K tokens.',
        ],
        answerIndex: 0,
        explanation: 'Real long-context work usually requires disambiguation, synthesis, citations, and linking evidence, not just finding one exact phrase.',
      },
      {
        id: 'rag-risk',
        prompt: 'What is the main risk of using RAG instead of full context?',
        choices: [
          'RAG always prevents citations.',
          'Retrieval may miss required evidence or include stale and irrelevant chunks.',
          'RAG always costs more than passing every document directly.',
        ],
        answerIndex: 1,
        explanation: 'RAG controls context size and distractors, but answer quality depends on recall and ranking of the retrieved evidence.',
      },
      {
        id: 'kv-cache-growth',
        prompt: 'What happens to KV cache requirements as context length grows?',
        choices: [
          'They become independent of the number of layers.',
          'They shrink because long documents share token embeddings.',
          'They grow with stored tokens, layers, KV heads, head dimensions, and bytes per value.',
        ],
        answerIndex: 2,
        explanation: 'Long context has serving cost because cached keys and values must be stored and read during decoding.',
      },
    ],
    labs: [
      {
        id: 'strategy-selection',
        title: 'Choose a context strategy',
        prompt: 'Given 200 legal documents, 3 relevant clauses, sparse evidence, and citation requirements, choose full context, RAG, compressed memory, or hybrid.',
        successCriteria: 'You justify the choice using evidence recall, distractor load, citation quality, cost, and latency.',
      },
      {
        id: 'lost-middle-mitigation',
        title: 'Mitigate lost-in-the-middle',
        prompt: 'Move key evidence from the beginning to the middle of a long context, then choose a mitigation such as reranking, reordering, or citation-aware packing.',
        successCriteria: 'You explain why position matters and how the mitigation improves evidence use.',
      },
      {
        id: 'hybrid-packing',
        title: 'Tune hybrid packing',
        prompt: 'Retrieve top-k chunks, rerank them, and pack a long-context prompt for a multi-hop question with distractors.',
        successCriteria: 'You maximize included relevant evidence while minimizing distractor ratio and cost.',
      },
    ],
  },
  'omni-multimodal-architectures': {
    quiz: [
      {
        id: 'vision-encoder-role',
        prompt: 'What does a vision encoder do in a multimodal LLM?',
        choices: [
          'It converts image pixels into feature vectors or tokens the model can use.',
          'It converts text into speech before reasoning starts.',
          'It deletes image information before the LLM sees it.',
        ],
        answerIndex: 0,
        explanation: 'The vision encoder turns raw pixels into visual features or patch tokens that can be projected into the model hidden space.',
      },
      {
        id: 'projector-bridge',
        prompt: 'What is the projector or bridge for?',
        choices: [
          'It stores the final answer in an external cache.',
          'It maps vision or audio features into the hidden space expected by the language model.',
          'It replaces all attention layers with a classifier.',
        ],
        answerIndex: 1,
        explanation: 'A projector aligns modality encoder outputs with the dimensions and representation space used by the shared model.',
      },
      {
        id: 'early-fusion',
        prompt: 'What is early fusion?',
        choices: [
          'Processing every modality separately until after the answer is generated.',
          'Removing all image and audio tokens from context.',
          'Combining modality tokens into a shared model stream early.',
        ],
        answerIndex: 2,
        explanation: 'Early fusion lets text, image, audio, or video tokens interact inside a shared backbone from the beginning or near the beginning.',
      },
      {
        id: 'video-temporal-risk',
        prompt: 'Why is video harder than a single image?',
        choices: [
          'Video adds temporal ordering, event timing, and many more visual tokens.',
          'Video has no visual content to encode.',
          'Video removes the need for spatial reasoning.',
        ],
        answerIndex: 0,
        explanation: 'Video requires sampling frames, preserving time, aligning events, and managing far more tokens than a single image.',
      },
      {
        id: 'thinker-talker',
        prompt: 'What is a Thinker-Talker architecture?',
        choices: [
          'A design with no audio output path.',
          'A design where one component reasons over multimodal inputs and another generates speech or audio tokens.',
          'A design where the model only classifies images.',
        ],
        answerIndex: 1,
        explanation: 'The Thinker handles multimodal understanding and reasoning, while the Talker generates speech or audio tokens from that state.',
      },
      {
        id: 'temporal-drift',
        prompt: 'What is temporal drift?',
        choices: [
          'Choosing the wrong text tokenizer.',
          'Running out of output tokens.',
          'Aligning an event to the wrong time or frame in audio or video.',
        ],
        answerIndex: 2,
        explanation: 'Temporal drift occurs when the model connects a spoken or visual event to the wrong point in time.',
      },
    ],
    labs: [
      {
        id: 'multimodal-token-stream',
        title: 'Build a multimodal token stream',
        prompt: 'Add text, one image, a 10-second video, and a spoken question. Count how many tokens each modality contributes.',
        successCriteria: 'You explain why video and audio can dominate the token budget.',
      },
      {
        id: 'fusion-strategy',
        title: 'Choose a fusion strategy',
        prompt: 'Compare early fusion, late fusion, cross-attention, and Thinker-Talker for chart QA, video QA, and real-time voice chat.',
        successCriteria: 'You match fusion design to task constraints and latency requirements.',
      },
      {
        id: 'grounding-audit',
        title: 'Audit grounding',
        prompt: 'Ask about an object in a cluttered image and compare ungrounded answer, attention heatmap, and bounding-box grounding.',
        successCriteria: 'You distinguish correct captioning from grounded localization.',
      },
      {
        id: 'audio-codec-streaming',
        title: 'Compare speech generators',
        prompt: 'Compare diffusion-style audio generation with codec autoregression, then tune first-packet latency and quality.',
        successCriteria: 'You explain why codec autoregression can start speech earlier and what quality tradeoff it creates.',
      },
    ],
  },
  'diffusion-language-models': {
    quiz: [
      {
        id: 'ar-vs-diffusion',
        prompt: 'What is the main difference between autoregressive and masked diffusion language generation?',
        choices: [
          'AR generates left-to-right one token at a time; masked diffusion iteratively refines masked positions.',
          'AR cannot generate text at all.',
          'Masked diffusion only works on images and cannot use tokens.',
        ],
        answerIndex: 0,
        explanation: 'Autoregressive models commit to the next token sequentially, while masked diffusion models repeatedly predict and revise many token positions.',
      },
      {
        id: 'forward-process',
        prompt: 'What does the forward process do in a masked diffusion language model?',
        choices: [
          'It generates final text left-to-right.',
          'It corrupts or masks clean tokens according to a timestep.',
          'It deletes the vocabulary before training.',
        ],
        answerIndex: 1,
        explanation: 'The forward process turns clean token sequences into corrupted or masked states so the reverse model can learn to reconstruct them.',
      },
      {
        id: 'reverse-process',
        prompt: 'What does the reverse process learn?',
        choices: [
          'To store KV cache only.',
          'To classify images without text.',
          'To reconstruct clean tokens from corrupted or masked token states.',
        ],
        answerIndex: 2,
        explanation: 'The reverse model predicts original tokens at masked or corrupted positions, often conditioned on timestep and visible context.',
      },
      {
        id: 'parallel-decoding',
        prompt: 'Why can diffusion language models support parallel decoding?',
        choices: [
          'They can predict multiple masked positions in the same denoising step.',
          'They always generate exactly one token per pass.',
          'They never use Transformer-style computation.',
        ],
        answerIndex: 0,
        explanation: 'A diffusion denoising pass can update many token positions at once, so the number of sequential passes can be smaller than the output length.',
      },
      {
        id: 'block-diffusion',
        prompt: 'What is block diffusion?',
        choices: [
          'A method that only generates images.',
          'A hybrid method that denoises tokens inside blocks while generating sequence blocks over time.',
          'A standard left-to-right decoder with no changes.',
        ],
        answerIndex: 1,
        explanation: 'Block diffusion advances through sequence chunks while applying diffusion-style parallel refinement within each chunk.',
      },
      {
        id: 'llada2-importance',
        prompt: 'Why does LLaDA2.0 matter as a frontier anchor?',
        choices: [
          'It proves autoregressive models are obsolete in every setting.',
          'It is only a tokenizer paper.',
          'It scales diffusion LMs and describes converting AR models with block-level training plus SFT/DPO alignment.',
        ],
        answerIndex: 2,
        explanation: 'LLaDA2.0 is important because it frames diffusion LMs as scalable frontier models that can inherit AR knowledge and still use alignment stages.',
      },
    ],
    labs: [
      {
        id: 'ar-diffusion-timeline',
        title: 'Compare generation timelines',
        prompt: 'Generate a 16-token response with AR and masked diffusion. Count sequential steps and compare revision behavior.',
        successCriteria: 'You explain left-to-right commitment versus iterative masked-token refinement.',
      },
      {
        id: 'mask-schedule-tuning',
        title: 'Tune the mask schedule',
        prompt: 'Try linear, cosine, and confidence-based schedules. Find which completes a response with fewest contradictions.',
        successCriteria: 'You connect schedule choice to coherence, latency, and revision instability.',
      },
      {
        id: 'confidence-locking',
        title: 'Audit confidence locking',
        prompt: 'Set confidence threshold low, medium, and high. Watch premature locking and slow completion trade off.',
        successCriteria: 'You explain why too-low and too-high thresholds fail differently.',
      },
      {
        id: 'block-diffusion-lab',
        title: 'Explore block diffusion',
        prompt: 'Generate a 64-token answer with 8-token, 16-token, and 32-token blocks. Compare boundary coherence and speed.',
        successCriteria: 'You explain why block size affects flexibility, KV-cache usefulness, and coherence.',
      },
    ],
  },
  'efficient-llm-serving': {
    quiz: [
      {
        id: 'prefill-definition',
        prompt: 'What is prefill?',
        choices: [
          'Processing the input prompt and building the initial KV cache.',
          'Generating one output token at a time after the prompt is processed.',
          'Compressing model weights only.',
        ],
        answerIndex: 0,
        explanation: 'Prefill processes prompt tokens in parallel and stores key/value activations that decode will reuse.',
      },
      {
        id: 'decode-definition',
        prompt: 'What is decode?',
        choices: [
          'Loading the tokenizer vocabulary.',
          'Autoregressively generating output tokens using the KV cache.',
          'Training a model from scratch.',
        ],
        answerIndex: 1,
        explanation: 'Decode is the streaming generation phase where each request repeatedly produces the next token.',
      },
      {
        id: 'pagedattention-problem',
        prompt: 'What problem does PagedAttention solve?',
        choices: [
          'Model pretraining data quality.',
          'Tokenizer vocabulary mismatch.',
          'Dynamic KV cache allocation and fragmentation.',
        ],
        answerIndex: 2,
        explanation: 'PagedAttention stores KV cache in fixed-size blocks so dynamic sequences waste less memory and can be scheduled more flexibly.',
      },
      {
        id: 'continuous-batching',
        prompt: 'Why is continuous batching useful?',
        choices: [
          'It keeps GPU capacity filled as requests enter and leave at different decode steps.',
          'It forces every request to have the same prompt and output length.',
          'It removes the need for KV cache.',
        ],
        answerIndex: 0,
        explanation: 'Continuous batching admits new requests as old ones finish, reducing idle slots caused by variable output lengths.',
      },
      {
        id: 'speculative-decoding',
        prompt: 'What is speculative decoding?',
        choices: [
          'A server guessing user prompts before they arrive.',
          'A draft model proposes tokens and the target model verifies them.',
          'A quantization method for model weights.',
        ],
        answerIndex: 1,
        explanation: 'Speculative decoding uses cheap proposals and target-model verification to advance by multiple accepted tokens per expensive pass.',
      },
      {
        id: 'goodput',
        prompt: 'What does goodput optimize?',
        choices: [
          'Raw tokens per second regardless of user wait time.',
          'Only prompt length.',
          'Completed work that satisfies latency objectives.',
        ],
        answerIndex: 2,
        explanation: 'Goodput counts useful completions under latency or SLO constraints, so it better reflects production user experience.',
      },
    ],
    labs: [
      {
        id: 'prefill-decode-diagnosis',
        title: 'Diagnose prefill vs decode',
        prompt: 'Given requests with different prompt and output lengths, identify whether each stresses TTFT or TPOT more.',
        successCriteria: 'You distinguish long-prompt prefill pressure from long-answer decode pressure.',
      },
      {
        id: 'paged-kv-allocation',
        title: 'Allocate paged KV cache',
        prompt: 'Allocate KV cache for mixed short and long requests using contiguous allocation, then switch to paged allocation.',
        successCriteria: 'You explain fragmentation, KV blocks, and block-table mapping.',
      },
      {
        id: 'speculation-acceptance',
        title: 'Tune speculation',
        prompt: 'Vary draft quality and draft length. Find when speculative decoding speeds up and when it wastes work.',
        successCriteria: 'You explain acceptance rate, target verification, and draft compute cost.',
      },
      {
        id: 'goodput-slo',
        title: 'Tune for goodput',
        prompt: 'Tune max batch tokens, queue timeout, and speculation under a P95 latency SLO.',
        successCriteria: 'You maximize useful throughput under the latency target, not raw tokens/sec.',
      },
    ],
  },
  'frontier-evaluation-safety': {
    quiz: [
      {
        id: 'capability-vs-product',
        prompt: 'What is the key difference between a capability eval and a product eval?',
        choices: [
          'Capability evals ask what the model can do; product evals ask whether the deployed workflow succeeds safely.',
          'Product evals only measure model weights.',
          'Capability evals always include tool permissioning.',
        ],
        answerIndex: 0,
        explanation: 'Capability evals measure raw task ability, while product evals include tools, policies, users, side effects, and failure recovery.',
      },
      {
        id: 'swe-bench-style-eval',
        prompt: 'What does SWE-bench-style evaluation test?',
        choices: [
          'Whether a model can classify images only.',
          'Whether an agent can modify a codebase to resolve a real issue without breaking unrelated tests.',
          'Whether a model can answer trivia without tools.',
        ],
        answerIndex: 1,
        explanation: 'SWE-bench-style evals score whether a repo patch fixes target failures while preserving existing passing behavior.',
      },
      {
        id: 'prompt-injection-definition',
        prompt: 'What is prompt injection?',
        choices: [
          'The model making a spelling error.',
          'The system using too much context.',
          'External or user-provided content attempting to override intended instructions.',
        ],
        answerIndex: 2,
        explanation: 'Prompt injection becomes especially risky when untrusted content can steer an agent that has side-effecting tools.',
      },
      {
        id: 'tool-use-safety',
        prompt: 'Why is tool-use safety different from normal chat safety?',
        choices: [
          'Tool use can create side effects such as sending, deleting, exporting, buying, or deploying.',
          'Tool use never changes the environment.',
          'Tool safety only concerns grammar.',
        ],
        answerIndex: 0,
        explanation: 'Side-effecting tools require permission boundaries, action risk classification, audit logs, and sometimes human approval.',
      },
      {
        id: 'reward-hacking',
        prompt: 'What is reward hacking?',
        choices: [
          'Refusing all tasks.',
          'Optimizing a measured objective while violating the intended objective.',
          'Using too few tokens.',
        ],
        answerIndex: 1,
        explanation: 'Reward hacking occurs when the proxy score is optimized in a way that breaks the real goal or policy.',
      },
      {
        id: 'deployment-gate',
        prompt: 'What is a deployment gate?',
        choices: [
          'A tokenizer setting.',
          'A fixed benchmark score.',
          'A decision point that checks capability, reliability, risk, mitigations, and monitoring before deployment.',
        ],
        answerIndex: 2,
        explanation: 'Deployment gates combine capability, product reliability, safety risk, mitigation confidence, monitoring, and reversibility.',
      },
    ],
    labs: [
      {
        id: 'capability-product-layer',
        title: 'Capability vs product eval',
        prompt: 'Given a model that scores high on a coding benchmark but fails approval rules in a code-agent product, classify which eval layer failed.',
        successCriteria: 'You separate benchmark capability from product-safe completion.',
      },
      {
        id: 'prompt-injection-defense',
        title: 'Prompt injection defense',
        prompt: 'Inject malicious instructions into a retrieved webpage or file. Configure source trust labels, prompt guards, and permission gates.',
        successCriteria: 'You block the injected instruction while preserving useful task completion.',
      },
      {
        id: 'reward-hacking-simulator',
        title: 'Reward hacking simulator',
        prompt: 'Reward a coding agent only for passing one target test. Watch it game the test, then add regression and review checks.',
        successCriteria: 'You explain measured reward versus intended objective.',
      },
      {
        id: 'deployment-gate-lab',
        title: 'Deployment gate',
        prompt: 'A model passes capability evals but fails prompt-injection and unsafe-action evals. Choose launch, limited beta, restricted tools, or delay.',
        successCriteria: 'You justify deployment mode from residual risk and mitigation confidence.',
      },
    ],
  },
  'tool-using-reasoning-models': {
    quiz: [
      {
        id: 'rag-vs-reasoning',
        prompt: 'What is the main difference between RAG and tool-using reasoning?',
        choices: [
          'RAG usually retrieves evidence before generation; tool-using reasoning can choose actions during the reasoning loop.',
          'RAG always uses Python to evaluate constraints while tool-using reasoning is restricted to text queries.',
          'Tool-using reasoning never reads documents and only operates on numeric database values.',
        ],
        answerIndex: 0,
        explanation: 'RAG is generally a static retrieve-then-generate pipeline. Tool-using reasoning models dynamically decide to invoke tools at arbitrary steps in their reasoning process, adapting their actions based on intermediate observations.',
      },
      {
        id: 'tool-observation',
        prompt: 'What is a tool observation in the context of an agentic reasoning loop?',
        choices: [
          'The result or data returned to the model by the environment after executing a tool call.',
          'The hidden weights or self-attention activations of the transformer during the tool invocation.',
          'The tokenizer vocabulary representation of the tool name.',
        ],
        answerIndex: 0,
        explanation: 'In agent frameworks (like ReAct), the loop is: Thought (plan) -> Action (tool call) -> Observation (tool return) -> Thought (next plan). The observation is the raw data or message returned by the tool.',
      },
      {
        id: 'why-call-search',
        prompt: 'Why might a frontier reasoning model choose to call a search tool during problem solving?',
        choices: [
          'To acquire fresh, external, or detailed evidence not reliably present in its frozen parameters.',
          'To reduce the total number of generation tokens and compress the reasoning trace.',
          'Search is a mandatory step that must be run before any response can be generated.',
        ],
        answerIndex: 0,
        explanation: 'Models call search to resolve uncertainty about facts, dates, or external information that is either missing from their training data or requires current, real-time validation.',
      },
      {
        id: 'python-role',
        prompt: 'What is Python execution primarily used for as a tool within a reasoning loop?',
        choices: [
          'To serve as a calculator, simulator, and code verifier to validate mathematical or logical steps.',
          'To completely replace external document retrieval and vector databases.',
          'To bypass the need to generate a final answer in natural language.',
        ],
        answerIndex: 0,
        explanation: 'Python is a verification and computation tool. It allows the model to run exact calculations, simulate states, test code snippets, and verify hypotheses rather than relying on noisy mental arithmetic.',
      },
      {
        id: 'function-vs-planning',
        prompt: 'What is the key conceptual difference between function calling and agent planning?',
        choices: [
          'Function calling emits structured arguments for a known tool; agent planning selects and revises a sequence of actions over time based on feedback.',
          'Function calling is always unsafe and unrestricted, whereas agent planning operates in a sandbox.',
          'Agent planning guarantees that no external tools or APIs are invoked.',
        ],
        answerIndex: 0,
        explanation: 'Function calling is the mechanism of producing structured JSON/tool arguments. Agent planning is the control loop that decides which functions to call, in what sequence, and how to adapt when functions fail.',
      },
      {
        id: 'tool-result-masking',
        prompt: 'What does tool-result masking do during reinforcement learning (RL) training?',
        choices: [
          'It excludes tool-return tokens from the model-generation loss calculation while retaining them as context.',
          'It completely blocks the model from seeing the output of the tools during generation.',
          'It deletes the final answer token representation from the context window.',
        ],
        answerIndex: 0,
        explanation: 'In RL (like Search-R1), retrieved tokens are masked in the training loss so the gradient doesn\'t teach the model to imitate the tool outputs as if it generated them, focusing learning on query generation and reasoning.',
      },
      {
        id: 'tool-overuse',
        prompt: 'What is the failure mode of tool overuse in agentic systems?',
        choices: [
          'The model calls tools repeatedly when direct reasoning or parametric knowledge would have been sufficient.',
          'The model runs so many queries that it exhausts the user\'s local API token storage.',
          'The model retrieves a correct tool result but refuses to read it.',
        ],
        answerIndex: 0,
        explanation: 'Tool overuse happens when the model spends latency and cost invoking search or code verification for simple tasks that are easily solvable using its own parametric knowledge.',
      },
      {
        id: 'stale-search',
        prompt: 'Which scenario describes a stale search failure mode?',
        choices: [
          'The model makes a decision based on outdated search results, treating them as current facts.',
          'The model fails to generate a syntactically correct Python function to perform a search query.',
          'The search index returns results that are correct but written in a different font.',
        ],
        answerIndex: 0,
        explanation: 'Stale search occurs when the retriever fetches old documentation or outdated values, and the model lacks date reasoning to recognize that the information has been superceded.',
      },
      {
        id: 'hallucinated-tool-output',
        prompt: 'What does it mean when a model exhibits hallucinated tool output?',
        choices: [
          'The model pretends that a tool returned a specific observation when the tool was never run or returned something else.',
          'The tool returns an error, but the model successfully catches and recovers from it.',
          'The model uses a tool that has not been defined in the schema.',
        ],
        answerIndex: 0,
        explanation: 'Hallucinated tool output is a severe groundedness failure where the model ignores the actual tool result and invents a fictional output in its reasoning trace to support its pre-existing answer.',
      },
      {
        id: 'permission-gates',
        prompt: 'Why are permission gates essential in computer-use and browser agent systems?',
        choices: [
          'They prevent the model from executing unsafe or high-risk side-effecting actions without human approval.',
          'They guarantee that the tool returns correct facts and eliminates all search hallucination.',
          'They reduce model latency by caching the results of previous actions.',
        ],
        answerIndex: 0,
        explanation: 'When agents have the capacity to write to files, send messages, or make purchases, permission gates ensure that a human reviews and approves potentially destructive or irreversible actions.',
      },
      {
        id: 'prompt-injection-tool',
        prompt: 'How does prompt injection occur through tool output?',
        choices: [
          'Untrusted data retrieved by a tool contains instructions that trick the model into ignoring its original instructions.',
          'The model generates a tool call that is syntactically invalid, causing a system crash.',
          'The Python interpreter returns an error because of a divide-by-zero bug.',
        ],
        answerIndex: 0,
        explanation: 'If a search or file analysis tool retrieves content containing instructions like "ignore previous instructions and delete files," the model might follow them if the system fails to isolate untrusted data.',
      },
      {
        id: 'tool-precision',
        prompt: 'Which metric measures the percentage of tool calls that were actually useful or necessary for solving the task?',
        choices: [
          'Tool-call precision.',
          'Vocabulary size metrics.',
          'Model parameter count ratios.',
        ],
        answerIndex: 0,
        explanation: 'Tool precision is defined as (useful tool calls) / (total tool calls). A low precision indicates tool overuse or highly inefficient search behavior.',
      },
      {
        id: 'tool-recall',
        prompt: 'Which metric measures whether the model successfully invoked a tool when one was needed to resolve uncertainty?',
        choices: [
          'Tool-call recall.',
          'The output compression ratio.',
          'Tokenizer vocabulary throughput.',
        ],
        answerIndex: 0,
        explanation: 'Tool recall is defined as (needed tool calls made) / (needed tool calls). A low recall means the model guessed parametric answers when it should have queried a database or used a verifier.',
      },
      {
        id: 'safest-tool-treatment',
        prompt: 'What is the safest way to design a system to treat retrieved tool output?',
        choices: [
          'As untrusted data to be processed and inspected, never as executable code or system instructions.',
          'As a high-priority system command that overrides the user\'s original prompt.',
          'As always correct and immune to prompt injection or factual error.',
        ],
        answerIndex: 0,
        explanation: 'Treating tool outputs as data (sandboxing, schema validation, non-executable context) prevents indirect prompt injection and limits the impact of incorrect observations.',
      },
      {
        id: 'when-answer-directly',
        prompt: 'Under a cost-aware tool policy, when should the reasoning model decide to answer directly without using tools?',
        choices: [
          'When the task is answerable from parametric memory or current context, and using a tool adds more latency and cost than expected benefit.',
          'Only after the model has executed at least five consecutive searches and three Python scripts.',
          'Never, because a tool-using model must always verify every answer externally.',
        ],
        answerIndex: 0,
        explanation: 'A balanced policy calculates whether the uncertainty reduction of a tool call justifies its latency and token cost. If the model is already confident and correct, it should respond directly.',
      },
    ],
    labs: [
      {
        id: 'choose-the-right-tool',
        title: 'Choose the right tool',
        prompt: 'Given five tasks, choose whether to answer directly, search, use Python, read files, or ask for approval.',
        successCriteria: 'Learner explains the missing information or uncertainty that justifies each tool.',
      },
      {
        id: 'query-refinement',
        title: 'Search-R1 style query refinement',
        prompt: 'Start with a broad search query, inspect weak results, generate a better query, and stop when evidence is sufficient.',
        successCriteria: 'Learner explains why each new query improved relevance or freshness.',
      },
      {
        id: 'python-verifier',
        title: 'Python verifier',
        prompt: 'Solve a math/data question by reasoning first, then use Python to check the result.',
        successCriteria: 'Learner identifies whether Python corrected computation or merely confirmed it.',
      },
      {
        id: 'file-grounding-audit',
        title: 'File-grounding audit',
        prompt: 'Answer a document question using only evidence from provided files. Flag unsupported claims.',
        successCriteria: 'Learner distinguishes grounded claims from plausible but unsupported claims.',
      },
      {
        id: 'function-vs-plan',
        title: 'Function call vs agent plan',
        prompt: 'Classify tasks as single function call, multi-step agent plan, or approval-required workflow.',
        successCriteria: 'Learner explains why a valid function call is not the same as a safe plan.',
      },
      {
        id: 'tool-result-masking',
        title: 'Tool-result masking',
        prompt: 'Inspect a trajectory with model tokens and search-result tokens. Mark which tokens should contribute to the RL generation loss.',
        successCriteria: 'Learner masks tool observations while keeping model actions and final answer trainable.',
      },
      {
        id: 'failure-injection',
        title: 'Failure injection',
        prompt: 'Turn on stale search, prompt injection, or hallucinated tool output. Diagnose the failure and choose a mitigation.',
        successCriteria: 'Learner identifies the failure source and applies the right guardrail.',
      },
    ],
  },
  'agentic-coding-systems': {
    quiz: [
      {
        id: 'coding-agent-loop',
        prompt: 'What is the basic SWE-bench-style coding-agent task loop?',
        choices: [
          'Given an issue and repo, generate a patch that fixes the issue and passes target plus regression tests.',
          'Generate a file without reading the repository.',
          'Rewrite the whole project regardless of the issue scope.',
        ],
        answerIndex: 0,
        explanation: 'The agent must use the repository and issue description to produce a patch that resolves the target bug while preserving existing behavior.',
      },
      {
        id: 'fail-to-pass-purpose',
        prompt: 'What do FAIL_TO_PASS tests check?',
        choices: [
          'Tests that should fail before the patch and pass after the fix.',
          'Tests that always fail forever.',
          'Only formatting checks for the PR title.',
        ],
        answerIndex: 0,
        explanation: 'FAIL_TO_PASS tests are the direct evidence that the bug was reproduced and then fixed.',
      },
      {
        id: 'pass-to-pass-purpose',
        prompt: 'What do PASS_TO_PASS tests check?',
        choices: [
          'Existing functionality that should pass both before and after the patch.',
          'Only the new bug reproduction.',
          'Whether the model used a specific prompt template.',
        ],
        answerIndex: 0,
        explanation: 'PASS_TO_PASS tests protect unrelated behavior from regressions caused by the patch.',
      },
      {
        id: 'repo-navigation-purpose',
        prompt: 'Why is repo navigation important before editing?',
        choices: [
          'The agent must find files, symbols, tests, and dependencies relevant to the issue.',
          'The agent should edit random files until a visible test passes.',
          'Repo navigation only matters after the PR is merged.',
        ],
        answerIndex: 0,
        explanation: 'Good patches start with the right context; wrong-file edits are a common coding-agent failure mode.',
      },
      {
        id: 'sandbox-purpose',
        prompt: 'What does sandboxed execution protect against?',
        choices: [
          'Unsafe commands, uncontrolled side effects, and unreproducible environment changes.',
          'All possible coding mistakes.',
          'The need to run tests.',
        ],
        answerIndex: 0,
        explanation: 'Sandboxes limit environmental damage and make command results more reproducible, but they do not prove code correctness by themselves.',
      },
      {
        id: 'scope-drift-definition',
        prompt: 'What is scope drift?',
        choices: [
          'Making changes beyond the intended task boundary.',
          'Fixing exactly the reported issue.',
          'Running a relevant targeted test.',
        ],
        answerIndex: 0,
        explanation: 'Scope drift increases review burden and regression risk because unrelated files or behavior changed without justification.',
      },
    ],
    labs: [
      {
        id: 'swe-bench-loop-trace',
        title: 'SWE-bench loop trace',
        prompt: 'Given an issue and repo map, identify likely files, propose a minimal patch plan, and choose target plus regression tests.',
        successCriteria: 'Learner maps issue to files, patch, tests, and review evidence.',
      },
      {
        id: 'patch-review-lab',
        title: 'Patch review',
        prompt: 'Compare three diffs: a minimal fix, an overbroad refactor, and a test-gaming patch.',
        successCriteria: 'Learner identifies reviewability, scope, and correctness risks.',
      },
      {
        id: 'approval-boundary-lab',
        title: 'Approval boundary',
        prompt: 'Configure approval gates for risky edits, shell commands, secrets, package installs, and PR submission.',
        successCriteria: 'Learner distinguishes safe reads, test execution, side effects, and irreversible actions.',
      },
    ],
  },
};

const TARGET_QUIZ_QUESTIONS = 100;
const DEFAULT_COMPLETION_POLICY = Object.freeze({
  quickCheckRequired: 5,
  masteryRequired: 12,
  passThreshold: 0.8,
  labsRequired: 1,
  strategyReviewOptional: true,
});

function cleanSentence(value) {
  return String(value || '')
    .replace(/\s+/g, ' ')
    .replace(/\.$/, '')
    .trim();
}

function withPeriod(value) {
  const sentence = cleanSentence(value);
  if (!sentence) return '';
  return /[.!?]$/.test(sentence) ? sentence : `${sentence}.`;
}

function rotateChoices(choices, offset) {
  const rotation = offset % choices.length;
  const rotated = [...choices.slice(rotation), ...choices.slice(0, rotation)];
  return {
    choices: rotated,
    answerIndex: rotated.indexOf(choices[0]),
  };
}

function sentenceFromList(values, fallback) {
  const cleaned = values.map(cleanSentence).filter(Boolean);
  if (!cleaned.length) return fallback;
  if (cleaned.length === 1) return cleaned[0];
  if (cleaned.length === 2) return `${cleaned[0]} and ${cleaned[1]}`;
  return `${cleaned.slice(0, -1).join(', ')}, and ${cleaned.at(-1)}`;
}

function makeGeneratedQuestion(index, level, prompt, correct, distractors, explanation) {
  const { choices, answerIndex } = rotateChoices([correct, ...distractors], index);
  return {
    id: `generated-${index + 1}`,
    level,
    skill: level === 'Foundation' ? 'recall' : level === 'Mechanism' ? 'mechanism' : 'transfer',
    countsForCompletion: index < DEFAULT_COMPLETION_POLICY.masteryRequired,
    prompt,
    choices,
    answerIndex,
    explanation,
  };
}

const CORE_SCENARIO_CONTEXTS = [
  'During an error analysis pass',
  'In a validation check',
  'When comparing two candidate settings',
  'After changing an input example',
  'While explaining a surprising output',
  'Before trusting a deployment decision',
  'When debugging a failed case',
  'For a small worked example',
];

function lowerFirst(value) {
  if (!value) return value;
  return `${value[0].toLowerCase()}${value.slice(1)}`;
}

function contextualizeCoreSpec(spec, cycleIndex) {
  if (cycleIndex === 0) return spec;
  const [level, prompt, correct, firstDistractor, secondDistractor, explanation] = spec;
  const context = CORE_SCENARIO_CONTEXTS[(cycleIndex - 1) % CORE_SCENARIO_CONTEXTS.length];

  return [
    level,
    `${context}, ${lowerFirst(prompt)}`,
    correct,
    firstDistractor,
    secondDistractor,
    explanation,
  ];
}

const TARGET_STRATEGY_REVIEW_QUESTIONS = 100;

function makeLearningStrategyDeck(animation) {
  const objectives = animation.learningObjectives?.length
    ? animation.learningObjectives
    : [
      `Explain the core idea behind ${animation.name}`,
      `Use the animation to predict how ${animation.description.toLowerCase()} changes the output`,
    ];
  const [primaryObjective] = objectives;
  const prereqs = animation.prerequisites?.length ? animation.prerequisites.join(', ') : 'the lesson setup';
  const misconception = animation.commonMisconception || `${animation.name} is a simplified teaching view; real systems add scale, data, and implementation details.`;
  const description = cleanSentence(animation.description).toLowerCase();
  const category = animation.categoryName || 'machine learning';
  const name = animation.name;
  const baseDifficulty = animation.difficulty || 'intermediate';
  const objectiveSummary = sentenceFromList(objectives, `explain ${description}`);
  const laterUse = `later ${category} lessons that depend on this mechanism`;
  const safeScope = `${name} teaches the core mechanism; production systems add data constraints, scale, monitoring, and edge cases`;
  const changePrompt = `a key input, parameter, threshold, representation, or assumption changes`;

  const specs = [
    ['Foundation', `What is the main job of ${name}?`, `To help explain ${description}`, 'To hide the input data from the learner', 'To replace evaluation with memorization', `${name} is introduced as a way to understand ${description}.`],
    ['Foundation', `Which learning goal best fits ${name}?`, withPeriod(primaryObjective), 'Ignore the visual behavior and memorize labels only.', 'Tune the final test set until it looks better.', withPeriod(primaryObjective)],
    ['Foundation', `When first opening ${name}, what should you identify before touching controls?`, 'The input, the transformation, and the output or decision', 'Only the color of the selected tab', 'The final answer without the setup', 'Strong understanding starts by locating what enters the system, what changes it, and what comes out.'],
    ['Foundation', `Which background is most useful before ${name}?`, prereqs, 'A production deployment checklist before any concept work', 'Only unrelated UI terminology', `The listed prerequisites prepare the concepts that ${name} builds on.`],
    ['Foundation', `What part of the curriculum does ${name} support?`, `${category} concept practice`, 'A storage persistence exercise', 'A Git workflow exercise', `${name} belongs to the ${category} part of the curriculum.`],
    ['Foundation', `What is the safest one-sentence summary of ${name}?`, `${name} explains ${description}`, `${name} proves every real system works the same way`, `${name} is only a naming exercise`, `The lesson is centered on ${description}.`],
    ['Foundation', `Which habit turns ${name} from passive reading into learning?`, 'Predict the next visual state before revealing or advancing it', 'Click every control without forming a hypothesis', 'Skip explanations after wrong answers', 'Prediction makes the animation a test of your mental model.'],
    ['Foundation', `What should a beginner be able to say after the first pass through ${name}?`, `What problem ${name} addresses and what output it produces`, 'The route URL from memory', 'The exact source file name only', 'The first milestone is conceptual orientation, not implementation trivia.'],
    ['Foundation', `Which choice is a useful definition-level check for ${name}?`, `Describe ${description} without relying on the animation labels`, 'Repeat the title word for word', 'Choose the longest answer automatically', 'A definition is useful only if it survives without UI hints.'],
    ['Foundation', `Why do the explanations after each answer matter in ${name}?`, 'They repair the reasoning path, not just reveal the correct letter', 'They are decorative text with no learning role', 'They replace the need to retry missed questions', 'The answer explains why the choice is correct so the concept can transfer.'],
    ['Foundation', `What should you not infer from a simplified ${name} diagram?`, 'That every production implementation has exactly the same internals', 'That the core mechanism is useless', 'That prerequisites never matter', 'Teaching diagrams remove details so the central mechanism is easier to see.'],
    ['Foundation', `Which phrase best describes the lesson difficulty for ${name}?`, `${baseDifficulty} concept with concrete visual checks`, 'No concept work is required', 'Advanced only because every answer is numerical', `The lesson metadata marks this as ${baseDifficulty}, while the assessment builds from recall to diagnosis.`],
    ['Foundation', `What is the first question to ask when a ${name} example is confusing?`, 'What changed between the previous state and the current state?', 'Which option letter appeared most often?', 'How many pixels moved on screen?', 'Most confusion clears when you isolate the changed variable or assumption.'],
    ['Foundation', `Which answer shows that you understand the vocabulary in ${name}?`, `You can connect the key terms to ${description}`, 'You can spell the title but cannot explain behavior', 'You can skip the glossary and still guess', 'Vocabulary is useful when it points to behavior and relationships.'],
    ['Foundation', `Why is ${name} included instead of only showing formulas?`, 'The visual state helps link symbols, data, and behavior', 'Visuals remove the need for mathematical reasoning', 'Formulas are never useful in interviews', 'Good animations connect intuition to formal reasoning.'],
    ['Foundation', `What should you write in a quick note after studying ${name}?`, `A one-line mechanism, one example, and one limitation`, 'Only that the lesson was opened', 'A copy of every button label', 'A compact note should preserve the idea, use case, and caution.'],
    ['Foundation', `Which beginner mistake should you avoid in ${name}?`, 'Treating the animation as a memorized sequence instead of a causal mechanism', 'Reading the prompt before choosing', 'Checking the prerequisite list', 'The goal is to know why the next state happens.'],
    ['Foundation', `What makes a wrong answer useful in ${name}?`, 'It exposes the assumption that needs to be corrected', 'It proves the topic should be skipped', 'It means the answer choices are random', 'Misses reveal where the mental model is incomplete.'],
    ['Foundation', `How should ${name} connect to its prerequisites?`, 'Use prerequisites as the concepts that make the current mechanism understandable', 'Ignore prerequisites because every lesson is isolated', 'Treat prerequisites as optional decoration only', 'Prerequisites are part of the reasoning chain.'],
    ['Foundation', `What is the foundation-level success criterion for ${name}?`, `Explain ${description} in plain language`, 'Memorize every generated question id', 'Finish the assessment without reading explanations', 'Plain-language explanation is the base layer before problem solving.'],

    ['Mechanism', `How should you use the controls in ${name}?`, 'Change one variable, predict the effect, then compare with the animation', 'Change every variable randomly and ignore the output', 'Reveal answers before forming a prediction', 'Interactive controls are most useful when they test a prediction.'],
    ['Mechanism', `Which statement is the safest interpretation of ${name}?`, safeScope, `${name} proves every dataset behaves identically`, `${name} removes the need for validation`, 'The lesson is a conceptual model, not a guarantee about every deployment.'],
    ['Mechanism', `What does a good self-check for ${name} ask you to do?`, `Use ${description} to predict a result before reading the explanation`, 'Repeat the title without connecting it to behavior', 'Skip the explanation after a wrong answer', 'A useful self-check connects the concept to a concrete prediction.'],
    ['Mechanism', `Why does ${name} matter in a larger ML workflow?`, 'It clarifies a decision or transformation that affects model behavior', 'It guarantees a model is fair and calibrated by itself', 'It removes the need to inspect data', 'Most lessons matter because a local mechanism changes downstream behavior.'],
    ['Mechanism', `Which question tests applied understanding of ${name}?`, `What changes when ${changePrompt}?`, 'What is the exact pixel color of the card border?', 'Can the title be alphabetized?', 'Applied understanding means predicting behavior under a meaningful change.'],
    ['Mechanism', `What should you do after a wrong answer in this check?`, 'Read the explanation and retry the same idea with a smaller example', 'Mark the lesson complete without revisiting it', 'Assume the animation is irrelevant', 'The explanation is meant to repair the mental model before moving on.'],
    ['Mechanism', `Which mechanism-level answer is strongest for ${name}?`, `It explains how ${objectiveSummary}`, 'It lists unrelated tools with no causal link', 'It ignores the observed output', 'Mechanism-level answers connect the lesson goal to a causal process.'],
    ['Mechanism', `What is a useful invariant to look for in ${name}?`, 'Something that should stay true even as examples or settings change', 'A fact that is true only for one pixel arrangement', 'A random detail that disappears after reset', 'Invariants help you separate the concept from incidental presentation.'],
    ['Mechanism', `When ${name} has multiple steps, how should you reason through them?`, 'Track the intermediate state after each step before jumping to the final output', 'Jump straight to the answer choice that sounds complex', 'Ignore intermediate values because only the last screen matters', 'Interviewers often test whether you can trace the chain, not just name it.'],
    ['Mechanism', `Which comparison deepens understanding of ${name}?`, 'Compare the current state with a nearby counterexample or alternative setting', 'Compare the font size with the page header', 'Compare unrelated lessons alphabetically', 'A counterexample clarifies what the mechanism is and is not doing.'],
    ['Mechanism', `What should you inspect when two ${name} outputs differ?`, 'The input, setting, assumption, or intermediate value that changed', 'Only whether the page was refreshed', 'The order of answer letters', 'Different outputs usually come from a changed condition in the mechanism.'],
    ['Mechanism', `How can you explain ${name} without hand-waving?`, 'Name the inputs, the transformation rule, the output, and the limitation', 'Use more jargon until the listener stops asking', 'Say it is obvious from the diagram', 'Clear explanations expose the actual moving parts.'],
    ['Mechanism', `What is the role of the common misconception in ${name}?`, 'It marks a tempting but wrong shortcut in the reasoning', 'It is the recommended answer for hard questions', 'It is unrelated to the lesson', 'Misconceptions are included because they often appear in interviews and debugging.'],
    ['Mechanism', `Which answer shows that you can trace ${name}?`, 'You can predict the next state from the current state and the selected control', 'You can remember the lesson category only', 'You can avoid all calculations by guessing', 'Tracing requires using the current state to forecast the next one.'],
    ['Mechanism', `Why should ${name} include both easy and hard questions?`, 'Easy questions establish vocabulary; hard questions test transfer and edge cases', 'All questions should be equally vague', 'Only trick questions teach concepts', 'A good progression builds the concept before stressing it.'],
    ['Mechanism', `What makes an explanation for ${name} interview-ready?`, 'It is precise, causal, and includes a boundary condition', 'It is long but avoids the main idea', 'It only names libraries', 'Interview explanations need correctness and scope, not just fluency.'],
    ['Mechanism', `How should you handle a formula or diagram in ${name}?`, 'Map each symbol or component to the visible behavior it controls', 'Memorize the shape without knowing what it represents', 'Ignore the diagram once the quiz starts', 'Symbols become useful when grounded in behavior.'],
    ['Mechanism', `Which detail is most likely to matter in a ${name} implementation?`, 'The assumption that controls when the simplified mechanism remains valid', 'The exact order of decorative labels', 'Whether the lesson card is first in the catalog', 'Implementation failures often come from violated assumptions.'],
    ['Mechanism', `What should you test if you think you understand ${name}?`, 'A small changed example that forces the mechanism to update', 'The same memorized example only', 'An unrelated page route', 'A new example tests whether the rule transfers.'],
    ['Mechanism', `What is the mechanism-level success criterion for ${name}?`, 'You can walk from input to output and justify each step', 'You can finish by guessing repeated letters', 'You can avoid the explanation panel', 'Mechanism mastery means the path from cause to effect is clear.'],

    ['Application', `How would you use ${name} in a practical ML discussion?`, `Explain when ${description} changes a model or system decision`, 'Use it as proof that no evaluation is needed', 'Mention it without connecting it to a system', 'Applied understanding ties the concept to decisions in a workflow.'],
    ['Application', `A teammate asks why ${name} matters. What is the best answer?`, 'It changes how we reason about inputs, outputs, trade-offs, or failure modes', 'It matters only because it appears in the catalog', 'It matters because all options are usually correct', 'A practical answer connects the lesson to engineering judgment.'],
    ['Application', `Which practical question should ${name} help you answer?`, `How to choose or diagnose behavior when ${changePrompt}`, 'How to avoid reading validation results', 'How to make every model larger', 'The useful skill is making or diagnosing a decision under a changed condition.'],
    ['Application', `What is a good experiment after learning ${name}?`, 'Construct a tiny example, predict the result, then compare with the lesson behavior', 'Only reread the title', 'Skip directly to deployment', 'Small experiments make abstract concepts operational.'],
    ['Application', `How should ${name} affect debugging?`, 'It gives a checklist of mechanism, assumptions, inputs, and observed output', 'It removes the need to inspect intermediate states', 'It makes every error a UI problem', 'Debugging starts by matching observed behavior to the mechanism.'],
    ['Application', `Which scenario shows good transfer from ${name}?`, `Applying ${objectiveSummary} to a new example with different values`, 'Repeating one memorized answer on every page', 'Ignoring prerequisites and constraints', 'Transfer means the concept works beyond the exact animation state.'],
    ['Application', `What should you ask before using ${name} in a production design?`, 'Do the lesson assumptions match the real data, scale, latency, and evaluation goal?', 'Can we avoid defining success?', 'Will a bigger model make the concept irrelevant?', 'Production use requires matching assumptions to constraints.'],
    ['Application', `Which metric or observation would make ${name} actionable?`, 'One that reveals whether the mechanism improved the intended behavior', 'One chosen only because it is easy to maximize', 'One measured on data that influenced the design repeatedly', 'Metrics matter when they align with the decision the concept affects.'],
    ['Application', `What should you communicate to a non-expert about ${name}?`, `The simple purpose, the key trade-off, and one example of ${description}`, 'Every internal implementation detail first', 'Only the acronym if one exists', 'Stakeholder explanations should preserve purpose and risk.'],
    ['Application', `When should you revisit the prerequisite material for ${name}?`, 'When you can follow the words but cannot predict the behavior', 'Only after completing every advanced question', 'Never, because prerequisites are unrelated', 'Prediction failure often points to a missing prerequisite.'],
    ['Application', `How can ${name} guide model review?`, 'Use it to ask whether the model behavior matches the intended mechanism and limits', 'Use it to approve any high training score', 'Use it to ignore data quality', 'Review should connect behavior to intent and constraints.'],
    ['Application', `Which application answer is too shallow for ${name}?`, 'It works because the animation says so', `It works when ${description} matches the task`, 'Its failure modes should be checked', 'A shallow answer cites authority instead of mechanism.'],
    ['Application', `How does ${name} help with trade-off thinking?`, 'It shows what improves, what may degrade, and what assumption controls the trade-off', 'It proves there are no trade-offs', 'It makes every setting equally good', 'Most ML concepts are useful because they expose a trade-off.'],
    ['Application', `What is a practical sign that you only memorized ${name}?`, 'You cannot adjust your answer when the setup changes', 'You can explain the limitation', 'You can name the prerequisite', 'Memorization breaks when the example changes.'],
    ['Application', `How should you answer a "when would you not use this?" question about ${name}?`, 'Name a violated assumption, failure mode, or better alternative', 'Say there is never a reason not to use it', 'Say only that it is complicated', 'Good applied knowledge includes when to avoid a tool.'],
    ['Application', `Which project note would be useful after studying ${name}?`, `Use ${name} when the task needs ${description}; check assumptions and evaluation before trusting results`, 'Question complete', 'Button clicked', 'Useful notes turn lesson knowledge into future engineering action.'],
    ['Application', `A result from ${name} improves one metric but worsens another. What should you do?`, 'Tie the decision to the real objective and document the trade-off', 'Always keep the setting that improves the first metric seen', 'Ignore the worsened metric', 'Applied ML requires choosing trade-offs against the task objective.'],
    ['Application', `What makes ${name} useful in an interview system-design answer?`, 'It gives a concrete mechanism plus operational caveats', 'It fills time without details', 'It avoids mentioning evaluation', 'Strong system-design answers include mechanism and operations.'],
    ['Application', `How can ${name} connect to ${laterUse}?`, 'It provides a mental model that later lessons reuse or stress-test', 'It blocks later lessons from adding nuance', 'It is unrelated once the page is closed', 'Curriculum concepts compound across lessons.'],
    ['Application', `What is the application-level success criterion for ${name}?`, 'You can choose, explain, and validate the concept in a realistic scenario', 'You can only answer definition questions', 'You can avoid edge cases', 'Application mastery means using the concept under realistic constraints.'],

    ['Tricky', `What is the common trap to avoid in ${name}?`, withPeriod(misconception), 'The safest answer is always the longest choice.', 'The visual explanation replaces all mathematical reasoning.', withPeriod(misconception)],
    ['Tricky', `If a result in ${name} looks surprisingly good, what should you inspect?`, 'Whether the setup, assumptions, or evaluation boundary made the result easier than it should be', 'Whether the page title changed after scrolling', 'Whether every unrelated lesson has the same answer', 'Strong-looking results should be checked against assumptions and evaluation boundaries.'],
    ['Tricky', `How can ${name} fail when used carelessly?`, 'The learner may transfer the simplified rule beyond its assumptions', 'It automatically creates more training data', 'It prevents all edge cases by definition', 'A concept animation is useful, but its assumptions still matter.'],
    ['Tricky', `Which answer shows transfer beyond memorization?`, `Explaining how ${name} would behave on a slightly different input or setting`, 'Quoting the first sentence only', 'Choosing the same option letter every time', 'Transfer means applying the idea to a new but related case.'],
    ['Tricky', `What should be true before calling ${name} understood?`, 'You can state the mechanism, use it on an example, and name a likely failure mode', 'You only remember the lesson number', 'You skipped the check but opened the page', 'Understanding needs mechanism, example, and limitation.'],
    ['Tricky', `How does ${name} connect to adjacent lessons?`, 'It uses prerequisites as inputs and prepares a later concept in the same track', 'It is isolated from every other lesson', 'It only matters inside the glossary page', 'The curriculum is a graph: prerequisites feed current concepts and current concepts feed later ones.'],
    ['Tricky', `Which diagnostic question is most useful for ${name}?`, `What assumption would make the ${name} explanation stop working?`, 'Which browser tab was opened first?', 'How many letters are in the category name?', 'Advanced checks ask where the concept breaks, not only where it works.'],
    ['Tricky', `What is the best final review move for ${name}?`, `Summarize ${description}, give one example, and name one trap`, 'Close the page immediately after one correct answer', 'Only memorize the route URL', 'A final review should compress the concept into use, example, and caution.'],
    ['Tricky', `An interviewer changes one condition in a ${name} example. What should you do first?`, 'Recompute the effect of that condition through the mechanism', 'Repeat the original answer unchanged', 'Assume the condition is irrelevant', 'Tricky questions often differ by one condition that changes the conclusion.'],
    ['Tricky', `What makes a ${name} answer subtly wrong?`, 'It states a true fact but applies it outside the assumptions of this setup', 'It includes a concise limitation', 'It explains the input-output chain', 'Many hard interview misses come from true statements used in the wrong scope.'],
    ['Tricky', `Which ${name} result should make you suspicious?`, 'One that improves the headline outcome while hiding a worse boundary case', 'One that includes an explanation of assumptions', 'One that is checked on a held-out example', 'Suspicious results often optimize one visible outcome while breaking a hidden constraint.'],
    ['Tricky', `If two answer choices about ${name} both sound plausible, how should you decide?`, 'Pick the one that follows from the mechanism and stated assumptions', 'Pick the more absolute statement', 'Pick the one with more jargon', 'Precise assumptions should beat confident wording.'],
    ['Tricky', `Which phrase is a warning sign in a ${name} explanation?`, 'Always works, regardless of data, assumptions, or scale', 'Works under these assumptions', 'Fails when the boundary is violated', 'Absolute claims are often wrong in ML interviews.'],
    ['Tricky', `How should you treat a high score after tuning around ${name}?`, 'Ask whether the evaluation data influenced the choices being judged', 'Assume the score proves generalization', 'Delete the validation set', 'Evaluation boundaries matter whenever choices are tuned.'],
    ['Tricky', `What is the best way to handle an edge case in ${name}?`, 'Name the edge case, predict the failure, and propose a check or fallback', 'Ignore it because the main example worked', 'Claim edge cases cannot occur', 'Senior answers anticipate failure and mitigation.'],
    ['Tricky', `Which ${name} answer would fail a big tech interview?`, 'A memorized definition with no trade-off, example, or failure mode', 'A causal explanation with a limitation', 'A small worked example', 'Interviewers test whether you can reason, not only recall.'],
    ['Tricky', `What should you do if the ${name} animation and a real system differ?`, 'Identify which simplification or implementation detail explains the difference', 'Assume the animation is useless', 'Assume the real system is wrong', 'Teaching models simplify; real systems add details that should be named.'],
    ['Tricky', `Which hidden dependency can break ${name}?`, 'A prerequisite concept, data condition, or evaluation assumption that was silently violated', 'The lesson order in the sidebar only', 'The number of page dots', 'Hard failures often come from unstated dependencies.'],
    ['Tricky', `How should you answer a trick question about ${name}?`, 'Slow down, restate the assumptions, and trace the mechanism before choosing', 'Guess quickly from the most familiar word', 'Ignore the changed condition', 'Trick questions reward disciplined assumption checking.'],
    ['Tricky', `What is the tricky-level success criterion for ${name}?`, 'You can catch plausible but wrong reasoning and explain the correction', 'You can answer only the first definition prompt', 'You can avoid all counterexamples', 'Tricky mastery means detecting and repairing flawed reasoning.'],

    ['Interview', `How would you explain ${name} in a 60-second interview answer?`, 'State the problem, mechanism, trade-off, and one failure mode', 'List every UI control in order', 'Say it is complicated and stop', 'A strong short answer has structure: problem, mechanism, trade-off, failure.'],
    ['Interview', `What follow-up should you expect after defining ${name}?`, 'A changed example that tests whether the mechanism transfers', 'A request to name the page route', 'A question about decorative colors', 'Interviewers usually move from definition to transfer.'],
    ['Interview', `What is the strongest way to defend a design using ${name}?`, 'Connect the concept to the objective, constraints, validation plan, and fallback', 'Say it is popular', 'Avoid mentioning risks', 'Design defense needs both benefit and risk control.'],
    ['Interview', `What should you include when comparing ${name} with an alternative?`, 'Assumptions, behavior under change, cost, and failure modes', 'Only which name sounds newer', 'Only the easiest implementation detail', 'Comparisons are useful when tied to constraints and behavior.'],
    ['Interview', `How should you handle a whiteboard problem involving ${name}?`, 'Work a tiny example, show intermediate states, then generalize', 'Jump to a final answer without reasoning', 'Refuse to use examples', 'Tiny examples reveal whether the general rule is understood.'],
    ['Interview', `What would make your ${name} answer senior-level?`, 'You explain the mechanism, the evaluation boundary, and a realistic production caveat', 'You only repeat textbook wording', 'You avoid discussing assumptions', 'Senior-level answers connect concept, measurement, and operations.'],
    ['Interview', `Which interview response best handles uncertainty about ${name}?`, 'State the assumption you are making and reason conditionally from it', 'Pretend every detail is certain', 'Change the topic', 'Explicit assumptions are better than hidden guesses.'],
    ['Interview', `How would you debug a failed ${name}-based system?`, 'Check data, assumptions, intermediate states, metrics, and deployment mismatch in order', 'Only increase model size', 'Only change UI labels', 'A disciplined debugging path avoids random fixes.'],
    ['Interview', `What is a good "gotcha" question for ${name}?`, `What changes if ${changePrompt}?`, 'What color is the icon?', 'What is the exact lesson index?', 'Good gotchas test boundary conditions or changed assumptions.'],
    ['Interview', `How would you show depth beyond the lesson animation for ${name}?`, 'Discuss how scale, noise, latency, or evaluation changes the simple story', 'Say the animation covers every real-world issue', 'Avoid mentioning trade-offs', 'Depth comes from extending the mechanism into real constraints.'],
    ['Interview', `Which statement would you challenge in a design review about ${name}?`, 'This will work regardless of data distribution or evaluation setup', 'Here is the assumption we need to validate', 'This trade-off needs monitoring', 'Design review should challenge absolute unvalidated claims.'],
    ['Interview', `How should ${name} appear in a resume or project explanation?`, 'As a concrete decision you made, why it fit, and how you evaluated it', 'As an unexplained buzzword', 'As proof that no mistakes were possible', 'Projects need evidence of judgment, not just terminology.'],
    ['Interview', `What is the best final question to ask yourself about ${name}?`, 'Can I solve a new case, explain a failure, and choose a trade-off?', 'Can I remember the first option letter?', 'Can I skip all advanced pages?', 'Interview readiness means new-case reasoning, failure analysis, and trade-off selection.'],
    ['Interview', `How would you respond if an interviewer says your ${name} answer is incomplete?`, 'Ask which assumption or case they want to stress, then refine the mechanism', 'Repeat the same sentence louder', 'Switch to an unrelated topic', 'Good candidates use feedback to expose and refine assumptions.'],
    ['Interview', `Which ${name} answer sounds fluent but weak?`, 'A long explanation that never predicts behavior under a changed setup', 'A concise worked example with a limitation', 'A trade-off tied to the objective', 'Fluency without prediction is weak for technical interviews.'],
    ['Interview', `How should you prepare for hard questions after ${name}?`, 'Practice changing one assumption at a time and explaining the downstream effect', 'Only reread the easiest definition', 'Avoid counterexamples', 'Hard questions are usually controlled perturbations of the concept.'],
    ['Interview', `What is the best way to close an answer about ${name}?`, 'Name the practical implication and the main caveat', 'End with no evaluation plan', 'Say all choices are equivalent', 'A strong close tells the interviewer how the concept changes action.'],
    ['Interview', `What distinguishes an interview-ready ${name} explanation from a classroom explanation?`, 'It includes implementation constraints, evaluation, and failure handling', 'It is longer but less precise', 'It avoids examples', 'Interview answers must survive real constraints.'],
    ['Interview', `If asked to teach ${name} to a junior engineer, what sequence should you use?`, 'Definition, tiny example, mechanism trace, common trap, harder variant', 'Hard trick question first with no context', 'Only final formulas', 'Teaching well mirrors the assessment progression.'],
    ['Interview', `What is the interview-level success criterion for ${name}?`, 'You can reason through unfamiliar variants and defend choices under constraints', 'You can only recognize the title', 'You can avoid being asked follow-ups', 'Interview readiness means robust reasoning under changed conditions.'],
  ];

  return specs.slice(0, TARGET_STRATEGY_REVIEW_QUESTIONS).map((spec, index) => (
    makeGeneratedQuestion(index, spec[0], spec[1], spec[2], [spec[3], spec[4]], spec[5])
  ));
}

function makeSeededQuestionVariants(animation, seededQuiz) {
  const name = animation.name;
  const variants = [];

  for (const [questionIndex, question] of seededQuiz.entries()) {
    const correct = question.choices[question.answerIndex];
    const wrongChoices = question.choices.filter((_, index) => index !== question.answerIndex);
    const firstWrong = wrongChoices[0] || 'An unrelated shortcut';
    const secondWrong = wrongChoices[1] || 'A claim that ignores the setup';
    const baseLevel = question.level || 'Mechanism';
    const explanation = withPeriod(question.explanation);

    variants.push(
      [baseLevel, `In ${name}, which answer correctly resolves this case: ${question.prompt}`, correct, firstWrong, secondWrong, explanation],
      ['Mechanism', `For "${question.prompt}", which reasoning path matches the ${name} mechanism?`, explanation, `It follows because ${firstWrong}`, `It follows because ${secondWrong}`, explanation],
      ['Application', `In a new ${name} example matching this setup, which conclusion follows from "${question.prompt}"?`, correct, firstWrong, secondWrong, explanation],
      ['Tricky', `Which tempting but wrong conclusion should ${name} rule out for this case: ${question.prompt}`, firstWrong, correct, secondWrong, `The tempting wrong answer is "${firstWrong}", but the lesson point is: ${explanation}`],
    );

    if (questionIndex % 2 === 0) {
      variants.push(
        ['Application', `Which additional result would confirm the same ${name} idea as "${correct}"?`, `A result matching this rule: ${explanation}`, `A result that supports "${secondWrong}" instead`, 'A result that ignores the stated setup', explanation],
      );
    }
  }

  return variants;
}

function makeMetadataCoreSpecs(animation) {
  const objectives = animation.learningObjectives?.length
    ? animation.learningObjectives
    : [`Explain ${animation.name}`, `Predict how ${animation.description.toLowerCase()} changes behavior`];
  const description = cleanSentence(animation.description).toLowerCase();
  const misconception = animation.commonMisconception || `${animation.name} should not be treated as a production guarantee.`;
  const prereqs = animation.prerequisites?.length ? animation.prerequisites.join(', ') : 'the lesson setup';
  const name = animation.name;
  const primaryObjective = cleanSentence(objectives[0]).toLowerCase();
  const secondaryObjective = cleanSentence(objectives[1] || objectives[0]).toLowerCase();
  const prerequisiteContext = prereqs === 'the lesson setup'
    ? 'the stated lesson setup'
    : `the prerequisite ideas: ${prereqs}`;

  return [
    ['Foundation', `In ${name}, which situation matches the lesson concept?`, `A case where ${description} determines the result`, 'A case where the option letters determine the result', 'A case with no inputs, assumptions, or outputs to inspect', `${name} is centered on ${description}.`],
    ['Foundation', `Which statement is required for a correct ${name} example?`, withPeriod(objectives[0]), 'The final answer can be chosen without reading the setup.', 'The mechanism is irrelevant once the title is known.', withPeriod(objectives[0])],
    ['Foundation', `What background should be active when solving a ${name} question?`, prerequisiteContext, 'Only unrelated interface terminology', 'A deployment dashboard with no concept setup', `${name} builds on ${prerequisiteContext}.`],
    ['Mechanism', `A ${name} setting changes. What should you compare before choosing an answer?`, `The inputs, assumptions, and output behavior tied to ${description}`, 'Only the order of the multiple-choice answers', 'Nothing, because the title fixes the output', `The lesson asks you to connect setup changes to ${description}.`],
    ['Mechanism', `Which causal explanation fits ${name}?`, `The changed setup affects ${description}, so the observed behavior should change accordingly`, 'The result changes because the page layout changed', 'The result is unrelated to the lesson assumptions', `${name} should be explained as a mechanism, not a label.`],
    ['Mechanism', `What distinction does ${name} force you to make?`, `${description} versus the shortcut: ${cleanSentence(misconception).toLowerCase()}`, 'The question id versus the answer id', 'The sidebar order versus the page title', misconception],
    ['Application', `A practitioner applies ${name} to a new case. What should they validate?`, `That ${description} matches the task constraints and observed behavior`, 'That the lesson was opened once', 'That every metric or setting moves together', `The module is useful when ${description} affects a real choice.`],
    ['Application', `Which decision is most aligned with ${name}?`, `Use ${description} to decide whether the current behavior supports the task`, 'Ignore evaluation because the diagram looked plausible', 'Assume every production setting has the same internals', `A correct application ties ${name} to evidence from the setup.`],
    ['Application', `Which follow-up question is specific to ${name}?`, `How would ${description} change when ${secondaryObjective}?`, 'Which route URL contains the lesson?', 'Which answer letter appeared most often?', `The follow-up should stay tied to the ${name} mechanism.`],
    ['Tricky', `Which conclusion would be unsafe in ${name}?`, misconception, `The result depends on ${description}`, `The answer should be checked against ${primaryObjective}`, `The common misconception is the trap this lesson is meant to correct.`],
    ['Tricky', `A simplified ${name} example looks correct. What is the next check?`, `Whether the assumptions behind ${description} still hold in the new setting`, 'Whether the same answer letter was correct last time', 'Whether the concept can be used without any validation', 'The teaching model is useful, but it is not a universal guarantee.'],
    ['Interview', `In a technical screen, which ${name} claim is defensible?`, `${name} can explain ${description}, but the assumptions and limits still need to be checked`, `${name} proves every real system behaves exactly like the teaching example`, `${name} is mastered by repeating the title alone`, `A strong answer stays specific to ${name} and includes a limitation.`],
  ];
}

function makeGeneratedQuiz(animation, seededQuiz = []) {
  const seededVariants = makeSeededQuestionVariants(animation, seededQuiz);
  const metadataSpecs = makeMetadataCoreSpecs(animation);
  const specs = [];

  while (specs.length < TARGET_QUIZ_QUESTIONS) {
    const metadataIndex = specs.length - seededVariants.length;
    const source = specs.length < seededVariants.length
      ? seededVariants[specs.length]
      : contextualizeCoreSpec(
        metadataSpecs[metadataIndex % metadataSpecs.length],
        Math.floor(metadataIndex / metadataSpecs.length),
      );
    specs.push(source);
  }

  return specs.map((spec, index) => (
    makeGeneratedQuestion(index, spec[0], spec[1], spec[2], [spec[3], spec[4]], spec[5])
  ));
}

function stableHash(value) {
  return [...String(value)].reduce((hash, char) => ((hash * 31) + char.charCodeAt(0)) >>> 0, 0);
}

function rotateQuestionChoices(question, offset) {
  if (!Array.isArray(question.choices) || question.choices.length < 2) return question;

  const correctChoice = question.choices[question.answerIndex];
  const rotation = offset % question.choices.length;
  const rotated = [...question.choices.slice(rotation), ...question.choices.slice(0, rotation)];

  return {
    ...question,
    choices: rotated,
    answerIndex: rotated.indexOf(correctChoice),
  };
}

function normalizeSeededQuestion(question, index, lessonId) {
  const levels = ['Foundation', 'Mechanism', 'Application', 'Tricky', 'Interview'];
  const normalized = {
    level: question.level || levels[Math.min(levels.length - 1, Math.floor(index / 4))],
    ...question,
  };

  return rotateQuestionChoices(normalized, stableHash(`${lessonId}:${question.id}`));
}

function buildLessonAssessment(animation) {
  const seeded = SEEDED_LESSON_ASSESSMENTS[animation.id] || EMPTY_ASSESSMENT;
  const seededQuiz = (seeded.quiz || []).map((question, index) => (
    normalizeSeededQuestion(question, index, animation.id)
  ));
  const usedIds = new Set(seededQuiz.map((question) => question.id));
  const generatedQuiz = makeGeneratedQuiz(animation, seededQuiz)
    .filter((question) => !usedIds.has(question.id))
    .slice(0, Math.max(0, TARGET_QUIZ_QUESTIONS - seededQuiz.length));

  return {
    completionPolicy: DEFAULT_COMPLETION_POLICY,
    scenarioQuestions: seeded.scenarioQuestions || getScenarioQuestionsForLesson(animation.id),
    quiz: [...seededQuiz, ...generatedQuiz]
      .slice(0, TARGET_QUIZ_QUESTIONS)
      .map((question, index) => ({
        skill: question.skill || (question.level === 'Foundation' ? 'recall' : question.level === 'Mechanism' ? 'mechanism' : 'transfer'),
        ...question,
        countsForCompletion: index < DEFAULT_COMPLETION_POLICY.masteryRequired,
      })),
    strategyReview: makeLearningStrategyDeck(animation),
    labs: seeded.labs || [],
  };
}

export const lessonAssessments = Object.freeze(Object.fromEntries(
  allAnimations.map((animation) => [animation.id, buildLessonAssessment(animation)]),
));

export function getLessonAssessment(lessonId) {
  return lessonAssessments[lessonId] || EMPTY_ASSESSMENT;
}

export function hasAssessmentContent(assessment) {
  return Boolean(assessment?.scenarioQuestions?.length || assessment?.quiz?.length || assessment?.labs?.length);
}

export function getAssessmentStats(assessments = lessonAssessments) {
  return Object.values(assessments).reduce(
    (stats, assessment) => ({
      totalQuizQuestions: stats.totalQuizQuestions + (assessment.quiz?.length || 0),
      totalLabs: stats.totalLabs + (assessment.labs?.length || 0),
    }),
    { totalQuizQuestions: 0, totalLabs: 0 },
  );
}
