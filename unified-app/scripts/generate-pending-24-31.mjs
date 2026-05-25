import { writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const outPath = join(__dirname, '../src/data/_pending-24-31.fragment.js');

function escapeStr(value) {
  return String(value)
    .replace(/\\/g, '\\\\')
    .replace(/'/g, "\\'")
    .replace(/\r\n/g, '\n')
    .replace(/\n/g, '\\n');
}

function renderLeaf(node) {
  const fields = Object.entries(node.tip)
    .map(([key, value]) => `              ${key}: '${escapeStr(value)}',`)
    .join('\n');
  const lesson = node.lessonId ? `\n            lessonId: '${node.lessonId}',` : '';
  return `          {
            id: '${node.id}',
            label: '${node.label.replace(/'/g, "\\'")}',
            tooltip: tip({
${fields}
            }),${lesson}
          }`;
}

function renderBranch(branch) {
  const children = branch.children.map(renderLeaf).join(',\n');
  return `      {
        id: '${branch.id}',
        label: '${branch.label}',
        type: '${branch.type}',
        children: [
${children}
        ],
      }`;
}

function renderMap(map) {
  const centerFields = Object.entries(map.center)
    .map(([key, value]) => `        ${key}: '${escapeStr(value)}',`)
    .join('\n');
  const branches = map.branches.map(renderBranch).join(',\n');
  return `  '${map.id}': {
    center: {
      id: '${map.id}',
      label: '${map.label}',
      type: 'current',
      tooltip: tip({
${centerFields}
      }),
    },
    branches: [
${branches}
    ],
  }`;
}

const MAPS = [
  {
    id: 'flow-matching',
    label: 'Flow Matching',
    center: {
      short: 'Flow matching learns a time-dependent velocity field v_θ(x,t) that transports samples along a continuous path from a simple noise distribution p₀ to data p₁.',
      intuition: 'Instead of predicting noise at many diffusion timesteps, train the network to point each particle in the direction it should move right now along a chosen probability path.',
      formula: 'L = \\mathbb{E}_{t,x_0,x_1}\\|v_\\theta(x_t,t) - u_t(x_t|x_0,x_1)\\|^2',
      why: 'Flow matching simplifies generative training with straight or scheduled paths, connects to diffusion and consistency models, and powers modern flow-based image and latent generators.',
      trap: 'Flow matching is not identical to classic DDPM epsilon prediction—the objective, path, and sampler must match the training setup.',
    },
    branches: [
      {
        id: 'prerequisites', label: 'Prerequisites', type: 'prerequisite',
        children: [
          { id: 'noise-distribution-fm', label: 'Noise distribution p₀', tip: { short: 'Start from a simple base like Gaussian N(0,I).', intuition: 'Generative models learn a map from easy-to-sample noise to complex data.', trap: 'Wrong base dimensionality breaks the transport path.' } },
          { id: 'data-distribution-fm', label: 'Data distribution p₁', tip: { short: 'Target distribution of real samples (images, latents).', intuition: 'Training pairs noise draws with data draws along a path.', trap: 'Mismatch between train and inference data scale breaks velocity learning.' } },
          { id: 'ode-flow-fm', label: 'ODE flow', tip: { short: 'dx/dt = v_θ(x,t) integrates particles from t=0 to t=1.', intuition: 'Sampling is numerical integration of the learned field.', trap: 'Euler steps with too few steps yield blurry or biased samples.' }, lessonId: 'diffusion-basics' },
          { id: 'neural-network-fm', label: 'Neural network', tip: { short: 'v_θ is a U-Net or transformer conditioned on time t.', intuition: 'Same backbone family as diffusion denoisers.', trap: 'Time conditioning must be injected consistently at train and sample time.' }, lessonId: 'neural-network' },
          { id: 'probability-path-fm', label: 'Probability path', tip: { short: 'A schedule x_t linking x_0 and x_1 for each t ∈ [0,1].', intuition: 'The path defines the target velocity u_t the network should match.', trap: 'Changing the path after training invalidates the learned field.' } },
          { id: 'gradient-descent-fm', label: 'Gradient descent', tip: { short: 'Minimize expected squared error between predicted and target velocity.', intuition: 'Conditional flow matching uses sampled (t, x_t) pairs each step.', trap: 'Bad learning rate or batch scale can destabilize velocity magnitudes.' }, lessonId: 'gradient-descent' },
        ],
      },
      {
        id: 'mechanism', label: 'Core mechanism', type: 'mechanism',
        children: [
          { id: 'sample-pair-fm', label: 'Sample (x₀, x₁) pair', tip: { short: 'Draw noise x₀ ~ p₀ and data x₁ ~ p₁ each training step.', intuition: 'Supervision comes from known endpoints of the transport.', trap: 'Using only x₁ without paired x₀ leaves the path undefined.' } },
          { id: 'sample-time-fm', label: 'Sample time t', tip: { short: 'Pick t uniformly or from a logit-normal schedule over [0,1].', intuition: 'Network must learn velocities across all times, not just endpoints.', trap: 'Never sampling mid-times yields poor intermediate dynamics.' } },
          { id: 'interpolate-path-fm', label: 'Construct x_t on path', tip: { short: 'Linear path: x_t = (1-t)x₀ + t x₁; or sigma schedules for curved paths.', intuition: 'x_t is the location where velocity is supervised.', formula: 'x_t = (1-t)x_0 + t x_1', trap: 'Path formula must match the u_t target used in the loss.' } },
          { id: 'target-velocity-fm', label: 'Target velocity u_t', tip: { short: 'For linear path, u_t = x₁ − x₀ (constant in t).', intuition: 'The network learns to output this direction at each (x_t, t).', formula: 'u_t = x_1 - x_0', trap: 'Using DDPM noise targets with flow paths mixes incompatible objectives.' } },
          { id: 'velocity-loss-fm', label: 'Velocity matching loss', tip: { short: 'L = ||v_θ(x_t,t) − u_t||² averaged over batch.', intuition: 'Regression on vector fields, not scalar noise levels.', trap: 'Forgetting to stop-gradient x_t targets causes unstable training.' } },
          { id: 'euler-sampling-fm', label: 'Euler sampling', tip: { short: 'Integrate x ← x + Δt · v_θ(x,t) from t=0 to 1.', intuition: 'More steps improve sample quality at higher compute.', trap: 'Large Δt with curved true dynamics accumulates integration error.' } },
        ],
      },
      {
        id: 'intuitions', label: 'Intuitions', type: 'intuition',
        children: [
          { id: 'river-flow-fm', label: 'River flow picture', tip: { short: 'Particles in noise are carried by a learned river toward the data manifold.', intuition: 'Velocity arrows show instantaneous direction, not the final destination alone.', trap: 'A locally correct field can still need many steps globally.' } },
          { id: 'straight-vs-curved-fm', label: 'Straight vs curved paths', tip: { short: 'Linear paths give constant targets; sigma schedules bend trajectories.', intuition: 'Curved paths can align with diffusion-like schedules for distillation.', trap: 'Straight-path training does not automatically transfer to curved inference.' } },
          { id: 'cfm-vs-diffusion-fm', label: 'CFM vs diffusion', tip: { short: 'Flow matching supervises velocity; DDPM often predicts noise ε.', intuition: 'Both build generative models but with different parameterizations.', trap: 'Swapping samplers across paradigms without retraining fails.' }, lessonId: 'diffusion-basics' },
          { id: 'fewer-timesteps-fm', label: 'Fewer integration steps', tip: { short: 'Well-learned fields can sample in 10–50 steps vs hundreds in diffusion.', intuition: 'Straight paths reduce accumulated discretization error.', trap: 'Aggressive step reduction still needs validation on FID or human eval.' } },
          { id: 'logit-normal-t-fm', label: 'Logit-normal time sampling', tip: { short: 'Bias t sampling toward mid-times where learning is hardest.', intuition: 'Uniform t can under-weight regions needing finer velocity detail.', trap: 'Extreme t biasing neglects endpoint behavior near t=0 or t=1.' } },
        ],
      },
      {
        id: 'formula-code', label: 'Formula / Code', type: 'formula',
        children: [
          { id: 'linear-path-formula-fm', label: 'Linear path', tip: { short: 'x_t = (1−t)x₀ + t x₁.', intuition: 'Optimal transport straight-line interpolation.', formula: 'x_t = (1-t)x_0 + t x_1', trap: 't outside [0,1] extrapolates beyond training support.' } },
          { id: 'cfm-loss-formula-fm', label: 'CFM loss', tip: { short: 'E[||v_θ(x_t,t) − (x₁−x₀)||²].', intuition: 'Conditional flow matching with linear conditional paths.', formula: 'L_{CFM} = \\mathbb{E}\\|v_\\theta(x_t,t) - (x_1-x_0)\\|^2', trap: 'Batch normalization of x_t changes effective velocity scale.' } },
          { id: 'euler-step-code-fm', label: 'Euler step', tip: { short: 'x = x + dt * model(x, t).', intuition: 'Simplest ODE solver for deployment.', code: 'for t in timesteps:\n    v = model(x, t)\n    x = x + dt * v', trap: 'dt sign must match increasing t from noise to data.' } },
          { id: 'sigma-schedule-fm', label: 'Sigma schedule', tip: { short: 'Curved paths use σ(t) blending noise and signal.', intuition: 'Connects flow training to diffusion noise schedules.', trap: 'Sigma must be differentiable and aligned with u_t derivation.' } },
          { id: 'torch-cfm-loop-fm', label: 'Training loop sketch', tip: { short: 'Sample batch, t, interpolate, predict v, MSE loss, backward.', intuition: 'No Markov chain of many noise levels required per step.', code: 'x_t = (1-t)*x0 + t*x1\nloss = F.mse_loss(model(x_t,t), x1-x0)', trap: 'x0 and x1 must share shape and device.' } },
        ],
      },
      {
        id: 'traps', label: 'Common traps', type: 'trap',
        children: [
          { id: 'ddpm-objective-trap-fm', label: 'DDPM objective confusion', tip: { short: 'Epsilon prediction loss is not flow-matching without re-derivation.', intuition: 'Objectives encode different probability paths.', trap: 'Copying diffusion code verbatim mis-trains the velocity field.' } },
          { id: 'path-mismatch-trap-fm', label: 'Path mismatch at inference', tip: { short: 'Sampler path must match training path assumptions.', intuition: 'Integration starts from p₀ regardless of how x_t was built in training.', trap: 'Using diffusion schedulers on flow-trained models degrades samples.' } },
          { id: 'too-few-steps-trap-fm', label: 'Too few Euler steps', tip: { short: 'Under-integration leaves samples noisy or off-manifold.', intuition: 'Quality vs speed tradeoff is explicit.', trap: 'One-step fantasies require distillation, not default flow matching.' } },
          { id: 'scale-mismatch-trap-fm', label: 'Data scale mismatch', tip: { short: 'Images in [0,1] vs [−1,1] change velocity magnitudes.', intuition: 'Normalize consistently across train and sample.', trap: 'Latent scaling with VAE must match flow training domain.' }, lessonId: 'vae' },
          { id: 'time-conditioning-trap-fm', label: 'Weak time conditioning', tip: { short: 'If t is ignored, the field cannot vary across the path.', intuition: 'Timestep embeddings or Fourier features are mandatory.', trap: 'Collapsing t to constant yields a single global direction field.' } },
        ],
      },
      {
        id: 'used-later', label: 'Used later', type: 'application',
        children: [
          { id: 'diffusion-basics-used-fm', label: 'Diffusion basics', tip: { short: 'Diffusion SDEs relate to flow paths and distillation targets.', intuition: 'Many models bridge flow matching and score-based diffusion.', trap: 'Not every diffusion checkpoint is a flow model.' }, lessonId: 'diffusion-basics' },
          { id: 'diffusion-sampling-used-fm', label: 'Diffusion sampling', tip: { short: 'Sampler design parallels ODE solvers for flows.', intuition: 'Step count and schedule tuning transfer conceptually.', trap: 'DDIM coefficients do not apply directly to raw flow Euler.' }, lessonId: 'diffusion-sampling' },
          { id: 'dit-used-fm', label: 'DiT', tip: { short: 'Transformer backbones predict velocity fields on patch tokens.', intuition: 'Architecture choice is orthogonal to flow objective.', trap: 'Patch size affects effective Lipschitz of the learned field.' }, lessonId: 'dit' },
          { id: 'unet-vs-dit-used-fm', label: 'UNet vs DiT', tip: { short: 'Compare inductive biases for spatial velocity prediction.', intuition: 'Both appear in flow-matching image pipelines.', trap: 'Switching backbone mid-project resets training schedule.' }, lessonId: 'unet-vs-dit' },
          { id: 'classifier-free-used-fm', label: 'Classifier-free guidance', tip: { short: 'Guidance mixes conditional and unconditional velocity fields at sample time.', intuition: 'Same trick as diffusion guidance on predicted updates.', trap: 'Guidance scale too high breaks flow dynamics.' }, lessonId: 'classifier-free-guidance' },
        ],
      },
    ],
  },
  {
    id: 'frontier-evaluation-safety',
    label: 'Frontier Evaluation & Safety',
    center: {
      short: 'Frontier evaluation separates capability benchmarks, product reliability, agent safety, and catastrophic-risk evidence before deployment decisions on powerful models.',
      intuition: 'A high MMLU score does not prove safe tool use, honest refusal, or robustness to jailbreaks—each claim needs its own measurement layer.',
      formula: 'Deploy\\;\\Leftrightarrow\\;Cap \\land Rel \\land AgentSafe \\land RiskEvidence',
      why: 'Shipping frontier LLMs without layered eval invites capability overhang, silent regressions, and governance gaps that single leaderboard numbers hide.',
      trap: 'One aggregate benchmark score is not a safety clearance—it can mask failure modes that only appear under agents, long context, or adversarial prompts.',
    },
    branches: [
      { id: 'prerequisites', label: 'Prerequisites', type: 'prerequisite', children: [
        { id: 'classification-metrics-fes', label: 'Classification metrics', tip: { short: 'Precision, recall, and F1 quantify binary eval outcomes.', intuition: 'Safety classifiers and refusal detectors are evaluated like classifiers.', trap: 'Accuracy alone hides imbalance on rare harmful classes.' }, lessonId: 'classification-metrics' },
        { id: 'calibration-fes', label: 'Calibration', tip: { short: 'Predicted probabilities should match empirical frequencies.', intuition: 'Confidence-gated routing depends on calibrated scores.', trap: 'Overconfident models bypass human review thresholds.' }, lessonId: 'calibration' },
        { id: 'ab-testing-fes', label: 'A/B testing', tip: { short: 'Compare product variants with controlled traffic splits.', intuition: 'Online eval complements offline benchmarks.', trap: 'Peeking and underpowered tests give false safety signals.' }, lessonId: 'ab-testing-foundations' },
        { id: 'model-monitoring-fes', label: 'Model monitoring', tip: { short: 'Track live drift, latency, and quality slices after launch.', intuition: 'Post-deploy eval is continuous, not a one-time gate.', trap: 'Offline-green models can regress in production data.' }, lessonId: 'model-monitoring' },
        { id: 'model-reliability-fes', label: 'Model reliability', tip: { short: 'Reliability spans OOD detection, uncertainty, and failure recovery.', intuition: 'Safety sits on top of baseline reliability practices.', trap: 'Skipping reliability makes safety patches whack-a-mole.' }, lessonId: 'model-reliability' },
        { id: 'transformer-fes', label: 'Transformer', tip: { short: 'Frontier models are large autoregressive or hybrid stacks.', intuition: 'Eval must stress the actual architecture and modality mix.', trap: 'Evaluating a distilled stub misses frontier failure modes.' }, lessonId: 'transformer' },
      ]},
      { id: 'mechanism', label: 'Core mechanism', type: 'mechanism', children: [
        { id: 'capability-layer-fes', label: 'Capability benchmarks', tip: { short: 'MMLU, coding, math, and multimodal tasks measure skill ceilings.', intuition: 'Answers “how capable is the base model?”', trap: 'Training on benchmark leakage inflates capability scores.' } },
        { id: 'reliability-layer-fes', label: 'Product reliability eval', tip: { short: 'Latency, hallucination rate, format adherence, and regression suites.', intuition: 'Answers “does the shipped product work for users?”', trap: 'Base model gains can break prompt templates or tools.' } },
        { id: 'agent-safety-layer-fes', label: 'Agent safety eval', tip: { short: 'Tool misuse, prompt injection, sandbox escapes, and multi-step harm.', intuition: 'Agents amplify single-turn failures across actions.', trap: 'Chat-safe models can be unsafe with code execution.' } },
        { id: 'red-team-layer-fes', label: 'Red teaming', tip: { short: 'Adversarial humans and models probe jailbreaks and policy edge cases.', intuition: 'Finds failures static benchmarks miss.', trap: 'One red-team round does not cover future attack creativity.' } },
        { id: 'guardrails-layer-fes', label: 'Guardrails stack', tip: { short: 'Input filters, output classifiers, tool policies, and human escalation.', intuition: 'Defense in depth when the base model fails.', trap: 'Guardrails add latency and can be circumvented in agents.' } },
        { id: 'risk-evidence-fes', label: 'Frontier risk evidence', tip: { short: 'Structured evidence on misuse, autonomy, and dual-use capabilities.', intuition: 'Governance needs auditable artifacts, not vibes.', trap: 'Public leaderboard wins are not substitute for internal risk review.' } },
      ]},
      { id: 'intuitions', label: 'Intuitions', type: 'intuition', children: [
        { id: 'layers-not-score-fes', label: 'Layers, not one score', tip: { short: 'Capability, reliability, agent safety, and risk are orthogonal axes.', intuition: 'A model can be smart yet unsafe or safe yet unusable.', trap: 'Averaging layers hides blocking failures on one axis.' } },
        { id: 'eval-games-fes', label: 'Eval gaming', tip: { short: 'Overfitting eval suites teaches to the test without real generalization.', intuition: 'Holdout tasks and dynamic benchmarks reduce gaming.', trap: 'Publishing eval prompts enables train-set contamination.' } },
        { id: 'agent-amplification-fes', label: 'Agent amplification', tip: { short: 'Tool access turns wording errors into filesystem or network actions.', intuition: 'Agent eval must include environment state, not text only.', trap: 'Single-turn refusal tests miss chained tool abuse.' } },
        { id: 'human-in-loop-fes', label: 'Human-in-the-loop', tip: { short: 'High-stakes decisions need escalation paths when automation is uncertain.', intuition: 'Eval should measure when to defer, not only when to answer.', trap: 'Automation bias treats model output as ground truth.' } },
        { id: 'continuous-eval-fes', label: 'Continuous evaluation', tip: { short: 'Each fine-tune, quantize, or prompt change reopens the eval gate.', intuition: 'Safety is a lifecycle, not a launch checkbox.', trap: 'Quantization can silently break refusal behaviors.' } },
      ]},
      { id: 'formula-code', label: 'Formula / Code', type: 'formula', children: [
        { id: 'pass-rate-fes', label: 'Pass@k eval', tip: { short: 'Fraction of tasks solved in k attempts for coding evals.', intuition: 'Stochastic sampling affects agent and code benchmarks.', formula: 'Pass@k = 1 - \\binom{n-c}{k}/\\binom{n}{k}', trap: 'Comparing pass@1 vs pass@10 without labeling misleads.' } },
        { id: 'refusal-rate-fes', label: 'Refusal rate', tip: { short: 'Share of harmful prompts correctly declined.', intuition: 'Balance against over-refusal on benign prompts.', trap: 'High refusal rate with high false-positive rate harms UX.' } },
        { id: 'jailbreak-success-fes', label: 'Jailbreak success rate', tip: { short: 'Adversarial prompt success fraction on policy violations.', intuition: 'Track over red-team rounds and model versions.', trap: 'Cherry-picked attack sets understate real risk.' } },
        { id: 'eval-harness-fes', label: 'Eval harness pattern', tip: { short: 'Prompt → model → parse → grade with deterministic or LLM judge.', intuition: 'Standardize reproducible eval pipelines.', code: 'for case in suite:\n    out = model(case.prompt)\n    scores.append(grader(out, case.rubric))', trap: 'Non-deterministic judges need variance reporting.' } },
        { id: 'regression-gate-fes', label: 'Regression gate', tip: { short: 'Block release if key metrics drop below thresholds vs baseline.', intuition: 'CI for models mirrors software test gates.', trap: 'Too many gates cause alert fatigue and bypass culture.' } },
      ]},
      { id: 'traps', label: 'Common traps', type: 'trap', children: [
        { id: 'leaderboard-trap-fes', label: 'Leaderboard clearance', tip: { short: 'SOTA on public bench ≠ safe to deploy with tools.', intuition: 'Benchmarks sample a tiny slice of real-world harm.', trap: 'Marketing SOTA while skipping agent red teams.' } },
        { id: 'static-suite-trap-fes', label: 'Static suite staleness', tip: { short: 'Fixed prompts get memorized or attacked systematically.', intuition: 'Rotate evals and use adaptive red teaming.', trap: 'Year-old jailbreak sets miss new attack templates.' } },
        { id: 'judge-bias-trap-fes', label: 'LLM judge bias', tip: { short: 'Automated graders inherit model preferences and length bias.', intuition: 'Calibrate judges against human labels.', trap: 'Self-grading with the same model family hides errors.' } },
        { id: 'over-refusal-trap-fes', label: 'Over-refusal ignored', tip: { short: 'Safety tuning can refuse benign medical or coding requests.', intuition: 'Measure false refusal rate alongside harm rate.', trap: 'Optimizing only for harm reduction cripples utility.' } },
        { id: 'post-quant-trap-fes', label: 'Post-quant eval skip', tip: { short: 'INT4/INT8 can break subtle refusal and format behaviors.', intuition: 'Eval the exact artifact users receive.', trap: 'FP16-safe policies fail after aggressive quantization.' } },
      ]},
      { id: 'used-later', label: 'Used later', type: 'application', children: [
        { id: 'model-fairness-used-fes', label: 'Model fairness', tip: { short: 'Slice evals reveal disparate harm or quality across demographics.', intuition: 'Safety and fairness intersect on refusals and stereotypes.', trap: 'Aggregate metrics hide slice-specific failures.' }, lessonId: 'model-fairness' },
        { id: 'model-debugging-used-fes', label: 'Model debugging', tip: { short: 'Trace eval failures to layers, prompts, or data slices.', intuition: 'Failed cases become debugging datasets.', trap: 'Fixing symptoms without root cause regresses elsewhere.' }, lessonId: 'model-debugging' },
        { id: 'agentic-used-fes', label: 'Agentic coding systems', tip: { short: 'Coding agents need sandbox evals for exfiltration and destructive commands.', intuition: 'Capability evals alone miss environment impact.', trap: 'Read-only agent demos hide write-capable failures.' }, lessonId: 'agentic-coding-systems' },
        { id: 'frontier-arch-used-fes', label: 'Frontier architecture overview', tip: { short: 'Architecture choices shift which eval layers matter most.', intuition: 'MoE, long context, and omni each add eval dimensions.', trap: 'Evaluating dense-chat harness on MoE multimodal stacks.' }, lessonId: 'frontier-llm-architecture-overview' },
        { id: 'ml-security-used-fes', label: 'ML security track', tip: { short: 'Adversarial robustness and supply-chain eval extend safety.', intuition: 'Security is complementary to policy alignment.', trap: 'Policy refusals do not stop weight theft or poisoning.' }, lessonId: 'ml-security-robustness-track' },
      ]},
    ],
  },
  {
    id: 'frontier-llm-architecture-overview',
    label: 'Frontier LLM Architecture Overview',
    center: {
      short: 'Frontier LLM architecture compares dense transformers, sparse MoE, compressed attention, long-context stacks, SSM hybrids, diffusion LMs, and omni multimodal designs by active compute, KV memory, context, generation order, and modality.',
      intuition: 'Follow one token through the stack: what changed for this token, and which bottleneck did that design choice buy or sell?',
      formula: 'Cost/token \\approx FLOPs_{active} + BW\\cdot KV_{bytes} + Mem_{params}',
      why: 'Architecture literacy prevents mistaking parameter count for inference cost and explains why frontier models diverge from vanilla GPT-2 stacks.',
      trap: 'Every improvement creates a new failure mode—smaller KV can mean extra projection FLOPs; MoE capacity can collapse under routing imbalance.',
    },
    branches: [
      { id: 'prerequisites', label: 'Prerequisites', type: 'prerequisite', children: [
        { id: 'transformer-flao', label: 'Transformer', tip: { short: 'Decoder-only stacks with self-attention and FFN blocks.', intuition: 'Baseline architecture all frontier variants extend.', trap: '2017 encoder-decoder defaults do not match modern LLM serving.' }, lessonId: 'transformer' },
        { id: 'self-attention-flao', label: 'Self-attention', tip: { short: 'O(T²) attention over sequence length T.', intuition: 'Memory and compute scale with context.', trap: 'Ignoring KV cache conflates prefill with decode costs.' }, lessonId: 'self-attention' },
        { id: 'kv-cache-flao', label: 'KV cache', tip: { short: 'Stored keys/values for past tokens during decode.', intuition: 'Dominates long-context memory bandwidth.', trap: 'Parameter count alone ignores cache bytes.' }, lessonId: 'kv-cache' },
        { id: 'gqa-flao', label: 'Grouped-query attention', tip: { short: 'Share KV heads across query heads.', intuition: 'First lever for cache compression.', trap: 'Extreme sharing hurts quality on some tasks.' }, lessonId: 'grouped-query-attention' },
        { id: 'transformer-families-flao', label: 'Architecture families', tip: { short: 'Catalog of dense, MoE, and hybrid block patterns.', intuition: 'Names map to repeatable design templates.', trap: 'Marketing names outpace consistent definitions.' }, lessonId: 'transformer-architecture-families' },
        { id: 'efficient-serving-flao', label: 'Efficient LLM serving', tip: { short: 'Batching, paging, and quantization affect real throughput.', intuition: 'Architecture and serving co-design determine SLA.', trap: 'Paper FLOPs omit memory bandwidth bounds.' }, lessonId: 'efficient-llm-serving' },
      ]},
      { id: 'mechanism', label: 'Core mechanism', type: 'mechanism', children: [
        { id: 'dense-baseline-flao', label: 'Dense baseline', tip: { short: 'Every token activates all FFN and attention parameters.', intuition: 'Simplest cost model: all params active each layer.', trap: 'Dense 70B feels heavier than sparse 400B active-8B.' } },
        { id: 'sparse-moe-flao', label: 'Sparse MoE layer', tip: { short: 'Router picks top-k experts per token; only those FFNs run.', intuition: 'Total params ≫ active params per token.', trap: 'All experts still occupy memory unless offloaded.' }, lessonId: 'frontier-moe-systems' },
        { id: 'attention-compressed-flao', label: 'Compressed attention', tip: { short: 'MLA, GQA, and sliding windows shrink KV footprint.', intuition: 'Trade stored bytes for query-time projection or locality.', trap: 'Compression quality loss shows at long context.' }, lessonId: 'multi-head-latent-attention' },
        { id: 'long-context-flao', label: 'Long-context stack', tip: { short: 'RoPE scaling, yarn, retrieval, and hybrid memory extend T.', intuition: 'Prefill cost grows with T; decode reads full cache.', trap: 'Claimed context window ≠ usable quality at max T.' }, lessonId: 'long-context-frontier-models' },
        { id: 'ssm-hybrid-flao', label: 'SSM / recurrent hybrid', tip: { short: 'State-space layers offer linear-time recurrence mixed with attention.', intuition: 'Some layers skip full quadratic attention.', trap: 'Hybrids still need careful eval on recall-heavy tasks.' } },
        { id: 'omni-modal-flao', label: 'Omni multimodal', tip: { short: 'Shared trunk encodes text, image, audio, and video tokens.', intuition: 'Modality-specific encoders feed a unified sequence.', trap: 'Token budget explodes with high-res vision or long audio.' }, lessonId: 'omni-multimodal-architectures' },
      ]},
      { id: 'intuitions', label: 'Intuitions', type: 'intuition', children: [
        { id: 'follow-one-token-flao', label: 'Follow one token', tip: { short: 'Trace active experts, cache bytes, and FLOPs for a single decode step.', intuition: 'Micro view clarifies macro architecture claims.', trap: 'Averaging over tokens hides routing hotspots.' } },
        { id: 'params-vs-active-flao', label: 'Total vs active params', tip: { short: 'MoE advertises total capacity; serving bills active compute.', intuition: 'VRAM holds all experts; FLOPs use only routed ones.', trap: 'Equating 400B MoE to 400B dense for latency.' } },
        { id: 'prefill-vs-decode-flao', label: 'Prefill vs decode', tip: { short: 'Prefill is compute-heavy; decode is often bandwidth-heavy.', intuition: 'Architecture knobs hit different phases differently.', trap: 'TTFT improvements do not fix TPOT if KV dominates.' } },
        { id: 'pareto-frontier-flao', label: 'Pareto frontier', tip: { short: 'No single architecture wins on cost, quality, context, and modalities.', intuition: 'Pick the family matching product bottleneck.', trap: 'Chasing all axes at once yields fragile stacks.' } },
        { id: 'modality-token-budget-flao', label: 'Modality token budget', tip: { short: 'Images and video become hundreds or thousands of tokens.', intuition: 'Multimodal context consumes window and cache fast.', trap: 'Text-only evals mis-rank omni models.' } },
      ]},
      { id: 'formula-code', label: 'Formula / Code', type: 'formula', children: [
        { id: 'kv-bytes-flao', label: 'KV cache bytes', tip: { short: 'KV ≈ 2 × layers × T × heads_kv × d_h × bytes.', intuition: 'Linear in context and layers.', formula: 'KV_{bytes} \\approx 2 L T H_{kv} d_h \\, bytes', trap: 'MLA latent cache uses different width than MHA formula.' } },
        { id: 'moe-active-flao', label: 'MoE active FLOPs', tip: { short: 'Active FFN ≈ top_k × expert_size per layer.', intuition: 'Router overhead is usually small vs experts.', formula: 'FLOPs_{MoE} \\approx k \\cdot FLOPs_{expert}', trap: 'Expert parallelism adds communication latency.' } },
        { id: 'attention-quadratic-flao', label: 'Attention cost', tip: { short: 'Prefill attention ~ O(T² d) per layer.', intuition: 'Quadratic in sequence for full attention.', trap: 'FlashAttention changes constants, not asymptotic prefill.' }, lessonId: 'flash-attention' },
        { id: 'compare-table-flao', label: 'Compare dimensions', tip: { short: 'Score families on active params, KV, context, gen order, modalities.', intuition: 'Question axes make tradeoffs explicit.', trap: 'Single-number leaderboard hides axis tradeoffs.' } },
        { id: 'precision-bytes-flao', label: 'Precision bytes', tip: { short: 'FP16=2, FP8=1, INT4=0.5 bytes per weight.', intuition: 'Quantization shifts memory but can hurt quality.', trap: 'Eval must use production precision.' } },
      ]},
      { id: 'traps', label: 'Common traps', type: 'trap', children: [
        { id: 'param-count-trap-flao', label: 'Parameter count hype', tip: { short: 'Total parameters overstate inference cost for MoE.', intuition: 'Active compute and memory traffic matter for SLA.', trap: 'Marketing compares total MoE to active dense unfairly.' } },
        { id: 'context-window-trap-flao', label: 'Context window on paper', tip: { short: 'Max T in config ≠ usable quality at max T.', intuition: 'Needle-in-haystack and long-doc evals at claimed length.', trap: 'RoPE extrapolation degrades without finetune.' } },
        { id: 'dense-moe-confusion-flao', label: 'Dense vs MoE confusion', tip: { short: 'MoE routing imbalance can mimic dense FLOPs with worse quality.', intuition: 'Load balancing losses exist for a reason.', trap: 'Assuming perfect expert utilization.' }, lessonId: 'frontier-moe-systems' },
        { id: 'multimodal-text-only-trap-flao', label: 'Text-only eval on omni', tip: { short: 'Omni models need vision/audio stress tests.', intuition: 'Joint training shifts text behavior too.', trap: 'MMLU-only reviews for multimodal products.' } },
        { id: 'ignore-serving-trap-flao', label: 'Ignore serving stack', tip: { short: 'Architecture wins vanish without batching and KV paging.', intuition: 'System co-design completes the story.', trap: 'Paper throughput without mention of hardware.' } },
      ]},
      { id: 'used-later', label: 'Used later', type: 'application', children: [
        { id: 'moe-lesson-used-flao', label: 'Frontier MoE systems', tip: { short: 'Deep dive on routing, load balance, and expert parallelism.', intuition: 'MoE is the largest active-compute lever.', trap: 'Skipping MoE chapter when deploying sparse models.' }, lessonId: 'frontier-moe-systems' },
        { id: 'long-context-used-flao', label: 'Long context frontier', tip: { short: 'Scaling laws and retrieval for massive contexts.', intuition: 'Extends architecture map with operational tactics.', trap: 'Bigger window without retrieval still loses needles.' }, lessonId: 'long-context-frontier-models' },
        { id: 'diffusion-lm-used-flao', label: 'Diffusion language models', tip: { short: 'Non-autoregressive generation order changes eval and serving.', intuition: 'Architecture map includes non-AR families.', trap: 'Applying AR KV intuition to diffusion LMs.' }, lessonId: 'diffusion-language-models' },
        { id: 'eval-safety-used-flao', label: 'Frontier evaluation & safety', tip: { short: 'Each architecture family needs tailored eval layers.', intuition: 'Agents on MoE omni stacks stress different failures.', trap: 'One eval harness for all families.' }, lessonId: 'frontier-evaluation-safety' },
        { id: 'mla-used-flao', label: 'Multi-head latent attention', tip: { short: 'MLA compresses KV for decode bandwidth wins.', intuition: 'Attention-compressed family exemplar.', trap: 'Ignoring projection FLOPs when claiming MLA wins.' }, lessonId: 'multi-head-latent-attention' },
      ]},
    ],
  },
  {
    id: 'frontier-moe-systems',
    label: 'Frontier MoE Systems',
    center: {
      short: 'Sparse mixture-of-experts (MoE) routes each token to top-k feed-forward experts so total capacity grows while active compute per token stays near a dense subset.',
      intuition: 'Think parameter budget vs active budget: many expert FFNs exist in memory, but only a few run for each token.',
      formula: 'y = \\sum_{i \\in TopK} g_i(x) \\cdot Expert_i(x)',
      why: 'MoE is a frontier lever for scaling model capacity without linearly scaling inference FLOPs—when routing, load balance, and serving are engineered correctly.',
      trap: 'Total expert parameters still consume VRAM; imbalanced routing can waste capacity or overload hot experts.',
    },
    branches: [
      { id: 'prerequisites', label: 'Prerequisites', type: 'prerequisite', children: [
        { id: 'transformer-moe', label: 'Transformer FFN block', tip: { short: 'Dense models use one large FFN per layer per token.', intuition: 'MoE replaces that FFN with a routed expert pool.', trap: 'Attention stack is usually still dense and shared.' }, lessonId: 'transformer' },
        { id: 'softmax-moe', label: 'Softmax gating', tip: { short: 'Router logits become expert weights over candidates.', intuition: 'Top-k picks discrete experts; weights scale contributions.', trap: 'Uniform routing wastes specialization.' }, lessonId: 'softmax' },
        { id: 'neural-network-moe', label: 'Neural network', tip: { short: 'Each expert is typically a two-layer MLP.', intuition: 'Experts differ by weights, not by architecture class.', trap: 'Expert count alone does not guarantee diversity.' }, lessonId: 'neural-network' },
        { id: 'efficient-serving-moe', label: 'Efficient LLM serving', tip: { short: 'Expert parallelism and all-to-all comm dominate MoE serving.', intuition: 'MoE shifts bottleneck from FLOPs to memory and network.', trap: 'Dense serving playbooks ignore expert dispatch latency.' }, lessonId: 'efficient-llm-serving' },
        { id: 'frontier-arch-moe', label: 'Frontier architecture map', tip: { short: 'MoE is one family in the frontier architecture landscape.', intuition: 'Compare active params and KV separately from MoE.', trap: 'Treating MoE as free capacity without routing cost.' }, lessonId: 'frontier-llm-architecture-overview' },
        { id: 'load-balancing-concept', label: 'Load balancing concept', tip: { short: 'Tokens should spread across experts, not collapse on one.', intuition: 'Auxiliary losses encourage uniform expert utilization.', trap: 'Perfect balance is impossible with skewed data domains.' } },
      ]},
      { id: 'mechanism', label: 'Core mechanism', type: 'mechanism', children: [
        { id: 'router-gate-moe', label: 'Router / gating', tip: { short: 'Linear gate maps hidden state to expert scores.', intuition: 'Each token picks its own expert subset.', trap: 'Router overfitting routes memorized tokens to one expert.' } },
        { id: 'top-k-routing-moe', label: 'Top-k routing', tip: { short: 'Activate k experts with highest gate scores per token.', intuition: 'k=1 is Switch Transformer style; k=2+ blends experts.', trap: 'Large k approaches dense compute without dense quality.' } },
        { id: 'shared-experts-moe', label: 'Shared experts', tip: { short: 'Always-on FFN plus routed experts on some stacks.', intuition: 'Shared path captures common features; routed path specializes.', trap: 'Shared width must be tuned vs routed pool.' } },
        { id: 'expert-parallelism-moe', label: 'Expert parallelism', tip: { short: 'Experts shard across devices; tokens all-to-all to owners.', intuition: 'Communication cost joins FLOPs in SLA math.', trap: 'Small batches hide catastrophic dispatch overhead.' } },
        { id: 'aux-loss-moe', label: 'Auxiliary load-balancing loss', tip: { short: 'Penalty when gate mass collapses on few experts.', intuition: 'Keeps training utilization high across experts.', trap: 'Too-strong aux loss hurts quality by forcing bad routes.' } },
        { id: 'capacity-factor-moe', label: 'Capacity factor', tip: { short: 'Limits tokens per expert per batch to avoid overload.', intuition: 'Dropped tokens when expert buffers overflow.', trap: 'Low capacity drops tokens; high capacity wastes compute.' } },
      ]},
      { id: 'intuitions', label: 'Intuitions', type: 'intuition', children: [
        { id: 'active-vs-total-moe', label: 'Active vs total parameters', tip: { short: 'VRAM stores all experts; FLOPs use only routed experts.', intuition: '400B total with 8B active is not 400B dense latency.', trap: 'Marketing total params as if all are active each token.' } },
        { id: 'specialization-moe', label: 'Expert specialization', tip: { short: 'Experts often align with domains like code, math, or languages.', intuition: 'Routing learns soft task clusters.', trap: 'Specialization is emergent, not guaranteed by architecture.' } },
        { id: 'routing-noise-moe', label: 'Routing noise', tip: { short: 'Training noise encourages exploration of expert assignments.', intuition: 'Prevents premature router collapse early in training.', trap: 'Too much noise prevents stable specialization.' } },
        { id: 'distillation-moe', label: 'MoE distillation', tip: { short: 'Distill routed teacher into dense or smaller MoE student.', intuition: 'Transfers routing signal and expert outputs.', trap: 'Student router may not mimic teacher load patterns.' } },
        { id: 'failure-debugger-moe', label: 'Failure debugger mindset', tip: { short: 'Watch expert histograms, dropped tokens, and domain skew.', intuition: 'MoE failures are systemic, not single-weight bugs.', trap: 'Debugging MoE like a dense model misses routing stats.' } },
      ]},
      { id: 'formula-code', label: 'Formula / Code', type: 'formula', children: [
        { id: 'gate-formula-moe', label: 'Gating formula', tip: { short: 'g = softmax(W_g x); take top-k of g.', intuition: 'Router is a small linear layer plus softmax.', formula: 'g = \\mathrm{softmax}(W_g x)', trap: 'Dtype of gate logits affects numerical stability.' } },
        { id: 'moe-output-moe', label: 'MoE layer output', tip: { short: 'Sum weighted expert outputs for selected indices.', intuition: 'Same shape as dense FFN output.', formula: 'y = \\sum_{i \\in TopK} g_i \\cdot E_i(x)', trap: 'Expert output dtype must match residual stream.' } },
        { id: 'aux-loss-formula-moe', label: 'Load-balancing aux loss', tip: { short: 'Encourage uniform fraction of tokens per expert.', intuition: 'Switch-style aux uses importance and load fractions.', trap: 'Aux on wrong tensor breaks gradient to router.' } },
        { id: 'active-param-estimate', label: 'Active param estimate', tip: { short: 'Active ≈ attention params + k × expert_size × layers.', intuition: 'Quick napkin math for serving planning.', trap: 'Shared experts and embeddings add fixed overhead.' } },
        { id: 'dispatch-sketch-moe', label: 'Dispatch sketch', tip: { short: 'Sort tokens by expert id, run batched expert MLPs, combine.', intuition: 'Implementation is gather-scatter heavy.', code: 'idx = topk(gates)\nout = scatter_add(expert[idx](x))', trap: 'Uneven expert batch sizes hurt GPU utilization.' } },
      ]},
      { id: 'traps', label: 'Common traps', type: 'trap', children: [
        { id: 'vram-all-experts-moe', label: 'VRAM holds all experts', tip: { short: 'Inactive experts still occupy memory on device.', intuition: 'MoE saves FLOPs more than RAM unless offloaded.', trap: 'Assuming MoE fits because active params are small.' } },
        { id: 'router-collapse-moe', label: 'Router collapse', tip: { short: 'One expert receives most tokens; others starve.', intuition: 'Shows up as flat histogram with one spike.', trap: 'Ignoring aux loss until late training.' } },
        { id: 'token-dropping-moe', label: 'Token dropping', tip: { short: 'Overflow expert buffers drop tokens silently in some systems.', intuition: 'Quality cliffs when load spikes.', trap: 'Capacity factor tuned only on average load.' } },
        { id: 'comm-latency-moe', label: 'All-to-all latency', tip: { short: 'Cross-device expert dispatch adds milliseconds.', intuition: 'Small batch inference suffers most.', trap: 'Benchmarking MoE on single GPU only.' } },
        { id: 'dense-quality-assumption-moe', label: 'Dense quality assumption', tip: { short: 'Same active params as dense does not guarantee same quality.', intuition: 'Routing and training stability matter.', trap: 'Undertrained routers waste expert capacity.' } },
      ]},
      { id: 'used-later', label: 'Used later', type: 'application', children: [
        { id: 'frontier-arch-used-moe', label: 'Frontier architecture overview', tip: { short: 'MoE sits beside compressed attention and long context.', intuition: 'Pick MoE when capacity is the bottleneck.', trap: 'Stacking every frontier knob at once.' }, lessonId: 'frontier-llm-architecture-overview' },
        { id: 'efficient-serving-used-moe', label: 'Efficient LLM serving', tip: { short: 'Schedulers must account for expert placement and dispatch.', intuition: 'MoE serving is a systems problem.', trap: 'Dense batching heuristics on sparse models.' }, lessonId: 'efficient-llm-serving' },
        { id: 'eval-safety-used-moe', label: 'Frontier evaluation', tip: { short: 'Eval MoE at production routing temperature and load.', intuition: 'Router stochasticity affects reproducibility.', trap: 'Eval only shared expert ablation.' }, lessonId: 'frontier-evaluation-safety' },
        { id: 'long-context-used-moe', label: 'Long context frontier', tip: { short: 'Long contexts multiply MoE dispatch volume per step.', intuition: 'Prefill routes many tokens simultaneously.', trap: 'Ignoring prefill expert hotspots.' }, lessonId: 'long-context-frontier-models' },
        { id: 'compression-track-used-moe', label: 'Efficient inference track', tip: { short: 'Quantization and offloading interact with expert sharding.', intuition: 'INT4 experts still need fast dispatch paths.', trap: 'Quantizing router differently from experts.' }, lessonId: 'efficient-inference-compression-track' },
      ]},
    ],
  },
  {
    id: 'glove',
    label: 'GloVe',
    center: {
      short: 'GloVe (Global Vectors) learns word embeddings by factorizing a weighted co-occurrence matrix so dot products approximate log co-occurrence ratios.',
      intuition: 'Corpus-wide word-pair counts become a global map; vectors are shaped so similar words share co-occurrence statistics.',
      formula: 'w_i^T w_j + b_i + b_j \\approx \\log X_{ij}',
      why: 'GloVe bridges count-based and prediction-based embeddings, remains a classic baseline, and clarifies how global statistics differ from local Word2Vec contexts.',
      trap: 'Raw co-occurrence counts need weighting—common words dominate unless you down-weight frequent pairs.',
    },
    branches: [
      { id: 'prerequisites', label: 'Prerequisites', type: 'prerequisite', children: [
        { id: 'cooccurrence-glove', label: 'Co-occurrence', tip: { short: 'Count how often words appear near each other in a corpus.', intuition: 'GloVe builds a word-word matrix from these counts.', trap: 'Window size changes which semantics are captured.' } },
        { id: 'dot-product-glove', label: 'Dot product', tip: { short: 'Similar vectors yield large dot products.', intuition: 'GloVe trains dot products to match log counts.', trap: 'Unnormalized dots confound vector length with similarity.' }, lessonId: 'dot-product' },
        { id: 'log-linear-glove', label: 'Log-linear model', tip: { short: 'Linear structure in log domain for count ratios.', intuition: 'Ratios encode relative co-occurrence strength.', trap: 'Zeros in counts need floor or weighting tricks.' } },
        { id: 'gradient-descent-glove', label: 'Gradient descent', tip: { short: 'Minimize weighted squared error on non-zero matrix cells.', intuition: 'Only observed pairs contribute to most updates.', trap: 'Learning rate too high diverges embedding norms.' }, lessonId: 'gradient-descent' },
        { id: 'word2vec-compare-glove', label: 'Word2Vec preview', tip: { short: 'Word2Vec learns from local prediction contexts.', intuition: 'Contrast local skip-gram with GloVe global matrix.', trap: 'Treating them as identical objectives.' }, lessonId: 'word2vec' },
        { id: 'vector-space-glove', label: 'Vector space', tip: { short: 'Each word maps to a dense d-dimensional vector.', intuition: 'Geometry encodes semantic and syntactic relations.', trap: 'Low d loses rare-word nuance; high d overfits small corpora.' } },
      ]},
      { id: 'mechanism', label: 'Core mechanism', type: 'mechanism', children: [
        { id: 'build-x-glove', label: 'Build co-occurrence matrix X', tip: { short: 'Slide context window; increment X_ij for word pairs.', intuition: 'Global corpus statistics aggregated once.', trap: 'Huge vocab makes X memory-heavy—use hashing or pruning.' } },
        { id: 'weight-function-glove', label: 'Weighting function f(X_ij)', tip: { short: 'Cap influence of very frequent pairs; zero or down-weight rare zeros.', intuition: 'Prevents stopwords from dominating the objective.', trap: 'No weighting lets the, of, and is swamp training.' } },
        { id: 'word-vectors-glove', label: 'Word vectors w_i', tip: { short: 'Learned embeddings for each vocabulary term.', intuition: 'Dot w_i^T w_j models affinity between words.', trap: 'Asymmetric roles need separate w and w_tilde in full formulation.' } },
        { id: 'bias-terms-glove', label: 'Bias terms b_i, b_j', tip: { short: 'Per-word offsets absorb frequency effects.', intuition: 'Lets vectors focus on relative similarity.', trap: 'Forgetting biases when reproducing from scratch.' } },
        { id: 'glove-loss-glove', label: 'GloVe loss', tip: { short: 'Weighted least squares on (w_i^T w_j + biases − log X_ij)².', intuition: 'Only train on non-zero or sampled cells for efficiency.', trap: 'Log(0) undefined—use log(X_ij) with X_ij>=1 floor.' } },
        { id: 'normalize-glove', label: 'Normalize embeddings', tip: { short: 'Optional L2 norm for cosine-style similarity at inference.', intuition: 'Comparisons often use normalized vectors.', trap: 'Training uses raw dots; inference norm changes geometry.' }, lessonId: 'cosine-similarity' },
      ]},
      { id: 'intuitions', label: 'Intuitions', type: 'intuition', children: [
        { id: 'global-vs-local-glove', label: 'Global vs local', tip: { short: 'GloVe uses whole-corpus counts; Word2Vec uses streaming contexts.', intuition: 'Global methods batch statistics; local methods scan sentences.', trap: 'Claiming GloVe ignores context—it uses fixed windows.' } },
        { id: 'ratio-intuition-glove', label: 'Ratio intuition', tip: { short: 'Ratios P(k|ice)/P(k|steam) highlight semantic dimensions.', intuition: 'Vector differences encode meaning directions.', trap: 'Single dimensions rarely align with one human concept.' } },
        { id: 'rare-word-glove', label: 'Rare word handling', tip: { short: 'Rare pairs get lower weights or are dropped.', intuition: 'Stabilizes training on sparse counts.', trap: 'Over-pruning loses niche terminology.' } },
        { id: 'analogy-glove', label: 'Linear analogies', tip: { short: 'king − man + woman ≈ queen emerges in good embeddings.', intuition: 'Offset vectors approximate relations.', trap: 'Analogies fail for ambiguous or polysemous words.' } },
        { id: 'subword-limit-glove', label: 'No subword units', tip: { short: 'Classic GloVe is word-level; OOV words have no vector.', intuition: 'FastText adds subwords for OOV robustness.', trap: 'Applying GloVe to morphologically rich langs without extension.' }, lessonId: 'fasttext' },
      ]},
      { id: 'formula-code', label: 'Formula / Code', type: 'formula', children: [
        { id: 'glove-objective-formula', label: 'GloVe objective', tip: { short: 'Minimize weighted squared log-co-occurrence error.', intuition: 'Core GloVe identity from the paper.', formula: 'J = \\sum_{i,j} f(X_{ij})(w_i^T w_j + b_i + b_j - \\log X_{ij})^2', trap: 'Symmetric vs asymmetric word vectors differ in full model.' } },
        { id: 'weight-cap-formula', label: 'Weight cap', tip: { short: 'f(x) = min((x/x_max)^alpha, 1) common choice.', intuition: 'Down-weights huge counts smoothly.', trap: 'Wrong x_max changes effective corpus influence.' } },
        { id: 'gensim-glove', label: 'Gensim / library train', tip: { short: 'Build co-occurrence, run GloVe trainer or use pretrained.', intuition: 'Production often uses pretrained checkpoints.', code: 'from gensim.models import KeyedVectors\nkv = KeyedVectors.load_word2vec_format("glove.txt")', trap: 'Header lines and dimension mismatch break loading.' } },
        { id: 'similarity-query-glove', label: 'Similarity query', tip: { short: 'most_similar uses cosine on normalized vectors.', intuition: 'Nearest neighbors validate embedding quality.', code: 'kv.most_similar("king", topn=10)', trap: 'Cosine on unnormalized vectors skews results.' } },
        { id: 'window-hyper-glove', label: 'Window hyperparameter', tip: { short: 'Context window size affects syntactic vs semantic capture.', intuition: 'Small windows: syntax; larger: topics.', trap: 'Changing window without rebuilding X invalidates training.' } },
      ]},
      { id: 'traps', label: 'Common traps', type: 'trap', children: [
        { id: 'raw-counts-trap-glove', label: 'Raw counts dominate', tip: { short: 'Function words co-occur everywhere without weighting.', intuition: 'Weighting is not optional on large corpora.', trap: 'Skipping f(X_ij) reproduces stopword geometry.' } },
        { id: 'word2vec-same-trap-glove', label: 'Same as Word2Vec', tip: { short: 'Different objective and data pipeline.', intuition: 'Compare on same corpus before judging superiority.', trap: 'Mixing hyperparameters across methods unfairly.' } },
        { id: 'oov-trap-glove', label: 'OOV words', tip: { short: 'Unknown tokens have no embedding at inference.', intuition: 'Use subword models or UNK handling.', trap: 'Deploying word-level GloVe on open-vocab social text.' }, lessonId: 'fasttext' },
        { id: 'static-embed-trap-glove', label: 'Static embeddings', tip: { short: 'One vector per word regardless of context sense.', intuition: 'Contextual models like BERT supersede for polysemy.', trap: 'Expecting bank disambiguation from static GloVe alone.' } },
        { id: 'small-corpus-trap-glove', label: 'Tiny corpus', tip: { short: 'Co-occurrence matrix too sparse for stable vectors.', intuition: 'Need billions of tokens for quality like published GloVe.', trap: 'Training GloVe on a few MB and expecting Wikipedia quality.' } },
      ]},
      { id: 'used-later', label: 'Used later', type: 'application', children: [
        { id: 'word2vec-used-glove', label: 'Word2Vec', tip: { short: 'Compare local prediction embeddings on same tasks.', intuition: 'Classic NLP embedding duo.', trap: 'Declaring winner without task-specific eval.' }, lessonId: 'word2vec' },
        { id: 'fasttext-used-glove', label: 'FastText', tip: { short: 'Subword embeddings fix OOV for morphologically rich text.', intuition: 'Evolution of static embedding line.', trap: 'Ignoring subwords in agglutinative languages.' }, lessonId: 'fasttext' },
        { id: 'embeddings-lesson-glove', label: 'Embeddings lesson', tip: { short: 'Modern pipelines start from embedding concepts GloVe popularized.', intuition: 'Foundation for transfer and fine-tuning stories.', trap: 'Skipping static era loses intuition for geometry.' }, lessonId: 'embeddings' },
        { id: 'cosine-used-glove', label: 'Cosine similarity', tip: { short: 'Evaluate neighbors and analogies with cosine metric.', intuition: 'Standard retrieval metric for static vectors.', trap: 'Euclidean distance on unnormalized GloVe vectors.' }, lessonId: 'cosine-similarity' },
        { id: 'tokenization-used-glove', label: 'Tokenization', tip: { short: 'Tokenizer choices change vocabulary and co-occurrence stats.', intuition: 'Preprocessing is part of embedding quality.', trap: 'Mixing tokenizers between train and downstream task.' }, lessonId: 'tokenization' },
      ]},
    ],
  },
  {
    id: 'gradient-problems',
    label: 'Gradient Problems',
    center: {
      short: 'Vanishing and exploding gradients occur when repeated multiplication of local derivatives through deep layers shrinks or blows up the backprop signal reaching early parameters.',
      intuition: 'Trace how repeated derivatives scale gradients layer by layer—depth amplifies init, activation slope, and weight scale choices.',
      formula: '\\frac{\\partial L}{\\partial W_1} = \\frac{\\partial L}{\\partial a_L}\\prod_{l=1}^{L-1}\\frac{\\partial a_{l+1}}{\\partial a_l}',
      why: 'Gradient health determines whether deep networks train at all—diagnosing vanishing vs exploding guides init, activations, normalization, residuals, and clipping.',
      trap: 'More layers can make credit assignment fragile; blaming only learning rate misses activation and initialization effects.',
    },
    branches: [
      { id: 'prerequisites', label: 'Prerequisites', type: 'prerequisite', children: [
        { id: 'backprop-gp', label: 'Backpropagation', tip: { short: 'Gradients flow backward via chain rule products.', intuition: 'Each layer multiplies one more local Jacobian factor.', trap: 'Broken local rule corrupts entire chain.' }, lessonId: 'computation-graph-backprop' },
        { id: 'chain-rule-gp', label: 'Chain rule', tip: { short: 'Deep gradients are products of many partial derivatives.', intuition: 'Products below 1 shrink; above 1 grow exponentially in depth.', trap: 'Assuming gradients stay order-one through depth.' } },
        { id: 'relu-gp', label: 'ReLU', tip: { short: 'Derivative 1 when active, 0 when inactive.', intuition: 'Dead ReLUs block gradient entirely.', trap: 'All-negative preactivations freeze units.' }, lessonId: 'relu' },
        { id: 'initialization-gp', label: 'Initialization', tip: { short: 'Starting weight scale sets initial gradient magnitudes.', intuition: 'Xavier/He init targets stable variance through layers.', trap: 'Default PyTorch init changed across versions—verify.' }, lessonId: 'initialization' },
        { id: 'gradient-descent-gp', label: 'Gradient descent', tip: { short: 'Updates use learning rate times gradient.', intuition: 'Tiny gradients mean frozen early layers even with correct LR.', trap: 'Cranking LR to fix vanishing causes explosion elsewhere.' }, lessonId: 'gradient-descent' },
        { id: 'sigmoid-tanh-gp', label: 'Sigmoid / tanh', tip: { short: 'Saturate with derivatives near zero at large |x|.', intuition: 'Classic source of vanishing in RNNs and old MLPs.', trap: 'Using sigmoid stacks without skip connections in deep nets.' } },
      ]},
      { id: 'mechanism', label: 'Core mechanism', type: 'mechanism', children: [
        { id: 'local-derivative-gp', label: 'Local derivative', tip: { short: 'Each layer contributes ∂a_{l+1}/∂a_l to the product.', intuition: 'Activation slope and weight norm dominate per layer.', trap: 'BatchNorm backward adds its own scaling factors.' } },
        { id: 'vanishing-gp', label: 'Vanishing gradients', tip: { short: 'Product of many factors << 1 reaches early layers.', intuition: 'Early weights get negligible updates.', trap: 'Loss still decreases while early layers stay random.' } },
        { id: 'exploding-gp', label: 'Exploding gradients', tip: { short: 'Product of many factors >> 1 yields NaN weights.', intuition: 'Often visible as loss spikes and Inf gradients.', trap: 'Only clipping symptoms without fixing init/activations.' } },
        { id: 'dead-relu-gp', label: 'Dead ReLU units', tip: { short: 'Units with always-negative input never activate or learn.', intuition: 'Zero gradient through inactive ReLU path.', trap: 'High LR can kill many units at once.' }, lessonId: 'leaky-relu' },
        { id: 'gradient-clipping-gp', label: 'Gradient clipping', tip: { short: 'Cap global norm or per-value gradients before optimizer step.', intuition: 'Stabilizes RNNs and deep stacks against spikes.', trap: 'Clipping every step hides persistent explosion cause.' } },
        { id: 'residual-mix-gp', label: 'Residual connections', tip: { short: 'Skip paths add identity routes that preserve gradient highways.', intuition: 'Residual stream carries signal around deep products.', trap: 'Bad scaling on residual branches still vanishes.' }, lessonId: 'residual-stream' },
      ]},
      { id: 'intuitions', label: 'Intuitions', type: 'intuition', children: [
        { id: 'signal-layer-gp', label: 'Which layer gets signal', tip: { short: 'Plot gradient norm vs layer index to see vanish or explode.', intuition: 'Diagnostic heatmaps guide fixes.', trap: 'Inspecting only final-layer grad misses depth pattern.' } },
        { id: 'depth-multiplier-gp', label: 'Depth as multiplier', tip: { short: 'Even 0.9 per layer becomes tiny after 50 layers.', intuition: 'Depth turns small mistakes into training failure.', trap: 'Copying shallow-net hyperparams to deep nets.' } },
        { id: 'activation-choice-gp', label: 'Activation choice', tip: { short: 'ReLU/LeakyReLU/GELU change typical derivative magnitudes.', intuition: 'Modern LMs use GELU/SwiGLU with careful init.', trap: 'Swapping activation without re-tuning init.' } },
        { id: 'normalization-help-gp', label: 'Normalization help', tip: { short: 'LayerNorm/BatchNorm stabilize activations entering next layer.', intuition: 'Keeps local derivatives in healthier range.', trap: 'Norm alone cannot fix dead ReLU deserts.' }, lessonId: 'layer-normalization' },
        { id: 'rnn-vs-transformer-gp', label: 'RNN vs transformer', tip: { short: 'RNNs unroll time; transformers unroll depth—both chain Jacobians.', intuition: 'Same math, different axis of repetition.', trap: 'Assuming transformers cannot vanish—they can in depth.' } },
      ]},
      { id: 'formula-code', label: 'Formula / Code', type: 'formula', children: [
        { id: 'product-formula-gp', label: 'Gradient product', tip: { short: 'Early-layer grad equals loss grad times Jacobian product.', intuition: 'Explains exponential scaling with depth.', formula: '\\frac{\\partial L}{\\partial W_1}=\\frac{\\partial L}{\\partial a_L}\\prod_l J_l', trap: 'Ill-conditioned Jacobians need numerical checks.' } },
        { id: 'clip-norm-code-gp', label: 'Global norm clip', tip: { short: 'Scale gradient if ||g|| exceeds threshold.', intuition: 'Standard RNN training stabilizer.', code: 'torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)', trap: 'max_norm too small stalls learning entirely.' } },
        { id: 'he-init-formula-gp', label: 'He initialization', tip: { short: 'Var(W) ≈ 2/fan_in for ReLU stacks.', intuition: 'Targets unit variance activations forward and backward.', formula: 'W \\sim \\mathcal{N}(0, \\sqrt{2/n_{in}})', trap: 'He init wrong for tanh or linear activations.' }, lessonId: 'initialization' },
        { id: 'leaky-slope-gp', label: 'Leaky ReLU slope', tip: { short: 'Small negative slope keeps gradient alive when inactive.', intuition: 'Prevents permanent zero derivative deserts.', trap: 'Slope too large reintroduces saturation.' }, lessonId: 'leaky-relu' },
        { id: 'grad-norm-monitor-gp', label: 'Grad norm monitor', tip: { short: 'Log per-layer gradient norms during training.', intuition: 'Early warning before NaN.', code: 'for n,p in model.named_parameters():\n    if p.grad is not None: log(n, p.grad.norm())', trap: 'Only logging total norm hides layer-specific vanish.' } },
      ]},
      { id: 'traps', label: 'Common traps', type: 'trap', children: [
        { id: 'lr-only-trap-gp', label: 'LR-only diagnosis', tip: { short: 'Vanishing needs init/activation/residual fixes, not just smaller LR.', intuition: 'LR cannot multiply zero dead-ReLU gradients.', trap: 'Endless LR grid search on untrainable stack.' } },
        { id: 'clip-masks-trap-gp', label: 'Clipping masks root cause', tip: { short: 'Frequent clipping means underlying instability persists.', intuition: 'Clip is a guardrail, not a cure.', trap: 'Training succeeds only because clip fires every step.' } },
        { id: 'dead-relu-ignore-gp', label: 'Ignore dead ReLUs', tip: { short: 'Many zero-gradient units silently reduce capacity.', intuition: 'Count fraction of active units per layer.', trap: 'Accuracy plateau with half the network dead.' } },
        { id: 'shallow-metric-trap-gp', label: 'Shallow-network metrics', tip: { short: 'Loss can improve while early layers remain untrained.', intuition: 'Later layers compensate until they cannot.', trap: 'Stopping when val loss flat without grad diagnostics.' } },
        { id: 'mixed-precision-trap-gp', label: 'Mixed precision underflow', tip: { short: 'FP16 grads can underflow to zero in deep models.', intuition: 'Loss scaling preserves small gradient bits.', trap: 'Vanishing that appears only in AMP training.' } },
      ]},
      { id: 'used-later', label: 'Used later', type: 'application', children: [
        { id: 'initialization-used-gp', label: 'Initialization', tip: { short: 'Proper init is first-line defense against gradient pathologies.', intuition: 'Pair init scheme with activation family.', trap: 'Transferring init from CNN to deep transformer without check.' }, lessonId: 'initialization' },
        { id: 'optimizers-used-gp', label: 'Optimizers', tip: { short: 'Adam adapts per-parameter scale but cannot fix zero gradients.', intuition: 'Optimizer choice second after signal reaches weights.', trap: 'Switching optimizer to fix dead units.' }, lessonId: 'optimizers' },
        { id: 'training-loop-used-gp', label: 'Training loop dynamics', tip: { short: 'Healthy loops show stable grad norms and loss descent together.', intuition: 'Monitor both curves, not loss alone.', trap: 'Loss cliff often preceded by grad norm spike.' }, lessonId: 'training-loop-dynamics' },
        { id: 'lstm-used-gp', label: 'LSTM', tip: { short: 'Gates were designed to mitigate RNN vanishing/exploding.', intuition: 'Historical context for gradient pathologies.', trap: 'Assuming LSTM eliminates all long-range gradient issues.' }, lessonId: 'lstm' },
        { id: 'backprop-used-gp', label: 'Computation graph', tip: { short: 'Backprop implements the chain products causing pathologies.', intuition: 'Debug graph ops when grads look wrong.', trap: 'Detach or stop-gradient mistakes mimic vanishing.' }, lessonId: 'computation-graph-backprop' },
      ]},
    ],
  },
  {
    id: 'joint-attention',
    label: 'Joint Attention',
    center: {
      short: 'Joint attention lets tokens from different modalities (text, image, audio) attend to each other in a shared sequence so cross-modal alignment emerges from standard attention scores.',
      intuition: 'Concatenate or fuse modality tokens, then let Q/K/V interact across modalities—which visual token should read which text token?',
      formula: '\\mathrm{Attention}(Q,K,V)=\\mathrm{softmax}\\left(\\frac{QK^T}{\\sqrt{d}}\\right)V',
      why: 'Joint attention powers multimodal LLMs, vision-language models, and omni architectures where grounding requires bidirectional cross-modal information flow.',
      trap: 'Joint attention can mix irrelevant signals too—modality-specific encoders and masks still matter.',
    },
    branches: [
      { id: 'prerequisites', label: 'Prerequisites', type: 'prerequisite', children: [
        { id: 'self-attention-ja', label: 'Self-attention', tip: { short: 'Queries attend over keys and mix values in one sequence.', intuition: 'Joint attention is self-attention on a concatenated multimodal sequence.', trap: 'Separate unimodal attentions miss cross-modal pairs.' }, lessonId: 'self-attention' },
        { id: 'attention-mechanism-ja', label: 'Attention mechanism', tip: { short: 'Q, K, V projections define what attends to what.', intuition: 'Shared or split projections define fusion style.', trap: 'Wrong projection sharing blocks cross-modal flow.' }, lessonId: 'attention-mechanism' },
        { id: 'embeddings-ja', label: 'Embeddings', tip: { short: 'Each modality maps raw inputs to token vectors.', intuition: 'Alignment starts in a common hidden dimension d.', trap: 'Mismatched d between encoders breaks concat.' }, lessonId: 'embeddings' },
        { id: 'positional-ja', label: 'Positional encoding', tip: { short: 'Positions distinguish order within and across modalities.', intuition: '2D or modality-specific pos encodings for images.', trap: 'Text-only RoPE on image patches misorders spatial info.' }, lessonId: 'positional-encoding' },
        { id: 'attention-masks-ja', label: 'Attention masks', tip: { short: 'Masks block illegal connections (causal, padding, modality).', intuition: 'Control which joint pairs are allowed.', trap: 'Over-masking prevents needed cross-attention.' }, lessonId: 'attention-masks' },
        { id: 'transformer-ja', label: 'Transformer', tip: { short: 'Stack joint attention layers with FFN blocks.', intuition: 'Depth builds hierarchical cross-modal features.', trap: 'Single shallow joint layer may underfit alignment.' }, lessonId: 'transformer' },
      ]},
      { id: 'mechanism', label: 'Core mechanism', type: 'mechanism', children: [
        { id: 'token-concat-ja', label: 'Token concatenation', tip: { short: 'Build sequence [text tokens | image tokens | ...].', intuition: 'One sequence enables full cross-attention matrix.', trap: 'Sequence length and memory explode with high-res vision.' } },
        { id: 'shared-projections-ja', label: 'Shared QKV projections', tip: { short: 'Same W_Q, W_K, W_V for all modalities in full joint attention.', intuition: 'Forces common representation space for scoring.', trap: 'Modality gap may need separate encoders before shared layers.' } },
        { id: 'cross-attention-vs-ja', label: 'Cross-attention variant', tip: { short: 'One modality queries another modality keys/values.', intuition: 'Asymmetric fusion: text queries image only.', trap: 'Confusing cross-attn with full bidirectional joint self-attn.' } },
        { id: 'bidirectional-fusion-ja', label: 'Bidirectional fusion', tip: { short: 'Text updates from vision and vision from text in same layer.', intuition: 'Full joint self-attention is inherently bidirectional among prefix.', trap: 'Causal LM masks limit which directions are legal at decode.' } },
        { id: 'modality-type-embed-ja', label: 'Modality type embeddings', tip: { short: 'Extra embedding tells model which modality a token belongs to.', intuition: 'Disambiguates otherwise similar position slots.', trap: 'Missing type tags blur text vs special tokens.' } },
        { id: 'compute-cost-ja', label: 'Compute cost', tip: { short: 'Attention over N tokens costs O(N²); vision adds many tokens.', intuition: 'Patch count drives multimodal memory.', trap: 'High-res images without pooling exceed context budget.' } },
      ]},
      { id: 'intuitions', label: 'Intuitions', type: 'intuition', children: [
        { id: 'grounding-ja', label: 'Visual grounding', tip: { short: 'Text token attends to object patches when answering about objects.', intuition: 'Attention maps visualize grounding (with caveats).', trap: 'Attention weights are not guaranteed causal explanations.' } },
        { id: 'early-vs-late-fusion-ja', label: 'Early vs late fusion', tip: { short: 'Joint early layers fuse raw tokens; late fusion merges high-level summaries.', intuition: 'Depth of fusion trades compute for alignment quality.', trap: 'Late fusion only may miss fine spatial grounding.' } },
        { id: 'modality-gap-ja', label: 'Modality gap', tip: { short: 'Image and text live in different statistical manifolds pre-alignment.', intuition: 'Contrastive pretraining (CLIP) reduces gap before joint LM.', trap: 'Random init joint training on small data fails.' }, lessonId: 'clip-encoder' },
        { id: 'causal-multimodal-ja', label: 'Causal multimodal decode', tip: { short: 'Autoregressive text gen attends to frozen image prefix.', intuition: 'Image acts as conditioning memory during decode.', trap: 'Updating image tokens mid-decode breaks cache assumptions.' } },
        { id: 'irrelevant-attend-ja', label: 'Irrelevant attention', tip: { short: 'Model may attend to wrong patches or spurious text tokens.', intuition: 'Needs data scale and objectives to sharpen alignment.', trap: 'Assuming joint attention always finds correct pairs.' } },
      ]},
      { id: 'formula-code', label: 'Formula / Code', type: 'formula', children: [
        { id: 'concat-formula-ja', label: 'Concatenated sequence', tip: { short: 'H = [H_text ; H_vision] then standard attention on H.', intuition: 'Single softmax over all token pairs (subject to mask).', formula: 'H = [H_t; H_v]', trap: 'Padding tokens need mask to exclude from softmax.' } },
        { id: 'joint-attn-formula-ja', label: 'Joint attention formula', tip: { short: 'Same scaled dot-product on stacked Q, K, V.', intuition: 'Cross blocks in QK^T are text-image scores.', formula: '\\mathrm{softmax}(QK^T/\\sqrt{d})V', trap: 'Scale sqrt(d) uses per-head dimension.' } },
        { id: 'cross-attn-code-ja', label: 'Cross-attention code', tip: { short: 'Q from modality A, K/V from modality B.', intuition: 'Common in encoder-decoder VL models.', code: 'cross = attention(q=text, k=image, v=image)', trap: 'Swapping Q/K sources changes information flow direction.' } },
        { id: 'modality-embed-code-ja', label: 'Modality embedding', tip: { short: 'Add learned vector per modality to token embeddings.', intuition: 'Tells layer which stream a token came from.', code: 'x = x + modality_embed[mod_id]', trap: 'Wrong mod_id tensor shape broadcasts incorrectly.' } },
        { id: 'mask-joint-code-ja', label: 'Joint mask pattern', tip: { short: 'Boolean mask forbids text→padding or future text tokens.', intuition: 'Combine causal and padding masks.', code: 'attn_mask = causal_mask & pad_mask', trap: 'Allowing text to attend wrong image pads adds noise.' } },
      ]},
      { id: 'traps', label: 'Common traps', type: 'trap', children: [
        { id: 'token-budget-trap-ja', label: 'Token budget blowup', tip: { short: 'Full-resolution vision tokens dominate memory.', intuition: 'Patch merge, pooling, or resamplers required.', trap: 'Dropping image resolution without telling the user.' } },
        { id: 'modality-gap-trap-ja', label: 'Skip alignment pretrain', tip: { short: 'Cold-start joint training on tiny data misaligns.', intuition: 'Use CLIP-style or caption pretraining first.', trap: 'Expecting GPT-style LM loss alone to align vision.' }, lessonId: 'clip-encoder' },
        { id: 'attn-heatmap-trap-ja', label: 'Attention heatmap faith', tip: { short: 'High attention weight ≠ causal importance.', intuition: 'Use probing or ablation for grounding claims.', trap: 'Publishing heatmaps as proof of understanding.' } },
        { id: 'causal-mask-trap-ja', label: 'Causal mask errors', tip: { short: 'Text must not attend to future text tokens during AR training.', intuition: 'Image prefix usually fully visible to all text positions.', trap: 'Applying causal mask across image blocks incorrectly.' } },
        { id: 'symmetry-trap-ja', label: 'Cross vs joint confusion', tip: { short: 'Cross-attn one-way differs from symmetric joint self-attn.', intuition: 'Pick fusion pattern to match architecture paper.', trap: 'Implementing cross-attn but evaluating as full joint.' } },
      ]},
      { id: 'used-later', label: 'Used later', type: 'application', children: [
        { id: 'multimodal-llm-ja', label: 'Multimodal LLM', tip: { short: 'Production VLMs stack joint or cross-attention layers.', intuition: 'Extends text-only LLM to images and more.', trap: 'Text-only eval on multimodal checkpoint.' }, lessonId: 'multimodal-llm' },
        { id: 'omni-ja', label: 'Omni multimodal architectures', tip: { short: 'Audio and video tokens join the joint sequence.', intuition: 'Same attention machinery, more modality encoders.', trap: 'Underestimating audio frame token counts.' }, lessonId: 'omni-multimodal-architectures' },
        { id: 'clip-used-ja', label: 'CLIP encoder', tip: { short: 'Contrastive pretraining supplies aligned image-text towers.', intuition: 'Often feeds vision tokens into joint LM.', trap: 'CLIP resolution limits vs downstream task needs.' }, lessonId: 'clip-encoder' },
        { id: 'frontier-arch-ja', label: 'Frontier architecture overview', tip: { short: 'Omni family uses joint attention in unified trunks.', intuition: 'Architecture map situates multimodal fusion.', trap: 'Ignoring modality-specific preprocessing in cost model.' }, lessonId: 'frontier-llm-architecture-overview' },
        { id: 'rag-ground-ja', label: 'RAG grounding', tip: { short: 'Retrieved text joins context for grounded answers.', intuition: 'Attention over retrieved chunks parallels multimodal fusion.', trap: 'Retrieval noise attended equally to user question.' }, lessonId: 'rag-reranking-grounding' },
      ]},
    ],
  },
  {
    id: 'knn-naive-bayes-svm',
    label: 'kNN, Naive Bayes, and SVM',
    center: {
      short: 'kNN, Naive Bayes, and SVM are classic classifiers: neighbor voting, probabilistic independence assumptions, and maximum-margin hyperplanes—each encodes a different inductive bias.',
      intuition: 'Compare neighbors, independence assumptions, and margins on the same points—which assumption fits this dataset?',
      formula: 'kNN: \\hat{y}=\\mathrm{majority}(k\\; neighbors)\\;|\\; NB: \\hat{y}=\\arg\\max_y P(y)\\prod_i P(x_i|y)\\;|\\; SVM: \\min \\|w\\|^2 \\; s.t.\\; y_i(w^Tx_i+b)\\geq 1',
      why: 'These baselines remain essential for teaching decision boundaries, probabilistic modeling, kernel methods, and when simple models beat deep nets on small tabular data.',
      trap: 'No classic model wins everywhere—kNN struggles in high dimensions, Naive Bayes breaks correlated features, SVM needs careful kernels and scaling.',
    },
    branches: [
      { id: 'prerequisites', label: 'Prerequisites', type: 'prerequisite', children: [
        { id: 'classification-metrics-knn', label: 'Classification metrics', tip: { short: 'Accuracy, precision, recall evaluate predicted labels.', intuition: 'Compare three models on same holdout split.', trap: 'Accuracy misleading on imbalanced classes.' }, lessonId: 'classification-metrics' },
        { id: 'feature-scaling-knn', label: 'Feature scaling', tip: { short: 'Normalize features so distance metrics are meaningful.', intuition: 'kNN and SVM are scale-sensitive.', trap: 'Raw pixel vs metadata features dominate kNN without scaling.' }, lessonId: 'feature-scaling-preprocessing' },
        { id: 'probability-knn', label: 'Probability basics', tip: { short: 'Naive Bayes outputs class posteriors from likelihoods and priors.', intuition: 'Bayes rule combines prior with likelihood product.', trap: 'Confusing likelihood with posterior.' }, lessonId: 'bayes-rule-ml' },
        { id: 'dot-product-knn', label: 'Dot product / hyperplane', tip: { short: 'Linear SVM finds w^T x + b = 0 separating classes.', intuition: 'Margin is distance to nearest points.', trap: 'Non-linear data needs kernels, not linear SVM only.' } },
        { id: 'train-test-knn', label: 'Train / validation split', tip: { short: 'Hold out data for unbiased comparison of three models.', intuition: 'kNN memorizes training set—evaluate on unseen points.', trap: 'Testing kNN on training neighbors inflates accuracy.' }, lessonId: 'train-validation-test-split' },
        { id: 'overfitting-knn', label: 'Overfitting', tip: { short: 'k=1 kNN overfits; large k underfits.', intuition: 'Each model has its own complexity knob.', trap: 'Picking k on test set leaks labels.' }, lessonId: 'overfitting' },
      ]},
      { id: 'mechanism', label: 'Core mechanism', type: 'mechanism', children: [
        { id: 'knn-vote', label: 'kNN majority vote', tip: { short: 'Label query by majority among k nearest training points.', intuition: 'Lazy learning—no explicit training phase.', trap: 'Ties need tie-break rules; odd k avoids ties in binary.' } },
        { id: 'distance-metric-knn', label: 'Distance metric', tip: { short: 'Euclidean distance common; Minkowski and cosine alternatives.', intuition: 'Metric choice defines neighborhood shape.', trap: 'High-dimensional Euclidean distances concentrate.' } },
        { id: 'nb-likelihood', label: 'Naive Bayes likelihood product', tip: { short: 'Assume features independent given class: P(x|y)=∏ P(x_i|y).', intuition: 'Fast scoring with closed-form counts or Gaussians.', trap: 'Correlated features violate independence badly.' } },
        { id: 'nb-prior', label: 'Class prior P(y)', tip: { short: 'Estimate base rate of each class from training counts.', intuition: 'Skewed priors shift decision boundary.', trap: 'Test set with different prior than training.' } },
        { id: 'svm-margin', label: 'SVM margin maximization', tip: { short: 'Find widest slab between classes with support vectors on edge.', intuition: 'Only support vectors matter for boundary.', trap: 'Outliers pull soft-margin slack if C too large.' } },
        { id: 'kernel-trick-knn', label: 'Kernel trick', tip: { short: 'Implicitly map x to higher dimensions for non-linear SVM.', intuition: 'RBF kernel common for curved boundaries.', trap: 'Kernel hyperparameters need cross-validation.' } },
      ]},
      { id: 'intuitions', label: 'Intuitions', type: 'intuition', children: [
        { id: 'three-biases-knn', label: 'Three inductive biases', tip: { short: 'Local smoothness (kNN), factorized likelihood (NB), max margin (SVM).', intuition: 'Bias determines which datasets each wins.', trap: 'Using one model for all tabular problems.' } },
        { id: 'curse-dimension-knn', label: 'Curse of dimensionality', tip: { short: 'In high D, all points become equidistant—kNN degrades.', intuition: 'Feature selection or dimensionality reduction helps kNN.', trap: 'Applying kNN to 10k-dimensional sparse text without reduction.' } },
        { id: 'nb-spam-intuition', label: 'Naive Bayes for text', tip: { short: 'Word counts with independence still work well for spam/ham.', intuition: 'Bag-of-words makes conditional independence less absurd.', trap: 'Bigram dependencies ignored unless feature engineered.' }, lessonId: 'bag-of-words' },
        { id: 'svm-support-intuition', label: 'Support vectors', tip: { short: 'Only near-boundary points define SVM solution.', intuition: 'Robust to distant points unlike kNN using all train points.', trap: 'Mislabeled support vectors swing boundary sharply.' } },
        { id: 'model-selection-knn', label: 'Model selection', tip: { short: 'Tune k, NB smoothing, SVM C and kernel on validation.', intuition: 'No free lunch—validate per dataset.', trap: 'Default sklearn params on every problem.' } },
      ]},
      { id: 'formula-code', label: 'Formula / Code', type: 'formula', children: [
        { id: 'knn-formula', label: 'kNN decision', tip: { short: 'Majority vote of labels of k smallest distances.', intuition: 'Weighted kNN weights closer neighbors more.', formula: '\\hat{y}=\\mathrm{mode}\\{y_i : x_i \\in N_k(x)\\}', trap: 'Forgetting to scale features before distance.' } },
        { id: 'nb-gaussian-formula', label: 'Gaussian NB', tip: { short: 'P(x_i|y) modeled as Gaussian per class per feature.', intuition: 'Continuous features need density model per dimension.', formula: 'P(x_i|y)=\\mathcal{N}(\\mu_{iy},\\sigma_{iy}^2)', trap: 'Zero variance features cause divide-by-zero.' } },
        { id: 'svm-primal-formula', label: 'SVM primal', tip: { short: 'Minimize ||w||² subject to functional margin constraints.', intuition: 'C parameter trades margin width vs slack violations.', formula: '\\min \\frac{1}{2}\\|w\\|^2 + C\\sum \\xi_i', trap: 'Unscaled features make margin meaningless.' } },
        { id: 'sklearn-trio-code', label: 'sklearn trio', tip: { short: 'KNeighborsClassifier, GaussianNB, SVC on same X_train.', intuition: 'Fair comparison needs shared pipeline.', code: 'from sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.naive_bayes import GaussianNB\nfrom sklearn.svm import SVC', trap: 'Different random_state or splits across models.' } },
        { id: 'pipeline-scale-code', label: 'Pipeline with scaler', tip: { short: 'StandardScaler before kNN/SVM in Pipeline.', intuition: 'Prevents leakage by fitting scaler on train only.', code: 'Pipeline([("scale", StandardScaler()), ("clf", SVC())])', trap: 'Scaling test data with full-dataset statistics.' }, lessonId: 'feature-scaling-preprocessing' },
      ]},
      { id: 'traps', label: 'Common traps', type: 'trap', children: [
        { id: 'unscaled-knn-trap', label: 'Unscaled kNN', tip: { short: 'One large-scale feature dominates distance.', intuition: 'Always scale for distance-based methods.', trap: 'Forgotten scaler in production pipeline.' } },
        { id: 'nb-correlation-trap', label: 'NB independence violation', tip: { short: 'Redundant features double-count evidence.', intuition: 'Use feature selection or models handling correlation.', trap: 'NB on raw multicollinear sensor channels.' } },
        { id: 'knn-test-memorize', label: 'kNN evaluates train set', tip: { short: 'Perfect train accuracy with k=1 is meaningless.', intuition: 'kNN is non-parametric memorization baseline.', trap: 'Reporting train accuracy for k=1 as success.' } },
        { id: 'svm-linear-only-trap', label: 'Linear SVM on nonlinear data', tip: { short: 'Needs RBF/poly kernel or feature engineering.', intuition: 'Visualize 2D before picking kernel.', trap: 'Blaming SVM without trying kernels.' } },
        { id: 'class-imbalance-trap-knn', label: 'Class imbalance ignored', tip: { short: 'Majority class wins kNN votes and NB priors.', intuition: 'Use class weights, SMOTE, or metrics beyond accuracy.', trap: '95% accuracy on 95% majority baseline.' }, lessonId: 'classification-metrics' },
      ]},
      { id: 'used-later', label: 'Used later', type: 'application', children: [
        { id: 'logistic-used-knn', label: 'Logistic regression', tip: { short: 'Linear probabilistic classifier compares to linear SVM.', intuition: 'Different loss, similar linear boundary sometimes.', trap: 'Assuming identical decision boundaries always.' }, lessonId: 'logistic-regression' },
        { id: 'tree-ensembles-used-knn', label: 'Tree ensembles', tip: { short: 'Random forests often beat kNN/NB/SVM on heterogeneous tabular.', intuition: 'Next step when linear/simple bias fails.', trap: 'Jumping to deep nets before strong tabular baselines.' }, lessonId: 'tree-ensembles' },
        { id: 'regularization-used-knn', label: 'Regularization', tip: { short: 'SVM margin is explicit regularization; kNN needs k as capacity control.', intuition: 'Unified view of complexity control.', trap: 'Unregularized high-C SVM overfits noise.' }, lessonId: 'regularization' },
        { id: 'cross-validation-used-knn', label: 'Cross-validation', tip: { short: 'Tune k, C, kernel with k-fold CV.', intuition: 'Stable hyperparameter picks for small data.', trap: 'Grid search on test set.' }, lessonId: 'cross-validation' },
        { id: 'bias-variance-used-knn', label: 'Bias-variance tradeoff', tip: { short: 'kNN high variance at small k; NB high bias; SVM balanced via C.', intuition: 'Place three classics on bias-variance spectrum.', trap: 'Ignoring variance of kNN at deployment with drift.' }, lessonId: 'bias-variance-tradeoff' },
      ]},
    ],
  },
];

writeFileSync(outPath, `// Auto-generated pending[24-31] — run generate-pending-24-31.mjs\n${MAPS.map(renderMap).join(',\n')},\n`, 'utf8');
console.log('Wrote', outPath, 'maps:', MAPS.map((m) => m.id).join(', '));
