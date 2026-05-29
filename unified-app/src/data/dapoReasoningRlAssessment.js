function stableHash(text) {
  let hash = 0;
  for (let index = 0; index < text.length; index += 1) {
    hash = (hash * 31 + text.charCodeAt(index)) >>> 0;
  }
  return hash.toString(16).padStart(8, '0');
}

function q(number, level, prompt, choices, answerIndex, explanation) {
  const id = `dapo-${String(number).padStart(3, '0')}`;
  const registryRotation = Number.parseInt(stableHash(`dapo-reasoning-rl:${id}`), 16) % choices.length;
  const targetAnswerIndex = ((number - 1) + registryRotation) % choices.length;
  const correctChoice = choices[answerIndex];
  const distractors = choices.filter((_, index) => index !== answerIndex);
  const balancedChoices = [null, null, null];
  balancedChoices[targetAnswerIndex] = correctChoice;
  let distractorIndex = 0;
  for (let index = 0; index < balancedChoices.length; index += 1) {
    if (balancedChoices[index] === null) {
      balancedChoices[index] = distractors[distractorIndex];
      distractorIndex += 1;
    }
  }

  return Object.freeze({
    id,
    level,
    prompt,
    choices: Object.freeze(balancedChoices),
    answerIndex: targetAnswerIndex,
    explanation,
    questionHash: stableHash(`${prompt}\n${balancedChoices.join('\n')}`),
  });
}

const DAPO_REASONING_SPECS = Object.freeze([
  q(1, 'Foundation', 'What problem setting does DAPO start from before adding its practical fixes?', [
    'GRPO-style reasoning RL where several completions for one prompt are scored and compared within the group.',
    'A retrieval-only pipeline where the language model policy is never updated.',
    'A pure supervised loop that always trains from one fixed demonstration per prompt.',
  ], 0, 'DAPO builds on group-relative reasoning RL, then changes how batches, clipping, loss aggregation, and length rewards behave.'),
  q(2, 'Foundation', 'What is the main reason DAPO matters for long reasoning traces?', [
    'It removes every sampling step and trains only on cached final answers.',
    'It makes GRPO-style training more reliable when long CoT rollouts create batch, clipping, token-credit, and length issues.',
    'It replaces policy optimization with a static prompt template.',
  ], 1, 'Long reasoning RL exposes failure modes that a clean GRPO objective does not solve by itself.'),
  q(3, 'Foundation', 'Which pair best describes the baseline signal DAPO inherits from GRPO?', [
    'Random prompt ordering and fixed teacher forcing.',
    'Embedding cosine scores and retrieval chunk overlap.',
    'Group reward statistics and advantages relative to sibling completions.',
  ], 2, 'The inherited signal compares completions sampled for the same prompt instead of requiring a separate value baseline.'),
  q(4, 'Foundation', 'Why can an all-correct group waste rollout budget in this lesson?', [
    'The completions have little reward contrast, so group-relative advantages carry weak learning signal.',
    'Correct completions cannot be represented as token sequences.',
    'Every correct group must be sent to a human grader before training.',
  ], 0, 'If every sibling wins, the update has little information about which behavior should become more likely.'),
  q(5, 'Foundation', 'Why can an all-wrong group be weak for group-relative learning?', [
    'Wrong groups always have too many tokens for the context window.',
    'There is no successful sibling to reinforce against the failed attempts.',
    'The policy ratio is undefined whenever a reward is zero.',
  ], 1, 'A useful relative update needs contrast, ideally including at least one successful and one failed completion.'),
  q(6, 'Foundation', 'What kind of prompt group usually gives the strongest DAPO training signal?', [
    'A trivial group where every sampled answer is correct.',
    'An impossible group where every sampled answer is wrong.',
    'A frontier group where some completions succeed and some fail.',
  ], 2, 'Frontier prompts create mixed outcomes, which produce nonzero relative advantages.'),
  q(7, 'Foundation', 'What is Dynamic Sampling designed to keep?', [
    'Prompt groups with mixed success and useful within-group comparisons.',
    'Only the shortest completion from every prompt.',
    'Only prompts whose answers are already memorized.',
  ], 0, 'Dynamic Sampling filters for groups that can actually contribute gradient signal.'),
  q(8, 'Foundation', 'What does Dynamic Sampling usually drop from the optimizer batch?', [
    'Groups with medium-length final answers.',
    'Groups that are all correct or all wrong after scoring.',
    'Groups that contain more than one completion for a prompt.',
  ], 1, 'All-correct and all-wrong groups are low contrast under group-relative training.'),
  q(9, 'Foundation', 'What does Clip-Higher change compared with symmetric PPO-style clipping?', [
    'It disables clipping for every negative update.',
    'It clips rewards instead of policy ratios.',
    'It raises the upper ratio bound while keeping the lower side controlled.',
  ], 2, 'DAPO separates lower and upper clip bounds so successful traces can get more room without loosening negative drift.'),
  q(10, 'Foundation', 'Why is the higher upper clip bound useful for positive-advantage traces?', [
    'It lets successful reasoning behavior grow more before the ratio cap blocks the update.',
    'It forces every token to receive the same probability as before training.',
    'It turns a failed answer into a correct answer at inference time.',
  ], 0, 'The lesson frames Clip-Higher as preserving exploration and growth for useful successful traces.'),
  q(11, 'Foundation', 'What stays controlled when Clip-Higher decouples the clipping bounds?', [
    'The lower bound that limits harmful probability-ratio movement.',
    'The number of visible quick-check questions on the page.',
    'The static order of source links in the lesson footer.',
  ], 0, 'The key is asymmetric clipping: more upper room for positive updates while keeping lower control.'),
  q(12, 'Foundation', 'What does token-level policy-gradient loss change for long completions?', [
    'It replaces every reward with a manually labeled token class.',
    'It aggregates update terms across response tokens instead of treating a long answer as one coarse block.',
    'It prevents the model from producing final answers.',
  ], 1, 'Long CoT responses contain many token decisions, so token-level aggregation gives denser credit.'),
  q(13, 'Foundation', 'Why is sample-level loss too coarse for some long-CoT training?', [
    'It hides many token decisions behind one response-sized advantage.',
    'It requires every prompt to have exactly one token.',
    'It always makes the reward variance negative.',
  ], 0, 'The lesson contrasts one coarse response block with many token-level policy decisions.'),
  q(14, 'Foundation', 'What does Overlong Reward Shaping replace?', [
    'A group-relative reward baseline.',
    'A tokenization vocabulary.',
    'A hard reward cliff around the maximum response length.',
  ], 2, 'A hard length cutoff can make nearly acceptable long traces receive a noisy zero signal.'),
  q(15, 'Foundation', 'What is the goal of smoothing length penalties near the max response length?', [
    'Give a more informative concision signal instead of an abrupt all-or-nothing penalty.',
    'Make every answer longer than the max length receive full reward.',
    'Remove the need to score correctness at all.',
  ], 0, 'Overlong shaping keeps length pressure but makes the penalty more gradual near the boundary.'),
  q(16, 'Foundation', 'Which four named techniques are emphasized as the DAPO recipe?', [
    'Beam search, cache eviction, quantization, and speculative decoding.',
    'Dynamic Sampling, Clip-Higher, Token-Level Policy Gradient Loss, and Overlong Reward Shaping.',
    'RAG chunking, reranking, citations, and abstention thresholds.',
  ], 1, 'Those four pieces are the named DAPO fixes for long-CoT group-relative RL.'),
  q(17, 'Foundation', 'Why is DAPO best understood as a recipe instead of one isolated equation?', [
    'It is only a serving-time method with no training objective.',
    'It avoids all reward models and all policy ratios.',
    'Batch filtering, reward shaping, clipping, and loss granularity interact during training.',
  ], 2, 'The same objective pieces behave differently depending on the data pipeline and diagnostics around them.'),
  q(18, 'Foundation', 'Which dashboard signal tells you whether Dynamic Sampling is finding useful groups?', [
    'Effective groups with mixed outcomes.',
    'Only the total parameter count of the base model.',
    'Only the CSS class used by the lesson header.',
  ], 0, 'Effective groups measure whether rollout budget is turning into contrastive training examples.'),
  q(19, 'Foundation', 'Which dashboard signal helps detect loss of exploration during reasoning RL?', [
    'Static prompt title length.',
    'Entropy or a similar diversity proxy.',
    'The number of source links in the lesson.',
  ], 1, 'The lesson uses entropy as a proxy for whether reasoning choices are collapsing.'),
  q(20, 'Foundation', 'Which statement best summarizes the DAPO training pipeline shown in the lesson?', [
    'Prompts create rollout groups, useful groups are kept, rewards are shaped, advantages are computed, and token-level clipped updates are applied.',
    'Prompts are skipped, rewards are ignored, and only inference-time decoding is changed.',
    'Every reasoning step is rewritten by a human before the optimizer sees it.',
  ], 0, 'The pipeline combines rollout filtering, shaped rewards, group-relative advantages, token-level loss, and Clip-Higher.'),

  q(21, 'Mechanism', 'In the lesson code, how is a group marked as having contrast?', [
    'Its accuracy is greater than 0 and less than 1.',
    'Its prompt text contains at least three words.',
    'Its max response length is exactly 2048 tokens.',
  ], 0, 'A mixed group has at least one success and at least one failure, so its accuracy lies strictly between 0 and 1.'),
  q(22, 'Mechanism', 'What happens to the effective batch if most rollout groups are all correct?', [
    'Dynamic Sampling has fewer useful groups to keep, so optimizer budget may be underfilled or low signal.',
    'Clip-Higher automatically invents wrong answers to restore contrast.',
    'Token-level loss turns every correct completion into a negative advantage.',
  ], 1, 'Dynamic Sampling improves quality, but it cannot create contrast when the rollout pool lacks frontier groups.'),
  q(23, 'Mechanism', 'What does increasing prompt difficulty too far tend to do in the DAPO simulation?', [
    'It guarantees every group becomes perfectly balanced.',
    'It removes the need for reward shaping.',
    'It can push groups toward all-wrong outcomes and reduce useful contrast.',
  ], 2, 'Difficulty should target the model frontier; too hard can leave no winning sibling.'),
  q(24, 'Mechanism', 'What does making prompts too easy tend to do to group-relative signal?', [
    'It creates all-correct groups with near-zero relative contrast.',
    'It makes the policy ratio impossible to compute.',
    'It forces the upper clip bound below the lower bound.',
  ], 0, 'Easy prompts consume rollout budget but may not distinguish better and worse reasoning traces.'),
  q(25, 'Mechanism', 'Why does group size affect DAPO batch quality?', [
    'Larger groups can make mixed outcomes more likely but also cost more rollout tokens.',
    'Group size changes the language vocabulary used by the model.',
    'Groups larger than two cannot have rewards.',
  ], 0, 'Group size is a cost-quality tradeoff: more siblings can reveal contrast but increase sampling cost.'),
  q(26, 'Mechanism', 'When Dynamic Sampling is active, which groups are sent forward in the lesson view?', [
    'Only groups with mixed success, up to the displayed batch limit.',
    'Only all-correct groups, because they are safest.',
    'Every group regardless of reward contrast.',
  ], 0, 'The interactive batch keeps contrastive groups instead of blindly optimizing on all rollouts.'),
  q(27, 'Mechanism', 'How does DAPO treat the GRPO advantage idea?', [
    'It keeps group-relative advantages but changes surrounding training mechanics.',
    'It replaces advantages with a retrieval score only.',
    'It requires a separate critic for every token in the trace.',
  ], 0, 'DAPO is not a rejection of GRPO; it is a practical recipe around group-relative advantages.'),
  q(28, 'Mechanism', 'What does a zero reward variance inside a group imply for relative advantages?', [
    'The model must have reached maximum entropy.',
    'Advantages are weak or zero because completions are not separated by reward.',
    'Clip-Higher must be disabled permanently.',
  ], 1, 'Without reward spread, the normalized relative signal has little direction.'),
  q(29, 'Mechanism', 'What does the upper clip slider represent in the DAPO lesson?', [
    'The maximum response length before shaping starts.',
    'The maximum number of prompt groups sampled.',
    'The active upper cap for policy probability ratios when Clip-Higher is used.',
  ], 2, 'Clip-Higher gives the upper probability-ratio cap a separate value from the lower side.'),
  q(30, 'Mechanism', 'What is the standard upper cap used as a baseline in the clipping chart?', [
    '1.20, shown as the symmetric PPO-style reference.',
    '0.00, because all positive updates are blocked.',
    '4096, because the cap is measured in tokens.',
  ], 0, 'The chart compares the active upper bound against a standard 1.20 reference.'),
  q(31, 'Mechanism', 'Why does DAPO keep the lower clip bound controlled?', [
    'To avoid making negative or harmful probability movement too loose.',
    'To force all positive-advantage traces to shrink.',
    'To convert reward values into prompt difficulty labels.',
  ], 0, 'The asymmetry is meant to give more room to good traces without losing guardrails on the other side.'),
  q(32, 'Mechanism', 'For a positive-advantage token with a high policy ratio, what can a higher upper clip allow?', [
    'A larger useful contribution before clipping clamps the term.',
    'A negative reward no matter what the verifier says.',
    'A shorter prompt string in the dataset.',
  ], 0, 'Raising the upper cap can preserve more learning signal for successful behavior.'),
  q(33, 'Mechanism', 'What does the token timeline show when token-level loss is active?', [
    'A separate ratio, clipped ratio, and contribution for each displayed response token role.',
    'Only one scalar label for the entire prompt database.',
    'The raw CSS layout of the assessment panel.',
  ], 0, 'The lesson makes token-level credit visible through per-token policy-ratio terms.'),
  q(34, 'Mechanism', 'What does the sample-level loss view emphasize?', [
    'The whole completion is treated as one coarse action.',
    'Every token receives an independent human-written answer key.',
    'Dynamic Sampling is disabled for all mixed groups.',
  ], 0, 'Sample-level aggregation is simpler but less detailed for long reasoning traces.'),
  q(35, 'Mechanism', 'How does token-level aggregation affect the token-gradient count metric?', [
    'It counts many contributing token terms from nonzero-advantage completions.',
    'It always sets the count to zero.',
    'It counts only prompt titles, not response tokens.',
  ], 0, 'When token-level loss is active, long useful completions contribute multiple gradient terms.'),
  q(36, 'Mechanism', 'What activates overlong shaping in the full DAPO mode?', [
    'The mode combines sampling, clipping, token credit, and shaped length rewards.',
    'Only all-wrong groups are kept and all rewards are set to one.',
    'The lower clip bound is removed from the objective.',
  ], 0, 'Full DAPO turns on the four main fixes together, including shaped length penalties.'),
  q(37, 'Mechanism', 'In the lesson, what does a hard overlong rule do to a correct completion past the max length?', [
    'It can drop the shaped reward to zero immediately.',
    'It raises the upper clip bound automatically.',
    'It marks the group as mixed regardless of other rewards.',
  ], 0, 'The hard rule creates the noisy cliff that Overlong Reward Shaping is meant to soften.'),
  q(38, 'Mechanism', 'What does the soft length margin control?', [
    'How early before the max length the gradual penalty begins.',
    'How many prompt groups are always all correct.',
    'How many answers appear on the assessment page.',
  ], 0, 'The margin defines the region where length pressure ramps up instead of appearing only at the cap.'),
  q(39, 'Mechanism', 'Why can an overlong rate metric matter even when mean reward looks acceptable?', [
    'A rising overlong rate can reveal length-control problems hidden by average reward.',
    'It proves the tokenizer has failed permanently.',
    'It means every completion is correct.',
  ], 0, 'Dashboard metrics need to catch failure modes that a single reward average can hide.'),
  q(40, 'Mechanism', 'What does the reward variance metric indicate in the DAPO dashboard?', [
    'How much contrast exists in the shaped rewards inside the batch.',
    'How many source links are visible.',
    'How many CSS variables define the theme.',
  ], 0, 'Reward variance is a proxy for whether the batch contains separable outcomes.'),
  q(41, 'Mechanism', 'What does clip fraction measure in the lesson dashboard?', [
    'The share of token ratios clamped by the active clipping range.',
    'The share of prompts that have no answer choices.',
    'The percentage of source links opened by the user.',
  ], 0, 'A high clip fraction can show that the policy update is frequently hitting ratio bounds.'),
  q(42, 'Mechanism', 'What does entropy collapse mean in this context?', [
    'Reasoning choices lose diversity, reducing exploration during training.',
    'The model begins using a larger tokenizer vocabulary.',
    'Every group becomes exactly half correct by definition.',
  ], 0, 'The lesson uses entropy as a training-health proxy for preserving diverse reasoning behavior.'),
  q(43, 'Mechanism', 'How can Clip-Higher help with entropy collapse risk?', [
    'By giving successful traces more room before positive updates are capped too tightly.',
    'By dropping every mixed group from the batch.',
    'By replacing all rewards with prompt lengths.',
  ], 0, 'If useful successful traces are capped too hard, the policy may lose promising behaviors.'),
  q(44, 'Mechanism', 'What does the active mode selector demonstrate?', [
    'Each DAPO fix can be viewed alone or combined in the full recipe.',
    'Only one prompt can ever be sampled for the lesson.',
    'The assessment data changes whenever a slider moves.',
  ], 0, 'The modes isolate GRPO baseline, each DAPO fix, and the full DAPO combination.'),
  q(45, 'Mechanism', 'Which sequence matches the pipeline shown in the DAPO recipe card?', [
    'Prompts, Dynamic Sampling, Reward, Advantage, Loss, Objective.',
    'Tokenizer, search index, reranker, citation, abstention, cache.',
    'Pixels, convolution, pooling, flattening, classifier, calibration.',
  ], 0, 'The recipe card orders the training pieces from rollout groups through clipped objective updates.'),
  q(46, 'Mechanism', 'What does the reward step combine in the recipe card?', [
    'A rule-based score plus length shaping.',
    'A retrieval chunk id plus a citation score.',
    'A fixed answer label plus no sampling.',
  ], 0, 'The lesson explicitly pairs correctness-style scoring with shaped length penalties.'),
  q(47, 'Mechanism', 'What does the loss step use after advantages are computed?', [
    'Token-level policy-gradient terms.',
    'Only a static supervised cross-entropy batch.',
    'Only a nearest-neighbor lookup table.',
  ], 0, 'DAPO emphasizes token-level policy-gradient loss for long reasoning traces.'),
  q(48, 'Mechanism', 'What does the objective step add at the end of the recipe?', [
    'A Clip-Higher update with asymmetric ratio bounds.',
    'A new tokenizer trained from scratch on every batch.',
    'A permanent ban on positive advantages.',
  ], 0, 'The final update applies clipped policy optimization with the DAPO clipping change.'),
  q(49, 'Mechanism', 'How should the reported AIME result in the lesson be treated?', [
    'As an author-reported result card, not a universal guarantee.',
    'As proof that DAPO always wins on every benchmark.',
    'As a replacement for checking training diagnostics.',
  ], 0, 'The lesson explicitly frames the result as reported evidence rather than a guarantee.'),
  q(50, 'Mechanism', 'What does the mini-dapo exercise pack reinforce?', [
    'Small implementations for group filtering, clipping, token loss, shaping, entropy, and health labels.',
    'Only frontend styling rules for the page.',
    'A retrieval database with no policy optimization.',
  ], 0, 'The lab mirrors the mechanics in small functions so the training recipe is concrete.'),

  q(51, 'Application', 'A batch has many prompts where all four sampled completions are correct. What is the first DAPO-style diagnosis?', [
    'The rollout pool is too easy and Dynamic Sampling will find little useful contrast.',
    'The upper clip must be below the lower clip.',
    'Token-level loss is impossible because rewards are positive.',
  ], 0, 'All-correct groups point to an easy-prompt or sampling issue before an optimizer issue.'),
  q(52, 'Application', 'A rollout pool has mostly all-wrong groups. Which adjustment is most aligned with the lesson?', [
    'Increase task difficulty even more to make the verifier stricter.',
    'Move toward frontier prompts or sampling settings that produce some successful siblings.',
    'Disable all reward scoring so every group is mixed.',
  ], 1, 'The goal is mixed success; all-wrong groups need a better frontier target or exploration.'),
  q(53, 'Application', 'You increase group size from 4 to 16 and effective groups rise, but rollout cost quadruples. What tradeoff are you seeing?', [
    'A larger group can improve contrast but consumes more sampling budget.',
    'A larger group removes the need for clipping.',
    'A larger group proves length shaping is harmful.',
  ], 0, 'DAPO tuning balances effective signal against the token cost of generating more siblings.'),
  q(54, 'Application', 'A training run shows high mean reward but very low reward variance. What should you suspect?', [
    'The groups may be too uniform, so group-relative updates have weak contrast.',
    'The tokenizer has too many merge rules.',
    'The dashboard is measuring only source link count.',
  ], 0, 'High average reward alone can hide a low-contrast batch.'),
  q(55, 'Application', 'Dynamic Sampling keeps only one group out of eight. What is the likely operational concern?', [
    'The effective batch may be too small even though the raw rollout pool was large.',
    'Every completion must have a positive advantage.',
    'The upper clip bound has become a max-length parameter.',
  ], 0, 'Filtering can expose that nominal batch size is not the same as useful optimizer batch size.'),
  q(56, 'Application', 'A model keeps producing diverse correct traces, but positive updates are clipped almost immediately. Which DAPO fix is relevant?', [
    'Clip-Higher, because it gives positive-advantage traces a larger upper ratio cap.',
    'Dropping all mixed groups from Dynamic Sampling.',
    'Replacing correctness rewards with only hard length cliffs.',
  ], 0, 'This is the use case for raising the upper clipping bound while keeping lower control.'),
  q(57, 'Application', 'A run has very high clip fraction after raising learning pressure. What should you inspect?', [
    'Whether ratio bounds and update size are causing too many token terms to be clamped.',
    'Whether all source links use the same icon.',
    'Whether every prompt is being scored by CSS.',
  ], 0, 'Clip fraction is a direct diagnostic for how often the clipped objective is active.'),
  q(58, 'Application', 'You want to preserve successful reasoning growth but avoid loose negative updates. Which setting is the DAPO-shaped move?', [
    'Raise the upper ratio cap while keeping the lower bound controlled.',
    'Raise both bounds without checking clip fraction.',
    'Remove rewards from failed samples.',
  ], 0, 'The lesson calls for decoupled clipping rather than simply making all movement looser.'),
  q(59, 'Application', 'A long correct solution is barely over the response limit and receives zero reward under a hard rule. What should you try?', [
    'Overlong Reward Shaping with a soft margin near the max length.',
    'Dropping all correct completions from the batch.',
    'Making token-level loss sample-level only forever.',
  ], 0, 'A smooth length penalty is designed for near-boundary cases like this.'),
  q(60, 'Application', 'The overlong rate rises while effective groups remain healthy. What does this imply?', [
    'The batch has signal, but length control still needs attention.',
    'Dynamic Sampling must be removed because every group is useless.',
    'All answers are necessarily wrong.',
  ], 0, 'Different dashboard signals diagnose different failure modes.'),
  q(61, 'Application', 'A model learns to write extremely long traces that often pass correctness checks. What DAPO component helps push concision?', [
    'Length reward shaping around the max response boundary.',
    'A larger upper clip with no length metric.',
    'A retrieval-only reranker.',
  ], 0, 'Overlong shaping can apply pressure against excessive length without a noisy cliff.'),
  q(62, 'Application', 'A long-CoT run improves final accuracy slowly, and useful completions have many reasoning tokens. Which loss granularity should you test?', [
    'Token-level aggregation so more response decisions contribute to the update.',
    'Only prompt-level labels with no completion tokens.',
    'No policy-gradient loss at all.',
  ], 0, 'Token-level loss is useful when long completions hide many update-relevant decisions.'),
  q(63, 'Application', 'A sample-level update treats a 2000-token solution as one action. What risk does that create?', [
    'Credit assignment is coarse across many token decisions.',
    'The group cannot have a reward value.',
    'The verifier must become a tokenizer.',
  ], 0, 'The lesson motivates token-level loss as a denser alternative for long traces.'),
  q(64, 'Application', 'A dashboard shows low entropy after many updates. Which concern should come first?', [
    'Reasoning diversity may be collapsing, so exploration and clipping behavior need review.',
    'The static lesson title is too short.',
    'All groups are guaranteed to be frontier groups.',
  ], 0, 'Low entropy is a health warning in long-CoT RL.'),
  q(65, 'Application', 'After enabling Dynamic Sampling, entropy improves but effective groups are still scarce. What is a reasonable next check?', [
    'Prompt difficulty and sampling settings, because the rollout pool may not contain enough mixed groups.',
    'Whether final answers are hidden from the user interface.',
    'Whether source links open in a new tab.',
  ], 0, 'Dynamic Sampling can filter good groups but cannot manufacture them from a poor rollout pool.'),
  q(66, 'Application', 'A batch has good effective groups and moderate entropy, but token-gradient count is low. Which setting may be inactive?', [
    'Token-level loss may be off or too few nonzero-advantage completions are present.',
    'The page source trail may have too many links.',
    'The reward curve must be hard-coded to one.',
  ], 0, 'Token-gradient count reflects loss granularity and nonzero-advantage completions.'),
  q(67, 'Application', 'A team reports only final benchmark score for DAPO training. What production review should ask for?', [
    'Diagnostics such as effective groups, entropy, clip fraction, overlong rate, and token-gradient count.',
    'Only screenshots of the header badge.',
    'Only the model parameter count.',
  ], 0, 'The lesson emphasizes training-health diagnostics, not just final result cards.'),
  q(68, 'Application', 'A run uses hard max-length penalties and shows unstable reward variance near the boundary. What is the DAPO-style fix?', [
    'Use linear or soft overlong shaping so near-boundary traces receive graded penalties.',
    'Drop all prompts with correct answers.',
    'Set all policy ratios to one by hand.',
  ], 0, 'Shaping reduces noise from abrupt length cliffs.'),
  q(69, 'Application', 'You see many all-correct groups after lowering difficulty. What change is most likely to recover signal?', [
    'Increase difficulty toward the model frontier instead of training on trivial prompts.',
    'Lower the upper clip to zero.',
    'Remove Dynamic Sampling from the pipeline.',
  ], 0, 'The useful region is the frontier where samples disagree.'),
  q(70, 'Application', 'You see many all-wrong groups after raising difficulty. What change is most likely to recover signal?', [
    'Lower difficulty or improve exploration so at least some siblings succeed.',
    'Hard-penalize all answers shorter than the max length.',
    'Convert every failed answer into a positive reward.',
  ], 0, 'The target is mixed success, not making the task impossible.'),
  q(71, 'Application', 'A verifier rewards final answer format too much and correctness too little. What can happen?', [
    'The model may optimize polished but wrong traces, so reward design needs audit.',
    'Dynamic Sampling becomes unnecessary for all future runs.',
    'The lower clip bound becomes a token length.',
  ], 0, 'DAPO still depends on meaningful rewards; better mechanics cannot save a bad scoring target.'),
  q(72, 'Application', 'A correct trace is overlong because it rambles before the final answer. Which two signals should be reviewed together?', [
    'Correctness reward and overlong shaping, because the model needs both solving and concision pressure.',
    'Only the number of visible quick checks and source links.',
    'Only the static route id and page title.',
  ], 0, 'Length shaping should complement correctness rather than replace it.'),
  q(73, 'Application', 'A model quickly stops exploring alternative solution paths after early wins. Which combination is most relevant?', [
    'Entropy monitoring plus Clip-Higher and sampling review.',
    'Only hard length penalties with no dashboard.',
    'Only fewer answer choices in the assessment.',
  ], 0, 'The lesson ties exploration health to entropy, clipping pressure, and useful batch construction.'),
  q(74, 'Application', 'A run has many useful groups but final updates are noisy because a few very long traces hit a reward cliff. What should you change first?', [
    'Smooth the length rule with Overlong Reward Shaping.',
    'Drop Dynamic Sampling even though groups are useful.',
    'Ignore overlong rate because mean reward exists.',
  ], 0, 'The described issue is specifically the noisy boundary that shaping addresses.'),
  q(75, 'Application', 'Before calling a DAPO run production-ready, what should the assessment cover?', [
    'Group signal, clipping behavior, token credit, length shaping, reward validity, and dashboard health.',
    'Only whether the page builds once.',
    'Only whether all completions are correct on easy prompts.',
  ], 0, 'A production review needs the interacting pieces, not only one metric.'),

  q(76, 'Tricky', 'Which answer rejects the trap: DAPO is just GRPO with a new name?', [
    'False; DAPO adds interacting fixes for sampling, clipping, token-level loss, and overlong rewards around GRPO-style signal.',
    'True; DAPO changes no training behavior beyond renaming GRPO.',
    'True; DAPO only edits the inference prompt and never changes optimization.',
  ], 0, 'The lesson misconception is that DAPO is merely a rename; the recipe changes the training system.'),
  q(77, 'Tricky', 'Which answer rejects the trap: Dynamic Sampling should keep every all-correct group because correct data is always best?', [
    'False; all-correct groups can lack within-group contrast even when the answers are correct.',
    'True; every all-correct group has the strongest possible relative advantage signal.',
    'True; all-correct groups automatically increase entropy.',
  ], 0, 'Correctness alone is not enough for group-relative learning; contrast matters.'),
  q(78, 'Tricky', 'Which answer rejects the trap: all-wrong groups are ideal because they show the model what not to do?', [
    'False; without a successful sibling, group-relative updates have no winner to reinforce.',
    'True; all-wrong groups always produce the largest positive advantages.',
    'True; DAPO requires rewards to be zero for every completion.',
  ], 0, 'All-wrong groups may reveal task difficulty, but they are weak optimizer batches for this signal.'),
  q(79, 'Tricky', 'Which answer rejects the trap: Clip-Higher simply removes policy safety bounds?', [
    'False; it raises the upper bound for positive updates while keeping a controlled lower bound.',
    'True; Clip-Higher means no ratio is ever clipped.',
    'True; Clip-Higher clips only prompt strings, not policy ratios.',
  ], 0, 'Clip-Higher is asymmetric clipping, not the absence of clipping.'),
  q(80, 'Tricky', 'Which answer rejects the trap: token-level loss means humans label every token?', [
    'False; it means policy-gradient terms are aggregated across generated tokens, not hand-labeled token supervision.',
    'True; every generated token needs a separate human answer key.',
    'True; token-level loss forbids sequence rewards.',
  ], 0, 'The token-level piece is about loss granularity for policy optimization.'),
  q(81, 'Tricky', 'Which answer rejects the trap: Overlong Reward Shaping rewards long answers more?', [
    'False; it smooths the penalty near the length boundary while still applying concision pressure.',
    'True; the longer the answer, the higher the shaped reward.',
    'True; shaping removes max response length from training.',
  ], 0, 'Shaping makes the length signal less noisy, not pro-verbosity.'),
  q(82, 'Tricky', 'Which answer rejects the trap: a high mean reward proves the DAPO batch is healthy?', [
    'False; low variance, low effective groups, entropy collapse, or overlong traces can still be problems.',
    'True; mean reward alone is a complete production diagnostic.',
    'True; clip fraction and entropy are unrelated to policy training.',
  ], 0, 'The lesson uses multiple dashboard signals because one average can hide failures.'),
  q(83, 'Tricky', 'Which answer rejects the trap: a large nominal batch guarantees a large useful batch?', [
    'False; many groups can be filtered out if they are all correct, all wrong, overlong, or low signal.',
    'True; raw rollout count is identical to effective optimizer signal.',
    'True; Dynamic Sampling never changes batch usefulness.',
  ], 0, 'DAPO distinguishes raw rollout volume from effective training signal.'),
  q(84, 'Tricky', 'Which answer rejects the trap: raising upper clip always improves training?', [
    'False; it should be checked against clip fraction, entropy, reward quality, and stability.',
    'True; higher upper clips are universally better with no diagnostics.',
    'True; upper clip changes prompt difficulty directly.',
  ], 0, 'Clip-Higher is a controlled tool, not a guarantee.'),
  q(85, 'Tricky', 'Which answer rejects the trap: Dynamic Sampling fixes a bad reward function?', [
    'False; it filters by observed outcomes but still depends on rewards that measure the right behavior.',
    'True; filtering makes every reward definition correct.',
    'True; reward design is irrelevant once groups are mixed.',
  ], 0, 'A mixed group based on a flawed reward can still optimize the wrong thing.'),
  q(86, 'Tricky', 'Which answer rejects the trap: token-level loss removes the need for group-relative advantages?', [
    'False; token-level aggregation still uses the policy-gradient signal derived from advantages.',
    'True; token-level loss replaces rewards with token positions only.',
    'True; every token gets positive advantage by default.',
  ], 0, 'Token-level loss changes where terms are aggregated, not the need for a learning signal.'),
  q(87, 'Tricky', 'Which answer rejects the trap: hard length cliffs are always cleaner than shaped penalties?', [
    'False; a hard cliff can create noisy jumps for nearly acceptable long traces.',
    'True; hard cliffs always provide the smoothest reward signal.',
    'True; shaped penalties cannot express concision pressure.',
  ], 0, 'The overlong fix exists because hard boundaries can be noisy.'),
  q(88, 'Tricky', 'Which answer rejects the trap: the reported AIME score proves universal DAPO superiority?', [
    'False; it is an author-reported result and must not replace task-specific evaluation.',
    'True; one reported score guarantees every future benchmark result.',
    'True; diagnostics are unnecessary after a result card appears.',
  ], 0, 'The lesson explicitly warns against treating the reported result as a universal guarantee.'),
  q(89, 'Tricky', 'Which answer rejects the trap: entropy collapse is harmless if some answers remain correct?', [
    'False; reduced diversity can harm exploration and future reasoning improvement.',
    'True; once any answer is correct, exploration no longer matters.',
    'True; entropy is only a frontend layout metric.',
  ], 0, 'Training health includes preserving useful exploration, not just current correctness.'),
  q(90, 'Tricky', 'Which answer rejects the trap: DAPO can be audited by reading only its final objective formula?', [
    'False; batch construction, reward shaping, clipping, loss granularity, and diagnostics must be reviewed together.',
    'True; one formula fully determines production behavior.',
    'True; rollout filtering is unrelated to optimization quality.',
  ], 0, 'The lesson frames DAPO as a recipe whose parts interact.'),

  q(91, 'Interview', 'In an interview, how would you explain DAPO in one precise sentence?', [
    'DAPO is a practical recipe for scaling GRPO-style reasoning RL with useful batch filtering, asymmetric clipping, token-level credit, and smoother length rewards.',
    'DAPO is a retrieval method that avoids policy optimization.',
    'DAPO is a supervised dataset format for one-answer prompts.',
  ], 0, 'A strong answer names the base method and the four practical fixes.'),
  q(92, 'Interview', 'How would you debug a DAPO run whose effective group count suddenly drops?', [
    'Inspect prompt difficulty, sampling diversity, reward/verifier behavior, and whether groups became all correct or all wrong.',
    'Only increase the number of visible UI cards.',
    'Only lower the max response length without looking at rewards.',
  ], 0, 'Effective group loss is a rollout and reward-signal problem before it is just an optimizer problem.'),
  q(93, 'Interview', 'How would you justify Clip-Higher to a skeptical reviewer?', [
    'It decouples upper and lower ratio bounds so positive-advantage traces can grow while negative movement remains controlled.',
    'It removes all clipping because clipping is never useful.',
    'It changes the verifier so every answer is correct.',
  ], 0, 'The justification should mention asymmetric bounds and the reason for preserving successful trace growth.'),
  q(94, 'Interview', 'How would you explain token-level loss without overselling it?', [
    'It gives denser policy-gradient aggregation across long responses, but it still depends on good rewards and stable advantages.',
    'It proves every token is logically correct.',
    'It removes the need for rollout groups.',
  ], 0, 'A production answer states both benefit and dependency.'),
  q(95, 'Interview', 'How would you choose between hard, linear, and soft overlong handling?', [
    'Compare boundary noise, overlong rate, correctness retention, and whether the shaped penalty gives useful concision pressure.',
    'Always choose hard cliffs because they cannot introduce noise.',
    'Always reward longer traces because reasoning needs more tokens.',
  ], 0, 'The right choice depends on diagnostics around length and reward behavior.'),
  q(96, 'Interview', 'What metrics would you put on a DAPO training dashboard?', [
    'Effective groups, reward variance, entropy, clip fraction, overlong rate, token-gradient count, and final task accuracy.',
    'Only final benchmark score and model size.',
    'Only prompt title length and source-link count.',
  ], 0, 'The dashboard should cover signal quality, exploration, clipping pressure, length behavior, gradient density, and outcomes.'),
  q(97, 'Interview', 'How would you respond if a teammate says Dynamic Sampling wastes data by dropping groups?', [
    'It trades raw rollout count for useful optimizer signal by removing groups with little relative contrast.',
    'It should drop mixed groups and keep only all-correct groups.',
    'It changes tokenization so no data is needed.',
  ], 0, 'The point is not to use every sample; it is to spend optimizer budget on informative comparisons.'),
  q(98, 'Interview', 'What production risk remains even with all four DAPO fixes enabled?', [
    'A flawed reward can still optimize polished wrong reasoning or excessive length, so reward audits and evaluations remain necessary.',
    'There is no remaining risk once the recipe is enabled.',
    'The model stops using tokens after token-level loss is enabled.',
  ], 0, 'DAPO mechanics cannot replace a valid objective and careful evaluation.'),
  q(99, 'Interview', 'How would you describe the relationship between DAPO and GRPO?', [
    'DAPO keeps GRPO-style group-relative learning and adds system-level fixes for long-CoT training failures.',
    'DAPO is unrelated to group-relative learning.',
    'DAPO is only a static inference prompt for GRPO models.',
  ], 0, 'DAPO should be described as a practical extension around the GRPO signal, not a completely separate idea.'),
  q(100, 'Interview', 'What final review would make a DAPO assessment production-ready?', [
    'Verify progressive coverage from GRPO basics through the four fixes, include traps only after setup, avoid answer leakage, and test all data invariants.',
    'Check only that there are ten easy questions.',
    'Remove explanations so the learner cannot review mistakes.',
  ], 0, 'A production-ready assessment needs correct sequencing, no spoilers, meaningful traps, and automated data checks.'),
]);

export const DAPO_REASONING_RL_QUIZ = Object.freeze(DAPO_REASONING_SPECS);
