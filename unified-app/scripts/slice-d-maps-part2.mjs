function branch(id, label, type, children) {
  return { id, label, type, children };
}

function leaf(id, label, tip, lessonId) {
  return { id, label, tip, ...(lessonId ? { lessonId } : {}) };
}

export const MAPS = [
  {
    id: 'k-means',
    label: 'K-Means Clustering',
    center: {
      short: 'K-means partitions unlabeled points into k clusters by alternating assignment to nearest centroids and recomputing each centroid as the cluster mean.',
      intuition: 'Centroids act like magnets—points join the closest one, then each magnet moves to the average of its group.',
      formula: '\\min_C \\sum_i \\lVert x_i - c_{a_i}\\rVert^2',
      why: 'K-means is a fast baseline for clustering, customer segmentation, embedding visualization, and initialization for deeper models.',
      trap: 'Lower inertia from larger k does not automatically mean a better or interpretable clustering.',
    },
    branches: [
      branch('prerequisites', 'Prerequisites', 'prerequisite', [
        leaf('vector-km', 'Vector', { short: 'Each data point is a numeric vector in feature space.', intuition: 'Distance is computed coordinate-wise.', trap: 'Categorical features need encoding before k-means.', lessonId: 'embeddings' }),
        leaf('euclidean-km', 'Euclidean distance', { short: 'Default metric is squared L2 distance to centroids.', intuition: 'Nearest centroid wins the point.', trap: 'Unscaled features make large-scale dimensions dominate.', lessonId: 'feature-scaling-preprocessing' }),
        leaf('mean-km', 'Mean', { short: 'Centroid update averages all assigned points.', intuition: 'The mean minimizes sum of squared distances within the cluster.', trap: 'Outliers pull centroids away from the bulk of the cluster.' }),
        leaf('unsupervised-km', 'Unsupervised learning', { short: 'No labels—structure is inferred from geometry alone.', intuition: 'Clusters are hypotheses about groupings.', trap: 'Clusters are not guaranteed to match human categories.' }),
        leaf('k-hyperparam-km', 'Choice of k', { short: 'k is set before running the algorithm.', intuition: 'Different k values yield different partitions.', trap: 'There is no universal correct k without domain goals.' }),
        leaf('inertia-km', 'Inertia', { short: 'Sum of squared distances from points to their assigned centroids.', intuition: 'Inertia decreases as k increases.', trap: 'Inertia alone cannot pick k—you need elbow or domain judgment.' }),
      ]),
      branch('mechanism', 'Core mechanism', 'mechanism', [
        leaf('init-centroids-km', 'Initialize centroids', { short: 'Pick k starting centers—random points or k-means++.', intuition: 'Bad starts can trap the algorithm in poor local minima.', trap: 'Single random restart may yield unstable clusters.' }),
        leaf('assign-step-km', 'Assignment step', { short: 'Each point joins the nearest centroid.', intuition: 'Hard assignment: every point belongs to exactly one cluster.', trap: 'Ties in distance require a consistent tie-break rule.' }),
        leaf('update-step-km', 'Update step', { short: 'Recompute each centroid as the mean of its assigned points.', intuition: 'Empty clusters need reinitialization or removal.', trap: 'An empty cluster leaves a centroid with no data support.' }),
        leaf('iterate-km', 'Iterate until convergence', { short: 'Repeat assign and update until assignments stop changing.', intuition: 'Each iteration lowers or maintains inertia.', trap: 'Convergence to a local minimum, not global optimum.' }),
        leaf('multiple-restarts-km', 'Multiple restarts', { short: 'Run k-means several times with different seeds; keep best inertia.', intuition: 'Reduces sensitivity to unlucky initialization.', trap: 'Still no guarantee of globally optimal partition.' }),
      ]),
      branch('intuitions', 'Intuitions', 'intuition', [
        leaf('magnets-km', 'Magnets picture', { short: 'Centroids pull points; points pull centroids toward their mass.', intuition: 'The dance stops at a stable assignment.', trap: 'Stable does not mean semantically meaningful.' }),
        leaf('voronoi-km', 'Voronoi regions', { short: 'Each centroid owns the region of points closest to it.', intuition: 'Cluster boundaries are hyperplanes between centroids.', trap: 'Boundaries are axis-aligned only in distance geometry, not feature importance.' }),
        leaf('spherical-km', 'Spherical clusters', { short: 'K-means prefers roughly equal-variance blob-shaped groups.', intuition: 'Elongated or nested shapes are a poor fit.', trap: 'Forcing k-means on crescent moons yields awkward splits.' }),
        leaf('scale-intuition-km', 'Feature scale', { short: 'Large-magnitude features dominate distance.', intuition: 'Standardize when units differ.', trap: 'Clustering raw income and age without scaling.', lessonId: 'feature-scaling-preprocessing' }),
        leaf('elbow-km', 'Elbow intuition', { short: 'Plot inertia vs k; look for diminishing returns.', intuition: 'Elbow is a heuristic, not a theorem.', trap: 'Elbows are often subjective on real data.' }),
      ]),
      branch('formula-code', 'Formula / Code', 'formula', [
        leaf('objective-km', 'Objective', { short: 'Minimize sum of squared distances to assigned centroids.', intuition: 'Assignment and mean update greedily reduce this objective.', formula: 'J=\\sum_{i=1}^{n}\\lVert x_i-c_{a_i}\\rVert^2', trap: 'Objective value depends on scaling of x.' }),
        leaf('assign-formula-km', 'Assignment rule', { short: 'a_i = argmin_j ||x_i - c_j||^2.', intuition: 'Each point picks the closest center.', trap: 'Using L1 distance changes the algorithm entirely.' }),
        leaf('update-formula-km', 'Centroid update', { short: 'c_j = mean of all x_i with a_i = j.', intuition: 'Mean is the L2 minimizer for fixed assignments.', trap: 'Empty cluster breaks the mean formula.' }),
        leaf('sklearn-km', 'sklearn sketch', { short: 'KMeans(n_clusters=k, n_init=10).fit(X).', intuition: 'Library handles restarts and empty-cluster handling.', code: 'from sklearn.cluster import KMeans\nkm = KMeans(n_clusters=3, n_init=10).fit(X)', trap: 'Forgetting to scale X before clustering.' }),
        leaf('inertia-code-km', 'Inertia check', { short: 'km.inertia_ reports final within-cluster sum of squares.', intuition: 'Compare across restarts with same k.', trap: 'Comparing inertia across different k without context.' }),
      ]),
      branch('traps', 'Common traps', 'trap', [
        leaf('k-too-large-km', 'k too large', { short: 'Every point can become its own cluster when k = n.', intuition: 'Inertia goes to zero but structure is lost.', trap: 'Chasing zero inertia with huge k.' }),
        leaf('local-min-km', 'Local minima', { short: 'Different seeds yield different partitions.', intuition: 'Always compare multiple restarts.', trap: 'Trusting one run without checking stability.' }),
        leaf('non-spherical-km', 'Non-spherical data', { short: 'K-means splits elongated clusters awkwardly.', intuition: 'Use spectral, GMM, or density methods when shapes differ.', trap: 'Forcing k-means on clearly non-convex manifolds.' }),
        leaf('outlier-trap-km', 'Outliers', { short: 'Outliers distort centroids and assignments.', intuition: 'Robust clustering or trimming may help.', trap: 'One bad point can define a whole cluster.' }),
        leaf('label-confusion-km', 'Clusters ≠ classes', { short: 'Unsupervised groups need not align with true labels.', intuition: 'Evaluate with domain sense, not only silhouette scores.', trap: 'Assuming cluster id maps to ground-truth category.' }),
      ]),
      branch('used-later', 'Used later', 'application', [
        leaf('pca-km', 'PCA', { short: 'Reduce dimension before clustering high-D embeddings.', intuition: 'Cluster in a variance-preserving subspace.', trap: 'PCA removes signal along low-variance but predictive directions.', lessonId: 'pca' }),
        leaf('feature-scaling-km', 'Feature scaling', { short: 'Standardize features so distance is fair across columns.', intuition: 'Preprocessing is part of the clustering pipeline.', trap: 'Fitting scaler on test data leaks information.', lessonId: 'feature-scaling-preprocessing' }),
        leaf('model-debugging-km', 'Model debugging', { short: 'Cluster embedding errors or users to find failure modes.', intuition: 'Exploratory clustering supports ML ops workflows.', trap: 'Treating exploratory clusters as production labels.', lessonId: 'model-debugging' }),
        leaf('recommender-km', 'Recommender systems', { short: 'User/item segmentation often starts with centroid clustering.', intuition: 'Centroids summarize behavior prototypes.', trap: 'Cold-start users do not sit near any centroid.', lessonId: 'recommender-systems-ranking-track' }),
        leaf('gmm-later-km', 'Mixture models', { short: 'Soft assignments generalize hard k-means partitions.', intuition: 'Gaussian mixtures allow elliptical clusters.', trap: 'Assuming k-means outputs are probabilities.' }),
      ]),
    ],
  },
  {
    id: 'knn-naive-bayes-svm',
    label: 'kNN, Naive Bayes, and SVM',
    center: {
      short: 'Three classic classifiers: kNN votes from local neighbors, Naive Bayes multiplies class-conditional feature likelihoods under independence, and SVM finds a maximum-margin separating hyperplane.',
      intuition: 'Same feature space, three different bets—local geometry, probabilistic independence, or global margin geometry.',
      formula: '\\hat{y}=\\mathrm{vote}_k(x)\\;|\\;\\arg\\max_y p(y)\\prod_j p(x_j|y)\\;|\\;\\mathrm{sign}(w^Tx+b)',
      why: 'These models remain strong baselines, teaching assumptions, and components in ensembles and text pipelines.',
      trap: 'They are not plug-compatible—scaling, independence, and margin geometry each change when a model wins.',
    },
    branches: [
      branch('prerequisites', 'Prerequisites', 'prerequisite', [
        leaf('vector-knbs', 'Feature vectors', { short: 'Each example is a point in feature space.', intuition: 'Distance, likelihood, and margin all read the same coordinates.', trap: 'Mixed feature types need encoding first.' }),
        leaf('classification-knbs', 'Classification', { short: 'Predict a discrete label from inputs.', intuition: 'Each model outputs class scores differently.', trap: 'Regression variants exist but this lesson focuses on labels.', lessonId: 'logistic-regression' }),
        leaf('distance-knbs', 'Distance', { short: 'kNN relies on how far query points are from stored examples.', intuition: 'Nearest neighbors dominate the vote.', trap: 'Unscaled features distort neighbor search.', lessonId: 'feature-scaling-preprocessing' }),
        leaf('probability-knbs', 'Probability', { short: 'Naive Bayes outputs class posteriors from priors and likelihoods.', intuition: 'Bayes rule combines evidence with base rates.', trap: 'Independence assumption is almost always approximate.', lessonId: 'bayes-rule-ml' }),
        leaf('hyperplane-knbs', 'Hyperplane', { short: 'SVM separates classes with a linear boundary w·x + b = 0.', intuition: 'Support vectors anchor the margin.', trap: 'Linear boundary fails when classes are not linearly separable without kernels.' }),
        leaf('train-test-knbs', 'Train / test split', { short: 'kNN stores training points; others learn parameters on train.', intuition: 'Evaluation must be on held-out data.', trap: 'kNN “memorizes” train set—test leakage inflates scores.', lessonId: 'train-validation-test-split' }),
      ]),
      branch('mechanism', 'Core mechanism', 'mechanism', [
        leaf('knn-vote-knbs', 'kNN voting', { short: 'Find k closest training points; majority label wins.', intuition: 'Small k is flexible; large k is smoother.', trap: 'Even k ties need a tie-break rule.' }),
        leaf('nb-likelihood-knbs', 'Naive Bayes likelihood', { short: 'Multiply P(x_j|y) across features assuming independence given y.', intuition: 'Log domain prevents numeric underflow.', formula: '\\hat{y}=\\arg\\max_y \\log p(y)+\\sum_j \\log p(x_j|y)', trap: 'Zero likelihoods from unseen feature values.' }),
        leaf('nb-variants-knbs', 'NB variants', { short: 'Gaussian for continuous; multinomial for counts; Bernoulli for binary.', intuition: 'Pick the likelihood model matching feature type.', trap: 'Using Bernoulli NB on raw word counts.' }),
        leaf('svm-margin-knbs', 'Maximum margin', { short: 'Choose w,b that maximize distance to nearest points of each class.', intuition: 'Support vectors lie on the margin boundary.', formula: '\\min \\frac{1}{2}\\lVert w\\rVert^2 \\text{ s.t. } y_i(w^Tx_i+b)\\geq 1', trap: 'Outliers can dominate the margin without slack.' }),
        leaf('svm-kernel-knbs', 'Kernel trick', { short: 'Implicitly map x to higher dimensions where separation is easier.', intuition: 'K(x,x\') replaces dot products in feature space.', trap: 'Wrong kernel or gamma overfits small data.' }),
      ]),
      branch('intuitions', 'Intuitions', 'intuition', [
        leaf('local-vs-global-knbs', 'Local vs global', { short: 'kNN is local; linear SVM is global; NB is generative per class.', intuition: 'Disagreement on a query reveals assumption mismatch.', trap: 'Picking the winner on one point without task context.' }),
        leaf('curse-dim-knbs', 'Curse of dimensionality', { short: 'In high dimensions all points look equally far—kNN suffers.', intuition: 'Distance concentration hurts neighbor meaning.', trap: 'Applying kNN to raw high-dimensional text without reduction.' }),
        leaf('nb-sparsity-knbs', 'NB for text', { short: 'Word counts with multinomial NB remain strong spam baselines.', intuition: 'Independence is wrong but often good enough.', trap: 'Ignoring stopwords and smoothing still hurts.' }),
        leaf('svm-support-knbs', 'Support vectors', { short: 'Only border points matter for the SVM boundary.', intuition: 'Interior points could be removed without changing the margin.', trap: 'Assuming all training points equally influence SVM.' }),
        leaf('scale-intuition-knbs', 'Scaling matters', { short: 'kNN and SVM with RBF kernels need comparable feature scales.', intuition: 'One large column dominates distance or margin.', trap: 'Training SVM on unscaled mixed-unit data.', lessonId: 'feature-scaling-preprocessing' }),
      ]),
      branch('formula-code', 'Formula / Code', 'formula', [
        leaf('knn-formula-knbs', 'kNN rule', { short: 'ŷ = mode of labels among k nearest training points.', intuition: 'Distance metric defines “nearest”.', trap: 'L2 on unscaled data is not neutral.' }),
        leaf('nb-log-formula-knbs', 'Log NB score', { short: 'score(y) = log p(y) + Σ_j log p(x_j|y).', intuition: 'Argmax over y picks the class.', trap: 'Forgetting Laplace smoothing on zeros.' }),
        leaf('svm-hinge-knbs', 'Hinge loss view', { short: 'Soft-margin SVM penalizes margin violations.', intuition: 'C trades margin width vs misclassification.', trap: 'Huge C overfits noise; tiny C underfits.' }),
        leaf('sklearn-knn-knbs', 'sklearn kNN', { short: 'KNeighborsClassifier(n_neighbors=k).fit(X,y).', code: 'from sklearn.neighbors import KNeighborsClassifier\nclf = KNeighborsClassifier(n_neighbors=5).fit(X, y)', trap: 'Not scaling X before neighbors.' }),
        leaf('sklearn-svm-knbs', 'sklearn SVM', { short: 'SVC(kernel="rbf", C=1.0, gamma="scale").', code: 'from sklearn.svm import SVC\nclf = SVC(kernel="rbf").fit(X, y)', trap: 'Grid search on test set instead of validation.' }),
      ]),
      branch('traps', 'Common traps', 'trap', [
        leaf('knn-storage-knbs', 'kNN storage cost', { short: 'Prediction scans or indexes the entire training set.', intuition: 'Latency grows with train size.', trap: 'Deploying kNN on millions of points without ANN index.' }),
        leaf('nb-independence-knbs', 'Independence violation', { short: 'Correlated features double-count evidence in NB.', intuition: 'Still works surprisingly often for text.', trap: 'Treating NB probabilities as well calibrated.', lessonId: 'calibration' }),
        leaf('svm-slow-knbs', 'SVM scale', { short: 'Kernel SVM training can be slow on large n.', intuition: 'Linear SVM or deep models may scale better.', trap: 'Default RBF SVM on 1M rows without approximation.' }),
        leaf('class-imbalance-knbs', 'Class imbalance', { short: 'Majority class wins kNN votes and NB priors.', intuition: 'Class weights or resampling may be needed.', trap: 'Reporting accuracy alone on imbalanced data.', lessonId: 'classification-metrics' }),
        leaf('data-leakage-knbs', 'Leakage in kNN', { short: 'Duplicates in train and test make neighbors cheat.', intuition: 'Near duplicates inflate kNN accuracy.', trap: 'Random split when users or documents repeat.', lessonId: 'data-leakage-deep-dive' }),
      ]),
      branch('used-later', 'Used later', 'application', [
        leaf('classification-metrics-knbs', 'Classification metrics', { short: 'Precision, recall, and F1 evaluate these classifiers beyond accuracy.', intuition: 'Threshold-free scores differ by model.', trap: 'One metric hides expensive false negatives.', lessonId: 'classification-metrics' }),
        leaf('roc-pr-knbs', 'ROC / PR curves', { short: 'Score-ranked evaluation compares models across thresholds.', intuition: 'SVM and NB produce scores; kNN can use neighbor vote fractions.', trap: 'Using ROC alone when positives are rare.', lessonId: 'roc-pr-curves' }),
        leaf('regularization-knbs', 'Regularization', { short: 'Modern linear models use penalized logistic regression instead of plain linear SVM in many pipelines.', intuition: 'C in SVM plays a similar role to inverse λ.', trap: 'Assuming regularization removes need for scaling.', lessonId: 'regularization' }),
        leaf('tree-ensembles-knbs', 'Tree ensembles', { short: 'Nonlinear boundaries without explicit kernels or neighbors.', intuition: 'Compare classic baselines before jumping to forests.', trap: 'Skipping simple baselines on tabular data.', lessonId: 'tree-ensembles' }),
        leaf('embeddings-knbs', 'Embeddings', { short: 'kNN retrieval in embedding space powers RAG and recommenders.', intuition: 'Neighbor search returns semantically similar items.', trap: 'Cosine neighbors are not guaranteed truth.', lessonId: 'embeddings' }),
      ]),
    ],
  },
  {
    id: 'kv-cache',
    label: 'KV Cache',
    center: {
      short: 'The KV cache stores previously computed key and value vectors during autoregressive decoding so each new token only projects fresh K/V and attends over cached history.',
      intuition: 'Old tokens keep the same keys and values—only the new token adds rows; queries still attend over the full visible context.',
      formula: 'K_{cache}, V_{cache} \\leftarrow [K_{old}, K_t], [V_{old}, V_t]',
      why: 'KV caching is essential for fast LLM inference, cutting redundant computation from quadratic reprojection to incremental appends.',
      trap: 'The cache does not skip attention—it skips repeated K/V projection for previous tokens.',
    },
    branches: [
      branch('prerequisites', 'Prerequisites', 'prerequisite', [
        leaf('self-attention-kvc', 'Self-attention', { short: 'Attention mixes values using query-key scores.', intuition: 'Cache stores K and V from past steps.', trap: 'Confusing cache with skipping attention entirely.', lessonId: 'self-attention' }),
        leaf('token-generation-kvc', 'Token generation loop', { short: 'Generate one token, append, repeat.', intuition: 'Each step adds one new query position.', trap: 'Batch prefills differ from single-token decode steps.', lessonId: 'transformer-token-generation' }),
        leaf('qkv-projections-kvc', 'Q / K / V projections', { short: 'Linear maps create query, key, and value vectors per token.', intuition: 'Only the newest token needs new projections each decode step.', trap: 'Reprojecting all history every step wastes compute.' }),
        leaf('causal-mask-kvc', 'Causal mask', { short: 'Decoder queries may not attend to future tokens.', intuition: 'Cache grows with past tokens only.', trap: 'Bidirectional caches differ from autoregressive caches.', lessonId: 'attention-masks' }),
        leaf('multi-head-kvc', 'Multi-head attention', { short: 'Each head has its own K/V tensors in the cache.', intuition: 'Memory scales with num_heads × context × head_dim.', trap: 'Forgetting head dimension when estimating memory.' }),
        leaf('softmax-kvc', 'Softmax', { short: 'Current query scores all cached keys then mixes values.', intuition: 'Attention math is unchanged—inputs are cached.', trap: 'Assuming softmax disappears with caching.', lessonId: 'softmax' }),
      ]),
      branch('mechanism', 'Core mechanism', 'mechanism', [
        leaf('prefill-kvc', 'Prefill phase', { short: 'Process the prompt in parallel; fill cache with all prompt K/V.', intuition: 'Prefill is compute-heavy but runs once per sequence.', trap: 'Treating long prefill like many decode steps for latency planning.' }),
        leaf('decode-step-kvc', 'Decode step', { short: 'Append one token; compute Q,K,V for it; extend cache.', intuition: 'Attention reads full cached K/V with the new query row.', trap: 'Forgetting to update cache length metadata.' }),
        leaf('cache-append-kvc', 'Cache append', { short: 'K_cache = concat(K_cache, K_new); same for V.', intuition: 'Append-only structure suits incremental generation.', formula: 'K_{cache} \\leftarrow [K_{cache}, K_t]', trap: 'In-place bugs when batch slots differ in length.' }),
        leaf('attention-over-cache-kvc', 'Attention over cache', { short: 'New query attends over all cached keys up to current length.', intuition: 'Score matrix row length equals context so far.', trap: 'Windowed caches truncate history unless sliding logic is explicit.' }),
        leaf('per-layer-cache-kvc', 'Per-layer cache', { short: 'Each transformer layer maintains separate K/V storage.', intuition: 'Total memory ≈ layers × heads × context × dim.', trap: 'Counting only one layer when budgeting GPU memory.' }),
      ]),
      branch('intuitions', 'Intuitions', 'intuition', [
        leaf('reuse-kvc', 'Reuse past work', { short: 'Previous tokens’ K/V are frozen after their step.', intuition: 'Like keeping solved subproblems on a memo pad.', trap: 'Cache invalidation when model weights change mid-session.' }),
        leaf('memory-growth-kvc', 'Memory grows with context', { short: 'Longer conversations need larger caches.', intuition: 'Memory bandwidth can dominate at long context.', trap: 'Assuming constant memory per generated token.' }),
        leaf('latency-tradeoff-kvc', 'Latency tradeoff', { short: 'Less matmul work per step; more data to read for attention.', intuition: 'Very long contexts shift bottleneck to memory IO.', trap: 'Ignoring bandwidth when celebrating fewer FLOPs.' }),
        leaf('batch-padding-kvc', 'Batch padding', { short: 'Batched sequences cache to different lengths with padding masks.', intuition: 'Serving batches heterogeneous prompts needs careful slot management.', trap: 'Padding leaks into attention without masks.' }),
        leaf('window-intuition-kvc', 'Sliding window', { short: 'Some models cache only the last W tokens.', intuition: 'Bounded memory trades away full history.', trap: 'Assuming infinite context when window is W.' }),
      ]),
      branch('formula-code', 'Formula / Code', 'formula', [
        leaf('cache-update-formula-kvc', 'Cache update', { short: 'Append new K,V along sequence dimension each decode step.', formula: 'K_{cache} \\leftarrow \\mathrm{concat}(K_{cache}, K_t)', intuition: 'Same for V and for every layer/head.', trap: 'Wrong concat axis corrupts head layout.' }),
        leaf('memory-formula-kvc', 'Memory estimate', { short: 'Bytes ≈ 2 × layers × heads × context × head_dim × dtype_bytes.', intuition: 'Factor 2 for K and V.', trap: 'Forgetting GQA shares fewer KV heads.', lessonId: 'grouped-query-attention' }),
        leaf('decode-code-kvc', 'Decode sketch', { short: 'Forward new token with past_key_values; receive extended cache.', code: 'out, cache = model(token, past_key_values=cache)\nnext = sample(out.logits[:, -1])', trap: 'Not passing cache between steps recomputes everything.' }),
        leaf('gqa-reduction-kvc', 'GQA memory savings', { short: 'Grouped-query attention stores fewer KV heads than query heads.', intuition: 'Queries share KV heads to shrink cache.', trap: 'Using MHA memory formulas on GQA models.', lessonId: 'grouped-query-attention' }),
        leaf('flash-decode-kvc', 'Flash decode', { short: 'Fused kernels still read cache for attention over history.', intuition: 'Kernel fusion changes schedule, not cached semantics.', trap: 'Thinking Flash Attention removes KV storage.', lessonId: 'flash-attention' }),
      ]),
      branch('traps', 'Common traps', 'trap', [
        leaf('skip-attention-trap-kvc', '“Skips attention” trap', { short: 'Cache skips K/V recomputation, not the attention reduction.', intuition: 'Softmax over cached keys still runs.', trap: 'Marketing language that implies attention is avoided.' }),
        leaf('unbounded-context-trap-kvc', 'Unbounded context fantasy', { short: 'Cache memory and bandwidth scale linearly with length.', intuition: 'Long-context models need compression or windows.', trap: 'Promising infinite chat without memory cost.' }),
        leaf('dtype-trap-kvc', 'KV dtype', { short: 'Quantized KV (INT8/FP8) saves memory but can harm quality.', intuition: 'Calibration matters for low-bit caches.', trap: 'Aggressive quantization without perplexity checks.' }),
        leaf('cache-invalidation-trap-kvc', 'Cache invalidation', { short: 'Editing prompt mid-stream or swapping LoRA adapters invalidates cache.', intuition: 'K/V depend on weights and prefix tokens.', trap: 'Reusing cache after weight update.' }),
        leaf('prefill-decode-trap-kvc', 'Prefill vs decode confusion', { short: 'Latency models differ: prefill is parallel; decode is sequential.', intuition: 'TTFT vs TPOT measure different phases.', trap: 'Benchmarking only one phase.' }),
      ]),
      branch('used-later', 'Used later', 'application', [
        leaf('token-gen-kvc', 'Token generation', { short: 'Generation loop appends tokens and extends cache each step.', intuition: 'Sampling and cache updates interleave.', trap: 'Forgetting cache when implementing greedy decode.', lessonId: 'transformer-token-generation' }),
        leaf('gqa-kvc', 'Grouped-query attention', { short: 'GQA reduces KV footprint for long contexts.', intuition: 'Serving stacks combine GQA + cache quantization.', trap: 'Over-sharing KV heads hurts quality.', lessonId: 'grouped-query-attention' }),
        leaf('efficient-inference-kvc', 'Efficient inference track', { short: 'Batching, paging, and speculative decoding interact with cache layout.', intuition: 'Production serving optimizes cache slots.', trap: 'Prototype ignores multi-tenant cache fragmentation.', lessonId: 'efficient-inference-compression-track' }),
        leaf('flash-attention-kvc', 'Flash Attention', { short: 'IO-aware attention kernels pair with cached decode.', intuition: 'Training uses flash; inference uses cache-aware flash.', trap: 'Assuming one kernel fits all phases.', lessonId: 'flash-attention' }),
        leaf('transformer-kvc', 'Transformer', { short: 'KV cache is an inference optimization atop the transformer block.', intuition: 'Architecture unchanged—serving adds cache state.', trap: 'Training code paths without cache awareness.', lessonId: 'transformer' }),
      ]),
    ],
  },
  {
    id: 'layer-normalization',
    label: 'Layer Normalization',
    center: {
      short: 'LayerNorm normalizes each token’s feature vector to zero mean and unit variance across features, then applies learned scale γ and shift β per dimension.',
      intuition: 'Stabilize activations inside one example before the next sublayer—each token is normalized on its own feature axis.',
      formula: '\\mathrm{LN}(x)=\\gamma\\odot\\frac{x-\\mu_x}{\\sqrt{\\sigma_x^2+\\epsilon}}+\\beta',
      why: 'LayerNorm stabilizes transformer training and inference, pairing with residual streams in GPT, BERT, and diffusion transformers.',
      trap: 'LayerNorm is not BatchNorm—it does not use statistics from other examples in the batch.',
    },
    branches: [
      branch('prerequisites', 'Prerequisites', 'prerequisite', [
        leaf('vector-ln', 'Vector', { short: 'Each token hidden state is a vector in ℝ^d.', intuition: 'Normalization runs across the d features of one token.', trap: 'Normalizing the wrong axis mixes tokens.' }),
        leaf('mean-variance-ln', 'Mean and variance', { short: 'Compute μ and σ² across features of one token.', intuition: 'Subtract mean; divide by std.', trap: 'Using batch statistics instead of token statistics.' }),
        leaf('residual-ln', 'Residual stream', { short: 'Transformers add sublayer outputs into a running hidden state.', intuition: 'Norm often wraps sublayers to control scale.', trap: 'Uncontrolled residual growth destabilizes deep stacks.', lessonId: 'residual-stream' }),
        leaf('affine-ln', 'Affine parameters', { short: 'Learnable γ and β restore representational capacity after normalization.', intuition: 'Without γ,β the layer could not recover useful scale.', trap: 'Assuming normalization always shrinks signal permanently.' }),
        leaf('transformer-ln', 'Transformer block', { short: 'Attention and MLP sublayers repeat with norms between them.', intuition: 'Pre-LN vs post-LN changes gradient paths.', trap: 'Swapping placement without retuning learning rate.', lessonId: 'transformer' }),
        leaf('epsilon-ln', 'Epsilon stabilizer', { short: 'Small ε in denominator prevents divide-by-zero.', intuition: 'Numerical guard in sqrt(σ² + ε).', trap: 'ε too large flattens activations excessively.' }),
      ]),
      branch('mechanism', 'Core mechanism', 'mechanism', [
        leaf('compute-stats-ln', 'Compute μ, σ per token', { short: 'Aggregate mean and variance across feature dimension d.', intuition: 'Each token gets its own normalization stats.', trap: 'Accidentally normalizing across batch dimension.' }),
        leaf('standardize-ln', 'Standardize', { short: 'x_hat = (x - μ) / sqrt(σ² + ε).', intuition: 'Centers and scales the token vector.', formula: '\\hat{x}=\\frac{x-\\mu_x}{\\sqrt{\\sigma_x^2+\\epsilon}}', trap: 'Forgetting ε when variance is tiny.' }),
        leaf('affine-transform-ln', 'Apply γ and β', { short: 'Output = γ ⊙ x_hat + β elementwise.', intuition: 'Model learns useful scale and shift after norm.', trap: 'Initializing γ to zero kills signal.' }),
        leaf('placement-ln', 'Pre-LN vs post-LN', { short: 'Norm before or after sublayer inside the residual path.', intuition: 'Pre-LN often trains deeper models more easily.', trap: 'Copying post-LN recipes onto pre-LN stacks without adjustment.' }),
        leaf('inference-ln', 'Inference behavior', { short: 'LayerNorm uses only the current token’s features—no batch stats.', intuition: 'Train and eval formulas match.', trap: 'Expecting running mean buffers like BatchNorm.' }),
      ]),
      branch('intuitions', 'Intuitions', 'intuition', [
        leaf('per-token-ln', 'Per-token thermostat', { short: 'Each token adjusts its own temperature before the next transform.', intuition: 'Prevents one token’s features from exploding.', trap: 'Thinking neighbors in the batch affect this token’s norm.' }),
        leaf('not-batch-ln', 'Not batch normalization', { short: 'BatchNorm normalizes across batch for each feature separately.', intuition: 'LayerNorm swaps the normalized axis.', trap: 'Dropping BatchNorm running stats logic into LayerNorm code.' }),
        leaf('scale-control-ln', 'Scale control in residuals', { short: 'Norm keeps residual additions from drifting magnitude.', intuition: 'Deep stacks accumulate writes; norm recenters.', trap: 'Removing norm in very deep models without replacement.' }),
        leaf('shift-invariance-ln', 'Shift sensitivity', { short: 'Adding constant to all features changes μ and cancels in x_hat.', intuition: 'LayerNorm removes shared offset across features.', trap: 'Expecting norm to fix bad input scaling from upstream.' }),
        leaf('rms-norm-ln', 'RMSNorm cousin', { short: 'Some models skip mean centering and scale by RMS only.', intuition: 'Simpler norm still controls magnitude.', trap: 'Assuming all transformers use full LayerNorm.' }),
      ]),
      branch('formula-code', 'Formula / Code', 'formula', [
        leaf('ln-formula-ln', 'LayerNorm formula', { short: 'LN(x) = γ * (x-μ)/sqrt(σ²+ε) + β.', formula: '\\mathrm{LN}(x)=\\gamma\\frac{x-\\mu_x}{\\sqrt{\\sigma_x^2+\\epsilon}}+\\beta', intuition: 'μ,σ computed over feature dim.', trap: 'Applying over sequence length by mistake.' }),
        leaf('pytorch-ln', 'PyTorch LayerNorm', { short: 'nn.LayerNorm(normalized_shape=d_model).', code: 'import torch.nn as nn\nln = nn.LayerNorm(d_model)\ny = ln(x)  # x shape [batch, seq, d_model]', trap: 'normalized_shape must match last dims.' }),
        leaf('pre-ln-code-ln', 'Pre-LN block', { short: 'x = x + attn(ln(x)); x = x + mlp(ln(x)).', intuition: 'Norm inside residual branch is common in GPT-style stacks.', trap: 'Post-LN code copied verbatim.' }),
        leaf('stats-dim-ln', 'Axis choice', { short: 'Normalize over hidden dimension, not batch or sequence.', intuition: 'Shape [B,T,D] → stats over D.', trap: 'Normalizing over T mixes token information.' }),
        leaf('gamma-init-ln', 'γ initialization', { short: 'Often initialize γ to 1 and β to 0.', intuition: 'Starts near identity then learns deviation.', trap: 'Zero γ initialization blocks learning.' }),
      ]),
      branch('traps', 'Common traps', 'trap', [
        leaf('batchnorm-confusion-ln', 'BatchNorm confusion', { short: 'BatchNorm needs batch statistics; LayerNorm does not.', intuition: 'Small batch BatchNorm is noisy; LayerNorm is batch-size agnostic.', trap: 'Using BatchNorm in variable-length NLP without care.', lessonId: 'dropout-batchnorm' }),
        leaf('wrong-axis-ln', 'Wrong axis', { short: 'Normalizing over sequence mixes tokens illegally.', intuition: 'Each token must keep private stats.', trap: 'LayerNorm over time dimension in transformers.' }),
        leaf('eval-mode-ln', 'Eval mode myth', { short: 'LayerNorm has no running stats to freeze unlike BatchNorm.', intuition: 'Dropout still changes train vs eval.', trap: 'Searching for running_mean in LayerNorm modules.' }),
        leaf('over-normalizing-ln', 'Over-normalizing', { short: 'Too many norm layers without need can slow learning.', intuition: 'Architecture recipes balance depth and norm placement.', trap: 'Adding norm everywhere as a generic fix.' }),
        leaf('scale-upstream-ln', 'Upstream scale', { short: 'LayerNorm does not replace sensible weight init or LR.', intuition: 'Bad initialization still hurts before first norm.', trap: 'Assuming norm alone prevents all explosion.', lessonId: 'initialization' }),
      ]),
      branch('used-later', 'Used later', 'application', [
        leaf('transformer-used-ln', 'Transformer', { short: 'Norm is part of every standard transformer block.', intuition: 'Works with attention and MLP residuals.', trap: 'Removing norm without alternative stabilization.', lessonId: 'transformer' }),
        leaf('gpt2-ln', 'GPT-2', { short: 'Decoder-only stacks rely on pre-LN in modern variants.', intuition: 'Generation stability depends on norm placement.', trap: 'Confusing legacy post-LN GPT with current recipes.', lessonId: 'gpt2-comprehensive' }),
        leaf('dropout-bn-ln', 'Dropout vs BatchNorm', { short: 'Regularization and batch norm behave differently from LayerNorm.', intuition: 'Training mode toggles affect dropout, not LN stats.', trap: 'Applying BatchNorm train/eval rules to LayerNorm.', lessonId: 'dropout-batchnorm' }),
        leaf('dit-ln', 'DiT', { short: 'Diffusion transformers use adaptive norm conditioned on timestep.', intuition: 'Conditioning injects t into scale/shift.', trap: 'Plain LayerNorm code on adaptive norm blocks.', lessonId: 'dit' }),
        leaf('training-dynamics-ln', 'Training loop dynamics', { short: 'Norm interacts with LR and warmup in deep training.', intuition: 'Watch activation scale across layers.', trap: 'Huge LR with post-LN can still diverge.', lessonId: 'training-loop-dynamics' }),
      ]),
    ],
  },
];
