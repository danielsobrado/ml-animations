---
title: "Embeddings - representing stuff as vectors"
date: 2024-11-23
draft: false
tags: ["embeddings", "vectors", "representation-learning"]
categories: ["NLP Fundamentals"]
---

Embeddings are everywhere now. Words, sentences, images, users, products... anything can be an embedding. But what are they actually?

## The basic idea

Convert discrete things into continuous vectors where similar things are close together.

Cat → [0.2, -0.5, 0.8, ...]
Dog → [0.3, -0.4, 0.7, ...]
Car → [-0.8, 0.3, -0.2, ...]

Cat and dog vectors are close. Car is far from both. That's the goal.

![Embedding Space](https://danielsobrado.github.io/ml-animations/animation/embeddings)

Explore: [Embeddings Animation](https://danielsobrado.github.io/ml-animations/animation/embeddings)

## Why not one-hot?

One-hot encoding for words:
```
cat = [1, 0, 0, 0, ...]  # 10000 dims for 10000 words
dog = [0, 1, 0, 0, ...]
car = [0, 0, 1, 0, ...]
```

Problems:
- Huge dimensionality
- All words equally distant
- No semantic information
- Sparse (wasteful)

Embeddings fix all of these.

## How dimensions get meaning

Each dimension doesn't have predefined meaning. The model learns useful dimensions during training.

Hypothetically:
- Dim 1 might encode "animal-ness"
- Dim 2 might encode "size"
- Dim 3 might encode "danger level"

But usually dimensions are entangled and not interpretable.

## Word embeddings

The first big success. Word2Vec, GloVe, FastText.

Trained on "words appearing in similar contexts have similar meanings."

```python
from gensim.models import Word2Vec

# similar words have similar vectors
model.wv.most_similar('king')
# [('queen', 0.8), ('prince', 0.7), ...]
```

Limitation: one vector per word. "Bank" (river) and "bank" (financial) get same vector.

## Sentence embeddings

Words are nice but you often need sentence or document level.

**Simple approach:** average word embeddings
```python
sentence_vec = np.mean([word_vec(w) for w in sentence])
```

Works okayish but loses word order.

**Better:** models trained for sentence similarity
- Sentence-BERT
- Universal Sentence Encoder
- E5, BGE (recent and good)

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(['This is sentence one', 'Another sentence'])
```

## Contextual embeddings

BERT and friends give different vectors based on context.

"I sat by the river bank" → bank_vector_1
"I went to the bank to deposit money" → bank_vector_2

Different vectors! Context matters.

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# each token gets context-dependent vector
outputs = model(**tokenizer("hello world", return_tensors='pt'))
embeddings = outputs.last_hidden_state  # [1, seq_len, 768]
```

## Image embeddings

CNN or Vision Transformer extracts features. Last layer before classification head = image embedding.

```python
from torchvision.models import resnet50

model = resnet50(pretrained=True)
# remove classification head
model = torch.nn.Sequential(*list(model.children())[:-1])

# image → 2048-dim vector
embedding = model(image).squeeze()
```

Or use CLIP for multi-modal embeddings (images and text in same space).

## Using embeddings

**Similarity search**

Find nearest neighbors in embedding space.

```python
from sklearn.metrics.pairwise import cosine_similarity

# find most similar to query
similarities = cosine_similarity([query_emb], all_embeddings)
top_k = np.argsort(similarities[0])[-k:]
```

**Clustering**

Group similar items.

```python
from sklearn.cluster import KMeans

clusters = KMeans(n_clusters=10).fit_predict(embeddings)
```

**Classification**

Embeddings as features for classifier.

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(embeddings, labels)
```

**Visualization**

Reduce to 2D/3D for plotting.

```python
from sklearn.manifold import TSNE

reduced = TSNE(n_components=2).fit_transform(embeddings)
plt.scatter(reduced[:, 0], reduced[:, 1])
```

## Embedding dimension

More dimensions = more capacity but:
- More compute
- More data needed
- Diminishing returns

Common choices:
- Word embeddings: 100-300
- Sentence embeddings: 384-768
- Image embeddings: 512-2048

## The curse of dimensionality

In high dimensions, everything is roughly equidistant. Distances become less meaningful.

Mitigation:
- Dimensionality reduction (PCA)
- Use cosine similarity (normalizes for magnitude)
- Careful about distance thresholds

## Training your own

When to train custom embeddings:
- Domain-specific vocabulary
- Languages not covered by pretrained
- Specific similarity notion needed

When to use pretrained:
- General purpose
- Limited data
- Quick start needed

Fine-tuning middle ground: start pretrained, adapt to your domain.

Explore how embeddings organize in space: [Embeddings Animation](https://danielsobrado.github.io/ml-animations/animation/embeddings)

---

Related:
- [Word2Vec](/posts/word2vec/)
- [Cosine Similarity](/posts/cosine-similarity/)
- [RAG uses embeddings](/posts/rag/)
