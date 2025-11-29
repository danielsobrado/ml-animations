---
title: "Word2Vec - where word embeddings started getting good"
date: 2024-11-26
draft: false
tags: ["word2vec", "embeddings", "nlp", "skip-gram", "cbow"]
categories: ["NLP Fundamentals"]
---

Before Word2Vec, representing words for ML was rough. One-hot encoding, TF-IDF... none of them captured meaning. Then Mikolov and team at Google published Word2Vec in 2013 and everything changed.

## The big idea

Words that appear in similar contexts have similar meanings. "Dog" and "cat" both appear near "pet", "cute", "fur". So their vectors should be close.

Train a simple neural network on a prediction task. The "side effect" is that word vectors learn to encode meaning.

![Word2Vec Training](https://danielsobrado.github.io/ml-animations/animation/word2vec)

See it in action: [Word2Vec Animation](https://danielsobrado.github.io/ml-animations/animation/word2vec)

## Two flavors

**Skip-gram:** Given center word, predict context words

"The cat sat on the mat"
- Center: "sat"
- Predict: "The", "cat", "on", "the"

**CBOW (Continuous Bag of Words):** Given context, predict center

- Context: "The", "cat", "on", "the"  
- Predict: "sat"

Skip-gram works better for smaller datasets and rare words. CBOW is faster and works well with frequent words.

## The architecture

Surprisingly simple:
- Input layer: one-hot encoded word
- Hidden layer: embedding dimension (typically 100-300)
- Output layer: vocabulary size, softmax

```
Input (V) → Hidden (D) → Output (V)
```

Where V = vocabulary size, D = embedding dimension

The magic happens in the hidden layer. Those weights become your word vectors.

## Training objective

For Skip-gram:

$$P(w_c | w_t) = \frac{\exp(v'_{w_c} \cdot v_{w_t})}{\sum_{w \in V} \exp(v'_w \cdot v_{w_t})}$$

Maximize probability of context words given center word.

That denominator sums over entire vocabulary. That's expensive.

## Making it fast

**Negative sampling**

Don't compute full softmax. Instead:
- Real context pairs: positive examples
- Random word pairs: negative examples

Only update weights for these few words per example.

```python
# pseudo-code
for center, context in training_data:
    loss = -log(sigmoid(dot(center_vec, context_vec)))
    # add k negative samples
    for neg_word in sample_negatives(k):
        loss += -log(sigmoid(-dot(center_vec, neg_vec)))
```

Much faster. Quality nearly as good.

**Hierarchical softmax**

Arrange vocabulary as binary tree. Prediction becomes path from root to word. Log(V) operations instead of V.

Less common now but was important historically.

## The famous analogies

"king - man + woman = queen"

This actually works (roughly). Vectors capture semantic relationships.

```python
# assuming we have word vectors
result = model['king'] - model['man'] + model['woman']
most_similar = find_nearest(result)  # returns 'queen'
```

Doesn't work perfectly. "king - man + woman" might return "king" itself if you don't filter. And many analogies fail. But the fact it works at all was surprising.

## What vectors learn

Word2Vec captures:
- Semantic similarity (dog ≈ cat)
- Syntactic patterns (walked ≈ talked in certain directions)
- Some analogies (bigger/smaller, country/capital)

But also:
- Dataset biases
- Frequency effects
- Only local context (window of ~5 words)

## Practical considerations

**Window size**

Smaller (2-5): captures syntactic similarity
Larger (5-10): captures topic/semantic similarity

**Embedding dimension**

More dimensions = more capacity but slower and needs more data
Common: 100-300 for most applications

**Minimum count**

Words appearing < N times get filtered. Rare words don't have enough context to learn good vectors.

**Subword info**

Original Word2Vec treats each word atomic. OOV words get no vector. FastText improved this by using character n-grams.

## Training your own

```python
from gensim.models import Word2Vec

sentences = [["the", "cat", "sat"], ["the", "dog", "ran"]]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1  # skip-gram
)

# get vector
cat_vec = model.wv['cat']

# find similar
model.wv.most_similar('cat')
```

Or use pretrained. GoogleNews vectors trained on 100B words, 3M vocabulary.

## Limitations

- No context: "bank" has same vector whether river or financial
- Only local window: misses document-level meaning
- No sentence/document embeddings natively
- Biases from training data

BERT and contextual embeddings solved the first issue. But Word2Vec is still useful for many applications.

Interactive exploration: [Word2Vec Animation](https://danielsobrado.github.io/ml-animations/animation/word2vec)

---

Related:
- [GloVe - Different Approach](/posts/glove/)
- [FastText - Adding Subwords](/posts/fasttext/)
- [Embeddings Overview](/posts/embeddings/)
