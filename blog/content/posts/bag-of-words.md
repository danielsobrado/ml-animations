---
title: "Bag of Words - the simplest text representation"
date: 2024-11-22
draft: false
tags: ["bag-of-words", "bow", "nlp", "text-representation", "tfidf"]
categories: ["NLP Fundamentals"]
---

Before embeddings there was Bag of Words. Still useful, still relevant for some tasks. And understanding it helps understand why newer methods are better.

## What is it?

Represent document as word counts. Ignore order completely.

"The cat sat on the mat"
"The dog sat on the log"

Vocabulary: [the, cat, sat, on, mat, dog, log]

Document 1: [2, 1, 1, 1, 1, 0, 0]
Document 2: [2, 0, 1, 1, 0, 1, 1]

That's it. Count each word.

![Bag of Words Process](https://danielsobrado.github.io/ml-animations/animation/bag-of-words)

See it visualized: [Bag of Words Animation](https://danielsobrado.github.io/ml-animations/animation/bag-of-words)

## Building it

```python
from collections import Counter

def bag_of_words(documents):
    # build vocabulary
    vocab = set()
    for doc in documents:
        vocab.update(doc.split())
    vocab = sorted(vocab)
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    
    # vectorize
    vectors = []
    for doc in documents:
        counts = Counter(doc.split())
        vec = [counts.get(w, 0) for w in vocab]
        vectors.append(vec)
    
    return vectors, vocab
```

Or just use sklearn:
```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)
```

## The problems

**Ignores word order**

"Dog bites man" and "Man bites dog" have identical BoW vectors. Completely different meaning.

**Sparse and high dimensional**

10,000 word vocabulary = 10,000 dim vectors. Mostly zeros.

**No semantic similarity**

"Happy" and "joyful" are as distant as "happy" and "angry". No meaning captured.

**Common words dominate**

"The", "is", "a" appear everywhere. Don't help distinguish documents.

## TF-IDF to the rescue

Term Frequency - Inverse Document Frequency

Weight words by:
- How often they appear in this document (TF)
- How rare they are across all documents (IDF)

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

$$\text{IDF}(t) = \log\frac{N}{|\{d : t \in d\}|}$$

Words appearing in every document get low weight. Rare, distinctive words get high weight.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)
```

## N-grams

Capture some word order by including consecutive word pairs (bigrams), triples (trigrams).

"The cat sat" with bigrams:
- Unigrams: [the, cat, sat]
- Bigrams: [the_cat, cat_sat]

```python
vectorizer = CountVectorizer(ngram_range=(1, 2))
```

Vocabulary explodes but captures more structure.

## When BoW still works

- Document classification (news categories, spam)
- Search and information retrieval (with TF-IDF)
- Baseline for comparison
- When you need interpretability
- Small datasets

## When it fails

- Sentiment analysis (word order matters)
- Question answering
- Anything requiring understanding
- Short texts (not enough words)

## Preprocessing matters

BoW benefits from:
- Lowercasing
- Removing punctuation
- Stop word removal
- Stemming/lemmatization

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    max_features=5000,
    ngram_range=(1, 2)
)
```

## Comparison with embeddings

| Aspect | BoW/TF-IDF | Embeddings |
|--------|------------|------------|
| Semantic similarity | No | Yes |
| Word order | No (partial with n-grams) | Yes |
| Dimensionality | High (vocab size) | Low (100-768) |
| Interpretable | Yes | No |
| Training data needed | None | Lots |
| Compute | Fast | Slower |

## Practical advice

Starting new NLP project?

1. Try TF-IDF first (baseline)
2. If not good enough, try sentence embeddings
3. If still not enough, fine-tune BERT

Surprised how often TF-IDF is "good enough" for classification tasks.

The animation shows how documents become vectors: [Bag of Words Animation](https://danielsobrado.github.io/ml-animations/animation/bag-of-words)

---

Related:
- [Embeddings - better representations](/posts/embeddings/)
- [Tokenization](/posts/tokenization/)
