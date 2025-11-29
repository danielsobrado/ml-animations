---
title: "FastText - handling words not in vocabulary"
date: 2024-11-24
draft: false
tags: ["fasttext", "embeddings", "nlp", "subword", "oov"]
categories: ["NLP Fundamentals"]
---

Word2Vec problem: what about misspellings? What about "unhappiness" when you only trained on "happy"? What about German compound words?

FastText fixes this by using subword information. Facebook AI released it in 2016.

## The core idea

Instead of one vector per word, break words into character n-grams.

"where" with n=3:
- "<wh", "whe", "her", "ere", "re>"
- Plus the word itself: "<where>"

Word vector = sum of all its n-gram vectors.

```python
def get_ngrams(word, min_n=3, max_n=6):
    word = '<' + word + '>'  # boundary markers
    ngrams = []
    for n in range(min_n, max_n + 1):
        for i in range(len(word) - n + 1):
            ngrams.append(word[i:i+n])
    return ngrams

get_ngrams('cat')
# ['<ca', 'cat', 'at>', '<cat', 'cat>', '<cat>']
```

![FastText Subwords](https://danielsobrado.github.io/ml-animations/animation/fasttext)

Interactive demo: [FastText Animation](https://danielsobrado.github.io/ml-animations/animation/fasttext)

## Why this works for OOV

Never seen "unhappyness" (misspelled)?

Break it into n-grams. Some of those n-grams appeared in:
- "happy"
- "unhappy"  
- "happiness"
- "sadness"

The vector is constructed from n-grams the model has seen. Not perfect but way better than nothing.

## Architecture

Same as Word2Vec (Skip-gram or CBOW) but:
- Word represented as sum of n-gram vectors
- More parameters (n-gram vocabulary larger than word vocabulary)
- Uses hashing to limit n-gram vocabulary size

```python
def word_vector(word, ngram_vectors):
    ngrams = get_ngrams(word)
    # hash ngrams to fixed vocabulary
    ngram_ids = [hash(ng) % bucket_size for ng in ngrams]
    return sum(ngram_vectors[id] for id in ngram_ids)
```

## Training

Similar to Word2Vec negative sampling:

```python
# pseudo-code
for center, context in training_pairs:
    center_vec = word_vector(center, ngram_vectors)
    context_vec = word_vector(context, ngram_vectors)
    
    # positive sample
    loss = -log(sigmoid(dot(center_vec, context_vec)))
    
    # negative samples
    for neg in negative_samples:
        neg_vec = word_vector(neg, ngram_vectors)
        loss += -log(sigmoid(-dot(center_vec, neg_vec)))
    
    # backprop through n-gram vectors
    update_ngram_vectors(loss)
```

## Practical advantages

**Morphologically rich languages**

Finnish, Turkish, German... words have many forms. FastText handles this naturally because related forms share n-grams.

"playing", "played", "plays" all share "play" n-grams.

**Typos and variations**

"learning", "leanring", "lerning" will have similar vectors.

**Rare words**

Word appearing once? In Word2Vec, vector is garbage. In FastText, n-grams have been seen in other words.

## The downsides

- More parameters to store
- Slower training (more vectors to update per word)
- Hash collisions can hurt quality
- Very short words have few n-grams

## Using FastText

Official library:
```python
import fasttext

# train
model = fasttext.train_unsupervised(
    'data.txt',
    model='skipgram',  # or 'cbow'
    dim=100,
    minn=3,  # min n-gram
    maxn=6,  # max n-gram
)

# get vector (works for any word!)
vec = model.get_word_vector('somemadeupword')

# similar words
model.get_nearest_neighbors('cat')
```

With Gensim:
```python
from gensim.models import FastText

model = FastText(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    min_n=3,
    max_n=6,
)
```

## Pretrained vectors

FastText released vectors for 157 languages. Trained on Wikipedia + Common Crawl.

Very useful for low-resource languages where training data is scarce.

## FastText for classification

FastText also has supervised mode for text classification. Very fast, surprisingly good.

```python
# train classifier
model = fasttext.train_supervised('train.txt')

# predict
model.predict("this is a test sentence")
```

Training file format:
```
__label__positive this movie was great
__label__negative terrible waste of time
```

## Comparing approaches

| Feature | Word2Vec | GloVe | FastText |
|---------|----------|-------|----------|
| OOV handling | No | No | Yes |
| Subword info | No | No | Yes |
| Speed | Fast | Medium | Slower |
| Memory | Low | High (matrix) | Medium |
| Incremental | Yes | No | Yes |

## When to use FastText

- Noisy text (social media, user input)
- Morphologically rich languages
- Small training data
- OOV words expected in production

When Word2Vec/GloVe might be enough:
- Clean text
- English or similar
- Large vocabulary coverage
- Speed critical

See the subword mechanism in action: [FastText Animation](https://danielsobrado.github.io/ml-animations/animation/fasttext)

---

Related:
- [Word2Vec Basics](/posts/word2vec/)
- [GloVe Approach](/posts/glove/)
- [Embeddings Overview](/posts/embeddings/)
