---
title: "Tokenization - how text becomes numbers"
date: 2024-11-21
draft: false
tags: ["tokenization", "nlp", "bpe", "wordpiece", "sentencepiece"]
categories: ["NLP Fundamentals"]
---

ML models don't read text. They need numbers. Tokenization bridges that gap. Seems simple but lots of nuance.

## The basic question

How do you split "Don't tokenize carelessly!"?

Options:
- Character level: ["D", "o", "n", "'", "t", ...]
- Word level: ["Don't", "tokenize", "carelessly", "!"]
- Subword: ["Don", "'t", "token", "ize", "care", "less", "ly", "!"]

Each has tradeoffs.

![Tokenization Methods](https://danielsobrado.github.io/ml-animations/animation/tokenization)

See different methods: [Tokenization Animation](https://danielsobrado.github.io/ml-animations/animation/tokenization)

## Character-level

Every character is a token.

Pros:
- Small vocabulary (just alphabet + symbols)
- No OOV (out of vocabulary) words
- Works for any language

Cons:
- Very long sequences
- Harder to learn word-level meaning
- More compute needed

Rarely used alone now.

## Word-level

Split on whitespace and punctuation.

```python
text = "Hello, world!"
tokens = text.split()  # ["Hello,", "world!"]
# or with regex
import re
tokens = re.findall(r'\w+|[^\w\s]', text)  # ["Hello", ",", "world", "!"]
```

Pros:
- Intuitive
- Short sequences
- Each token meaningful

Cons:
- Huge vocabulary (every word form)
- OOV problems ("unbelievable" not in vocab?)
- Morphology issues (run, runs, running = 3 tokens)

## Subword tokenization

The sweet spot. Break unknown words into known pieces.

"unhappiness" → ["un", "happi", "ness"]

Model can understand new words from known components.

### BPE - Byte Pair Encoding

Start with characters. Repeatedly merge most frequent pairs.

```
Corpus: "low low low lower lowest"

Initial: l o w </w> l o w </w> l o w </w> l o w e r </w> l o w e s t </w>

Iteration 1: merge "l o" → "lo"
lo w </w> lo w </w> lo w </w> lo w e r </w> lo w e s t </w>

Iteration 2: merge "lo w" → "low"
low </w> low </w> low </w> low e r </w> low e s t </w>

... continue until vocabulary size reached
```

GPT-2/3/4 use BPE.

### WordPiece

Similar to BPE but uses likelihood instead of frequency.

Merge pair that maximizes:
$$\frac{P(ab)}{P(a)P(b)}$$

BERT uses WordPiece. Tokens start with ## if not word start.

"unhappiness" → ["un", "##happi", "##ness"]

### Unigram / SentencePiece

Start with large vocabulary. Remove tokens that hurt likelihood least.

SentencePiece treats spaces as characters (▁). Language agnostic.

"hello world" → ["▁hello", "▁world"]

Used by T5, LLaMA.

## In practice

Using transformers library:

```python
from transformers import AutoTokenizer

# BERT (WordPiece)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
tokens = tokenizer.tokenize("unhappiness")
# ['un', '##happiness']

# GPT-2 (BPE)
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokens = tokenizer.tokenize("unhappiness")
# ['un', 'happiness']

# T5 (SentencePiece)
tokenizer = AutoTokenizer.from_pretrained('t5-base')
tokens = tokenizer.tokenize("unhappiness")
# ['▁un', 'happiness']
```

## Vocabulary size

Bigger vocab:
- Shorter sequences
- Each token more meaningful
- More parameters in embedding layer

Smaller vocab:
- Longer sequences
- Better generalization
- Less memory

Common sizes:
- BERT: 30,000
- GPT-2: 50,257
- LLaMA: 32,000

## Token counting matters

APIs charge per token. Context windows limited by tokens.

```python
# count tokens
tokens = tokenizer.encode(text)
num_tokens = len(tokens)
```

Rule of thumb for English: 1 token ≈ 4 characters, 100 tokens ≈ 75 words

But varies! Code and non-English can be token-expensive.

## Special tokens

Every tokenizer has special tokens:
- `[CLS]`, `[SEP]` (BERT)
- `<s>`, `</s>` (many models)
- `<pad>` for padding
- `<unk>` for unknown (rare with subword)
- `<mask>` for MLM training

```python
tokenizer.special_tokens_map
# {'unk_token': '[UNK]', 'sep_token': '[SEP]', ...}
```

## Common gotchas

**Whitespace handling**

Different tokenizers handle spaces differently. Some strip, some include.

**Case sensitivity**

"Hello" vs "hello" - uncased models lowercase everything.

**Numbers**

"123456" might be one token or many. Check!

**Non-English**

Some tokenizers are English-centric. Chinese, Arabic, etc. might get poor tokenization.

## Building custom tokenizer

For domain-specific text:

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

trainer = trainers.BpeTrainer(vocab_size=10000)
tokenizer.train(files=['data.txt'], trainer=trainer)
```

Worth it for specialized domains (code, biology, legal).

See tokenization methods compared: [Tokenization Animation](https://danielsobrado.github.io/ml-animations/animation/tokenization)

---

Related:
- [BERT Tokenization](/posts/bert/)
- [Embeddings from tokens](/posts/embeddings/)
