---
title: "BERT explained - bidirectional transformers"
date: 2024-11-28
draft: false
tags: ["bert", "transformers", "nlp", "pre-training"]
categories: ["Machine Learning"]
---

BERT changed everything in NLP when it came out in 2018. Suddenly fine-tuning pretrained models became the default approach. Let me explain what makes it special.

## What's different about BERT

Unlike GPT which reads left-to-right, BERT reads in both directions. Bidirectional.

Consider: "The bank by the river was steep"

Left-to-right model sees "bank" before knowing about "river". Has to guess meaning.

BERT sees full context. "River" helps understand "bank" means riverbank, not financial institution.

![BERT Demo](https://danielsobrado.github.io/ml-animations/animation/bert)

Interactive demo: [BERT Animation](https://danielsobrado.github.io/ml-animations/animation/bert)

## Architecture

BERT uses only the encoder part of transformer. No decoder.

Two sizes released:
- BERT-Base: 12 layers, 768 hidden, 12 heads, 110M params
- BERT-Large: 24 layers, 1024 hidden, 16 heads, 340M params

Input format is specific. Uses special tokens:
- [CLS] at start - used for classification
- [SEP] to separate sentences
- Segment embeddings to distinguish sentence A vs B

## Pre-training objectives

This is the clever part. Two tasks:

**Masked Language Modeling (MLM):**
Randomly mask 15% of tokens. Model predicts masked tokens using context from both directions.

"The [MASK] sat on the mat" → predict "cat"

Unlike autoregressive models, this forces bidirectional understanding.

**Next Sentence Prediction (NSP):**
Given two sentences, predict if B follows A in original text.

This helps with tasks needing sentence-pair understanding like QA.

btw the NSP objective is somewhat controversial. Later work (RoBERTa) showed it might not be that helpful.

## Why MLM works

The masking strategy is interesting:
- 80% of time: replace with [MASK] token
- 10% of time: replace with random token
- 10% of time: keep original

Why not always mask? Because [MASK] never appears in fine-tuning. The random replacement helps model be robust.

## Fine-tuning BERT

Pre-trained BERT is general language understanding. Fine-tune for specific task.

**Classification:** Use [CLS] token representation, add classifier head

**Token classification (NER):** Use each token's representation, classify each

**Question answering:** Input [CLS] question [SEP] passage [SEP], predict answer span

Fine-tuning typically takes just few epochs. 2-4 epochs with small learning rate (2e-5 to 5e-5).

## The visualization shows

Check out the interactive BERT demo:

- Tokenization including special tokens
- Bidirectional attention patterns  
- How [CLS] aggregates information
- Masking process visualization

[BERT Animation](https://danielsobrado.github.io/ml-animations/animation/bert)

## Practical tips from my experience

Learning rates matter a lot. Too high and you lose pre-trained knowledge. Too low and fine-tuning takes forever.

Batch size affects performance. Larger batches often help but need gradient accumulation if VRAM limited.

The [CLS] token representation isn't always best for similarity tasks. Sometimes mean pooling all tokens works better.

## BERT variants

Many improvements since original:

- **RoBERTa**: More data, longer training, no NSP
- **ALBERT**: Parameter sharing for efficiency
- **DistilBERT**: Smaller, faster, 97% performance
- **ELECTRA**: Different pre-training objective

For production, DistilBERT is often good enough and much faster.

## Limitations

Max 512 tokens. Longer documents need chunking or different architecture.

Fixed vocabulary. Novel words get broken into subwords which can hurt performance.

Pre-training is expensive. Fine-tuning is cheap but still needs labeled data.

---

Related posts:
- [Transformer Architecture](/posts/transformer-architecture/)
- [Fine-Tuning Guide](/posts/fine-tuning/)
- [Tokenization Explained](/posts/tokenization/)
