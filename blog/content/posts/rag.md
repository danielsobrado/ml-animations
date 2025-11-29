---
title: "RAG explained - Retrieval Augmented Generation"
date: 2024-11-27
draft: false
tags: ["rag", "retrieval", "llm", "embeddings", "vector-search"]
categories: ["Machine Learning"]
---

LLMs know lots of stuff. But they don't know YOUR stuff. And they make things up sometimes. RAG helps with both problems.

## The problem with vanilla LLMs

Ask GPT about your company's internal policy. It will confidently tell you something... that's completely wrong. It doesn't have access to your docs.

Training or fine-tuning on your data? Expensive. Slow to update. Data might leak into model weights.

RAG is simpler: just give the model relevant context at inference time.

![RAG Pipeline](https://danielsobrado.github.io/ml-animations/animation/rag)

Explore the pipeline: [RAG Animation](https://danielsobrado.github.io/ml-animations/animation/rag)

## How it works

1. **Chunk your documents** - split into manageable pieces
2. **Embed chunks** - convert to vectors with embedding model
3. **Store in vector DB** - for fast similarity search
4. **At query time:**
   - Embed the question
   - Find similar chunks
   - Include them in LLM prompt
   - LLM answers using that context

Simple idea. The devil is in the details.

## Chunking strategies

Most basic: fixed size chunks with overlap

```python
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks
```

Better: respect document structure
- Split on paragraphs
- Keep headers with their content
- Don't break mid-sentence

Even better: semantic chunking
- Group by topic
- Use model to find natural boundaries

## Embedding models

The embedding model matters A LOT. Common choices:

- OpenAI ada-002 (1536 dims, good quality)
- sentence-transformers (open source, various sizes)
- Cohere embed (good multilingual)

Considerations:
- Dimension size affects storage and speed
- Quality varies by domain
- Some tuned for retrieval specifically

## Vector stores

Where you store embeddings. Options:

**Lightweight:**
- FAISS (Facebook, in-memory)
- Annoy (Spotify, file-based)
- ChromaDB (easy to use)

**Production scale:**
- Pinecone (managed, expensive)
- Weaviate (self-hosted or managed)
- Milvus (open source, scalable)
- pgvector (Postgres extension)

For prototypes, ChromaDB or FAISS. For production, depends on scale.

## Retrieval methods

**Basic: k-NN search**

Find k most similar vectors. Simple but works.

**Hybrid: keyword + semantic**

Combine BM25 (keyword) with vector similarity. Catches things pure semantic misses.

```python
# pseudo-code
semantic_results = vector_db.search(query_embedding, k=10)
keyword_results = bm25_search(query_text, k=10)
combined = rerank(semantic_results + keyword_results)
```

**Re-ranking**

Retrieve more candidates, then use cross-encoder to rerank. More accurate but slower.

## The prompt engineering part

Retrieved context goes into prompt:

```
Context:
{retrieved_chunks}

Question: {user_question}

Answer based only on the provided context. If the answer isn't in the context, say you don't know.
```

That last part matters - tells model to not make stuff up.

## Common issues

**Chunk too small:** loses context, fragments don't make sense alone

**Chunk too big:** retrieval less precise, wastes context window

**Wrong k:** too few misses relevant info, too many adds noise

**Embedding mismatch:** query and documents need same embedding model

**No answer in retrieved docs:** model might hallucinate anyway

## Advanced patterns

**Parent-child chunks**

Store small chunks for retrieval, but return larger parent chunk for context.

**HyDE - Hypothetical Document Embeddings**

Generate hypothetical answer, embed that, search for similar. Sometimes helps with short queries.

**Query expansion**

Rephrase query multiple ways, search with each, combine results.

**Multi-step retrieval**

Use first retrieval to understand query better, then do refined second retrieval.

## Evaluation

Hard to evaluate properly. Metrics to consider:

- **Retrieval quality:** are right chunks returned?
- **Answer quality:** is final answer correct?
- **Faithfulness:** does answer use retrieved context?

Manual evaluation often needed. Or use LLM-as-judge (but that's not perfect either).

## My experience

RAG works surprisingly well for:
- Internal docs Q&A
- Customer support
- Code documentation search

Struggles with:
- Complex multi-hop reasoning
- Very precise numerical questions
- When relevant info spread across many docs

The interactive visualization shows the full flow: [RAG Animation](https://danielsobrado.github.io/ml-animations/animation/rag)

---

Related:
- [Embeddings Explained](/posts/embeddings/)
- [Cosine Similarity](/posts/cosine-similarity/)
