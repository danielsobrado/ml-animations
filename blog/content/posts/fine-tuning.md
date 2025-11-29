---
title: "Fine-Tuning pretrained models - what actually works"
date: 2024-11-28
draft: false
tags: ["fine-tuning", "transfer-learning", "lora", "qlora"]
categories: ["Machine Learning"]
---

You got a pretrained model. Now you want it to do your specific task. Fine-tuning is the answer but it's more nuanced than tutorials suggest.

## Why fine-tune?

Training from scratch needs:
- Massive data (billions of tokens for LLMs)
- Lots of compute (thousands of GPU hours)
- Expertise to get it working

Fine-tuning needs:
- Your specific task data (hundreds to thousands examples)
- Modest compute (few hours on single GPU often)
- Still some expertise but way less

The pretrained model already knows language. You're just teaching it your task.

![Fine-Tuning Methods](https://danielsobrado.github.io/ml-animations/animation/fine-tuning)

Check the interactive comparison: [Fine-Tuning Animation](https://danielsobrado.github.io/ml-animations/animation/fine-tuning)

## Full fine-tuning

Update all model weights on your data.

Pros:
- Maximum flexibility
- Can significantly change model behavior

Cons:
- Need separate copy of weights per task
- Risk of catastrophic forgetting
- Need more compute and memory

For BERT-size models, full fine-tuning is still practical. For larger models, gets expensive fast.

## LoRA - Low-Rank Adaptation

The clever insight: weight updates during fine-tuning are often low-rank. Don't need full matrices.

Instead of updating full weight matrix W, add low-rank decomposition:
$$W' = W + BA$$

where B is d×r and A is r×d, with r << d

Only train B and A. Original W stays frozen.

Benefits:
- Much fewer trainable params (often 0.1-1% of original)
- Can merge back into original weights for inference
- Multiple adaptations with same base model

The visualization shows how LoRA works with different rank settings.

## QLoRA - Quantized LoRA

Fine-tune on consumer GPUs by combining:
- 4-bit quantization of base model
- LoRA adapters in higher precision
- Paged optimizers to handle memory spikes

This is how people fine-tune 7B+ models on single GPU.

Memory savings are dramatic. 65B model that needs 130GB+ can fit in 24GB VRAM.

## What I've learned

**Learning rate matters more than you think**

Full fine-tuning: 1e-5 to 5e-5 typical
LoRA: can go higher, 1e-4 to 3e-4

Too high = catastrophic forgetting
Too low = barely changes anything

**Batch size vs gradient accumulation**

Larger effective batch usually helps stability. If VRAM limited, use gradient accumulation.

```python
# effective batch = batch_size * gradient_accumulation_steps
# if you want 32 but can only fit 8:
batch_size = 8
gradient_accumulation_steps = 4
```

**Early stopping**

Fine-tuning overfits fast. 2-4 epochs often enough. Watch validation loss.

## Practical example

Let's say you want to fine-tune for classification.

Full fine-tuning:
```python
from transformers import AutoModelForSequenceClassification, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    num_labels=2
)

trainer = Trainer(
    model=model,
    train_dataset=train_data,
    args=TrainingArguments(
        learning_rate=2e-5,
        num_train_epochs=3,
        per_device_train_batch_size=16,
    )
)
trainer.train()
```

With LoRA:
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,  # rank
    lora_alpha=32,
    target_modules=["query", "value"],
    lora_dropout=0.1,
)

model = get_peft_model(model, lora_config)
# only ~0.1% params trainable now
```

## When to use what

**Full fine-tuning:**
- Small models (BERT-size)
- Plenty of data
- Task very different from pre-training

**LoRA:**
- Larger models (1B+)
- Limited compute
- Multiple tasks from same base

**QLoRA:**
- Large models on consumer hardware
- Very memory constrained

## Common mistakes

1. Learning rate too high, model forgets everything
2. Training too long, overfits to small dataset
3. Not using proper validation set
4. Ignoring class imbalance
5. Wrong loss function for task

The visualization helps understand different approaches: [Fine-Tuning Animation](https://danielsobrado.github.io/ml-animations/animation/fine-tuning)

---

Related:
- [BERT Architecture](/posts/bert/)
- [Layer Normalization](/posts/layer-normalization/)
