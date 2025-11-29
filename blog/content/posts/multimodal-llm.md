---
title: "Multimodal LLMs - when models see and read"
date: 2024-11-12
draft: false
tags: ["multimodal", "llm", "vision-language", "clip"]
categories: ["Advanced Models"]
---

Text-only LLMs are impressive but limited. Can't show them a picture and ask "what's wrong here?" Multimodal models bridge that gap.

## What is multimodal?

Model that processes multiple types of input: text + images, text + audio, etc.

GPT-4V, Gemini, LLaVA, Claude 3 - all multimodal. They can "see" images and reason about them.

![Multimodal Architecture](https://danielsobrado.github.io/ml-animations/animation/multimodal-llm)

See the architecture: [Multimodal LLM Animation](https://danielsobrado.github.io/ml-animations/animation/multimodal-llm)

## How it works (simplified)

1. **Image encoder:** Converts image to embeddings (usually ViT)
2. **Projection:** Maps image embeddings to LLM's space
3. **LLM:** Processes combined text + image tokens
4. **Output:** Text as usual

```
Image → [Image Encoder] → [Projection] → Image Tokens
                                              ↓
Text → [Tokenizer] → Text Tokens → [Combined] → [LLM] → Output
```

## The image encoder

Usually Vision Transformer (ViT) pretrained on images.

Split image into patches (16x16 or 14x14). Each patch becomes a token.
224×224 image with 16×16 patches = 196 image tokens.

```python
# Pseudo-code
patches = split_into_patches(image, patch_size=16)  # [N, 16, 16, 3]
patch_embeddings = linear_projection(patches)  # [N, D]
image_tokens = vit_encoder(patch_embeddings)  # [N, D]
```

## CLIP - the foundation

Contrastive Language-Image Pretraining (OpenAI, 2021).

Train image and text encoders together:
- Matching image-text pairs should have similar embeddings
- Non-matching pairs should be different

```
"A photo of a dog" ↔ [image of dog]   → high similarity
"A photo of a dog" ↔ [image of cat]   → low similarity
```

CLIP's image encoder is used in many multimodal models.

## Connecting vision to language

The tricky part: LLM expects text tokens. Image encoder outputs image features.

**Option 1: Linear projection**
Simple matrix maps image features to text embedding space.
```python
image_tokens = projection(image_features)  # shape matches text embeddings
```

**Option 2: Q-Former (BLIP-2)**
Learned queries attend to image features, produce fixed-length output.

**Option 3: Perceiver resampler (Flamingo)**
Cross-attention to resample variable-length image features.

## Training strategies

**Stage 1: Align vision and language**
Train projection layer while freezing encoders.
Dataset: image-caption pairs.

**Stage 2: Fine-tune for instruction following**
Unfreeze more, train on conversation data.
"What's in this image?" + image → detailed description

**Stage 3: RLHF (optional)**
Refine based on human preferences.

## LLaVA approach

Popular open-source multimodal model.

1. Pretrained CLIP ViT for vision
2. Pretrained LLaMA for language
3. Simple linear projection connecting them
4. Two-stage training: alignment then instruction tuning

Surprisingly effective given simplicity.

## Capabilities

What multimodal models can do:
- Describe images in detail
- Answer questions about images
- OCR and document understanding
- Compare multiple images
- Chart and diagram interpretation
- Some spatial reasoning

What they struggle with:
- Precise counting
- Fine spatial relationships
- Multiple objects with complex relations
- Small text in images
- Mathematical reasoning from diagrams

## Using multimodal APIs

```python
# OpenAI GPT-4V
response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }
    ]
)

# Anthropic Claude
message = client.messages.create(
    model="claude-3-opus-20240229",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64_image}},
                {"type": "text", "text": "Describe this image"}
            ]
        }
    ]
)
```

## Resolution and tokens

Higher resolution = more tokens = more cost and latency.

Most APIs resize or tile images. GPT-4V uses:
- Low detail: 85 tokens
- High detail: up to 1445 tokens (for large images)

Consider resolution vs quality tradeoff.

## Beyond vision

Multimodal extends to:
- **Audio:** Whisper-like encoders + LLM
- **Video:** Frame sampling + vision encoder
- **3D:** Point cloud encoders

Same principle: encode modality, project to LLM space, generate text.

## Building your own

With open models:
```python
from transformers import LlavaForConditionalGeneration

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")

# Inference
inputs = processor(text=prompt, images=image, return_tensors="pt")
output = model.generate(**inputs)
```

See how modalities combine: [Multimodal LLM Animation](https://danielsobrado.github.io/ml-animations/animation/multimodal-llm)

---

Related:
- [Embeddings representation](/posts/embeddings/)
- [Transformer architecture](/posts/transformer-architecture/)
- [RAG with images](/posts/rag/)
