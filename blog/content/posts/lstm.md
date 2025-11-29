---
title: "LSTM - learning long-term dependencies"
date: 2024-11-16
draft: false
tags: ["lstm", "rnn", "sequence-modeling", "neural-networks"]
categories: ["Neural Networks"]
---

RNNs have short memory. Information from early timesteps fades quickly. LSTMs fix this with gates that control information flow.

Before transformers dominated, LSTM was the go-to for sequence tasks.

## The vanilla RNN problem

Simple RNN:
$$h_t = \tanh(W_h h_{t-1} + W_x x_t)$$

Information from step 0 must pass through all intermediate steps to reach step 100. It gets multiplied by weights each time.

If weights < 1: vanishing gradients
If weights > 1: exploding gradients

Long sequences = trouble.

## LSTM architecture

LSTM adds a "cell state" - a highway for information to flow unimpeded.

Four gates control what happens:

![LSTM Gates](https://danielsobrado.github.io/ml-animations/animation/lstm)

Interactive breakdown: [LSTM Animation](https://danielsobrado.github.io/ml-animations/animation/lstm)

### Forget gate

Decides what to throw away from cell state.

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

Output: 0 = forget completely, 1 = keep everything

### Input gate

Decides what new information to store.

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

### Cell state update

Combine forget and input:

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

Old cell state (maybe partially forgotten) + new information (maybe partially added)

### Output gate

Decides what to output based on cell state.

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \odot \tanh(C_t)$$

## Why it works

Cell state C can flow unchanged across many timesteps (if forget gate ≈ 1 and input gate ≈ 0).

Gradients flow through the cell state without repeated multiplication by small numbers.

Information from step 0 can reach step 100 if the network learns to keep forget gate open.

## Code

PyTorch makes it simple:
```python
lstm = nn.LSTM(
    input_size=100,
    hidden_size=256,
    num_layers=2,
    batch_first=True,
    dropout=0.1,
    bidirectional=True
)

# Forward pass
output, (h_n, c_n) = lstm(x)
# output: [batch, seq_len, hidden_size * num_directions]
# h_n: final hidden state
# c_n: final cell state
```

From scratch (simplified):
```python
class LSTMCell:
    def __init__(self, input_size, hidden_size):
        # Combined weight matrix for all gates
        self.W = nn.Linear(input_size + hidden_size, 4 * hidden_size)
    
    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=-1)
        gates = self.W(combined)
        
        i, f, o, g = gates.chunk(4, dim=-1)
        
        i = torch.sigmoid(i)  # input gate
        f = torch.sigmoid(f)  # forget gate
        o = torch.sigmoid(o)  # output gate
        g = torch.tanh(g)     # candidate cell
        
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        
        return h_new, c_new
```

## Bidirectional LSTM

Process sequence in both directions. For tasks where full context available (not generation).

```python
lstm = nn.LSTM(..., bidirectional=True)
# hidden_size doubles (forward + backward)
```

Often helps for classification, tagging, etc.

## Stacked LSTMs

Multiple LSTM layers for more capacity:

```python
lstm = nn.LSTM(..., num_layers=3)
```

Each layer processes output of layer below. More layers = more abstraction.

Dropout between layers to prevent overfitting.

## GRU - simplified variant

Gated Recurrent Unit. Similar idea, fewer parameters.

Two gates instead of four:
- Reset gate
- Update gate

$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t])$$
$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t])$$
$$\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t])$$
$$h_t = (1-z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

No separate cell state. Often works comparably to LSTM.

## LSTM vs Transformers

**LSTM:**
- Sequential processing (can't parallelize across timesteps)
- Good for online/streaming
- Lower memory during inference
- Works with limited data

**Transformers:**
- Parallel processing
- Better at very long dependencies
- Needs more data and compute
- Current SOTA for most tasks

LSTM isn't obsolete. For streaming applications, edge devices, or limited data, still useful.

## Common patterns

**Sequence classification:**
```python
output, (h_n, c_n) = lstm(sequence)
# use h_n[-1] (last layer's final hidden)
logits = classifier(h_n[-1])
```

**Sequence tagging:**
```python
output, _ = lstm(sequence)
# use output at each timestep
logits = classifier(output)
```

**Generation:**
```python
# autoregressively
h, c = None, None
for t in range(seq_len):
    output, (h, c) = lstm(input_t, (h, c))
    next_token = sample(output)
```

See how gates control information flow: [LSTM Animation](https://danielsobrado.github.io/ml-animations/animation/lstm)

---

Related:
- [Transformers replaced LSTMs](/posts/transformer-architecture/)
- [Attention Mechanism](/posts/attention-mechanism-part1/)
