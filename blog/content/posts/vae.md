---
title: "VAE - learning to generate with latent spaces"
date: 2024-11-13
draft: false
tags: ["vae", "variational-autoencoder", "generative", "latent-space"]
categories: ["Advanced Models"]
---

Autoencoders compress data. VAEs do that plus learn meaningful latent space you can sample from. The "variational" part makes all the difference.

## Regular autoencoder first

Encoder compresses input x to latent code z.
Decoder reconstructs x from z.

```python
class Autoencoder(nn.Module):
    def __init__(self):
        self.encoder = nn.Sequential(...)  # x → z
        self.decoder = nn.Sequential(...)  # z → x'
    
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon
```

Problem: latent space has no structure. Sampling random z gives garbage.

## VAE difference

Instead of encoding to a point, encode to a distribution (mean and variance).

$$q(z|x) = \mathcal{N}(\mu(x), \sigma^2(x))$$

Sample z from this distribution. Decode sample.

![VAE Architecture](https://danielsobrado.github.io/ml-animations/animation/vae)

See the latent space: [VAE Animation](https://danielsobrado.github.io/ml-animations/animation/vae)

## The reparameterization trick

Can't backprop through random sampling. Trick: reparameterize.

Instead of:
$$z \sim \mathcal{N}(\mu, \sigma^2)$$

Do:
$$z = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)$$

Now z is differentiable w.r.t. μ and σ. Gradients flow.

```python
def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
```

## VAE loss

Two parts:

**Reconstruction loss:** Output should match input
$$\mathcal{L}_{recon} = ||x - \text{decoder}(z)||^2$$

**KL divergence:** Latent distribution should be close to prior N(0,1)
$$\mathcal{L}_{KL} = -\frac{1}{2}\sum(1 + \log\sigma^2 - \mu^2 - \sigma^2)$$

Total loss:
$$\mathcal{L} = \mathcal{L}_{recon} + \beta \cdot \mathcal{L}_{KL}$$

## Full code

```python
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_var = nn.Linear(256, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss
```

## Why KL divergence?

Without KL term, encoder would collapse each input to a point. No overlap between latent codes.

KL forces latent distributions to be close to N(0,1). This creates structure:
- Similar inputs → overlapping distributions
- Smooth interpolation possible
- Random sampling from prior gives valid outputs

## Beta-VAE

Increase β to learn more disentangled representations:

$$\mathcal{L} = \mathcal{L}_{recon} + \beta \cdot \mathcal{L}_{KL}$$

β > 1: more pressure on latent structure, possibly blurrier reconstructions
β < 1: better reconstructions, less structured latent space

## What you can do

**Generation:** Sample z ~ N(0,1), decode
```python
z = torch.randn(batch_size, latent_dim)
samples = vae.decode(z)
```

**Interpolation:** Move smoothly between two inputs
```python
z1 = vae.encode(x1)[0]  # mean
z2 = vae.encode(x2)[0]
for alpha in [0, 0.25, 0.5, 0.75, 1.0]:
    z_interp = alpha * z1 + (1-alpha) * z2
    x_interp = vae.decode(z_interp)
```

**Attribute manipulation:** If dimensions are disentangled
```python
z[0, dimension_for_smile] += 1.0  # hypothetically
```

## Limitations

- Blurry outputs (MSE loss averages uncertainty)
- Hard to scale to high resolution
- Mode collapse still possible
- Disentanglement not guaranteed

GANs often give sharper outputs. Diffusion models currently dominate image generation. But VAEs have nice theoretical properties and stable training.

## Conditional VAE

Condition on labels or other info:

```python
def encode(self, x, y):
    h = self.encoder(torch.cat([x, y], dim=-1))
    ...

def decode(self, z, y):
    return self.decoder(torch.cat([z, y], dim=-1))
```

Generate specific class: sample z, provide target y.

Explore the latent space interactively: [VAE Animation](https://danielsobrado.github.io/ml-animations/animation/vae)

---

Related:
- [Cross-Entropy for probability](/posts/cross-entropy/)
- [Embeddings as representations](/posts/embeddings/)
