//! Variational Autoencoder (VAE) for Latent Diffusion
//!
//! The VAE compresses images into a lower-dimensional latent space where
//! diffusion happens. This is what makes Stable Diffusion "latent" - we
//! don't diffuse in pixel space but in this compressed representation.
//!
//! Key insight: A 512x512 RGB image has 786,432 values. The VAE compresses
//! this to roughly 64x64x4 = 16,384 values - a 48x reduction! This makes
//! diffusion tractable on consumer hardware.

use crate::tensor::Tensor;
use crate::nn::{Conv2d, Linear, GroupNorm};

/// Gaussian distribution for reparameterization trick
/// 
/// The VAE encoder outputs mean and log-variance of a Gaussian.
/// We sample from this distribution using the reparameterization trick:
/// z = mean + std * epsilon, where epsilon ~ N(0,1)
/// 
/// This trick allows gradients to flow through the sampling operation.
pub struct DiagonalGaussian {
    pub mean: Tensor,
    pub logvar: Tensor,
    pub std: Tensor,
}

impl DiagonalGaussian {
    /// Create distribution from encoder output
    /// 
    /// The encoder outputs a tensor that we split into mean and logvar.
    /// We clamp logvar to prevent numerical instability.
    pub fn new(params: &Tensor, latent_channels: usize) -> Self {
        let shape = params.shape();
        let channels = shape[1];
        
        // Split along channel dimension
        // First half is mean, second half is logvar
        let half_channels = channels / 2;
        assert_eq!(half_channels, latent_channels, "Channel mismatch");
        
        // For simplicity, we'll compute mean and logvar from the full tensor
        // In practice, you'd slice the tensor
        let mean = params.clone(); // Would slice: params[:, :latent_channels]
        let logvar = params.clone(); // Would slice: params[:, latent_channels:]
        
        // Clamp logvar for numerical stability
        // Too negative = near-zero variance (deterministic)
        // Too positive = huge variance (unstable)
        let logvar_clamped = logvar.clamp(-30.0, 20.0);
        
        // std = exp(0.5 * logvar) = sqrt(variance)
        let std = logvar_clamped.mul_scalar(0.5).exp();
        
        DiagonalGaussian {
            mean,
            logvar: logvar_clamped,
            std,
        }
    }
    
    /// Sample from the distribution using reparameterization trick
    /// 
    /// Instead of sampling z ~ N(mean, std), we sample:
    /// epsilon ~ N(0, 1)
    /// z = mean + std * epsilon
    /// 
    /// This allows backpropagation through the sampling.
    pub fn sample(&self) -> Tensor {
        let epsilon = Tensor::randn(self.mean.shape());
        self.mean.add(&self.std.mul(&epsilon))
    }
    
    /// KL divergence from standard normal N(0,1)
    /// 
    /// KL(q(z|x) || p(z)) where p(z) = N(0,1)
    /// = -0.5 * sum(1 + logvar - mean^2 - exp(logvar))
    /// 
    /// This regularizes the latent space to be close to standard normal.
    pub fn kl_divergence(&self) -> f32 {
        // KL = -0.5 * (1 + logvar - mean^2 - var)
        let mean_sq = self.mean.mul(&self.mean);
        let var = self.logvar.exp();
        
        // Sum over all dimensions
        let kl_per_element = self.logvar
            .add_scalar(1.0)
            .sub(&mean_sq)
            .sub(&var)
            .mul_scalar(-0.5);
        
        kl_per_element.mean()
    }
}

/// Simple ResNet block for VAE
/// 
/// ResNet blocks allow deeper networks by adding skip connections.
/// The formula is: output = input + F(input)
/// where F is two convolutions with normalization and activation.
pub struct ResNetBlock {
    norm1: GroupNorm,
    conv1: Conv2d,
    norm2: GroupNorm,
    conv2: Conv2d,
    channels: usize,
}

impl ResNetBlock {
    pub fn new(channels: usize) -> Self {
        ResNetBlock {
            norm1: GroupNorm::new(32, channels),
            conv1: Conv2d::new(channels, channels, 3, 1, 1),
            norm2: GroupNorm::new(32, channels),
            conv2: Conv2d::new(channels, channels, 3, 1, 1),
            channels,
        }
    }
    
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let h = self.norm1.forward(x);
        let h = h.silu();
        let h = self.conv1.forward(&h);
        
        let h = self.norm2.forward(&h);
        let h = h.silu();
        let h = self.conv2.forward(&h);
        
        // Skip connection
        x.add(&h)
    }
}

/// ResNet block with channel projection
/// 
/// When input and output channels differ, we need a 1x1 convolution
/// on the skip connection to match dimensions.
pub struct ResNetBlockProjection {
    shortcut: Conv2d,
    norm1: GroupNorm,
    conv1: Conv2d,
    norm2: GroupNorm,
    conv2: Conv2d,
}

impl ResNetBlockProjection {
    pub fn new(in_channels: usize, out_channels: usize) -> Self {
        ResNetBlockProjection {
            shortcut: Conv2d::new(in_channels, out_channels, 1, 1, 0),
            norm1: GroupNorm::new(32, in_channels),
            conv1: Conv2d::new(in_channels, out_channels, 3, 1, 1),
            norm2: GroupNorm::new(32, out_channels),
            conv2: Conv2d::new(out_channels, out_channels, 3, 1, 1),
        }
    }
    
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let shortcut = self.shortcut.forward(x);
        
        let h = self.norm1.forward(x);
        let h = h.silu();
        let h = self.conv1.forward(&h);
        
        let h = self.norm2.forward(&h);
        let h = h.silu();
        let h = self.conv2.forward(&h);
        
        shortcut.add(&h)
    }
}

/// Self-attention block for VAE
/// 
/// Attention helps the VAE capture global dependencies in the image.
/// At lower resolutions (like 32x32), attention is computationally feasible.
pub struct AttentionBlock {
    norm: GroupNorm,
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    channels: usize,
}

impl AttentionBlock {
    pub fn new(channels: usize) -> Self {
        AttentionBlock {
            norm: GroupNorm::new(32, channels),
            to_q: Linear::new(channels, channels),
            to_k: Linear::new(channels, channels),
            to_v: Linear::new(channels, channels),
            to_out: Linear::new(channels, channels),
            channels,
        }
    }
    
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.shape();
        let batch = shape[0];
        let channels = shape[1];
        let height = shape[2];
        let width = shape[3];
        let seq_len = height * width;
        
        // Normalize and reshape to [batch, seq_len, channels]
        let h = self.norm.forward(x);
        let h = h.reshape(&[batch, channels, seq_len]);
        let h = h.transpose(); // [batch, seq_len, channels]
        
        // Compute Q, K, V
        let q = self.to_q.forward(&h);
        let k = self.to_k.forward(&h);
        let v = self.to_v.forward(&h);
        
        // Attention: softmax(Q @ K^T / sqrt(d)) @ V
        let scale = (channels as f32).sqrt();
        let attn = q.matmul(&k.transpose()).mul_scalar(1.0 / scale);
        let attn = attn.softmax();
        let out = attn.matmul(&v);
        
        // Project output
        let out = self.to_out.forward(&out);
        
        // Reshape back to [batch, channels, height, width]
        let out = out.transpose();
        let out = out.reshape(&[batch, channels, height, width]);
        
        // Residual connection
        x.add(&out)
    }
}

/// Downsample block for encoder
/// 
/// Reduces spatial dimensions by 2x using strided convolution.
/// Strided convolution is preferred over pooling as it's learnable.
pub struct Downsample {
    conv: Option<Conv2d>,
}

impl Downsample {
    pub fn new(in_channels: usize, out_channels: usize, use_conv: bool) -> Self {
        let conv = if use_conv {
            Some(Conv2d::new(in_channels, out_channels, 3, 2, 0))
        } else {
            None
        };
        Downsample { conv }
    }
    
    pub fn forward(&self, x: &Tensor) -> Tensor {
        if let Some(ref conv) = self.conv {
            // Manual asymmetric padding for precision
            let x_padded = x.pad(&[0, 1, 0, 1]); // Right and bottom padding
            conv.forward(&x_padded)
        } else {
            x.clone()
        }
    }
}

/// Upsample block for decoder
/// 
/// Increases spatial dimensions by 2x using nearest neighbor interpolation
/// followed by convolution. This avoids checkerboard artifacts that
/// transposed convolutions can produce.
pub struct Upsample {
    conv: Option<Conv2d>,
}

impl Upsample {
    pub fn new(in_channels: usize, out_channels: usize, use_conv: bool) -> Self {
        let conv = if use_conv {
            Some(Conv2d::new(in_channels, out_channels, 3, 1, 1))
        } else {
            None
        };
        Upsample { conv }
    }
    
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Nearest neighbor upsampling 2x
        let upsampled = x.upsample_nearest(2);
        
        if let Some(ref conv) = self.conv {
            conv.forward(&upsampled)
        } else {
            upsampled
        }
    }
}

/// Down block: ResNet blocks + Downsample
pub struct DownBlock {
    resnet1: Box<dyn ResNetBlockTrait>,
    resnet2: ResNetBlock,
    downsample: Downsample,
}

trait ResNetBlockTrait {
    fn forward(&self, x: &Tensor) -> Tensor;
}

impl ResNetBlockTrait for ResNetBlock {
    fn forward(&self, x: &Tensor) -> Tensor {
        ResNetBlock::forward(self, x)
    }
}

impl ResNetBlockTrait for ResNetBlockProjection {
    fn forward(&self, x: &Tensor) -> Tensor {
        ResNetBlockProjection::forward(self, x)
    }
}

/// Up block: ResNet blocks + Upsample  
pub struct UpBlock {
    resnet1: Box<dyn ResNetBlockTrait>,
    resnet2: ResNetBlock,
    resnet3: ResNetBlock,
    upsample: Upsample,
}

/// Middle block: ResNet + Attention + ResNet
pub struct MidBlock {
    resnet1: ResNetBlock,
    attention: AttentionBlock,
    resnet2: ResNetBlock,
}

impl MidBlock {
    pub fn new(channels: usize) -> Self {
        MidBlock {
            resnet1: ResNetBlock::new(channels),
            attention: AttentionBlock::new(channels),
            resnet2: ResNetBlock::new(channels),
        }
    }
    
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let h = self.resnet1.forward(x);
        let h = self.attention.forward(&h);
        self.resnet2.forward(&h)
    }
}

/// VAE Encoder
/// 
/// Compresses images from [B, 3, H, W] to latent [B, latent_channels, H/8, W/8]
/// 
/// Architecture:
/// - Initial conv: 3 -> 128 channels
/// - 4 down blocks: progressively increase channels and reduce spatial size
/// - Mid block with attention
/// - Output conv to latent space
pub struct Encoder {
    conv_in: Conv2d,
    down_blocks: Vec<(ResNetBlockProjection, ResNetBlock, Downsample)>,
    mid_block: MidBlock,
    conv_norm_out: GroupNorm,
    conv_out: Conv2d,
}

impl Encoder {
    pub fn new(in_channels: usize, base_channels: usize, quant_channels: usize) -> Self {
        // Channel progression: 128 -> 128 -> 256 -> 512 -> 512
        let down_configs = [
            (base_channels, base_channels, true),
            (base_channels, base_channels * 2, true),
            (base_channels * 2, base_channels * 4, true),
            (base_channels * 4, base_channels * 4, false), // No downsample on last
        ];
        
        let conv_in = Conv2d::new(in_channels, base_channels, 3, 1, 1);
        
        let mut down_blocks = Vec::new();
        for (in_ch, out_ch, use_conv) in down_configs.iter() {
            let resnet1 = if in_ch != out_ch {
                ResNetBlockProjection::new(*in_ch, *out_ch)
            } else {
                // Use same-channel block
                ResNetBlockProjection::new(*in_ch, *out_ch)
            };
            let resnet2 = ResNetBlock::new(*out_ch);
            let downsample = Downsample::new(*out_ch, *out_ch, *use_conv);
            down_blocks.push((resnet1, resnet2, downsample));
        }
        
        let final_channels = base_channels * 4;
        let mid_block = MidBlock::new(final_channels);
        let conv_norm_out = GroupNorm::new(32, final_channels);
        let conv_out = Conv2d::new(final_channels, quant_channels, 3, 1, 1);
        
        Encoder {
            conv_in,
            down_blocks,
            mid_block,
            conv_norm_out,
            conv_out,
        }
    }
    
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut h = self.conv_in.forward(x);
        
        for (resnet1, resnet2, downsample) in &self.down_blocks {
            h = resnet1.forward(&h);
            h = resnet2.forward(&h);
            h = downsample.forward(&h);
        }
        
        h = self.mid_block.forward(&h);
        h = self.conv_norm_out.forward(&h);
        h = h.silu();
        self.conv_out.forward(&h)
    }
}

/// VAE Decoder
/// 
/// Reconstructs images from latent [B, latent_channels, H/8, W/8] to [B, 3, H, W]
/// 
/// Architecture mirrors the encoder but in reverse:
/// - Input conv from latent space
/// - Mid block with attention  
/// - 4 up blocks: progressively decrease channels and increase spatial size
/// - Output conv: 128 -> 3 channels
pub struct Decoder {
    conv_in: Conv2d,
    mid_block: MidBlock,
    up_blocks: Vec<(ResNetBlockProjection, ResNetBlock, ResNetBlock, Upsample)>,
    conv_norm_out: GroupNorm,
    conv_out: Conv2d,
}

impl Decoder {
    pub fn new(output_channels: usize, latent_channels: usize, base_channels: usize) -> Self {
        let up_configs = [
            (512, 512, true),
            (512, 512, true),
            (512, 256, true),
            (256, 128, false),
        ];
        
        let initial_channels = base_channels * 4; // 512
        let conv_in = Conv2d::new(latent_channels, initial_channels, 3, 1, 1);
        let mid_block = MidBlock::new(initial_channels);
        
        let mut up_blocks = Vec::new();
        for (in_ch, out_ch, use_conv) in up_configs.iter() {
            let resnet1 = ResNetBlockProjection::new(*in_ch, *out_ch);
            let resnet2 = ResNetBlock::new(*out_ch);
            let resnet3 = ResNetBlock::new(*out_ch);
            let upsample = Upsample::new(*out_ch, *out_ch, *use_conv);
            up_blocks.push((resnet1, resnet2, resnet3, upsample));
        }
        
        let conv_norm_out = GroupNorm::new(32, base_channels);
        let conv_out = Conv2d::new(base_channels, output_channels, 3, 1, 1);
        
        Decoder {
            conv_in,
            mid_block,
            up_blocks,
            conv_norm_out,
            conv_out,
        }
    }
    
    pub fn forward(&self, z: &Tensor) -> Tensor {
        let mut h = self.conv_in.forward(z);
        h = self.mid_block.forward(&h);
        
        for (resnet1, resnet2, resnet3, upsample) in &self.up_blocks {
            h = resnet1.forward(&h);
            h = resnet2.forward(&h);
            h = resnet3.forward(&h);
            h = upsample.forward(&h);
        }
        
        h = self.conv_norm_out.forward(&h);
        h = h.silu();
        h = self.conv_out.forward(&h);
        
        // Tanh to bound output to [-1, 1]
        h.tanh()
    }
}

/// Complete Variational Autoencoder
/// 
/// The VAE is the key to making diffusion models practical. It provides:
/// 1. Compression: 48x reduction in data dimensionality
/// 2. Regularization: Latent space is approximately standard normal
/// 3. Semantic structure: Similar images have similar latents
/// 
/// SD3 uses 16 latent channels (up from SD1's 4) for higher quality.
pub struct VAE {
    pub encoder: Encoder,
    pub decoder: Decoder,
    pub latent_channels: usize,
    pub scaling_factor: f32,
    pub shift_factor: f32,
}

impl VAE {
    /// Create a new VAE
    /// 
    /// - latent_channels: 16 for SD3, 4 for SD1
    /// - scaling_factor: Normalizes latent distribution  
    /// - shift_factor: Centers the distribution
    pub fn new(latent_channels: usize) -> Self {
        let base_channels = 128;
        let quant_channels = latent_channels * 2; // Mean + logvar
        
        VAE {
            encoder: Encoder::new(3, base_channels, quant_channels),
            decoder: Decoder::new(3, latent_channels, base_channels),
            latent_channels,
            scaling_factor: 1.5305,  // SD3 value
            shift_factor: 0.0609,    // SD3 value
        }
    }
    
    /// Encode image to latent distribution
    /// 
    /// Returns a DiagonalGaussian that you can sample from.
    pub fn encode(&self, x: &Tensor) -> DiagonalGaussian {
        let params = self.encoder.forward(x);
        DiagonalGaussian::new(&params, self.latent_channels)
    }
    
    /// Decode latent to image
    pub fn decode(&self, z: &Tensor) -> Tensor {
        self.decoder.forward(z)
    }
    
    /// Normalize latent for diffusion
    /// 
    /// The raw VAE output has its own distribution. We normalize it
    /// to be closer to N(0,1) which diffusion models expect.
    pub fn normalize_latent(&self, z: &Tensor) -> Tensor {
        z.sub_scalar(self.shift_factor).mul_scalar(self.scaling_factor)
    }
    
    /// Denormalize latent after diffusion
    pub fn denormalize_latent(&self, z: &Tensor) -> Tensor {
        z.mul_scalar(1.0 / self.scaling_factor).add_scalar(self.shift_factor)
    }
    
    /// Full encode-decode pass (for testing)
    pub fn forward(&self, x: &Tensor) -> (Tensor, DiagonalGaussian) {
        let dist = self.encode(x);
        let z = dist.sample();
        let recon = self.decode(&z);
        (recon, dist)
    }
}

/// Compute VAE reconstruction loss
/// 
/// Loss = MSE(reconstruction, original) + beta * KL_divergence
/// 
/// The beta term (usually 0.0001 to 0.001) controls the trade-off between
/// reconstruction quality and latent space regularity.
pub fn vae_loss(recon: &Tensor, original: &Tensor, dist: &DiagonalGaussian, beta: f32) -> f32 {
    let mse = recon.sub(original).pow(2.0).mean();
    let kl = dist.kl_divergence();
    mse + beta * kl
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_vae_shapes() {
        let vae = VAE::new(16);
        
        // Test with small image for speed
        let x = Tensor::randn(&[1, 3, 64, 64]);
        
        // Encode
        let dist = vae.encode(&x);
        let z = dist.sample();
        
        // Check latent shape: [1, 16, 8, 8] (8x spatial reduction)
        assert_eq!(z.shape()[0], 1);
        assert_eq!(z.shape()[1], 16);
        assert_eq!(z.shape()[2], 8);
        assert_eq!(z.shape()[3], 8);
        
        // Decode
        let recon = vae.decode(&z);
        
        // Check reconstruction shape matches input
        assert_eq!(recon.shape(), x.shape());
    }
    
    #[test]
    fn test_diagonal_gaussian() {
        let mean = Tensor::zeros(&[2, 4, 8, 8]);
        let logvar = Tensor::zeros(&[2, 4, 8, 8]);
        
        let dist = DiagonalGaussian {
            mean,
            logvar: logvar.clone(),
            std: logvar.exp().mul_scalar(0.5),
        };
        
        // Sample should have same shape as mean
        let z = dist.sample();
        assert_eq!(z.shape(), &[2, 4, 8, 8]);
        
        // KL divergence of N(0,1) from N(0,1) should be ~0
        let kl = dist.kl_divergence();
        assert!(kl.abs() < 0.1, "KL should be near 0 for standard normal");
    }
}
