//! U-Net Architecture - The noise prediction network
//!
//! A simplified U-Net for image denoising, similar to what Stable Diffusion uses
//! but much smaller for educational purposes.

use crate::nn::{Conv2d, GroupNorm, Linear};
use crate::tensor::Tensor;
use crate::diffusion::get_timestep_embedding;

/// Residual Block with GroupNorm and time embedding
#[derive(Debug, Clone)]
pub struct ResBlock {
    pub conv1: Conv2d,
    pub conv2: Conv2d,
    pub norm1: GroupNorm,
    pub norm2: GroupNorm,
    pub time_mlp: Linear,
    pub skip_conv: Option<Conv2d>,
    pub channels: usize,
}

impl ResBlock {
    pub fn new(in_channels: usize, out_channels: usize, time_emb_dim: usize) -> Self {
        let skip_conv = if in_channels != out_channels {
            Some(Conv2d::new(in_channels, out_channels, 1, 1, 0))
        } else {
            None
        };

        ResBlock {
            conv1: Conv2d::new(in_channels, out_channels, 3, 1, 1),
            conv2: Conv2d::new(out_channels, out_channels, 3, 1, 1),
            norm1: GroupNorm::new(8.min(out_channels), out_channels),
            norm2: GroupNorm::new(8.min(out_channels), out_channels),
            time_mlp: Linear::new(time_emb_dim, out_channels),
            skip_conv,
            channels: out_channels,
        }
    }

    pub fn forward(&self, x: &Tensor, time_emb: &Tensor) -> Tensor {
        let shape = x.shape();
        let batch = shape[0];
        let height = shape[2];
        let width = shape[3];

        // First conv + norm
        let h = self.conv1.forward(x);
        let h = self.norm1.forward(&h);
        let h = h.silu();

        // Add time embedding
        // time_emb shape: [batch, time_emb_dim]
        // Need to project and add to spatial features
        let time_proj = self.time_mlp.forward(time_emb);
        let time_proj = time_proj.silu();
        
        // Broadcast time projection to spatial dimensions
        // shape: [batch, channels] -> [batch, channels, height, width]
        let time_proj = time_proj.reshape(&[batch, self.channels, 1, 1]);
        let time_proj = time_proj.broadcast(&[batch, self.channels, height, width]);
        
        let h = h.add(&time_proj);

        // Second conv + norm
        let h = self.conv2.forward(&h);
        let h = self.norm2.forward(&h);
        let h = h.silu();

        // Skip connection
        let skip = match &self.skip_conv {
            Some(conv) => conv.forward(x),
            None => x.clone(),
        };

        h.add(&skip)
    }

    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = self.conv1.parameters();
        params.extend(self.conv2.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.norm2.parameters());
        params.extend(self.time_mlp.parameters());
        if let Some(skip) = &self.skip_conv {
            params.extend(skip.parameters());
        }
        params
    }
}

/// Downsampling block (stride 2 conv)
#[derive(Debug, Clone)]
pub struct Downsample {
    pub conv: Conv2d,
}

impl Downsample {
    pub fn new(channels: usize) -> Self {
        Downsample {
            conv: Conv2d::new(channels, channels, 3, 2, 1),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        self.conv.forward(x)
    }

    pub fn parameters(&self) -> Vec<&Tensor> {
        self.conv.parameters()
    }
}

/// Upsampling block (nearest neighbor + conv)
#[derive(Debug, Clone)]
pub struct Upsample {
    pub conv: Conv2d,
}

impl Upsample {
    pub fn new(channels: usize) -> Self {
        Upsample {
            conv: Conv2d::new(channels, channels, 3, 1, 1),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.shape();
        let batch = shape[0];
        let channels = shape[1];
        let height = shape[2];
        let width = shape[3];

        // Nearest neighbor upsampling 2x
        let new_height = height * 2;
        let new_width = width * 2;
        
        let mut upsampled = Tensor::zeros(&[batch, channels, new_height, new_width]);
        let x_contig = x.to_contiguous();
        let x_data = x_contig.as_slice();
        let up_data = upsampled.as_mut_slice();

        for b in 0..batch {
            for c in 0..channels {
                for h in 0..height {
                    for w in 0..width {
                        let val = x_data[b * channels * height * width + c * height * width + h * width + w];
                        // Copy to 2x2 block
                        for dh in 0..2 {
                            for dw in 0..2 {
                                let nh = h * 2 + dh;
                                let nw = w * 2 + dw;
                                up_data[b * channels * new_height * new_width + c * new_height * new_width + nh * new_width + nw] = val;
                            }
                        }
                    }
                }
            }
        }

        self.conv.forward(&upsampled)
    }

    pub fn parameters(&self) -> Vec<&Tensor> {
        self.conv.parameters()
    }
}

/// Simplified U-Net for diffusion models
/// 
/// Architecture:
/// - Encoder: progressively downsamples while increasing channels
/// - Middle: bottleneck with attention
/// - Decoder: progressively upsamples while decreasing channels
/// - Skip connections between encoder and decoder
#[derive(Debug)]
pub struct UNet {
    // Time embedding
    pub time_mlp1: Linear,
    pub time_mlp2: Linear,
    pub time_emb_dim: usize,
    
    // Initial convolution
    pub init_conv: Conv2d,
    
    // Encoder blocks
    pub down1: ResBlock,
    pub down2: ResBlock,
    pub down3: ResBlock,
    pub pool1: Downsample,
    pub pool2: Downsample,
    
    // Middle (bottleneck)
    pub mid1: ResBlock,
    pub mid2: ResBlock,
    
    // Decoder blocks
    pub up3: ResBlock,
    pub up2: ResBlock,
    pub up1: ResBlock,
    pub upsample2: Upsample,
    pub upsample1: Upsample,
    
    // Final convolution
    pub final_norm: GroupNorm,
    pub final_conv: Conv2d,
    
    // Channel configuration
    pub channels: [usize; 4],
}

impl UNet {
    /// Create a new U-Net
    /// 
    /// Args:
    ///   in_channels: Input image channels (e.g., 3 for RGB)
    ///   model_channels: Base number of channels (multiplied at each level)
    ///   out_channels: Output channels (same as input for noise prediction)
    pub fn new(in_channels: usize, model_channels: usize, out_channels: usize) -> Self {
        let time_emb_dim = model_channels * 4;
        
        // Channel progression: model_channels -> 2x -> 4x -> 4x
        let channels = [
            model_channels,
            model_channels * 2,
            model_channels * 4,
            model_channels * 4,
        ];

        UNet {
            // Time embedding MLP
            time_mlp1: Linear::new(model_channels, time_emb_dim),
            time_mlp2: Linear::new(time_emb_dim, time_emb_dim),
            time_emb_dim,
            
            // Initial conv: in_channels -> model_channels
            init_conv: Conv2d::new(in_channels, channels[0], 3, 1, 1),
            
            // Encoder
            down1: ResBlock::new(channels[0], channels[1], time_emb_dim),
            pool1: Downsample::new(channels[1]),
            down2: ResBlock::new(channels[1], channels[2], time_emb_dim),
            pool2: Downsample::new(channels[2]),
            down3: ResBlock::new(channels[2], channels[3], time_emb_dim),
            
            // Middle
            mid1: ResBlock::new(channels[3], channels[3], time_emb_dim),
            mid2: ResBlock::new(channels[3], channels[3], time_emb_dim),
            
            // Decoder (note: channels doubled for skip connections)
            up3: ResBlock::new(channels[3] + channels[3], channels[2], time_emb_dim),
            upsample2: Upsample::new(channels[2]),
            up2: ResBlock::new(channels[2] + channels[2], channels[1], time_emb_dim),
            upsample1: Upsample::new(channels[1]),
            up1: ResBlock::new(channels[1] + channels[1], channels[0], time_emb_dim),
            
            // Final
            final_norm: GroupNorm::new(8.min(channels[0]), channels[0]),
            final_conv: Conv2d::new(channels[0], out_channels, 3, 1, 1),
            
            channels,
        }
    }

    /// Forward pass
    /// 
    /// Args:
    ///   x: Noisy image [batch, channels, height, width]
    ///   timesteps: Current timestep for each batch element
    /// 
    /// Returns:
    ///   Predicted noise [batch, channels, height, width]
    pub fn forward(&self, x: &Tensor, timesteps: &[usize]) -> Tensor {
        let _batch = x.shape()[0];
        
        // 1. Time embedding
        let t_emb = get_timestep_embedding(timesteps, self.channels[0]);
        let t_emb = self.time_mlp1.forward(&t_emb);
        let t_emb = t_emb.silu();
        let t_emb = self.time_mlp2.forward(&t_emb);
        
        // 2. Initial convolution
        let h = self.init_conv.forward(x);
        
        // 3. Encoder (save for skip connections)
        let h1 = self.down1.forward(&h, &t_emb);      // [batch, c1, H, W]
        let h = self.pool1.forward(&h1);               // [batch, c1, H/2, W/2]
        
        let h2 = self.down2.forward(&h, &t_emb);      // [batch, c2, H/2, W/2]
        let h = self.pool2.forward(&h2);               // [batch, c2, H/4, W/4]
        
        let h3 = self.down3.forward(&h, &t_emb);      // [batch, c3, H/4, W/4]
        
        // 4. Middle
        let h = self.mid1.forward(&h3, &t_emb);
        let h = self.mid2.forward(&h, &t_emb);
        
        // 5. Decoder with skip connections
        let h = Tensor::concat(&[&h, &h3], 1);         // Concatenate along channel dim
        let h = self.up3.forward(&h, &t_emb);
        let h = self.upsample2.forward(&h);
        
        let h = Tensor::concat(&[&h, &h2], 1);
        let h = self.up2.forward(&h, &t_emb);
        let h = self.upsample1.forward(&h);
        
        let h = Tensor::concat(&[&h, &h1], 1);
        let h = self.up1.forward(&h, &t_emb);
        
        // 6. Final output
        let h = self.final_norm.forward(&h);
        let h = h.silu();
        self.final_conv.forward(&h)
    }

    /// Get all trainable parameters
    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![];
        
        params.extend(self.time_mlp1.parameters());
        params.extend(self.time_mlp2.parameters());
        params.extend(self.init_conv.parameters());
        
        params.extend(self.down1.parameters());
        params.extend(self.down2.parameters());
        params.extend(self.down3.parameters());
        params.extend(self.pool1.parameters());
        params.extend(self.pool2.parameters());
        
        params.extend(self.mid1.parameters());
        params.extend(self.mid2.parameters());
        
        params.extend(self.up3.parameters());
        params.extend(self.up2.parameters());
        params.extend(self.up1.parameters());
        params.extend(self.upsample2.parameters());
        params.extend(self.upsample1.parameters());
        
        params.extend(self.final_norm.parameters());
        params.extend(self.final_conv.parameters());
        
        params
    }

    /// Count total number of parameters
    pub fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.numel()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resblock() {
        let block = ResBlock::new(32, 64, 128);
        let x = Tensor::randn(&[2, 32, 16, 16]);
        let t = Tensor::randn(&[2, 128]);
        let y = block.forward(&x, &t);
        assert_eq!(y.shape(), &[2, 64, 16, 16]);
    }

    #[test]
    fn test_downsample() {
        let down = Downsample::new(64);
        let x = Tensor::randn(&[2, 64, 32, 32]);
        let y = down.forward(&x);
        assert_eq!(y.shape(), &[2, 64, 16, 16]);
    }

    #[test]
    fn test_upsample() {
        let up = Upsample::new(64);
        let x = Tensor::randn(&[2, 64, 16, 16]);
        let y = up.forward(&x);
        assert_eq!(y.shape(), &[2, 64, 32, 32]);
    }

    #[test]
    fn test_unet_forward() {
        let unet = UNet::new(3, 32, 3);
        let x = Tensor::randn(&[1, 3, 32, 32]);
        let timesteps = vec![100];
        let y = unet.forward(&x, &timesteps);
        assert_eq!(y.shape(), &[1, 3, 32, 32]);
        
        println!("UNet parameters: {}", unet.num_parameters());
    }
}
