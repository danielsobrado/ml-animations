//! Tensor implementation - the foundation of our neural network
//!
//! We build on top of ndarray but add neural-network friendly operations.

use ndarray::{Array, Axis, IxDyn};
use ndarray_rand::RandomExt;
use rand_distr::{Normal, Uniform};
// std::ops traits are used by Tensor internally

/// A multi-dimensional tensor for neural network operations
#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Array<f32, IxDyn>,
}

impl Tensor {
    /// Create a new tensor from an ndarray
    pub fn new(data: Array<f32, IxDyn>) -> Self {
        Tensor { data }
    }

    /// Create a tensor filled with zeros
    pub fn zeros(shape: &[usize]) -> Self {
        Tensor {
            data: Array::zeros(IxDyn(shape)),
        }
    }

    /// Create a tensor filled with ones
    pub fn ones(shape: &[usize]) -> Self {
        Tensor {
            data: Array::ones(IxDyn(shape)),
        }
    }

    /// Create a tensor with random values from normal distribution
    pub fn randn(shape: &[usize]) -> Self {
        let normal = Normal::new(0.0, 1.0).unwrap();
        Tensor {
            data: Array::random(IxDyn(shape), normal),
        }
    }

    /// Create a tensor with random values uniformly distributed
    pub fn rand_uniform(shape: &[usize], low: f32, high: f32) -> Self {
        let uniform = Uniform::new(low, high);
        Tensor {
            data: Array::random(IxDyn(shape), uniform),
        }
    }

    /// Create a tensor from a Vec with the given shape
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Self {
        let expected_len: usize = shape.iter().product();
        assert_eq!(data.len(), expected_len, 
            "Data length {} doesn't match shape {:?} (expected {})", 
            data.len(), shape, expected_len);
        Tensor {
            data: Array::from_shape_vec(IxDyn(shape), data).unwrap(),
        }
    }

    /// Xavier/Glorot initialization - good for linear layers
    pub fn xavier_init(fan_in: usize, fan_out: usize) -> Self {
        let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();
        Self::rand_uniform(&[fan_in, fan_out], -limit, limit)
    }

    /// Kaiming/He initialization - good for ReLU networks
    pub fn kaiming_init(fan_in: usize, fan_out: usize) -> Self {
        let std = (2.0 / fan_in as f32).sqrt();
        let normal = Normal::new(0.0, std).unwrap();
        Tensor {
            data: Array::random(IxDyn(&[fan_in, fan_out]), normal),
        }
    }

    /// Get the shape of the tensor
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// Get total number of elements
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Reshape the tensor
    pub fn reshape(&self, new_shape: &[usize]) -> Self {
        // First make contiguous, then reshape
        let contiguous = self.to_contiguous();
        Tensor {
            data: contiguous.data.into_shape(IxDyn(new_shape)).unwrap(),
        }
    }

    /// Make tensor contiguous in memory
    pub fn to_contiguous(&self) -> Self {
        if self.data.is_standard_layout() {
            self.clone()
        } else {
            Tensor {
                data: self.data.as_standard_layout().to_owned(),
            }
        }
    }

    /// Transpose last two dimensions (for matrix multiplication)
    pub fn transpose(&self) -> Self {
        let ndim = self.data.ndim();
        let mut axes: Vec<usize> = (0..ndim).collect();
        if ndim >= 2 {
            axes.swap(ndim - 1, ndim - 2);
        }
        Tensor {
            data: self.data.clone().permuted_axes(axes),
        }
    }

    /// Matrix multiplication (for 2D tensors)
    pub fn matmul(&self, other: &Tensor) -> Self {
        // For simplicity, handle 2D case
        let a = self.data.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
        let b = other.data.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
        let result = a.dot(&b);
        Tensor {
            data: result.into_dyn(),
        }
    }

    /// Element-wise addition
    pub fn add(&self, other: &Tensor) -> Self {
        Tensor {
            data: &self.data + &other.data,
        }
    }

    /// Add scalar
    pub fn add_scalar(&self, scalar: f32) -> Self {
        Tensor {
            data: &self.data + scalar,
        }
    }

    /// Element-wise subtraction
    pub fn sub(&self, other: &Tensor) -> Self {
        Tensor {
            data: &self.data - &other.data,
        }
    }

    /// Element-wise multiplication (Hadamard product)
    pub fn mul(&self, other: &Tensor) -> Self {
        Tensor {
            data: &self.data * &other.data,
        }
    }

    /// Multiply by scalar
    pub fn mul_scalar(&self, scalar: f32) -> Self {
        Tensor {
            data: &self.data * scalar,
        }
    }

    /// Element-wise division
    pub fn div(&self, other: &Tensor) -> Self {
        Tensor {
            data: &self.data / &other.data,
        }
    }

    /// Divide by scalar
    pub fn div_scalar(&self, scalar: f32) -> Self {
        Tensor {
            data: &self.data / scalar,
        }
    }

    /// Square root element-wise
    pub fn sqrt(&self) -> Self {
        Tensor {
            data: self.data.mapv(|x| x.sqrt()),
        }
    }

    /// Square element-wise
    pub fn square(&self) -> Self {
        Tensor {
            data: self.data.mapv(|x| x * x),
        }
    }

    /// Exponential element-wise
    pub fn exp(&self) -> Self {
        Tensor {
            data: self.data.mapv(|x| x.exp()),
        }
    }

    /// Natural log element-wise
    pub fn ln(&self) -> Self {
        Tensor {
            data: self.data.mapv(|x| x.ln()),
        }
    }

    /// Sum all elements
    pub fn sum(&self) -> f32 {
        self.data.sum()
    }

    /// Mean of all elements
    pub fn mean(&self) -> f32 {
        self.data.mean().unwrap()
    }

    /// Sum along axis
    pub fn sum_axis(&self, axis: usize) -> Self {
        Tensor {
            data: self.data.sum_axis(Axis(axis)).into_dyn(),
        }
    }

    /// Mean along axis
    pub fn mean_axis(&self, axis: usize) -> Self {
        Tensor {
            data: self.data.mean_axis(Axis(axis)).unwrap().into_dyn(),
        }
    }

    /// ReLU activation
    pub fn relu(&self) -> Self {
        Tensor {
            data: self.data.mapv(|x| x.max(0.0)),
        }
    }

    /// Sigmoid activation
    pub fn sigmoid(&self) -> Self {
        Tensor {
            data: self.data.mapv(|x| 1.0 / (1.0 + (-x).exp())),
        }
    }

    /// Tanh activation
    pub fn tanh(&self) -> Self {
        Tensor {
            data: self.data.mapv(|x| x.tanh()),
        }
    }

    /// GELU activation (used in transformers)
    pub fn gelu(&self) -> Self {
        Tensor {
            data: self.data.mapv(|x| {
                0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x.powi(3))).tanh())
            }),
        }
    }

    /// SiLU/Swish activation (used in modern diffusion models)
    pub fn silu(&self) -> Self {
        Tensor {
            data: self.data.mapv(|x| x / (1.0 + (-x).exp())),
        }
    }

    /// Softmax along last axis
    pub fn softmax(&self) -> Self {
        let max_val = self.data.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_data = self.data.mapv(|x| (x - max_val).exp());
        let sum = exp_data.sum();
        Tensor {
            data: exp_data / sum,
        }
    }

    /// Clamp values to range
    pub fn clamp(&self, min: f32, max: f32) -> Self {
        Tensor {
            data: self.data.mapv(|x| x.clamp(min, max)),
        }
    }

    /// Get underlying data as slice (panics if not contiguous - use to_vec for safety)
    pub fn as_slice(&self) -> &[f32] {
        self.data.as_slice().expect("Tensor not contiguous - use to_contiguous() first or to_vec()")
    }

    /// Get underlying data as Vec (always works, but copies data)
    pub fn to_vec(&self) -> Vec<f32> {
        self.data.iter().cloned().collect()
    }

    /// Get mutable slice (panics if not contiguous)
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        self.data.as_slice_mut().expect("Tensor not contiguous - use to_contiguous() first")
    }

    /// Concatenate tensors along axis
    pub fn concat(tensors: &[&Tensor], axis: usize) -> Self {
        let views: Vec<_> = tensors.iter().map(|t| t.data.view()).collect();
        let result = ndarray::concatenate(Axis(axis), &views).unwrap();
        Tensor {
            data: result.into_dyn(),
        }
    }

    /// Broadcast to shape
    pub fn broadcast(&self, shape: &[usize]) -> Self {
        let broadcast = self.data.broadcast(IxDyn(shape)).unwrap();
        Tensor {
            data: broadcast.to_owned(),
        }
    }

    /// Subtract a scalar from all elements
    pub fn sub_scalar(&self, val: f32) -> Self {
        Tensor {
            data: self.data.mapv(|x| x - val),
        }
    }

    /// Element-wise power
    pub fn pow(&self, exp: f32) -> Self {
        Tensor {
            data: self.data.mapv(|x| x.powf(exp)),
        }
    }

    /// Pad tensor with zeros (2D/4D support)
    /// 
    /// For 4D [B, H, W, C] with padding [top, bottom, left, right]
    pub fn pad(&self, padding: &[usize]) -> Self {
        let shape = self.shape();
        match shape.len() {
            4 => {
                // Assuming [B, H, W, C] format and padding [top, bottom, left, right]
                let (batch, height, width, channels) = (shape[0], shape[1], shape[2], shape[3]);
                let (top, bottom, left, right) = if padding.len() == 4 {
                    (padding[0], padding[1], padding[2], padding[3])
                } else {
                    (0, 0, 0, 0)
                };
                
                let new_h = height + top + bottom;
                let new_w = width + left + right;
                
                let mut output = vec![0.0f32; batch * new_h * new_w * channels];
                let data = self.to_vec();
                
                for b in 0..batch {
                    for h in 0..height {
                        for w in 0..width {
                            for c in 0..channels {
                                let src_idx = ((b * height + h) * width + w) * channels + c;
                                let dst_h = h + top;
                                let dst_w = w + left;
                                let dst_idx = ((b * new_h + dst_h) * new_w + dst_w) * channels + c;
                                output[dst_idx] = data[src_idx];
                            }
                        }
                    }
                }
                
                Tensor::from_vec(output, &[batch, new_h, new_w, channels])
            }
            _ => self.clone() // Return unchanged for unsupported dimensions
        }
    }

    /// Nearest-neighbor upsampling by scale factor
    pub fn upsample_nearest(&self, scale: usize) -> Self {
        let shape = self.shape();
        match shape.len() {
            4 => {
                let (batch, height, width, channels) = (shape[0], shape[1], shape[2], shape[3]);
                let new_h = height * scale;
                let new_w = width * scale;
                
                let mut output = vec![0.0f32; batch * new_h * new_w * channels];
                let data = self.to_vec();
                
                for b in 0..batch {
                    for h in 0..new_h {
                        for w in 0..new_w {
                            for c in 0..channels {
                                let src_h = h / scale;
                                let src_w = w / scale;
                                let src_idx = ((b * height + src_h) * width + src_w) * channels + c;
                                let dst_idx = ((b * new_h + h) * new_w + w) * channels + c;
                                output[dst_idx] = data[src_idx];
                            }
                        }
                    }
                }
                
                Tensor::from_vec(output, &[batch, new_h, new_w, channels])
            }
            _ => self.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let t = Tensor::zeros(&[2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.numel(), 6);
    }

    #[test]
    fn test_tensor_operations() {
        let a = Tensor::ones(&[2, 2]);
        let b = Tensor::ones(&[2, 2]).mul_scalar(2.0);
        let c = a.add(&b);
        assert_eq!(c.mean(), 3.0);
    }

    #[test]
    fn test_matmul() {
        let a = Tensor::ones(&[2, 3]);
        let b = Tensor::ones(&[3, 4]);
        let c = a.matmul(&b);
        assert_eq!(c.shape(), &[2, 4]);
        assert_eq!(c.mean(), 3.0);
    }

    #[test]
    fn test_activations() {
        let t = Tensor::randn(&[10]);
        let _ = t.relu();
        let _ = t.sigmoid();
        let _ = t.gelu();
        let _ = t.silu();
    }
}
