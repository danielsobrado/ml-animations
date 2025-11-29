//! Demo: SD3-Style Diffusion Components
//!
//! This demonstrates the new components:
//! - Tokenizers (BPE, Unigram)
//! - Flow Matching scheduler
//! - VAE concepts
//! - DiT concepts
//!
//! Note: Full integration requires weight loading and
//! shape tuning. This demo shows the concepts work individually.

use mini_diffusion::Tensor;

fn main() {
    println!("=== Mini-Diffusion: SD3-Style Components Demo ===\n");
    
    // 1. Tokenizers
    demo_tokenizers();
    
    // 2. Flow Matching
    demo_flow_matching();
    
    // 3. Tensor Operations for SD3
    demo_tensor_ops();
    
    println!("\n✅ All component demos completed successfully!");
}

fn demo_tokenizers() {
    println!("--- BPE Tokenizer (CLIP-style) ---");
    let bpe = mini_diffusion::BPETokenizer::new(77);
    
    let text = "a photo of a cat";
    let tokens = bpe.encode(text);
    println!("Input: \"{}\"", text);
    println!("Tokens: {:?}", tokens);
    println!("Vocab size: {}\n", bpe.vocab_size());
    
    println!("--- Unigram Tokenizer (T5-style) ---");
    let unigram = mini_diffusion::UnigramTokenizer::new(512);
    
    let tokens = unigram.encode(text);
    println!("Input: \"{}\"", text);
    println!("Tokens: {:?}", tokens);
    println!("Decoded: \"{}\"\n", unigram.decode(&tokens));
}

fn demo_flow_matching() {
    use mini_diffusion::flow::{FlowMatchingScheduler, EulerSolver, LogitNormalSampler};
    
    println!("--- Flow Matching Scheduler ---");
    let scheduler = FlowMatchingScheduler::new(20);
    
    println!("Number of inference steps: {}", scheduler.num_inference_steps);
    println!("Sigma schedule (first 5): {:?}", &scheduler.sigmas[..5.min(scheduler.sigmas.len())]);
    
    // Demonstrate logit-normal sampling
    let sampler = LogitNormalSampler::default();
    let timesteps: Vec<f32> = (0..5).map(|_| sampler.sample()).collect();
    println!("Sampled timesteps (logit-normal): {:?}", timesteps);
    
    // Demonstrate add_noise and velocity
    let data = Tensor::ones(&[2, 2]);
    let noise = Tensor::randn(&[2, 2]);
    
    let noisy = scheduler.add_noise(&data, &noise, 0.5);
    let velocity = scheduler.get_velocity(&data, &noise);
    
    println!("Data shape: {:?}", data.shape());
    println!("Noisy (σ=0.5) shape: {:?}", noisy.shape());
    println!("Velocity shape: {:?}", velocity.shape());
    
    println!("\n--- Euler Solver ---");
    let solver = EulerSolver::new(20);
    
    let x = Tensor::randn(&[4, 4]);
    let v = Tensor::randn(&[4, 4]);
    
    let x_next = solver.step(&x, &v, 0);
    println!("Step 0 -> 1: {:?} -> {:?}\n", x.shape(), x_next.shape());
}

fn demo_tensor_ops() {
    println!("--- New Tensor Operations ---");
    
    // from_vec
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    println!("from_vec([1,2,3,4], [2,2]): {:?}", t.to_vec());
    
    // sub_scalar
    let t2 = t.sub_scalar(1.0);
    println!("sub_scalar(1.0): {:?}", t2.to_vec());
    
    // pow
    let t3 = t.pow(2.0);
    println!("pow(2.0): {:?}", t3.to_vec());
    
    // Mean
    println!("mean(): {}", t.mean());
    
    // Pad (4D tensor)
    let t4d = Tensor::ones(&[1, 2, 2, 1]);
    let padded = t4d.pad(&[1, 1, 1, 1]); // top, bottom, left, right
    println!("pad([2,2] -> [4,4]): {:?}", padded.shape());
    
    // Upsample
    let upsampled = t4d.upsample_nearest(2);
    println!("upsample_nearest(2x): {:?} -> {:?}", t4d.shape(), upsampled.shape());
    
    println!();
}
