//! Generation binary - Generate images using a trained model
//!
//! Usage: cargo run --bin generate -- [options]

use mini_diffusion::{
    DiffusionConfig, NoiseScheduler, UNet,
    sampling::{Sampler, SamplerConfig, save_images},
};

fn main() {
    println!("Mini Diffusion - Image Generation");
    println!("==================================\n");

    // Configuration
    let diffusion_config = DiffusionConfig {
        num_timesteps: 1000,
        beta_start: 1e-4,
        beta_end: 0.02,
        schedule: "cosine".to_string(),
    };

    let sampler_config = SamplerConfig {
        num_steps: 50,       // DDIM allows fewer steps
        guidance_scale: 1.0, // No guidance for unconditional
        use_ddim: true,      // Faster, deterministic
        eta: 0.0,            // Fully deterministic
    };

    // Image settings
    let batch_size = 4;
    let image_size = 32;
    let model_channels = 64;

    println!("Sampler Configuration:");
    println!("  - Steps: {} (out of {} training steps)", sampler_config.num_steps, diffusion_config.num_timesteps);
    println!("  - Method: {}", if sampler_config.use_ddim { "DDIM" } else { "DDPM" });
    println!("  - Eta: {}", sampler_config.eta);
    println!();

    // Create model (in real usage, load from checkpoint)
    println!("Creating model...");
    let model = UNet::new(3, model_channels, 3);
    println!("  - Parameters: {}", model.num_parameters());
    println!();

    // Note: Model has random weights - output will be noise
    println!("Note: Model has random weights (no training)");
    println!("      Output will look like colored noise.\n");

    // Create sampler
    let noise_scheduler = NoiseScheduler::new(diffusion_config);
    let sampler = Sampler::new(sampler_config, noise_scheduler);

    // Generate images
    println!("Generating {} images of size {}x{}...", batch_size, image_size, image_size);
    let generated = sampler.sample(&model, &[batch_size, 3, image_size, image_size]);

    println!("\nGeneration complete!");
    println!("  - Output shape: {:?}", generated.shape());
    let data = generated.to_vec();
    println!("  - Value range: [{:.3}, {:.3}]", 
        data.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
        data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
    );

    // Save images
    println!("\nSaving images...");
    match save_images(&generated, "generated") {
        Ok(()) => println!("Images saved successfully!"),
        Err(e) => println!("Error saving images: {}", e),
    }

    println!("\n==================================");
    println!("Generation demo complete!");
}
