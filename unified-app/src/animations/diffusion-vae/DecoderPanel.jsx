import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, ChevronRight, Sparkles } from 'lucide-react';

export default function DecoderPanel() {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [generationMode, setGenerationMode] = useState(false);
  const intervalRef = useRef(null);

  const steps = [
    {
      title: 'Latent Vector z',
      description: 'The sampled latent vector z (or a generated sample from N(0,I)) enters the decoder.',
      highlight: 'latent'
    },
    {
      title: 'First Hidden Layer',
      description: 'Linear transformation expands z from 20 dimensions to 200, beginning the reconstruction process.',
      highlight: 'hidden1'
    },
    {
      title: 'Second Hidden Layer',
      description: 'Further expansion to 400 dimensions with ReLU activation, adding capacity for complex features.',
      highlight: 'hidden2'
    },
    {
      title: 'Output Layer',
      description: 'Final linear layer maps to 784 dimensions (28×28 image). Sigmoid squashes values to [0,1] for pixel intensities.',
      highlight: 'output'
    },
    {
      title: 'Reconstructed Image',
      description: 'The 784-dimensional output is reshaped to 28×28 to form the reconstructed (or generated) image.',
      highlight: 'image'
    }
  ];

  useEffect(() => {
    if (isPlaying) {
      intervalRef.current = setInterval(() => {
        setCurrentStep(prev => {
          if (prev >= steps.length - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, 2000);
    }
    return () => clearInterval(intervalRef.current);
  }, [isPlaying]);

  const reset = () => {
    setCurrentStep(0);
    setIsPlaying(false);
  };

  const NeuronLayer = ({ count, width, color, label, active, sublabel }) => (
    <div className="flex flex-col items-center">
      <div 
        className={`relative transition-all duration-500 ${active ? 'scale-110' : 'scale-100'}`}
        style={{ width: `${width}px` }}
      >
        <div className={`h-32 rounded-lg ${color} ${active ? 'ring-2 ring-white ring-offset-2 ring-offset-transparent' : ''} flex flex-col items-center justify-center transition-all`}>
          <p className="text-2xl font-bold">{count}</p>
          <p className="text-xs opacity-80">neurons</p>
        </div>
      </div>
      <p className={`mt-2 text-sm font-medium transition-colors ${active ? 'text-white' : 'text-gray-400'}`}>{label}</p>
      {sublabel && <p className="text-xs text-gray-700 dark:text-gray-500">{sublabel}</p>}
    </div>
  );

  const generateRandomImage = () => {
    return Array(16).fill(0).map(() => Math.random() > 0.4 ? 1 : 0);
  };

  const [randomImage, setRandomImage] = useState(generateRandomImage());

  return (
    <div className="space-y-6">
      {/* Title */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">
          Decoder: <span className="text-orange-600 dark:text-orange-400">From Latent to Data</span>
        </h2>
        <p className="text-gray-800 dark:text-gray-400">
          The decoder learns to reconstruct data from latent representations
        </p>
      </div>

      {/* Mode Toggle */}
      <div className="flex justify-center gap-4">
        <button
          onClick={() => setGenerationMode(false)}
          className={`px-4 py-2 rounded-lg font-medium transition-all ${
            !generationMode ? 'bg-orange-600 text-white' : 'bg-white/10 text-gray-800 dark:text-gray-400'
          }`}
        >
          Reconstruction Mode
        </button>
        <button
          onClick={() => { setGenerationMode(true); setRandomImage(generateRandomImage()); }}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all ${
            generationMode ? 'bg-purple-600 text-white' : 'bg-white/10 text-gray-800 dark:text-gray-400'
          }`}
        >
          <Sparkles size={18} />
          Generation Mode
        </button>
      </div>

      {/* Mode Description */}
      <div className={`rounded-xl p-4 border ${generationMode ? 'bg-purple-900/20 border-purple-500/30' : 'bg-orange-900/20 border-orange-500/30'}`}>
        {generationMode ? (
          <p className="text-gray-700 dark:text-gray-300">
            <strong className="text-purple-600 dark:text-purple-400">Generation Mode:</strong> Sample z ~ N(0, I) directly (no encoder needed!) 
            and pass through decoder to generate new data that looks like the training distribution.
          </p>
        ) : (
          <p className="text-gray-700 dark:text-gray-300">
            <strong className="text-orange-600 dark:text-orange-400">Reconstruction Mode:</strong> The encoder produces z from input x, 
            then the decoder reconstructs x̂ ≈ x. Loss = how different is x̂ from x?
          </p>
        )}
      </div>

      {/* Controls */}
      <div className="flex justify-center gap-3">
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          className="flex items-center gap-2 px-4 py-2 bg-orange-600 hover:bg-orange-700 rounded-lg transition-colors"
        >
          {isPlaying ? <Pause size={18} /> : <Play size={18} />}
          {isPlaying ? 'Pause' : 'Play Animation'}
        </button>
        <button
          onClick={reset}
          className="flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 rounded-lg transition-colors"
        >
          <RotateCcw size={18} />
          Reset
        </button>
        {generationMode && (
          <button
            onClick={() => setRandomImage(generateRandomImage())}
            className="flex items-center gap-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg transition-colors"
          >
            <Sparkles size={18} />
            New Sample
          </button>
        )}
      </div>

      {/* Step Progress */}
      <div className="flex justify-center gap-2">
        {steps.map((_, i) => (
          <button
            key={i}
            onClick={() => { setCurrentStep(i); setIsPlaying(false); }}
            className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium transition-all ${
              i === currentStep 
                ? 'bg-orange-500 text-white scale-110' 
                : i < currentStep 
                ? 'bg-orange-900 text-orange-300' 
                : 'bg-white/10 text-gray-700 dark:text-gray-500'
            }`}
          >
            {i + 1}
          </button>
        ))}
      </div>

      {/* Step Description */}
      <div className="bg-orange-900/20 border border-orange-500/30 rounded-xl p-4">
        <h3 className="font-bold text-orange-600 dark:text-orange-400">Step {currentStep + 1}: {steps[currentStep].title}</h3>
        <p className="text-gray-700 dark:text-gray-300 mt-1">{steps[currentStep].description}</p>
      </div>

      {/* Architecture Visualization */}
      <div className="bg-black/30 rounded-2xl p-8 border border-white/10">
        <div className="flex items-center justify-between gap-2">
          {/* Latent */}
          <div className={`transition-all duration-500 ${steps[currentStep].highlight === 'latent' ? 'scale-110' : ''}`}>
            <div className={`w-20 h-20 rounded-full flex flex-col items-center justify-center ${
              generationMode 
                ? 'bg-gradient-to-br from-purple-500 to-violet-700' 
                : 'bg-gradient-to-br from-violet-500 to-purple-700'
            } ${steps[currentStep].highlight === 'latent' ? 'ring-2 ring-white sample-bounce' : ''}`}>
              <p className="text-lg font-bold">z</p>
              <p className="text-xs opacity-80">{generationMode ? '~N(0,I)' : 'encoded'}</p>
            </div>
            <p className={`text-center text-sm mt-2 ${steps[currentStep].highlight === 'latent' ? 'text-purple-300' : 'text-gray-500'}`}>
              {generationMode ? 'Sampled' : 'Latent'}
            </p>
            <p className="text-center text-xs text-gray-700 dark:text-gray-500">[batch, 20]</p>
          </div>

          <ChevronRight className="text-gray-800 dark:text-gray-600" />

          <NeuronLayer 
            count={200} 
            width={50} 
            color="bg-gradient-to-b from-orange-600 to-orange-800"
            label="Hidden 1"
            sublabel="ReLU"
            active={steps[currentStep].highlight === 'hidden1'}
          />

          <ChevronRight className="text-gray-800 dark:text-gray-600" />

          <NeuronLayer 
            count={400} 
            width={60} 
            color="bg-gradient-to-b from-orange-600 to-orange-800"
            label="Hidden 2"
            sublabel="ReLU"
            active={steps[currentStep].highlight === 'hidden2'}
          />

          <ChevronRight className="text-gray-800 dark:text-gray-600" />

          <NeuronLayer 
            count={784} 
            width={80} 
            color="bg-gradient-to-b from-red-600 to-red-800"
            label="Output"
            sublabel="Sigmoid"
            active={steps[currentStep].highlight === 'output'}
          />

          <ChevronRight className="text-gray-800 dark:text-gray-600" />

          {/* Reconstructed Image */}
          <div className={`transition-all duration-500 ${steps[currentStep].highlight === 'image' ? 'scale-110' : ''}`}>
            <div className={`w-24 h-24 rounded-xl bg-gradient-to-br from-pink-500 to-rose-600 flex items-center justify-center ${steps[currentStep].highlight === 'image' ? 'ring-2 ring-white' : ''}`}>
              <div className="grid grid-cols-4 gap-0.5">
                {randomImage.map((cell, i) => (
                  <div key={i} className={`w-3 h-3 rounded-sm transition-all ${cell ? 'bg-white/90' : 'bg-white/20'}`} />
                ))}
              </div>
            </div>
            <p className={`text-center text-sm mt-2 ${steps[currentStep].highlight === 'image' ? 'text-pink-300' : 'text-gray-500'}`}>
              {generationMode ? 'Generated' : 'Reconstructed'}
            </p>
            <p className="text-center text-xs text-gray-700 dark:text-gray-500">28×28</p>
          </div>
        </div>
      </div>

      {/* Code Example */}
      <div className="bg-black/40 rounded-xl p-4 border border-white/10">
        <p className="text-sm text-gray-800 dark:text-gray-400 mb-3">PyTorch Decoder Implementation:</p>
        <pre className="text-sm overflow-x-auto">
          <code className="text-orange-300">{`class Decoder(nn.Module):
    def __init__(self, latent_dim=20, hidden_dim=400, output_dim=784):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        # Sigmoid for pixel values in [0, 1]
        return torch.sigmoid(self.fc_out(h))

# Generation: sample from prior and decode
def generate(decoder, num_samples=10, latent_dim=20):
    z = torch.randn(num_samples, latent_dim)  # Sample z ~ N(0, I)
    with torch.no_grad():
        generated = decoder(z)
    return generated.view(-1, 1, 28, 28)  # Reshape to images`}</code>
        </pre>
      </div>

      {/* Key Points */}
      <div className="grid md:grid-cols-2 gap-4">
        <div className="bg-orange-900/20 rounded-xl p-4 border border-orange-500/30">
          <h4 className="font-bold text-orange-300 mb-2">Why Sigmoid Output?</h4>
          <p className="text-sm text-gray-700 dark:text-gray-300">
            For images normalized to [0,1], sigmoid constrains outputs to valid pixel values. 
            This also works well with Binary Cross-Entropy loss for reconstruction.
          </p>
        </div>
        <div className="bg-purple-900/20 rounded-xl p-4 border border-purple-500/30">
          <h4 className="font-bold text-purple-300 mb-2">Decoder as Generator</h4>
          <p className="text-sm text-gray-700 dark:text-gray-300">
            After training, the decoder alone is a generative model! Sample z randomly,
            decode it, and you get novel samples from the learned data distribution.
          </p>
        </div>
      </div>

      {/* Architecture Variants */}
      <div className="bg-black/30 rounded-xl p-4 border border-white/10">
        <h4 className="font-bold mb-3">🏗️ Decoder Architecture Variants</h4>
        <div className="grid md:grid-cols-3 gap-4">
          <div className="bg-white/5 rounded-lg p-3">
            <h5 className="font-medium text-orange-600 dark:text-orange-400">MLP Decoder</h5>
            <p className="text-xs text-gray-800 dark:text-gray-400 mt-1">Fully connected layers. Simple but works for small images like MNIST.</p>
          </div>
          <div className="bg-white/5 rounded-lg p-3">
            <h5 className="font-medium text-orange-600 dark:text-orange-400">Transposed Conv (Deconv)</h5>
            <p className="text-xs text-gray-800 dark:text-gray-400 mt-1">ConvTranspose2d layers for upsampling. Better for larger images.</p>
          </div>
          <div className="bg-white/5 rounded-lg p-3">
            <h5 className="font-medium text-orange-600 dark:text-orange-400">Upsample + Conv</h5>
            <p className="text-xs text-gray-800 dark:text-gray-400 mt-1">Nearest/bilinear upsample followed by conv. Avoids checkerboard artifacts.</p>
          </div>
        </div>
      </div>
    </div>
  );
}
