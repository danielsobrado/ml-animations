import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, ChevronRight } from 'lucide-react';

export default function EncoderPanel() {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const intervalRef = useRef(null);

  const steps = [
    {
      title: 'Input Image',
      description: 'A 28×28 grayscale image (like MNIST digit) is flattened into a 784-dimensional vector.',
      highlight: 'input'
    },
    {
      title: 'First Hidden Layer',
      description: 'Linear transformation + activation compresses 784 → 400 dimensions, learning important features.',
      highlight: 'hidden1'
    },
    {
      title: 'Second Hidden Layer',
      description: 'Further compression to 200 dimensions, extracting higher-level abstractions.',
      highlight: 'hidden2'
    },
    {
      title: 'Mean (μ) Output',
      description: 'A linear layer outputs the mean vector μ of the latent distribution. Each dimension is a center point.',
      highlight: 'mu'
    },
    {
      title: 'Log-Variance (log σ²) Output',
      description: 'A separate linear layer outputs log σ² (log-variance for numerical stability). σ = exp(0.5 × log σ²).',
      highlight: 'logvar'
    },
    {
      title: 'Reparameterization',
      description: 'Sample ε ~ N(0,1), then compute z = μ + σ ⊙ ε. This allows gradients to flow through sampling!',
      highlight: 'reparam'
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
      }, 2500);
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

  return (
    <div className="space-y-6">
      {/* Title */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">
          Encoder: <span className="text-green-400">From Data to Distribution</span>
        </h2>
        <p className="text-gray-800 dark:text-gray-400">
          The encoder learns to map input data to latent distribution parameters μ and σ
        </p>
      </div>

      {/* Controls */}
      <div className="flex justify-center gap-3">
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg transition-colors"
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
      </div>

      {/* Step Progress */}
      <div className="flex justify-center gap-2">
        {steps.map((_, i) => (
          <button
            key={i}
            onClick={() => { setCurrentStep(i); setIsPlaying(false); }}
            className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium transition-all ${
              i === currentStep 
                ? 'bg-green-500 text-white scale-110' 
                : i < currentStep 
                ? 'bg-green-900 text-green-300' 
                : 'bg-white/10 text-gray-700 dark:text-gray-500'
            }`}
          >
            {i + 1}
          </button>
        ))}
      </div>

      {/* Step Description */}
      <div className="bg-green-900/20 border border-green-500/30 rounded-xl p-4">
        <h3 className="font-bold text-green-400">Step {currentStep + 1}: {steps[currentStep].title}</h3>
        <p className="text-gray-700 dark:text-gray-300 mt-1">{steps[currentStep].description}</p>
      </div>

      {/* Architecture Visualization */}
      <div className="bg-black/30 rounded-2xl p-8 border border-white/10">
        <div className="flex items-center justify-between gap-2">
          <NeuronLayer 
            count={784} 
            width={80} 
            color="bg-gradient-to-b from-blue-600 to-blue-800"
            label="Input"
            sublabel="28×28 = 784"
            active={steps[currentStep].highlight === 'input'}
          />

          <ChevronRight className="text-gray-800 dark:text-gray-600" />

          <NeuronLayer 
            count={400} 
            width={60} 
            color="bg-gradient-to-b from-green-600 to-green-800"
            label="Hidden 1"
            sublabel="ReLU"
            active={steps[currentStep].highlight === 'hidden1'}
          />

          <ChevronRight className="text-gray-800 dark:text-gray-600" />

          <NeuronLayer 
            count={200} 
            width={50} 
            color="bg-gradient-to-b from-green-600 to-green-800"
            label="Hidden 2"
            sublabel="ReLU"
            active={steps[currentStep].highlight === 'hidden2'}
          />

          <ChevronRight className="text-gray-800 dark:text-gray-600" />

          {/* Split into mu and logvar */}
          <div className="flex flex-col gap-4">
            <div className={`transition-all duration-500 ${steps[currentStep].highlight === 'mu' ? 'scale-110' : ''}`}>
              <div className={`w-16 h-14 rounded-lg bg-gradient-to-br from-purple-500 to-purple-700 flex flex-col items-center justify-center ${steps[currentStep].highlight === 'mu' ? 'ring-2 ring-white' : ''}`}>
                <p className="text-lg font-bold">20</p>
                <p className="text-xs">μ</p>
              </div>
              <p className={`text-center text-xs mt-1 ${steps[currentStep].highlight === 'mu' ? 'text-purple-300' : 'text-gray-500'}`}>Mean</p>
            </div>
            <div className={`transition-all duration-500 ${steps[currentStep].highlight === 'logvar' ? 'scale-110' : ''}`}>
              <div className={`w-16 h-14 rounded-lg bg-gradient-to-br from-pink-500 to-pink-700 flex flex-col items-center justify-center ${steps[currentStep].highlight === 'logvar' ? 'ring-2 ring-white' : ''}`}>
                <p className="text-lg font-bold">20</p>
                <p className="text-xs">log σ²</p>
              </div>
              <p className={`text-center text-xs mt-1 ${steps[currentStep].highlight === 'logvar' ? 'text-pink-300' : 'text-gray-500'}`}>Log-Var</p>
            </div>
          </div>

          <ChevronRight className="text-gray-800 dark:text-gray-600" />

          {/* Reparameterization */}
          <div className={`transition-all duration-500 ${steps[currentStep].highlight === 'reparam' ? 'scale-110' : ''}`}>
            <div className={`w-20 h-20 rounded-full bg-gradient-to-br from-violet-500 to-purple-700 flex flex-col items-center justify-center ${steps[currentStep].highlight === 'reparam' ? 'ring-2 ring-white sample-bounce' : ''}`}>
              <p className="text-lg font-bold">z</p>
              <p className="text-xs opacity-80">latent</p>
            </div>
            <p className={`text-center text-xs mt-1 ${steps[currentStep].highlight === 'reparam' ? 'text-purple-300' : 'text-gray-500'}`}>Sample</p>
          </div>
        </div>

        {/* Reparameterization Formula */}
        {steps[currentStep].highlight === 'reparam' && (
          <div className="mt-6 p-4 bg-purple-900/30 rounded-xl border border-purple-500/30">
            <p className="text-center font-mono text-lg">
              <span className="text-purple-600 dark:text-purple-400">z</span> = 
              <span className="text-purple-300"> μ</span> + 
              <span className="text-pink-300"> σ</span> ⊙ 
              <span className="text-yellow-300"> ε</span>
              <span className="text-gray-800 dark:text-sm ml-2">(where ε ~ N(0, I))</span>
            </p>
            <p className="text-center text-sm text-gray-800 dark:text-gray-400 mt-2">
              σ = exp(0.5 × log σ²) — computing σ from log-variance for numerical stability
            </p>
          </div>
        )}
      </div>

      {/* Code Example */}
      <div className="bg-black/40 rounded-xl p-4 border border-white/10">
        <p className="text-sm text-gray-800 dark:text-gray-400 mb-3">PyTorch Encoder Implementation:</p>
        <pre className="text-sm overflow-x-auto">
          <code className="text-green-300">{`class Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)      # Mean
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)  # Log-variance
    
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # σ = exp(0.5 × log σ²)
        eps = torch.randn_like(std)     # ε ~ N(0, I)
        return mu + std * eps           # z = μ + σ ⊙ ε`}</code>
        </pre>
      </div>

      {/* Key Points */}
      <div className="grid md:grid-cols-2 gap-4">
        <div className="bg-purple-900/20 rounded-xl p-4 border border-purple-500/30">
          <h4 className="font-bold text-purple-300 mb-2">Why Two Outputs?</h4>
          <p className="text-sm text-gray-700 dark:text-gray-300">
            The encoder outputs both μ and log σ² because we're learning a <strong>distribution</strong>, 
            not a single point. Each latent dimension has its own mean and variance.
          </p>
        </div>
        <div className="bg-green-900/20 rounded-xl p-4 border border-green-500/30">
          <h4 className="font-bold text-green-300 mb-2">Why Log-Variance?</h4>
          <p className="text-sm text-gray-700 dark:text-gray-300">
            We output log σ² instead of σ because: (1) variance must be positive, but log can be any real number,
            (2) it's more numerically stable during training.
          </p>
        </div>
      </div>
    </div>
  );
}
