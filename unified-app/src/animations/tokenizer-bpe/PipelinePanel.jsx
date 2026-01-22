import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, ArrowRight, ChevronRight } from 'lucide-react';
import gsap from 'gsap';

function PipelinePanel() {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const containerRef = useRef(null);
  const timelineRef = useRef(null);

  const inputText = "a cat sitting on a couch";

  const steps = [
    {
      id: 0,
      title: 'Input Text',
      description: 'Raw prompt enters the pipeline',
      visual: 'text',
      data: inputText
    },
    {
      id: 1,
      title: 'Preprocessing',
      description: 'Lowercase conversion and normalization',
      visual: 'preprocess',
      data: inputText.toLowerCase()
    },
    {
      id: 2,
      title: 'BPE Tokenization',
      description: 'Split into subword tokens',
      visual: 'tokens',
      data: ['a</w>', 'cat</w>', 'sitting</w>', 'on</w>', 'a</w>', 'couch</w>']
    },
    {
      id: 3,
      title: 'Add Special Tokens',
      description: 'Wrap with start and end tokens',
      visual: 'special',
      data: ['<|startoftext|>', 'a</w>', 'cat</w>', 'sitting</w>', 'on</w>', 'a</w>', 'couch</w>', '<|endoftext|>']
    },
    {
      id: 4,
      title: 'Token → ID',
      description: 'Look up each token in vocabulary',
      visual: 'ids',
      data: [49406, 320, 2368, 4919, 529, 320, 3725, 49407]
    },
    {
      id: 5,
      title: 'Padding',
      description: 'Pad to fixed length (77 tokens)',
      visual: 'padded',
      data: { ids: [49406, 320, 2368, 4919, 529, 320, 3725, 49407], padded: 69 }
    },
    {
      id: 6,
      title: 'Ready for Encoder',
      description: 'Token IDs ready for embedding lookup',
      visual: 'final',
      data: 'Shape: [1, 77] → Embedding [1, 77, 768]'
    }
  ];

  useEffect(() => {
    if (isPlaying && currentStep < steps.length - 1) {
      const timer = setTimeout(() => {
        setCurrentStep(prev => prev + 1);
      }, 1500);
      return () => clearTimeout(timer);
    } else if (currentStep >= steps.length - 1) {
      setIsPlaying(false);
    }
  }, [isPlaying, currentStep]);

  useEffect(() => {
    if (containerRef.current) {
      gsap.fromTo('.step-content',
        { opacity: 0, x: 20 },
        { opacity: 1, x: 0, duration: 0.4 }
      );
    }
  }, [currentStep]);

  const reset = () => {
    setIsPlaying(false);
    setCurrentStep(0);
  };

  const renderStepVisual = () => {
    const step = steps[currentStep];

    switch (step.visual) {
      case 'text':
        return (
          <div className="text-center p-8">
            <div className="text-sm text-gray-800 dark:text-gray-400 mb-2">Raw Text Input</div>
            <div className="text-2xl text-white font-mono bg-black/40 px-6 py-4 rounded-lg inline-block">
              "{step.data}"
            </div>
          </div>
        );
      
      case 'preprocess':
        return (
          <div className="text-center p-8">
            <div className="text-sm text-gray-800 dark:text-gray-400 mb-2">After Preprocessing</div>
            <div className="text-2xl text-orange-300 font-mono bg-black/40 px-6 py-4 rounded-lg inline-block">
              "{step.data}"
            </div>
            <div className="mt-2 text-xs text-gray-700 dark:text-gray-500">Converted to lowercase</div>
          </div>
        );
      
      case 'tokens':
        return (
          <div className="text-center p-6">
            <div className="text-sm text-gray-800 dark:text-gray-400 mb-4">BPE Tokens</div>
            <div className="flex flex-wrap justify-center gap-2">
              {step.data.map((token, i) => (
                <span key={i} className="px-3 py-2 bg-blue-500/20 text-blue-300 rounded-lg font-mono text-sm">
                  {token}
                </span>
              ))}
            </div>
          </div>
        );
      
      case 'special':
        return (
          <div className="text-center p-6">
            <div className="text-sm text-gray-800 dark:text-gray-400 mb-4">With Special Tokens</div>
            <div className="flex flex-wrap justify-center gap-2">
              {step.data.map((token, i) => (
                <span key={i} className={`px-3 py-2 rounded-lg font-mono text-sm ${
                  token.includes('|') ? 'bg-red-500/20 text-red-300' : 'bg-blue-500/20 text-blue-300'
                }`}>
                  {token}
                </span>
              ))}
            </div>
          </div>
        );
      
      case 'ids':
        return (
          <div className="text-center p-6">
            <div className="text-sm text-gray-800 dark:text-gray-400 mb-4">Token IDs</div>
            <div className="flex flex-wrap justify-center gap-2">
              {step.data.map((id, i) => (
                <div key={i} className="flex flex-col items-center">
                  <span className={`px-3 py-2 rounded-lg font-mono text-sm ${
                    id === 49406 || id === 49407 ? 'bg-red-500/20 text-red-300' : 'bg-yellow-500/20 text-yellow-300'
                  }`}>
                    {id}
                  </span>
                  <span className="text-xs text-gray-700 dark:text-gray-500 mt-1">[{i}]</span>
                </div>
              ))}
            </div>
          </div>
        );
      
      case 'padded':
        return (
          <div className="text-center p-6">
            <div className="text-sm text-gray-800 dark:text-gray-400 mb-4">Padded Sequence (77 tokens)</div>
            <div className="flex flex-wrap justify-center gap-1">
              {step.data.ids.map((id, i) => (
                <span key={i} className={`px-2 py-1 rounded text-xs font-mono ${
                  id === 49406 || id === 49407 ? 'bg-red-500/30 text-red-300' : 'bg-yellow-500/30 text-yellow-300'
                }`}>
                  {id}
                </span>
              ))}
              <span className="px-2 py-1 rounded text-xs font-mono bg-gray-500/30 text-gray-800 dark:text-gray-400">
                + {step.data.padded} × 49407
              </span>
            </div>
            <div className="mt-4 text-xs text-gray-700 dark:text-gray-500">
              {step.data.ids.length} actual tokens + {step.data.padded} padding tokens = 77 total
            </div>
          </div>
        );
      
      case 'final':
        return (
          <div className="text-center p-8">
            <div className="text-sm text-gray-800 dark:text-gray-400 mb-4">Ready for Text Encoder</div>
            <div className="text-xl text-green-300 font-mono bg-green-500/10 px-6 py-4 rounded-lg inline-block border border-green-500/30">
              {step.data}
            </div>
            <div className="mt-4 text-sm text-gray-800 dark:text-gray-400">
              Each token ID will be looked up in embedding table
            </div>
          </div>
        );
      
      default:
        return null;
    }
  };

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-orange-600 dark:text-orange-400 mb-2">Tokenization Pipeline</h2>
        <p className="text-gray-700 dark:text-gray-300 max-w-3xl mx-auto">
          Watch the complete tokenization process from raw text to encoder-ready token IDs.
        </p>
      </div>

      {/* Controls */}
      <div className="flex justify-center gap-4">
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          className="flex items-center gap-2 px-6 py-3 bg-orange-500 hover:bg-orange-600 text-white rounded-xl transition-colors"
        >
          {isPlaying ? <Pause size={18} /> : <Play size={18} />}
          {isPlaying ? 'Pause' : 'Play'}
        </button>
        <button
          onClick={reset}
          className="flex items-center gap-2 px-6 py-3 bg-black/30 hover:bg-black/50 text-white rounded-xl transition-colors"
        >
          <RotateCcw size={18} />
          Reset
        </button>
      </div>

      {/* Progress Steps */}
      <div className="flex justify-between items-center bg-black/30 rounded-xl p-4 overflow-x-auto">
        {steps.map((step, i) => (
          <React.Fragment key={step.id}>
            <button
              onClick={() => { setCurrentStep(i); setIsPlaying(false); }}
              className={`flex flex-col items-center min-w-[80px] p-2 rounded-lg transition-all ${
                i === currentStep ? 'bg-orange-500/20' : 'hover:bg-white/5'
              }`}
            >
              <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ${
                i < currentStep ? 'bg-green-500 text-white' :
                i === currentStep ? 'bg-orange-500 text-white' :
                'bg-gray-700 text-gray-800 dark:text-gray-400'
              }`}>
                {i + 1}
              </div>
              <div className={`text-xs mt-1 text-center ${
                i === currentStep ? 'text-orange-400' : 'text-gray-500'
              }`}>
                {step.title}
              </div>
            </button>
            {i < steps.length - 1 && (
              <ChevronRight size={16} className={`flex-shrink-0 ${
                i < currentStep ? 'text-green-500' : 'text-gray-600'
              }`} />
            )}
          </React.Fragment>
        ))}
      </div>

      {/* Current Step Display */}
      <div ref={containerRef} className="bg-black/40 rounded-xl p-6">
        <div className="step-content">
          <div className="text-center mb-4">
            <h3 className="text-xl font-semibold text-orange-600 dark:text-orange-400">
              Step {currentStep + 1}: {steps[currentStep].title}
            </h3>
            <p className="text-gray-800 dark:text-sm">{steps[currentStep].description}</p>
          </div>
          {renderStepVisual()}
        </div>
      </div>

      {/* Full Pipeline Code */}
      <div className="bg-black/40 rounded-xl p-6">
        <h3 className="text-gray-700 dark:text-gray-300 font-semibold mb-4">Complete Pipeline Code</h3>
        <div className="bg-black/60 rounded-lg p-4 font-mono text-sm overflow-x-auto">
          <pre className="text-gray-700 dark:text-gray-300">
{`# Full CLIP tokenization pipeline
def tokenize(text: str, max_length: int = 77) -> List[int]:
    # 1. Preprocess
    text = text.lower().strip()
    
    # 2. BPE tokenization
    tokens = bpe_encode(text)
    
    # 3. Add special tokens
    tokens = [START_TOKEN] + tokens + [END_TOKEN]
    
    # 4. Convert to IDs
    token_ids = [vocab[t] for t in tokens]
    
    # 5. Pad or truncate to max_length
    if len(token_ids) > max_length:
        token_ids = token_ids[:max_length-1] + [END_TOKEN_ID]
    else:
        token_ids += [END_TOKEN_ID] * (max_length - len(token_ids))
    
    return token_ids  # Shape: [77]`}
          </pre>
        </div>
      </div>
    </div>
  );
}

export default PipelinePanel;
