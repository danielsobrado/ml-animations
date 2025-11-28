import React, { useState, useEffect } from 'react';
import { Play, Pause, RotateCcw, ArrowRight, Scissors, Info } from 'lucide-react';

export default function TokenizationPanel() {
  const [inputText, setInputText] = useState("I love playing football!");
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const steps = [
    { title: 'Original Text', description: 'Start with raw input text' },
    { title: 'Basic Tokenization', description: 'Split by whitespace and punctuation' },
    { title: 'WordPiece Tokenization', description: 'Break unknown words into subwords' },
    { title: 'Add Special Tokens', description: 'Add [CLS] at start, [SEP] at end' },
    { title: 'Token IDs', description: 'Convert tokens to vocabulary indices' },
  ];

  // WordPiece simulation
  const tokenize = () => {
    const text = inputText.toLowerCase();
    // Simulated WordPiece tokenization
    const basicTokens = text.match(/[\w]+|[^\s\w]/g) || [];
    
    const wordPieceTokens = basicTokens.flatMap(token => {
      // Simulate breaking "playing" into "play" + "##ing"
      if (token === 'playing') return ['play', '##ing'];
      if (token === 'football') return ['foot', '##ball'];
      if (token === 'understanding') return ['under', '##stand', '##ing'];
      if (token === 'unbelievable') return ['un', '##believ', '##able'];
      return [token];
    });

    const withSpecial = ['[CLS]', ...wordPieceTokens, '[SEP]'];
    
    // Simulated token IDs
    const tokenIds = withSpecial.map((token, i) => {
      const vocab = {
        '[CLS]': 101, '[SEP]': 102, '[MASK]': 103, '[PAD]': 0,
        'i': 1045, 'love': 2293, 'play': 2377, '##ing': 2075,
        'foot': 3329, '##ball': 3608, 'the': 1996, 'cat': 4937,
        '!': 999, '.': 1012, ',': 1010
      };
      return vocab[token] || (1000 + i);
    });

    return { basicTokens, wordPieceTokens, withSpecial, tokenIds };
  };

  const { basicTokens, wordPieceTokens, withSpecial, tokenIds } = tokenize();

  useEffect(() => {
    if (isPlaying) {
      const interval = setInterval(() => {
        setCurrentStep(prev => {
          if (prev >= steps.length - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, 2000);
      return () => clearInterval(interval);
    }
  }, [isPlaying]);

  const reset = () => {
    setCurrentStep(0);
    setIsPlaying(false);
  };

  return (
    <div className="space-y-6">
      {/* Title */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">
          Tokenization: <span className="text-green-400">From Text to Tokens</span>
        </h2>
        <p className="text-gray-400">
          How BERT breaks down text into processable units using WordPiece
        </p>
      </div>

      {/* Input Section */}
      <div className="bg-black/30 rounded-xl p-4 border border-white/10">
        <label className="text-sm text-gray-400 mb-2 block">Try your own text:</label>
        <input
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          className="w-full bg-black/30 border border-white/20 rounded-lg px-4 py-3 text-lg focus:border-green-500 focus:outline-none transition-colors"
          placeholder="Enter text to tokenize..."
        />
      </div>

      {/* Controls */}
      <div className="flex justify-center gap-3">
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg transition-colors"
        >
          {isPlaying ? <Pause size={18} /> : <Play size={18} />}
          {isPlaying ? 'Pause' : 'Play Steps'}
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
                : 'bg-white/10 text-gray-500'
            }`}
          >
            {i + 1}
          </button>
        ))}
      </div>

      {/* Current Step Description */}
      <div className="bg-green-900/20 border border-green-500/30 rounded-xl p-4">
        <h3 className="font-bold text-green-400">Step {currentStep + 1}: {steps[currentStep].title}</h3>
        <p className="text-gray-300 mt-1">{steps[currentStep].description}</p>
      </div>

      {/* Visualization */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        {/* Step 0: Original */}
        <div className={`transition-all duration-500 ${currentStep >= 0 ? 'opacity-100' : 'opacity-30'}`}>
          <div className="flex items-center gap-2 mb-2">
            <span className="text-sm font-medium text-gray-400">Original Text:</span>
          </div>
          <div className="bg-blue-900/30 rounded-lg p-4 text-xl font-mono border border-blue-500/30">
            {inputText}
          </div>
        </div>

        {currentStep >= 1 && (
          <>
            <div className="flex justify-center my-4">
              <ArrowRight className="text-gray-500 rotate-90" />
            </div>
            
            {/* Step 1: Basic Tokens */}
            <div className={`transition-all duration-500 ${currentStep >= 1 ? 'opacity-100' : 'opacity-30'}`}>
              <div className="flex items-center gap-2 mb-2">
                <Scissors size={16} className="text-yellow-400" />
                <span className="text-sm font-medium text-gray-400">Basic Tokens:</span>
              </div>
              <div className="flex flex-wrap gap-2">
                {basicTokens.map((token, i) => (
                  <span 
                    key={i}
                    className="bg-yellow-600/30 border border-yellow-500/50 px-3 py-2 rounded-lg font-mono mask-reveal"
                    style={{ animationDelay: `${i * 100}ms` }}
                  >
                    {token}
                  </span>
                ))}
              </div>
            </div>
          </>
        )}

        {currentStep >= 2 && (
          <>
            <div className="flex justify-center my-4">
              <ArrowRight className="text-gray-500 rotate-90" />
            </div>
            
            {/* Step 2: WordPiece */}
            <div className={`transition-all duration-500 ${currentStep >= 2 ? 'opacity-100' : 'opacity-30'}`}>
              <div className="flex items-center gap-2 mb-2">
                <span className="text-sm font-medium text-gray-400">WordPiece Tokens:</span>
                <span className="text-xs text-purple-400">(## = continuation)</span>
              </div>
              <div className="flex flex-wrap gap-2">
                {wordPieceTokens.map((token, i) => (
                  <span 
                    key={i}
                    className={`px-3 py-2 rounded-lg font-mono mask-reveal ${
                      token.startsWith('##') 
                        ? 'bg-purple-600/30 border border-purple-500/50 text-purple-300' 
                        : 'bg-green-600/30 border border-green-500/50 text-green-300'
                    }`}
                    style={{ animationDelay: `${i * 100}ms` }}
                  >
                    {token}
                  </span>
                ))}
              </div>
            </div>
          </>
        )}

        {currentStep >= 3 && (
          <>
            <div className="flex justify-center my-4">
              <ArrowRight className="text-gray-500 rotate-90" />
            </div>
            
            {/* Step 3: With Special Tokens */}
            <div className={`transition-all duration-500 ${currentStep >= 3 ? 'opacity-100' : 'opacity-30'}`}>
              <div className="flex items-center gap-2 mb-2">
                <span className="text-sm font-medium text-gray-400">With Special Tokens:</span>
              </div>
              <div className="flex flex-wrap gap-2">
                {withSpecial.map((token, i) => (
                  <span 
                    key={i}
                    className={`px-3 py-2 rounded-lg font-mono mask-reveal ${
                      token.startsWith('[') 
                        ? 'bg-red-600/30 border border-red-500/50 text-red-300 font-bold' 
                        : token.startsWith('##')
                        ? 'bg-purple-600/30 border border-purple-500/50 text-purple-300'
                        : 'bg-green-600/30 border border-green-500/50 text-green-300'
                    }`}
                    style={{ animationDelay: `${i * 100}ms` }}
                  >
                    {token}
                  </span>
                ))}
              </div>
            </div>
          </>
        )}

        {currentStep >= 4 && (
          <>
            <div className="flex justify-center my-4">
              <ArrowRight className="text-gray-500 rotate-90" />
            </div>
            
            {/* Step 4: Token IDs */}
            <div className={`transition-all duration-500 ${currentStep >= 4 ? 'opacity-100' : 'opacity-30'}`}>
              <div className="flex items-center gap-2 mb-2">
                <span className="text-sm font-medium text-gray-400">Token IDs (vocab indices):</span>
              </div>
              <div className="flex flex-wrap gap-2">
                {withSpecial.map((token, i) => (
                  <div key={i} className="flex flex-col items-center mask-reveal" style={{ animationDelay: `${i * 100}ms` }}>
                    <span className="text-xs text-gray-500 mb-1">{token}</span>
                    <span className="bg-cyan-600/30 border border-cyan-500/50 px-3 py-2 rounded-lg font-mono text-cyan-300">
                      {tokenIds[i]}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
      </div>

      {/* Special Tokens Explanation */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
          <Info size={20} className="text-blue-400" />
          BERT Special Tokens
        </h3>
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
          {[
            { token: '[CLS]', id: 101, desc: 'Classification token - always first. Its output embedding is used for sentence-level tasks.', color: 'red' },
            { token: '[SEP]', id: 102, desc: 'Separator token - marks end of sentence or separates sentence pairs.', color: 'orange' },
            { token: '[MASK]', id: 103, desc: 'Mask token - used during pre-training to hide words for prediction.', color: 'purple' },
            { token: '[PAD]', id: 0, desc: 'Padding token - fills sequences to equal length in a batch.', color: 'gray' },
          ].map(item => (
            <div key={item.token} className={`bg-${item.color}-900/20 rounded-xl p-4 border border-${item.color}-500/30`}>
              <div className="flex items-center gap-2 mb-2">
                <span className={`font-mono font-bold text-${item.color}-400`}>{item.token}</span>
                <span className="text-xs text-gray-500">ID: {item.id}</span>
              </div>
              <p className="text-xs text-gray-400">{item.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* WordPiece Deep Dive */}
      <div className="bg-gradient-to-br from-purple-900/30 to-blue-900/30 rounded-2xl p-6 border border-purple-500/30">
        <h3 className="text-lg font-bold mb-4">🔤 WordPiece Tokenization Deep Dive</h3>
        <div className="space-y-4">
          <p className="text-gray-300">
            WordPiece breaks words into subword units, handling unknown words by decomposing them:
          </p>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-black/30 rounded-lg p-4">
              <p className="text-sm text-gray-400 mb-2">Example: "unbelievable"</p>
              <div className="flex flex-wrap gap-2">
                <span className="bg-purple-600/30 px-2 py-1 rounded text-sm">un</span>
                <span className="text-gray-500">+</span>
                <span className="bg-purple-600/30 px-2 py-1 rounded text-sm">##believ</span>
                <span className="text-gray-500">+</span>
                <span className="bg-purple-600/30 px-2 py-1 rounded text-sm">##able</span>
              </div>
              <p className="text-xs text-gray-500 mt-2">## indicates continuation of previous token</p>
            </div>
            <div className="bg-black/30 rounded-lg p-4">
              <p className="text-sm text-gray-400 mb-2">Why WordPiece?</p>
              <ul className="text-sm space-y-1">
                <li className="flex items-start gap-2">
                  <span className="text-green-400">✓</span>
                  <span>Handles unknown words gracefully</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-400">✓</span>
                  <span>Reduces vocabulary size (~30k tokens)</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-400">✓</span>
                  <span>Shares representations between similar words</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      {/* Code Example */}
      <div className="bg-black/40 rounded-xl p-4 border border-white/10">
        <p className="text-sm text-gray-400 mb-3">🐍 Python Code (using Transformers):</p>
        <pre className="text-sm overflow-x-auto">
          <code className="text-green-300">{`from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text = "I love playing football!"
tokens = tokenizer.tokenize(text)
# ['i', 'love', 'play', '##ing', 'foot', '##ball', '!']

input_ids = tokenizer.encode(text)
# [101, 1045, 2293, 2377, 2075, 3329, 3608, 999, 102]
#  [CLS]  i   love  play  ##ing foot ##ball  !  [SEP]

# Or use the full pipeline:
encoding = tokenizer(text, return_tensors='pt')
# {'input_ids': tensor([[101, 1045, ...]]),
#  'token_type_ids': tensor([[0, 0, ...]]),
#  'attention_mask': tensor([[1, 1, ...]])}`}</code>
        </pre>
      </div>
    </div>
  );
}
