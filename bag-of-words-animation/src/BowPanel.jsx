import React, { useState, useEffect } from 'react';
import { Play, Pause, RotateCcw, Hash, Eye, EyeOff } from 'lucide-react';

export default function BowPanel() {
  const [inputText, setInputText] = useState("The cat sat on the mat. The dog sat on the rug.");
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [showStopwords, setShowStopwords] = useState(true);

  const stopwords = ['the', 'on', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'of', 'to', 'and'];

  const steps = [
    { title: 'Original Text', description: 'Start with raw text input' },
    { title: 'Tokenization', description: 'Split text into individual tokens (words)' },
    { title: 'Lowercasing', description: 'Convert all tokens to lowercase for consistency' },
    { title: 'Remove Punctuation', description: 'Strip punctuation marks' },
    { title: 'Build Vocabulary', description: 'Create set of unique words' },
    { title: 'Count Frequencies', description: 'Count occurrences of each word' },
    { title: 'Create Vector', description: 'Generate the final BoW vector' },
  ];

  // Process text through steps
  const processText = () => {
    const sentences = inputText.split(/[.!?]+/).filter(s => s.trim());
    
    // Step 1: Original
    const original = inputText;
    
    // Step 2: Tokenize
    const tokens = inputText.split(/\s+/);
    
    // Step 3: Lowercase
    const lowercased = tokens.map(t => t.toLowerCase());
    
    // Step 4: Remove punctuation
    const cleaned = lowercased.map(t => t.replace(/[^a-z]/g, '')).filter(t => t);
    
    // Step 5: Build vocabulary (optionally filter stopwords)
    let vocab = [...new Set(cleaned)].sort();
    if (!showStopwords) {
      vocab = vocab.filter(w => !stopwords.includes(w));
    }
    
    // Step 6: Count frequencies
    const counts = {};
    cleaned.forEach(word => {
      if (showStopwords || !stopwords.includes(word)) {
        counts[word] = (counts[word] || 0) + 1;
      }
    });
    
    // Step 7: Create vector
    const vector = vocab.map(word => counts[word] || 0);
    
    return { original, tokens, lowercased, cleaned, vocab, counts, vector };
  };

  const processed = processText();

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
    <div className="space-y-6 pb-20">
      {/* Title */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">
          <span className="text-green-400">Bag of Words</span>: Count-Based Representation
        </h2>
        <p className="text-gray-400">
          The simplest way to convert text into numbers
        </p>
      </div>

      {/* Input */}
      <div className="bg-black/30 rounded-xl p-4 border border-white/10">
        <label className="text-sm text-gray-400 mb-2 block">Enter your text:</label>
        <textarea
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          className="w-full bg-gray-800 border border-gray-600 rounded-lg p-3 text-white resize-none"
          rows={2}
          placeholder="Enter text to convert to Bag of Words..."
        />
      </div>

      {/* Controls */}
      <div className="flex flex-wrap justify-center gap-3">
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
        <button
          onClick={() => setShowStopwords(!showStopwords)}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
            showStopwords ? 'bg-white/10 hover:bg-white/20' : 'bg-red-600 hover:bg-red-700'
          }`}
        >
          {showStopwords ? <Eye size={18} /> : <EyeOff size={18} />}
          {showStopwords ? 'With Stopwords' : 'No Stopwords'}
        </button>
      </div>

      {/* Step Progress */}
      <div className="flex flex-wrap justify-center gap-2">
        {steps.map((step, i) => (
          <button
            key={i}
            onClick={() => { setCurrentStep(i); setIsPlaying(false); }}
            className={`px-3 py-1 rounded-full text-sm transition-all ${
              i === currentStep 
                ? 'bg-green-500 text-black scale-110' 
                : i < currentStep 
                ? 'bg-green-900 text-green-300' 
                : 'bg-white/10 text-gray-500'
            }`}
          >
            {i + 1}
          </button>
        ))}
      </div>

      {/* Current Step */}
      <div className="bg-green-900/20 border border-green-500/30 rounded-xl p-4">
        <h3 className="font-bold text-green-400">Step {currentStep + 1}: {steps[currentStep].title}</h3>
        <p className="text-gray-300 mt-1">{steps[currentStep].description}</p>
      </div>

      {/* Visualization */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        {/* Step 0-1: Original/Tokens */}
        {currentStep <= 1 && (
          <div className="space-y-4">
            <h4 className="text-sm text-gray-400">
              {currentStep === 0 ? 'Original Text:' : 'Tokenized:'}
            </h4>
            <div className="flex flex-wrap gap-2">
              {currentStep === 0 ? (
                <p className="text-xl text-white font-mono">{processed.original}</p>
              ) : (
                processed.tokens.map((token, i) => (
                  <span 
                    key={i} 
                    className="px-3 py-1 bg-blue-600/30 rounded text-blue-300 font-mono animate-fadeIn"
                    style={{ animationDelay: `${i * 50}ms` }}
                  >
                    {token}
                  </span>
                ))
              )}
            </div>
          </div>
        )}

        {/* Step 2: Lowercase */}
        {currentStep === 2 && (
          <div className="space-y-4">
            <h4 className="text-sm text-gray-400">Lowercased Tokens:</h4>
            <div className="flex flex-wrap gap-2">
              {processed.lowercased.map((token, i) => (
                <span 
                  key={i} 
                  className="px-3 py-1 bg-purple-600/30 rounded text-purple-300 font-mono animate-fadeIn"
                  style={{ animationDelay: `${i * 50}ms` }}
                >
                  {token}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Step 3: Clean */}
        {currentStep === 3 && (
          <div className="space-y-4">
            <h4 className="text-sm text-gray-400">Cleaned (no punctuation):</h4>
            <div className="flex flex-wrap gap-2">
              {processed.cleaned.map((token, i) => (
                <span 
                  key={i} 
                  className={`px-3 py-1 rounded font-mono animate-fadeIn ${
                    !showStopwords && stopwords.includes(token)
                      ? 'bg-red-600/30 text-red-300 line-through'
                      : 'bg-green-600/30 text-green-300'
                  }`}
                  style={{ animationDelay: `${i * 50}ms` }}
                >
                  {token}
                </span>
              ))}
            </div>
            {!showStopwords && (
              <p className="text-xs text-red-400">Stopwords will be removed</p>
            )}
          </div>
        )}

        {/* Step 4: Vocabulary */}
        {currentStep === 4 && (
          <div className="space-y-4">
            <h4 className="text-sm text-gray-400">Vocabulary ({processed.vocab.length} unique words):</h4>
            <div className="flex flex-wrap gap-2">
              {processed.vocab.map((word, i) => (
                <div 
                  key={i} 
                  className="flex items-center gap-1 px-3 py-1 bg-yellow-600/30 rounded animate-fadeIn"
                  style={{ animationDelay: `${i * 100}ms` }}
                >
                  <span className="text-xs text-gray-400">{i}:</span>
                  <span className="text-yellow-300 font-mono">{word}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Step 5: Counts */}
        {currentStep === 5 && (
          <div className="space-y-4">
            <h4 className="text-sm text-gray-400">Word Frequencies:</h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {Object.entries(processed.counts).sort((a, b) => b[1] - a[1]).map(([word, count], i) => (
                <div 
                  key={i} 
                  className="bg-orange-900/30 rounded-lg p-3 text-center animate-fadeIn"
                  style={{ animationDelay: `${i * 100}ms` }}
                >
                  <p className="font-mono text-orange-300">{word}</p>
                  <p className="text-2xl font-bold text-white">{count}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Step 6: Vector */}
        {currentStep === 6 && (
          <div className="space-y-4">
            <h4 className="text-sm text-gray-400">Final BoW Vector:</h4>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr>
                    {processed.vocab.map((word, i) => (
                      <th key={i} className="px-2 py-1 text-yellow-400 font-mono text-center border-b border-white/10">
                        {word}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    {processed.vector.map((count, i) => (
                      <td 
                        key={i} 
                        className="px-2 py-2 text-center font-mono animate-fadeIn"
                        style={{ animationDelay: `${i * 100}ms` }}
                      >
                        <span className={`inline-block w-8 h-8 rounded-full flex items-center justify-center ${
                          count > 0 ? 'bg-green-600 text-white' : 'bg-gray-700 text-gray-400'
                        }`}>
                          {count}
                        </span>
                      </td>
                    ))}
                  </tr>
                </tbody>
              </table>
            </div>
            <div className="bg-gray-800 rounded-lg p-3 mt-4">
              <p className="text-xs text-gray-400 mb-1">Vector representation:</p>
              <p className="font-mono text-green-400 break-all">
                [{processed.vector.join(', ')}]
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Key Points */}
      <div className="grid md:grid-cols-2 gap-4">
        <div className="bg-green-900/20 rounded-xl p-4 border border-green-500/30">
          <h4 className="font-bold text-green-400 mb-2">✅ Advantages</h4>
          <ul className="text-sm text-gray-300 space-y-1">
            <li>• Simple and fast to compute</li>
            <li>• Easy to understand and interpret</li>
            <li>• Works well for document classification</li>
            <li>• Fixed-size representation</li>
          </ul>
        </div>
        <div className="bg-red-900/20 rounded-xl p-4 border border-red-500/30">
          <h4 className="font-bold text-red-400 mb-2">❌ Limitations</h4>
          <ul className="text-sm text-gray-300 space-y-1">
            <li>• Ignores word order ("dog bites man" = "man bites dog")</li>
            <li>• No semantic meaning captured</li>
            <li>• Common words dominate (need TF-IDF)</li>
            <li>• High dimensionality with large vocab</li>
          </ul>
        </div>
      </div>

      {/* Code Example */}
      <div className="bg-black/40 rounded-xl p-4 border border-white/10">
        <p className="text-sm text-gray-400 mb-3">🐍 Python with scikit-learn:</p>
        <pre className="text-sm overflow-x-auto">
          <code className="text-green-300">{`from sklearn.feature_extraction.text import CountVectorizer

# Create the vectorizer
vectorizer = CountVectorizer()

# Example documents
documents = [
    "The cat sat on the mat",
    "The dog ran in the park"
]

# Fit and transform
bow_matrix = vectorizer.fit_transform(documents)

# Get vocabulary
print("Vocabulary:", vectorizer.get_feature_names_out())
# ['cat', 'dog', 'in', 'mat', 'on', 'park', 'ran', 'sat', 'the']

# Get BoW vectors
print("BoW Matrix:")
print(bow_matrix.toarray())
# [[1 0 0 1 1 0 0 1 2]    <- "The cat sat on the mat"
#  [0 1 1 0 0 1 1 0 2]]   <- "The dog ran in the park"`}</code>
        </pre>
      </div>
    </div>
  );
}
