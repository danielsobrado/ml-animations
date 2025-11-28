import React, { useState, useEffect } from 'react';
import { Play, Pause, RotateCcw, Calculator, BarChart } from 'lucide-react';

export default function TfIdfPanel() {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [selectedDoc, setSelectedDoc] = useState(0);
  const [selectedWord, setSelectedWord] = useState('cat');

  const documents = [
    "The cat sat on the mat",
    "The dog chased the cat",
    "The cat and the dog played"
  ];

  const steps = [
    { title: 'Document Corpus', description: 'Start with a collection of documents' },
    { title: 'Term Frequency (TF)', description: 'Count word occurrences in each document, normalized by doc length' },
    { title: 'Document Frequency (DF)', description: 'Count how many documents contain each word' },
    { title: 'Inverse Document Frequency (IDF)', description: 'Calculate IDF = log(N / DF) to penalize common words' },
    { title: 'TF-IDF Score', description: 'Multiply TF × IDF to get final weights' },
  ];

  // Process documents
  const processDocuments = () => {
    const N = documents.length;
    
    // Tokenize and clean
    const tokenized = documents.map(doc => 
      doc.toLowerCase().replace(/[^a-z\s]/g, '').split(/\s+/).filter(w => w)
    );

    // Build vocabulary
    const allWords = tokenized.flat();
    const vocab = [...new Set(allWords)].sort();

    // Calculate TF for each document
    const tf = tokenized.map(tokens => {
      const counts = {};
      tokens.forEach(t => counts[t] = (counts[t] || 0) + 1);
      const result = {};
      vocab.forEach(word => {
        result[word] = (counts[word] || 0) / tokens.length;
      });
      return result;
    });

    // Calculate DF
    const df = {};
    vocab.forEach(word => {
      df[word] = tokenized.filter(tokens => tokens.includes(word)).length;
    });

    // Calculate IDF
    const idf = {};
    vocab.forEach(word => {
      idf[word] = Math.log(N / df[word]);
    });

    // Calculate TF-IDF
    const tfidf = tf.map(docTf => {
      const result = {};
      vocab.forEach(word => {
        result[word] = docTf[word] * idf[word];
      });
      return result;
    });

    return { tokenized, vocab, tf, df, idf, tfidf, N };
  };

  const data = processDocuments();

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
      }, 3000);
      return () => clearInterval(interval);
    }
  }, [isPlaying]);

  const reset = () => {
    setCurrentStep(0);
    setIsPlaying(false);
  };

  const formatNum = (n) => n.toFixed(3);

  return (
    <div className="space-y-6 pb-20">
      {/* Title */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">
          <span className="text-yellow-400">TF-IDF</span>: Term Frequency × Inverse Document Frequency
        </h2>
        <p className="text-gray-400">
          Weight words by importance: frequent in document, rare across corpus
        </p>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap justify-center gap-3">
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          className="flex items-center gap-2 px-4 py-2 bg-yellow-600 hover:bg-yellow-700 rounded-lg transition-colors"
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
      <div className="flex flex-wrap justify-center gap-2">
        {steps.map((step, i) => (
          <button
            key={i}
            onClick={() => { setCurrentStep(i); setIsPlaying(false); }}
            className={`px-3 py-1 rounded-full text-sm transition-all ${
              i === currentStep 
                ? 'bg-yellow-500 text-black scale-110' 
                : i < currentStep 
                ? 'bg-yellow-900 text-yellow-300' 
                : 'bg-white/10 text-gray-500'
            }`}
          >
            {i + 1}. {step.title}
          </button>
        ))}
      </div>

      {/* Current Step */}
      <div className="bg-yellow-900/20 border border-yellow-500/30 rounded-xl p-4">
        <h3 className="font-bold text-yellow-400">Step {currentStep + 1}: {steps[currentStep].title}</h3>
        <p className="text-gray-300 mt-1">{steps[currentStep].description}</p>
      </div>

      {/* Main Visualization */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        {/* Step 0: Documents */}
        {currentStep === 0 && (
          <div className="space-y-4">
            <h4 className="text-lg font-bold text-gray-300">Document Corpus (N = {data.N})</h4>
            {documents.map((doc, i) => (
              <div 
                key={i}
                className="bg-blue-900/30 rounded-lg p-4 border border-blue-500/30 animate-fadeIn"
                style={{ animationDelay: `${i * 200}ms` }}
              >
                <span className="text-blue-400 font-mono text-sm">Doc {i + 1}:</span>
                <p className="text-white mt-1">"{doc}"</p>
              </div>
            ))}
          </div>
        )}

        {/* Step 1: TF */}
        {currentStep === 1 && (
          <div className="space-y-4">
            <div className="flex items-center gap-4 mb-4">
              <h4 className="text-lg font-bold text-gray-300">Term Frequency (TF)</h4>
              <select
                value={selectedDoc}
                onChange={(e) => setSelectedDoc(parseInt(e.target.value))}
                className="bg-gray-700 border border-gray-600 rounded px-3 py-1"
              >
                {documents.map((_, i) => (
                  <option key={i} value={i}>Document {i + 1}</option>
                ))}
              </select>
            </div>
            
            <div className="bg-blue-900/20 rounded-lg p-3 mb-4">
              <p className="text-sm text-blue-400">Formula: TF(t, d) = count(t in d) / |d|</p>
              <p className="text-xs text-gray-400 mt-1">Word count divided by total words in document</p>
            </div>

            <p className="text-sm text-gray-400 mb-2">"{documents[selectedDoc]}"</p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {data.vocab.map((word, i) => {
                const tfValue = data.tf[selectedDoc][word];
                return (
                  <div 
                    key={i}
                    className={`rounded-lg p-3 text-center animate-fadeIn ${
                      tfValue > 0 ? 'bg-green-900/30 border border-green-500/30' : 'bg-gray-800/30'
                    }`}
                    style={{ animationDelay: `${i * 50}ms` }}
                  >
                    <p className="font-mono text-sm">{word}</p>
                    <p className={`text-xl font-bold ${tfValue > 0 ? 'text-green-400' : 'text-gray-500'}`}>
                      {formatNum(tfValue)}
                    </p>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Step 2: DF */}
        {currentStep === 2 && (
          <div className="space-y-4">
            <h4 className="text-lg font-bold text-gray-300">Document Frequency (DF)</h4>
            <div className="bg-purple-900/20 rounded-lg p-3 mb-4">
              <p className="text-sm text-purple-400">DF(t) = number of documents containing term t</p>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {data.vocab.map((word, i) => (
                <div 
                  key={i}
                  className="bg-purple-900/30 rounded-lg p-3 text-center border border-purple-500/30 animate-fadeIn"
                  style={{ animationDelay: `${i * 50}ms` }}
                >
                  <p className="font-mono text-sm">{word}</p>
                  <p className="text-xl font-bold text-purple-400">
                    {data.df[word]} / {data.N}
                  </p>
                  <p className="text-xs text-gray-500">
                    in {data.df[word]} doc{data.df[word] > 1 ? 's' : ''}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Step 3: IDF */}
        {currentStep === 3 && (
          <div className="space-y-4">
            <h4 className="text-lg font-bold text-gray-300">Inverse Document Frequency (IDF)</h4>
            <div className="bg-orange-900/20 rounded-lg p-3 mb-4">
              <p className="text-sm text-orange-400">IDF(t) = log(N / DF(t))</p>
              <p className="text-xs text-gray-400 mt-1">
                Rare words → high IDF, Common words → low IDF
              </p>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {data.vocab.map((word, i) => {
                const idfValue = data.idf[word];
                return (
                  <div 
                    key={i}
                    className="bg-orange-900/30 rounded-lg p-3 text-center border border-orange-500/30 animate-fadeIn"
                    style={{ animationDelay: `${i * 50}ms` }}
                  >
                    <p className="font-mono text-sm">{word}</p>
                    <p className={`text-xl font-bold ${
                      idfValue > 0.5 ? 'text-orange-400' : 'text-gray-400'
                    }`}>
                      {formatNum(idfValue)}
                    </p>
                    <p className="text-xs text-gray-500">
                      log({data.N}/{data.df[word]})
                    </p>
                  </div>
                );
              })}
            </div>

            <div className="bg-gray-800 rounded-lg p-4 mt-4">
              <p className="text-sm text-gray-400 mb-2">Notice:</p>
              <ul className="text-sm space-y-1">
                <li className="text-red-400">• "the" appears in all docs → IDF = 0 (least important)</li>
                <li className="text-green-400">• Unique words → IDF {">"} 0 (more important)</li>
              </ul>
            </div>
          </div>
        )}

        {/* Step 4: TF-IDF */}
        {currentStep === 4 && (
          <div className="space-y-4">
            <div className="flex items-center gap-4 mb-4">
              <h4 className="text-lg font-bold text-gray-300">TF-IDF Scores</h4>
              <select
                value={selectedDoc}
                onChange={(e) => setSelectedDoc(parseInt(e.target.value))}
                className="bg-gray-700 border border-gray-600 rounded px-3 py-1"
              >
                {documents.map((_, i) => (
                  <option key={i} value={i}>Document {i + 1}</option>
                ))}
              </select>
            </div>

            <div className="bg-yellow-900/20 rounded-lg p-3 mb-4">
              <p className="text-sm text-yellow-400">TF-IDF(t, d) = TF(t, d) × IDF(t)</p>
              <p className="text-xs text-gray-400 mt-1">
                High score = word is important to this specific document
              </p>
            </div>

            <p className="text-sm text-gray-400 mb-2">"{documents[selectedDoc]}"</p>
            
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {data.vocab
                .map(word => ({ word, score: data.tfidf[selectedDoc][word] }))
                .sort((a, b) => b.score - a.score)
                .map(({ word, score }, i) => (
                  <div 
                    key={i}
                    className={`rounded-lg p-3 text-center animate-fadeIn ${
                      score > 0 ? 'bg-yellow-900/30 border border-yellow-500/30' : 'bg-gray-800/30 border border-gray-600/30'
                    }`}
                    style={{ animationDelay: `${i * 50}ms` }}
                  >
                    <p className="font-mono text-sm">{word}</p>
                    <p className={`text-xl font-bold ${score > 0 ? 'text-yellow-400' : 'text-gray-500'}`}>
                      {formatNum(score)}
                    </p>
                    {score > 0 && (
                      <p className="text-xs text-gray-500">
                        {formatNum(data.tf[selectedDoc][word])} × {formatNum(data.idf[word])}
                      </p>
                    )}
                  </div>
                ))}
            </div>

            {/* TF-IDF Matrix */}
            <div className="mt-6 overflow-x-auto">
              <h5 className="text-sm text-gray-400 mb-2">Complete TF-IDF Matrix:</h5>
              <table className="w-full text-sm">
                <thead>
                  <tr>
                    <th className="px-2 py-1 text-left text-gray-400">Doc</th>
                    {data.vocab.map((word, i) => (
                      <th key={i} className="px-2 py-1 text-center text-yellow-400 font-mono">{word}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {documents.map((_, docIdx) => (
                    <tr key={docIdx} className={docIdx === selectedDoc ? 'bg-yellow-900/20' : ''}>
                      <td className="px-2 py-1 text-gray-400">Doc {docIdx + 1}</td>
                      {data.vocab.map((word, i) => (
                        <td key={i} className="px-2 py-1 text-center font-mono">
                          <span className={data.tfidf[docIdx][word] > 0 ? 'text-green-400' : 'text-gray-600'}>
                            {formatNum(data.tfidf[docIdx][word])}
                          </span>
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>

      {/* Formula Summary */}
      <div className="bg-gradient-to-r from-yellow-900/20 to-orange-900/20 rounded-xl p-6 border border-yellow-500/30">
        <h4 className="font-bold text-yellow-400 mb-4">📐 TF-IDF Formulas</h4>
        <div className="grid md:grid-cols-3 gap-4">
          <div className="bg-black/30 rounded-lg p-4">
            <h5 className="text-green-400 font-medium mb-2">Term Frequency</h5>
            <div className="bg-black/50 rounded p-2 font-mono text-sm text-center">
              TF(t,d) = count(t,d) / |d|
            </div>
            <p className="text-xs text-gray-400 mt-2">Normalized word count in document</p>
          </div>
          <div className="bg-black/30 rounded-lg p-4">
            <h5 className="text-purple-400 font-medium mb-2">Inverse Doc Frequency</h5>
            <div className="bg-black/50 rounded p-2 font-mono text-sm text-center">
              IDF(t) = log(N / DF(t))
            </div>
            <p className="text-xs text-gray-400 mt-2">Penalizes common words</p>
          </div>
          <div className="bg-black/30 rounded-lg p-4">
            <h5 className="text-yellow-400 font-medium mb-2">TF-IDF</h5>
            <div className="bg-black/50 rounded p-2 font-mono text-sm text-center">
              TF-IDF = TF × IDF
            </div>
            <p className="text-xs text-gray-400 mt-2">Final importance weight</p>
          </div>
        </div>
      </div>

      {/* Code Example */}
      <div className="bg-black/40 rounded-xl p-4 border border-white/10">
        <p className="text-sm text-gray-400 mb-3">🐍 Python with scikit-learn:</p>
        <pre className="text-sm overflow-x-auto">
          <code className="text-yellow-300">{`from sklearn.feature_extraction.text import TfidfVectorizer

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Example documents
documents = [
    "The cat sat on the mat",
    "The dog chased the cat",
    "The cat and the dog played"
]

# Fit and transform
tfidf_matrix = vectorizer.fit_transform(documents)

# Get feature names
print("Vocabulary:", vectorizer.get_feature_names_out())

# Get TF-IDF scores for first document
print("\\nDoc 1 TF-IDF scores:")
for word, score in zip(vectorizer.get_feature_names_out(), tfidf_matrix[0].toarray()[0]):
    if score > 0:
        print(f"  {word}: {score:.4f}")

# Calculate cosine similarity between documents
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(tfidf_matrix)
print("\\nDocument Similarity Matrix:")
print(similarity)`}</code>
        </pre>
      </div>
    </div>
  );
}
