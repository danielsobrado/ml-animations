import React, { useState } from 'react';

export default function Step1Tokenization({ onComplete, onNext }) {
    const [inputText, setInputText] = useState("Hello, GPT-2!");
    const [tokens, setTokens] = useState([]);
    const [quizAnswer, setQuizAnswer] = useState('');
    const [quizFeedback, setQuizFeedback] = useState('');

    // Simplified BPE tokenization simulation
    const tokenize = (text) => {
        // Simple word-level tokenization for demonstration
        const simpleTokens = text.split(/(\s+|[,.!?])/g).filter(t => t.trim());
        setTokens(simpleTokens);
    };

    const checkQuiz = () => {
        const correct = quizAnswer.toLowerCase().includes('subword');
        setQuizFeedback(correct
            ? '✓ Correct! BPE breaks text into subword units, allowing the model to handle unknown words.'
            : '✗ Try again. Think about how BPE handles rare or unknown words.'
        );
        if (correct) onComplete();
    };

    return (
        <div className="space-y-8">
            <div>
                <h2 className="text-3xl font-bold mb-2">Step 1: Tokenization & Embeddings</h2>
                <p className="text-gray-400">How text becomes numbers that GPT-2 can understand</p>
            </div>

            {/* Explanation */}
            <div className="bg-gray-800 rounded-lg p-6 space-y-4">
                <h3 className="text-xl font-semibold text-emerald-400">What is Tokenization?</h3>
                <p className="text-gray-300">
                    GPT-2 can't process raw text - it needs numbers. <strong>Tokenization</strong> converts text into a sequence of tokens (subword units).
                </p>
                <p className="text-gray-300">
                    GPT-2 uses <strong>Byte-Pair Encoding (BPE)</strong> with a vocabulary of 50,257 tokens. This allows it to:
                </p>
                <ul className="list-disc list-inside space-y-1 text-gray-300 ml-4">
                    <li>Handle any text (including rare words)</li>
                    <li>Break unknown words into known subwords</li>
                    <li>Keep common words as single tokens</li>
                </ul>
            </div>

            {/* Interactive Demo */}
            <div className="bg-gray-800 rounded-lg p-6 space-y-4">
                <h3 className="text-xl font-semibold text-emerald-400">Try it Yourself</h3>
                <div>
                    <label className="block text-sm text-gray-400 mb-2">Enter text:</label>
                    <input
                        type="text"
                        value={inputText}
                        onChange={(e) => setInputText(e.target.value)}
                        className="w-full bg-gray-700 text-white px-4 py-2 rounded border border-gray-600 focus:border-emerald-500 focus:outline-none"
                        placeholder="Type anything..."
                    />
                </div>
                <button
                    onClick={() => tokenize(inputText)}
                    className="px-6 py-2 bg-emerald-600 hover:bg-emerald-700 rounded font-semibold transition-colors"
                >
                    Tokenize
                </button>

                {tokens.length > 0 && (
                    <div className="mt-4">
                        <div className="text-sm text-gray-400 mb-2">Tokens ({tokens.length}):</div>
                        <div className="flex flex-wrap gap-2">
                            {tokens.map((token, i) => (
                                <div key={i} className="bg-emerald-900 text-emerald-100 px-3 py-1 rounded text-sm font-mono">
                                    {token}
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>

            {/* Embedding Explanation */}
            <div className="bg-gray-800 rounded-lg p-6 space-y-4">
                <h3 className="text-xl font-semibold text-emerald-400">Token Embeddings</h3>
                <p className="text-gray-300">
                    Each token is converted to a <strong>learned embedding vector</strong> of size 768 (for GPT-2 Small).
                </p>
                <div className="bg-gray-900 p-4 rounded font-mono text-sm text-gray-300">
                    Token "Hello" → Token ID: 15496 → Embedding: [0.23, -0.45, 0.12, ..., 0.67] (768 dimensions)
                </div>
                <p className="text-gray-300">
                    These embeddings are <strong>learned during training</strong> so that similar tokens have similar vectors.
                </p>
            </div>

            {/* Exercise */}
            <div className="bg-blue-900 bg-opacity-30 border border-blue-700 rounded-lg p-6 space-y-4">
                <h3 className="text-xl font-semibold text-blue-400">📝 Exercise</h3>
                <p className="text-gray-300">
                    Why does GPT-2 use Byte-Pair Encoding instead of word-level tokenization?
                </p>
                <textarea
                    value={quizAnswer}
                    onChange={(e) => setQuizAnswer(e.target.value)}
                    className="w-full bg-gray-700 text-white px-4 py-2 rounded border border-gray-600 focus:border-blue-500 focus:outline-none h-24"
                    placeholder="Your answer..."
                />
                <button
                    onClick={checkQuiz}
                    className="px-6 py-2 bg-blue-600 hover:bg-blue-700 rounded font-semibold transition-colors"
                >
                    Check Answer
                </button>
                {quizFeedback && (
                    <div className={`p-3 rounded ${quizFeedback.startsWith('✓') ? 'bg-green-900 text-green-200' : 'bg-red-900 text-red-200'}`}>
                        {quizFeedback}
                    </div>
                )}
            </div>

            {/* Navigation */}
            <div className="flex justify-end">
                <button
                    onClick={onNext}
                    className="px-6 py-3 bg-emerald-600 hover:bg-emerald-700 rounded font-semibold transition-colors"
                >
                    Next: Positional Encoding →
                </button>
            </div>
        </div>
    );
}
