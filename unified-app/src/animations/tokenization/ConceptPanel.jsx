import React, { useState, useEffect } from 'react';
import { Type, Hash, Layers, ArrowRight, Lightbulb } from 'lucide-react';

const EXAMPLES = [
    { text: "Hello world!", description: "Simple greeting" },
    { text: "I'm learning tokenization", description: "Contractions & long words" },
    { text: "GPT-4 is amazing!", description: "Hyphenated terms" },
    { text: "The quick brown fox jumps", description: "Common English" },
    { text: "日本語テスト", description: "Japanese (Unicode)" },
];

// Simulated character-level tokenization
const charTokenize = (text) => text.split('');

// Simulated word-level tokenization
const wordTokenize = (text) => text.split(/(\s+|[.,!?;:'"()-])/).filter(t => t.trim());

// Simulated subword tokenization (BPE-like)
const subwordTokenize = (text) => {
    // Simplified subword simulation
    const result = [];
    const words = text.split(/(\s+)/);
    
    words.forEach(word => {
        if (word.match(/^\s+$/)) {
            result.push(word);
        } else if (word.length <= 3) {
            result.push(word);
        } else {
            // Split longer words into subwords
            const common = ['ing', 'tion', 'ed', 'er', 'est', 'ly', 'ment', 'ness', 'ize', 'ful'];
            let remaining = word;
            let found = false;
            
            for (const suffix of common) {
                if (remaining.toLowerCase().endsWith(suffix) && remaining.length > suffix.length + 2) {
                    result.push(remaining.slice(0, -suffix.length));
                    result.push('##' + suffix);
                    found = true;
                    break;
                }
            }
            
            if (!found) {
                if (word.length > 6) {
                    result.push(word.slice(0, Math.ceil(word.length / 2)));
                    result.push('##' + word.slice(Math.ceil(word.length / 2)));
                } else {
                    result.push(word);
                }
            }
        }
    });
    
    return result.filter(t => t);
};

const COLORS = [
    'bg-blue-100 border-blue-300 text-blue-800',
    'bg-green-100 border-green-300 text-green-800',
    'bg-purple-100 border-purple-300 text-purple-800',
    'bg-orange-100 border-orange-300 text-orange-800',
    'bg-pink-100 border-pink-300 text-pink-800',
    'bg-yellow-100 border-yellow-300 text-yellow-800',
    'bg-cyan-100 border-cyan-300 text-cyan-800',
    'bg-red-100 border-red-300 text-red-800',
];

export default function ConceptPanel() {
    const [inputText, setInputText] = useState("Hello, I'm learning tokenization!");
    const [tokenMethod, setTokenMethod] = useState('subword');
    const [tokens, setTokens] = useState([]);
    const [animatingIdx, setAnimatingIdx] = useState(-1);

    useEffect(() => {
        let result;
        switch (tokenMethod) {
            case 'char':
                result = charTokenize(inputText);
                break;
            case 'word':
                result = wordTokenize(inputText);
                break;
            case 'subword':
            default:
                result = subwordTokenize(inputText);
                break;
        }
        setTokens(result);
        
        // Animate tokens appearing
        setAnimatingIdx(-1);
        result.forEach((_, i) => {
            setTimeout(() => setAnimatingIdx(i), i * 100);
        });
    }, [inputText, tokenMethod]);

    return (
        <div className="p-8 h-full">
            <div className="max-w-4xl mx-auto">
                {/* Introduction */}
                <div className="text-center mb-8">
                    <h2 className="text-3xl font-bold text-indigo-900 mb-4">What is Tokenization?</h2>
                    <p className="text-lg text-slate-700 leading-relaxed max-w-2xl mx-auto">
                        Tokenization is the process of breaking text into smaller pieces called <strong>tokens</strong>. 
                        These tokens are the fundamental units that language models understand and process.
                    </p>
                </div>

                {/* Why Tokenization Matters */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
                    <div className="bg-blue-50 p-5 rounded-xl border-2 border-blue-100">
                        <div className="bg-blue-100 w-10 h-10 rounded-full flex items-center justify-center mb-3 text-blue-600">
                            <Type size={20} />
                        </div>
                        <h3 className="font-bold text-blue-900 mb-2">Text → Numbers</h3>
                        <p className="text-blue-800 text-sm">
                            Computers can't read text directly. Tokens are mapped to numerical IDs that models can process.
                        </p>
                    </div>

                    <div className="bg-green-50 p-5 rounded-xl border-2 border-green-100">
                        <div className="bg-green-100 w-10 h-10 rounded-full flex items-center justify-center mb-3 text-green-600">
                            <Hash size={20} />
                        </div>
                        <h3 className="font-bold text-green-900 mb-2">Vocabulary Size</h3>
                        <p className="text-green-800 text-sm">
                            A fixed vocabulary of tokens (e.g., 50,000) can represent any text, balancing coverage and efficiency.
                        </p>
                    </div>

                    <div className="bg-purple-50 p-5 rounded-xl border-2 border-purple-100">
                        <div className="bg-purple-100 w-10 h-10 rounded-full flex items-center justify-center mb-3 text-purple-600">
                            <Layers size={20} />
                        </div>
                        <h3 className="font-bold text-purple-900 mb-2">Subword Magic</h3>
                        <p className="text-purple-800 text-sm">
                            Subword tokenization handles rare words by breaking them into known pieces: "unhappy" → "un" + "happy"
                        </p>
                    </div>
                </div>

                {/* Interactive Demo */}
                <div className="bg-slate-50 rounded-xl p-6 mb-8">
                    <h3 className="text-xl font-bold text-slate-800 mb-4">Try It Yourself</h3>
                    
                    {/* Input */}
                    <div className="mb-4">
                        <label className="block text-sm font-medium text-slate-700 mb-2">Enter text:</label>
                        <input
                            type="text"
                            value={inputText}
                            onChange={(e) => setInputText(e.target.value)}
                            className="w-full p-3 border-2 border-slate-200 rounded-lg focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 transition-all"
                            placeholder="Type something..."
                        />
                    </div>

                    {/* Quick Examples */}
                    <div className="flex flex-wrap gap-2 mb-4">
                        {EXAMPLES.map((ex, i) => (
                            <button
                                key={i}
                                onClick={() => setInputText(ex.text)}
                                className="px-3 py-1 text-xs bg-white border border-slate-200 rounded-full hover:bg-slate-100 transition-colors"
                                title={ex.description}
                            >
                                {ex.text.slice(0, 15)}{ex.text.length > 15 ? '...' : ''}
                            </button>
                        ))}
                    </div>

                    {/* Method Selection */}
                    <div className="flex gap-4 mb-6">
                        {[
                            { id: 'char', label: 'Character-level', desc: 'Every character is a token' },
                            { id: 'word', label: 'Word-level', desc: 'Split on spaces/punctuation' },
                            { id: 'subword', label: 'Subword (BPE)', desc: 'Balanced approach used by GPT' },
                        ].map(method => (
                            <label
                                key={method.id}
                                className={`flex-1 p-3 rounded-lg border-2 cursor-pointer transition-all ${
                                    tokenMethod === method.id
                                        ? 'border-indigo-500 bg-indigo-50'
                                        : 'border-slate-200 hover:border-slate-300'
                                }`}
                            >
                                <input
                                    type="radio"
                                    name="method"
                                    value={method.id}
                                    checked={tokenMethod === method.id}
                                    onChange={(e) => setTokenMethod(e.target.value)}
                                    className="sr-only"
                                />
                                <div className="font-bold text-slate-800">{method.label}</div>
                                <div className="text-xs text-slate-600">{method.desc}</div>
                            </label>
                        ))}
                    </div>

                    {/* Visualization */}
                    <div className="bg-white rounded-lg p-4 border border-slate-200">
                        <div className="flex items-center gap-2 mb-3">
                            <span className="text-sm font-medium text-slate-600">Original:</span>
                            <span className="font-mono bg-slate-100 px-2 py-1 rounded">{inputText}</span>
                        </div>
                        
                        <div className="flex items-center gap-2 mb-3">
                            <ArrowRight className="text-slate-400" size={20} />
                            <span className="text-sm font-medium text-slate-600">Tokens ({tokens.length}):</span>
                        </div>

                        <div className="flex flex-wrap gap-2">
                            {tokens.map((token, i) => (
                                <span
                                    key={i}
                                    className={`inline-flex items-center px-3 py-1 rounded-lg border-2 font-mono text-sm transition-all duration-300 ${
                                        COLORS[i % COLORS.length]
                                    } ${i <= animatingIdx ? 'opacity-100 scale-100' : 'opacity-0 scale-75'}`}
                                >
                                    <span className="opacity-50 mr-1 text-xs">{i}</span>
                                    {token === ' ' ? '␣' : token}
                                </span>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Key Insight */}
                <div className="bg-amber-50 p-4 rounded-xl border border-amber-200 flex items-start gap-3">
                    <Lightbulb className="text-amber-600 flex-shrink-0 mt-1" size={24} />
                    <div>
                        <h4 className="font-bold text-amber-900 mb-1">Key Insight</h4>
                        <p className="text-amber-800 text-sm">
                            Different tokenization methods produce different numbers of tokens for the same text.
                            <strong> Subword tokenization</strong> (like BPE) is the most common in modern LLMs because it 
                            balances vocabulary size with the ability to handle rare or novel words.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}
