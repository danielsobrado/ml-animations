import React, { useState, useEffect } from 'react';

export default function TextPanel() {
    const [inputText, setInputText] = useState("I like cats. I like dogs. Cats are cool. Dogs are fun. I like code.");
    const [generatedText, setGeneratedText] = useState("");
    const [chain, setChain] = useState({});
    const [isGenerating, setIsGenerating] = useState(false);

    // Build Markov Chain (Bigrams)
    useEffect(() => {
        const words = inputText.toLowerCase().replace(/[.]/g, ' .').split(/\s+/).filter(w => w);
        const newChain = {};

        for (let i = 0; i < words.length - 1; i++) {
            const current = words[i];
            const next = words[i + 1];

            if (!newChain[current]) newChain[current] = [];
            newChain[current].push(next);
        }

        setChain(newChain);
    }, [inputText]);

    const generate = () => {
        setIsGenerating(true);
        setGeneratedText("");

        // Pick random start word
        const keys = Object.keys(chain);
        if (keys.length === 0) return;

        let currentWord = keys[Math.floor(Math.random() * keys.length)];
        let result = [currentWord];

        let count = 0;
        const maxWords = 20;

        const interval = setInterval(() => {
            const nextOptions = chain[currentWord];

            if (!nextOptions || nextOptions.length === 0 || count >= maxWords || currentWord === '.') {
                clearInterval(interval);
                setIsGenerating(false);
                return;
            }

            const nextWord = nextOptions[Math.floor(Math.random() * nextOptions.length)];
            result.push(nextWord);
            setGeneratedText(result.join(' ').replace(/ \./g, '.')); // Fix punctuation spacing

            currentWord = nextWord;
            count++;
        }, 200);
    };

    return (
        <div className="p-8 h-full flex flex-col items-center overflow-y-auto">
            <div className="max-w-3xl w-full text-center mb-8">
                <h2 className="text-3xl font-bold text-pink-400 mb-4">Text Generation (Mini-LLM)</h2>
                <p className="text-lg text-slate-300 leading-relaxed">
                    Large Language Models are essentially giant Markov Chains.
                    <br />
                    They predict the <strong>next word</strong> based on the current context.
                </p>
            </div>

            <div className="grid md:grid-cols-2 gap-8 w-full max-w-6xl">
                {/* Input */}
                <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 flex flex-col">
                    <h3 className="font-bold text-white mb-4">1. Training Data</h3>
                    <textarea
                        value={inputText}
                        onChange={(e) => setInputText(e.target.value)}
                        className="w-full h-40 bg-slate-900 border border-slate-600 rounded-lg p-4 text-slate-300 focus:border-pink-500 outline-none resize-none"
                        placeholder="Enter some text here to train the model..."
                    />
                    <p className="text-xs text-slate-500 mt-2">
                        The model learns which words follow which. Try adding more sentences!
                    </p>
                </div>

                {/* Visualization of Chain */}
                <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 flex flex-col h-[300px] overflow-hidden">
                    <h3 className="font-bold text-white mb-4">2. The Learned Chain (Dictionary)</h3>
                    <div className="flex-1 overflow-y-auto font-mono text-sm space-y-2 pr-2">
                        {Object.entries(chain).map(([word, nextWords]) => (
                            <div key={word} className="bg-slate-900 p-2 rounded border border-slate-700">
                                <span className="text-pink-400 font-bold">{word}</span>
                                <span className="text-slate-500"> ➔ </span>
                                <span className="text-slate-300">[{nextWords.join(', ')}]</span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Generation */}
            <div className="w-full max-w-6xl mt-8">
                <button
                    onClick={generate}
                    disabled={isGenerating}
                    className={`w-full py-4 rounded-xl font-bold text-xl transition-all shadow-lg ${isGenerating
                            ? 'bg-slate-700 text-slate-500 cursor-not-allowed'
                            : 'bg-gradient-to-r from-pink-600 to-purple-600 text-white hover:scale-[1.02]'
                        }`}
                >
                    {isGenerating ? 'Generating...' : '✨ Generate New Text'}
                </button>

                {generatedText && (
                    <div className="mt-8 p-8 bg-slate-900 rounded-2xl border-2 border-pink-500/50 text-center">
                        <h3 className="text-slate-500 text-sm uppercase tracking-wider mb-4">Output</h3>
                        <p className="text-2xl text-white font-serif leading-relaxed">
                            "{generatedText}"
                        </p>
                    </div>
                )}
            </div>
        </div>
    );
}
