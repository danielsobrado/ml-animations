import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, ChevronRight, ChevronLeft, Lightbulb } from 'lucide-react';
import gsap from 'gsap';

// BPE Algorithm Steps Simulation
const runBPE = (text, maxMerges = 10) => {
    const steps = [];
    
    // Step 1: Character-level tokenization with end-of-word marker
    let tokens = text.split(/\s+/).map(word => 
        word.split('').map((c, i, arr) => i === arr.length - 1 ? c + '</w>' : c)
    ).flat();
    
    // For visualization, we track the original words
    const words = text.split(/\s+/);
    let wordTokens = words.map(word => 
        word.split('').map((c, i, arr) => i === arr.length - 1 ? c + '</w>' : c)
    );
    
    steps.push({
        title: 'Initial Characters',
        description: 'Split each word into characters. Add </w> to mark word endings.',
        wordTokens: JSON.parse(JSON.stringify(wordTokens)),
        vocabulary: [...new Set(wordTokens.flat())],
        merge: null
    });
    
    // Run BPE merges
    for (let i = 0; i < maxMerges; i++) {
        // Count pairs
        const pairCounts = {};
        wordTokens.forEach(wt => {
            for (let j = 0; j < wt.length - 1; j++) {
                const pair = wt[j] + ' ' + wt[j + 1];
                pairCounts[pair] = (pairCounts[pair] || 0) + 1;
            }
        });
        
        if (Object.keys(pairCounts).length === 0) break;
        
        // Find most frequent pair
        const bestPair = Object.entries(pairCounts).reduce((a, b) => 
            b[1] > a[1] ? b : a
        );
        
        if (bestPair[1] < 2) break; // Stop if no pair appears more than once
        
        const [pairStr, count] = bestPair;
        const [first, second] = pairStr.split(' ');
        const merged = first + second;
        
        // Apply merge
        wordTokens = wordTokens.map(wt => {
            const newWt = [];
            let j = 0;
            while (j < wt.length) {
                if (j < wt.length - 1 && wt[j] === first && wt[j + 1] === second) {
                    newWt.push(merged);
                    j += 2;
                } else {
                    newWt.push(wt[j]);
                    j++;
                }
            }
            return newWt;
        });
        
        steps.push({
            title: `Merge #${i + 1}: "${first}" + "${second}"`,
            description: `Found pair "${first} ${second}" appears ${count} times. Merge into "${merged}"`,
            wordTokens: JSON.parse(JSON.stringify(wordTokens)),
            vocabulary: [...new Set(wordTokens.flat())],
            merge: { first, second, merged, count }
        });
    }
    
    return steps;
};

const COLORS = [
    'bg-blue-200 border-blue-400',
    'bg-green-200 border-green-400',
    'bg-purple-200 border-purple-400',
    'bg-orange-200 border-orange-400',
    'bg-pink-200 border-pink-400',
    'bg-yellow-200 border-yellow-400',
    'bg-cyan-200 border-cyan-400',
    'bg-red-200 border-red-400',
];

export default function BPEPanel() {
    const [inputText, setInputText] = useState('low lower lowest');
    const [steps, setSteps] = useState([]);
    const [currentStep, setCurrentStep] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const tokensRef = useRef(null);

    useEffect(() => {
        const result = runBPE(inputText, 8);
        setSteps(result);
        setCurrentStep(0);
    }, [inputText]);

    useEffect(() => {
        if (!isPlaying) return;
        
        const timer = setInterval(() => {
            setCurrentStep(prev => {
                if (prev >= steps.length - 1) {
                    setIsPlaying(false);
                    return prev;
                }
                return prev + 1;
            });
        }, 2000);
        
        return () => clearInterval(timer);
    }, [isPlaying, steps.length]);

    // Animate tokens when step changes
    useEffect(() => {
        if (tokensRef.current) {
            gsap.fromTo(
                tokensRef.current.querySelectorAll('.token-item'),
                { scale: 0.8, opacity: 0 },
                { scale: 1, opacity: 1, duration: 0.3, stagger: 0.05 }
            );
        }
    }, [currentStep]);

    const currentStepData = steps[currentStep] || { wordTokens: [], vocabulary: [], title: '', description: '' };

    return (
        <div className="p-8 h-full">
            <div className="max-w-5xl mx-auto">
                {/* Header */}
                <div className="text-center mb-6">
                    <h2 className="text-3xl font-bold text-indigo-900 mb-2">Byte Pair Encoding (BPE)</h2>
                    <p className="text-slate-600">
                        Watch how BPE iteratively merges the most frequent pairs of tokens
                    </p>
                </div>

                {/* Input */}
                <div className="mb-6">
                    <label className="block text-sm font-medium text-slate-700 mb-2">Training Text:</label>
                    <div className="flex gap-2">
                        <input
                            type="text"
                            value={inputText}
                            onChange={(e) => setInputText(e.target.value)}
                            className="flex-1 p-3 border-2 border-slate-200 rounded-lg focus:border-indigo-500"
                            placeholder="Enter training text..."
                        />
                        <div className="flex gap-1">
                            {['low lower lowest', 'happy happier happiest', 'play playing played'].map(ex => (
                                <button
                                    key={ex}
                                    onClick={() => setInputText(ex)}
                                    className="px-3 py-1 text-xs bg-slate-100 rounded-lg hover:bg-slate-200 whitespace-nowrap"
                                >
                                    {ex.slice(0, 12)}...
                                </button>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Controls */}
                <div className="flex items-center justify-center gap-4 mb-6">
                    <button
                        onClick={() => setCurrentStep(0)}
                        className="p-2 rounded-lg bg-slate-100 hover:bg-slate-200 transition-colors"
                        title="Reset"
                    >
                        <RotateCcw size={20} />
                    </button>
                    <button
                        onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
                        disabled={currentStep === 0}
                        className="p-2 rounded-lg bg-slate-100 hover:bg-slate-200 transition-colors disabled:opacity-50"
                    >
                        <ChevronLeft size={20} />
                    </button>
                    <button
                        onClick={() => setIsPlaying(!isPlaying)}
                        className={`px-6 py-2 rounded-lg font-bold transition-colors ${
                            isPlaying ? 'bg-red-500 text-white' : 'bg-indigo-600 text-white'
                        }`}
                    >
                        {isPlaying ? <Pause size={20} /> : <Play size={20} />}
                    </button>
                    <button
                        onClick={() => setCurrentStep(Math.min(steps.length - 1, currentStep + 1))}
                        disabled={currentStep === steps.length - 1}
                        className="p-2 rounded-lg bg-slate-100 hover:bg-slate-200 transition-colors disabled:opacity-50"
                    >
                        <ChevronRight size={20} />
                    </button>
                    <span className="text-sm text-slate-600">
                        Step {currentStep + 1} / {steps.length}
                    </span>
                </div>

                {/* Current Step Info */}
                <div className="bg-indigo-50 rounded-xl p-4 mb-6 border border-indigo-200">
                    <h3 className="font-bold text-indigo-900 text-lg">{currentStepData.title}</h3>
                    <p className="text-indigo-800">{currentStepData.description}</p>
                    {currentStepData.merge && (
                        <div className="mt-2 flex items-center gap-2">
                            <span className="px-2 py-1 bg-indigo-200 rounded font-mono">{currentStepData.merge.first}</span>
                            <span className="text-indigo-600">+</span>
                            <span className="px-2 py-1 bg-indigo-200 rounded font-mono">{currentStepData.merge.second}</span>
                            <span className="text-indigo-600">→</span>
                            <span className="px-2 py-1 bg-green-200 rounded font-mono font-bold">{currentStepData.merge.merged}</span>
                            <span className="text-sm text-indigo-600">({currentStepData.merge.count} occurrences)</span>
                        </div>
                    )}
                </div>

                {/* Tokenization Visualization */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Words and their tokens */}
                    <div className="bg-slate-50 rounded-xl p-4">
                        <h4 className="font-bold text-slate-800 mb-3">Current Tokenization</h4>
                        <div ref={tokensRef} className="space-y-3">
                            {inputText.split(/\s+/).map((word, wordIdx) => (
                                <div key={wordIdx} className="flex items-center gap-2">
                                    <span className="font-mono text-slate-600 w-20">{word}:</span>
                                    <div className="flex flex-wrap gap-1">
                                        {(currentStepData.wordTokens[wordIdx] || []).map((token, tokenIdx) => (
                                            <span
                                                key={tokenIdx}
                                                className={`token-item px-2 py-1 rounded border-2 font-mono text-sm ${
                                                    COLORS[(wordIdx + tokenIdx) % COLORS.length]
                                                }`}
                                            >
                                                {token}
                                            </span>
                                        ))}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Vocabulary */}
                    <div className="bg-slate-50 rounded-xl p-4">
                        <h4 className="font-bold text-slate-800 mb-3">
                            Vocabulary ({currentStepData.vocabulary?.length || 0} tokens)
                        </h4>
                        <div className="flex flex-wrap gap-2 max-h-48 overflow-y-auto">
                            {currentStepData.vocabulary?.map((token, i) => (
                                <span
                                    key={i}
                                    className={`px-2 py-1 rounded border font-mono text-xs ${
                                        currentStepData.merge?.merged === token
                                            ? 'bg-green-200 border-green-400 ring-2 ring-green-500'
                                            : 'bg-white border-slate-300'
                                    }`}
                                >
                                    {token}
                                </span>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Algorithm Explanation */}
                <div className="mt-6 bg-amber-50 p-4 rounded-xl border border-amber-200">
                    <div className="flex items-start gap-3">
                        <Lightbulb className="text-amber-600 flex-shrink-0 mt-1" size={24} />
                        <div>
                            <h4 className="font-bold text-amber-900 mb-2">BPE Algorithm</h4>
                            <ol className="text-amber-800 text-sm space-y-1 list-decimal list-inside">
                                <li>Start with character-level tokens (add end-of-word marker)</li>
                                <li>Count all adjacent token pairs across the corpus</li>
                                <li>Merge the most frequent pair into a new token</li>
                                <li>Repeat until vocabulary size is reached or no more merges</li>
                            </ol>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
