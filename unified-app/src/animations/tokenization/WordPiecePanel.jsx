import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, ChevronRight, ChevronLeft, Lightbulb, ArrowRight } from 'lucide-react';
import gsap from 'gsap';

// WordPiece Algorithm Simulation (greedy longest-match)
const simulateWordPiece = (text, vocabulary) => {
    const steps = [];
    const words = text.split(/\s+/);
    
    words.forEach(word => {
        const wordSteps = [];
        let remaining = word;
        const tokens = [];
        let isFirst = true;
        
        wordSteps.push({
            action: 'start',
            word,
            remaining,
            tokens: [],
            description: `Processing word: "${word}"`
        });
        
        while (remaining.length > 0) {
            let found = false;
            
            // Try longest match first
            for (let end = remaining.length; end > 0; end--) {
                const substr = isFirst ? remaining.slice(0, end) : '##' + remaining.slice(0, end);
                const checkStr = isFirst ? remaining.slice(0, end) : remaining.slice(0, end);
                
                if (vocabulary.includes(substr) || vocabulary.includes(checkStr)) {
                    const token = isFirst ? checkStr : '##' + checkStr;
                    tokens.push(token);
                    
                    wordSteps.push({
                        action: 'match',
                        word,
                        tried: substr,
                        matched: token,
                        remaining: remaining.slice(end),
                        tokens: [...tokens],
                        description: `Found "${token}" in vocabulary`
                    });
                    
                    remaining = remaining.slice(end);
                    isFirst = false;
                    found = true;
                    break;
                }
            }
            
            if (!found) {
                // Unknown token (single character)
                const token = isFirst ? remaining[0] : '##' + remaining[0];
                tokens.push('[UNK]');
                
                wordSteps.push({
                    action: 'unknown',
                    word,
                    tried: remaining,
                    remaining: remaining.slice(1),
                    tokens: [...tokens],
                    description: `No match found, using [UNK] for "${remaining[0]}"`
                });
                
                remaining = remaining.slice(1);
                isFirst = false;
            }
        }
        
        wordSteps.push({
            action: 'complete',
            word,
            tokens,
            remaining: '',
            description: `Complete: "${word}" → [${tokens.join(', ')}]`
        });
        
        steps.push(...wordSteps);
    });
    
    return steps;
};

// Sample vocabulary (subset of BERT-like vocabulary)
const SAMPLE_VOCAB = [
    // Common words
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
    'un', 'happy', 'happi', 'play', 'learn', 'token', 'ization', 'embed',
    // Subwords
    '##ing', '##ed', '##er', '##est', '##ly', '##ness', '##ment', '##tion', '##ize',
    '##s', '##es', '##ful', '##less', '##able', '##ible', '##al', '##ial',
    // Characters (fallback)
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    '##a', '##b', '##c', '##d', '##e', '##f', '##g', '##h', '##i', '##j',
    '##k', '##l', '##m', '##n', '##o', '##p', '##q', '##r', '##s', '##t',
    '##u', '##v', '##w', '##x', '##y', '##z'
];

export default function WordPiecePanel() {
    const [inputText, setInputText] = useState('unhappiness playing');
    const [steps, setSteps] = useState([]);
    const [currentStep, setCurrentStep] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const visualRef = useRef(null);

    useEffect(() => {
        const result = simulateWordPiece(inputText.toLowerCase(), SAMPLE_VOCAB);
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
        }, 1500);
        
        return () => clearInterval(timer);
    }, [isPlaying, steps.length]);

    useEffect(() => {
        if (visualRef.current) {
            gsap.fromTo(
                visualRef.current,
                { opacity: 0, y: 10 },
                { opacity: 1, y: 0, duration: 0.3 }
            );
        }
    }, [currentStep]);

    const currentStepData = steps[currentStep] || {};

    return (
        <div className="p-8 h-full">
            <div className="max-w-5xl mx-auto">
                {/* Header */}
                <div className="text-center mb-6">
                    <h2 className="text-3xl font-bold text-indigo-900 mb-2">WordPiece Tokenization</h2>
                    <p className="text-slate-600">
                        Used by BERT - greedy longest-match-first algorithm
                    </p>
                </div>

                {/* Input */}
                <div className="mb-6">
                    <label className="block text-sm font-medium text-slate-700 mb-2">Text to tokenize:</label>
                    <div className="flex gap-2">
                        <input
                            type="text"
                            value={inputText}
                            onChange={(e) => setInputText(e.target.value)}
                            className="flex-1 p-3 border-2 border-slate-200 rounded-lg focus:border-indigo-500"
                            placeholder="Enter text to tokenize..."
                        />
                        <div className="flex gap-1">
                            {['unhappiness playing', 'tokenization is fun', 'embeddings learned'].map(ex => (
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
                        className="p-2 rounded-lg bg-slate-100 hover:bg-slate-200"
                    >
                        <RotateCcw size={20} />
                    </button>
                    <button
                        onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
                        disabled={currentStep === 0}
                        className="p-2 rounded-lg bg-slate-100 hover:bg-slate-200 disabled:opacity-50"
                    >
                        <ChevronLeft size={20} />
                    </button>
                    <button
                        onClick={() => setIsPlaying(!isPlaying)}
                        className={`px-6 py-2 rounded-lg font-bold ${
                            isPlaying ? 'bg-red-500 text-white' : 'bg-indigo-600 text-white'
                        }`}
                    >
                        {isPlaying ? <Pause size={20} /> : <Play size={20} />}
                    </button>
                    <button
                        onClick={() => setCurrentStep(Math.min(steps.length - 1, currentStep + 1))}
                        disabled={currentStep === steps.length - 1}
                        className="p-2 rounded-lg bg-slate-100 hover:bg-slate-200 disabled:opacity-50"
                    >
                        <ChevronRight size={20} />
                    </button>
                    <span className="text-sm text-slate-600">
                        Step {currentStep + 1} / {steps.length}
                    </span>
                </div>

                {/* Visualization */}
                <div ref={visualRef} className="bg-slate-50 rounded-xl p-6 mb-6">
                    {/* Current Step Info */}
                    <div className={`p-4 rounded-lg mb-4 ${
                        currentStepData.action === 'match' ? 'bg-green-100 border border-green-300' :
                        currentStepData.action === 'unknown' ? 'bg-red-100 border border-red-300' :
                        currentStepData.action === 'complete' ? 'bg-blue-100 border border-blue-300' :
                        'bg-white border border-slate-200'
                    }`}>
                        <p className="font-medium">{currentStepData.description}</p>
                    </div>

                    {/* Word Processing Visual */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        {/* Original Word */}
                        <div className="bg-white p-4 rounded-lg border">
                            <h4 className="text-sm font-medium text-slate-600 mb-2">Current Word</h4>
                            <div className="text-2xl font-mono font-bold text-slate-800">
                                {currentStepData.word || '-'}
                            </div>
                        </div>

                        {/* Remaining */}
                        <div className="bg-white p-4 rounded-lg border">
                            <h4 className="text-sm font-medium text-slate-600 mb-2">Remaining</h4>
                            <div className="text-2xl font-mono">
                                {currentStepData.remaining !== undefined ? (
                                    currentStepData.remaining || <span className="text-green-600">✓ Done</span>
                                ) : '-'}
                            </div>
                        </div>

                        {/* Current Match */}
                        <div className="bg-white p-4 rounded-lg border">
                            <h4 className="text-sm font-medium text-slate-600 mb-2">Trying to Match</h4>
                            <div className="text-2xl font-mono text-indigo-600">
                                {currentStepData.tried || '-'}
                            </div>
                        </div>
                    </div>

                    {/* Tokens So Far */}
                    <div className="mt-4 p-4 bg-white rounded-lg border">
                        <h4 className="text-sm font-medium text-slate-600 mb-2">Tokens Generated</h4>
                        <div className="flex flex-wrap gap-2">
                            {currentStepData.tokens?.length > 0 ? (
                                currentStepData.tokens.map((token, i) => (
                                    <span
                                        key={i}
                                        className={`px-3 py-1 rounded-lg font-mono ${
                                            token.startsWith('##') 
                                                ? 'bg-purple-100 border-2 border-purple-300' 
                                                : token === '[UNK]'
                                                    ? 'bg-red-100 border-2 border-red-300'
                                                    : 'bg-blue-100 border-2 border-blue-300'
                                        }`}
                                    >
                                        {token}
                                    </span>
                                ))
                            ) : (
                                <span className="text-slate-400">No tokens yet</span>
                            )}
                        </div>
                    </div>
                </div>

                {/* Vocabulary Panel */}
                <div className="bg-slate-50 rounded-xl p-4 mb-6">
                    <h4 className="font-bold text-slate-800 mb-3">Sample Vocabulary (subset)</h4>
                    <div className="flex flex-wrap gap-1 max-h-32 overflow-y-auto">
                        {SAMPLE_VOCAB.slice(0, 40).map((token, i) => (
                            <span
                                key={i}
                                className={`px-2 py-0.5 rounded text-xs font-mono ${
                                    token.startsWith('##')
                                        ? 'bg-purple-100 text-purple-800'
                                        : 'bg-blue-100 text-blue-800'
                                }`}
                            >
                                {token}
                            </span>
                        ))}
                        <span className="text-slate-500 text-xs">...and more</span>
                    </div>
                </div>

                {/* Key Differences */}
                <div className="bg-amber-50 p-4 rounded-xl border border-amber-200">
                    <div className="flex items-start gap-3">
                        <Lightbulb className="text-amber-600 flex-shrink-0 mt-1" size={24} />
                        <div>
                            <h4 className="font-bold text-amber-900 mb-2">WordPiece vs BPE</h4>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                                <div>
                                    <h5 className="font-bold text-amber-800">WordPiece</h5>
                                    <ul className="text-amber-800 list-disc list-inside">
                                        <li>Uses ## prefix for continuation</li>
                                        <li>Greedy longest-match decoding</li>
                                        <li>Used by BERT, DistilBERT</li>
                                    </ul>
                                </div>
                                <div>
                                    <h5 className="font-bold text-amber-800">BPE</h5>
                                    <ul className="text-amber-800 list-disc list-inside">
                                        <li>Uses &lt;/w&gt; suffix for word end</li>
                                        <li>Pair frequency-based merging</li>
                                        <li>Used by GPT, RoBERTa</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
