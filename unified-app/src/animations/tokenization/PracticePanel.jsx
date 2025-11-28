import React, { useState, useEffect } from 'react';
import { FlaskConical, CheckCircle, XCircle, Lightbulb, RefreshCw } from 'lucide-react';

// Quiz questions about tokenization
const QUIZZES = [
    {
        id: 1,
        question: 'Why do modern LLMs use subword tokenization instead of word-level?',
        options: [
            'It\'s faster to compute',
            'It can handle rare/unknown words by breaking them into known pieces',
            'It produces fewer tokens',
            'It requires less memory'
        ],
        correct: 1,
        explanation: 'Subword tokenization handles Out-of-Vocabulary (OOV) words by breaking them into known subword units. For example, "unhappiness" becomes "un" + "happi" + "ness".'
    },
    {
        id: 2,
        question: 'What does the "##" prefix mean in WordPiece tokenization?',
        options: [
            'It marks the start of a new word',
            'It indicates a special token',
            'It means this token continues the previous word (not a word start)',
            'It represents a number'
        ],
        correct: 2,
        explanation: 'In WordPiece, "##" indicates that this subword is a continuation of the previous token, not the start of a new word. E.g., "playing" → ["play", "##ing"]'
    },
    {
        id: 3,
        question: 'In BPE, what determines which token pairs get merged?',
        options: [
            'Alphabetical order',
            'The length of the resulting token',
            'Frequency - most common pairs are merged first',
            'Random selection'
        ],
        correct: 2,
        explanation: 'BPE merges the most frequently occurring adjacent token pairs. This creates a vocabulary of common subwords that efficiently represent the training corpus.'
    },
    {
        id: 4,
        question: 'Which tokenization method does GPT-4 primarily use?',
        options: [
            'Character-level tokenization',
            'Word-level tokenization',
            'BPE (Byte Pair Encoding)',
            'WordPiece'
        ],
        correct: 2,
        explanation: 'GPT models use BPE (specifically cl100k_base for GPT-4), which efficiently balances vocabulary size with the ability to represent any text.'
    },
    {
        id: 5,
        question: 'What happens when a tokenizer encounters a word not in its vocabulary?',
        options: [
            'It throws an error',
            'It skips the word',
            'It breaks the word into smaller known subwords or characters',
            'It replaces it with a random word'
        ],
        correct: 2,
        explanation: 'Subword tokenizers can represent ANY text by breaking unknown words into known subword pieces or even individual characters as a fallback.'
    }
];

// Simulated tokenizer for hands-on practice
const tokenize = (text) => {
    const result = [];
    const words = text.split(/(\s+)/);
    
    words.forEach(word => {
        if (word.match(/^\s+$/)) {
            result.push({ token: word === ' ' ? '▁' : '▁'.repeat(word.length), type: 'space' });
        } else if (word.length <= 3) {
            result.push({ token: word, type: 'word' });
        } else {
            // Split into subwords
            const common = ['ing', 'tion', 'ment', 'ness', 'ful', 'less', 'able', 'ly', 'ed', 'er', 'est', 's', 'es'];
            let remaining = word;
            let isFirst = true;
            
            // Check for common prefixes
            const prefixes = ['un', 're', 'pre', 'dis', 'mis', 'over', 'under'];
            for (const prefix of prefixes) {
                if (remaining.toLowerCase().startsWith(prefix) && remaining.length > prefix.length + 2) {
                    result.push({ token: prefix, type: 'prefix' });
                    remaining = remaining.slice(prefix.length);
                    isFirst = false;
                    break;
                }
            }
            
            // Check for common suffixes
            let suffixFound = null;
            for (const suffix of common) {
                if (remaining.toLowerCase().endsWith(suffix) && remaining.length > suffix.length + 1) {
                    suffixFound = suffix;
                    break;
                }
            }
            
            if (suffixFound) {
                const stem = remaining.slice(0, -suffixFound.length);
                result.push({ token: isFirst ? stem : '##' + stem, type: 'stem' });
                result.push({ token: '##' + suffixFound, type: 'suffix' });
            } else {
                result.push({ token: isFirst ? remaining : '##' + remaining, type: 'word' });
            }
        }
    });
    
    return result;
};

export default function PracticePanel() {
    const [activeSection, setActiveSection] = useState('quiz');
    const [currentQuiz, setCurrentQuiz] = useState(0);
    const [selectedAnswer, setSelectedAnswer] = useState(null);
    const [showResult, setShowResult] = useState(false);
    const [score, setScore] = useState(0);
    const [practiceText, setPracticeText] = useState('The unhappiness of tokenization is overwhelming');
    const [tokens, setTokens] = useState([]);

    useEffect(() => {
        setTokens(tokenize(practiceText));
    }, [practiceText]);

    const handleAnswer = (index) => {
        setSelectedAnswer(index);
        setShowResult(true);
        if (index === QUIZZES[currentQuiz].correct) {
            setScore(s => s + 1);
        }
    };

    const nextQuestion = () => {
        if (currentQuiz < QUIZZES.length - 1) {
            setCurrentQuiz(c => c + 1);
            setSelectedAnswer(null);
            setShowResult(false);
        }
    };

    const resetQuiz = () => {
        setCurrentQuiz(0);
        setSelectedAnswer(null);
        setShowResult(false);
        setScore(0);
    };

    const quiz = QUIZZES[currentQuiz];

    return (
        <div className="p-8 h-full">
            <div className="max-w-4xl mx-auto">
                <div className="text-center mb-6">
                    <h2 className="text-3xl font-bold text-indigo-900 mb-2">Practice Lab</h2>
                    <p className="text-slate-600">Test your understanding of tokenization</p>
                </div>

                {/* Section Tabs */}
                <div className="flex justify-center gap-4 mb-8">
                    <button
                        onClick={() => setActiveSection('quiz')}
                        className={`px-6 py-2 rounded-lg font-bold transition-all ${
                            activeSection === 'quiz'
                                ? 'bg-indigo-600 text-white'
                                : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                        }`}
                    >
                        📝 Quiz
                    </button>
                    <button
                        onClick={() => setActiveSection('sandbox')}
                        className={`px-6 py-2 rounded-lg font-bold transition-all ${
                            activeSection === 'sandbox'
                                ? 'bg-indigo-600 text-white'
                                : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                        }`}
                    >
                        🧪 Sandbox
                    </button>
                </div>

                {activeSection === 'quiz' ? (
                    <div className="bg-slate-50 rounded-xl p-6">
                        {/* Progress */}
                        <div className="flex justify-between items-center mb-4">
                            <span className="text-sm text-slate-600">
                                Question {currentQuiz + 1} of {QUIZZES.length}
                            </span>
                            <span className="text-sm font-bold text-indigo-600">
                                Score: {score}/{QUIZZES.length}
                            </span>
                        </div>

                        {/* Progress Bar */}
                        <div className="w-full bg-slate-200 rounded-full h-2 mb-6">
                            <div
                                className="bg-indigo-600 h-2 rounded-full transition-all"
                                style={{ width: `${((currentQuiz + 1) / QUIZZES.length) * 100}%` }}
                            />
                        </div>

                        {/* Question */}
                        <div className="bg-white rounded-lg p-6 mb-4 border">
                            <h3 className="text-lg font-bold text-slate-800 mb-4">{quiz.question}</h3>
                            
                            <div className="space-y-3">
                                {quiz.options.map((option, i) => (
                                    <button
                                        key={i}
                                        onClick={() => !showResult && handleAnswer(i)}
                                        disabled={showResult}
                                        className={`w-full p-3 rounded-lg border-2 text-left transition-all ${
                                            showResult
                                                ? i === quiz.correct
                                                    ? 'bg-green-100 border-green-400'
                                                    : i === selectedAnswer
                                                        ? 'bg-red-100 border-red-400'
                                                        : 'bg-slate-50 border-slate-200'
                                                : selectedAnswer === i
                                                    ? 'bg-indigo-50 border-indigo-400'
                                                    : 'bg-white border-slate-200 hover:border-indigo-300'
                                        }`}
                                    >
                                        <div className="flex items-center gap-3">
                                            {showResult && i === quiz.correct && (
                                                <CheckCircle className="text-green-600" size={20} />
                                            )}
                                            {showResult && i === selectedAnswer && i !== quiz.correct && (
                                                <XCircle className="text-red-600" size={20} />
                                            )}
                                            <span>{option}</span>
                                        </div>
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* Explanation */}
                        {showResult && (
                            <div className={`p-4 rounded-lg mb-4 ${
                                selectedAnswer === quiz.correct
                                    ? 'bg-green-50 border border-green-200'
                                    : 'bg-amber-50 border border-amber-200'
                            }`}>
                                <div className="flex items-start gap-3">
                                    <Lightbulb className={selectedAnswer === quiz.correct ? 'text-green-600' : 'text-amber-600'} size={20} />
                                    <p className={selectedAnswer === quiz.correct ? 'text-green-800' : 'text-amber-800'}>
                                        {quiz.explanation}
                                    </p>
                                </div>
                            </div>
                        )}

                        {/* Navigation */}
                        <div className="flex justify-between">
                            <button
                                onClick={resetQuiz}
                                className="flex items-center gap-2 px-4 py-2 text-slate-600 hover:text-slate-800"
                            >
                                <RefreshCw size={16} /> Reset
                            </button>
                            
                            {showResult && currentQuiz < QUIZZES.length - 1 && (
                                <button
                                    onClick={nextQuestion}
                                    className="px-6 py-2 bg-indigo-600 text-white rounded-lg font-bold hover:bg-indigo-700"
                                >
                                    Next Question →
                                </button>
                            )}
                            
                            {showResult && currentQuiz === QUIZZES.length - 1 && (
                                <div className="text-lg font-bold text-indigo-600">
                                    Final Score: {score}/{QUIZZES.length} 🎉
                                </div>
                            )}
                        </div>
                    </div>
                ) : (
                    <div className="bg-slate-50 rounded-xl p-6">
                        <h3 className="text-xl font-bold text-slate-800 mb-4">Tokenization Sandbox</h3>
                        
                        <div className="mb-4">
                            <label className="block text-sm font-medium text-slate-700 mb-2">
                                Enter text to tokenize:
                            </label>
                            <textarea
                                value={practiceText}
                                onChange={(e) => setPracticeText(e.target.value)}
                                className="w-full p-3 border-2 border-slate-200 rounded-lg focus:border-indigo-500 resize-none"
                                rows={3}
                                placeholder="Type something to see how it gets tokenized..."
                            />
                        </div>

                        {/* Quick Examples */}
                        <div className="flex flex-wrap gap-2 mb-6">
                            {[
                                'unhappiness',
                                'tokenization is cool',
                                'GPT-4 loves embeddings',
                                'المرحبا',
                                '🎉 emoji test'
                            ].map(ex => (
                                <button
                                    key={ex}
                                    onClick={() => setPracticeText(ex)}
                                    className="px-3 py-1 text-xs bg-white border rounded-full hover:bg-slate-100"
                                >
                                    {ex}
                                </button>
                            ))}
                        </div>

                        {/* Token Output */}
                        <div className="bg-white rounded-lg p-4 border">
                            <div className="flex justify-between items-center mb-3">
                                <h4 className="font-bold text-slate-700">Tokens ({tokens.length})</h4>
                                <span className="text-xs text-slate-500">
                                    ~{(practiceText.length / 4).toFixed(0)} GPT tokens estimate
                                </span>
                            </div>
                            
                            <div className="flex flex-wrap gap-2">
                                {tokens.map((t, i) => (
                                    <div
                                        key={i}
                                        className={`px-3 py-1 rounded-lg font-mono text-sm flex items-center gap-1 ${
                                            t.type === 'prefix' ? 'bg-orange-100 border-2 border-orange-300' :
                                            t.type === 'suffix' ? 'bg-purple-100 border-2 border-purple-300' :
                                            t.type === 'stem' ? 'bg-blue-100 border-2 border-blue-300' :
                                            t.type === 'space' ? 'bg-slate-100 border-2 border-slate-300' :
                                            'bg-green-100 border-2 border-green-300'
                                        }`}
                                    >
                                        <span className="text-xs text-slate-500">{i}</span>
                                        {t.token}
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Legend */}
                        <div className="mt-4 flex flex-wrap gap-4 text-xs">
                            <div className="flex items-center gap-1">
                                <span className="w-4 h-4 bg-orange-100 border-2 border-orange-300 rounded"></span>
                                <span>Prefix</span>
                            </div>
                            <div className="flex items-center gap-1">
                                <span className="w-4 h-4 bg-blue-100 border-2 border-blue-300 rounded"></span>
                                <span>Stem</span>
                            </div>
                            <div className="flex items-center gap-1">
                                <span className="w-4 h-4 bg-purple-100 border-2 border-purple-300 rounded"></span>
                                <span>Suffix (##)</span>
                            </div>
                            <div className="flex items-center gap-1">
                                <span className="w-4 h-4 bg-green-100 border-2 border-green-300 rounded"></span>
                                <span>Word</span>
                            </div>
                            <div className="flex items-center gap-1">
                                <span className="w-4 h-4 bg-slate-100 border-2 border-slate-300 rounded"></span>
                                <span>Space</span>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
