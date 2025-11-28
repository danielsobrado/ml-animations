import React, { useState, useEffect } from 'react';
import { FlaskConical, CheckCircle, XCircle, Lightbulb, RefreshCw, Calculator } from 'lucide-react';

// Quiz questions about embeddings
const QUIZZES = [
    {
        id: 1,
        question: 'What is the main purpose of word embeddings?',
        options: [
            'To compress text files',
            'To represent words as dense numerical vectors that capture semantic meaning',
            'To count word frequencies',
            'To translate between languages'
        ],
        correct: 1,
        explanation: 'Embeddings convert words into dense vectors where similar words are close together. This allows neural networks to process and understand semantic relationships between words.'
    },
    {
        id: 2,
        question: 'Why is cosine similarity preferred over Euclidean distance for comparing text embeddings?',
        options: [
            'It\'s faster to compute',
            'It produces larger numbers',
            'It measures angle (direction) rather than magnitude, making it robust to document length',
            'It always returns positive values'
        ],
        correct: 2,
        explanation: 'Cosine similarity measures the angle between vectors, ignoring their magnitudes. This makes it ideal for text where a longer document shouldn\'t be considered "further" from a short one with similar content.'
    },
    {
        id: 3,
        question: 'What does the famous equation "King - Man + Woman ≈ Queen" demonstrate?',
        options: [
            'Embeddings can do arithmetic',
            'Embeddings capture semantic relationships and analogies through vector operations',
            'Queens are better than kings',
            'Word2Vec is always accurate'
        ],
        correct: 1,
        explanation: 'This shows that embeddings encode semantic relationships as directions in vector space. The "gender direction" (man→woman) can be applied to other words to find analogous relationships.'
    },
    {
        id: 4,
        question: 'How many dimensions do modern language model embeddings typically have?',
        options: [
            '2-3 dimensions',
            '50-100 dimensions',
            '768-12,288 dimensions',
            'Millions of dimensions'
        ],
        correct: 2,
        explanation: 'BERT uses 768 dimensions, GPT-3 uses 12,288 dimensions. Higher dimensions can capture more nuanced relationships but require more computation.'
    },
    {
        id: 5,
        question: 'What happens to words that are semantically similar in embedding space?',
        options: [
            'They have identical embeddings',
            'They are far apart in the vector space',
            'They are close together in the vector space',
            'They have opposite embeddings'
        ],
        correct: 2,
        explanation: 'Similar words cluster together in embedding space. "Happy", "joyful", and "glad" will be close to each other, while "sad" will be further away.'
    }
];

// Embeddings for analogy exercise
const ANALOGY_EMBEDDINGS = {
    'man': [1, 0],
    'woman': [0.9, 0.1],
    'king': [1.5, 0.8],
    'queen': [1.4, 0.9],
    'uncle': [1.2, 0.3],
    'aunt': [1.1, 0.4],
    'boy': [0.8, -0.1],
    'girl': [0.7, 0],
};

export default function PracticePanel() {
    const [activeSection, setActiveSection] = useState('quiz');
    const [currentQuiz, setCurrentQuiz] = useState(0);
    const [selectedAnswer, setSelectedAnswer] = useState(null);
    const [showResult, setShowResult] = useState(false);
    const [score, setScore] = useState(0);
    
    // Analogy exercise state
    const [analogyWord1, setAnalogyWord1] = useState('man');
    const [analogyWord2, setAnalogyWord2] = useState('woman');
    const [analogyWord3, setAnalogyWord3] = useState('king');
    const [analogyResult, setAnalogyResult] = useState(null);

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

    const calculateAnalogy = () => {
        const v1 = ANALOGY_EMBEDDINGS[analogyWord1];
        const v2 = ANALOGY_EMBEDDINGS[analogyWord2];
        const v3 = ANALOGY_EMBEDDINGS[analogyWord3];
        
        // v3 - v1 + v2 = result
        const resultVector = [
            v3[0] - v1[0] + v2[0],
            v3[1] - v1[1] + v2[1]
        ];
        
        // Find closest word
        let closestWord = '';
        let minDist = Infinity;
        
        Object.entries(ANALOGY_EMBEDDINGS).forEach(([word, vec]) => {
            if (word === analogyWord1 || word === analogyWord2 || word === analogyWord3) return;
            const dist = Math.sqrt(
                (vec[0] - resultVector[0]) ** 2 + 
                (vec[1] - resultVector[1]) ** 2
            );
            if (dist < minDist) {
                minDist = dist;
                closestWord = word;
            }
        });
        
        setAnalogyResult({
            vector: resultVector,
            closestWord,
            distance: minDist
        });
    };

    const quiz = QUIZZES[currentQuiz];
    const words = Object.keys(ANALOGY_EMBEDDINGS);

    return (
        <div className="p-8 h-full">
            <div className="max-w-4xl mx-auto">
                <div className="text-center mb-6">
                    <h2 className="text-3xl font-bold text-indigo-900 mb-2">Practice Lab</h2>
                    <p className="text-slate-600">Test your understanding of embeddings</p>
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
                        onClick={() => setActiveSection('analogy')}
                        className={`px-6 py-2 rounded-lg font-bold transition-all ${
                            activeSection === 'analogy'
                                ? 'bg-indigo-600 text-white'
                                : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                        }`}
                    >
                        🧮 Analogy Calculator
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
                        <h3 className="text-xl font-bold text-slate-800 mb-4">Word Analogy Calculator</h3>
                        <p className="text-slate-600 mb-6">
                            Try the famous word analogy: A is to B as C is to ?
                        </p>

                        {/* Formula Display */}
                        <div className="bg-white rounded-lg p-4 border mb-6">
                            <div className="flex items-center justify-center gap-3 flex-wrap text-lg">
                                <select
                                    value={analogyWord3}
                                    onChange={(e) => setAnalogyWord3(e.target.value)}
                                    className="px-3 py-2 border-2 border-purple-300 rounded-lg bg-purple-50 font-bold"
                                >
                                    {words.map(w => <option key={w} value={w}>{w}</option>)}
                                </select>
                                <span className="text-slate-500">−</span>
                                <select
                                    value={analogyWord1}
                                    onChange={(e) => setAnalogyWord1(e.target.value)}
                                    className="px-3 py-2 border-2 border-blue-300 rounded-lg bg-blue-50 font-bold"
                                >
                                    {words.map(w => <option key={w} value={w}>{w}</option>)}
                                </select>
                                <span className="text-slate-500">+</span>
                                <select
                                    value={analogyWord2}
                                    onChange={(e) => setAnalogyWord2(e.target.value)}
                                    className="px-3 py-2 border-2 border-blue-300 rounded-lg bg-blue-50 font-bold"
                                >
                                    {words.map(w => <option key={w} value={w}>{w}</option>)}
                                </select>
                                <span className="text-slate-500">=</span>
                                <span className="px-3 py-2 border-2 border-green-300 rounded-lg bg-green-50 font-bold min-w-[80px] text-center">
                                    {analogyResult?.closestWord || '?'}
                                </span>
                            </div>
                            <p className="text-center text-sm text-slate-500 mt-2">
                                "{analogyWord1}" is to "{analogyWord2}" as "{analogyWord3}" is to "?"
                            </p>
                        </div>

                        <div className="flex justify-center mb-6">
                            <button
                                onClick={calculateAnalogy}
                                className="px-6 py-3 bg-indigo-600 text-white rounded-lg font-bold hover:bg-indigo-700 flex items-center gap-2"
                            >
                                <Calculator size={20} />
                                Calculate Analogy
                            </button>
                        </div>

                        {/* Result */}
                        {analogyResult && (
                            <div className="bg-green-50 rounded-lg p-4 border border-green-200">
                                <h4 className="font-bold text-green-900 mb-2">Result</h4>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                    <div>
                                        <div className="text-sm text-green-700">Computed Vector:</div>
                                        <div className="font-mono bg-white p-2 rounded mt-1">
                                            [{analogyResult.vector[0].toFixed(2)}, {analogyResult.vector[1].toFixed(2)}]
                                        </div>
                                    </div>
                                    <div>
                                        <div className="text-sm text-green-700">Closest Word:</div>
                                        <div className="text-2xl font-bold text-green-900 mt-1">
                                            {analogyResult.closestWord}
                                        </div>
                                        <div className="text-xs text-green-600">
                                            Distance: {analogyResult.distance.toFixed(3)}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* Suggested analogies */}
                        <div className="mt-6">
                            <h4 className="font-medium text-slate-700 mb-2">Try these analogies:</h4>
                            <div className="flex flex-wrap gap-2">
                                {[
                                    { a: 'man', b: 'woman', c: 'king', result: 'queen' },
                                    { a: 'man', b: 'woman', c: 'uncle', result: 'aunt' },
                                    { a: 'man', b: 'woman', c: 'boy', result: 'girl' },
                                ].map((analogy, i) => (
                                    <button
                                        key={i}
                                        onClick={() => {
                                            setAnalogyWord1(analogy.a);
                                            setAnalogyWord2(analogy.b);
                                            setAnalogyWord3(analogy.c);
                                            setAnalogyResult(null);
                                        }}
                                        className="px-3 py-1 text-sm bg-white border rounded-full hover:bg-slate-100"
                                    >
                                        {analogy.c} - {analogy.a} + {analogy.b} = {analogy.result}
                                    </button>
                                ))}
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
