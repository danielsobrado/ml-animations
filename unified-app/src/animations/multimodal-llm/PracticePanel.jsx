import React, { useState } from 'react';
import { CheckCircle, XCircle, Lightbulb, RefreshCw } from 'lucide-react';

const QUIZZES = [
    {
        question: 'What is the main challenge in building multimodal LLMs?',
        options: [
            'Making the model smaller',
            'Aligning different modalities into a shared representation space',
            'Collecting more text data',
            'Reducing training time'
        ],
        correct: 1,
        explanation: 'The key challenge is getting images, text, and audio to share a common embedding space so the language model can reason about all of them together.'
    },
    {
        question: 'What does a Vision Transformer (ViT) do in a multimodal LLM?',
        options: [
            'Generates images from text',
            'Splits images into patches and encodes them as vectors',
            'Translates between languages',
            'Compresses images for storage'
        ],
        correct: 1,
        explanation: 'ViT treats an image as a sequence of patches (like words in a sentence), encodes each patch, and produces visual embeddings that can be processed alongside text.'
    },
    {
        question: 'How do text tokens "understand" image content in cross-attention?',
        options: [
            'They are converted to images',
            'They attend to visual token embeddings during the attention mechanism',
            'They are processed separately and never interact',
            'The image is converted to text first'
        ],
        correct: 1,
        explanation: 'In cross-attention, text tokens can attend to (look at) visual tokens. When processing "What animal...", the attention weights for "animal" will be high on patches showing the animal.'
    },
    {
        question: 'Why is the projection layer important in multimodal models?',
        options: [
            'It makes images look better',
            'It maps visual embeddings to the same dimensionality as text embeddings',
            'It speeds up training',
            'It reduces model size'
        ],
        correct: 1,
        explanation: 'Visual encoders and LLMs often have different embedding dimensions. The projection layer bridges this gap, allowing visual features to be processed as if they were text tokens.'
    },
    {
        question: 'What is "early fusion" in multimodal architectures?',
        options: [
            'Combining outputs after separate processing',
            'Merging different modalities at the input level before main processing',
            'Training vision and language models separately',
            'Using only one modality'
        ],
        correct: 1,
        explanation: 'Early fusion combines modalities at the input stage, allowing deep cross-modal interactions throughout the entire network. Late fusion keeps them separate until the end.'
    }
];

// Multimodal scenarios for practice
const SCENARIOS = [
    {
        id: 1,
        image: '🏔️',
        imageDesc: 'Mountain landscape',
        question: 'What season is shown in this photo?',
        context: 'The image shows snow-capped peaks with green meadows below',
        answer: 'The image likely shows late spring or early summer - snow remains on peaks while meadows are green.'
    },
    {
        id: 2,
        image: '📊',
        imageDesc: 'Bar chart',
        question: 'What does this chart tell us about Q4 sales?',
        context: 'A bar chart showing quarterly sales with Q4 being the tallest bar',
        answer: 'Q4 shows the highest sales of the year, indicating strong holiday season performance.'
    },
    {
        id: 3,
        image: '🔧',
        imageDesc: 'Mechanical part',
        question: 'What might be wrong with this component?',
        context: 'An image of a gear with visible wear marks and discoloration',
        answer: 'The component shows signs of wear and possible overheating, suggesting it may need replacement.'
    }
];

export default function PracticePanel() {
    const [activeSection, setActiveSection] = useState('quiz');
    const [currentQuiz, setCurrentQuiz] = useState(0);
    const [selectedAnswer, setSelectedAnswer] = useState(null);
    const [showResult, setShowResult] = useState(false);
    const [score, setScore] = useState(0);
    const [selectedScenario, setSelectedScenario] = useState(0);
    const [showScenarioAnswer, setShowScenarioAnswer] = useState(false);

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
    const scenario = SCENARIOS[selectedScenario];

    return (
        <div className="p-8 h-full">
            <div className="max-w-4xl mx-auto">
                <div className="text-center mb-6">
                    <h2 className="text-3xl font-bold text-indigo-900 mb-2">Practice Lab</h2>
                    <p className="text-slate-600">Test your understanding of multimodal LLMs</p>
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
                        onClick={() => setActiveSection('scenarios')}
                        className={`px-6 py-2 rounded-lg font-bold transition-all ${
                            activeSection === 'scenarios'
                                ? 'bg-indigo-600 text-white'
                                : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                        }`}
                    >
                        🎯 Scenarios
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

                        <div className="w-full bg-slate-200 rounded-full h-2 mb-6">
                            <div
                                className="bg-indigo-600 h-2 rounded-full transition-all"
                                style={{ width: `${((currentQuiz + 1) / QUIZZES.length) * 100}%` }}
                            />
                        </div>

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
                        <h3 className="text-xl font-bold text-slate-800 mb-4">Multimodal Reasoning Scenarios</h3>
                        <p className="text-slate-600 mb-6">
                            Explore how a multimodal LLM would process these image + text combinations
                        </p>

                        {/* Scenario Selection */}
                        <div className="flex gap-2 mb-6">
                            {SCENARIOS.map((s, i) => (
                                <button
                                    key={s.id}
                                    onClick={() => {
                                        setSelectedScenario(i);
                                        setShowScenarioAnswer(false);
                                    }}
                                    className={`px-4 py-2 rounded-lg transition-all ${
                                        selectedScenario === i
                                            ? 'bg-indigo-600 text-white'
                                            : 'bg-white border hover:bg-slate-100'
                                    }`}
                                >
                                    {s.image} Scenario {s.id}
                                </button>
                            ))}
                        </div>

                        {/* Scenario Display */}
                        <div className="bg-white rounded-lg p-6 border mb-6">
                            <div className="flex gap-6">
                                {/* Image */}
                                <div className="flex-shrink-0">
                                    <div className="w-32 h-32 bg-slate-100 rounded-lg flex items-center justify-center text-6xl border-2 border-slate-200">
                                        {scenario.image}
                                    </div>
                                    <div className="text-center text-sm text-slate-500 mt-2">{scenario.imageDesc}</div>
                                </div>

                                {/* Question */}
                                <div className="flex-1">
                                    <div className="text-sm text-slate-500 mb-1">User Question:</div>
                                    <div className="text-lg font-medium text-slate-800 mb-4">
                                        "{scenario.question}"
                                    </div>

                                    <div className="text-sm text-slate-500 mb-1">Image Context:</div>
                                    <div className="text-sm text-slate-600 bg-slate-50 p-2 rounded mb-4">
                                        {scenario.context}
                                    </div>

                                    {!showScenarioAnswer ? (
                                        <button
                                            onClick={() => setShowScenarioAnswer(true)}
                                            className="px-4 py-2 bg-indigo-600 text-white rounded-lg font-medium hover:bg-indigo-700"
                                        >
                                            Show Model Response
                                        </button>
                                    ) : (
                                        <div className="bg-green-50 p-4 rounded-lg border border-green-200">
                                            <div className="text-sm text-green-600 mb-1">Model Response:</div>
                                            <div className="text-green-800 font-medium">{scenario.answer}</div>
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>

                        {/* How it works */}
                        <div className="bg-indigo-50 p-4 rounded-lg border border-indigo-200">
                            <h4 className="font-bold text-indigo-900 mb-2">How the Model Processes This:</h4>
                            <ol className="text-indigo-800 text-sm space-y-1 list-decimal list-inside">
                                <li>Vision encoder extracts features from the image ({scenario.image})</li>
                                <li>Text tokenizer processes the question</li>
                                <li>Visual and text tokens are combined</li>
                                <li>Cross-attention allows "season" or "sales" to attend to relevant visual features</li>
                                <li>Language model generates contextually appropriate response</li>
                            </ol>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
