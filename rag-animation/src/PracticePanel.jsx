import React, { useState } from 'react';
import { CheckCircle2, XCircle, RotateCcw, Brain, Database, Search, FileText, Lightbulb } from 'lucide-react';

const quizQuestions = [
    {
        id: 1,
        question: 'What is the main purpose of RAG?',
        options: [
            'To retrain the LLM on new data',
            'To augment LLM responses with retrieved external knowledge',
            'To compress the model size',
            'To speed up inference time'
        ],
        correct: 1,
        explanation: 'RAG augments LLM responses by retrieving relevant information from an external knowledge base, without retraining the model.'
    },
    {
        id: 2,
        question: 'What is the purpose of "chunking" in a RAG pipeline?',
        options: [
            'To compress documents for storage',
            'To split documents into smaller pieces for better retrieval',
            'To remove duplicate content',
            'To translate documents'
        ],
        correct: 1,
        explanation: 'Chunking splits large documents into smaller pieces (typically 256-512 tokens) so that relevant sections can be retrieved and fit into the LLM context window.'
    },
    {
        id: 3,
        question: 'Why are embeddings used in RAG?',
        options: [
            'To encrypt the documents',
            'To convert text into vectors for similarity search',
            'To compress the text',
            'To translate between languages'
        ],
        correct: 1,
        explanation: 'Embeddings convert text into dense vectors, enabling semantic similarity search to find relevant content even when exact keywords don\'t match.'
    },
    {
        id: 4,
        question: 'What does "Top-K retrieval" refer to?',
        options: [
            'Retrieving documents from K different databases',
            'Retrieving the K most similar chunks to the query',
            'Using K different embedding models',
            'Generating K different responses'
        ],
        correct: 1,
        explanation: 'Top-K retrieval returns the K most similar document chunks to the query based on vector similarity scores.'
    },
    {
        id: 5,
        question: 'What is a key advantage of RAG over fine-tuning?',
        options: [
            'RAG produces faster responses',
            'RAG doesn\'t require GPUs',
            'RAG knowledge can be updated without retraining',
            'RAG uses less memory'
        ],
        correct: 2,
        explanation: 'RAG allows knowledge updates by simply adding or modifying documents in the knowledge base, without expensive model retraining.'
    }
];

export default function PracticePanel() {
    const [mode, setMode] = useState('quiz'); // 'quiz' or 'sandbox'
    const [currentQuestion, setCurrentQuestion] = useState(0);
    const [answers, setAnswers] = useState({});
    const [showResults, setShowResults] = useState(false);
    const [selectedAnswer, setSelectedAnswer] = useState(null);
    const [showExplanation, setShowExplanation] = useState(false);

    // Sandbox state
    const [sandboxQuery, setSandboxQuery] = useState('');
    const [sandboxDocs, setSandboxDocs] = useState([
        { id: 1, text: 'Python is a high-level programming language known for its readability.', enabled: true },
        { id: 2, text: 'JavaScript runs in web browsers and enables interactive websites.', enabled: true },
        { id: 3, text: 'Machine learning models can be trained using Python libraries like TensorFlow.', enabled: true },
        { id: 4, text: 'The weather forecast shows sunny skies for the weekend.', enabled: true },
    ]);
    const [sandboxResult, setSandboxResult] = useState(null);

    const handleAnswer = (optionIndex) => {
        setSelectedAnswer(optionIndex);
        setShowExplanation(true);
        setAnswers({ ...answers, [currentQuestion]: optionIndex });
    };

    const nextQuestion = () => {
        if (currentQuestion < quizQuestions.length - 1) {
            setCurrentQuestion(currentQuestion + 1);
            setSelectedAnswer(null);
            setShowExplanation(false);
        } else {
            setShowResults(true);
        }
    };

    const resetQuiz = () => {
        setCurrentQuestion(0);
        setAnswers({});
        setShowResults(false);
        setSelectedAnswer(null);
        setShowExplanation(false);
    };

    const getScore = () => {
        let correct = 0;
        Object.entries(answers).forEach(([q, a]) => {
            if (quizQuestions[parseInt(q)].correct === a) correct++;
        });
        return correct;
    };

    // Simple keyword-based "retrieval" for sandbox
    const performRetrieval = () => {
        const queryWords = sandboxQuery.toLowerCase().split(/\s+/);
        const enabledDocs = sandboxDocs.filter(d => d.enabled);
        
        const scored = enabledDocs.map(doc => {
            const docWords = doc.text.toLowerCase();
            let score = 0;
            queryWords.forEach(word => {
                if (word.length > 2 && docWords.includes(word)) {
                    score += 1;
                }
            });
            return { ...doc, score: score / queryWords.length };
        });

        const sorted = scored.sort((a, b) => b.score - a.score);
        const topResult = sorted[0];

        if (topResult && topResult.score > 0) {
            setSandboxResult({
                retrieved: topResult,
                response: `Based on the retrieved context: "${topResult.text.substring(0, 50)}...", here is a response about your query.`
            });
        } else {
            setSandboxResult({
                retrieved: null,
                response: 'No relevant documents found. The model would need to rely on its training data alone.'
            });
        }
    };

    const q = quizQuestions[currentQuestion];

    return (
        <div className="p-6 h-full overflow-y-auto">
            <div className="max-w-4xl mx-auto">
                {/* Mode Toggle */}
                <div className="flex justify-center gap-4 mb-6">
                    <button
                        onClick={() => setMode('quiz')}
                        className={`px-6 py-2 rounded-lg font-medium transition-all ${
                            mode === 'quiz'
                                ? 'bg-indigo-500 text-white'
                                : 'bg-white border hover:bg-slate-50'
                        }`}
                    >
                        📝 Quiz
                    </button>
                    <button
                        onClick={() => setMode('sandbox')}
                        className={`px-6 py-2 rounded-lg font-medium transition-all ${
                            mode === 'sandbox'
                                ? 'bg-indigo-500 text-white'
                                : 'bg-white border hover:bg-slate-50'
                        }`}
                    >
                        🧪 RAG Sandbox
                    </button>
                </div>

                {mode === 'quiz' ? (
                    // Quiz Mode
                    <div className="bg-white rounded-xl p-6 shadow-lg border">
                        {!showResults ? (
                            <>
                                {/* Progress */}
                                <div className="flex justify-between items-center mb-6">
                                    <span className="text-sm text-slate-500">
                                        Question {currentQuestion + 1} of {quizQuestions.length}
                                    </span>
                                    <div className="flex gap-1">
                                        {quizQuestions.map((_, i) => (
                                            <div
                                                key={i}
                                                className={`w-3 h-3 rounded-full ${
                                                    i < currentQuestion ? 'bg-green-500' :
                                                    i === currentQuestion ? 'bg-indigo-500' : 'bg-slate-200'
                                                }`}
                                            />
                                        ))}
                                    </div>
                                </div>

                                {/* Question */}
                                <h3 className="text-xl font-bold text-slate-800 mb-6">{q.question}</h3>

                                {/* Options */}
                                <div className="space-y-3 mb-6">
                                    {q.options.map((option, i) => {
                                        const isSelected = selectedAnswer === i;
                                        const isCorrect = q.correct === i;
                                        const showCorrectness = showExplanation;

                                        return (
                                            <button
                                                key={i}
                                                onClick={() => !showExplanation && handleAnswer(i)}
                                                disabled={showExplanation}
                                                className={`w-full text-left p-4 rounded-lg border transition-all ${
                                                    showCorrectness
                                                        ? isCorrect
                                                            ? 'bg-green-100 border-green-500 text-green-800'
                                                            : isSelected
                                                                ? 'bg-red-100 border-red-500 text-red-800'
                                                                : 'bg-slate-50 border-slate-200'
                                                        : isSelected
                                                            ? 'bg-indigo-100 border-indigo-500'
                                                            : 'bg-white border-slate-200 hover:bg-slate-50'
                                                }`}
                                            >
                                                <div className="flex items-center gap-3">
                                                    {showCorrectness && isCorrect && (
                                                        <CheckCircle2 className="text-green-500" size={20} />
                                                    )}
                                                    {showCorrectness && isSelected && !isCorrect && (
                                                        <XCircle className="text-red-500" size={20} />
                                                    )}
                                                    <span>{option}</span>
                                                </div>
                                            </button>
                                        );
                                    })}
                                </div>

                                {/* Explanation */}
                                {showExplanation && (
                                    <div className="bg-blue-50 p-4 rounded-lg border border-blue-200 mb-4">
                                        <div className="flex items-start gap-2">
                                            <Lightbulb className="text-blue-500 flex-shrink-0 mt-1" size={18} />
                                            <p className="text-blue-800">{q.explanation}</p>
                                        </div>
                                    </div>
                                )}

                                {/* Next Button */}
                                {showExplanation && (
                                    <button
                                        onClick={nextQuestion}
                                        className="w-full py-3 bg-indigo-500 text-white rounded-lg font-medium hover:bg-indigo-600"
                                    >
                                        {currentQuestion < quizQuestions.length - 1 ? 'Next Question' : 'See Results'}
                                    </button>
                                )}
                            </>
                        ) : (
                            // Results
                            <div className="text-center">
                                <div className="text-6xl mb-4">
                                    {getScore() === quizQuestions.length ? '🏆' : getScore() >= 3 ? '⭐' : '📚'}
                                </div>
                                <h3 className="text-2xl font-bold text-slate-800 mb-2">
                                    Quiz Complete!
                                </h3>
                                <p className="text-xl text-slate-600 mb-6">
                                    You scored <span className="font-bold text-indigo-600">{getScore()}</span> out of {quizQuestions.length}
                                </p>
                                <button
                                    onClick={resetQuiz}
                                    className="flex items-center gap-2 mx-auto px-6 py-3 bg-indigo-500 text-white rounded-lg hover:bg-indigo-600"
                                >
                                    <RotateCcw size={18} />
                                    Try Again
                                </button>
                            </div>
                        )}
                    </div>
                ) : (
                    // Sandbox Mode
                    <div className="space-y-4">
                        <div className="bg-white rounded-xl p-6 shadow-lg border">
                            <h3 className="text-lg font-bold text-slate-800 mb-4 flex items-center gap-2">
                                <Database size={20} className="text-purple-500" />
                                Knowledge Base
                            </h3>
                            <div className="space-y-2 mb-4">
                                {sandboxDocs.map(doc => (
                                    <div
                                        key={doc.id}
                                        className={`flex items-start gap-3 p-3 rounded-lg border ${
                                            doc.enabled ? 'bg-green-50 border-green-200' : 'bg-slate-50 border-slate-200 opacity-50'
                                        }`}
                                    >
                                        <input
                                            type="checkbox"
                                            checked={doc.enabled}
                                            onChange={() => {
                                                setSandboxDocs(docs =>
                                                    docs.map(d =>
                                                        d.id === doc.id ? { ...d, enabled: !d.enabled } : d
                                                    )
                                                );
                                            }}
                                            className="mt-1"
                                        />
                                        <span className="text-sm text-slate-700">{doc.text}</span>
                                    </div>
                                ))}
                            </div>
                            <button
                                onClick={() => setSandboxDocs(docs => [
                                    ...docs,
                                    { id: Date.now(), text: 'New document content...', enabled: true }
                                ])}
                                className="text-sm text-indigo-600 hover:text-indigo-800"
                            >
                                + Add Document
                            </button>
                        </div>

                        <div className="bg-white rounded-xl p-6 shadow-lg border">
                            <h3 className="text-lg font-bold text-slate-800 mb-4 flex items-center gap-2">
                                <Search size={20} className="text-blue-500" />
                                Query
                            </h3>
                            <div className="flex gap-3">
                                <input
                                    type="text"
                                    value={sandboxQuery}
                                    onChange={(e) => setSandboxQuery(e.target.value)}
                                    placeholder="Ask something related to your documents..."
                                    className="flex-1 px-4 py-2 border rounded-lg focus:ring-2 focus:ring-indigo-500"
                                />
                                <button
                                    onClick={performRetrieval}
                                    disabled={!sandboxQuery.trim()}
                                    className="px-6 py-2 bg-indigo-500 text-white rounded-lg hover:bg-indigo-600 disabled:opacity-50"
                                >
                                    Retrieve & Generate
                                </button>
                            </div>
                        </div>

                        {sandboxResult && (
                            <div className="bg-white rounded-xl p-6 shadow-lg border">
                                <h3 className="text-lg font-bold text-slate-800 mb-4 flex items-center gap-2">
                                    <Brain size={20} className="text-green-500" />
                                    Result
                                </h3>
                                
                                {sandboxResult.retrieved ? (
                                    <div className="space-y-3">
                                        <div className="bg-green-50 p-3 rounded-lg border border-green-200">
                                            <div className="text-xs text-green-600 font-medium mb-1">
                                                Retrieved Context (Score: {(sandboxResult.retrieved.score * 100).toFixed(0)}%)
                                            </div>
                                            <p className="text-green-800">{sandboxResult.retrieved.text}</p>
                                        </div>
                                        <div className="bg-purple-50 p-3 rounded-lg border border-purple-200">
                                            <div className="text-xs text-purple-600 font-medium mb-1">
                                                Generated Response
                                            </div>
                                            <p className="text-purple-800">{sandboxResult.response}</p>
                                        </div>
                                    </div>
                                ) : (
                                    <div className="bg-amber-50 p-3 rounded-lg border border-amber-200">
                                        <p className="text-amber-800">{sandboxResult.response}</p>
                                    </div>
                                )}
                            </div>
                        )}

                        <div className="bg-amber-50 p-4 rounded-xl border border-amber-200">
                            <h4 className="font-bold text-amber-900 mb-2 flex items-center gap-2">
                                <Lightbulb size={18} />
                                Tips
                            </h4>
                            <ul className="text-sm text-amber-800 space-y-1">
                                <li>• Try enabling/disabling documents to see how retrieval changes</li>
                                <li>• Use keywords that appear in your documents</li>
                                <li>• Notice how irrelevant documents (like weather) don't get retrieved for tech queries</li>
                            </ul>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
