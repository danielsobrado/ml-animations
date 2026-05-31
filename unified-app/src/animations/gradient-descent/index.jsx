import React, { useState, useCallback, Suspense, lazy } from 'react';
import { Play, LineChart, FlaskConical } from 'lucide-react';

// Lazy load panels
const GradientDescentPanel = lazy(() => import('./GradientDescentPanel'));
const LossHistoryPanel = lazy(() => import('./LossHistoryPanel'));
const PracticePanel = lazy(() => import('./PracticePanel'));

// Tab configuration
const tabs = [
    { id: 'descent', label: '1. Gradient Descent', icon: Play, color: 'from-blue-500 to-cyan-500' },
    { id: 'history', label: '2. Loss History', icon: LineChart, color: 'from-green-500 to-emerald-500' },
    { id: 'practice', label: '3. Practice Lab', icon: FlaskConical, color: 'from-rose-500 to-red-500' },
];

// Loading fallback
function LoadingPanel() {
    return (
        <div className="flex items-center justify-center p-12">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
        </div>
    );
}

export default function GradientDescentAnimation() {
    const [activeTab, setActiveTab] = useState('descent');
    const [learningRate, setLearningRate] = useState(0.1);
    const [startWeight, setStartWeight] = useState(4);
    const [stepHistory, setStepHistory] = useState([]);

    const handleStepChange = useCallback((iteration, weight, loss) => {
        setStepHistory(prev => {
            if (iteration === 0) {
                return [{ iteration, weight, loss }];
            }
            // Only add if it's a new iteration
            if (prev.length === 0 || prev[prev.length - 1].iteration !== iteration) {
                return [...prev, { iteration, weight, loss }];
            }
            return prev;
        });
    }, []);

    const handleParamsChange = useCallback((nextLearningRate, nextStartWeight) => {
        setLearningRate(nextLearningRate);
        setStartWeight(nextStartWeight);
        setStepHistory([]);
    }, []);

    const renderPanel = () => {
        switch (activeTab) {
            case 'descent':
                return (
                    <Suspense fallback={<LoadingPanel />}>
                        <GradientDescentPanel
                            learningRate={learningRate}
                            startWeight={startWeight}
                            onStepChange={handleStepChange}
                        />
                    </Suspense>
                );
            case 'history':
                return (
                    <Suspense fallback={<LoadingPanel />}>
                        <LossHistoryPanel history={stepHistory} />
                    </Suspense>
                );
            case 'practice':
                return (
                    <Suspense fallback={<LoadingPanel />}>
                        <PracticePanel
                            learningRate={learningRate}
                            startWeight={startWeight}
                            onParamsChange={handleParamsChange}
                        />
                    </Suspense>
                );
            default:
                return (
                    <Suspense fallback={<LoadingPanel />}>
                        <GradientDescentPanel
                            learningRate={learningRate}
                            startWeight={startWeight}
                            onStepChange={handleStepChange}
                        />
                    </Suspense>
                );
        }
    };

    return (
        <div className="flex flex-col h-full">
            {/* Navigation Tabs */}
            <nav className="bg-white/50 backdrop-blur-sm border-b border-slate-200 sticky top-0 z-10">
                <div className="px-4 overflow-x-auto">
                    <div className="flex space-x-1 py-2">
                        {tabs.map((tab) => (
                            <button
                                key={tab.id}
                                onClick={() => setActiveTab(tab.id)}
                                className={`flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium transition-all whitespace-nowrap ${
                                    activeTab === tab.id
                                        ? `bg-gradient-to-r ${tab.color} text-white shadow-lg scale-105`
                                        : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                                }`}
                            >
                                <tab.icon size={18} />
                                {tab.label}
                            </button>
                        ))}
                    </div>
                </div>
            </nav>

            {/* Panel Content */}
            <div className="flex-1 overflow-auto">
                {renderPanel()}
            </div>
        </div>
    );
}
