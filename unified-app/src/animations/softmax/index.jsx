import React, { useState, Suspense, lazy } from 'react';
import { Play, LineChart, FlaskConical } from 'lucide-react';
import { classifySoftmaxSharpness, computeSoftmax, softmaxMetrics } from '../../data/softmaxModel';

// Lazy load panels
const SoftmaxAnimationPanel = lazy(() => import('./SoftmaxAnimationPanel'));
const SoftmaxGraphPanel = lazy(() => import('./SoftmaxGraphPanel'));
const PracticePanel = lazy(() => import('./PracticePanel'));

// Tab configuration
const tabs = [
    { id: 'animation', label: '1. Animation', icon: Play, color: 'from-blue-500 to-cyan-500' },
    { id: 'graph', label: '2. Softmax Graph', icon: LineChart, color: 'from-green-500 to-emerald-500' },
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

export default function SoftmaxAnimation() {
    const [activeTab, setActiveTab] = useState('animation');
    const [logits, setLogits] = useState([2.0, 1.0, 0.1]);
    const [temperature, setTemperature] = useState(1);
    const probabilities = computeSoftmax(logits, temperature);
    const metrics = softmaxMetrics(probabilities);
    const sharpness = classifySoftmaxSharpness(probabilities);

    const renderPanel = () => {
        switch (activeTab) {
            case 'animation':
                return <Suspense fallback={<LoadingPanel />}><SoftmaxAnimationPanel /></Suspense>;
            case 'graph':
                return (
                    <Suspense fallback={<LoadingPanel />}>
                        <SoftmaxGraphPanel logits={logits} probabilities={probabilities} isActive />
                    </Suspense>
                );
            case 'practice':
                return (
                    <Suspense fallback={<LoadingPanel />}>
                        <PracticePanel
                            logits={logits}
                            probabilities={probabilities}
                            temperature={temperature}
                            onLogitsChange={setLogits}
                            onTemperatureChange={setTemperature}
                        />
                    </Suspense>
                );
            default:
                return <Suspense fallback={<LoadingPanel />}><SoftmaxAnimationPanel /></Suspense>;
        }
    };

    return (
        <div className="ua-softmax-stage">
            <nav className="ua-segmented-tabs" aria-label="Softmax views">
                {tabs.map((tab) => (
                    <button
                        type="button"
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id)}
                        className={activeTab === tab.id ? 'active' : ''}
                    >
                        <tab.icon size={16} />
                        {tab.label}
                    </button>
                ))}
            </nav>

            <div className="ua-metrics-row" aria-label="Softmax metrics">
                <div>
                    <span>τ</span>
                    <strong>{temperature.toFixed(2)}</strong>
                    <small>temperature</small>
                </div>
                <div>
                    <span>max p</span>
                    <strong>{(metrics.maxProbability * 100).toFixed(1)}%</strong>
                    <small>{sharpness.label}</small>
                </div>
                <div>
                    <span>H</span>
                    <strong>{metrics.entropy.toFixed(2)}</strong>
                    <small>entropy bits</small>
                </div>
                <div>
                    <span>Δ</span>
                    <strong>{metrics.margin.toFixed(2)}</strong>
                    <small>margin</small>
                </div>
            </div>

            <div className="ua-softmax-panel">
                {renderPanel()}
            </div>
        </div>
    );
}
