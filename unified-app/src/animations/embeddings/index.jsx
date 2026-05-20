import React, { useState, Suspense, lazy } from 'react';
import { Tabs } from '../../_design-system/ui';

// Lazy load panels
const AlgebraPanel = lazy(() => import('./AlgebraPanel'));
const SimilarityPanel = lazy(() => import('./SimilarityPanel'));
const SpacePanel = lazy(() => import('./SpacePanel'));

// Tab configuration
const tabs = [
    { id: 'algebra', label: 'Word Algebra' },
    { id: 'similarity', label: 'Similarity Lab' },
    { id: 'space', label: '3D Semantic Space' },
];

// Loading fallback
function LoadingPanel() {
    return (
        <div className="flex items-center justify-center p-12">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-cyan-500"></div>
        </div>
    );
}

export default function EmbeddingsAnimation() {
    const [activeTab, setActiveTab] = useState('algebra');

    const renderPanel = () => {
        switch (activeTab) {
            case 'algebra':
                return <Suspense fallback={<LoadingPanel />}><AlgebraPanel /></Suspense>;
            case 'similarity':
                return <Suspense fallback={<LoadingPanel />}><SimilarityPanel /></Suspense>;
            case 'space':
                return <Suspense fallback={<LoadingPanel />}><SpacePanel /></Suspense>;
            default:
                return <Suspense fallback={<LoadingPanel />}><AlgebraPanel /></Suspense>;
        }
    };

    return (
        <div className="flex flex-col h-full">
            <Tabs tabs={tabs} active={activeTab} onChange={setActiveTab} />

            {/* Panel Content */}
            <div className="flex-1 overflow-auto">
                {renderPanel()}
            </div>
        </div>
    );
}
