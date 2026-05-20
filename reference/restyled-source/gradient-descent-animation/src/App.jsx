import React, { useState, useCallback } from 'react';
import GradientDescentPanel from './GradientDescentPanel';
import LossHistoryPanel from './LossHistoryPanel';
import PracticePanel from './PracticePanel';
import { Page, Header, EquationStrip, Eq, Readouts } from './_design-system/ui';

export default function App() {
    const [learningRate, setLearningRate] = useState(0.25);
    const [startWeight, setStartWeight] = useState(3.0);
    const [history, setHistory] = useState([]);
    const [current, setCurrent] = useState({ iteration: 0, weight: 3.0 });

    const handleParamsChange = useCallback((lr, sw) => {
        setLearningRate(lr);
        setStartWeight(sw);
        setHistory([]);
        setCurrent({ iteration: 0, weight: sw });
    }, []);

    const handleStepChange = useCallback((iteration, weight) => {
        setCurrent({ iteration, weight });
        setHistory((prev) => {
            const next = [...prev];
            next[iteration] = { iteration, weight, loss: weight * weight };
            return next;
        });
    }, []);

    return (
        <Page>
            <Header
                eyebrow={['Chapter 04', 'Optimization', '§ 4.1']}
                right={<span><Eq tex="\alpha" /> = {learningRate.toFixed(2)} · {history.length} iterations</span>}
                title="Gradient descent on a convex loss"
                subtitle={
                    <>
                        A scalar walk-through of the simplest optimizer there is — how a single learning rate <Eq tex="\alpha" />{' '}
                        bends a parameter <Eq tex="w" /> toward a minimum, and what changes when we set it too small or too large.
                    </>
                }
            />

            <EquationStrip
                label="Update rule"
                tex="w_{t+1} \;=\; w_t \;-\; \alpha \, \nabla_{\!w}\, \mathcal{L}(w_t)"
                meta={<>with <Eq tex="\mathcal{L}(w) = w^2" /></>}
            />

            <div className="page-body">
                <div className="main-col">
                    <GradientDescentPanel
                        learningRate={learningRate}
                        startWeight={startWeight}
                        history={history}
                        current={current}
                        onStepChange={handleStepChange}
                    />
                    <div className="readouts-and-history">
                        <ReadoutsBlock current={current} />
                        <LossHistoryPanel history={history} current={current} />
                    </div>
                </div>
                <aside className="side-col">
                    <PracticePanel
                        learningRate={learningRate}
                        startWeight={startWeight}
                        onParamsChange={handleParamsChange}
                    />
                </aside>
            </div>
        </Page>
    );
}

function ReadoutsBlock({ current }) {
    const w = current.weight;
    const grad = 2 * w;
    const loss = w * w;
    const rows = [
        { label: 'iteration', tex: 't', value: String(current.iteration) },
        { label: 'weight', tex: 'w_t', value: w.toFixed(3) },
        { label: 'gradient', tex: '\\nabla\\mathcal{L}', value: grad.toFixed(3) },
        { label: 'loss', tex: '\\mathcal{L}(w_t)', value: loss.toFixed(3) },
    ];
    // Inline import to keep this file self-contained at the bottom
    return <Readouts rows={rows} />;
}
