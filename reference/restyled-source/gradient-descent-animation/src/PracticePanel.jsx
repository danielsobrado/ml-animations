import React from 'react';
import { Aside, ParamSlider, Eq } from './_design-system/ui';

export default function PracticePanel({ learningRate, startWeight, onParamsChange }) {
    const handleLR = (v) => onParamsChange(v, startWeight);
    const handleW0 = (v) => onParamsChange(learningRate, v);

    const lrHint = learningRate < 0.05
        ? { text: 'α < 0.05: convergence is slow', tone: 'warn' }
        : learningRate >= 1.0
            ? { text: 'α ≥ 1: iterates will diverge', tone: 'bad' }
            : learningRate > 0.9
                ? { text: 'α > 0.9: expect oscillation', tone: 'warn' }
                : { text: 'within stable region (α < 1)', tone: 'ok' };

    return (
        <Aside heading="Controls">
            <ParamSlider
                label="Learning rate"
                tex="\alpha"
                value={learningRate}
                min={0.01} max={1.05} step={0.01}
                onChange={handleLR}
                format={(v) => v.toFixed(2)}
                hint={lrHint.text}
                hintTone={lrHint.tone}
            />
            <ParamSlider
                label="Initial weight"
                tex="w_0"
                value={startWeight}
                min={-5} max={5} step={0.1}
                onChange={handleW0}
                format={(v) => (v >= 0 ? '+' : '') + v.toFixed(2)}
                hint="symmetric about the origin"
                hintTone="ok"
            />

            <p>
                Try <Eq tex="\alpha = 0.01" /> — convergence is monotone but slow. At <Eq tex="\alpha = 0.95" />{' '}
                the iterates oscillate; past <Eq tex="\alpha = 1" /> they diverge.
            </p>
            <p>
                The update <Eq tex="w_{t+1} = (1 - 2\alpha)\, w_t" /> contracts whenever{' '}
                <strong>|1 − 2α| &lt; 1</strong>, i.e. <Eq tex="0 < \alpha < 1" />.
            </p>
        </Aside>
    );
}
