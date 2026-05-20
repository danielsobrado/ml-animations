import React from 'react';
import { Minus, Plus } from 'lucide-react';
import { classifySoftmaxSharpness, nudgeLogit } from '../../data/softmaxModel';

export default function PracticePanel({
    logits,
    probabilities,
    temperature,
    onLogitsChange,
    onTemperatureChange,
}) {
    const readout = classifySoftmaxSharpness(probabilities);

    const handleChange = (index, value) => {
        const newLogits = [...logits];
        newLogits[index] = parseFloat(value);
        onLogitsChange(newLogits);
    };

    const nudge = (index, delta) => {
        onLogitsChange(nudgeLogit(logits, index, delta));
    };

    return (
        <div className="ua-softmax-lab">
            <div className="ua-softmax-lab-head">
                <div>
                    <span>Temperature lab</span>
                    <h2>Live logits to probabilities</h2>
                </div>
                <div className={`ua-softmax-readout ${readout.tone}`}>
                    <strong>{readout.label}</strong>
                    <small>{readout.description}</small>
                </div>
            </div>

            <label className="ua-temperature-slider">
                <span>τ temperature</span>
                <input
                    type="range"
                    min="0.25"
                    max="3"
                    step="0.05"
                    value={temperature}
                    onChange={(event) => onTemperatureChange(parseFloat(event.target.value))}
                />
                <strong>{temperature.toFixed(2)}</strong>
            </label>

            <div className="ua-logit-controls">
                {logits.map((val, i) => (
                    <div key={i} className="ua-logit-row">
                        <label>z{i + 1}</label>
                        <button type="button" onClick={() => nudge(i, -0.25)} aria-label={`Decrease logit ${i + 1}`}>
                            <Minus size={14} />
                        </button>
                        <input
                            type="range"
                            min="-5"
                            max="5"
                            step="0.1"
                            value={val}
                            onChange={(e) => handleChange(i, e.target.value)}
                            className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                        />
                        <button type="button" onClick={() => nudge(i, 0.25)} aria-label={`Increase logit ${i + 1}`}>
                            <Plus size={14} />
                        </button>
                        <input
                            type="number"
                            value={val}
                            onChange={(e) => handleChange(i, e.target.value)}
                            step="0.1"
                        />
                        <output>{(probabilities[i] * 100).toFixed(1)}%</output>
                    </div>
                ))}
            </div>

            <div className="ua-softmax-note">
                <p>Adjust the logits to see how probabilities change.</p>
                <p className="mt-1">Notice that increasing one logit decreases the probabilities of others!</p>
            </div>
        </div>
    );
}
