import React from 'react';
import {
  ArrowDown,
  ArrowUp,
  CheckCircle,
  Minus,
  Play,
  RotateCcw,
  XCircle,
} from 'lucide-react';
import { GRPO_CANDIDATES, DEFAULT_GRPO_WEIGHTS } from './grpoLabData';
import { GRPO_STARTER_CODE } from './grpoStarterCode';
import { buildGrpoCheckScript } from './grpoChecker';
import { runPythonInBrowser } from '../runtime/pyodideClient';

const WEIGHT_CONTROLS = [
  {
    key: 'correctnessWeight',
    label: 'Correctness reward',
    min: 0,
    max: 2,
    step: 0.1,
  },
  {
    key: 'formatWeight',
    label: 'Format reward',
    min: 0,
    max: 0.5,
    step: 0.05,
  },
  {
    key: 'languagePenalty',
    label: 'Language penalty',
    min: 0,
    max: 0.5,
    step: 0.05,
  },
  {
    key: 'lengthPenalty',
    label: 'Length penalty',
    min: 0,
    max: 0.05,
    step: 0.005,
  },
];

function parseLabOutput(stdout) {
  const lines = stdout.trim().split('\n').filter(Boolean);
  const last = lines[lines.length - 1];
  return JSON.parse(last);
}

function formatNumber(value) {
  return Number.isFinite(value) ? value.toFixed(3) : '-';
}

function directionMeta(direction) {
  if (direction === 'up') {
    return {
      label: 'Increase',
      className: 'ua-grpo-direction up',
      Icon: ArrowUp,
    };
  }

  if (direction === 'down') {
    return {
      label: 'Decrease',
      className: 'ua-grpo-direction down',
      Icon: ArrowDown,
    };
  }

  return {
    label: 'Neutral',
    className: 'ua-grpo-direction neutral',
    Icon: Minus,
  };
}

function getCandidateResult(result, index) {
  if (!result) {
    return {
      reward: null,
      advantage: null,
      direction: null,
    };
  }

  return {
    reward: result.rewards[index],
    advantage: result.advantages[index],
    direction: result.directions[index],
  };
}

export default function GrpoAdvantageLab() {
  const [code, setCode] = React.useState(GRPO_STARTER_CODE);
  const [weights, setWeights] = React.useState(DEFAULT_GRPO_WEIGHTS);
  const [running, setRunning] = React.useState(false);
  const [runtimeHasLoaded, setRuntimeHasLoaded] = React.useState(false);
  const [result, setResult] = React.useState(null);
  const [error, setError] = React.useState(null);

  function updateWeight(key, value) {
    setWeights((current) => ({
      ...current,
      [key]: Number(value),
    }));
  }

  function resetLab() {
    setCode(GRPO_STARTER_CODE);
    setWeights(DEFAULT_GRPO_WEIGHTS);
    setResult(null);
    setError(null);
  }

  async function runLab() {
    setRunning(true);
    setError(null);
    setResult(null);

    const script = buildGrpoCheckScript(code, GRPO_CANDIDATES, weights);
    const response = await runPythonInBrowser(script);

    setRunning(false);
    setRuntimeHasLoaded(true);

    if (response.error) {
      setError(response.stderr ? `${response.error}\n\n${response.stderr}` : response.error);
      setResult(null);
      return;
    }

    try {
      setResult(parseLabOutput(response.stdout));
    } catch (err) {
      setError(`Could not parse lab output: ${err.message}`);
      setResult(null);
    }
  }

  return (
    <section className="ua-grpo-lab-panel">
      <div className="ua-grpo-lab-head">
        <span>Interactive Lab</span>
        <div>
          <h2>Compute GRPO group advantages</h2>
          <p>
            GRPO compares multiple sampled answers for the same prompt. A trace does not need an
            absolute perfect score to be reinforced; it needs to be better than the group baseline.
            In this lab, you will compute reward, normalize rewards inside the group, and watch
            which traces receive positive or negative update pressure.
          </p>
        </div>
      </div>

      <div className="ua-grpo-lab-note">
        <strong>Teaching abstraction:</strong>
        <span>
          This lab uses a simplified reward and advantage calculation. Production GRPO also includes
          policy ratios, clipping, KL control, batching details, and optimizer dynamics.
        </span>
      </div>

      <div className="ua-grpo-lab-controls" aria-label="Reward weights">
        {WEIGHT_CONTROLS.map((control) => (
          <label key={control.key} className="ua-grpo-slider">
            <span>
              {control.label}
              <strong>{weights[control.key].toFixed(control.step < 0.01 ? 3 : 2)}</strong>
            </span>
            <input
              type="range"
              min={control.min}
              max={control.max}
              step={control.step}
              value={weights[control.key]}
              onChange={(event) => updateWeight(control.key, event.target.value)}
              className="ds-range"
            />
          </label>
        ))}
      </div>

      <div className="ua-grpo-lab-grid">
        <div className="ua-grpo-lab-card ua-grpo-candidates">
          <div className="ua-grpo-card-head">
            <h3>Candidate solution group</h3>
            <span>Solve: 3x + 2 = 11</span>
          </div>

          <div className="ua-grpo-table-wrap">
            <table className="ua-grpo-table">
              <thead>
                <tr>
                  <th>Candidate</th>
                  <th>Trace</th>
                  <th>Correct</th>
                  <th>Format</th>
                  <th>Tokens</th>
                  <th>Mixed</th>
                  <th>Reward</th>
                  <th>Advantage</th>
                  <th>Update</th>
                </tr>
              </thead>
              <tbody>
                {GRPO_CANDIDATES.map((candidate, index) => {
                  const rowResult = getCandidateResult(result, index);
                  const meta = directionMeta(rowResult.direction);
                  const DirectionIcon = meta.Icon;

                  return (
                    <tr key={candidate.id}>
                      <td className="ua-grpo-id">{candidate.id}</td>
                      <td>
                        <strong>{candidate.label}</strong>
                        <small>{candidate.summary}</small>
                      </td>
                      <td>{candidate.isCorrect ? 'Yes' : 'No'}</td>
                      <td>{candidate.hasValidFormat ? 'Yes' : 'No'}</td>
                      <td>{candidate.tokenCount}</td>
                      <td>{candidate.languageMixed ? 'Yes' : 'No'}</td>
                      <td>{rowResult.reward === null ? '-' : formatNumber(rowResult.reward)}</td>
                      <td>{rowResult.advantage === null ? '-' : formatNumber(rowResult.advantage)}</td>
                      <td>
                        {rowResult.direction ? (
                          <span className={meta.className}>
                            <DirectionIcon aria-hidden="true" />
                            {meta.label}
                          </span>
                        ) : (
                          '-'
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        <div className="ua-grpo-lab-card ua-grpo-python-card">
          <div className="ua-grpo-card-head">
            <h3>Python editor</h3>
            <span>Implement reward, normalization, and update direction</span>
          </div>

          <textarea
            value={code}
            onChange={(event) => setCode(event.target.value)}
            rows={24}
            spellCheck={false}
            className="ua-grpo-code-editor"
            aria-label="GRPO Python lab code"
          />

          <div className="ua-grpo-actions">
            <button
              type="button"
              data-math-control
              onClick={runLab}
              disabled={running}
              className="ds-btn ua-grpo-run-button"
            >
              <Play aria-hidden="true" />
              {running && !runtimeHasLoaded ? 'Loading Python runtime...' : running ? 'Running checks...' : 'Run checks'}
            </button>
            <button
              type="button"
              data-math-control
              onClick={resetLab}
              disabled={running}
              className="ds-btn ua-grpo-reset-button"
            >
              <RotateCcw aria-hidden="true" />
              Reset starter
            </button>
          </div>
        </div>

        <div className="ua-grpo-lab-card ua-grpo-feedback-card">
          <div className="ua-grpo-card-head">
            <h3>Checks and update pressure</h3>
            <span>Positive advantage reinforces traces above the group baseline</span>
          </div>

          {error && <pre className="ua-grpo-lab-error">{error}</pre>}

          {!result && !error && (
            <div className="ua-grpo-empty-state">
              <p>
                {running
                  ? 'Python is checking your reward, advantage, and update functions.'
                  : 'Run the starter code to see failing checks, then complete each TODO.'}
              </p>
            </div>
          )}

          {result && (
            <div className="ua-grpo-results">
              <div className={`ua-grpo-status ${result.passed ? 'passed' : 'pending'}`}>
                {result.passed ? (
                  <CheckCircle aria-hidden="true" />
                ) : (
                  <XCircle aria-hidden="true" />
                )}
                <strong>{result.passed ? 'All checks passed' : 'Not passed yet'}</strong>
              </div>

              <ul className="ua-grpo-check-list">
                {result.checks.map((check) => (
                  <li key={check.name} className={check.passed ? 'passed' : 'failed'}>
                    {check.passed ? (
                      <CheckCircle aria-hidden="true" />
                    ) : (
                      <XCircle aria-hidden="true" />
                    )}
                    {check.name}
                  </li>
                ))}
              </ul>

              <div className="ua-grpo-advantage-stack">
                {GRPO_CANDIDATES.map((candidate, index) => {
                  const advantage = result.advantages[index];
                  const direction = result.directions[index];
                  const meta = directionMeta(direction);
                  const DirectionIcon = meta.Icon;
                  const barWidth = `${Math.min(Math.abs(advantage) / 2, 1) * 100}%`;

                  return (
                    <div key={candidate.id} className="ua-grpo-advantage-row">
                      <span className="ua-grpo-advantage-label">{candidate.id}</span>
                      <div className="ua-grpo-advantage-track">
                        <span
                          className={`ua-grpo-advantage-bar ${direction}`}
                          style={{ width: barWidth }}
                        />
                      </div>
                      <span className={meta.className}>
                        <DirectionIcon aria-hidden="true" />
                        {formatNumber(advantage)}
                      </span>
                    </div>
                  );
                })}
              </div>

              {result.passed && (
                <p className="ua-grpo-completion-copy">
                  Notice that reward design controls what the model learns. If format reward is too
                  strong, a polished wrong trace can look better than it should. If length penalty
                  is too strong, the model may underthink hard problems. GRPO optimizes the reward
                  surface you provide, not your unstated intention.
                </p>
              )}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
