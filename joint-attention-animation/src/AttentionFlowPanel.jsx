import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw } from 'lucide-react';
import gsap from 'gsap';

export default function AttentionFlowPanel() {
  const [selectedSource, setSelectedSource] = useState(null);
  const [isAnimating, setIsAnimating] = useState(false);
  const svgRef = useRef(null);

  const imgTokens = 16; // 4x4 grid for visualization
  const txtTokens = 4;
  const totalTokens = imgTokens + txtTokens;

  // Token positions
  const getTokenPosition = (idx, total) => {
    const cols = Math.ceil(Math.sqrt(total));
    const row = Math.floor(idx / cols);
    const col = idx % cols;
    return {
      x: 60 + col * 50,
      y: 60 + row * 50,
    };
  };

  const handleTokenClick = (idx) => {
    setSelectedSource(selectedSource === idx ? null : idx);
  };

  useEffect(() => {
    if (selectedSource !== null && svgRef.current) {
      const lines = svgRef.current.querySelectorAll('.attention-line');
      gsap.fromTo(lines, 
        { strokeDashoffset: 100 },
        { strokeDashoffset: 0, duration: 0.5, stagger: 0.02 }
      );
    }
  }, [selectedSource]);

  const runFullAnimation = () => {
    setIsAnimating(true);
    let current = 0;
    const interval = setInterval(() => {
      setSelectedSource(current);
      current++;
      if (current >= totalTokens) {
        clearInterval(interval);
        setTimeout(() => {
          setSelectedSource(null);
          setIsAnimating(false);
        }, 1000);
      }
    }, 300);
  };

  return (
    <div className="space-y-8">
      {/* Title */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">
          Attention <span className="text-violet-400">Flow</span>
        </h2>
        <p className="text-gray-400">
          Visualize how each token attends to every other token
        </p>
      </div>

      {/* Main Visualization */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <div className="flex justify-center gap-4 mb-6">
          <button
            onClick={runFullAnimation}
            disabled={isAnimating}
            className="px-4 py-2 rounded-lg bg-violet-600 hover:bg-violet-500 disabled:opacity-50 flex items-center gap-2"
          >
            <Play size={18} />
            Animate All
          </button>
          <button
            onClick={() => setSelectedSource(null)}
            className="px-4 py-2 rounded-lg bg-white/10 hover:bg-white/20 flex items-center gap-2"
          >
            <RotateCcw size={18} />
            Clear
          </button>
        </div>

        <p className="text-center text-sm text-gray-400 mb-4">
          Click on any token to see what it attends to (all other tokens!)
        </p>

        {/* Interactive SVG */}
        <div className="flex justify-center">
          <svg ref={svgRef} width="400" height="350" className="bg-black/30 rounded-xl">
            {/* Grid labels */}
            <text x="130" y="25" fill="#60a5fa" className="text-xs">Image Tokens</text>
            <text x="300" y="25" fill="#fb923c" className="text-xs">Text</text>
            
            {/* Attention lines from selected source */}
            {selectedSource !== null && (
              <g>
                {[...Array(totalTokens)].map((_, targetIdx) => {
                  if (targetIdx === selectedSource) return null;
                  const source = selectedSource < imgTokens 
                    ? getTokenPosition(selectedSource, imgTokens)
                    : { x: 300, y: 80 + (selectedSource - imgTokens) * 50 };
                  const target = targetIdx < imgTokens
                    ? getTokenPosition(targetIdx, imgTokens)
                    : { x: 300, y: 80 + (targetIdx - imgTokens) * 50 };
                  
                  const isImgToTxt = selectedSource < imgTokens && targetIdx >= imgTokens;
                  const isTxtToImg = selectedSource >= imgTokens && targetIdx < imgTokens;
                  const strokeColor = isImgToTxt ? '#60a5fa' : isTxtToImg ? '#fb923c' : '#a855f7';
                  const opacity = Math.random() * 0.5 + 0.3; // Random attention weight
                  
                  return (
                    <line
                      key={targetIdx}
                      className="attention-line"
                      x1={source.x + 15}
                      y1={source.y + 15}
                      x2={target.x + 15}
                      y2={target.y + 15}
                      stroke={strokeColor}
                      strokeWidth={opacity * 3}
                      strokeOpacity={opacity}
                      strokeDasharray="100"
                      strokeDashoffset="0"
                    />
                  );
                })}
              </g>
            )}

            {/* Image tokens (4x4 grid) */}
            {[...Array(imgTokens)].map((_, i) => {
              const pos = getTokenPosition(i, imgTokens);
              const isSelected = selectedSource === i;
              return (
                <g key={`img-${i}`} onClick={() => handleTokenClick(i)} className="cursor-pointer">
                  <rect
                    x={pos.x}
                    y={pos.y}
                    width="30"
                    height="30"
                    rx="4"
                    fill={isSelected ? '#3b82f6' : '#1e40af'}
                    stroke={isSelected ? '#60a5fa' : 'transparent'}
                    strokeWidth="2"
                    className="transition-all hover:fill-blue-500"
                  />
                  <text
                    x={pos.x + 15}
                    y={pos.y + 19}
                    fill="white"
                    textAnchor="middle"
                    className="text-xs font-bold pointer-events-none"
                  >
                    {i}
                  </text>
                </g>
              );
            })}

            {/* Separator */}
            <line x1="260" y1="50" x2="260" y2="280" stroke="white" strokeOpacity="0.2" strokeDasharray="4" />

            {/* Text tokens */}
            {[...Array(txtTokens)].map((_, i) => {
              const isSelected = selectedSource === imgTokens + i;
              return (
                <g key={`txt-${i}`} onClick={() => handleTokenClick(imgTokens + i)} className="cursor-pointer">
                  <rect
                    x="280"
                    y={60 + i * 50}
                    width="50"
                    height="30"
                    rx="4"
                    fill={isSelected ? '#ea580c' : '#9a3412'}
                    stroke={isSelected ? '#fb923c' : 'transparent'}
                    strokeWidth="2"
                    className="transition-all hover:fill-orange-500"
                  />
                  <text
                    x="305"
                    y={79 + i * 50}
                    fill="white"
                    textAnchor="middle"
                    className="text-xs font-bold pointer-events-none"
                  >
                    T{i}
                  </text>
                </g>
              );
            })}

            {/* Legend */}
            <g transform="translate(20, 300)">
              <rect x="0" y="0" width="12" height="12" fill="#3b82f6" rx="2" />
              <text x="18" y="10" fill="#9ca3af" className="text-xs">Img→All</text>
              <rect x="80" y="0" width="12" height="12" fill="#f97316" rx="2" />
              <text x="98" y="10" fill="#9ca3af" className="text-xs">Txt→All</text>
              <rect x="160" y="0" width="12" height="12" fill="#a855f7" rx="2" />
              <text x="178" y="10" fill="#9ca3af" className="text-xs">Same type</text>
            </g>
          </svg>
        </div>

        {selectedSource !== null && (
          <div className="mt-4 text-center p-3 bg-violet-900/30 rounded-lg">
            <p className="text-violet-300">
              Token <span className="font-bold">{selectedSource < imgTokens ? selectedSource : `T${selectedSource - imgTokens}`}</span> attends to all {totalTokens - 1} other tokens
            </p>
          </div>
        )}
      </div>

      {/* Attention Matrix Visualization */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-xl font-bold mb-4">Attention Matrix View</h3>
        <p className="text-sm text-gray-400 mb-4">
          The full attention matrix for joint attention (simplified 4×4 img + 2 text)
        </p>
        
        <div className="overflow-x-auto">
          <table className="mx-auto text-xs">
            <thead>
              <tr>
                <th className="p-1"></th>
                {[...Array(6)].map((_, i) => (
                  <th key={i} className={`p-1 ${i < 4 ? 'text-blue-400' : 'text-orange-400'}`}>
                    {i < 4 ? `I${i}` : `T${i-4}`}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[...Array(6)].map((_, row) => (
                <tr key={row}>
                  <td className={`p-1 font-bold ${row < 4 ? 'text-blue-400' : 'text-orange-400'}`}>
                    {row < 4 ? `I${row}` : `T${row-4}`}
                  </td>
                  {[...Array(6)].map((_, col) => {
                    const val = Math.random() * 0.8 + 0.1;
                    const isImgToImg = row < 4 && col < 4;
                    const isTxtToTxt = row >= 4 && col >= 4;
                    const isCross = !isImgToImg && !isTxtToTxt;
                    let bg;
                    if (isImgToImg) bg = `rgba(59, 130, 246, ${val})`;
                    else if (isTxtToTxt) bg = `rgba(249, 115, 22, ${val})`;
                    else bg = `rgba(168, 85, 247, ${val})`;
                    
                    return (
                      <td
                        key={col}
                        className="w-10 h-10 text-center border border-white/10"
                        style={{ background: bg }}
                      >
                        {val.toFixed(1)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="mt-4 flex justify-center gap-6 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-blue-500/60" />
            <span className="text-gray-400">Image↔Image</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-orange-500/60" />
            <span className="text-gray-400">Text↔Text</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-violet-500/60" />
            <span className="text-gray-400">Cross-modal</span>
          </div>
        </div>
      </div>

      {/* Key Insight */}
      <div className="bg-gradient-to-r from-violet-900/30 to-fuchsia-900/30 rounded-xl p-6 border border-violet-500/30">
        <h3 className="font-bold text-violet-300 mb-2">💡 Key Insight</h3>
        <p className="text-gray-300">
          In joint attention, the attention matrix is <strong>fully dense</strong> - every token (both image and text) 
          can attend to every other token. This creates four types of attention:
        </p>
        <ul className="mt-3 text-sm text-gray-400 space-y-1">
          <li>• <span className="text-blue-400">Image→Image:</span> Spatial coherence, object consistency</li>
          <li>• <span className="text-orange-400">Text→Text:</span> Language understanding, context</li>
          <li>• <span className="text-violet-400">Image→Text:</span> Image regions query text for guidance</li>
          <li>• <span className="text-violet-400">Text→Image:</span> Text concepts attend to relevant image regions</li>
        </ul>
      </div>
    </div>
  );
}
