import React, { useEffect, useRef, useState } from 'react';
import { ZoomIn, ZoomOut, RotateCcw, Lightbulb } from 'lucide-react';

// Sample word embeddings (2D for visualization)
const WORDS = [
    { word: 'king', x: 0.8, y: 0.85, category: 'royalty' },
    { word: 'queen', x: 0.7, y: 0.8, category: 'royalty' },
    { word: 'prince', x: 0.75, y: 0.7, category: 'royalty' },
    { word: 'princess', x: 0.65, y: 0.65, category: 'royalty' },
    { word: 'man', x: 0.5, y: 0.6, category: 'person' },
    { word: 'woman', x: 0.4, y: 0.55, category: 'person' },
    { word: 'boy', x: 0.55, y: 0.45, category: 'person' },
    { word: 'girl', x: 0.45, y: 0.4, category: 'person' },
    { word: 'cat', x: -0.3, y: 0.5, category: 'animal' },
    { word: 'dog', x: -0.2, y: 0.55, category: 'animal' },
    { word: 'lion', x: -0.15, y: 0.65, category: 'animal' },
    { word: 'tiger', x: -0.1, y: 0.6, category: 'animal' },
    { word: 'car', x: 0.2, y: -0.5, category: 'vehicle' },
    { word: 'truck', x: 0.3, y: -0.45, category: 'vehicle' },
    { word: 'bus', x: 0.35, y: -0.55, category: 'vehicle' },
    { word: 'bicycle', x: 0.1, y: -0.4, category: 'vehicle' },
    { word: 'happy', x: -0.6, y: -0.2, category: 'emotion' },
    { word: 'sad', x: -0.65, y: -0.3, category: 'emotion' },
    { word: 'angry', x: -0.55, y: -0.35, category: 'emotion' },
    { word: 'joyful', x: -0.5, y: -0.15, category: 'emotion' },
    { word: 'apple', x: -0.7, y: 0.2, category: 'food' },
    { word: 'banana', x: -0.75, y: 0.15, category: 'food' },
    { word: 'orange', x: -0.65, y: 0.25, category: 'food' },
    { word: 'pizza', x: -0.5, y: 0.1, category: 'food' },
];

const CATEGORY_COLORS = {
    'royalty': { bg: '#c4b5fd', border: '#8b5cf6', fill: '#7c3aed' },
    'person': { bg: '#93c5fd', border: '#3b82f6', fill: '#2563eb' },
    'animal': { bg: '#86efac', border: '#22c55e', fill: '#16a34a' },
    'vehicle': { bg: '#fdba74', border: '#f97316', fill: '#ea580c' },
    'emotion': { bg: '#f9a8d4', border: '#ec4899', fill: '#db2777' },
    'food': { bg: '#fcd34d', border: '#f59e0b', fill: '#d97706' },
};

export default function VectorSpacePanel() {
    const canvasRef = useRef(null);
    const [zoom, setZoom] = useState(1);
    const [hoveredWord, setHoveredWord] = useState(null);
    const [selectedCategory, setSelectedCategory] = useState(null);
    const [showClusters, setShowClusters] = useState(true);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        const centerX = width / 2;
        const centerY = height / 2;
        const scale = Math.min(width, height) * 0.4 * zoom;

        // Clear canvas
        ctx.fillStyle = '#f8fafc';
        ctx.fillRect(0, 0, width, height);

        // Draw grid
        ctx.strokeStyle = '#e2e8f0';
        ctx.lineWidth = 1;
        for (let i = -1; i <= 1; i += 0.25) {
            // Vertical lines
            ctx.beginPath();
            ctx.moveTo(centerX + i * scale, 0);
            ctx.lineTo(centerX + i * scale, height);
            ctx.stroke();
            // Horizontal lines
            ctx.beginPath();
            ctx.moveTo(0, centerY - i * scale);
            ctx.lineTo(width, centerY - i * scale);
            ctx.stroke();
        }

        // Draw axes
        ctx.strokeStyle = '#94a3b8';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(0, centerY);
        ctx.lineTo(width, centerY);
        ctx.moveTo(centerX, 0);
        ctx.lineTo(centerX, height);
        ctx.stroke();

        // Draw axis labels
        ctx.fillStyle = '#64748b';
        ctx.font = '12px sans-serif';
        ctx.fillText('Dimension 1 →', width - 80, centerY - 10);
        ctx.save();
        ctx.translate(centerX + 10, 20);
        ctx.fillText('↑ Dimension 2', 0, 0);
        ctx.restore();

        // Draw cluster backgrounds if enabled
        if (showClusters) {
            const categories = [...new Set(WORDS.map(w => w.category))];
            categories.forEach(cat => {
                if (selectedCategory && selectedCategory !== cat) return;
                
                const catWords = WORDS.filter(w => w.category === cat);
                const color = CATEGORY_COLORS[cat];
                
                // Calculate cluster center and radius
                const avgX = catWords.reduce((s, w) => s + w.x, 0) / catWords.length;
                const avgY = catWords.reduce((s, w) => s + w.y, 0) / catWords.length;
                const maxDist = Math.max(...catWords.map(w => 
                    Math.sqrt((w.x - avgX) ** 2 + (w.y - avgY) ** 2)
                ));
                
                ctx.beginPath();
                ctx.arc(
                    centerX + avgX * scale,
                    centerY - avgY * scale,
                    (maxDist + 0.15) * scale,
                    0, Math.PI * 2
                );
                ctx.fillStyle = color.bg + '40';
                ctx.fill();
                ctx.strokeStyle = color.border + '60';
                ctx.lineWidth = 2;
                ctx.stroke();
            });
        }

        // Draw words
        WORDS.forEach(word => {
            const isSelected = selectedCategory ? word.category === selectedCategory : true;
            const isHovered = hoveredWord === word.word;
            const color = CATEGORY_COLORS[word.category];
            
            const x = centerX + word.x * scale;
            const y = centerY - word.y * scale;
            
            // Draw point
            ctx.beginPath();
            ctx.arc(x, y, isHovered ? 10 : 6, 0, Math.PI * 2);
            ctx.fillStyle = isSelected ? color.fill : '#cbd5e1';
            ctx.fill();
            ctx.strokeStyle = isHovered ? '#1e293b' : color.border;
            ctx.lineWidth = isHovered ? 3 : 2;
            ctx.stroke();
            
            // Draw label
            ctx.fillStyle = isSelected ? '#1e293b' : '#94a3b8';
            ctx.font = isHovered ? 'bold 14px sans-serif' : '12px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(word.word, x, y - 12);
        });

        // Draw origin
        ctx.fillStyle = '#ef4444';
        ctx.beginPath();
        ctx.arc(centerX, centerY, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = '#64748b';
        ctx.font = '10px sans-serif';
        ctx.fillText('(0,0)', centerX + 10, centerY + 15);

    }, [zoom, hoveredWord, selectedCategory, showClusters]);

    const handleCanvasMove = (e) => {
        const canvas = canvasRef.current;
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        const scale = Math.min(canvas.width, canvas.height) * 0.4 * zoom;
        
        // Find closest word
        let closest = null;
        let minDist = Infinity;
        
        WORDS.forEach(word => {
            const wx = centerX + word.x * scale;
            const wy = centerY - word.y * scale;
            const dist = Math.sqrt((x - wx) ** 2 + (y - wy) ** 2);
            if (dist < 20 && dist < minDist) {
                minDist = dist;
                closest = word.word;
            }
        });
        
        setHoveredWord(closest);
    };

    const categories = [...new Set(WORDS.map(w => w.category))];

    return (
        <div className="p-6 h-full">
            <div className="max-w-6xl mx-auto">
                <div className="text-center mb-4">
                    <h2 className="text-2xl font-bold text-indigo-900 mb-2">Vector Space Visualization</h2>
                    <p className="text-slate-600">
                        Words plotted in 2D space - similar words cluster together
                    </p>
                </div>

                <div className="flex gap-4">
                    {/* Controls */}
                    <div className="w-64 flex-shrink-0">
                        <div className="bg-slate-50 rounded-xl p-4 mb-4">
                            <h3 className="font-bold text-slate-800 mb-3">Controls</h3>
                            
                            {/* Zoom */}
                            <div className="flex items-center gap-2 mb-4">
                                <button
                                    onClick={() => setZoom(z => Math.max(0.5, z - 0.25))}
                                    className="p-2 rounded-lg bg-white border hover:bg-slate-100"
                                >
                                    <ZoomOut size={16} />
                                </button>
                                <span className="text-sm text-slate-600">{(zoom * 100).toFixed(0)}%</span>
                                <button
                                    onClick={() => setZoom(z => Math.min(2, z + 0.25))}
                                    className="p-2 rounded-lg bg-white border hover:bg-slate-100"
                                >
                                    <ZoomIn size={16} />
                                </button>
                                <button
                                    onClick={() => setZoom(1)}
                                    className="p-2 rounded-lg bg-white border hover:bg-slate-100"
                                >
                                    <RotateCcw size={16} />
                                </button>
                            </div>

                            {/* Show clusters toggle */}
                            <label className="flex items-center gap-2 mb-4 cursor-pointer">
                                <input
                                    type="checkbox"
                                    checked={showClusters}
                                    onChange={(e) => setShowClusters(e.target.checked)}
                                    className="rounded"
                                />
                                <span className="text-sm text-slate-700">Show cluster regions</span>
                            </label>

                            {/* Category Filter */}
                            <h4 className="text-sm font-medium text-slate-700 mb-2">Filter by Category</h4>
                            <div className="space-y-1">
                                <button
                                    onClick={() => setSelectedCategory(null)}
                                    className={`w-full text-left px-3 py-1.5 rounded text-sm ${
                                        !selectedCategory ? 'bg-indigo-100 text-indigo-800' : 'hover:bg-slate-100'
                                    }`}
                                >
                                    All Categories
                                </button>
                                {categories.map(cat => (
                                    <button
                                        key={cat}
                                        onClick={() => setSelectedCategory(cat)}
                                        className={`w-full text-left px-3 py-1.5 rounded text-sm flex items-center gap-2 ${
                                            selectedCategory === cat ? 'bg-indigo-100 text-indigo-800' : 'hover:bg-slate-100'
                                        }`}
                                    >
                                        <span 
                                            className="w-3 h-3 rounded-full" 
                                            style={{ backgroundColor: CATEGORY_COLORS[cat].fill }}
                                        />
                                        {cat}
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* Hovered Word Info */}
                        {hoveredWord && (
                            <div className="bg-white rounded-xl p-4 border shadow-lg">
                                <h4 className="font-bold text-slate-800 mb-2">{hoveredWord}</h4>
                                {(() => {
                                    const w = WORDS.find(w => w.word === hoveredWord);
                                    return w && (
                                        <>
                                            <div className="text-sm text-slate-600 mb-2">
                                                Category: <span className="font-medium">{w.category}</span>
                                            </div>
                                            <div className="font-mono text-xs bg-slate-100 p-2 rounded">
                                                [{w.x.toFixed(2)}, {w.y.toFixed(2)}]
                                            </div>
                                        </>
                                    );
                                })()}
                            </div>
                        )}
                    </div>

                    {/* Canvas */}
                    <div className="flex-1">
                        <canvas
                            ref={canvasRef}
                            width={700}
                            height={500}
                            onMouseMove={handleCanvasMove}
                            onMouseLeave={() => setHoveredWord(null)}
                            className="border rounded-xl bg-slate-50 cursor-crosshair w-full"
                            style={{ maxWidth: '700px' }}
                        />
                    </div>
                </div>

                {/* Key Observations */}
                <div className="mt-6 bg-amber-50 p-4 rounded-xl border border-amber-200">
                    <div className="flex items-start gap-3">
                        <Lightbulb className="text-amber-600 flex-shrink-0 mt-1" size={20} />
                        <div>
                            <h4 className="font-bold text-amber-900 mb-1">What to Notice</h4>
                            <ul className="text-amber-800 text-sm space-y-1">
                                <li>• <strong>Clustering:</strong> Words from the same category naturally group together</li>
                                <li>• <strong>Distance = Similarity:</strong> "King" and "Queen" are close; "King" and "Bicycle" are far</li>
                                <li>• <strong>Directions have meaning:</strong> The gender direction (man→woman) is consistent across pairs</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
