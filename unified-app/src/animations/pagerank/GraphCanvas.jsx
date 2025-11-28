import React, { useRef, useState, useEffect } from 'react';
import * as d3 from 'd3-force';

export default function GraphCanvas({ nodes, setNodes, links, setLinks }) {
    const svgRef = useRef(null);
    const [selectedNode, setSelectedNode] = useState(null);
    const [dragLine, setDragLine] = useState(null);

    // Helper to find node at coordinates
    const findNode = (x, y) => {
        return nodes.find(n => Math.hypot(n.x - x, n.y - y) < 30);
    };

    const handleMouseDown = (e) => {
        const rect = svgRef.current.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const node = findNode(x, y);

        if (node) {
            if (e.shiftKey) {
                // Start creating a link
                setDragLine({ source: node, x, y });
            } else {
                // Select/Drag node (simplified for now)
                setSelectedNode(node);
            }
        } else {
            // Add new node
            const id = String.fromCharCode(65 + nodes.length); // A, B, C...
            if (nodes.length < 26) {
                setNodes([...nodes, { id, x, y }]);
            }
        }
    };

    const handleMouseMove = (e) => {
        const rect = svgRef.current.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        if (dragLine) {
            setDragLine({ ...dragLine, x, y });
        } else if (selectedNode) {
            // Move node
            setNodes(nodes.map(n => n.id === selectedNode.id ? { ...n, x, y } : n));
        }
    };

    const handleMouseUp = (e) => {
        const rect = svgRef.current.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const targetNode = findNode(x, y);

        if (dragLine && targetNode && targetNode.id !== dragLine.source.id) {
            // Create link if unique
            if (!links.some(l => l.source === dragLine.source.id && l.target === targetNode.id)) {
                setLinks([...links, { source: dragLine.source.id, target: targetNode.id }]);
            }
        }

        setDragLine(null);
        setSelectedNode(null);
    };

    return (
        <div className="p-8 h-full flex flex-col items-center">
            <div className="max-w-3xl w-full text-center mb-8">
                <h2 className="text-3xl font-bold text-indigo-900 mb-4">Graph Builder</h2>
                <p className="text-lg text-slate-700 leading-relaxed">
                    Build your mini-internet!
                    <br />
                    <span className="text-sm bg-slate-100 px-2 py-1 rounded border border-slate-300 mx-1">Click Empty Space</span> to Add Node.
                    <span className="text-sm bg-slate-100 px-2 py-1 rounded border border-slate-300 mx-1">Shift + Drag</span> from Node to Link.
                </p>
            </div>

            <div className="w-full max-w-4xl h-[500px] bg-slate-50 rounded-xl border-2 border-slate-200 shadow-inner relative overflow-hidden">
                <svg
                    ref={svgRef}
                    className="w-full h-full cursor-crosshair"
                    onMouseDown={handleMouseDown}
                    onMouseMove={handleMouseMove}
                    onMouseUp={handleMouseUp}
                >
                    <defs>
                        <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="28" refY="3.5" orient="auto">
                            <polygon points="0 0, 10 3.5, 0 7" fill="#94a3b8" />
                        </marker>
                    </defs>

                    {/* Links */}
                    {links.map((link, i) => {
                        const source = nodes.find(n => n.id === link.source);
                        const target = nodes.find(n => n.id === link.target);
                        if (!source || !target) return null;
                        return (
                            <line
                                key={i}
                                x1={source.x} y1={source.y}
                                x2={target.x} y2={target.y}
                                stroke="#94a3b8" strokeWidth="2"
                                markerEnd="url(#arrowhead)"
                            />
                        );
                    })}

                    {/* Drag Line */}
                    {dragLine && (
                        <line
                            x1={dragLine.source.x} y1={dragLine.source.y}
                            x2={dragLine.x} y2={dragLine.y}
                            stroke="#cbd5e1" strokeWidth="2" strokeDasharray="5 5"
                        />
                    )}

                    {/* Nodes */}
                    {nodes.map(node => (
                        <g key={node.id} transform={`translate(${node.x},${node.y})`}>
                            <circle r="20" fill="white" stroke="#4f46e5" strokeWidth="3" className="cursor-move hover:fill-indigo-50" />
                            <text dy="5" textAnchor="middle" className="font-bold text-indigo-900 pointer-events-none select-none">{node.id}</text>
                        </g>
                    ))}
                </svg>

                <button
                    onClick={() => { setNodes([]); setLinks([]); }}
                    className="absolute top-4 right-4 px-4 py-2 bg-white text-red-600 border border-red-200 rounded-lg shadow-sm hover:bg-red-50 font-bold text-sm"
                >
                    Clear All
                </button>
            </div>
        </div>
    );
}
