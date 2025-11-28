import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import gsap from 'gsap';

const STEPS = [
    { id: 'inputs', title: '1. Inputs Arrive', desc: 'Previous Hidden State (h_t-1) and New Input (x_t) merge.' },
    { id: 'forget', title: '2. Forget Gate', desc: 'Sigmoid layer decides what to keep from old Cell State.' },
    { id: 'input_gate', title: '3. Input Gate', desc: 'Sigmoid layer decides which values to update.' },
    { id: 'candidates', title: '4. Candidate Values', desc: 'Tanh layer creates new candidate values.' },
    { id: 'prep_update', title: '5. Prepare Update', desc: 'Multiply Input Gate * Candidates.' },
    { id: 'cell_update', title: '6. Update Cell State', desc: 'Old State * Forget + New Values. The Conveyor Belt moves!' },
    { id: 'output_gate', title: '7. Output Gate', desc: 'Sigmoid layer decides what to output.' },
    { id: 'hidden_update', title: '8. Hidden State', desc: 'Tanh(Cell State) * Output Gate. The final result.' }
];

export default function FlowPanel() {
    const containerRef = useRef(null);
    const [step, setStep] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const sceneRef = useRef(null);
    const objectsRef = useRef({});

    useEffect(() => {
        if (!containerRef.current) return;

        const width = containerRef.current.clientWidth;
        const height = 400;

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xf8fafc); // Slate-50
        sceneRef.current = scene;

        const camera = new THREE.OrthographicCamera(
            width / -2, width / 2, height / 2, height / -2, 0.1, 1000
        );
        camera.position.z = 100;

        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(width, height);
        containerRef.current.appendChild(renderer.domElement);

        // Helper to create blocks
        const createBlock = (color, x, y, label) => {
            const group = new THREE.Group();
            const geometry = new THREE.BoxGeometry(40, 40, 10);
            const material = new THREE.MeshBasicMaterial({ color });
            const mesh = new THREE.Mesh(geometry, material);
            group.add(mesh);

            // Label
            const canvas = document.createElement('canvas');
            canvas.width = 128; canvas.height = 64;
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = 'white';
            ctx.font = 'bold 40px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(label, 64, 32);
            const texture = new THREE.CanvasTexture(canvas);
            const spriteMat = new THREE.SpriteMaterial({ map: texture });
            const sprite = new THREE.Sprite(spriteMat);
            sprite.scale.set(40, 20, 1);
            sprite.position.z = 6;
            group.add(sprite);

            group.position.set(x, y, 0);
            scene.add(group);
            return group;
        };

        // Create static structure (pipes/gates)
        // ... (Simplified for brevity, would be more complex in full implementation)

        // Dynamic Objects
        objectsRef.current.h_prev = createBlock(0x3b82f6, -200, -100, 'h-1'); // Blue
        objectsRef.current.x_t = createBlock(0x22c55e, -200, -150, 'x_t'); // Green
        objectsRef.current.c_prev = createBlock(0xeab308, -250, 100, 'C-1'); // Gold

        // Gates (initially hidden)
        objectsRef.current.forget = createBlock(0xef4444, -100, 0, 'f_t'); // Red
        objectsRef.current.input = createBlock(0x22c55e, 0, 0, 'i_t'); // Green
        objectsRef.current.cand = createBlock(0xf97316, 50, -50, 'C~'); // Orange
        objectsRef.current.output = createBlock(0x3b82f6, 150, 0, 'o_t'); // Blue

        const animate = () => {
            requestAnimationFrame(animate);
            renderer.render(scene, camera);
        };
        animate();

        return () => {
            renderer.dispose();
            if (containerRef.current?.contains(renderer.domElement)) {
                containerRef.current.removeChild(renderer.domElement);
            }
        };
    }, []);

    const animateStep = (stepIndex) => {
        const objs = objectsRef.current;
        // Reset positions for demo
        if (stepIndex === 0) {
            gsap.set(objs.h_prev.position, { x: -200, y: -100 });
            gsap.set(objs.x_t.position, { x: -200, y: -150 });
        }

        switch (stepIndex) {
            case 1: // Inputs Arrive
                gsap.to(objs.h_prev.position, { x: -150, duration: 1 });
                gsap.to(objs.x_t.position, { x: -150, duration: 1 });
                break;
            case 2: // Forget Gate
                gsap.to(objs.forget.scale, { x: 1.2, y: 1.2, duration: 0.5, yoyo: true, repeat: 1 });
                break;
            // ... Add animations for other steps
        }
    };

    const nextStep = () => {
        if (step < STEPS.length) {
            setStep(step + 1);
            animateStep(step + 1);
        }
    };

    return (
        <div className="p-4 h-full flex flex-col">
            <div className="flex justify-between items-center mb-4">
                <h2 className="text-2xl font-bold text-slate-800">Bit-by-Bit Animation</h2>
                <div className="flex gap-2">
                    <button
                        onClick={() => setStep(0)}
                        className="px-4 py-2 bg-slate-200 rounded hover:bg-slate-300 font-bold"
                    >
                        Reset
                    </button>
                    <button
                        onClick={nextStep}
                        className="px-4 py-2 bg-indigo-600 text-white rounded hover:bg-indigo-700 font-bold"
                    >
                        Next Step ({step}/{STEPS.length})
                    </button>
                </div>
            </div>

            <div ref={containerRef} className="flex-1 bg-slate-100 rounded-xl border border-slate-300 relative overflow-hidden" />

            <div className="mt-4 p-4 bg-white rounded-xl border border-slate-200 shadow-sm min-h-[100px]">
                <h3 className="font-bold text-lg text-indigo-900">{STEPS[step > 0 ? step - 1 : 0].title}</h3>
                <p className="text-slate-600">{STEPS[step > 0 ? step - 1 : 0].desc}</p>
            </div>
        </div>
    );
}
