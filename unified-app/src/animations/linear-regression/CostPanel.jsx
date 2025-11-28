import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

export default function CostPanel() {
    const containerRef = useRef(null);
    const [slope, setSlope] = useState(1);
    const [intercept, setIntercept] = useState(0);

    // Fixed dataset for cost calculation
    const data = [
        { x: 1, y: 2 },
        { x: 2, y: 3 },
        { x: 3, y: 5 },
        { x: 4, y: 4 },
        { x: 5, y: 6 }
    ];

    // Calculate MSE for a given m, b
    const calculateMSE = (m, b) => {
        let error = 0;
        data.forEach(p => {
            const pred = m * p.x + b;
            error += Math.pow(p.y - pred, 2);
        });
        return error / data.length;
    };

    const currentMSE = calculateMSE(slope, intercept);

    useEffect(() => {
        if (!containerRef.current) return;

        const width = containerRef.current.clientWidth;
        const height = 500;

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xf8fafc);

        const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
        camera.position.set(5, 5, 5);
        camera.lookAt(0, 0, 0);

        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(width, height);
        containerRef.current.appendChild(renderer.domElement);

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;

        // Lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(5, 10, 5);
        scene.add(dirLight);

        // Cost Surface Mesh
        const geometry = new THREE.PlaneGeometry(6, 6, 50, 50);
        const positions = geometry.attributes.position;

        // Map x -> slope (-1 to 3), y -> intercept (-2 to 4)
        // We need to scale the plane coordinates to parameter space
        for (let i = 0; i < positions.count; i++) {
            const x = positions.getX(i); // -3 to 3
            const y = positions.getY(i); // -3 to 3

            const m = x / 1.5 + 1; // Map to roughly -1 to 3
            const b = y / 1.5 + 1; // Map to roughly -1 to 3

            const z = calculateMSE(m, b) / 5; // Scale down height
            positions.setZ(i, z);
        }
        geometry.computeVertexNormals();
        geometry.rotateX(-Math.PI / 2); // Rotate to horizontal

        const material = new THREE.MeshStandardMaterial({
            color: 0x6366f1,
            roughness: 0.4,
            metalness: 0.2,
            side: THREE.DoubleSide,
            wireframe: false
        });
        const surface = new THREE.Mesh(geometry, material);
        scene.add(surface);

        // Wireframe overlay
        const wireframeMat = new THREE.MeshBasicMaterial({ color: 0x4338ca, wireframe: true, transparent: true, opacity: 0.2 });
        const wireframe = new THREE.Mesh(geometry, wireframeMat);
        scene.add(wireframe);

        // Current Position Marker
        const markerGeo = new THREE.SphereGeometry(0.2, 32, 32);
        const markerMat = new THREE.MeshStandardMaterial({ color: 0xef4444, emissive: 0x991b1b });
        const marker = new THREE.Mesh(markerGeo, markerMat);
        scene.add(marker);

        // Animation Loop
        const animate = () => {
            requestAnimationFrame(animate);
            controls.update();

            // Update marker position based on state
            // Inverse mapping of the geometry transformation above
            const x = (slope - 1) * 1.5;
            const z = (intercept - 1) * 1.5; // Y in plane geometry became Z after rotation
            const y = calculateMSE(slope, intercept) / 5;

            marker.position.set(x, y, -z); // Note: Z axis flip in Three.js coordinate system relative to plane logic often needs checking

            renderer.render(scene, camera);
        };
        animate();

        return () => {
            renderer.dispose();
            if (containerRef.current?.contains(renderer.domElement)) {
                containerRef.current.removeChild(renderer.domElement);
            }
        };
    }, [slope, intercept]); // Re-run if slope/intercept changes? No, passed via closure/ref usually better but this works for simple demo

    return (
        <div className="p-8 h-full flex flex-col items-center">
            <div className="max-w-3xl w-full text-center mb-8">
                <h2 className="text-3xl font-bold text-indigo-900 mb-4">The Cost Landscape</h2>
                <p className="text-lg text-slate-700 leading-relaxed">
                    Every combination of Slope and Intercept has a "Cost" (MSE).
                    <br />
                    Finding the best line means finding the <strong>lowest point</strong> in this bowl.
                </p>
            </div>

            <div className="flex flex-col md:flex-row gap-8 w-full max-w-6xl h-[500px]">
                {/* Controls */}
                <div className="w-full md:w-80 flex flex-col gap-6 bg-white p-6 rounded-xl shadow-lg border border-slate-200 z-10">
                    <h3 className="font-bold text-slate-500 uppercase text-xs mb-4">Navigate the Landscape</h3>

                    <div className="mb-6">
                        <label className="flex justify-between text-sm font-bold text-slate-700 mb-2">
                            Slope (m): <span className="text-indigo-600">{slope.toFixed(2)}</span>
                        </label>
                        <input
                            type="range" min="-1" max="3" step="0.1"
                            value={slope}
                            onChange={(e) => setSlope(Number(e.target.value))}
                            className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
                        />
                    </div>

                    <div className="mb-6">
                        <label className="flex justify-between text-sm font-bold text-slate-700 mb-2">
                            Intercept (b): <span className="text-indigo-600">{intercept.toFixed(2)}</span>
                        </label>
                        <input
                            type="range" min="-1" max="3" step="0.1"
                            value={intercept}
                            onChange={(e) => setIntercept(Number(e.target.value))}
                            className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
                        />
                    </div>

                    <div className="mt-auto">
                        <p className="text-sm text-slate-500 mb-1">Current Cost (MSE)</p>
                        <p className="text-3xl font-mono font-bold text-slate-800">{currentMSE.toFixed(3)}</p>
                    </div>
                </div>

                {/* 3D View */}
                <div ref={containerRef} className="flex-1 rounded-xl overflow-hidden shadow-inner border border-slate-300 relative">
                    <div className="absolute top-4 left-4 bg-white/80 p-2 rounded text-xs font-mono pointer-events-none">
                        3D Cost Surface
                    </div>
                </div>
            </div>
        </div>
    );
}
