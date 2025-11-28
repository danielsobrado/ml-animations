import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { CSS2DRenderer, CSS2DObject } from 'three/examples/jsm/renderers/CSS2DRenderer';

export default function SpacePanel() {
    const containerRef = useRef(null);
    const [selectedCluster, setSelectedCluster] = useState(null);

    // Mock Embeddings Data
    const clusters = {
        animals: { color: 0xf87171, words: ['Dog', 'Cat', 'Lion', 'Tiger', 'Wolf', 'Puppy', 'Kitten'] },
        tech: { color: 0x60a5fa, words: ['Computer', 'Laptop', 'Phone', 'Code', 'Algorithm', 'AI', 'Robot'] },
        food: { color: 0xfacc15, words: ['Pizza', 'Burger', 'Sushi', 'Pasta', 'Bread', 'Cheese', 'Apple'] },
        cities: { color: 0x4ade80, words: ['Paris', 'London', 'Tokyo', 'New York', 'Berlin', 'Rome', 'Dubai'] }
    };

    useEffect(() => {
        if (!containerRef.current) return;

        const width = containerRef.current.clientWidth;
        const height = 500;

        // Scene
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0f172a); // Slate-900
        scene.fog = new THREE.FogExp2(0x0f172a, 0.05);

        // Camera
        const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
        camera.position.set(15, 10, 15);

        // Renderers
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(width, height);
        containerRef.current.appendChild(renderer.domElement);

        const labelRenderer = new CSS2DRenderer();
        labelRenderer.setSize(width, height);
        labelRenderer.domElement.style.position = 'absolute';
        labelRenderer.domElement.style.top = '0px';
        labelRenderer.domElement.style.pointerEvents = 'none'; // Allow clicks to pass through
        containerRef.current.appendChild(labelRenderer.domElement);

        // Controls
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.autoRotate = true;
        controls.autoRotateSpeed = 0.5;

        // Generate Words
        const objects = [];

        Object.entries(clusters).forEach(([key, data], clusterIdx) => {
            // Cluster centers
            const cx = (Math.random() - 0.5) * 20;
            const cy = (Math.random() - 0.5) * 20;
            const cz = (Math.random() - 0.5) * 20;

            data.words.forEach(word => {
                // Random offset from center
                const x = cx + (Math.random() - 0.5) * 4;
                const y = cy + (Math.random() - 0.5) * 4;
                const z = cz + (Math.random() - 0.5) * 4;

                // Visual Dot
                const geometry = new THREE.SphereGeometry(0.2, 16, 16);
                const material = new THREE.MeshBasicMaterial({ color: data.color });
                const mesh = new THREE.Mesh(geometry, material);
                mesh.position.set(x, y, z);
                scene.add(mesh);

                // Text Label
                const div = document.createElement('div');
                div.className = 'text-xs font-bold text-white bg-black/50 px-2 py-1 rounded backdrop-blur-sm';
                div.textContent = word;
                const label = new CSS2DObject(div);
                label.position.set(0, 0.4, 0);
                mesh.add(label);

                objects.push({ mesh, cluster: key });
            });
        });

        // Stars/Particles background
        const starsGeo = new THREE.BufferGeometry();
        const starsCount = 1000;
        const posArray = new Float32Array(starsCount * 3);
        for (let i = 0; i < starsCount * 3; i++) {
            posArray[i] = (Math.random() - 0.5) * 100;
        }
        starsGeo.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
        const starsMat = new THREE.PointsMaterial({ size: 0.1, color: 0xffffff, transparent: true, opacity: 0.5 });
        const starsMesh = new THREE.Points(starsGeo, starsMat);
        scene.add(starsMesh);

        // Animation Loop
        const animate = () => {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
            labelRenderer.render(scene, camera);
        };
        animate();

        // Handle Resize
        const handleResize = () => {
            if (!containerRef.current) return;
            const w = containerRef.current.clientWidth;
            const h = containerRef.current.clientHeight; // Use clientHeight to match container
            camera.aspect = w / h;
            camera.updateProjectionMatrix();
            renderer.setSize(w, h);
            labelRenderer.setSize(w, h);
        };
        window.addEventListener('resize', handleResize);

        return () => {
            window.removeEventListener('resize', handleResize);
            renderer.dispose();
            if (containerRef.current) {
                containerRef.current.innerHTML = ''; // Clear both renderers
            }
        };
    }, []); // Re-run only on mount (simplified)

    return (
        <div className="p-4 h-full flex flex-col">
            <div className="flex justify-between items-center mb-4">
                <h2 className="text-2xl font-bold text-cyan-400">3D Semantic Galaxy</h2>
                <div className="flex gap-2">
                    {Object.entries(clusters).map(([key, data]) => (
                        <div key={key} className="flex items-center gap-2 px-3 py-1 bg-slate-700 rounded-full text-xs">
                            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#' + data.color.toString(16) }}></div>
                            <span className="capitalize text-slate-300">{key}</span>
                        </div>
                    ))}
                </div>
            </div>

            <div className="bg-slate-700/50 rounded-xl border border-slate-600 p-2 text-center text-sm text-slate-400 mb-2">
                Drag to rotate • Scroll to zoom • Words with similar meanings cluster together
            </div>

            <div ref={containerRef} className="flex-1 rounded-xl overflow-hidden shadow-inner border border-slate-700 relative min-h-[400px]" />
        </div>
    );
}
