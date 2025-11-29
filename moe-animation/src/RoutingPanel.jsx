import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import gsap from 'gsap';

export default function RoutingPanel({ numExperts, topK, batchSize, onGenerate }) {
    const containerRef = useRef(null);
    const sceneRef = useRef(null);
    const rendererRef = useRef(null);
    const [isAnimating, setIsAnimating] = useState(false);

    useEffect(() => {
        if (!containerRef.current) return;

        const width = containerRef.current.clientWidth;
        const height = 500;

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0f172a); // Slate-900
        sceneRef.current = scene;

        const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
        camera.position.z = 50;
        camera.position.y = 10;

        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setSize(width, height);
        containerRef.current.appendChild(renderer.domElement);
        rendererRef.current = renderer;

        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        scene.add(ambientLight);
        const pointLight = new THREE.PointLight(0xffffff, 1);
        pointLight.position.set(20, 20, 20);
        scene.add(pointLight);

        // Router (Gate)
        const routerGeometry = new THREE.TorusGeometry(5, 0.5, 16, 100);
        const routerMaterial = new THREE.MeshStandardMaterial({
            color: 0x00f3ff,
            emissive: 0x00f3ff,
            emissiveIntensity: 0.5
        });
        const router = new THREE.Mesh(routerGeometry, routerMaterial);
        router.position.set(0, 0, 0);
        scene.add(router);

        // Experts
        const experts = [];
        const radius = 25;
        const expertColors = [0xff00ff, 0x00ff9f, 0xbc13fe, 0xffea00, 0xff0055, 0x00ccff, 0xff9900, 0x99ff00];

        for (let i = 0; i < numExperts; i++) {
            const angle = (i / numExperts) * Math.PI * 2; // Semicircle layout? No, circle for now
            // Let's do a semi-circle layout for better visibility
            const x = Math.cos(angle) * radius;
            const y = Math.sin(angle) * radius * 0.5; // Flattened circle
            const z = -20;

            const geometry = new THREE.BoxGeometry(4, 6, 2);
            const material = new THREE.MeshStandardMaterial({
                color: expertColors[i % expertColors.length],
                transparent: true,
                opacity: 0.8
            });
            const expert = new THREE.Mesh(geometry, material);
            expert.position.set(x, y, z);

            // Label (simplified as a smaller box for now)
            const labelGeo = new THREE.PlaneGeometry(3, 1);
            const labelMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
            const label = new THREE.Mesh(labelGeo, labelMat);
            label.position.set(0, 4, 0);
            expert.add(label);

            scene.add(expert);
            experts.push(expert);
        }

        // Animation Loop
        let animationId;
        const animate = () => {
            animationId = requestAnimationFrame(animate);
            router.rotation.z += 0.01;
            renderer.render(scene, camera);
        };
        animate();

        return () => {
            cancelAnimationFrame(animationId);
            renderer.dispose();
            if (containerRef.current?.contains(renderer.domElement)) {
                containerRef.current.removeChild(renderer.domElement);
            }
        };
    }, [numExperts]);

    const generateBatch = () => {
        if (isAnimating || !sceneRef.current) return;
        setIsAnimating(true);
        if (onGenerate) onGenerate();

        const scene = sceneRef.current;
        const tokens = [];

        // Create tokens
        for (let i = 0; i < batchSize; i++) {
            const geometry = new THREE.SphereGeometry(0.8, 16, 16);
            const material = new THREE.MeshStandardMaterial({ color: 0xffffff });
            const token = new THREE.Mesh(geometry, material);

            // Start position (left side or bottom)
            token.position.set(0, -20 - i * 2, 10);
            scene.add(token);
            tokens.push(token);

            // Animate
            const tl = gsap.timeline();

            // 1. Move to Router
            tl.to(token.position, {
                x: 0,
                y: 0,
                z: 0,
                duration: 1,
                ease: "power2.out",
                delay: i * 0.1
            });

            // 2. Route to Top-K Experts
            // For simplicity, pick random experts for now (logic should be passed in prop ideally)
            const targetIndices = [];
            while (targetIndices.length < topK) {
                const idx = Math.floor(Math.random() * numExperts);
                if (!targetIndices.includes(idx)) targetIndices.push(idx);
            }

            // If topK > 1, we need to clone the token visually or split it
            // For visual simplicity, let's just move the main token to the first expert
            // and spawn "ghost" tokens for others if needed.

            const expert = scene.children.find(c => c.geometry.type === 'BoxGeometry' && scene.children.indexOf(c) > 2 + targetIndices[0]); // Hacky find
            // Better: we need to store expert references. 
            // Re-finding them:
            const expertMeshes = scene.children.filter(c => c.geometry.type === 'BoxGeometry');

            targetIndices.forEach((expertIdx, k) => {
                const targetExpert = expertMeshes[expertIdx];

                if (k === 0) {
                    // Main token moves
                    tl.to(token.position, {
                        x: targetExpert.position.x,
                        y: targetExpert.position.y,
                        z: targetExpert.position.z,
                        duration: 1,
                        ease: "power2.inOut",
                        onComplete: () => {
                            // Flash expert
                            gsap.to(targetExpert.material, { emissive: 0xffffff, duration: 0.1, yoyo: true, repeat: 1 });
                        }
                    });

                    // Fade out
                    tl.to(token.material, { opacity: 0, duration: 0.5, delay: 0.5 });
                } else {
                    // Spawn ghost token for other experts
                    // (Simplified for this iteration: just route to primary)
                }
            });
        }

        setTimeout(() => {
            setIsAnimating(false);
            tokens.forEach(t => scene.remove(t));
        }, (batchSize * 0.1 + 3) * 1000);
    };

    return (
        <div className="relative w-full h-full bg-slate-900 rounded-xl overflow-hidden border border-slate-700 shadow-2xl">
            <div ref={containerRef} className="w-full h-[500px]" />

            <div className="absolute bottom-4 left-4 right-4 flex justify-between items-center pointer-events-none">
                <div className="bg-slate-800/80 backdrop-blur p-4 rounded-lg border border-slate-600 pointer-events-auto">
                    <h3 className="text-neon-blue font-bold mb-2">Router Visualization</h3>
                    <p className="text-sm text-slate-300 max-w-xs">
                        Tokens (white spheres) enter the Router (ring) and are dispatched to the Top-{topK} Experts (colored boxes) based on learned gating weights.
                    </p>
                </div>

                <button
                    onClick={generateBatch}
                    disabled={isAnimating}
                    className={`pointer-events-auto px-6 py-3 rounded-lg font-bold text-black transition-all transform hover:scale-105 ${isAnimating
                            ? 'bg-gray-500 cursor-not-allowed'
                            : 'bg-neon-green hover:bg-white shadow-[0_0_15px_rgba(0,255,159,0.5)]'
                        }`}
                >
                    {isAnimating ? 'Routing...' : 'Generate Batch'}
                </button>
            </div>
        </div>
    );
}
