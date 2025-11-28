import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

export default function LandscapePanel() {
    const containerRef = useRef(null);
    const [ballPos, setBallPos] = useState({ x: 0, z: 0 });
    const ballRef = useRef(null);
    const isDescendingRef = useRef(false);

    // Rastrigin-like function for terrain: z = x^2 + y^2 - 10*cos(2*PI*x) - 10*cos(2*PI*y)
    // Simplified for visual clarity: z = x^2 + y^2 + sin(5x) + sin(5y)
    const getHeight = (x, z) => {
        return (x * x + z * z) * 0.1 + Math.sin(x * 2) + Math.sin(z * 2);
    };

    const getGradient = (x, z) => {
        const delta = 0.01;
        const h = getHeight(x, z);
        const h_dx = getHeight(x + delta, z);
        const h_dz = getHeight(x, z + delta);
        return {
            x: (h_dx - h) / delta,
            z: (h_dz - h) / delta
        };
    };

    useEffect(() => {
        if (!containerRef.current) return;

        const width = containerRef.current.clientWidth;
        const height = 500;

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xf0f9ff);
        scene.fog = new THREE.Fog(0xf0f9ff, 10, 50);

        const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
        camera.position.set(10, 10, 10);
        camera.lookAt(0, 0, 0);

        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(width, height);
        renderer.shadowMap.enabled = true;
        containerRef.current.appendChild(renderer.domElement);

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;

        // Lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(10, 20, 10);
        dirLight.castShadow = true;
        scene.add(dirLight);

        // Terrain Mesh
        const geometry = new THREE.PlaneGeometry(20, 20, 100, 100);
        const positions = geometry.attributes.position;
        for (let i = 0; i < positions.count; i++) {
            const x = positions.getX(i);
            const z = positions.getY(i); // Plane is initially vertical, so Y is Z
            positions.setZ(i, getHeight(x, z));
        }
        geometry.computeVertexNormals();
        geometry.rotateX(-Math.PI / 2); // Rotate to horizontal

        const material = new THREE.MeshStandardMaterial({
            color: 0x3b82f6,
            roughness: 0.8,
            flatShading: true,
            side: THREE.DoubleSide
        });
        const terrain = new THREE.Mesh(geometry, material);
        terrain.receiveShadow = true;
        scene.add(terrain);

        // Ball
        const ballGeo = new THREE.SphereGeometry(0.3, 32, 32);
        const ballMat = new THREE.MeshStandardMaterial({ color: 0xef4444 });
        const ball = new THREE.Mesh(ballGeo, ballMat);
        ball.castShadow = true;
        scene.add(ball);
        ballRef.current = ball;

        // Animation Loop
        const animate = () => {
            requestAnimationFrame(animate);
            controls.update();

            if (isDescendingRef.current && ballRef.current) {
                const pos = ballRef.current.position;
                const grad = getGradient(pos.x, pos.z);
                const lr = 0.05;

                pos.x -= grad.x * lr;
                pos.z -= grad.z * lr;
                pos.y = getHeight(pos.x, pos.z) + 0.3;

                // Stop if gradient is small
                if (Math.abs(grad.x) < 0.01 && Math.abs(grad.z) < 0.01) {
                    isDescendingRef.current = false;
                }
            }

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

    const dropBall = () => {
        if (ballRef.current) {
            // Random position
            const x = (Math.random() - 0.5) * 10;
            const z = (Math.random() - 0.5) * 10;
            ballRef.current.position.set(x, getHeight(x, z) + 0.3, z);
            isDescendingRef.current = true;
        }
    };

    return (
        <div className="p-4 h-full flex flex-col">
            <div className="flex justify-between items-center mb-4">
                <h2 className="text-2xl font-bold text-slate-800">3D Landscape Explorer</h2>
                <button
                    onClick={dropBall}
                    className="px-6 py-2 bg-indigo-600 text-white rounded-lg font-bold hover:bg-indigo-700 shadow"
                >
                    🎲 Drop Ball & Descend
                </button>
            </div>
            <div className="bg-slate-100 rounded-xl border border-slate-300 p-2 text-center text-sm text-slate-500 mb-2">
                Drag to rotate • Scroll to zoom • Watch the ball find the local minimum
            </div>
            <div ref={containerRef} className="flex-1 rounded-xl overflow-hidden shadow-inner border border-slate-300" />
        </div>
    );
}
