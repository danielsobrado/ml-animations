import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Text } from '@react-three/drei';
import * as THREE from 'three';

function LossSurface() {
    // Create a mesh for f(x,y) = 0.1x^2 + 0.1y^2 + sin(2x)*0.5
    // A nice bumpy bowl
    const geometry = useMemo(() => {
        const geo = new THREE.PlaneGeometry(6, 6, 64, 64);
        const pos = geo.attributes.position;
        for (let i = 0; i < pos.count; i++) {
            const x = pos.getX(i);
            const y = pos.getY(i); // This is actually Z in 3D space usually, but Plane is XY
            // Let's map plane XY to world XZ, and height to Y
            const z = 0.2 * (x * x + y * y) - 0.5 * Math.cos(2 * x) * Math.cos(2 * y);
            pos.setZ(i, z); // Set Z (which we'll rotate to be Y)
        }
        geo.computeVertexNormals();
        return geo;
    }, []);

    return (
        <mesh geometry={geometry} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
            <meshStandardMaterial
                color="#10b981"
                roughness={0.4}
                metalness={0.1}
                side={THREE.DoubleSide}
                wireframe={false}
            />
        </mesh>
    );
}

function Grid() {
    return <gridHelper args={[10, 10, 0xffffff, 0x333333]} position={[0, -1, 0]} />;
}

export default function LandscapePanel() {
    return (
        <div className="w-full h-full min-h-[500px] bg-slate-900 rounded-xl overflow-hidden relative">
            <div className="absolute top-4 left-4 z-10 bg-slate-900/80 p-4 rounded-lg backdrop-blur text-slate-200 pointer-events-none">
                <h3 className="font-bold text-lg text-emerald-600 dark:text-emerald-400">The Loss Landscape</h3>
                <p className="text-sm">Drag to rotate. Scroll to zoom.</p>
                <p className="text-xs text-slate-800 dark:text-slate-400 mt-2">
                    The goal of training is to find the lowest point (Global Minimum).
                </p>
            </div>

            <Canvas shadows>
                <PerspectiveCamera makeDefault position={[5, 4, 5]} />
                <OrbitControls enableDamping />

                <ambientLight intensity={0.5} />
                <directionalLight position={[5, 10, 5]} intensity={1} castShadow />
                <pointLight position={[-5, 5, -5]} intensity={0.5} color="#ec4899" />

                <LossSurface />
                <Grid />

                {/* Ball at a local minimum */}
                <mesh position={[0, -0.5, 0]}>
                    <sphereGeometry args={[0.2, 32, 32]} />
                    <meshStandardMaterial color="#fbbf24" emissive="#fbbf24" emissiveIntensity={0.5} />
                </mesh>
            </Canvas>
        </div>
    );
}
