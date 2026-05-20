import React, { useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
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
        <div className="w-full h-full min-h-[500px] overflow-hidden relative border border-[var(--ds-border)] bg-[var(--ds-paper)]">
            <div className="absolute top-4 left-4 z-10 border border-[var(--ds-border)] bg-[var(--ds-paper)]/90 p-4 text-[var(--ds-ink)] pointer-events-none">
                <h3 className="font-bold text-lg text-[var(--ds-accent)]">The Loss Landscape</h3>
                <p className="text-sm">The surface rotates slowly to reveal the training objective.</p>
                <p className="text-xs text-[var(--ds-muted)] mt-2">
                    The goal of training is to find the lowest point (Global Minimum).
                </p>
            </div>

            <Canvas shadows camera={{ position: [5, 4, 5], fov: 50 }}>

                <ambientLight intensity={0.5} />
                <directionalLight position={[5, 10, 5]} intensity={1} castShadow />
                <pointLight position={[-5, 5, -5]} intensity={0.35} color="#244a7f" />

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
