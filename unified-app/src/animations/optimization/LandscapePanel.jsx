import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

export default function LandscapePanel() {
    const containerRef = useRef(null);

    useEffect(() => {
        const container = containerRef.current;
        if (!container) return;

        const width = container.clientWidth || 900;
        const height = 500;
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xfefcf7);

        const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 1000);
        camera.position.set(5, 4, 5);

        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        renderer.setSize(width, height);
        renderer.shadowMap.enabled = true;
        container.appendChild(renderer.domElement);

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.autoRotate = true;
        controls.autoRotateSpeed = 0.35;

        const geometry = new THREE.PlaneGeometry(6, 6, 72, 72);
        const pos = geometry.attributes.position;
        for (let i = 0; i < pos.count; i += 1) {
            const x = pos.getX(i);
            const y = pos.getY(i);
            const z = 0.2 * (x * x + y * y) - 0.5 * Math.cos(2 * x) * Math.cos(2 * y);
            pos.setZ(i, z);
        }
        geometry.computeVertexNormals();

        const surface = new THREE.Mesh(
            geometry,
            new THREE.MeshStandardMaterial({
                color: 0x3a6a3a,
                roughness: 0.58,
                metalness: 0.02,
                side: THREE.DoubleSide,
            })
        );
        surface.rotation.x = -Math.PI / 2;
        surface.receiveShadow = true;
        scene.add(surface);

        const grid = new THREE.GridHelper(10, 10, 0xb6ac93, 0xece6d3);
        grid.position.y = -1;
        scene.add(grid);

        const marker = new THREE.Mesh(
            new THREE.SphereGeometry(0.2, 32, 32),
            new THREE.MeshStandardMaterial({ color: 0xa85a3a, emissive: 0x6b2f1c, emissiveIntensity: 0.2 })
        );
        marker.position.set(0, -0.5, 0);
        marker.castShadow = true;
        scene.add(marker);

        scene.add(new THREE.AmbientLight(0xffffff, 0.65));
        const key = new THREE.DirectionalLight(0xffffff, 1);
        key.position.set(5, 10, 5);
        key.castShadow = true;
        scene.add(key);
        scene.add(new THREE.PointLight(0x264273, 0.4, 20));

        let animationFrame = 0;
        const animate = () => {
            animationFrame = requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        };
        animate();

        const handleResize = () => {
            if (!containerRef.current) return;
            const nextWidth = containerRef.current.clientWidth || width;
            camera.aspect = nextWidth / height;
            camera.updateProjectionMatrix();
            renderer.setSize(nextWidth, height);
        };
        window.addEventListener('resize', handleResize);

        return () => {
            cancelAnimationFrame(animationFrame);
            window.removeEventListener('resize', handleResize);
            controls.dispose();
            geometry.dispose();
            surface.material.dispose();
            marker.geometry.dispose();
            marker.material.dispose();
            renderer.dispose();
            if (container.contains(renderer.domElement)) {
                container.removeChild(renderer.domElement);
            }
        };
    }, []);

    return (
        <div className="w-full min-h-[500px] overflow-hidden relative border border-[var(--ds-rule)] bg-[var(--ds-paper)]">
            <div className="absolute top-4 left-4 z-10 border border-[var(--ds-rule)] bg-[var(--ds-paper)]/90 p-4 text-[var(--ds-ink)] pointer-events-none max-w-sm">
                <h3 className="font-bold text-lg text-[var(--ds-accent)]">The Loss Landscape</h3>
                <p className="text-sm">The surface rotates slowly to reveal the training objective.</p>
                <p className="text-xs text-[var(--ds-mute)] mt-2">
                    Training moves parameters toward the lowest visible region.
                </p>
            </div>
            <div ref={containerRef} className="min-h-[500px]" />
        </div>
    );
}
