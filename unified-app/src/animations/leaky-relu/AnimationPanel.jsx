import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import gsap from 'gsap';

// Leaky ReLU example: X = [2, 1, 3], W = [1, -1, 1], b = -5, α = 0.01
// Dot product: 2*1 + 1*(-1) + 3*1 = 2 - 1 + 3 = 4
// z = 4 + (-5) = -1
// Leaky ReLU(-1) = 0.01 * (-1) = -0.01

const ALPHA = 0.01;

export default function AnimationPanel({ onStepChange }) {
  const containerRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const animationRef = useRef(null);
  const [step, setStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const X = [2, 1, 3];
  const W = [1, -1, 1];
  const b = -5;
  const dotProduct = X.reduce((sum, x, i) => sum + x * W[i], 0); // 4
  const zValue = dotProduct + b; // -1
  const leakyReluOutput = zValue > 0 ? zValue : ALPHA * zValue; // -0.01

  useEffect(() => {
    if (onStepChange) {
      onStepChange(step, zValue, leakyReluOutput);
    }
  }, [step, onStepChange, zValue, leakyReluOutput]);

  useEffect(() => {
    if (!containerRef.current) return;

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf8f9fa);
    sceneRef.current = scene;

    const width = containerRef.current.clientWidth;
    const height = 400;

    const camera = new THREE.OrthographicCamera(
      -width / 2, width / 2,
      height / 2, -height / 2,
      0.1, 1000
    );
    camera.position.z = 100;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
    scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight.position.set(0, 1, 1);
    scene.add(directionalLight);

    // Animation loop
    const animate = () => {
      animationRef.current = requestAnimationFrame(animate);
      renderer.render(scene, camera);
    };
    animate();

    // Handle resize
    const handleResize = () => {
      if (!containerRef.current) return;
      const newWidth = containerRef.current.clientWidth;
      camera.left = -newWidth / 2;
      camera.right = newWidth / 2;
      camera.updateProjectionMatrix();
      renderer.setSize(newWidth, height);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(animationRef.current);
      renderer.dispose();
      if (containerRef.current && renderer.domElement) {
        containerRef.current.removeChild(renderer.domElement);
      }
    };
  }, []);

  // Update visualization based on step
  useEffect(() => {
    if (!sceneRef.current) return;
    const scene = sceneRef.current;

    // Clear previous objects
    while (scene.children.length > 2) {
      scene.remove(scene.children[scene.children.length - 1]);
    }

    const createBox = (color, x, y, label, value) => {
      const group = new THREE.Group();
      
      const geometry = new THREE.BoxGeometry(50, 50, 10);
      const material = new THREE.MeshLambertMaterial({ color });
      const mesh = new THREE.Mesh(geometry, material);
      group.add(mesh);

      // Add border
      const edges = new THREE.EdgesGeometry(geometry);
      const lineMaterial = new THREE.LineBasicMaterial({ color: 0x333333 });
      const line = new THREE.LineSegments(edges, lineMaterial);
      group.add(line);

      group.position.set(x, y, 0);
      group.userData = { label, value };
      return group;
    };

    const createLabel = (text, x, y, size = 16, color = '#333') => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      canvas.width = 256;
      canvas.height = 64;
      ctx.fillStyle = color;
      ctx.font = `bold ${size * 2}px Arial`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(text, 128, 32);

      const texture = new THREE.CanvasTexture(canvas);
      const material = new THREE.SpriteMaterial({ map: texture });
      const sprite = new THREE.Sprite(material);
      sprite.position.set(x, y, 1);
      sprite.scale.set(128, 32, 1);
      return sprite;
    };

    // Step 0: Initial state - show nothing
    if (step === 0) {
      const label = createLabel('Click Play to start', 0, 0, 20);
      scene.add(label);
    }

    // Step 1: Show input X
    if (step >= 1) {
      const xLabel = createLabel('X (Input)', -200, 140, 18, '#2563eb');
      scene.add(xLabel);
      
      X.forEach((val, i) => {
        const box = createBox(0x3b82f6, -200, 80 - i * 60, 'x', val);
        scene.add(box);
        const valLabel = createLabel(val.toString(), -200, 80 - i * 60, 20, '#000');
        scene.add(valLabel);
      });
    }

    // Step 2: Show weights W
    if (step >= 2) {
      const wLabel = createLabel('W (Weights)', -80, 140, 18, '#16a34a');
      scene.add(wLabel);
      
      W.forEach((val, i) => {
        const box = createBox(0x22c55e, -80, 80 - i * 60, 'w', val);
        scene.add(box);
        const valLabel = createLabel(val.toString(), -80, 80 - i * 60, 20, '#000');
        scene.add(valLabel);
      });
    }

    // Step 3: Show bias b
    if (step >= 3) {
      const bLabel = createLabel('b (Bias)', 40, 140, 18, '#ca8a04');
      scene.add(bLabel);
      
      const box = createBox(0xeab308, 40, 20, 'b', b);
      scene.add(box);
      const valLabel = createLabel(b.toString(), 40, 20, 20, '#000');
      scene.add(valLabel);
    }

    // Step 4: Show dot product calculation
    if (step >= 4) {
      const dotLabel = createLabel('X · W', 160, 140, 18, '#7c3aed');
      scene.add(dotLabel);
      
      const calcLabel = createLabel(`${X[0]}×${W[0]} + ${X[1]}×${W[1]} + ${X[2]}×${W[2]}`, 160, 80, 12, '#666');
      scene.add(calcLabel);
      
      const resultLabel = createLabel(`= ${dotProduct}`, 160, 40, 20, '#7c3aed');
      scene.add(resultLabel);
    }

    // Step 5: Show z = dot + bias
    if (step >= 5) {
      const zLabel = createLabel('z = X·W + b', 160, -20, 16, '#dc2626');
      scene.add(zLabel);
      
      const zCalcLabel = createLabel(`= ${dotProduct} + (${b})`, 160, -60, 14, '#666');
      scene.add(zCalcLabel);
      
      const box = createBox(0xef4444, 160, -110, 'z', zValue);
      scene.add(box);
      const zValLabel = createLabel(zValue.toString(), 160, -110, 20, '#000');
      scene.add(zValLabel);
    }

    // Step 6: Apply Leaky ReLU
    if (step >= 6) {
      const leakyReluLabel = createLabel('Leaky ReLU(z)', 300, 100, 18, '#ea580c');
      scene.add(leakyReluLabel);
      
      const formulaLabel = createLabel(`α = ${ALPHA}`, 300, 60, 14, '#666');
      scene.add(formulaLabel);
      
      const conditionLabel = createLabel(zValue > 0 ? 'z > 0 → z' : `z ≤ 0 → α × z`, 300, 20, 14, '#666');
      scene.add(conditionLabel);
      
      const calcLabel = createLabel(zValue > 0 ? `= ${zValue}` : `= ${ALPHA} × ${zValue}`, 300, -20, 14, '#666');
      scene.add(calcLabel);
      
      const box = createBox(0xf97316, 300, -80, 'leaky_relu', leakyReluOutput);
      scene.add(box);
      const outputLabel = createLabel(leakyReluOutput.toFixed(2), 300, -80, 20, '#000');
      scene.add(outputLabel);
    }

  }, [step, X, W, b, dotProduct, zValue, leakyReluOutput]);

  const playAnimation = () => {
    if (isPlaying) return;
    setIsPlaying(true);
    setStep(0);

    const timeline = gsap.timeline({
      onComplete: () => setIsPlaying(false)
    });

    for (let i = 1; i <= 6; i++) {
      timeline.call(() => setStep(i), null, i * 1.2);
    }
  };

  const resetAnimation = () => {
    setStep(0);
    setIsPlaying(false);
  };

  const stepLabels = [
    'Ready',
    'Input Vector X',
    'Weight Vector W',
    'Bias b',
    'Dot Product X · W',
    'Linear Combination z',
    'Apply Leaky ReLU'
  ];

  return (
    <div className="p-4">
      <h2 className="text-xl font-bold text-gray-800 mb-3 text-center">Animation Demo</h2>
      
      <div ref={containerRef} className="w-full rounded-lg overflow-hidden border border-gray-200" />
      
      <div className="mt-4 flex justify-center gap-4">
        <button
          onClick={playAnimation}
          disabled={isPlaying}
          className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed font-medium"
        >
          {isPlaying ? 'Playing...' : 'Play'}
        </button>
        <button
          onClick={resetAnimation}
          className="px-6 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 font-medium"
        >
          Reset
        </button>
      </div>

      <div className="mt-4 text-center">
        <p className="text-lg font-semibold text-gray-700">
          Step {step}: {stepLabels[step]}
        </p>
        <div className="mt-2 flex justify-center gap-1">
          {stepLabels.map((_, i) => (
            <div
              key={i}
              className={`w-3 h-3 rounded-full ${i <= step ? 'bg-blue-600' : 'bg-gray-300'}`}
            />
          ))}
        </div>
      </div>

      <div className="mt-4 p-3 bg-orange-50 rounded-lg border border-orange-200">
        <p className="text-sm text-gray-700">
          <strong>Leaky ReLU Formula:</strong> f(x) = x if x &gt; 0, else α × x
        </p>
        <p className="text-sm text-gray-600 mt-1">
          Where α is a small constant (typically 0.01). Unlike ReLU, Leaky ReLU allows a small gradient when the input is negative.
        </p>
      </div>
    </div>
  );
}
