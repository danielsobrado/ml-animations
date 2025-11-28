import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import gsap from 'gsap';

const matrixA = [[2, 3], [1, 4]];
const matrixB = [[1, 0, 2], [3, 2, 1]];
const result = [[11, 6, 7], [13, 8, 6]];

const COLORS = {
  matrixA: 0x5b9bd5,
  matrixB: 0x70ad47,
  result: 0xed7d31,
  highlight: 0xffc000,
  text: 0x333333,
  bg: 0xffffff
};

const STEPS = [
  { row: 0, col: 0, calc: '(2×1) + (3×3) = 11' },
  { row: 0, col: 1, calc: '(2×0) + (3×2) = 6' },
  { row: 0, col: 2, calc: '(2×2) + (3×1) = 7' },
  { row: 1, col: 0, calc: '(1×1) + (4×3) = 13' },
  { row: 1, col: 1, calc: '(1×0) + (4×2) = 8' },
  { row: 1, col: 2, calc: '(1×2) + (4×1) = 6' },
];

export default function AnimationPanel() {
  const containerRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const [step, setStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [explanation, setExplanation] = useState('Click Play to start the animation');
  const objectsRef = useRef({ cellsA: [], cellsB: [], cellsR: [], labels: [] });

  useEffect(() => {
    if (!containerRef.current) return;

    objectsRef.current = { cellsA: [], cellsB: [], cellsR: [], labels: [] };

    const width = containerRef.current.clientWidth;
    const height = 400;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(COLORS.bg);
    sceneRef.current = scene;

    const camera = new THREE.OrthographicCamera(
      width / -2, width / 2, height / 2, height / -2, 0.1, 1000
    );
    camera.position.z = 100;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    containerRef.current.appendChild(renderer.domElement);

    const cellSize = 40;
    const gap = 4;

    const createCell = (value, x, y, color) => {
      const group = new THREE.Group();
      
      const geometry = new THREE.PlaneGeometry(cellSize, cellSize);
      const material = new THREE.MeshBasicMaterial({ 
        color, 
        transparent: true, 
        opacity: 0.8 
      });
      const mesh = new THREE.Mesh(geometry, material);
      group.add(mesh);

      const border = new THREE.LineSegments(
        new THREE.EdgesGeometry(geometry),
        new THREE.LineBasicMaterial({ color: 0xffffff, linewidth: 2 })
      );
      group.add(border);

      const canvas = document.createElement('canvas');
      canvas.width = 128;
      canvas.height = 128;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = 'black';
      ctx.font = 'bold 64px Arial';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(value.toString(), 64, 64);

      const texture = new THREE.CanvasTexture(canvas);
      const labelMaterial = new THREE.SpriteMaterial({ map: texture, transparent: true });
      const label = new THREE.Sprite(labelMaterial);
      label.scale.set(32, 32, 1);
      group.add(label);

      group.position.set(x, y, 0);
      group.userData = { value, originalColor: color, mesh, label, canvas, texture };
      scene.add(group);
      return group;
    };

    const createLabel = (text, x, y, size = 40) => {
      const canvas = document.createElement('canvas');
      canvas.width = 512;
      canvas.height = 128;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = 'black';
      ctx.font = `bold ${size * 2}px Arial`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(text, 256, 64);

      const texture = new THREE.CanvasTexture(canvas);
      const material = new THREE.SpriteMaterial({ map: texture });
      const sprite = new THREE.Sprite(material);
      sprite.scale.set(160, 40, 1);
      sprite.position.set(x, y, 0);
      scene.add(sprite);
      return sprite;
    };

    const startX = -220;
    const matrixAX = startX;
    const matrixBX = startX + 140;
    const resultX = startX + 330;
    const topY = 80;

    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        const cell = createCell(
          matrixA[i][j],
          matrixAX + j * (cellSize + gap),
          topY - i * (cellSize + gap),
          COLORS.matrixA
        );
        cell.userData.row = i;
        cell.userData.col = j;
        objectsRef.current.cellsA.push(cell);
      }
    }

    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 3; j++) {
        const cell = createCell(
          matrixB[i][j],
          matrixBX + j * (cellSize + gap),
          topY - i * (cellSize + gap),
          COLORS.matrixB
        );
        cell.userData.row = i;
        cell.userData.col = j;
        objectsRef.current.cellsB.push(cell);
      }
    }

    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 3; j++) {
        const cell = createCell(
          '?',
          resultX + j * (cellSize + gap),
          topY - i * (cellSize + gap),
          COLORS.result
        );
        cell.userData.row = i;
        cell.userData.col = j;
        cell.userData.resultValue = result[i][j];
        cell.visible = false;
        objectsRef.current.cellsR.push(cell);
      }
    }

    createLabel('A', matrixAX + 20, topY + 50);
    createLabel('×', matrixAX + 85, topY - 20, 28);
    createLabel('B', matrixBX + 44, topY + 50);
    createLabel('=', resultX - 30, topY - 20, 28);
    createLabel('C', resultX + 44, topY + 50);

    const formulaY = -100;
    createLabel('Row × Column = Element', 0, formulaY + 40, 18);

    rendererRef.current = renderer;

    let animationId;
    const animate = () => {
      animationId = requestAnimationFrame(animate);
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
  }, []);

  const highlightCells = (rowA, colB) => {
    const { cellsA, cellsB } = objectsRef.current;

    cellsA.forEach(cell => {
      gsap.to(cell.userData.mesh.material.color, {
        r: (COLORS.matrixA >> 16 & 255) / 255,
        g: (COLORS.matrixA >> 8 & 255) / 255,
        b: (COLORS.matrixA & 255) / 255,
        duration: 0.3
      });
      gsap.to(cell.scale, { x: 1, y: 1, duration: 0.3 });
    });

    cellsB.forEach(cell => {
      gsap.to(cell.userData.mesh.material.color, {
        r: (COLORS.matrixB >> 16 & 255) / 255,
        g: (COLORS.matrixB >> 8 & 255) / 255,
        b: (COLORS.matrixB & 255) / 255,
        duration: 0.3
      });
      gsap.to(cell.scale, { x: 1, y: 1, duration: 0.3 });
    });

    cellsA.filter(c => c.userData.row === rowA).forEach((cell, i) => {
      gsap.to(cell.userData.mesh.material.color, {
        r: 0.4, g: 0.7, b: 1, duration: 0.3, delay: i * 0.1
      });
      gsap.to(cell.scale, { x: 1.2, y: 1.2, duration: 0.3, delay: i * 0.1 });
    });

    cellsB.filter(c => c.userData.col === colB).forEach((cell, i) => {
      gsap.to(cell.userData.mesh.material.color, {
        r: 0.6, g: 0.9, b: 0.5, duration: 0.3, delay: i * 0.1
      });
      gsap.to(cell.scale, { x: 1.2, y: 1.2, duration: 0.3, delay: i * 0.1 });
    });
  };

  const playAnimation = async () => {
    if (isPlaying) return;
    setIsPlaying(true);

    for (let i = step; i < STEPS.length; i++) {
      const { row, col, calc } = STEPS[i];
      setStep(i + 1);
      setExplanation(`Step ${i + 1}: Row ${row + 1} × Column ${col + 1}\n${calc}`);
      highlightCells(row, col);
      showResultCell(row, col);
      await new Promise(r => setTimeout(r, 1500));
    }

    setExplanation('Complete! Each result element is the dot product of a row from A and column from B.');
    setIsPlaying(false);
  };

  const showResultCell = (rowA, colB) => {
    const { cellsR } = objectsRef.current;
    const resultCell = cellsR.find(c => c.userData.row === rowA && c.userData.col === colB);
    if (resultCell && !resultCell.visible) {
      resultCell.visible = true;
      resultCell.scale.set(0.01, 0.01, 0.01);
      gsap.to(resultCell.scale, { x: 1, y: 1, z: 1, duration: 0.5, ease: 'back.out' });
      
      setTimeout(() => {
        const canvas = resultCell.userData.canvas;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = 'black';
        ctx.font = 'bold 64px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(resultCell.userData.resultValue.toString(), 64, 64);
        resultCell.userData.texture.needsUpdate = true;
      }, 300);
    }
  };

  const goToStep = (targetStep) => {
    if (isPlaying) return;
    
    const { cellsR } = objectsRef.current;
    
    for (let i = 0; i < STEPS.length; i++) {
      const { row, col } = STEPS[i];
      const resultCell = cellsR.find(c => c.userData.row === row && c.userData.col === col);
      
      if (i < targetStep) {
        if (resultCell && !resultCell.visible) {
          resultCell.visible = true;
          resultCell.scale.set(1, 1, 1);
          const canvas = resultCell.userData.canvas;
          const ctx = canvas.getContext('2d');
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          ctx.fillStyle = 'black';
          ctx.font = 'bold 64px Arial';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(resultCell.userData.resultValue.toString(), 64, 64);
          resultCell.userData.texture.needsUpdate = true;
        }
      } else {
        if (resultCell && resultCell.visible) {
          resultCell.visible = false;
          resultCell.scale.set(0.01, 0.01, 0.01);
          const canvas = resultCell.userData.canvas;
          const ctx = canvas.getContext('2d');
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          ctx.fillStyle = 'black';
          ctx.font = 'bold 64px Arial';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText('?', 64, 64);
          resultCell.userData.texture.needsUpdate = true;
        }
      }
    }

    setStep(targetStep);
    
    if (targetStep === 0) {
      resetHighlights();
      setExplanation('Click Play or use Next to step through the animation');
    } else if (targetStep > STEPS.length) {
      setExplanation('Complete! Each result element is the dot product of a row from A and column from B.');
    } else {
      const { row, col, calc } = STEPS[targetStep - 1];
      setExplanation(`Step ${targetStep}: Row ${row + 1} × Column ${col + 1}\n${calc}`);
      highlightCells(row, col);
      showResultCell(row, col);
    }
  };

  const nextStep = () => {
    if (isPlaying || step >= STEPS.length) return;
    goToStep(step + 1);
  };

  const prevStep = () => {
    if (isPlaying || step <= 0) return;
    goToStep(step - 1);
  };

  const resetHighlights = () => {
    const { cellsA, cellsB } = objectsRef.current;
    
    cellsA.forEach(cell => {
      gsap.to(cell.userData.mesh.material.color, {
        r: (COLORS.matrixA >> 16 & 255) / 255,
        g: (COLORS.matrixA >> 8 & 255) / 255,
        b: (COLORS.matrixA & 255) / 255,
        duration: 0.3
      });
      gsap.to(cell.scale, { x: 1, y: 1, duration: 0.3 });
    });

    cellsB.forEach(cell => {
      gsap.to(cell.userData.mesh.material.color, {
        r: (COLORS.matrixB >> 16 & 255) / 255,
        g: (COLORS.matrixB >> 8 & 255) / 255,
        b: (COLORS.matrixB & 255) / 255,
        duration: 0.3
      });
      gsap.to(cell.scale, { x: 1, y: 1, duration: 0.3 });
    });
  };

  const reset = () => {
    if (isPlaying) return;
    goToStep(0);
  };

  return (
    <div className="flex flex-col items-center p-3">
      <h2 className="text-xl font-bold text-gray-800 mb-2">Animation Demo</h2>
      
      <div ref={containerRef} className="w-full rounded-lg overflow-hidden shadow-lg bg-white" />
      
      <div className="mt-2 p-2 bg-white rounded-lg w-full text-center shadow">
        <p className="text-gray-800 whitespace-pre-line text-sm">{explanation}</p>
      </div>
      
      <div className="flex items-center gap-2 mt-2">
        <button
          onClick={prevStep}
          disabled={isPlaying || step <= 0}
          className="px-3 py-1 bg-blue-500 hover:bg-blue-600 disabled:bg-gray-400 disabled:cursor-not-allowed text-white font-bold rounded-lg transition-colors text-sm"
        >
          ← Prev
        </button>
        
        <div className="px-3 py-1 bg-gray-200 rounded-lg font-mono text-gray-700 min-w-[80px] text-center text-sm">
          {step} / {STEPS.length}
        </div>
        
        <button
          onClick={nextStep}
          disabled={isPlaying || step >= STEPS.length}
          className="px-3 py-1 bg-blue-500 hover:bg-blue-600 disabled:bg-gray-400 disabled:cursor-not-allowed text-white font-bold rounded-lg transition-colors text-sm"
        >
          Next →
        </button>
      </div>

      <div className="flex gap-2 mt-2">
        <button
          onClick={playAnimation}
          disabled={isPlaying || step >= STEPS.length}
          className="px-4 py-2 bg-green-500 hover:bg-green-600 disabled:bg-gray-400 disabled:cursor-not-allowed text-white font-bold rounded-lg transition-colors text-sm"
        >
          {isPlaying ? 'Playing...' : '▶ Play'}
        </button>
        <button
          onClick={reset}
          disabled={isPlaying}
          className="px-4 py-2 bg-red-500 hover:bg-red-600 disabled:bg-gray-400 disabled:cursor-not-allowed text-white font-bold rounded-lg transition-colors text-sm"
        >
          ↺ Reset
        </button>
      </div>
    </div>
  );
}
