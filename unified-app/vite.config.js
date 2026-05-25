import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/ml-animations/',
  resolve: {
    alias: {
      '@': '/src',
    },
  },
  build: {
    chunkSizeWarningLimit: 800,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('/src/data/conceptMaps.js')) return 'concept-maps-data';
          if (id.includes('/src/data/lessonAssessments.js')) return 'assessment-data';
          if (id.includes('/src/data/glossaryRepository.js')) return 'glossary-data';
          if (id.includes('/src/data/curriculumDepth.js')) return 'curriculum-depth-data';
          if (id.includes('/src/data/animationLearning.js')) return 'learning-model-data';
          if (id.includes('/src/data/scenarioQuestions.js')) return 'assessment-data';
          if (id.includes('/src/labs/runtime/')) return 'lab-runtime';
          if (!id.includes('node_modules')) return undefined;
          if (id.includes('recharts')) return 'charts';
          if (id.includes('three')) return 'three';
          if (id.includes('react-router-dom')) return 'router';
          if (id.includes('lucide-react')) return 'icons';
          if (id.includes('mind-elixir')) return 'mindmap';
          if (id.includes('react') || id.includes('scheduler')) return 'react-vendor';
          if (id.includes('framer-motion')) return 'motion';
          if (id.includes('gsap')) return 'animation-vendor';
          if (id.includes('katex')) return 'math-rendering';
          return undefined;
        },
      },
    },
  },
})
