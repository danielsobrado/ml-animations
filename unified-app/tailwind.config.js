/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#f0f9ff',
          100: '#e0f2fe',
          200: '#bae6fd',
          300: '#7dd3fc',
          400: '#38bdf8',
          500: '#0ea5e9',
          600: '#0284c7',
          700: '#0369a1',
          800: '#075985',
          900: '#0c4a6e',
          950: '#082f49',
        },
      },
      animation: {
        'pulse-glow': 'pulse-glow 2s ease-in-out infinite',
        'flow-line': 'flow-line 1s ease-out forwards',
        'slide-in': 'slide-in 0.3s ease-out',
        'fade-in': 'fade-in 0.2s ease-out',
      },
      keyframes: {
        'pulse-glow': {
          '0%, 100%': { boxShadow: '0 0 5px currentColor' },
          '50%': { boxShadow: '0 0 20px currentColor, 0 0 30px currentColor' },
        },
        'flow-line': {
          '0%': { strokeDashoffset: '100' },
          '100%': { strokeDashoffset: '0' },
        },
        'slide-in': {
          '0%': { transform: 'translateX(-10px)', opacity: '0' },
          '100%': { transform: 'translateX(0)', opacity: '1' },
        },
        'fade-in': {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
      },
    },
  },
  plugins: [],
}
