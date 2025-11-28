/**
 * Animation Integration Helper
 * 
 * This script helps integrate animations from the original separate folders
 * into the unified application structure.
 * 
 * Usage: node scripts/integrate-animation.js <animation-id>
 * Example: node scripts/integrate-animation.js transformer
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const ANIMATIONS_DIR = path.join(__dirname, '..', 'src', 'animations');
const ROOT_DIR = path.join(__dirname, '..', '..');

function getAnimationSourcePath(animationId) {
  return path.join(ROOT_DIR, `${animationId}-animation`, 'src');
}

function getAnimationDestPath(animationId) {
  return path.join(ANIMATIONS_DIR, animationId);
}

// Color replacements for theme support
const colorReplacements = [
  // Background colors
  { from: /bg-slate-900/g, to: 'bg-slate-50 dark:bg-slate-900' },
  { from: /bg-slate-800/g, to: 'bg-white dark:bg-slate-800' },
  { from: /bg-slate-700/g, to: 'bg-slate-100 dark:bg-slate-700' },
  
  // Text colors
  { from: /text-white(?!\s*dark:)/g, to: 'text-slate-900 dark:text-white' },
  { from: /text-slate-400(?!\s*dark:)/g, to: 'text-slate-600 dark:text-slate-400' },
  { from: /text-slate-300(?!\s*dark:)/g, to: 'text-slate-700 dark:text-slate-300' },
  
  // Border colors
  { from: /border-slate-700(?!\s*dark:)/g, to: 'border-slate-200 dark:border-slate-700' },
  { from: /border-slate-600(?!\s*dark:)/g, to: 'border-slate-300 dark:border-slate-600' },
];

function applyThemeReplacements(content) {
  let result = content;
  for (const replacement of colorReplacements) {
    result = result.replace(replacement.from, replacement.to);
  }
  return result;
}

function integrateAnimation(animationId) {
  const sourcePath = getAnimationSourcePath(animationId);
  const destPath = getAnimationDestPath(animationId);
  
  // Check if source exists
  if (!fs.existsSync(sourcePath)) {
    console.error(`Source animation not found: ${sourcePath}`);
    process.exit(1);
  }
  
  // Create destination directory
  if (!fs.existsSync(destPath)) {
    fs.mkdirSync(destPath, { recursive: true });
  }
  
  // Get all JSX files
  const files = fs.readdirSync(sourcePath).filter(f => f.endsWith('.jsx'));
  
  // Copy and transform files
  for (const file of files) {
    const sourceFile = path.join(sourcePath, file);
    let content = fs.readFileSync(sourceFile, 'utf-8');
    
    // Skip main.jsx as it's not needed
    if (file === 'main.jsx') continue;
    
    // Apply theme replacements
    content = applyThemeReplacements(content);
    
    // Rename App.jsx to index.jsx
    const destFile = file === 'App.jsx' ? 'index.jsx' : file;
    
    fs.writeFileSync(path.join(destPath, destFile), content);
    console.log(`✓ Copied and transformed: ${file} -> ${destFile}`);
  }
  
  // Update the animation registry
  updateRegistry(animationId);
  
  console.log(`\n✅ Animation "${animationId}" integrated successfully!`);
  console.log(`\nNext steps:`);
  console.log(`1. Review the files in src/animations/${animationId}/`);
  console.log(`2. Test the animation at /animation/${animationId}`);
  console.log(`3. Fine-tune any remaining theme colors`);
}

function updateRegistry(animationId) {
  const registryPath = path.join(ANIMATIONS_DIR, 'index.js');
  let content = fs.readFileSync(registryPath, 'utf-8');
  
  // Check if already registered
  if (content.includes(`'${animationId}':`)) {
    console.log(`Registry already contains ${animationId}`);
    return;
  }
  
  // Add new import line
  const importLine = `  '${animationId}': lazy(() => import('./${animationId}')),`;
  
  // Find the closing brace of animationRegistry
  const insertPoint = content.indexOf('};');
  content = content.slice(0, insertPoint) + importLine + '\n' + content.slice(insertPoint);
  
  fs.writeFileSync(registryPath, content);
  console.log(`✓ Updated animation registry`);
}

// Main
const animationId = process.argv[2];

if (!animationId) {
  console.log('Usage: node scripts/integrate-animation.js <animation-id>');
  console.log('Example: node scripts/integrate-animation.js transformer');
  process.exit(1);
}

integrateAnimation(animationId);
