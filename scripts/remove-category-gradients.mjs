import fs from 'fs';
import path from 'path';

const file = path.join(process.cwd(), 'unified-app', 'src', 'data', 'animations.js');
let source = fs.readFileSync(file, 'utf8');

source = source.replace(/\n\s*color: 'from-[^']+',/g, '');
source = source.replace(/\n\s*categoryColor: category\.color,/g, '');

fs.writeFileSync(file, source);
console.log('Removed category gradient metadata.');
