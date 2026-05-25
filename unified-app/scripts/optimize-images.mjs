import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import sharp from 'sharp';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '../..');

const defaults = {
  format: 'jpeg',
  quality: 82,
  maxWidth: 1600,
  maxHeight: 1600,
  dryRun: false,
  keepPng: false,
  overwrite: false,
};

const outputExtensions = {
  jpeg: '.jpg',
  webp: '.webp',
  avif: '.avif',
};

function printUsage() {
  console.log(`
Usage:
  npm run images:optimize -- [folder] [options]

Defaults:
  folder      all existing ../*-animation/images folders
  format      jpeg
  quality     82
  max size    1600x1600

Options:
  --format=jpeg|webp|avif
  --quality=1-100
  --max-width=NUMBER
  --max-height=NUMBER
  --keep-png       Keep source PNGs after successful conversion
  --overwrite      Replace existing optimized files
  --dry-run        Show what would be converted
`);
}

function parseArgs(argv) {
  const options = { ...defaults };
  let folder = null;

  for (const arg of argv) {
    if (arg === '--help' || arg === '-h') {
      printUsage();
      process.exit(0);
    }

    if (!arg.startsWith('--')) {
      folder = path.resolve(process.cwd(), arg);
      continue;
    }

    const [rawKey, rawValue] = arg.slice(2).split('=');
    const key = rawKey.replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());

    if (key === 'dryRun' || key === 'keepPng' || key === 'overwrite') {
      options[key] = true;
      continue;
    }

    if (!(key in options)) {
      throw new Error(`Unknown option: ${arg}`);
    }

    options[key] = rawValue;
  }

  options.format = String(options.format).toLowerCase();
  if (!outputExtensions[options.format]) {
    throw new Error(`Unsupported format "${options.format}". Use jpeg, webp, or avif.`);
  }

  options.quality = parseIntegerOption('quality', options.quality, 1, 100);
  options.maxWidth = parseIntegerOption('max-width', options.maxWidth, 1);
  options.maxHeight = parseIntegerOption('max-height', options.maxHeight, 1);

  return { folder, options };
}

function parseIntegerOption(name, value, min, max = Number.MAX_SAFE_INTEGER) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isInteger(parsed) || parsed < min || parsed > max) {
    throw new Error(`--${name} must be an integer from ${min} to ${max}.`);
  }
  return parsed;
}

async function getPngFiles(folder) {
  const entries = await fs.readdir(folder, { withFileTypes: true });
  return entries
    .filter((entry) => entry.isFile() && entry.name.toLowerCase().endsWith('.png'))
    .map((entry) => path.join(folder, entry.name))
    .sort((a, b) => a.localeCompare(b));
}

async function getDefaultImageFolders() {
  const entries = await fs.readdir(repoRoot, { withFileTypes: true });
  const lessonImageFolders = entries
    .filter((entry) => entry.isDirectory() && entry.name.endsWith('-animation'))
    .map((entry) => path.join(repoRoot, entry.name, 'images'));
  const existingFolders = [];

  for (const folder of lessonImageFolders) {
    try {
      const stats = await fs.stat(folder);
      if (stats.isDirectory()) {
        existingFolders.push(folder);
      }
    } catch {
      // Lesson has no images folder. Skip it.
    }
  }

  return existingFolders.sort((a, b) => a.localeCompare(b));
}

function formatBytes(bytes) {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
}

async function convertPng(file, options) {
  const sourceStats = await fs.stat(file);
  const outputPath = path.join(
    path.dirname(file),
    `${path.basename(file, path.extname(file))}${outputExtensions[options.format]}`,
  );
  const tempPath = `${outputPath}.tmp`;

  if (!options.overwrite) {
    try {
      await fs.access(outputPath);
      return {
        status: 'skipped',
        file,
        outputPath,
        reason: 'output exists; pass --overwrite to replace it',
      };
    } catch {
      // No existing output. Continue.
    }
  }

  if (options.dryRun) {
    return {
      status: 'dry-run',
      file,
      outputPath,
      sourceBytes: sourceStats.size,
    };
  }

  let pipeline = sharp(file)
    .rotate()
    .resize({
      width: options.maxWidth,
      height: options.maxHeight,
      fit: 'inside',
      withoutEnlargement: true,
    });

  if (options.format === 'jpeg') {
    pipeline = pipeline
      .flatten({ background: '#ffffff' })
      .jpeg({ quality: options.quality, mozjpeg: true, progressive: true });
  } else if (options.format === 'webp') {
    pipeline = pipeline.webp({ quality: options.quality, effort: 4 });
  } else if (options.format === 'avif') {
    pipeline = pipeline.avif({ quality: options.quality, effort: 4 });
  }

  await pipeline.toFile(tempPath);

  const outputStats = await fs.stat(tempPath);
  if (outputStats.size === 0) {
    await fs.unlink(tempPath);
    throw new Error(`Converted file was empty: ${tempPath}`);
  }

  if (options.overwrite) {
    await fs.rm(outputPath, { force: true });
  }

  await fs.rename(tempPath, outputPath);

  if (!options.keepPng) {
    await fs.unlink(file);
  }

  return {
    status: 'converted',
    file,
    outputPath,
    sourceBytes: sourceStats.size,
    outputBytes: outputStats.size,
    removedSource: !options.keepPng,
  };
}

async function main() {
  const { folder, options } = parseArgs(process.argv.slice(2));
  const folders = folder ? [folder] : await getDefaultImageFolders();
  const pngFilesByFolder = [];

  for (const imageFolder of folders) {
    pngFilesByFolder.push({
      folder: imageFolder,
      files: await getPngFiles(imageFolder),
    });
  }

  const pngFiles = pngFilesByFolder.flatMap((entry) => entry.files);

  if (pngFiles.length === 0) {
    const folderDescription = folder ? folder : 'lesson image folders';
    console.log(`No PNG files found in ${folderDescription}`);
    return;
  }

  console.log(`Optimizing ${pngFiles.length} PNG file(s) across ${pngFilesByFolder.length} folder(s)`);

  const results = [];
  for (const file of pngFiles) {
    results.push(await convertPng(file, options));
  }

  for (const result of results) {
    const sourceName = path.basename(result.file);
    const outputName = path.basename(result.outputPath);

    if (result.status === 'converted') {
      const reduction = 100 - (result.outputBytes / result.sourceBytes) * 100;
      const sourceRemoved = result.removedSource ? '; removed PNG' : '; kept PNG';
      console.log(
        `${sourceName} -> ${outputName}: ${formatBytes(result.sourceBytes)} to ${formatBytes(
          result.outputBytes,
        )} (${reduction.toFixed(1)}% smaller${sourceRemoved})`,
      );
    } else if (result.status === 'dry-run') {
      console.log(`${sourceName} -> ${outputName}: ${formatBytes(result.sourceBytes)} source`);
    } else {
      console.log(`${sourceName}: skipped (${result.reason})`);
    }
  }
}

main().catch((error) => {
  console.error(error.message);
  process.exit(1);
});
