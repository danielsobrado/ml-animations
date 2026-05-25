const lessonImageModules = import.meta.glob('../../../*-animation/images/*.{jpg,jpeg,webp,avif,png}', {
  eager: true,
  import: 'default',
  query: '?url',
});

function parseLessonImagePath(filePath) {
  const normalizedPath = filePath.replace(/\\/g, '/');
  const match = normalizedPath.match(/\/([^/]+)-animation\/images\/([^/]+)$/);

  if (!match) return null;

  return {
    lessonId: match[1],
    filename: match[2],
  };
}

function withoutExtension(filename) {
  return filename.replace(/\.[^.]+$/, '');
}

function titleFromSlug(slug) {
  return slug
    .split('-')
    .filter(Boolean)
    .map((word) => word.slice(0, 1).toUpperCase() + word.slice(1))
    .join(' ');
}

function imageSortKey(image) {
  return image.filename.toLowerCase();
}

const lessonImages = Object.entries(lessonImageModules)
  .flatMap(([filePath, src]) => {
    const parsedPath = parseLessonImagePath(filePath);
    if (!parsedPath) return [];

    const { lessonId, filename } = parsedPath;
    const slug = withoutExtension(filename);

    return [{
      filename,
      lessonId,
      slug,
      src,
      title: titleFromSlug(slug),
    }];
  })
  .sort((a, b) => imageSortKey(a).localeCompare(imageSortKey(b)));

export function getLessonImages(lessonId, lessonName) {
  const matches = lessonImages.filter((image) => image.lessonId === lessonId);

  return matches
    .map((image, index) => ({
      ...image,
      alt: `${lessonName} lesson visual${matches.length > 1 ? ` ${index + 1}` : ''}: ${image.title}`,
    }));
}
