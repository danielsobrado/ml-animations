export const POINTS = Object.freeze([
  [0.8, 1.0], [1.1, 1.4], [1.4, 0.9], [1.7, 1.3], [0.9, 1.8],
  [4.1, 1.0], [4.6, 1.3], [4.9, 0.8], [5.2, 1.5], [4.4, 1.8],
  [2.5, 4.4], [2.9, 4.9], [3.3, 4.3], [3.6, 4.8], [2.7, 5.3],
  [5.3, 4.7], [5.7, 5.1], [6.0, 4.4], [6.4, 5.0],
]);

export const INITIAL_CENTROIDS = Object.freeze([
  [1, 1],
  [5.5, 1.1],
  [3.1, 5],
  [6, 4.8],
]);

export const COLORS = Object.freeze(['#2563eb', '#dc2626', '#16a34a', '#9333ea']);

export function distance(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

export function assign(points, centroids) {
  return points.map((point) => {
    const distances = centroids.map((centroid) => distance(point, centroid));
    return distances.indexOf(Math.min(...distances));
  });
}

export function updateCentroids(points, assignments, centroids) {
  return centroids.map((centroid, cluster) => {
    const members = points.filter((_, index) => assignments[index] === cluster);
    if (!members.length) return centroid;
    return [
      members.reduce((sum, point) => sum + point[0], 0) / members.length,
      members.reduce((sum, point) => sum + point[1], 0) / members.length,
    ];
  });
}

export function inertia(points, assignments, centroids) {
  return points.reduce((sum, point, index) => sum + distance(point, centroids[assignments[index]]) ** 2, 0);
}

export function runKMeans(k, iterations) {
  let centroids = INITIAL_CENTROIDS.slice(0, k);
  let assignments = assign(POINTS, centroids);

  for (let step = 0; step < iterations; step += 1) {
    centroids = updateCentroids(POINTS, assignments, centroids);
    assignments = assign(POINTS, centroids);
  }

  return {
    centroids,
    assignments,
    inertia: inertia(POINTS, assignments, centroids),
  };
}

export function toScreen([x, y]) {
  return [40 + x * 46, 330 - y * 48];
}
