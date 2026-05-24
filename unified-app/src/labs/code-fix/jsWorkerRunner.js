let nextId = 1;

export function runJavaScriptExercise({ userCode, testCode, timeoutMs = 1200 }) {
  const id = nextId++;

  return new Promise((resolve) => {
    const worker = new Worker(
      new URL('./jsEvalWorker.js', import.meta.url),
      { type: 'module' }
    );

    const timeout = window.setTimeout(() => {
      worker.terminate();
      resolve({
        ok: false,
        results: [],
        error: 'Execution timed out. Check for an infinite loop.',
      });
    }, timeoutMs);

    worker.onmessage = (event) => {
      if (event.data.id !== id) return;

      window.clearTimeout(timeout);
      worker.terminate();

      resolve({
        ok: event.data.ok,
        results: event.data.results || [],
        error: event.data.error,
      });
    };

    worker.postMessage({ id, userCode, testCode });
  });
}
