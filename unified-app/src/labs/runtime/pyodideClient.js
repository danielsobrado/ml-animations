let worker;
let nextId = 1;
const pending = new Map();

function getWorker() {
  if (!worker) {
    worker = new Worker(new URL('./pyodideWorker.js', import.meta.url), { type: 'module' });

    worker.onmessage = (event) => {
      const { id, ...payload } = event.data;
      const resolver = pending.get(id);
      if (!resolver) return;

      pending.delete(id);
      resolver(payload);
    };

    worker.onerror = (event) => {
      pending.forEach((resolve) => {
        resolve({
          stdout: '',
          stderr: '',
          error: event.message || 'Python worker failed.',
        });
      });
      pending.clear();
    };
  }

  return worker;
}

export function runPythonInBrowser(python) {
  const id = nextId;
  nextId += 1;

  return new Promise((resolve) => {
    pending.set(id, resolve);
    getWorker().postMessage({ id, python });
  });
}
