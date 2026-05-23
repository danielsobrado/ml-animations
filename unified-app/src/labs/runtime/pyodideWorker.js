const PYODIDE_URL = 'https://cdn.jsdelivr.net/pyodide/v0.29.4/full/pyodide.mjs';
const MAX_STREAM_CHARS = 12000;

let pyodideReadyPromise;

async function getPyodide() {
  if (!pyodideReadyPromise) {
    pyodideReadyPromise = import(/* @vite-ignore */ PYODIDE_URL).then(({ loadPyodide }) => loadPyodide());
  }

  return pyodideReadyPromise;
}

function appendLimited(current, text) {
  if (current.length >= MAX_STREAM_CHARS) return current;
  const remaining = MAX_STREAM_CHARS - current.length;
  const next = current + text.slice(0, remaining);
  return text.length > remaining ? `${next}\n[output truncated]` : next;
}

self.onmessage = async (event) => {
  const { id, python } = event.data;
  let stdout = '';
  let stderr = '';

  try {
    const pyodide = await getPyodide();

    pyodide.setStdout({
      batched: (text) => {
        stdout = appendLimited(stdout, text);
      },
    });

    pyodide.setStderr({
      batched: (text) => {
        stderr = appendLimited(stderr, text);
      },
    });

    await pyodide.loadPackagesFromImports(python);
    await pyodide.runPythonAsync(python);

    self.postMessage({
      id,
      stdout,
      stderr,
      error: null,
    });
  } catch (error) {
    self.postMessage({
      id,
      stdout,
      stderr,
      error: error?.message || String(error),
    });
  }
};
