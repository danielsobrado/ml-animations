self.onmessage = (event) => {
  const { id, userCode, testCode } = event.data;

  try {
    const run = new Function(`
      "use strict";
      ${userCode}

      return (function runTests() {
        ${testCode}
      })();
    `);

    const results = run();

    self.postMessage({
      id,
      ok: true,
      results,
      error: null,
    });
  } catch (error) {
    self.postMessage({
      id,
      ok: false,
      results: [],
      error: error?.message || String(error),
    });
  }
};
