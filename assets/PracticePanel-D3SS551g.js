import{a as v,j as e,k as y}from"./react-vendor-Cdu38Wyn.js";import{F as G,I as U,aZ as F,aQ as X,ar as W,$ as Q,_ as H}from"./icons-C7miCxLM.js";let Y=1;function $({userCode:n,testCode:u,timeoutMs:c=1200}){const r=Y++;return new Promise(a=>{const s=new Worker(new URL("/ml-animations/assets/jsEvalWorker-BiN8n73y.js",import.meta.url),{type:"module"}),w=window.setTimeout(()=>{s.terminate(),a({ok:!1,results:[],error:"Execution timed out. Check for an infinite loop."})},c);s.onmessage=f=>{f.data.id===r&&(window.clearTimeout(w),s.terminate(),a({ok:f.data.ok,results:f.data.results||[],error:f.data.error}))},s.postMessage({id:r,userCode:n,testCode:u})})}function J(n,u){return u?"error":n!=null&&n.length?n.every(c=>c.passed)?"passed":"failed":"idle"}function I(n){return typeof n=="string"?n:JSON.stringify(n)}const _=/(\/\/.*|\/\*[\s\S]*?\*\/|(["'`])(?:\\.|(?!\2)[^\\])*\2|\b(?:const|let|var|function|return|if|else|for|while|new|throw|true|false|null|undefined)\b|\b\d+(?:\.\d+)?\b|\b[a-zA-Z_$][\w$]*(?=\s*\())/g;function Z(n){const u=[];let c=0;for(const r of n.matchAll(_)){r.index>c&&u.push(n.slice(c,r.index));const a=r[0];let s="plain";a.startsWith("//")||a.startsWith("/*")?s="comment":/^["'`]/.test(a)?s="string":/^\d/.test(a)?s="number":/^(const|let|var|function|return|if|else|for|while|new|throw|true|false|null|undefined)$/.test(a)?s="keyword":s="call",u.push(e.jsx("span",{className:`ua-code-token-${s}`,children:a},`${r.index}-${a}`)),c=r.index+a.length}return c<n.length&&u.push(n.slice(c)),u}function K({exercises:n}){var E;const[u,c]=v.useState(0),r=n[u],a=v.useRef(null),[s,w]=v.useState(()=>Object.fromEntries(n.map(t=>[t.id,t.starterCode]))),[f,k]=v.useState({}),[A,D]=v.useState({}),[M,B]=v.useState(!1),[S,O]=v.useState(!1),T=s[r.id],o=A[r.id],j=J(o==null?void 0:o.results,o==null?void 0:o.error),C=f[r.id]||0,L=r.hints.slice(0,C),q=!!(o||C>0),z=v.useMemo(()=>{const t=[];return n.forEach((l,h)=>{const m=l.group||"Exercises",p=t[t.length-1];if((p==null?void 0:p.name)===m){p.items.push({exercise:l,index:h});return}t.push({name:m,items:[{exercise:l,index:h}]})}),t},[n]);async function P(){B(!0),O(!1);const t=await $({userCode:T,testCode:r.testCode});D(l=>({...l,[r.id]:t})),B(!1)}function i(){w(t=>({...t,[r.id]:r.starterCode})),D(t=>({...t,[r.id]:null})),k(t=>({...t,[r.id]:0})),O(!1)}function d(){k(t=>({...t,[r.id]:Math.min(r.hints.length,C+1)}))}function g(){w(t=>({...t,[r.id]:r.solution})),O(!1)}function x(t){a.current&&(a.current.scrollTop=t.currentTarget.scrollTop,a.current.scrollLeft=t.currentTarget.scrollLeft)}const N=Object.values(A).filter(t=>{var l;return((l=t==null?void 0:t.results)==null?void 0:l.length)&&t.results.every(h=>h.passed)}).length;return e.jsxs("section",{className:"ua-codefix-lab",children:[e.jsxs("div",{className:"ua-codefix-head",children:[e.jsx("span",{children:"Code Completion-style lab"}),e.jsx("h2",{children:"Fix the TODOs, run the tests"}),e.jsx("p",{children:"Each exercise is almost complete. Change the smallest piece of code needed to make the tests pass."})]}),e.jsx("div",{className:"ua-codefix-progress",children:z.map(t=>{const l=t.items.filter(({exercise:h})=>{var p;const m=A[h.id];return((p=m==null?void 0:m.results)==null?void 0:p.length)&&m.results.every(R=>R.passed)}).length;return e.jsxs("div",{className:"ua-codefix-progress-group",children:[e.jsxs("div",{className:"ua-codefix-progress-label",children:[e.jsx("strong",{children:t.name}),e.jsxs("span",{children:[l,"/",t.items.length]})]}),e.jsx("div",{className:"ua-codefix-progress-steps",children:t.items.map(({exercise:h,index:m})=>{const p=A[h.id],R=J(p==null?void 0:p.results,p==null?void 0:p.error),V=R==="passed"?G:U;return e.jsxs("button",{type:"button",onClick:()=>{c(m),O(!1)},className:`ua-codefix-step ${m===u?"active":""} ${R}`,children:[e.jsx(V,{size:15,"aria-hidden":"true"}),e.jsxs("span",{children:[h.stepLabel||`${m+1}.`," ",h.title]})]},h.id)})})]},t.name)})}),e.jsxs("div",{className:"ua-codefix-grid",children:[e.jsxs("article",{className:"ua-codefix-card ua-codefix-instructions",children:[e.jsx("span",{children:r.difficulty}),e.jsx("h3",{children:r.title}),e.jsx("p",{children:r.objective}),e.jsxs("div",{className:"ua-codefix-concept",children:[e.jsx("strong",{children:"Concept"}),e.jsx("p",{children:r.concept})]}),e.jsxs("div",{className:"ua-codefix-explanation",children:[e.jsx("strong",{children:"After you pass"}),e.jsx("p",{children:r.explanation})]})]}),e.jsxs("article",{className:"ua-codefix-card ua-codefix-editor-card",children:[e.jsxs("div",{className:"ua-codefix-card-head",children:[e.jsxs("div",{children:[e.jsx("span",{children:"Editor"}),e.jsx("h3",{children:"Complete the TODO"})]}),e.jsxs("button",{type:"button",onClick:i,children:[e.jsx(F,{size:14,"aria-hidden":"true"}),"Reset"]})]}),e.jsxs("div",{className:"ua-codefix-editor-shell",children:[e.jsx("pre",{className:"ua-codefix-highlight","aria-hidden":"true",ref:a,children:Z(T)}),e.jsx("textarea",{className:"ua-codefix-editor",value:T,spellCheck:!1,"aria-label":`${r.title} code editor`,onScroll:x,onChange:t=>w(l=>({...l,[r.id]:t.target.value}))})]}),e.jsxs("div",{className:"ua-codefix-actions",children:[e.jsxs("button",{type:"button",onClick:P,disabled:M,children:[e.jsx(X,{size:15,"aria-hidden":"true"}),M?"Running...":"Run tests"]}),e.jsxs("button",{type:"button",onClick:d,disabled:C>=r.hints.length,children:[e.jsx(W,{size:15,"aria-hidden":"true"}),C===0?"Show hint":"Next hint"]}),e.jsxs("button",{type:"button",onClick:()=>O(t=>!t),disabled:!q,title:q?void 0:"Run tests or use a hint before revealing the solution.",children:[S?e.jsx(Q,{size:15,"aria-hidden":"true"}):e.jsx(H,{size:15,"aria-hidden":"true"}),S?"Hide solution":q?"See solution":"Try first"]})]})]}),e.jsxs("article",{className:"ua-codefix-card ua-codefix-feedback",children:[e.jsx("span",{children:"Checks"}),e.jsxs("h3",{children:[j==="passed"&&"All tests passed",j==="failed"&&"Keep going",j==="error"&&"Code error",j==="idle"&&"Run tests to begin"]}),(o==null?void 0:o.error)&&e.jsx("pre",{className:"ua-codefix-error",children:o.error}),((E=o==null?void 0:o.results)==null?void 0:E.length)>0?e.jsx("ul",{className:"ua-codefix-checks",children:o.results.map(t=>e.jsxs("li",{className:t.passed?"passed":"failed",children:[e.jsxs("strong",{children:[t.passed?"Pass":"Fail",": ",t.name]}),!t.passed&&e.jsxs("small",{children:["Expected ",I(t.expected),", got ",I(t.actual)]})]},t.name))}):e.jsx("p",{className:"ua-codefix-empty",children:"Run the tests. If one fails, use the smallest hint that helps."}),L.length>0&&e.jsxs("div",{className:"ua-codefix-hints",children:[e.jsx("strong",{children:"Hints"}),L.map((t,l)=>t.includes(`
`)?e.jsxs("div",{className:"ua-codefix-hint",children:[e.jsxs("b",{children:["Hint ",l+1,":"]}),e.jsx("pre",{className:"ua-codefix-hint-code",children:t})]},t):e.jsxs("p",{children:[e.jsxs("b",{children:["Hint ",l+1,":"]})," ",t]},t))]}),S&&e.jsxs("div",{className:"ua-codefix-solution",children:[e.jsx("strong",{children:"Solution"}),e.jsx("pre",{children:r.solution}),e.jsx("button",{type:"button",onClick:g,children:"Apply solution to editor"})]})]})]}),e.jsxs("div",{className:"ua-codefix-footer",children:[e.jsxs("strong",{children:[N," / ",n.length]}),e.jsx("span",{children:"exercises passed"})]})]})}const ee=[{id:"dot-product-first-pair",stepLabel:"1.1",group:"Dot product",title:"First matching pair",concept:"A dot product starts by multiplying entries with the same index.",objective:"Replace one number with the first pair product.",difficulty:"warmup",starterCode:`function firstPairProduct(a, b) {
  // TODO: replace 0 with the product of the first entries.
  return 0;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({
    name,
    actual,
    expected,
    passed: Object.is(actual, expected),
  });
}

check('first pair in [1, 2] dot [3, 4]', firstPairProduct([1, 2], [3, 4]), 3);
check('first pair in [0, 5] dot [10, 2]', firstPairProduct([0, 5], [10, 2]), 0);
check('first pair in [-1, 2] dot [3, 5]', firstPairProduct([-1, 2], [3, 5]), -3);

return results;`,hints:["Use index 0 for the first entry of each vector.","The first pair product is a[0] times b[0].","return a[0] * b[0];"],solution:`function firstPairProduct(a, b) {
  return a[0] * b[0];
}`,explanation:"The first contribution to a dot product comes from multiplying the two index-0 entries."},{id:"dot-product-two-pairs",stepLabel:"1.2",group:"Dot product",title:"Add two pair products",concept:"A two-entry dot product adds the first pair product and the second pair product.",objective:"Replace one expression with the missing second pair product.",difficulty:"warmup",starterCode:`function dot2(a, b) {
  const first = a[0] * b[0];
  const second = 0; // TODO: replace 0.

  return first + second;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({
    name,
    actual,
    expected,
    passed: Object.is(actual, expected),
  });
}

check('dot2([1, 2], [3, 4])', dot2([1, 2], [3, 4]), 11);
check('dot2([0, 5], [10, 2])', dot2([0, 5], [10, 2]), 10);
check('dot2([-1, 2], [3, 5])', dot2([-1, 2], [3, 5]), 7);

return results;`,hints:["The second pair uses index 1 in both arrays.","Keep the existing return line. Only fix the value assigned to second.","const second = a[1] * b[1];"],solution:`function dot2(a, b) {
  const first = a[0] * b[0];
  const second = a[1] * b[1];

  return first + second;
}`,explanation:"A two-entry dot product is a[0] * b[0] plus a[1] * b[1]."},{id:"dot-product-loop-update",stepLabel:"1.3",group:"Dot product",title:"Loop over every pair",concept:"The loop repeats the same pair-product rule for vectors of any length.",objective:"Complete the one accumulator update inside the loop.",difficulty:"core",starterCode:`function dot(a, b) {
  let total = 0;

  for (let i = 0; i < a.length; i++) {
    // TODO: replace 0 with the current pair product.
    total += 0;
  }

  return total;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({
    name,
    actual,
    expected,
    passed: Object.is(actual, expected),
  });
}

check('dot([1, 2], [3, 4])', dot([1, 2], [3, 4]), 11);
check('dot([0, 5], [10, 2])', dot([0, 5], [10, 2]), 10);
check('dot([-1, 2], [3, 5])', dot([-1, 2], [3, 5]), 7);
check('dot([2, 2, 2], [1, 2, 3])', dot([2, 2, 2], [1, 2, 3]), 12);

return results;`,hints:["Inside the loop, i points to the current matching pair.","Add a[i] times b[i] into total.","total += a[i] * b[i];"],solution:`function dot(a, b) {
  let total = 0;

  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }

  return total;
}`,explanation:"The loop version is the same rule as dot2, repeated until every matching pair has contributed."},{id:"matrix-cell-one-term",stepLabel:"2.1",group:"Matrix cell",title:"One cell, first term",concept:"One matrix-product cell begins with A[row][0] times B[0][col].",objective:"Replace one expression with the first term of a row-column dot product.",difficulty:"core",starterCode:`function firstCellTerm(A, B, row, col) {
  // TODO: replace 0 with the first row-column product.
  return 0;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({
    name,
    actual,
    expected,
    passed: Object.is(actual, expected),
  });
}

const A = [
  [1, 2],
  [3, 1],
];

const B = [
  [2, 1, 3],
  [1, 4, 2],
];

check('first term for C[0][0]', firstCellTerm(A, B, 0, 0), 2);
check('first term for C[0][2]', firstCellTerm(A, B, 0, 2), 3);
check('first term for C[1][1]', firstCellTerm(A, B, 1, 1), 3);

return results;`,hints:["Use the selected row from A, the selected column from B, and k = 0.","The first term is A[row][0] times B[0][col].","return A[row][0] * B[0][col];"],solution:`function firstCellTerm(A, B, row, col) {
  return A[row][0] * B[0][col];
}`,explanation:"A matrix cell is a dot product; this is the first product in that dot product."},{id:"matrix-cell-loop-update",stepLabel:"2.2",group:"Matrix cell",title:"One cell loop",concept:"The index k moves across a row of A and down a column of B.",objective:"Complete the one accumulator update for a matrix cell.",difficulty:"core",starterCode:`function matrixCell(A, B, row, col) {
  let total = 0;

  for (let k = 0; k < B.length; k++) {
    // TODO: replace 0 with the current row-column product.
    total += 0;
  }

  return total;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({
    name,
    actual,
    expected,
    passed: Object.is(actual, expected),
  });
}

const A = [
  [1, 2],
  [3, 1],
];

const B = [
  [2, 1, 3],
  [1, 4, 2],
];

check('C[0][0]', matrixCell(A, B, 0, 0), 4);
check('C[0][1]', matrixCell(A, B, 0, 1), 9);
check('C[0][2]', matrixCell(A, B, 0, 2), 7);
check('C[1][0]', matrixCell(A, B, 1, 0), 7);
check('C[1][1]', matrixCell(A, B, 1, 1), 7);
check('C[1][2]', matrixCell(A, B, 1, 2), 11);

return results;`,hints:["Use k as the shared index between A and B.","A[row][k] chooses the next entry in the row. B[k][col] chooses the next entry in the column.","total += A[row][k] * B[k][col];"],solution:`function matrixCell(A, B, row, col) {
  let total = 0;

  for (let k = 0; k < B.length; k++) {
    total += A[row][k] * B[k][col];
  }

  return total;
}`,explanation:"The complete cell is the sum of every row-column product for that row and column."},{id:"matrix-multiply-column-count",stepLabel:"3.1",group:"Matrix multiplication",title:"Output column count",concept:"The product A * B has one output column for each column in B.",objective:"Replace one number so the inner loop visits every output column.",difficulty:"challenge",starterCode:`function matrixCell(A, B, row, col) {
  let total = 0;
  for (let k = 0; k < B.length; k++) {
    total += A[row][k] * B[k][col];
  }
  return total;
}

function matmul(A, B) {
  const rows = A.length;
  const cols = 0; // TODO: replace 0 with the number of output columns.

  const C = [];

  for (let i = 0; i < rows; i++) {
    const row = [];

    for (let j = 0; j < cols; j++) {
      row.push(matrixCell(A, B, i, j));
    }

    C.push(row);
  }

  return C;
}`,testCode:`const results = [];

function sameMatrix(actual, expected) {
  return JSON.stringify(actual) === JSON.stringify(expected);
}

function check(name, actual, expected) {
  results.push({
    name,
    actual: JSON.stringify(actual),
    expected: JSON.stringify(expected),
    passed: sameMatrix(actual, expected),
  });
}

check(
  '2x2 times 2x3',
  matmul(
    [[1, 2], [3, 1]],
    [[2, 1, 3], [1, 4, 2]]
  ),
  [[4, 9, 7], [7, 7, 11]]
);

check(
  '2x3 times 3x1',
  matmul(
    [[1, 2, 3], [4, 5, 6]],
    [[1], [2], [3]]
  ),
  [[14], [32]]
);

return results;`,hints:["The number of output columns comes from the first row of B.","B[0] is the first row of B. Its length is the number of columns.","const cols = B[0].length;"],solution:`function matrixCell(A, B, row, col) {
  let total = 0;
  for (let k = 0; k < B.length; k++) {
    total += A[row][k] * B[k][col];
  }
  return total;
}

function matmul(A, B) {
  const rows = A.length;
  const cols = B[0].length;

  const C = [];

  for (let i = 0; i < rows; i++) {
    const row = [];

    for (let j = 0; j < cols; j++) {
      row.push(matrixCell(A, B, i, j));
    }

    C.push(row);
  }

  return C;
}`,explanation:"The shape of A * B is rows of A by columns of B, so the inner loop must run once per column in B."},{id:"matrix-multiply-push-cell",stepLabel:"3.2",group:"Matrix multiplication",title:"Push each computed cell",concept:"The nested loops choose each output position; matrixCell computes the value for that position.",objective:"Replace one argument so each row receives the computed cell value.",difficulty:"challenge",starterCode:`function matrixCell(A, B, row, col) {
  let total = 0;
  for (let k = 0; k < B.length; k++) {
    total += A[row][k] * B[k][col];
  }
  return total;
}

function matmul(A, B) {
  const rows = A.length;
  const cols = B[0].length;

  const C = [];

  for (let i = 0; i < rows; i++) {
    const row = [];

    for (let j = 0; j < cols; j++) {
      // TODO: replace 0 with the computed C[i][j] value.
      row.push(0);
    }

    C.push(row);
  }

  return C;
}`,testCode:`const results = [];

function sameMatrix(actual, expected) {
  return JSON.stringify(actual) === JSON.stringify(expected);
}

function check(name, actual, expected) {
  results.push({
    name,
    actual: JSON.stringify(actual),
    expected: JSON.stringify(expected),
    passed: sameMatrix(actual, expected),
  });
}

check(
  '2x2 times 2x3',
  matmul(
    [[1, 2], [3, 1]],
    [[2, 1, 3], [1, 4, 2]]
  ),
  [[4, 9, 7], [7, 7, 11]]
);

check(
  'identity matrix',
  matmul(
    [[1, 0], [0, 1]],
    [[5, 6], [7, 8]]
  ),
  [[5, 6], [7, 8]]
);

check(
  '2x3 times 3x1',
  matmul(
    [[1, 2, 3], [4, 5, 6]],
    [[1], [2], [3]]
  ),
  [[14], [32]]
);

return results;`,hints:["You already have matrixCell(A, B, i, j). Use it inside the nested loops.","The outer loop chooses output row i. The inner loop chooses output column j.","row.push(matrixCell(A, B, i, j));"],solution:`function matrixCell(A, B, row, col) {
  let total = 0;
  for (let k = 0; k < B.length; k++) {
    total += A[row][k] * B[k][col];
  }
  return total;
}

function matmul(A, B) {
  const rows = A.length;
  const cols = B[0].length;

  const C = [];

  for (let i = 0; i < rows; i++) {
    const row = [];

    for (let j = 0; j < cols; j++) {
      row.push(matrixCell(A, B, i, j));
    }

    C.push(row);
  }

  return C;
}`,explanation:"The full matrix product is the matrixCell rule repeated for every row and every column."},{id:"vector-norm-square-entry",stepLabel:"4.1",group:"Vector norm",title:"Square one entry",concept:"A vector norm starts by squaring each entry so negative and positive values both contribute positively.",objective:"Replace one number with the square of the first entry.",difficulty:"warmup",starterCode:`function firstSquaredEntry(v) {
  // TODO: replace 0 with the square of the first entry.
  return 0;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({
    name,
    actual,
    expected,
    passed: Object.is(actual, expected),
  });
}

check('firstSquaredEntry([3, 4])', firstSquaredEntry([3, 4]), 9);
check('firstSquaredEntry([-5, 2])', firstSquaredEntry([-5, 2]), 25);
check('firstSquaredEntry([0, 7])', firstSquaredEntry([0, 7]), 0);

return results;`,hints:["Use index 0 for the first entry.","Squaring means multiplying the value by itself.","return v[0] * v[0];"],solution:`function firstSquaredEntry(v) {
  return v[0] * v[0];
}`,explanation:"The Euclidean norm is based on squared entries, so negative values still add positive length."},{id:"vector-norm-sum-squares",stepLabel:"4.2",group:"Vector norm",title:"Sum every square",concept:"The squared length of a vector is the sum of its squared entries.",objective:"Complete the accumulator update inside the loop.",difficulty:"core",starterCode:`function sumSquares(v) {
  let total = 0;

  for (let i = 0; i < v.length; i++) {
    // TODO: add the square of the current entry.
    total += 0;
  }

  return total;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({
    name,
    actual,
    expected,
    passed: Object.is(actual, expected),
  });
}

check('sumSquares([3, 4])', sumSquares([3, 4]), 25);
check('sumSquares([1, 2, 2])', sumSquares([1, 2, 2]), 9);
check('sumSquares([-1, -2, -3])', sumSquares([-1, -2, -3]), 14);
check('sumSquares([0, 0, 0])', sumSquares([0, 0, 0]), 0);

return results;`,hints:["Inside the loop, v[i] is the current entry.","Add v[i] times v[i] into total.","total += v[i] * v[i];"],solution:`function sumSquares(v) {
  let total = 0;

  for (let i = 0; i < v.length; i++) {
    total += v[i] * v[i];
  }

  return total;
}`,explanation:"The squared norm is the vector dotted with itself: v dot v."},{id:"vector-norm-full",stepLabel:"4.3",group:"Vector norm",title:"Vector norm",concept:"The Euclidean norm is the square root of the sum of squared entries.",objective:"Replace the final return value with the Euclidean norm.",difficulty:"core",starterCode:`function sumSquares(v) {
  let total = 0;
  for (let i = 0; i < v.length; i++) {
    total += v[i] * v[i];
  }
  return total;
}

function norm(v) {
  const squaredLength = sumSquares(v);

  // TODO: return the square root of squaredLength.
  return squaredLength;
}`,testCode:`const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({
    name,
    actual,
    expected,
    passed: approxEqual(actual, expected),
  });
}

check('norm([3, 4])', norm([3, 4]), 5);
check('norm([1, 2, 2])', norm([1, 2, 2]), 3);
check('norm([0, 0, 0])', norm([0, 0, 0]), 0);
check('norm([-6, 8])', norm([-6, 8]), 10);

return results;`,hints:["JavaScript has Math.sqrt for square roots.","The norm is Math.sqrt(sumSquares(v)).","return Math.sqrt(squaredLength);"],solution:`function sumSquares(v) {
  let total = 0;
  for (let i = 0; i < v.length; i++) {
    total += v[i] * v[i];
  }
  return total;
}

function norm(v) {
  const squaredLength = sumSquares(v);
  return Math.sqrt(squaredLength);
}`,explanation:"The Euclidean norm is the vector length: sqrt(v1^2 + v2^2 + ... + vn^2)."},{id:"cosine-numerator",stepLabel:"5.1",group:"Cosine similarity",title:"Cosine numerator",concept:"Cosine similarity uses the dot product as its numerator.",objective:"Replace one expression with the dot product of u and v.",difficulty:"warmup",starterCode:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function cosineNumerator(u, v) {
  // TODO: return the dot product of u and v.
  return 0;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({
    name,
    actual,
    expected,
    passed: Object.is(actual, expected),
  });
}

check('numerator [1, 0] and [0, 1]', cosineNumerator([1, 0], [0, 1]), 0);
check('numerator [1, 2] and [3, 4]', cosineNumerator([1, 2], [3, 4]), 11);
check('numerator [-1, 2] and [3, 5]', cosineNumerator([-1, 2], [3, 5]), 7);

return results;`,hints:["The helper function dot(a, b) is already available.","Cosine similarity starts with u dot v.","return dot(u, v);"],solution:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function cosineNumerator(u, v) {
  return dot(u, v);
}`,explanation:"The dot product measures alignment, but its raw size also depends on vector lengths."},{id:"cosine-denominator",stepLabel:"5.2",group:"Cosine similarity",title:"Cosine denominator",concept:"Cosine similarity divides by both vector lengths so only direction remains.",objective:"Replace one expression with norm(u) times norm(v).",difficulty:"core",starterCode:`function norm(v) {
  let total = 0;
  for (let i = 0; i < v.length; i++) {
    total += v[i] * v[i];
  }
  return Math.sqrt(total);
}

function cosineDenominator(u, v) {
  // TODO: return the product of the two norms.
  return 1;
}`,testCode:`const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({
    name,
    actual,
    expected,
    passed: approxEqual(actual, expected),
  });
}

check('denominator [3, 4] and [1, 0]', cosineDenominator([3, 4], [1, 0]), 5);
check('denominator [3, 4] and [0, 5]', cosineDenominator([3, 4], [0, 5]), 25);
check('denominator [1, 2, 2] and [2, 0, 0]', cosineDenominator([1, 2, 2], [2, 0, 0]), 6);

return results;`,hints:["The denominator removes the effect of vector length.","Use norm(u) and norm(v).","return norm(u) * norm(v);"],solution:`function norm(v) {
  let total = 0;
  for (let i = 0; i < v.length; i++) {
    total += v[i] * v[i];
  }
  return Math.sqrt(total);
}

function cosineDenominator(u, v) {
  return norm(u) * norm(v);
}`,explanation:"Dividing by both norms turns raw dot product into directional similarity."},{id:"cosine-similarity-full",stepLabel:"5.3",group:"Cosine similarity",title:"Cosine similarity",concept:"Cosine similarity is dot product divided by the product of vector lengths.",objective:"Complete the final cosine formula.",difficulty:"core",starterCode:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function norm(v) {
  return Math.sqrt(dot(v, v));
}

function cosineSimilarity(u, v) {
  const numerator = dot(u, v);
  const denominator = norm(u) * norm(v);

  // TODO: return cosine similarity.
  return 0;
}`,testCode:`const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({
    name,
    actual,
    expected,
    passed: approxEqual(actual, expected),
  });
}

check('same direction', cosineSimilarity([1, 0], [5, 0]), 1);
check('perpendicular', cosineSimilarity([1, 0], [0, 1]), 0);
check('opposite direction', cosineSimilarity([1, 0], [-2, 0]), -1);
check('classic example', cosineSimilarity([1, 2], [3, 4]), 11 / (Math.sqrt(5) * 5));

return results;`,hints:["The numerator and denominator are already computed.","Cosine similarity = numerator / denominator.","return numerator / denominator;"],solution:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function norm(v) {
  return Math.sqrt(dot(v, v));
}

function cosineSimilarity(u, v) {
  const numerator = dot(u, v);
  const denominator = norm(u) * norm(v);
  return numerator / denominator;
}`,explanation:"Cosine similarity compares direction. It equals 1 for same direction, 0 for perpendicular, and -1 for opposite direction."},{id:"transpose-one-entry",stepLabel:"6.1",group:"Transpose",title:"Transpose one entry",concept:"Transposing swaps row and column coordinates.",objective:"Return the transposed value at T[row][col].",difficulty:"warmup",starterCode:`function transposedEntry(A, row, col) {
  // TODO: return the value that appears at T[row][col].
  return 0;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

const A = [
  [1, 2, 3],
  [4, 5, 6],
];

check('T[0][0]', transposedEntry(A, 0, 0), 1);
check('T[1][0]', transposedEntry(A, 1, 0), 2);
check('T[2][1]', transposedEntry(A, 2, 1), 6);

return results;`,hints:["T[row][col] comes from A[col][row].","Transpose swaps the indices.","return A[col][row];"],solution:`function transposedEntry(A, row, col) {
  return A[col][row];
}`,explanation:"Transpose flips a matrix over its diagonal: rows become columns and columns become rows."},{id:"transpose-output-shape",stepLabel:"6.2",group:"Transpose",title:"Transpose shape",concept:"A matrix with m rows and n columns transposes into n rows and m columns.",objective:"Return the shape of the transposed matrix.",difficulty:"warmup",starterCode:`function transposeShape(A) {
  const rows = A.length;
  const cols = A[0].length;

  // TODO: return [transposedRows, transposedCols].
  return [rows, cols];
}`,testCode:`const results = [];

function sameArray(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function check(name, actual, expected) {
  results.push({
    name,
    actual: JSON.stringify(actual),
    expected: JSON.stringify(expected),
    passed: sameArray(actual, expected),
  });
}

check('2x3 becomes 3x2', transposeShape([[1, 2, 3], [4, 5, 6]]), [3, 2]);
check('3x1 becomes 1x3', transposeShape([[1], [2], [3]]), [1, 3]);
check('1x4 becomes 4x1', transposeShape([[1, 2, 3, 4]]), [4, 1]);

return results;`,hints:["The old number of columns becomes the new number of rows.","The old number of rows becomes the new number of columns.","return [cols, rows];"],solution:`function transposeShape(A) {
  const rows = A.length;
  const cols = A[0].length;
  return [cols, rows];
}`,explanation:"Transpose swaps the shape: m x n becomes n x m."},{id:"transpose-full",stepLabel:"6.3",group:"Transpose",title:"Full transpose",concept:"Build each transposed row by reading down one original column.",objective:"Complete the value pushed into each transposed row.",difficulty:"core",starterCode:`function transpose(A) {
  const rows = A.length;
  const cols = A[0].length;
  const T = [];

  for (let j = 0; j < cols; j++) {
    const row = [];

    for (let i = 0; i < rows; i++) {
      // TODO: push the value that belongs at T[j][i].
      row.push(0);
    }

    T.push(row);
  }

  return T;
}`,testCode:`const results = [];

function sameMatrix(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function check(name, actual, expected) {
  results.push({
    name,
    actual: JSON.stringify(actual),
    expected: JSON.stringify(expected),
    passed: sameMatrix(actual, expected),
  });
}

check('2x3 transpose', transpose([[1, 2, 3], [4, 5, 6]]), [[1, 4], [2, 5], [3, 6]]);
check('3x1 transpose', transpose([[1], [2], [3]]), [[1, 2, 3]]);
check('1x3 transpose', transpose([[7, 8, 9]]), [[7], [8], [9]]);

return results;`,hints:["The outer loop j chooses an original column.","The inner loop i moves down the original rows.","row.push(A[i][j]);"],solution:`function transpose(A) {
  const rows = A.length;
  const cols = A[0].length;
  const T = [];

  for (let j = 0; j < cols; j++) {
    const row = [];

    for (let i = 0; i < rows; i++) {
      row.push(A[i][j]);
    }

    T.push(row);
  }

  return T;
}`,explanation:"The j-th row of the transpose is the j-th column of the original matrix."},{id:"matrix-shape-read",stepLabel:"7.1",group:"Shape compatibility",title:"Read matrix shape",concept:"A matrix shape is rows x columns.",objective:"Return [rows, columns] for a matrix.",difficulty:"warmup",starterCode:`function shape(A) {
  const rows = A.length;

  // TODO: replace 0 with the number of columns.
  const cols = 0;

  return [rows, cols];
}`,testCode:`const results = [];

function sameArray(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function check(name, actual, expected) {
  results.push({
    name,
    actual: JSON.stringify(actual),
    expected: JSON.stringify(expected),
    passed: sameArray(actual, expected),
  });
}

check('2x3', shape([[1, 2, 3], [4, 5, 6]]), [2, 3]);
check('3x1', shape([[1], [2], [3]]), [3, 1]);
check('1x2', shape([[9, 8]]), [1, 2]);

return results;`,hints:["Rows are A.length.","Columns are the length of the first row.","const cols = A[0].length;"],solution:`function shape(A) {
  const rows = A.length;
  const cols = A[0].length;
  return [rows, cols];
}`,explanation:"A matrix with 2 rows and 3 columns has shape 2 x 3."},{id:"matrix-shape-can-multiply",stepLabel:"7.2",group:"Shape compatibility",title:"Can these multiply?",concept:"A * B is valid only when columns of A equal rows of B.",objective:"Return true when A and B have compatible shapes.",difficulty:"core",starterCode:`function canMultiply(A, B) {
  const colsA = A[0].length;
  const rowsB = B.length;

  // TODO: return whether the inner dimensions match.
  return false;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('2x3 times 3x2 is valid', canMultiply([[1,2,3],[4,5,6]], [[1,2],[3,4],[5,6]]), true);
check('2x2 times 3x2 is invalid', canMultiply([[1,2],[3,4]], [[1,2],[3,4],[5,6]]), false);
check('1x3 times 3x1 is valid', canMultiply([[1,2,3]], [[1],[2],[3]]), true);
check('3x1 times 3x1 is invalid', canMultiply([[1],[2],[3]], [[1],[2],[3]]), false);

return results;`,hints:["Only the inner dimensions matter.","A is m x n and B is n x p.","return colsA === rowsB;"],solution:`function canMultiply(A, B) {
  const colsA = A[0].length;
  const rowsB = B.length;
  return colsA === rowsB;
}`,explanation:"Matrix multiplication works when each row of A has the same length as each column of B."},{id:"matrix-shape-guard",stepLabel:"7.3",group:"Shape compatibility",title:"Guard matrix multiplication",concept:"Good matrix code checks shape compatibility before computing.",objective:"Throw an error when matrix shapes are incompatible.",difficulty:"challenge",starterCode:`function canMultiply(A, B) {
  return A[0].length === B.length;
}

function matmulShapeCheck(A, B) {
  // TODO: if shapes are incompatible, throw new Error('Incompatible shapes').
  return 'ok';
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

function catchesError(fn) {
  try {
    fn();
    return false;
  } catch (error) {
    return error.message === 'Incompatible shapes';
  }
}

check(
  'valid shape returns ok',
  matmulShapeCheck([[1,2,3]], [[1],[2],[3]]),
  'ok'
);

check(
  'invalid shape throws',
  catchesError(() => matmulShapeCheck([[1,2]], [[1,2], [3,4], [5,6]])),
  true
);

return results;`,hints:["Use canMultiply(A, B).","If canMultiply returns false, throw an Error.",`if (!canMultiply(A, B)) {
  throw new Error('Incompatible shapes');
}`],solution:`function canMultiply(A, B) {
  return A[0].length === B.length;
}

function matmulShapeCheck(A, B) {
  if (!canMultiply(A, B)) {
    throw new Error('Incompatible shapes');
  }

  return 'ok';
}`,explanation:"Shape checking turns a silent wrong computation into a clear mathematical error."},{id:"matrix-vector-one-row",stepLabel:"8.1",group:"Matrix-vector multiplication",title:"One row times vector",concept:"A matrix-vector output entry is one row of the matrix dotted with the vector.",objective:"Compute one output entry.",difficulty:"core",starterCode:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function rowTimesVector(A, x, row) {
  // TODO: return row "row" of A dotted with x.
  return 0;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

const A = [
  [1, 2, 3],
  [4, 5, 6],
];

const x = [1, 2, 3];

check('row 0 times x', rowTimesVector(A, x, 0), 14);
check('row 1 times x', rowTimesVector(A, x, 1), 32);

return results;`,hints:["A[row] gives the selected row.","Use the dot helper.","return dot(A[row], x);"],solution:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function rowTimesVector(A, x, row) {
  return dot(A[row], x);
}`,explanation:"Matrix-vector multiplication applies the dot-product rule once per matrix row."},{id:"matrix-vector-full",stepLabel:"8.2",group:"Matrix-vector multiplication",title:"Matrix-vector multiplication",concept:"A matrix-vector product stacks one dot product per matrix row.",objective:"Push each row dot product into the output vector.",difficulty:"core",starterCode:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function matvec(A, x) {
  const y = [];

  for (let row = 0; row < A.length; row++) {
    // TODO: push the output entry for this row.
    y.push(0);
  }

  return y;
}`,testCode:`const results = [];

function sameArray(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function check(name, actual, expected) {
  results.push({
    name,
    actual: JSON.stringify(actual),
    expected: JSON.stringify(expected),
    passed: sameArray(actual, expected),
  });
}

check('2x3 times 3-vector', matvec([[1,2,3],[4,5,6]], [1,2,3]), [14, 32]);
check('identity times vector', matvec([[1,0],[0,1]], [7, 8]), [7, 8]);
check('zero matrix', matvec([[0,0],[0,0]], [5, 6]), [0, 0]);

return results;`,hints:["For each row, compute dot(A[row], x).","Push the dot product into y.","y.push(dot(A[row], x));"],solution:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function matvec(A, x) {
  const y = [];

  for (let row = 0; row < A.length; row++) {
    y.push(dot(A[row], x));
  }

  return y;
}`,explanation:"Matrix-vector multiplication is the same row-column idea, but the second object has only one column."},{id:"identity-diagonal-check",stepLabel:"9.1",group:"Identity matrix",title:"Diagonal entries",concept:"In an identity matrix, diagonal entries are 1.",objective:"Return whether a row and column index are on the diagonal.",difficulty:"warmup",starterCode:`function isDiagonal(row, col) {
  // TODO: return true when row and col are the same.
  return false;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('0,0 is diagonal', isDiagonal(0, 0), true);
check('1,1 is diagonal', isDiagonal(1, 1), true);
check('0,1 is not diagonal', isDiagonal(0, 1), false);
check('2,0 is not diagonal', isDiagonal(2, 0), false);

return results;`,hints:["A diagonal entry has the same row and column index.","Compare row and col.","return row === col;"],solution:`function isDiagonal(row, col) {
  return row === col;
}`,explanation:"The identity matrix has 1s exactly where row index equals column index."},{id:"identity-entry",stepLabel:"9.2",group:"Identity matrix",title:"Identity entry",concept:"Identity entries are 1 on the diagonal and 0 everywhere else.",objective:"Return the identity matrix value for one position.",difficulty:"warmup",starterCode:`function identityEntry(row, col) {
  // TODO: return 1 on the diagonal, 0 otherwise.
  return 0;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('I[0][0]', identityEntry(0, 0), 1);
check('I[1][1]', identityEntry(1, 1), 1);
check('I[0][1]', identityEntry(0, 1), 0);
check('I[2][0]', identityEntry(2, 0), 0);

return results;`,hints:["Use a conditional expression.","If row === col, return 1. Otherwise return 0.","return row === col ? 1 : 0;"],solution:`function identityEntry(row, col) {
  return row === col ? 1 : 0;
}`,explanation:"The identity matrix leaves vectors unchanged because it copies each coordinate onto itself."},{id:"identity-full",stepLabel:"9.3",group:"Identity matrix",title:"Build identity matrix",concept:"An n x n identity matrix has 1s on the diagonal and 0s elsewhere.",objective:"Push the correct entry into each row.",difficulty:"core",starterCode:`function identity(n) {
  const I = [];

  for (let row = 0; row < n; row++) {
    const values = [];

    for (let col = 0; col < n; col++) {
      // TODO: push the identity value for this row and column.
      values.push(0);
    }

    I.push(values);
  }

  return I;
}`,testCode:`const results = [];

function sameMatrix(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function check(name, actual, expected) {
  results.push({
    name,
    actual: JSON.stringify(actual),
    expected: JSON.stringify(expected),
    passed: sameMatrix(actual, expected),
  });
}

check('identity(1)', identity(1), [[1]]);
check('identity(2)', identity(2), [[1,0],[0,1]]);
check('identity(3)', identity(3), [[1,0,0],[0,1,0],[0,0,1]]);

return results;`,hints:["Use row === col to detect the diagonal.","Push 1 on the diagonal and 0 elsewhere.","values.push(row === col ? 1 : 0);"],solution:`function identity(n) {
  const I = [];

  for (let row = 0; row < n; row++) {
    const values = [];

    for (let col = 0; col < n; col++) {
      values.push(row === col ? 1 : 0);
    }

    I.push(values);
  }

  return I;
}`,explanation:"The identity matrix is the multiplicative do-nothing matrix: I * x = x."},{id:"projection-unit-scale",stepLabel:"10.1",group:"Projection",title:"Projection scale onto unit vector",concept:"When the basis vector is unit length, the projection scale is just a dot product.",objective:"Return the dot product of v and unitBasis.",difficulty:"core",starterCode:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function projectionScaleUnit(v, unitBasis) {
  // TODO: return the scale of v along unitBasis.
  return 0;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('onto x-axis', projectionScaleUnit([3, 4], [1, 0]), 3);
check('onto y-axis', projectionScaleUnit([3, 4], [0, 1]), 4);
check('negative direction', projectionScaleUnit([-2, 5], [1, 0]), -2);

return results;`,hints:["A unit basis vector already has length 1.","The amount of v along that direction is v dot unitBasis.","return dot(v, unitBasis);"],solution:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function projectionScaleUnit(v, unitBasis) {
  return dot(v, unitBasis);
}`,explanation:"For a unit direction u, the projection scale is v dot u."},{id:"projection-unit-vector",stepLabel:"10.2",group:"Projection",title:"Projection vector onto unit direction",concept:"The projection vector is scale times the unit direction.",objective:"Replace the TODO with scale times the current basis coordinate.",difficulty:"core",starterCode:`function projectOntoUnit(v, unitBasis) {
  let scale = 0;

  for (let i = 0; i < v.length; i++) {
    scale += v[i] * unitBasis[i];
  }

  const projection = [];

  for (let i = 0; i < unitBasis.length; i++) {
    // TODO: push scale times this basis coordinate.
    projection.push(0);
  }

  return projection;
}`,testCode:`const results = [];

function sameArray(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function check(name, actual, expected) {
  results.push({
    name,
    actual: JSON.stringify(actual),
    expected: JSON.stringify(expected),
    passed: sameArray(actual, expected),
  });
}

check('project [3,4] onto x-axis', projectOntoUnit([3,4], [1,0]), [3,0]);
check('project [3,4] onto y-axis', projectOntoUnit([3,4], [0,1]), [0,4]);
check('project [-2,5] onto x-axis', projectOntoUnit([-2,5], [1,0]), [-2,0]);

return results;`,hints:["The scale is already computed.","Each projection coordinate is scale * unitBasis[i].","projection.push(scale * unitBasis[i]);"],solution:`function projectOntoUnit(v, unitBasis) {
  let scale = 0;

  for (let i = 0; i < v.length; i++) {
    scale += v[i] * unitBasis[i];
  }

  const projection = [];

  for (let i = 0; i < unitBasis.length; i++) {
    projection.push(scale * unitBasis[i]);
  }

  return projection;
}`,explanation:"Projection keeps only the part of v that lies along the chosen unit direction."},{id:"projection-nonunit-vector",stepLabel:"10.3",group:"Projection",title:"Projection onto any vector",concept:"For a non-unit basis b, divide by b dot b before multiplying by b.",objective:"Complete the projection scale formula.",difficulty:"challenge",starterCode:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function projectOnto(v, b) {
  // TODO: replace 0 with the correct projection scale.
  const scale = 0;

  return b.map((entry) => scale * entry);
}`,testCode:`const results = [];

function approxArray(a, b, tolerance = 1e-9) {
  return a.length === b.length && a.every((value, index) => Math.abs(value - b[index]) <= tolerance);
}

function check(name, actual, expected) {
  results.push({
    name,
    actual: JSON.stringify(actual),
    expected: JSON.stringify(expected),
    passed: approxArray(actual, expected),
  });
}

check('project [3,4] onto [2,0]', projectOnto([3,4], [2,0]), [3,0]);
check('project [3,4] onto [0,2]', projectOnto([3,4], [0,2]), [0,4]);
check('project [2,2] onto [1,1]', projectOnto([2,2], [1,1]), [2,2]);
check('project [2,0] onto [1,1]', projectOnto([2,0], [1,1]), [1,1]);

return results;`,hints:["For non-unit b, the scale is (v dot b) / (b dot b).","The denominator corrects for the length of b.","const scale = dot(v, b) / dot(b, b);"],solution:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function projectOnto(v, b) {
  const scale = dot(v, b) / dot(b, b);
  return b.map((entry) => scale * entry);
}`,explanation:"Projection onto b is ((v dot b) / (b dot b)) * b. The denominator handles non-unit basis vectors."},{id:"least-squares-prediction",stepLabel:"11.1",group:"Least-squares residual",title:"Prediction Ax",concept:"Least squares compares the target vector b with the prediction Ax.",objective:"Use matrix-vector multiplication to compute Ax.",difficulty:"core",starterCode:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function matvec(A, x) {
  const y = [];

  for (let row = 0; row < A.length; row++) {
    y.push(dot(A[row], x));
  }

  return y;
}

function prediction(A, x) {
  // TODO: return Ax.
  return [];
}`,testCode:`const results = [];

function sameArray(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function check(name, actual, expected) {
  results.push({
    name,
    actual: JSON.stringify(actual),
    expected: JSON.stringify(expected),
    passed: sameArray(actual, expected),
  });
}

check('prediction 2x2', prediction([[1,2],[3,4]], [1,1]), [3,7]);
check('prediction 2x3', prediction([[1,2,3],[4,5,6]], [1,2,3]), [14,32]);

return results;`,hints:["The helper matvec(A, x) already computes Ax.","prediction should return the model output.","return matvec(A, x);"],solution:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function matvec(A, x) {
  const y = [];

  for (let row = 0; row < A.length; row++) {
    y.push(dot(A[row], x));
  }

  return y;
}

function prediction(A, x) {
  return matvec(A, x);
}`,explanation:"In least squares, Ax is the model output that tries to match b using the columns of A."},{id:"least-squares-residual-vector",stepLabel:"11.2",group:"Least-squares residual",title:"Residual vector",concept:"The residual is target minus prediction: r = b - Ax.",objective:"Complete the residual coordinate formula.",difficulty:"core",starterCode:`function residualVector(b, yHat) {
  const residual = [];

  for (let i = 0; i < b.length; i++) {
    // TODO: push target minus prediction.
    residual.push(0);
  }

  return residual;
}`,testCode:`const results = [];

function sameArray(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function check(name, actual, expected) {
  results.push({
    name,
    actual: JSON.stringify(actual),
    expected: JSON.stringify(expected),
    passed: sameArray(actual, expected),
  });
}

check('residual [5, 10] - [3, 7]', residualVector([5,10], [3,7]), [2,3]);
check('zero residual', residualVector([1,2,3], [1,2,3]), [0,0,0]);
check('negative residual', residualVector([1,1], [4,0]), [-3,1]);

return results;`,hints:["Residual means what is left over after prediction.","Use b[i] - yHat[i].","residual.push(b[i] - yHat[i]);"],solution:`function residualVector(b, yHat) {
  const residual = [];

  for (let i = 0; i < b.length; i++) {
    residual.push(b[i] - yHat[i]);
  }

  return residual;
}`,explanation:"The residual vector points from the prediction Ax to the observed target b."},{id:"least-squares-residual-sum-squares",stepLabel:"11.3",group:"Least-squares residual",title:"Residual sum of squares",concept:"Least squares minimizes the squared length of the residual vector.",objective:"Complete the squared-residual accumulator.",difficulty:"challenge",starterCode:`function residualSumSquares(b, yHat) {
  let total = 0;

  for (let i = 0; i < b.length; i++) {
    const residual = b[i] - yHat[i];

    // TODO: add the squared residual.
    total += 0;
  }

  return total;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('RSS [5,10] vs [3,7]', residualSumSquares([5,10], [3,7]), 13);
check('RSS zero', residualSumSquares([1,2,3], [1,2,3]), 0);
check('RSS negative residuals', residualSumSquares([1,1], [4,0]), 10);

return results;`,hints:["Squared residual means residual times residual.","Add residual * residual into total.","total += residual * residual;"],solution:`function residualSumSquares(b, yHat) {
  let total = 0;

  for (let i = 0; i < b.length; i++) {
    const residual = b[i] - yHat[i];
    total += residual * residual;
  }

  return total;
}`,explanation:"Least squares minimizes RSS, the squared length of the error vector b - Ax."},{id:"orthogonality-dot-zero",stepLabel:"12.1",group:"Orthogonality",title:"Zero dot product",concept:"Two vectors are orthogonal when their dot product is zero.",objective:"Complete the boolean check for zero dot product.",difficulty:"core",starterCode:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function hasZeroDot(a, b) {
  // TODO: return true when the dot product is exactly zero.
  return false;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('standard basis vectors', hasZeroDot([1, 0], [0, 1]), true);
check('non-orthogonal vectors', hasZeroDot([1, 2], [3, 4]), false);
check('integer orthogonal pair', hasZeroDot([2, -1], [1, 2]), true);

return results;`,hints:["Orthogonal means dot(a, b) equals zero.","Use the dot helper.","return dot(a, b) === 0;"],solution:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function hasZeroDot(a, b) {
  return dot(a, b) === 0;
}`,explanation:"Orthogonality is the geometric meaning of a zero dot product."},{id:"orthogonality-tolerance",stepLabel:"12.2",group:"Orthogonality",title:"Orthogonal with tolerance",concept:"Floating-point computations often need a tolerance instead of exact equality.",objective:"Check whether the absolute dot product is small enough.",difficulty:"core",starterCode:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function areOrthogonal(a, b, tolerance = 1e-9) {
  // TODO: return true if |dot(a, b)| is at most tolerance.
  return false;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('standard basis vectors', areOrthogonal([1, 0], [0, 1]), true);
check('opposite diagonal pair', areOrthogonal([1, 1], [1, -1]), true);
check('non-orthogonal vectors', areOrthogonal([1, 2], [3, 4]), false);
check('nearly zero dot product', areOrthogonal([1, 0], [1e-10, 1]), true);

return results;`,hints:["Use Math.abs.","Check whether the absolute dot product is <= tolerance.","return Math.abs(dot(a, b)) <= tolerance;"],solution:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function areOrthogonal(a, b, tolerance = 1e-9) {
  return Math.abs(dot(a, b)) <= tolerance;
}`,explanation:"In real numerical code, zero often means close enough to zero."},{id:"projection-residual-orthogonal",stepLabel:"12.3",group:"Orthogonality",title:"Projection residual is orthogonal",concept:"After projecting v onto b, the leftover residual is orthogonal to b.",objective:"Return the dot product between the residual and b.",difficulty:"challenge",starterCode:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function projectOnto(v, b) {
  const scale = dot(v, b) / dot(b, b);
  return b.map((entry) => scale * entry);
}

function residualAfterProjection(v, b) {
  const projection = projectOnto(v, b);
  return v.map((entry, i) => entry - projection[i]);
}

function residualDotBasis(v, b) {
  const residual = residualAfterProjection(v, b);

  // TODO: return residual dotted with b.
  return 999;
}`,testCode:`const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: approxEqual(actual, expected) });
}

check('x-axis projection residual', residualDotBasis([3, 4], [1, 0]), 0);
check('diagonal projection residual', residualDotBasis([2, 0], [1, 1]), 0);
check('non-unit projection residual', residualDotBasis([5, 2], [2, 1]), 0);

return results;`,hints:["The residual is already computed.","Use dot(residual, b).","return dot(residual, b);"],solution:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function projectOnto(v, b) {
  const scale = dot(v, b) / dot(b, b);
  return b.map((entry) => scale * entry);
}

function residualAfterProjection(v, b) {
  const projection = projectOnto(v, b);
  return v.map((entry, i) => entry - projection[i]);
}

function residualDotBasis(v, b) {
  const residual = residualAfterProjection(v, b);
  return dot(residual, b);
}`,explanation:"Projection leaves behind an error vector that is perpendicular to the projection direction."},{id:"projection-matrix-outer-product",stepLabel:"13.1",group:"Projection matrix",title:"Outer product",concept:"For a unit vector u, the projection matrix onto u is u times u^T.",objective:"Compute one entry of the outer product.",difficulty:"core",starterCode:`function outerEntry(u, row, col) {
  // TODO: return u[row] times u[col].
  return 0;
}`,testCode:`const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: approxEqual(actual, expected) });
}

check('x-axis top-left', outerEntry([1, 0], 0, 0), 1);
check('x-axis off diagonal', outerEntry([1, 0], 0, 1), 0);
check('y-axis bottom-right', outerEntry([0, 1], 1, 1), 1);
check('diagonal unit vector', outerEntry([0.6, 0.8], 0, 1), 0.48);

return results;`,hints:["Outer product combines one coordinate from the row and one from the column.","Use u[row] and u[col].","return u[row] * u[col];"],solution:`function outerEntry(u, row, col) {
  return u[row] * u[col];
}`,explanation:"The outer product builds a matrix from a vector by multiplying every pair of coordinates."},{id:"projection-matrix-unit",stepLabel:"13.2",group:"Projection matrix",title:"Projection matrix onto unit vector",concept:"A projection matrix onto a unit vector u is P = u times u^T.",objective:"Push the correct outer-product entry into each row.",difficulty:"core",starterCode:`function projectionMatrixUnit(u) {
  const P = [];

  for (let row = 0; row < u.length; row++) {
    const values = [];

    for (let col = 0; col < u.length; col++) {
      // TODO: push the projection matrix entry.
      values.push(0);
    }

    P.push(values);
  }

  return P;
}`,testCode:`const results = [];

function approxMatrix(a, b, tolerance = 1e-9) {
  return a.length === b.length && a.every((row, i) => (
    row.length === b[i].length && row.every((value, j) => Math.abs(value - b[i][j]) <= tolerance)
  ));
}

function check(name, actual, expected) {
  results.push({
    name,
    actual: JSON.stringify(actual),
    expected: JSON.stringify(expected),
    passed: approxMatrix(actual, expected),
  });
}

check('x-axis projection matrix', projectionMatrixUnit([1, 0]), [[1, 0], [0, 0]]);
check('y-axis projection matrix', projectionMatrixUnit([0, 1]), [[0, 0], [0, 1]]);
check('diagonal unit projection matrix', projectionMatrixUnit([0.6, 0.8]), [[0.36, 0.48], [0.48, 0.64]]);

return results;`,hints:["Use the outer product rule.","Each entry is u[row] * u[col].","values.push(u[row] * u[col]);"],solution:`function projectionMatrixUnit(u) {
  const P = [];

  for (let row = 0; row < u.length; row++) {
    const values = [];

    for (let col = 0; col < u.length; col++) {
      values.push(u[row] * u[col]);
    }

    P.push(values);
  }

  return P;
}`,explanation:"A projection matrix stores the projection operation as a matrix."},{id:"projection-matrix-apply",stepLabel:"13.3",group:"Projection matrix",title:"Apply projection matrix",concept:"Applying a projection matrix means matrix-vector multiplication.",objective:"Use matvec to apply P to v.",difficulty:"core",starterCode:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function matvec(A, x) {
  const y = [];

  for (let row = 0; row < A.length; row++) {
    y.push(dot(A[row], x));
  }

  return y;
}

function applyProjectionMatrix(P, v) {
  // TODO: return P times v.
  return [];
}`,testCode:`const results = [];

function approxArray(a, b, tolerance = 1e-9) {
  return a.length === b.length && a.every((value, index) => Math.abs(value - b[index]) <= tolerance);
}

function check(name, actual, expected) {
  results.push({
    name,
    actual: JSON.stringify(actual),
    expected: JSON.stringify(expected),
    passed: approxArray(actual, expected),
  });
}

check('project with x-axis matrix', applyProjectionMatrix([[1,0],[0,0]], [3,4]), [3,0]);
check('project with y-axis matrix', applyProjectionMatrix([[0,0],[0,1]], [3,4]), [0,4]);
check('project with diagonal matrix', applyProjectionMatrix([[0.5,0.5],[0.5,0.5]], [2,0]), [1,1]);

return results;`,hints:["Projection matrix application is just matrix-vector multiplication.","Use the matvec helper.","return matvec(P, v);"],solution:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function matvec(A, x) {
  const y = [];

  for (let row = 0; row < A.length; row++) {
    y.push(dot(A[row], x));
  }

  return y;
}

function applyProjectionMatrix(P, v) {
  return matvec(P, v);
}`,explanation:"Projection matrices turn geometric projection into a normal matrix-vector operation."},{id:"projection-matrix-idempotent",stepLabel:"13.4",group:"Projection matrix",title:"Projecting twice changes nothing",concept:"Projection matrices satisfy P squared = P.",objective:"Return P applied twice to v.",difficulty:"challenge",starterCode:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function matvec(A, x) {
  return A.map((row) => dot(row, x));
}

function projectTwice(P, v) {
  const once = matvec(P, v);

  // TODO: apply P to once.
  return [];
}`,testCode:`const results = [];

function approxArray(a, b, tolerance = 1e-9) {
  return a.length === b.length && a.every((value, index) => Math.abs(value - b[index]) <= tolerance);
}

function check(name, actual, expected) {
  results.push({
    name,
    actual: JSON.stringify(actual),
    expected: JSON.stringify(expected),
    passed: approxArray(actual, expected),
  });
}

check('project twice onto x-axis', projectTwice([[1,0],[0,0]], [3,4]), [3,0]);
check('project twice onto y-axis', projectTwice([[0,0],[0,1]], [3,4]), [0,4]);
check('project twice onto diagonal', projectTwice([[0.5,0.5],[0.5,0.5]], [2,0]), [1,1]);

return results;`,hints:["The variable once is already P times v.","Apply P to once using matvec.","return matvec(P, once);"],solution:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function matvec(A, x) {
  return A.map((row) => dot(row, x));
}

function projectTwice(P, v) {
  const once = matvec(P, v);
  return matvec(P, once);
}`,explanation:"After a vector is already projected onto a subspace, projecting it again does not move it."},{id:"normal-equations-left",stepLabel:"14.1",group:"Normal equations",title:"Compute A^T A",concept:"The left side of the normal equations is A^T A.",objective:"Return transpose(A) times A.",difficulty:"challenge",starterCode:`function transpose(A) {
  const rows = A.length;
  const cols = A[0].length;
  const T = [];

  for (let j = 0; j < cols; j++) {
    const row = [];
    for (let i = 0; i < rows; i++) {
      row.push(A[i][j]);
    }
    T.push(row);
  }

  return T;
}

function matrixCell(A, B, row, col) {
  let total = 0;
  for (let k = 0; k < B.length; k++) {
    total += A[row][k] * B[k][col];
  }
  return total;
}

function matmul(A, B) {
  const C = [];
  for (let i = 0; i < A.length; i++) {
    const row = [];
    for (let j = 0; j < B[0].length; j++) {
      row.push(matrixCell(A, B, i, j));
    }
    C.push(row);
  }
  return C;
}

function normalLeft(A) {
  // TODO: return A^T A.
  return [];
}`,testCode:`const results = [];

function sameMatrix(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function check(name, actual, expected) {
  results.push({
    name,
    actual: JSON.stringify(actual),
    expected: JSON.stringify(expected),
    passed: sameMatrix(actual, expected),
  });
}

check('line design matrix normal left', normalLeft([[1, 1], [1, 2], [1, 3]]), [[3, 6], [6, 14]]);
check('identity normal left', normalLeft([[1, 0], [0, 1]]), [[1, 0], [0, 1]]);

return results;`,hints:["First compute transpose(A).","Then multiply transpose(A) by A.","return matmul(transpose(A), A);"],solution:`function transpose(A) {
  const rows = A.length;
  const cols = A[0].length;
  const T = [];

  for (let j = 0; j < cols; j++) {
    const row = [];
    for (let i = 0; i < rows; i++) {
      row.push(A[i][j]);
    }
    T.push(row);
  }

  return T;
}

function matrixCell(A, B, row, col) {
  let total = 0;
  for (let k = 0; k < B.length; k++) {
    total += A[row][k] * B[k][col];
  }
  return total;
}

function matmul(A, B) {
  const C = [];
  for (let i = 0; i < A.length; i++) {
    const row = [];
    for (let j = 0; j < B[0].length; j++) {
      row.push(matrixCell(A, B, i, j));
    }
    C.push(row);
  }
  return C;
}

function normalLeft(A) {
  return matmul(transpose(A), A);
}`,explanation:"Normal equations use A^T A to summarize how columns of A interact with each other."},{id:"normal-equations-right",stepLabel:"14.2",group:"Normal equations",title:"Compute A^T b",concept:"The right side of the normal equations is A^T b.",objective:"Return transpose(A) times b.",difficulty:"challenge",starterCode:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function transpose(A) {
  const T = [];
  for (let j = 0; j < A[0].length; j++) {
    const row = [];
    for (let i = 0; i < A.length; i++) {
      row.push(A[i][j]);
    }
    T.push(row);
  }
  return T;
}

function matvec(A, x) {
  return A.map((row) => dot(row, x));
}

function normalRight(A, b) {
  // TODO: return A^T b.
  return [];
}`,testCode:`const results = [];

function sameArray(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function check(name, actual, expected) {
  results.push({
    name,
    actual: JSON.stringify(actual),
    expected: JSON.stringify(expected),
    passed: sameArray(actual, expected),
  });
}

check('line design matrix normal right', normalRight([[1, 1], [1, 2], [1, 3]], [2, 3, 5]), [10, 23]);
check('identity normal right', normalRight([[1, 0], [0, 1]], [7, 8]), [7, 8]);

return results;`,hints:["The right side is A transpose times b.","Use transpose(A) and matvec.","return matvec(transpose(A), b);"],solution:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function transpose(A) {
  const T = [];
  for (let j = 0; j < A[0].length; j++) {
    const row = [];
    for (let i = 0; i < A.length; i++) {
      row.push(A[i][j]);
    }
    T.push(row);
  }
  return T;
}

function matvec(A, x) {
  return A.map((row) => dot(row, x));
}

function normalRight(A, b) {
  return matvec(transpose(A), b);
}`,explanation:"A^T b measures how each column of A aligns with the target vector b."},{id:"solve-2x2-system",stepLabel:"14.3",group:"Normal equations",title:"Solve 2x2 system",concept:"A small normal equation can be solved with the 2x2 inverse formula.",objective:"Complete the determinant formula.",difficulty:"challenge",starterCode:`function solve2x2(M, y) {
  const a = M[0][0];
  const b = M[0][1];
  const c = M[1][0];
  const d = M[1][1];

  // TODO: compute the determinant ad - bc.
  const det = 1;

  const x0 = (d * y[0] - b * y[1]) / det;
  const x1 = (-c * y[0] + a * y[1]) / det;

  return [x0, x1];
}`,testCode:`const results = [];

function approxArray(a, b, tolerance = 1e-9) {
  return a.length === b.length && a.every((value, index) => Math.abs(value - b[index]) <= tolerance);
}

function check(name, actual, expected) {
  results.push({
    name,
    actual: JSON.stringify(actual),
    expected: JSON.stringify(expected),
    passed: approxArray(actual, expected),
  });
}

check('identity system', solve2x2([[1,0],[0,1]], [7,8]), [7,8]);
check('diagonal system', solve2x2([[2,0],[0,4]], [6,8]), [3,2]);
check('full 2x2 system', solve2x2([[3,1],[1,2]], [9,8]), [2,3]);

return results;`,hints:["The determinant of [[a,b],[c,d]] is ad - bc.","Use the variables already assigned.","const det = a * d - b * c;"],solution:`function solve2x2(M, y) {
  const a = M[0][0];
  const b = M[0][1];
  const c = M[1][0];
  const d = M[1][1];

  const det = a * d - b * c;

  const x0 = (d * y[0] - b * y[1]) / det;
  const x1 = (-c * y[0] + a * y[1]) / det;

  return [x0, x1];
}`,explanation:"For tiny least-squares examples, a 2x2 solver lets learners see the whole normal-equation pipeline."},{id:"line-fit-design-matrix",stepLabel:"15.1",group:"Least-squares line fit",title:"Design matrix for a line",concept:"A line y = b + mx can be written with rows [1, x].",objective:"Push [1, x] for each input value.",difficulty:"core",starterCode:`function designMatrix(xs) {
  const A = [];

  for (let i = 0; i < xs.length; i++) {
    const x = xs[i];

    // TODO: push the row for intercept + slope.
    A.push([]);
  }

  return A;
}`,testCode:`const results = [];

function sameMatrix(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function check(name, actual, expected) {
  results.push({
    name,
    actual: JSON.stringify(actual),
    expected: JSON.stringify(expected),
    passed: sameMatrix(actual, expected),
  });
}

check('three x values', designMatrix([1, 2, 3]), [[1,1],[1,2],[1,3]]);
check('two x values', designMatrix([0, 5]), [[1,0],[1,5]]);

return results;`,hints:["The first entry is always 1 for the intercept.","The second entry is x for the slope.","A.push([1, x]);"],solution:`function designMatrix(xs) {
  const A = [];

  for (let i = 0; i < xs.length; i++) {
    const x = xs[i];
    A.push([1, x]);
  }

  return A;
}`,explanation:"The column of 1s lets the model learn an intercept; the x column lets it learn a slope."},{id:"line-fit-predict-one",stepLabel:"15.2",group:"Least-squares line fit",title:"Predict with intercept and slope",concept:"A fitted line predicts yHat = intercept + slope * x.",objective:"Complete the prediction formula.",difficulty:"warmup",starterCode:`function predictLine(params, x) {
  const intercept = params[0];
  const slope = params[1];

  // TODO: return intercept + slope * x.
  return 0;
}`,testCode:`const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: approxEqual(actual, expected) });
}

check('intercept only at x=0', predictLine([2, 3], 0), 2);
check('positive slope', predictLine([2, 3], 4), 14);
check('fractional slope', predictLine([-1, 0.5], 6), 2);

return results;`,hints:["params[0] is intercept.","params[1] is slope.","return intercept + slope * x;"],solution:`function predictLine(params, x) {
  const intercept = params[0];
  const slope = params[1];
  return intercept + slope * x;
}`,explanation:"This is the simplest linear regression prediction formula."},{id:"line-fit-normal-equations",stepLabel:"15.3",group:"Least-squares line fit",title:"Fit line with normal equations",concept:"Least squares solves (A^T A)w = A^T y.",objective:"Return the solved parameter vector.",difficulty:"challenge",starterCode:`function fitLineFromNormalEquations(left, right) {
  // left is A^T A and right is A^T y.
  // TODO: solve the 2x2 system.
  return [0, 0];
}`,testCode:`const results = [];

function approxArray(a, b, tolerance = 1e-9) {
  return a.length === b.length && a.every((value, index) => Math.abs(value - b[index]) <= tolerance);
}

function check(name, actual, expected) {
  results.push({
    name,
    actual: JSON.stringify(actual),
    expected: JSON.stringify(expected),
    passed: approxArray(actual, expected),
  });
}

check('line y = x', fitLineFromNormalEquations([[3, 6], [6, 14]], [6, 14]), [0, 1]);
check('line y = 1 + x', fitLineFromNormalEquations([[3, 6], [6, 14]], [9, 20]), [1, 1]);

return results;`,hints:["Reuse the 2x2 solve formula.","Let a, b, c, d be the entries of left, and y be right.","Return [(d*y0 - b*y1)/det, (-c*y0 + a*y1)/det]."],solution:`function fitLineFromNormalEquations(left, right) {
  const a = left[0][0];
  const b = left[0][1];
  const c = left[1][0];
  const d = left[1][1];
  const det = a * d - b * c;

  return [
    (d * right[0] - b * right[1]) / det,
    (-c * right[0] + a * right[1]) / det,
  ];
}`,explanation:"This completes the algebra bridge from matrix multiplication to linear regression."},{id:"mean-basic",stepLabel:"16.1",group:"Centering and covariance",title:"Mean",concept:"The mean is the average value.",objective:"Divide the sum by the number of entries.",difficulty:"warmup",starterCode:`function mean(values) {
  let total = 0;

  for (let i = 0; i < values.length; i++) {
    total += values[i];
  }

  // TODO: return the average.
  return total;
}`,testCode:`const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: approxEqual(actual, expected) });
}

check('mean of three values', mean([1,2,3]), 2);
check('mean of two values', mean([10,20]), 15);
check('mean around zero', mean([-1,1]), 0);

return results;`,hints:["Average means total divided by count.","The count is values.length.","return total / values.length;"],solution:`function mean(values) {
  let total = 0;

  for (let i = 0; i < values.length; i++) {
    total += values[i];
  }

  return total / values.length;
}`,explanation:"Centering and covariance both start by finding the mean."},{id:"center-vector",stepLabel:"16.2",group:"Centering and covariance",title:"Center a vector",concept:"Centering subtracts the mean so the values have average zero.",objective:"Push value minus mean into the centered vector.",difficulty:"core",starterCode:`function mean(values) {
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function center(values) {
  const mu = mean(values);
  const centered = [];

  for (let i = 0; i < values.length; i++) {
    // TODO: subtract the mean from the current value.
    centered.push(0);
  }

  return centered;
}`,testCode:`const results = [];

function approxArray(a, b, tolerance = 1e-9) {
  return a.length === b.length && a.every((value, index) => Math.abs(value - b[index]) <= tolerance);
}

function check(name, actual, expected) {
  results.push({
    name,
    actual: JSON.stringify(actual),
    expected: JSON.stringify(expected),
    passed: approxArray(actual, expected),
  });
}

check('center [1,2,3]', center([1,2,3]), [-1,0,1]);
check('center [10,20]', center([10,20]), [-5,5]);
check('center [-1,1]', center([-1,1]), [-1,1]);

return results;`,hints:["The mean is stored in mu.","Centered value = original value - mean.","centered.push(values[i] - mu);"],solution:`function mean(values) {
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function center(values) {
  const mu = mean(values);
  const centered = [];

  for (let i = 0; i < values.length; i++) {
    centered.push(values[i] - mu);
  }

  return centered;
}`,explanation:"Centering moves the data cloud so its average lies at zero."},{id:"covariance-basic",stepLabel:"16.3",group:"Centering and covariance",title:"Covariance",concept:"Covariance measures whether two centered variables move together.",objective:"Accumulate the product of centered coordinates.",difficulty:"challenge",starterCode:`function mean(values) {
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function covariance(x, y) {
  const meanX = mean(x);
  const meanY = mean(y);
  let total = 0;

  for (let i = 0; i < x.length; i++) {
    const centeredX = x[i] - meanX;
    const centeredY = y[i] - meanY;

    // TODO: add the product of the centered values.
    total += 0;
  }

  return total / x.length;
}`,testCode:`const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: approxEqual(actual, expected) });
}

check('positive covariance', covariance([1,2,3], [1,2,3]), 2 / 3);
check('negative covariance', covariance([1,2,3], [3,2,1]), -2 / 3);
check('zero covariance with constant y', covariance([1,2,3], [5,5,5]), 0);

return results;`,hints:["Covariance multiplies centered values.","Add centeredX * centeredY.","total += centeredX * centeredY;"],solution:`function mean(values) {
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function covariance(x, y) {
  const meanX = mean(x);
  const meanY = mean(y);
  let total = 0;

  for (let i = 0; i < x.length; i++) {
    const centeredX = x[i] - meanX;
    const centeredY = y[i] - meanY;
    total += centeredX * centeredY;
  }

  return total / x.length;
}`,explanation:"Positive covariance means variables tend to move together; negative covariance means they move in opposite directions."},{id:"column-mean",stepLabel:"17.1",group:"PCA bridge",title:"Column mean",concept:"PCA centers each feature column before measuring variance directions.",objective:"Compute the mean of one matrix column.",difficulty:"core",starterCode:`function columnMean(X, col) {
  let total = 0;

  for (let row = 0; row < X.length; row++) {
    // TODO: add the value from this row and column.
    total += 0;
  }

  return total / X.length;
}`,testCode:`const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: approxEqual(actual, expected) });
}

check('first column mean', columnMean([[1,2],[3,4],[5,6]], 0), 3);
check('second column mean', columnMean([[1,2],[3,4],[5,6]], 1), 4);
check('single column mean', columnMean([[10], [20]], 0), 15);

return results;`,hints:["Use X[row][col].","Add the selected column value for each row.","total += X[row][col];"],solution:`function columnMean(X, col) {
  let total = 0;

  for (let row = 0; row < X.length; row++) {
    total += X[row][col];
  }

  return total / X.length;
}`,explanation:"Column means are feature means. PCA centers features, not individual rows."},{id:"center-matrix-columns",stepLabel:"17.2",group:"PCA bridge",title:"Center matrix columns",concept:"Centering a data matrix subtracts each feature column mean.",objective:"Push the centered value for each cell.",difficulty:"challenge",starterCode:`function columnMean(X, col) {
  let total = 0;
  for (let row = 0; row < X.length; row++) {
    total += X[row][col];
  }
  return total / X.length;
}

function centerColumns(X) {
  const rows = X.length;
  const cols = X[0].length;
  const centered = [];

  for (let row = 0; row < rows; row++) {
    const values = [];

    for (let col = 0; col < cols; col++) {
      const mu = columnMean(X, col);

      // TODO: push X[row][col] minus the column mean.
      values.push(0);
    }

    centered.push(values);
  }

  return centered;
}`,testCode:`const results = [];

function approxMatrix(a, b, tolerance = 1e-9) {
  return a.length === b.length && a.every((row, i) => (
    row.length === b[i].length && row.every((value, j) => Math.abs(value - b[i][j]) <= tolerance)
  ));
}

function check(name, actual, expected) {
  results.push({
    name,
    actual: JSON.stringify(actual),
    expected: JSON.stringify(expected),
    passed: approxMatrix(actual, expected),
  });
}

check('center 3x2 matrix', centerColumns([[1,2],[3,4],[5,6]]), [[-2,-2],[0,0],[2,2]]);
check('center 2x1 matrix', centerColumns([[10],[20]]), [[-5],[5]]);

return results;`,hints:["Each feature column gets its own mean.","Subtract mu from the current cell.","values.push(X[row][col] - mu);"],solution:`function columnMean(X, col) {
  let total = 0;
  for (let row = 0; row < X.length; row++) {
    total += X[row][col];
  }
  return total / X.length;
}

function centerColumns(X) {
  const rows = X.length;
  const cols = X[0].length;
  const centered = [];

  for (let row = 0; row < rows; row++) {
    const values = [];

    for (let col = 0; col < cols; col++) {
      const mu = columnMean(X, col);
      values.push(X[row][col] - mu);
    }

    centered.push(values);
  }

  return centered;
}`,explanation:"PCA looks for directions of spread after removing the average feature values."},{id:"pca-project-row",stepLabel:"17.3",group:"PCA bridge",title:"Project onto a component",concept:"A PCA score is a dot product between a centered data row and a component direction.",objective:"Return the dot product between row and component.",difficulty:"core",starterCode:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function pcaScore(centeredRow, component) {
  // TODO: return the score along this component.
  return 0;
}`,testCode:`const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: approxEqual(actual, expected) });
}

check('score on first axis', pcaScore([3, 4], [1, 0]), 3);
check('score on second axis', pcaScore([3, 4], [0, 1]), 4);
check('score on diagonal component', pcaScore([2, 2], [1 / Math.sqrt(2), 1 / Math.sqrt(2)]), 2 * Math.sqrt(2));

return results;`,hints:["A component is a direction vector.","The coordinate along that direction is a dot product.","return dot(centeredRow, component);"],solution:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function pcaScore(centeredRow, component) {
  return dot(centeredRow, component);
}`,explanation:"PCA projection turns high-dimensional centered data into coordinates along chosen directions."},{id:"gram-schmidt-subtract-projection",stepLabel:"18.1",group:"Orthonormal bases",title:"Subtract the projection",concept:"Gram-Schmidt removes the part of a vector that points in a previous basis direction.",objective:"Complete u = v - projection.",difficulty:"core",starterCode:`function subtractVectors(a, b) {
  const result = [];

  for (let i = 0; i < a.length; i++) {
    // TODO: push a[i] minus b[i].
    result.push(0);
  }

  return result;
}`,testCode:`const results = [];

function sameArray(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function check(name, actual, expected) {
  results.push({
    name,
    actual: JSON.stringify(actual),
    expected: JSON.stringify(expected),
    passed: sameArray(actual, expected),
  });
}

check('subtract [3, 4] - [3, 0]', subtractVectors([3, 4], [3, 0]), [0, 4]);
check('subtract [2, 2] - [1, 1]', subtractVectors([2, 2], [1, 1]), [1, 1]);
check('subtract [-1, 5] - [2, 1]', subtractVectors([-1, 5], [2, 1]), [-3, 4]);

return results;`,hints:["Subtract coordinate by coordinate.","The residual keeps what is left after removing the projection.","result.push(a[i] - b[i]);"],solution:`function subtractVectors(a, b) {
  const result = [];

  for (let i = 0; i < a.length; i++) {
    result.push(a[i] - b[i]);
  }

  return result;
}`,explanation:"Gram-Schmidt repeatedly subtracts projections so the remaining vector is orthogonal to earlier basis vectors."},{id:"normalize-vector",stepLabel:"18.2",group:"Orthonormal bases",title:"Normalize a vector",concept:"Normalizing turns a vector into a unit vector without changing its direction.",objective:"Divide each coordinate by the vector norm.",difficulty:"core",starterCode:`function norm(v) {
  let total = 0;

  for (let i = 0; i < v.length; i++) {
    total += v[i] * v[i];
  }

  return Math.sqrt(total);
}

function normalize(v) {
  const length = norm(v);
  const result = [];

  for (let i = 0; i < v.length; i++) {
    // TODO: push the normalized coordinate.
    result.push(0);
  }

  return result;
}`,testCode:`const results = [];

function approxArray(a, b, tolerance = 1e-9) {
  return a.length === b.length && a.every((value, index) => Math.abs(value - b[index]) <= tolerance);
}

function check(name, actual, expected) {
  results.push({
    name,
    actual: JSON.stringify(actual),
    expected: JSON.stringify(expected),
    passed: approxArray(actual, expected),
  });
}

check('normalize [3, 4]', normalize([3, 4]), [0.6, 0.8]);
check('normalize [0, 5]', normalize([0, 5]), [0, 1]);
check('normalize [-6, 8]', normalize([-6, 8]), [-0.6, 0.8]);

return results;`,hints:["The vector length is already stored in length.","Each coordinate should be divided by length.","result.push(v[i] / length);"],solution:`function norm(v) {
  let total = 0;

  for (let i = 0; i < v.length; i++) {
    total += v[i] * v[i];
  }

  return Math.sqrt(total);
}

function normalize(v) {
  const length = norm(v);
  const result = [];

  for (let i = 0; i < v.length; i++) {
    result.push(v[i] / length);
  }

  return result;
}`,explanation:"A unit vector has length 1. Orthonormal bases are made of unit vectors that are mutually perpendicular."},{id:"gram-schmidt-one-step",stepLabel:"18.3",group:"Orthonormal bases",title:"One Gram-Schmidt step",concept:"To make a new vector orthogonal to q, subtract its projection onto q.",objective:"Return v minus its projection onto unit vector q.",difficulty:"challenge",starterCode:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function gramSchmidtResidual(v, q) {
  // q is already a unit vector.
  const scale = dot(v, q);
  const projection = q.map((entry) => scale * entry);

  // TODO: return v minus projection.
  return [];
}`,testCode:`const results = [];

function approxArray(a, b, tolerance = 1e-9) {
  return a.length === b.length && a.every((value, index) => Math.abs(value - b[index]) <= tolerance);
}

function check(name, actual, expected) {
  results.push({
    name,
    actual: JSON.stringify(actual),
    expected: JSON.stringify(expected),
    passed: approxArray(actual, expected),
  });
}

check('remove x-axis part', gramSchmidtResidual([3, 4], [1, 0]), [0, 4]);
check('remove y-axis part', gramSchmidtResidual([3, 4], [0, 1]), [3, 0]);
check('remove diagonal part', gramSchmidtResidual([2, 0], [1 / Math.sqrt(2), 1 / Math.sqrt(2)]), [1, -1]);

return results;`,hints:["The projection has already been computed.","Subtract projection[i] from v[i].","return v.map((entry, i) => entry - projection[i]);"],solution:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function gramSchmidtResidual(v, q) {
  const scale = dot(v, q);
  const projection = q.map((entry) => scale * entry);
  return v.map((entry, i) => entry - projection[i]);
}`,explanation:"This is the heart of Gram-Schmidt: remove the component already explained by a previous basis direction."},{id:"qr-extract-column",stepLabel:"19.1",group:"QR bridge",title:"Extract a column",concept:"QR works with columns of a matrix, so first you need to read a column as a vector.",objective:"Push A[row][col] for every row.",difficulty:"warmup",starterCode:`function column(A, col) {
  const values = [];

  for (let row = 0; row < A.length; row++) {
    // TODO: push the entry from this row and selected column.
    values.push(0);
  }

  return values;
}`,testCode:`const results = [];

function sameArray(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function check(name, actual, expected) {
  results.push({
    name,
    actual: JSON.stringify(actual),
    expected: JSON.stringify(expected),
    passed: sameArray(actual, expected),
  });
}

const A = [
  [1, 2, 3],
  [4, 5, 6],
  [7, 8, 9],
];

check('column 0', column(A, 0), [1, 4, 7]);
check('column 1', column(A, 1), [2, 5, 8]);
check('column 2', column(A, 2), [3, 6, 9]);

return results;`,hints:["A[row][col] picks one entry from the selected column.","Loop over rows while col stays fixed.","values.push(A[row][col]);"],solution:`function column(A, col) {
  const values = [];

  for (let row = 0; row < A.length; row++) {
    values.push(A[row][col]);
  }

  return values;
}`,explanation:"QR decomposition turns matrix columns into orthonormal directions."},{id:"qr-r-entry",stepLabel:"19.2",group:"QR bridge",title:"One R entry",concept:"In QR, R[i][j] measures how much column j of A points along q_i.",objective:"Return q_i dot a_j.",difficulty:"core",starterCode:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function rEntry(qi, aj) {
  // TODO: return the alignment between qi and aj.
  return 0;
}`,testCode:`const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({
    name,
    actual,
    expected,
    passed: approxEqual(actual, expected),
  });
}

check('x-axis with [3,4]', rEntry([1, 0], [3, 4]), 3);
check('y-axis with [3,4]', rEntry([0, 1], [3, 4]), 4);
check('diagonal with [2,0]', rEntry([1 / Math.sqrt(2), 1 / Math.sqrt(2)], [2, 0]), Math.sqrt(2));

return results;`,hints:["R stores dot products between Q columns and A columns.","Use dot(qi, aj).","return dot(qi, aj);"],solution:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function rEntry(qi, aj) {
  return dot(qi, aj);
}`,explanation:"R tells how to combine the orthonormal Q columns to reconstruct A."},{id:"qr-reconstruct",stepLabel:"19.3",group:"QR bridge",title:"Reconstruct with QR",concept:"If A = QR, multiplying Q and R should recover A.",objective:"Return Q times R.",difficulty:"challenge",starterCode:`function matrixCell(A, B, row, col) {
  let total = 0;
  for (let k = 0; k < B.length; k++) {
    total += A[row][k] * B[k][col];
  }
  return total;
}

function matmul(A, B) {
  const C = [];

  for (let row = 0; row < A.length; row++) {
    const values = [];

    for (let col = 0; col < B[0].length; col++) {
      values.push(matrixCell(A, B, row, col));
    }

    C.push(values);
  }

  return C;
}

function reconstructFromQR(Q, R) {
  // TODO: return Q times R.
  return [];
}`,testCode:`const results = [];

function sameMatrix(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function check(name, actual, expected) {
  results.push({
    name,
    actual: JSON.stringify(actual),
    expected: JSON.stringify(expected),
    passed: sameMatrix(actual, expected),
  });
}

check('identity Q', reconstructFromQR([[1, 0], [0, 1]], [[3, 4], [0, 5]]), [[3, 4], [0, 5]]);
check('simple Q and R', reconstructFromQR([[1, 0], [0, 1]], [[1, 2, 3], [4, 5, 6]]), [[1, 2, 3], [4, 5, 6]]);

return results;`,hints:["QR reconstruction is ordinary matrix multiplication.","Use the matmul helper.","return matmul(Q, R);"],solution:`function matrixCell(A, B, row, col) {
  let total = 0;
  for (let k = 0; k < B.length; k++) {
    total += A[row][k] * B[k][col];
  }
  return total;
}

function matmul(A, B) {
  const C = [];

  for (let row = 0; row < A.length; row++) {
    const values = [];

    for (let col = 0; col < B[0].length; col++) {
      values.push(matrixCell(A, B, row, col));
    }

    C.push(values);
  }

  return C;
}

function reconstructFromQR(Q, R) {
  return matmul(Q, R);
}`,explanation:"QR is useful because Q is geometrically nice and R is easy to solve with, but together they still represent the original matrix."},{id:"determinant-2x2",stepLabel:"20.1",group:"Determinant and invertibility",title:"2x2 determinant",concept:"The determinant of [[a,b],[c,d]] is ad - bc.",objective:"Complete the determinant formula.",difficulty:"core",starterCode:`function det2(M) {
  const a = M[0][0];
  const b = M[0][1];
  const c = M[1][0];
  const d = M[1][1];

  // TODO: return ad - bc.
  return 0;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('identity determinant', det2([[1, 0], [0, 1]]), 1);
check('scale determinant', det2([[2, 0], [0, 3]]), 6);
check('shear determinant', det2([[1, 2], [3, 4]]), -2);
check('singular determinant', det2([[1, 2], [2, 4]]), 0);

return results;`,hints:["Use the variables a, b, c, and d.","Multiply the diagonal a*d, then subtract the off-diagonal b*c.","return a * d - b * c;"],solution:`function det2(M) {
  const a = M[0][0];
  const b = M[0][1];
  const c = M[1][0];
  const d = M[1][1];

  return a * d - b * c;
}`,explanation:"For 2D matrices, determinant measures signed area scaling."},{id:"determinant-invertible",stepLabel:"20.2",group:"Determinant and invertibility",title:"Is the matrix invertible?",concept:"A square matrix is invertible only if its determinant is nonzero.",objective:"Return whether the 2x2 matrix is invertible.",difficulty:"core",starterCode:`function det2(M) {
  const a = M[0][0];
  const b = M[0][1];
  const c = M[1][0];
  const d = M[1][1];

  return a * d - b * c;
}

function isInvertible2(M) {
  // TODO: return true when det2(M) is not zero.
  return false;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('identity is invertible', isInvertible2([[1, 0], [0, 1]]), true);
check('scale is invertible', isInvertible2([[2, 0], [0, 3]]), true);
check('rank-deficient is not invertible', isInvertible2([[1, 2], [2, 4]]), false);
check('zero matrix is not invertible', isInvertible2([[0, 0], [0, 0]]), false);

return results;`,hints:["A zero determinant means area collapses to zero.","Check det2(M) !== 0.","return det2(M) !== 0;"],solution:`function det2(M) {
  const a = M[0][0];
  const b = M[0][1];
  const c = M[1][0];
  const d = M[1][1];

  return a * d - b * c;
}

function isInvertible2(M) {
  return det2(M) !== 0;
}`,explanation:"If the determinant is zero, the transformation collapses area and cannot be reversed."},{id:"inverse-2x2",stepLabel:"20.3",group:"Determinant and invertibility",title:"2x2 inverse",concept:"The inverse of [[a,b],[c,d]] is 1/det times [[d,-b],[-c,a]].",objective:"Complete the inverse matrix entries.",difficulty:"challenge",starterCode:`function inverse2(M) {
  const a = M[0][0];
  const b = M[0][1];
  const c = M[1][0];
  const d = M[1][1];

  const det = a * d - b * c;

  // TODO: return the 2x2 inverse.
  return [
    [0, 0],
    [0, 0],
  ];
}`,testCode:`const results = [];

function approxMatrix(a, b, tolerance = 1e-9) {
  return a.length === b.length && a.every((row, i) =>
    row.length === b[i].length &&
    row.every((value, j) => Math.abs(value - b[i][j]) <= tolerance)
  );
}

function check(name, actual, expected) {
  results.push({
    name,
    actual: JSON.stringify(actual),
    expected: JSON.stringify(expected),
    passed: approxMatrix(actual, expected),
  });
}

check('inverse identity', inverse2([[1, 0], [0, 1]]), [[1, 0], [0, 1]]);
check('inverse diagonal', inverse2([[2, 0], [0, 4]]), [[0.5, 0], [0, 0.25]]);
check('inverse [[1,2],[3,4]]', inverse2([[1, 2], [3, 4]]), [[-2, 1], [1.5, -0.5]]);

return results;`,hints:["Use the formula 1/det times [[d, -b], [-c, a]].","Each entry should be divided by det.","return [[d / det, -b / det], [-c / det, a / det]];"],solution:`function inverse2(M) {
  const a = M[0][0];
  const b = M[0][1];
  const c = M[1][0];
  const d = M[1][1];

  const det = a * d - b * c;

  return [
    [d / det, -b / det],
    [-c / det, a / det],
  ];
}`,explanation:"The inverse reverses a linear transformation when the determinant is nonzero."},{id:"change-basis-one-coordinate",stepLabel:"21.1",group:"Change of basis",title:"One coordinate in a new basis",concept:"For an orthonormal basis, a coordinate is a dot product with the basis vector.",objective:"Return v dot basisVector.",difficulty:"core",starterCode:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function coordinateInBasis(v, basisVector) {
  // TODO: return the coordinate of v along basisVector.
  return 0;
}`,testCode:`const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({
    name,
    actual,
    expected,
    passed: approxEqual(actual, expected),
  });
}

check('x-coordinate', coordinateInBasis([3, 4], [1, 0]), 3);
check('y-coordinate', coordinateInBasis([3, 4], [0, 1]), 4);
check('diagonal coordinate', coordinateInBasis([2, 0], [1 / Math.sqrt(2), 1 / Math.sqrt(2)]), Math.sqrt(2));

return results;`,hints:["In an orthonormal basis, projection coordinates are dot products.","Use the dot helper.","return dot(v, basisVector);"],solution:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function coordinateInBasis(v, basisVector) {
  return dot(v, basisVector);
}`,explanation:"A coordinate says how much of the vector points along a basis direction."},{id:"change-basis-all-coordinates",stepLabel:"21.2",group:"Change of basis",title:"All coordinates in a new basis",concept:"Coordinates in an orthonormal basis come from dotting with every basis vector.",objective:"Push each basis coordinate into the result.",difficulty:"core",starterCode:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function coordinatesInBasis(v, basisVectors) {
  const coords = [];

  for (let i = 0; i < basisVectors.length; i++) {
    // TODO: push the coordinate along this basis vector.
    coords.push(0);
  }

  return coords;
}`,testCode:`const results = [];

function approxArray(a, b, tolerance = 1e-9) {
  return a.length === b.length && a.every((value, index) => Math.abs(value - b[index]) <= tolerance);
}

function check(name, actual, expected) {
  results.push({
    name,
    actual: JSON.stringify(actual),
    expected: JSON.stringify(expected),
    passed: approxArray(actual, expected),
  });
}

check('standard basis', coordinatesInBasis([3, 4], [[1, 0], [0, 1]]), [3, 4]);
check('swapped basis', coordinatesInBasis([3, 4], [[0, 1], [1, 0]]), [4, 3]);
check('diagonal basis', coordinatesInBasis([2, 0], [[1 / Math.sqrt(2), 1 / Math.sqrt(2)], [1 / Math.sqrt(2), -1 / Math.sqrt(2)]]), [Math.sqrt(2), Math.sqrt(2)]);

return results;`,hints:["Loop over every basis vector.","Each coordinate is dot(v, basisVectors[i]).","coords.push(dot(v, basisVectors[i]));"],solution:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function coordinatesInBasis(v, basisVectors) {
  const coords = [];

  for (let i = 0; i < basisVectors.length; i++) {
    coords.push(dot(v, basisVectors[i]));
  }

  return coords;
}`,explanation:"Changing to an orthonormal basis is measuring the vector along each new direction."},{id:"change-basis-reconstruct",stepLabel:"21.3",group:"Change of basis",title:"Reconstruct from coordinates",concept:"A vector can be rebuilt by adding coordinate-scaled basis vectors.",objective:"Add coords[j] times basisVectors[j][i] to each output coordinate.",difficulty:"challenge",starterCode:`function reconstructFromBasis(coords, basisVectors) {
  const dimension = basisVectors[0].length;
  const v = Array(dimension).fill(0);

  for (let j = 0; j < basisVectors.length; j++) {
    for (let i = 0; i < dimension; i++) {
      // TODO: add this coordinate-scaled basis entry.
      v[i] += 0;
    }
  }

  return v;
}`,testCode:`const results = [];

function approxArray(a, b, tolerance = 1e-9) {
  return a.length === b.length && a.every((value, index) => Math.abs(value - b[index]) <= tolerance);
}

function check(name, actual, expected) {
  results.push({
    name,
    actual: JSON.stringify(actual),
    expected: JSON.stringify(expected),
    passed: approxArray(actual, expected),
  });
}

check('standard basis', reconstructFromBasis([3, 4], [[1, 0], [0, 1]]), [3, 4]);
check('swapped basis', reconstructFromBasis([4, 3], [[0, 1], [1, 0]]), [3, 4]);
check('diagonal basis', reconstructFromBasis([Math.sqrt(2), Math.sqrt(2)], [[1 / Math.sqrt(2), 1 / Math.sqrt(2)], [1 / Math.sqrt(2), -1 / Math.sqrt(2)]]), [2, 0]);

return results;`,hints:["Each coordinate scales one basis vector.","Add coords[j] * basisVectors[j][i] into v[i].","v[i] += coords[j] * basisVectors[j][i];"],solution:`function reconstructFromBasis(coords, basisVectors) {
  const dimension = basisVectors[0].length;
  const v = Array(dimension).fill(0);

  for (let j = 0; j < basisVectors.length; j++) {
    for (let i = 0; i < dimension; i++) {
      v[i] += coords[j] * basisVectors[j][i];
    }
  }

  return v;
}`,explanation:"Coordinates are not the vector itself; they are instructions for combining basis directions."},{id:"eigen-rayleigh-numerator",stepLabel:"22.1",group:"Eigenvalues",title:"Rayleigh numerator",concept:"The Rayleigh quotient estimates how much A scales a direction v.",objective:"Return v dot Av.",difficulty:"core",starterCode:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function rayleighNumerator(v, Av) {
  // TODO: return v dotted with Av.
  return 0;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('v dot Av', rayleighNumerator([1, 0], [3, 0]), 3);
check('v dot Av 2d', rayleighNumerator([1, 2], [5, 6]), 17);
check('negative', rayleighNumerator([-1, 2], [3, 5]), 7);

return results;`,hints:["The dot helper is already available.","Rayleigh numerator is dot(v, Av).","return dot(v, Av);"],solution:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function rayleighNumerator(v, Av) {
  return dot(v, Av);
}`,explanation:"If v is an eigenvector, Av points in the same direction and the Rayleigh quotient returns its eigenvalue."},{id:"eigen-rayleigh-quotient",stepLabel:"22.2",group:"Eigenvalues",title:"Rayleigh quotient",concept:"The Rayleigh quotient is (v dot Av) / (v dot v).",objective:"Complete the quotient formula.",difficulty:"core",starterCode:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function rayleighQuotient(v, Av) {
  const numerator = dot(v, Av);
  const denominator = dot(v, v);

  // TODO: return numerator divided by denominator.
  return 0;
}`,testCode:`const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({
    name,
    actual,
    expected,
    passed: approxEqual(actual, expected),
  });
}

check('eigen direction scale 3', rayleighQuotient([1, 0], [3, 0]), 3);
check('eigen direction scale 2', rayleighQuotient([0, 2], [0, 4]), 2);
check('general vector', rayleighQuotient([1, 1], [3, 5]), 4);

return results;`,hints:["The numerator and denominator are already computed.","The quotient is numerator / denominator.","return numerator / denominator;"],solution:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function rayleighQuotient(v, Av) {
  const numerator = dot(v, Av);
  const denominator = dot(v, v);
  return numerator / denominator;
}`,explanation:"The Rayleigh quotient estimates the scaling factor of A along the direction v."},{id:"eigen-power-step",stepLabel:"22.3",group:"Eigenvalues",title:"One power iteration step",concept:"Power iteration repeatedly applies A and normalizes to find a dominant eigenvector.",objective:"Return the normalized version of Av.",difficulty:"challenge",starterCode:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function matvec(A, x) {
  return A.map((row) => dot(row, x));
}

function norm(v) {
  return Math.sqrt(dot(v, v));
}

function powerStep(A, v) {
  const Av = matvec(A, v);
  const length = norm(Av);

  // TODO: return Av normalized to unit length.
  return Av;
}`,testCode:`const results = [];

function approxArray(a, b, tolerance = 1e-9) {
  return a.length === b.length && a.every((value, index) => Math.abs(value - b[index]) <= tolerance);
}

function check(name, actual, expected) {
  results.push({
    name,
    actual: JSON.stringify(actual),
    expected: JSON.stringify(expected),
    passed: approxArray(actual, expected),
  });
}

check('diagonal matrix favors x', powerStep([[3, 0], [0, 1]], [1, 0]), [1, 0]);
check('diagonal matrix favors y', powerStep([[1, 0], [0, 4]], [0, 1]), [0, 1]);
check('scale vector', powerStep([[2, 0], [0, 2]], [3, 4]), [0.6, 0.8]);

return results;`,hints:["Av and length are already computed.","Normalize by dividing each entry of Av by length.","return Av.map((entry) => entry / length);"],solution:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function matvec(A, x) {
  return A.map((row) => dot(row, x));
}

function norm(v) {
  return Math.sqrt(dot(v, v));
}

function powerStep(A, v) {
  const Av = matvec(A, v);
  const length = norm(Av);
  return Av.map((entry) => entry / length);
}`,explanation:"Power iteration applies the matrix, then rescales the result so the vector does not explode in length."},{id:"low-rank-scaled-outer-entry",stepLabel:"23.1",group:"Low-rank approximation",title:"Scaled outer product entry",concept:"A rank-1 matrix can be written as sigma times u v^T.",objective:"Return sigma times u[row] times v[col].",difficulty:"core",starterCode:`function rankOneEntry(sigma, u, v, row, col) {
  // TODO: return sigma * u[row] * v[col].
  return 0;
}`,testCode:`const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({
    name,
    actual,
    expected,
    passed: approxEqual(actual, expected),
  });
}

check('entry 0,0', rankOneEntry(2, [1, 0], [3, 4], 0, 0), 6);
check('entry 0,1', rankOneEntry(2, [1, 0], [3, 4], 0, 1), 8);
check('entry 1,0 zero', rankOneEntry(2, [1, 0], [3, 4], 1, 0), 0);
check('fractional', rankOneEntry(5, [0.6, 0.8], [1, 0], 1, 0), 4);

return results;`,hints:["A rank-1 approximation uses one singular value and two direction vectors.","Use sigma, u[row], and v[col].","return sigma * u[row] * v[col];"],solution:`function rankOneEntry(sigma, u, v, row, col) {
  return sigma * u[row] * v[col];
}`,explanation:"A rank-1 matrix is the outer product u v^T scaled by sigma."},{id:"low-rank-build-rank-one",stepLabel:"23.2",group:"Low-rank approximation",title:"Build a rank-1 matrix",concept:"A rank-1 approximation fills every cell with sigma * u_i * v_j.",objective:"Push the scaled outer-product entry into each row.",difficulty:"core",starterCode:`function rankOneMatrix(sigma, u, v) {
  const A = [];

  for (let row = 0; row < u.length; row++) {
    const values = [];

    for (let col = 0; col < v.length; col++) {
      // TODO: push sigma * u[row] * v[col].
      values.push(0);
    }

    A.push(values);
  }

  return A;
}`,testCode:`const results = [];

function approxMatrix(a, b, tolerance = 1e-9) {
  return a.length === b.length && a.every((row, i) =>
    row.length === b[i].length &&
    row.every((value, j) => Math.abs(value - b[i][j]) <= tolerance)
  );
}

function check(name, actual, expected) {
  results.push({
    name,
    actual: JSON.stringify(actual),
    expected: JSON.stringify(expected),
    passed: approxMatrix(actual, expected),
  });
}

check('simple rank one', rankOneMatrix(2, [1, 0], [3, 4]), [[6, 8], [0, 0]]);
check('column rank one', rankOneMatrix(3, [1, 2], [1]), [[3], [6]]);
check('identity-like direction', rankOneMatrix(1, [1, 1], [1, -1]), [[1, -1], [1, -1]]);

return results;`,hints:["This is the same formula for every row and column.","Use sigma * u[row] * v[col].","values.push(sigma * u[row] * v[col]);"],solution:`function rankOneMatrix(sigma, u, v) {
  const A = [];

  for (let row = 0; row < u.length; row++) {
    const values = [];

    for (let col = 0; col < v.length; col++) {
      values.push(sigma * u[row] * v[col]);
    }

    A.push(values);
  }

  return A;
}`,explanation:"Low-rank approximation builds a matrix by adding a few simple rank-1 patterns."},{id:"low-rank-frobenius-error",stepLabel:"23.3",group:"Low-rank approximation",title:"Approximation error",concept:"The Frobenius error is the sum of squared entrywise differences between a matrix and its approximation.",objective:"Add the squared difference for each cell.",difficulty:"challenge",starterCode:`function frobeniusErrorSquared(A, Ahat) {
  let total = 0;

  for (let row = 0; row < A.length; row++) {
    for (let col = 0; col < A[0].length; col++) {
      const diff = A[row][col] - Ahat[row][col];

      // TODO: add squared difference.
      total += 0;
    }
  }

  return total;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('zero error', frobeniusErrorSquared([[1, 2], [3, 4]], [[1, 2], [3, 4]]), 0);
check('single difference', frobeniusErrorSquared([[1, 2], [3, 4]], [[1, 2], [3, 5]]), 1);
check('multiple differences', frobeniusErrorSquared([[1, 2], [3, 4]], [[0, 0], [0, 0]]), 30);

return results;`,hints:["Frobenius error squares every entry difference.","The difference is already stored in diff.","total += diff * diff;"],solution:`function frobeniusErrorSquared(A, Ahat) {
  let total = 0;

  for (let row = 0; row < A.length; row++) {
    for (let col = 0; col < A[0].length; col++) {
      const diff = A[row][col] - Ahat[row][col];
      total += diff * diff;
    }
  }

  return total;
}`,explanation:"Low-rank approximation keeps the most important patterns and measures what was lost with reconstruction error."},{id:"absolute-error",stepLabel:"24.1",group:"Numerical stability",title:"Absolute error",concept:"Absolute error measures how far an approximation is from the true value.",objective:"Return the absolute difference between trueValue and approxValue.",difficulty:"warmup",starterCode:`function absoluteError(trueValue, approxValue) {
  // TODO: return the absolute difference.
  return 0;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('error 10 vs 8', absoluteError(10, 8), 2);
check('error 8 vs 10', absoluteError(8, 10), 2);
check('error 5 vs 5', absoluteError(5, 5), 0);
check('error -3 vs 2', absoluteError(-3, 2), 5);

return results;`,hints:["Use Math.abs.","The difference is trueValue - approxValue.","return Math.abs(trueValue - approxValue);"],solution:`function absoluteError(trueValue, approxValue) {
  return Math.abs(trueValue - approxValue);
}`,explanation:"Absolute error is the raw distance between a true value and an approximation."},{id:"relative-error",stepLabel:"24.2",group:"Numerical stability",title:"Relative error",concept:"Relative error compares error to the size of the true value.",objective:"Return absolute error divided by absolute true value.",difficulty:"core",starterCode:`function relativeError(trueValue, approxValue) {
  const error = Math.abs(trueValue - approxValue);

  // TODO: divide error by the size of trueValue.
  return error;
}`,testCode:`const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({
    name,
    actual,
    expected,
    passed: approxEqual(actual, expected),
  });
}

check('10 vs 9', relativeError(10, 9), 0.1);
check('100 vs 99', relativeError(100, 99), 0.01);
check('-50 vs -45', relativeError(-50, -45), 0.1);

return results;`,hints:["Relative error asks: how large is the error compared with the true value?","Use Math.abs(trueValue) in the denominator.","return error / Math.abs(trueValue);"],solution:`function relativeError(trueValue, approxValue) {
  const error = Math.abs(trueValue - approxValue);
  return error / Math.abs(trueValue);
}`,explanation:"A raw error of 1 is huge if the true value is 2, but tiny if the true value is 1,000,000."},{id:"condition-number-from-singular-values",stepLabel:"24.3",group:"Numerical stability",title:"Condition number",concept:"A condition number compares the largest and smallest singular values.",objective:"Return max singular value divided by min singular value.",difficulty:"core",starterCode:`function conditionNumber(singularValues) {
  const largest = Math.max(...singularValues);
  const smallest = Math.min(...singularValues);

  // TODO: return largest divided by smallest.
  return 0;
}`,testCode:`const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({
    name,
    actual,
    expected,
    passed: approxEqual(actual, expected),
  });
}

check('well-conditioned', conditionNumber([5, 4, 2]), 2.5);
check('identity-like', conditionNumber([1, 1, 1]), 1);
check('ill-conditioned', conditionNumber([100, 1, 0.01]), 10000);

return results;`,hints:["Condition number is largest scale divided by smallest scale.","The largest and smallest variables are already computed.","return largest / smallest;"],solution:`function conditionNumber(singularValues) {
  const largest = Math.max(...singularValues);
  const smallest = Math.min(...singularValues);
  return largest / smallest;
}`,explanation:"A high condition number means some directions are stretched much more than others, making solutions sensitive to noise."},{id:"detect-ill-conditioning",stepLabel:"24.4",group:"Numerical stability",title:"Detect ill-conditioning",concept:"A large condition number warns that small input noise may become large output error.",objective:"Return true when condition number exceeds the threshold.",difficulty:"core",starterCode:`function isIllConditioned(singularValues, threshold = 1000) {
  const largest = Math.max(...singularValues);
  const smallest = Math.min(...singularValues);
  const condition = largest / smallest;

  // TODO: return whether condition is greater than threshold.
  return false;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('identity-like not ill-conditioned', isIllConditioned([1, 1, 1]), false);
check('moderate not ill-conditioned by default', isIllConditioned([100, 2]), false);
check('large condition is ill-conditioned', isIllConditioned([100, 0.01]), true);
check('custom threshold', isIllConditioned([20, 1], 10), true);

return results;`,hints:["The condition number is already computed.","Compare condition with threshold.","return condition > threshold;"],solution:`function isIllConditioned(singularValues, threshold = 1000) {
  const largest = Math.max(...singularValues);
  const smallest = Math.min(...singularValues);
  const condition = largest / smallest;

  return condition > threshold;
}`,explanation:"Ill-conditioned systems can produce unstable answers even when the formula is mathematically correct."},{id:"pseudoinverse-invert-singular-values",stepLabel:"25.1",group:"Pseudoinverse bridge",title:"Invert singular values",concept:"The pseudoinverse inverts nonzero singular values.",objective:"Return 1 / sigma for a nonzero singular value.",difficulty:"warmup",starterCode:`function invertSingularValue(sigma) {
  // TODO: return the reciprocal of sigma.
  return sigma;
}`,testCode:`const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({
    name,
    actual,
    expected,
    passed: approxEqual(actual, expected),
  });
}

check('invert 2', invertSingularValue(2), 0.5);
check('invert 4', invertSingularValue(4), 0.25);
check('invert 0.5', invertSingularValue(0.5), 2);

return results;`,hints:["The reciprocal of sigma is one divided by sigma.","Use 1 / sigma.","return 1 / sigma;"],solution:`function invertSingularValue(sigma) {
  return 1 / sigma;
}`,explanation:"The pseudoinverse reverses directions that the matrix scales, but only where the scale is not zero."},{id:"pseudoinverse-threshold-singular-values",stepLabel:"25.2",group:"Pseudoinverse bridge",title:"Threshold tiny singular values",concept:"Very small singular values can amplify noise, so pseudoinverses often threshold them.",objective:"Return 0 when sigma is too small, otherwise return 1 / sigma.",difficulty:"core",starterCode:`function safeInvertSingularValue(sigma, tolerance = 1e-6) {
  // TODO: return 0 if sigma is below tolerance; otherwise return 1 / sigma.
  return 0;
}`,testCode:`const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({
    name,
    actual,
    expected,
    passed: approxEqual(actual, expected),
  });
}

check('invert 2', safeInvertSingularValue(2), 0.5);
check('invert 4', safeInvertSingularValue(4), 0.25);
check('tiny value becomes zero', safeInvertSingularValue(1e-9), 0);
check('custom tolerance', safeInvertSingularValue(0.01, 0.1), 0);

return results;`,hints:["Use an if statement or ternary expression.","If sigma < tolerance, return 0.","return sigma < tolerance ? 0 : 1 / sigma;"],solution:`function safeInvertSingularValue(sigma, tolerance = 1e-6) {
  return sigma < tolerance ? 0 : 1 / sigma;
}`,explanation:"Thresholding prevents tiny singular values from exploding into huge inverse scales."},{id:"pseudoinverse-sigma-plus",stepLabel:"25.3",group:"Pseudoinverse bridge",title:"Build Sigma-plus diagonal",concept:"Sigma-plus contains inverted singular values on the diagonal.",objective:"Push the safe inverted value on the diagonal and 0 elsewhere.",difficulty:"challenge",starterCode:`function safeInvertSingularValue(sigma, tolerance = 1e-6) {
  return sigma < tolerance ? 0 : 1 / sigma;
}

function sigmaPlus(singularValues) {
  const Splus = [];

  for (let row = 0; row < singularValues.length; row++) {
    const values = [];

    for (let col = 0; col < singularValues.length; col++) {
      // TODO: push inverted singular value on diagonal, 0 otherwise.
      values.push(999);
    }

    Splus.push(values);
  }

  return Splus;
}`,testCode:`const results = [];

function approxMatrix(a, b, tolerance = 1e-9) {
  return a.length === b.length && a.every((row, i) =>
    row.length === b[i].length &&
    row.every((value, j) => Math.abs(value - b[i][j]) <= tolerance)
  );
}

function check(name, actual, expected) {
  results.push({
    name,
    actual: JSON.stringify(actual),
    expected: JSON.stringify(expected),
    passed: approxMatrix(actual, expected),
  });
}

check('two singular values', sigmaPlus([2, 4]), [[0.5, 0], [0, 0.25]]);
check('three singular values', sigmaPlus([1, 2, 5]), [[1,0,0],[0,0.5,0],[0,0,0.2]]);
check('tiny singular value', sigmaPlus([2, 1e-9]), [[0.5, 0], [0, 0]]);

return results;`,hints:["Use row === col to detect the diagonal.","On the diagonal, use safeInvertSingularValue(singularValues[row]).","values.push(row === col ? safeInvertSingularValue(singularValues[row]) : 0);"],solution:`function safeInvertSingularValue(sigma, tolerance = 1e-6) {
  return sigma < tolerance ? 0 : 1 / sigma;
}

function sigmaPlus(singularValues) {
  const Splus = [];

  for (let row = 0; row < singularValues.length; row++) {
    const values = [];

    for (let col = 0; col < singularValues.length; col++) {
      values.push(row === col ? safeInvertSingularValue(singularValues[row]) : 0);
    }

    Splus.push(values);
  }

  return Splus;
}`,explanation:"Sigma-plus is the diagonal scaling matrix used inside the SVD formula for the pseudoinverse."},{id:"pseudoinverse-apply",stepLabel:"25.4",group:"Pseudoinverse bridge",title:"Apply pseudoinverse",concept:"A pseudoinverse solution is x = Aplus b.",objective:"Return Aplus times b.",difficulty:"core",starterCode:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function matvec(A, x) {
  return A.map((row) => dot(row, x));
}

function solveWithPseudoinverse(Aplus, b) {
  // TODO: return Aplus times b.
  return [];
}`,testCode:`const results = [];

function approxArray(a, b, tolerance = 1e-9) {
  return a.length === b.length && a.every((value, index) => Math.abs(value - b[index]) <= tolerance);
}

function check(name, actual, expected) {
  results.push({
    name,
    actual: JSON.stringify(actual),
    expected: JSON.stringify(expected),
    passed: approxArray(actual, expected),
  });
}

check('identity pseudoinverse', solveWithPseudoinverse([[1,0],[0,1]], [7,8]), [7,8]);
check('diagonal pseudoinverse', solveWithPseudoinverse([[0.5,0],[0,0.25]], [6,8]), [3,2]);
check('rectangular-like Aplus', solveWithPseudoinverse([[1,0,0],[0,0.5,0]], [3,8,10]), [3,4]);

return results;`,hints:["Solving with a pseudoinverse is matrix-vector multiplication.","Use the matvec helper.","return matvec(Aplus, b);"],solution:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function matvec(A, x) {
  return A.map((row) => dot(row, x));
}

function solveWithPseudoinverse(Aplus, b) {
  return matvec(Aplus, b);
}`,explanation:"The pseudoinverse gives a least-squares or minimum-norm solution when an ordinary inverse is unavailable."},{id:"gd-prediction-error",stepLabel:"26.1",group:"Gradient descent least squares",title:"Prediction error",concept:"Gradient descent updates parameters using prediction error.",objective:"Return prediction minus target.",difficulty:"warmup",starterCode:`function predictionError(prediction, target) {
  // TODO: return prediction minus target.
  return 0;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('overprediction', predictionError(10, 7), 3);
check('underprediction', predictionError(4, 9), -5);
check('perfect prediction', predictionError(5, 5), 0);

return results;`,hints:["Error is signed: prediction - target.","Positive means prediction was too high.","return prediction - target;"],solution:`function predictionError(prediction, target) {
  return prediction - target;
}`,explanation:"Signed error tells gradient descent which direction the prediction is wrong."},{id:"gd-one-weight-gradient",stepLabel:"26.2",group:"Gradient descent least squares",title:"One weight gradient",concept:"For squared error, the gradient contribution is error times feature value.",objective:"Return error * feature.",difficulty:"core",starterCode:`function oneWeightGradient(error, feature) {
  // TODO: return this feature's gradient contribution.
  return 0;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('positive error, positive feature', oneWeightGradient(3, 2), 6);
check('negative error, positive feature', oneWeightGradient(-5, 2), -10);
check('positive error, zero feature', oneWeightGradient(3, 0), 0);
check('negative feature', oneWeightGradient(4, -2), -8);

return results;`,hints:["The gradient scales with how much this feature contributed.","Multiply error by feature.","return error * feature;"],solution:`function oneWeightGradient(error, feature) {
  return error * feature;
}`,explanation:"If a feature is large, the weight connected to it gets a larger update signal."},{id:"gd-gradient-vector",stepLabel:"26.3",group:"Gradient descent least squares",title:"Gradient vector",concept:"Each weight receives error times its matching feature.",objective:"Push error * x[i] for every feature.",difficulty:"core",starterCode:`function gradientForExample(error, x) {
  const gradient = [];

  for (let i = 0; i < x.length; i++) {
    // TODO: push the gradient for weight i.
    gradient.push(0);
  }

  return gradient;
}`,testCode:`const results = [];

function sameArray(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function check(name, actual, expected) {
  results.push({
    name,
    actual: JSON.stringify(actual),
    expected: JSON.stringify(expected),
    passed: sameArray(actual, expected),
  });
}

check('error 3', gradientForExample(3, [1, 2, 3]), [3, 6, 9]);
check('error -2', gradientForExample(-2, [1, 0, 4]), [-2, 0, -8]);
check('zero error', gradientForExample(0, [5, 6]), [0, 0]);

return results;`,hints:["The same error multiplies every feature.","For weight i, use error * x[i].","gradient.push(error * x[i]);"],solution:`function gradientForExample(error, x) {
  const gradient = [];

  for (let i = 0; i < x.length; i++) {
    gradient.push(error * x[i]);
  }

  return gradient;
}`,explanation:"The gradient vector tells every weight how to move to reduce squared error."},{id:"gd-weight-update",stepLabel:"26.4",group:"Gradient descent least squares",title:"One gradient descent update",concept:"Gradient descent subtracts learningRate times gradient.",objective:"Update one weight coordinate.",difficulty:"core",starterCode:`function updateWeights(weights, gradient, learningRate) {
  const updated = [];

  for (let i = 0; i < weights.length; i++) {
    // TODO: subtract learningRate times gradient[i].
    updated.push(weights[i]);
  }

  return updated;
}`,testCode:`const results = [];

function approxArray(a, b, tolerance = 1e-9) {
  return a.length === b.length && a.every((value, index) => Math.abs(value - b[index]) <= tolerance);
}

function check(name, actual, expected) {
  results.push({
    name,
    actual: JSON.stringify(actual),
    expected: JSON.stringify(expected),
    passed: approxArray(actual, expected),
  });
}

check('simple update', updateWeights([1, 2], [3, 4], 0.1), [0.7, 1.6]);
check('negative gradient', updateWeights([1, 2], [-1, 2], 0.5), [1.5, 1]);
check('zero gradient', updateWeights([5, 6], [0, 0], 0.1), [5, 6]);

return results;`,hints:["Gradient descent moves opposite the gradient.","New weight = old weight - learningRate * gradient.","updated.push(weights[i] - learningRate * gradient[i]);"],solution:`function updateWeights(weights, gradient, learningRate) {
  const updated = [];

  for (let i = 0; i < weights.length; i++) {
    updated.push(weights[i] - learningRate * gradient[i]);
  }

  return updated;
}`,explanation:"The learning rate controls the size of the step downhill."},{id:"logistic-logit-dot",stepLabel:"27.1",group:"Logistic regression bridge",title:"Logit is a dot product",concept:"Logistic regression first computes a linear score: w dot x + b.",objective:"Return dot(weights, x) plus bias.",difficulty:"core",starterCode:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function logit(weights, x, bias) {
  // TODO: return w dot x + bias.
  return 0;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('simple logit', logit([1, 2], [3, 4], 0), 11);
check('with bias', logit([1, 2], [3, 4], -1), 10);
check('negative weight', logit([-1, 2], [3, 5], 1), 8);

return results;`,hints:["Use the dot helper.","The linear score is dot(weights, x) + bias.","return dot(weights, x) + bias;"],solution:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function logit(weights, x, bias) {
  return dot(weights, x) + bias;
}`,explanation:"Logistic regression is linear algebra plus a sigmoid. The dot product creates the score."},{id:"logistic-sigmoid",stepLabel:"27.2",group:"Logistic regression bridge",title:"Sigmoid",concept:"Sigmoid turns any real-valued logit into a value between 0 and 1.",objective:"Complete the sigmoid formula.",difficulty:"core",starterCode:`function sigmoid(z) {
  // TODO: return 1 / (1 + exp(-z)).
  return z;
}`,testCode:`const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({
    name,
    actual,
    expected,
    passed: approxEqual(actual, expected),
  });
}

check('sigmoid(0)', sigmoid(0), 0.5);
check('sigmoid(log 3)', sigmoid(Math.log(3)), 0.75);
check('sigmoid(-log 3)', sigmoid(-Math.log(3)), 0.25);

return results;`,hints:["Use Math.exp.","The formula is 1 / (1 + Math.exp(-z)).","return 1 / (1 + Math.exp(-z));"],solution:`function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}`,explanation:"Sigmoid converts a linear score into a probability-like value."},{id:"logistic-predict-probability",stepLabel:"27.3",group:"Logistic regression bridge",title:"Predict probability",concept:"A logistic model predicts sigmoid(w dot x + b).",objective:"Apply sigmoid to the logit.",difficulty:"core",starterCode:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

function predictProbability(weights, x, bias) {
  const z = dot(weights, x) + bias;

  // TODO: return sigmoid of z.
  return z;
}`,testCode:`const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({
    name,
    actual,
    expected,
    passed: approxEqual(actual, expected),
  });
}

check('probability 0.5', predictProbability([0, 0], [3, 4], 0), 0.5);
check('probability 0.75', predictProbability([1], [Math.log(3)], 0), 0.75);
check('probability with bias', predictProbability([1], [0], Math.log(3)), 0.75);

return results;`,hints:["z is already the linear score.","Apply sigmoid(z).","return sigmoid(z);"],solution:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

function predictProbability(weights, x, bias) {
  const z = dot(weights, x) + bias;
  return sigmoid(z);
}`,explanation:"Logistic regression turns feature-weight alignment into a probability."},{id:"logistic-binary-cross-entropy",stepLabel:"27.4",group:"Logistic regression bridge",title:"Binary cross-entropy",concept:"Binary cross-entropy penalizes confident wrong probabilities heavily.",objective:"Complete the loss formula for one label and probability.",difficulty:"challenge",starterCode:`function binaryCrossEntropy(y, p) {
  // TODO: return -(y log p + (1-y) log(1-p)).
  return 0;
}`,testCode:`const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({
    name,
    actual,
    expected,
    passed: approxEqual(actual, expected),
  });
}

check('positive label p=0.5', binaryCrossEntropy(1, 0.5), -Math.log(0.5));
check('negative label p=0.5', binaryCrossEntropy(0, 0.5), -Math.log(0.5));
check('positive label p=0.8', binaryCrossEntropy(1, 0.8), -Math.log(0.8));
check('negative label p=0.2', binaryCrossEntropy(0, 0.2), -Math.log(0.8));

return results;`,hints:["Use Math.log.","The formula is negative of y log p plus (1-y) log(1-p).","return -(y * Math.log(p) + (1 - y) * Math.log(1 - p));"],solution:`function binaryCrossEntropy(y, p) {
  return -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
}`,explanation:"Cross-entropy rewards high probability on the true class and punishes confident wrong predictions."},{id:"attention-one-score",stepLabel:"28.1",group:"Attention algebra bridge",title:"One attention score",concept:"One attention score is a query vector dotted with a key vector.",objective:"Return query dot key.",difficulty:"core",starterCode:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function attentionScore(query, key) {
  // TODO: return query dot key.
  return 0;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('score 1', attentionScore([1, 2], [3, 4]), 11);
check('orthogonal score', attentionScore([1, 0], [0, 1]), 0);
check('negative score', attentionScore([-1, 2], [3, 5]), 7);

return results;`,hints:["Attention starts with similarity scores.","Similarity here is dot product.","return dot(query, key);"],solution:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function attentionScore(query, key) {
  return dot(query, key);
}`,explanation:"In transformer attention, QK^T is a matrix of query-key dot products."},{id:"attention-scale-score",stepLabel:"28.2",group:"Attention algebra bridge",title:"Scale attention score",concept:"Attention scores are divided by sqrt(d) to keep logits from growing too large.",objective:"Divide the raw score by Math.sqrt(d).",difficulty:"core",starterCode:`function scaleAttentionScore(rawScore, d) {
  // TODO: return rawScore divided by sqrt(d).
  return rawScore;
}`,testCode:`const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({
    name,
    actual,
    expected,
    passed: approxEqual(actual, expected),
  });
}

check('scale by sqrt 4', scaleAttentionScore(8, 4), 4);
check('scale by sqrt 9', scaleAttentionScore(12, 9), 4);
check('scale by sqrt 1', scaleAttentionScore(7, 1), 7);

return results;`,hints:["Use Math.sqrt(d).","Scaled dot-product attention divides by the square root of dimension.","return rawScore / Math.sqrt(d);"],solution:`function scaleAttentionScore(rawScore, d) {
  return rawScore / Math.sqrt(d);
}`,explanation:"Scaling keeps attention logits numerically stable before softmax."},{id:"attention-softmax-denominator",stepLabel:"28.3",group:"Attention algebra bridge",title:"Softmax denominator",concept:"Softmax normalizes exponentiated scores so weights sum to 1.",objective:"Accumulate Math.exp(score) for every score.",difficulty:"core",starterCode:`function softmaxDenominator(scores) {
  let total = 0;

  for (let i = 0; i < scores.length; i++) {
    // TODO: add exp of this score.
    total += 0;
  }

  return total;
}`,testCode:`const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({
    name,
    actual,
    expected,
    passed: approxEqual(actual, expected),
  });
}

check('zeros', softmaxDenominator([0, 0]), 2);
check('one zero', softmaxDenominator([0]), 1);
check('mixed', softmaxDenominator([0, Math.log(3)]), 4);

return results;`,hints:["Softmax uses exponentials.","Use Math.exp(scores[i]).","total += Math.exp(scores[i]);"],solution:`function softmaxDenominator(scores) {
  let total = 0;

  for (let i = 0; i < scores.length; i++) {
    total += Math.exp(scores[i]);
  }

  return total;
}`,explanation:"Softmax turns raw attention scores into normalized attention weights."},{id:"attention-softmax-weights",stepLabel:"28.4",group:"Attention algebra bridge",title:"Softmax weights",concept:"Each softmax weight is exp(score) divided by the sum of all exp(scores).",objective:"Push one normalized softmax weight per score.",difficulty:"challenge",starterCode:`function softmax(scores) {
  let denominator = 0;

  for (let i = 0; i < scores.length; i++) {
    denominator += Math.exp(scores[i]);
  }

  const weights = [];

  for (let i = 0; i < scores.length; i++) {
    // TODO: push the normalized softmax weight.
    weights.push(0);
  }

  return weights;
}`,testCode:`const results = [];

function approxArray(a, b, tolerance = 1e-9) {
  return a.length === b.length && a.every((value, index) => Math.abs(value - b[index]) <= tolerance);
}

function check(name, actual, expected) {
  results.push({
    name,
    actual: JSON.stringify(actual),
    expected: JSON.stringify(expected),
    passed: approxArray(actual, expected),
  });
}

check('two equal scores', softmax([0, 0]), [0.5, 0.5]);
check('one option', softmax([0]), [1]);
check('log ratio', softmax([0, Math.log(3)]), [0.25, 0.75]);

return results;`,hints:["The denominator is already computed.","Weight i is exp(scores[i]) / denominator.","weights.push(Math.exp(scores[i]) / denominator);"],solution:`function softmax(scores) {
  let denominator = 0;

  for (let i = 0; i < scores.length; i++) {
    denominator += Math.exp(scores[i]);
  }

  const weights = [];

  for (let i = 0; i < scores.length; i++) {
    weights.push(Math.exp(scores[i]) / denominator);
  }

  return weights;
}`,explanation:"Attention weights are a probability distribution over which values to read."},{id:"attention-weighted-value-sum",stepLabel:"28.5",group:"Attention algebra bridge",title:"Weighted value sum",concept:"The attention output is a weighted sum of value vectors.",objective:"Add weight times value coordinate into the output.",difficulty:"challenge",starterCode:`function weightedValueSum(weights, values) {
  const dimension = values[0].length;
  const output = Array(dimension).fill(0);

  for (let token = 0; token < values.length; token++) {
    for (let dim = 0; dim < dimension; dim++) {
      // TODO: add this token's weighted value coordinate.
      output[dim] += 0;
    }
  }

  return output;
}`,testCode:`const results = [];

function approxArray(a, b, tolerance = 1e-9) {
  return a.length === b.length && a.every((value, index) => Math.abs(value - b[index]) <= tolerance);
}

function check(name, actual, expected) {
  results.push({
    name,
    actual: JSON.stringify(actual),
    expected: JSON.stringify(expected),
    passed: approxArray(actual, expected),
  });
}

check('choose first value', weightedValueSum([1, 0], [[3, 4], [10, 20]]), [3, 4]);
check('average two values', weightedValueSum([0.5, 0.5], [[2, 4], [6, 8]]), [4, 6]);
check('weighted mix', weightedValueSum([0.25, 0.75], [[0, 4], [8, 0]]), [6, 1]);

return results;`,hints:["Each token contributes weight[token] times its value vector.","For each coordinate, add weights[token] * values[token][dim].","output[dim] += weights[token] * values[token][dim];"],solution:`function weightedValueSum(weights, values) {
  const dimension = values[0].length;
  const output = Array(dimension).fill(0);

  for (let token = 0; token < values.length; token++) {
    for (let dim = 0; dim < dimension; dim++) {
      output[dim] += weights[token] * values[token][dim];
    }
  }

  return output;
}`,explanation:"Attention does not return the most-attended token; it returns a mixture of value vectors."},{id:"derivative-line-slope",stepLabel:"29.1",group:"Derivative basics",title:"Slope of a line",concept:"The derivative of f(x) = mx + b is the constant slope m.",objective:"Return the slope m.",difficulty:"warmup",starterCode:`function derivativeOfLine(m, b, x) {
  // f(x) = m*x + b
  // TODO: return the derivative with respect to x.
  return 0;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('slope 2', derivativeOfLine(2, 5, 10), 2);
check('slope -3', derivativeOfLine(-3, 1, 7), -3);
check('slope 0', derivativeOfLine(0, 100, 4), 0);

return results;`,hints:["The derivative of m*x + b is m.","b disappears because constants do not change with x.","return m;"],solution:`function derivativeOfLine(m, b, x) {
  return m;
}`,explanation:"A derivative measures local change. For a straight line, the local change is the same everywhere."},{id:"derivative-square",stepLabel:"29.2",group:"Derivative basics",title:"Derivative of x^2",concept:"The derivative of x^2 is 2x.",objective:"Return 2 * x.",difficulty:"warmup",starterCode:`function derivativeSquare(x) {
  // f(x) = x*x
  // TODO: return f'(x).
  return 0;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('x=0', derivativeSquare(0), 0);
check('x=3', derivativeSquare(3), 6);
check('x=-4', derivativeSquare(-4), -8);
check('x=10', derivativeSquare(10), 20);

return results;`,hints:["Power rule: d/dx x^2 = 2x.","The slope grows as x gets farther from 0.","return 2 * x;"],solution:`function derivativeSquare(x) {
  return 2 * x;
}`,explanation:"For squared loss, gradients grow with the size of the error."},{id:"derivative-squared-error",stepLabel:"29.3",group:"Derivative basics",title:"Squared-error derivative",concept:"For loss L = (prediction - target)^2, the derivative with respect to prediction is 2(prediction - target).",objective:"Return the gradient of squared error with respect to prediction.",difficulty:"core",starterCode:`function squaredErrorGradient(prediction, target) {
  const error = prediction - target;

  // TODO: return d/dprediction of error^2.
  return 0;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('prediction too high', squaredErrorGradient(10, 7), 6);
check('prediction too low', squaredErrorGradient(4, 9), -10);
check('perfect prediction', squaredErrorGradient(5, 5), 0);

return results;`,hints:["Squared error is error^2.","Derivative of error^2 with respect to prediction is 2 * error.","return 2 * error;"],solution:`function squaredErrorGradient(prediction, target) {
  const error = prediction - target;
  return 2 * error;
}`,explanation:"The gradient is positive when prediction is too high, negative when too low, and zero when perfect."},{id:"numerical-derivative",stepLabel:"29.4",group:"Derivative basics",title:"Numerical derivative",concept:"A derivative can be approximated by measuring a tiny change in function output.",objective:"Complete the finite-difference formula.",difficulty:"core",starterCode:`function numericalDerivative(f, x, h = 1e-5) {
  // TODO: return (f(x + h) - f(x)) / h.
  return 0;
}`,testCode:`const results = [];

function approxEqual(a, b, tolerance = 1e-3) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: approxEqual(actual, expected) });
}

check('derivative of x^2 at 3', numericalDerivative((x) => x * x, 3), 6);
check('derivative of 2x+1 at 5', numericalDerivative((x) => 2 * x + 1, 5), 2);
check('derivative of x^3 at 2', numericalDerivative((x) => x * x * x, 2), 12);

return results;`,hints:["Look at how much f changes after a tiny step h.","Divide output change by input change.","return (f(x + h) - f(x)) / h;"],solution:`function numericalDerivative(f, x, h = 1e-5) {
  return (f(x + h) - f(x)) / h;
}`,explanation:"Numerical derivatives are useful for checking gradients, though exact backprop is usually more efficient."},{id:"chain-rule-two-links",stepLabel:"30.1",group:"Chain rule",title:"Two-link chain rule",concept:"The chain rule multiplies local derivatives along a path.",objective:"Return outerGradient * innerGradient.",difficulty:"warmup",starterCode:`function chainTwo(outerGradient, innerGradient) {
  // TODO: return the product of the two local gradients.
  return 0;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('2 then 3', chainTwo(2, 3), 6);
check('-1 then 5', chainTwo(-1, 5), -5);
check('zero stops gradient', chainTwo(10, 0), 0);

return results;`,hints:["Chain rule multiplies derivatives.","If one local derivative is zero, the path gradient is zero.","return outerGradient * innerGradient;"],solution:`function chainTwo(outerGradient, innerGradient) {
  return outerGradient * innerGradient;
}`,explanation:"Backprop is repeated chain rule: gradients flow backward by multiplying local derivatives."},{id:"chain-through-square",stepLabel:"30.2",group:"Chain rule",title:"Chain through square",concept:"If y = z^2 and z depends on x, then dy/dx = 2z * dz/dx.",objective:"Return 2 * z * dzdx.",difficulty:"core",starterCode:`function chainThroughSquare(z, dzdx) {
  // y = z^2
  // TODO: return dy/dx.
  return 0;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('z=3 dzdx=2', chainThroughSquare(3, 2), 12);
check('z=-4 dzdx=1', chainThroughSquare(-4, 1), -8);
check('z=5 dzdx=0', chainThroughSquare(5, 0), 0);

return results;`,hints:["Derivative of z^2 with respect to z is 2z.","Then multiply by dz/dx.","return 2 * z * dzdx;"],solution:`function chainThroughSquare(z, dzdx) {
  return 2 * z * dzdx;
}`,explanation:"The outer function contributes 2z; the inner function contributes dz/dx."},{id:"chain-through-sigmoid",stepLabel:"30.3",group:"Chain rule",title:"Chain through sigmoid",concept:"The derivative of sigmoid output s with respect to its input is s(1-s).",objective:"Return upstreamGradient * s * (1 - s).",difficulty:"core",starterCode:`function chainThroughSigmoid(sigmoidOutput, upstreamGradient) {
  // TODO: return the downstream gradient.
  return 0;
}`,testCode:`const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: approxEqual(actual, expected) });
}

check('s=0.5 upstream=1', chainThroughSigmoid(0.5, 1), 0.25);
check('s=0.8 upstream=2', chainThroughSigmoid(0.8, 2), 0.32);
check('s=0.1 upstream=3', chainThroughSigmoid(0.1, 3), 0.27);

return results;`,hints:["Sigmoid derivative uses the output: s * (1 - s).","Multiply by upstreamGradient.","return upstreamGradient * sigmoidOutput * (1 - sigmoidOutput);"],solution:`function chainThroughSigmoid(sigmoidOutput, upstreamGradient) {
  return upstreamGradient * sigmoidOutput * (1 - sigmoidOutput);
}`,explanation:"Sigmoid gradients shrink near 0 and 1, which is one reason saturated sigmoids can learn slowly."},{id:"chain-rule-add-paths",stepLabel:"30.4",group:"Chain rule",title:"Add gradients from multiple paths",concept:"When one variable affects loss through multiple paths, gradients add.",objective:"Return pathA + pathB.",difficulty:"core",starterCode:`function addGradientPaths(pathA, pathB) {
  // TODO: return the total gradient from both paths.
  return 0;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('two positive paths', addGradientPaths(2, 3), 5);
check('opposing paths', addGradientPaths(10, -4), 6);
check('one path zero', addGradientPaths(7, 0), 7);

return results;`,hints:["Gradients from different downstream branches add together.","This happens in computation graphs with reused values.","return pathA + pathB;"],solution:`function addGradientPaths(pathA, pathB) {
  return pathA + pathB;
}`,explanation:"Backprop sums contributions when a value is used by more than one downstream operation."},{id:"neuron-weighted-input",stepLabel:"31.1",group:"One neuron",title:"Weighted input",concept:"A neuron first computes a dot product between weights and inputs.",objective:"Return dot(weights, x).",difficulty:"core",starterCode:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function weightedInput(weights, x) {
  // TODO: return the weighted sum.
  return 0;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('simple weighted input', weightedInput([1, 2], [3, 4]), 11);
check('zero weight', weightedInput([0, 5], [10, 2]), 10);
check('negative weight', weightedInput([-1, 2], [3, 5]), 7);

return results;`,hints:["A neuron uses the same dot product you learned earlier.","Use the dot helper.","return dot(weights, x);"],solution:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function weightedInput(weights, x) {
  return dot(weights, x);
}`,explanation:"Every dense neuron starts as a dot product."},{id:"neuron-add-bias",stepLabel:"31.2",group:"One neuron",title:"Add bias",concept:"A bias shifts the neuron before activation.",objective:"Return weighted sum plus bias.",difficulty:"warmup",starterCode:`function preActivation(weightedSum, bias) {
  // TODO: return weightedSum plus bias.
  return weightedSum;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('positive bias', preActivation(10, 2), 12);
check('negative bias', preActivation(10, -3), 7);
check('zero bias', preActivation(5, 0), 5);

return results;`,hints:["Bias is added after the weighted sum.","Return weightedSum + bias.","return weightedSum + bias;"],solution:`function preActivation(weightedSum, bias) {
  return weightedSum + bias;
}`,explanation:"Bias lets the neuron shift its decision boundary or activation threshold."},{id:"neuron-relu-forward",stepLabel:"31.3",group:"One neuron",title:"ReLU activation",concept:"ReLU keeps positive values and turns negative values into zero.",objective:"Return max(0, z).",difficulty:"warmup",starterCode:`function relu(z) {
  // TODO: return max(0, z).
  return z;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('positive', relu(3), 3);
check('negative', relu(-4), 0);
check('zero', relu(0), 0);

return results;`,hints:["Use Math.max.","ReLU is max(0, z).","return Math.max(0, z);"],solution:`function relu(z) {
  return Math.max(0, z);
}`,explanation:"ReLU adds nonlinearity by gating off negative pre-activations."},{id:"neuron-forward-full",stepLabel:"31.4",group:"One neuron",title:"Full neuron forward pass",concept:"A simple neuron computes ReLU(w dot x + b).",objective:"Return relu(dot(weights, x) + bias).",difficulty:"core",starterCode:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function relu(z) {
  return Math.max(0, z);
}

function neuronForward(weights, x, bias) {
  // TODO: return ReLU of weighted input plus bias.
  return 0;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('positive neuron', neuronForward([1, 2], [3, 4], -5), 6);
check('negative clipped', neuronForward([1, 1], [1, 1], -5), 0);
check('zero boundary', neuronForward([1, 1], [1, 1], -2), 0);

return results;`,hints:["First compute dot(weights, x) + bias.","Then pass it through relu.","return relu(dot(weights, x) + bias);"],solution:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function relu(z) {
  return Math.max(0, z);
}

function neuronForward(weights, x, bias) {
  return relu(dot(weights, x) + bias);
}`,explanation:"Dense neural networks are built from many versions of this pattern."},{id:"backprop-bias-gradient",stepLabel:"32.1",group:"One-neuron backprop",title:"Bias gradient",concept:"For z = w dot x + b, the derivative of z with respect to b is 1.",objective:"Return upstreamGradient.",difficulty:"warmup",starterCode:`function biasGradient(upstreamGradient) {
  // TODO: return dL/db.
  return 0;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('upstream 3', biasGradient(3), 3);
check('upstream -2', biasGradient(-2), -2);
check('upstream 0', biasGradient(0), 0);

return results;`,hints:["Bias is added directly.","dz/db = 1, so dL/db = upstreamGradient * 1.","return upstreamGradient;"],solution:`function biasGradient(upstreamGradient) {
  return upstreamGradient;
}`,explanation:"Bias receives the same upstream gradient because it shifts z by one unit per one unit of bias."},{id:"backprop-one-weight",stepLabel:"32.2",group:"One-neuron backprop",title:"One weight gradient",concept:"For z = w dot x + b, dL/dw_i = upstreamGradient * x_i.",objective:"Return upstreamGradient * inputValue.",difficulty:"core",starterCode:`function weightGradient(upstreamGradient, inputValue) {
  // TODO: return dL/dw for one weight.
  return 0;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('input 2 upstream 3', weightGradient(3, 2), 6);
check('input 0 upstream 3', weightGradient(3, 0), 0);
check('negative upstream', weightGradient(-2, 5), -10);

return results;`,hints:["A weight is multiplied by its input.","The input scales the gradient for that weight.","return upstreamGradient * inputValue;"],solution:`function weightGradient(upstreamGradient, inputValue) {
  return upstreamGradient * inputValue;
}`,explanation:"Weights connected to larger inputs receive larger gradient signals."},{id:"backprop-weight-vector",stepLabel:"32.3",group:"One-neuron backprop",title:"Weight-gradient vector",concept:"Each weight gradient is upstreamGradient times the matching input.",objective:"Push upstreamGradient * x[i] for each weight.",difficulty:"core",starterCode:`function weightGradients(upstreamGradient, x) {
  const gradients = [];

  for (let i = 0; i < x.length; i++) {
    // TODO: push the gradient for weight i.
    gradients.push(0);
  }

  return gradients;
}`,testCode:`const results = [];

function sameArray(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function check(name, actual, expected) {
  results.push({
    name,
    actual: JSON.stringify(actual),
    expected: JSON.stringify(expected),
    passed: sameArray(actual, expected),
  });
}

check('upstream 3', weightGradients(3, [1, 2, 3]), [3, 6, 9]);
check('upstream -2', weightGradients(-2, [1, 0, 4]), [-2, 0, -8]);
check('upstream 0', weightGradients(0, [5, 6]), [0, 0]);

return results;`,hints:["Loop over the input vector.","Each gradient is upstreamGradient times x[i].","gradients.push(upstreamGradient * x[i]);"],solution:`function weightGradients(upstreamGradient, x) {
  const gradients = [];

  for (let i = 0; i < x.length; i++) {
    gradients.push(upstreamGradient * x[i]);
  }

  return gradients;
}`,explanation:"Backprop through a dense neuron produces one gradient per weight."},{id:"backprop-input-gradient",stepLabel:"32.4",group:"One-neuron backprop",title:"Input gradients",concept:"The gradient into each input is upstreamGradient times the matching weight.",objective:"Push upstreamGradient * weights[i].",difficulty:"core",starterCode:`function inputGradients(upstreamGradient, weights) {
  const gradients = [];

  for (let i = 0; i < weights.length; i++) {
    // TODO: push the gradient for input i.
    gradients.push(0);
  }

  return gradients;
}`,testCode:`const results = [];

function sameArray(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function check(name, actual, expected) {
  results.push({
    name,
    actual: JSON.stringify(actual),
    expected: JSON.stringify(expected),
    passed: sameArray(actual, expected),
  });
}

check('upstream 3', inputGradients(3, [1, 2, 3]), [3, 6, 9]);
check('upstream -2', inputGradients(-2, [1, 0, 4]), [-2, 0, -8]);
check('upstream 0', inputGradients(0, [5, 6]), [0, 0]);

return results;`,hints:["Inputs receive gradients through weights.","Each input gradient is upstreamGradient times weights[i].","gradients.push(upstreamGradient * weights[i]);"],solution:`function inputGradients(upstreamGradient, weights) {
  const gradients = [];

  for (let i = 0; i < weights.length; i++) {
    gradients.push(upstreamGradient * weights[i]);
  }

  return gradients;
}`,explanation:"This is how gradients flow backward from one layer into the previous layer."},{id:"relu-derivative",stepLabel:"33.1",group:"Activation gradients",title:"ReLU derivative",concept:"ReLU passes gradient only when the input was positive.",objective:"Return 1 for positive z, otherwise 0.",difficulty:"warmup",starterCode:`function reluDerivative(z) {
  // TODO: return 1 if z > 0, otherwise 0.
  return 0;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('positive', reluDerivative(3), 1);
check('negative', reluDerivative(-4), 0);
check('zero', reluDerivative(0), 0);

return results;`,hints:["ReLU is active only when z > 0.","Use a ternary expression.","return z > 0 ? 1 : 0;"],solution:`function reluDerivative(z) {
  return z > 0 ? 1 : 0;
}`,explanation:"A negative ReLU input blocks gradient, which can create dead units."},{id:"relu-backprop",stepLabel:"33.2",group:"Activation gradients",title:"Backprop through ReLU",concept:"The upstream gradient is kept only if ReLU was active.",objective:"Multiply upstreamGradient by the ReLU derivative.",difficulty:"core",starterCode:`function reluDerivative(z) {
  return z > 0 ? 1 : 0;
}

function reluBackward(upstreamGradient, z) {
  // TODO: return upstreamGradient times reluDerivative(z).
  return upstreamGradient;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('active ReLU', reluBackward(5, 3), 5);
check('inactive ReLU', reluBackward(5, -3), 0);
check('zero ReLU', reluBackward(5, 0), 0);
check('negative upstream active', reluBackward(-2, 4), -2);

return results;`,hints:["Backprop multiplies by local derivative.","reluDerivative(z) is either 1 or 0.","return upstreamGradient * reluDerivative(z);"],solution:`function reluDerivative(z) {
  return z > 0 ? 1 : 0;
}

function reluBackward(upstreamGradient, z) {
  return upstreamGradient * reluDerivative(z);
}`,explanation:"ReLU either passes the gradient through unchanged or blocks it entirely."},{id:"sigmoid-derivative-output",stepLabel:"33.3",group:"Activation gradients",title:"Sigmoid derivative",concept:"If s = sigmoid(z), then ds/dz = s(1-s).",objective:"Return s * (1 - s).",difficulty:"core",starterCode:`function sigmoidDerivativeFromOutput(s) {
  // TODO: return s * (1 - s).
  return 0;
}`,testCode:`const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: approxEqual(actual, expected) });
}

check('s=0.5', sigmoidDerivativeFromOutput(0.5), 0.25);
check('s=0.8', sigmoidDerivativeFromOutput(0.8), 0.16);
check('s=0.1', sigmoidDerivativeFromOutput(0.1), 0.09);

return results;`,hints:["Use the sigmoid output s directly.","Derivative is s times one minus s.","return s * (1 - s);"],solution:`function sigmoidDerivativeFromOutput(s) {
  return s * (1 - s);
}`,explanation:"Sigmoid gradients are largest near 0.5 and small near saturated outputs 0 or 1."},{id:"sigmoid-backprop",stepLabel:"33.4",group:"Activation gradients",title:"Backprop through sigmoid",concept:"Sigmoid backprop multiplies upstream gradient by s(1-s).",objective:"Return upstreamGradient * s * (1 - s).",difficulty:"core",starterCode:`function sigmoidBackward(upstreamGradient, sigmoidOutput) {
  // TODO: apply the sigmoid local derivative.
  return 0;
}`,testCode:`const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: approxEqual(actual, expected) });
}

check('s=0.5 upstream=1', sigmoidBackward(1, 0.5), 0.25);
check('s=0.8 upstream=2', sigmoidBackward(2, 0.8), 0.32);
check('s=0.1 upstream=3', sigmoidBackward(3, 0.1), 0.27);

return results;`,hints:["Local derivative is sigmoidOutput * (1 - sigmoidOutput).","Multiply by upstreamGradient.","return upstreamGradient * sigmoidOutput * (1 - sigmoidOutput);"],solution:`function sigmoidBackward(upstreamGradient, sigmoidOutput) {
  return upstreamGradient * sigmoidOutput * (1 - sigmoidOutput);
}`,explanation:"Sigmoid saturation can shrink gradients during backprop."},{id:"one-hot-target",stepLabel:"34.1",group:"Softmax cross-entropy",title:"One-hot target",concept:"Classification targets are often represented as one-hot vectors.",objective:"Return 1 at targetIndex and 0 elsewhere.",difficulty:"warmup",starterCode:`function oneHot(numClasses, targetIndex) {
  const y = [];

  for (let i = 0; i < numClasses; i++) {
    // TODO: push 1 for the target index, otherwise 0.
    y.push(0);
  }

  return y;
}`,testCode:`const results = [];

function sameArray(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function check(name, actual, expected) {
  results.push({
    name,
    actual: JSON.stringify(actual),
    expected: JSON.stringify(expected),
    passed: sameArray(actual, expected),
  });
}

check('class 0 of 3', oneHot(3, 0), [1, 0, 0]);
check('class 1 of 3', oneHot(3, 1), [0, 1, 0]);
check('class 2 of 4', oneHot(4, 2), [0, 0, 1, 0]);

return results;`,hints:["Compare i with targetIndex.","Push 1 if they match, otherwise 0.","y.push(i === targetIndex ? 1 : 0);"],solution:`function oneHot(numClasses, targetIndex) {
  const y = [];

  for (let i = 0; i < numClasses; i++) {
    y.push(i === targetIndex ? 1 : 0);
  }

  return y;
}`,explanation:"A one-hot vector says which class is the true class."},{id:"cross-entropy-one-hot",stepLabel:"34.2",group:"Softmax cross-entropy",title:"Cross-entropy from true class probability",concept:"For a one-hot label, cross-entropy is -log(probability of the true class).",objective:"Return -Math.log(probabilities[targetIndex]).",difficulty:"core",starterCode:`function crossEntropyFromTarget(probabilities, targetIndex) {
  // TODO: return negative log probability of the true class.
  return 0;
}`,testCode:`const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: approxEqual(actual, expected) });
}

check('p=0.5', crossEntropyFromTarget([0.5, 0.5], 0), -Math.log(0.5));
check('p=0.8', crossEntropyFromTarget([0.1, 0.8, 0.1], 1), -Math.log(0.8));
check('p=0.25', crossEntropyFromTarget([0.25, 0.25, 0.5], 0), -Math.log(0.25));

return results;`,hints:["Only the probability assigned to the true class matters for one-hot cross-entropy.","Use probabilities[targetIndex].","return -Math.log(probabilities[targetIndex]);"],solution:`function crossEntropyFromTarget(probabilities, targetIndex) {
  return -Math.log(probabilities[targetIndex]);
}`,explanation:"Cross-entropy strongly penalizes assigning low probability to the true class."},{id:"softmax-cross-entropy-gradient",stepLabel:"34.3",group:"Softmax cross-entropy",title:"Softmax + CE gradient",concept:"For softmax followed by cross-entropy, the logit gradient is probabilities minus one-hot target.",objective:"Push probabilities[i] - target[i].",difficulty:"challenge",starterCode:`function softmaxCrossEntropyGradient(probabilities, targetIndex) {
  const gradient = [];

  for (let i = 0; i < probabilities.length; i++) {
    const target = i === targetIndex ? 1 : 0;

    // TODO: push probability minus target.
    gradient.push(0);
  }

  return gradient;
}`,testCode:`const results = [];

function approxArray(a, b, tolerance = 1e-9) {
  return a.length === b.length && a.every((value, index) => Math.abs(value - b[index]) <= tolerance);
}

function check(name, actual, expected) {
  results.push({
    name,
    actual: JSON.stringify(actual),
    expected: JSON.stringify(expected),
    passed: approxArray(actual, expected),
  });
}

check('binary target 0', softmaxCrossEntropyGradient([0.7, 0.3], 0), [-0.3, 0.3]);
check('binary target 1', softmaxCrossEntropyGradient([0.7, 0.3], 1), [0.7, -0.7]);
check('three classes', softmaxCrossEntropyGradient([0.1, 0.8, 0.1], 1), [0.1, -0.2, 0.1]);

return results;`,hints:["This is the famous simplification: gradient = p - y.","target is already 1 for the true class and 0 otherwise.","gradient.push(probabilities[i] - target);"],solution:`function softmaxCrossEntropyGradient(probabilities, targetIndex) {
  const gradient = [];

  for (let i = 0; i < probabilities.length; i++) {
    const target = i === targetIndex ? 1 : 0;
    gradient.push(probabilities[i] - target);
  }

  return gradient;
}`,explanation:"The true class gets pushed up when its probability is too low; other classes get pushed down."},{id:"softmax-gradient-sum-zero",stepLabel:"34.4",group:"Softmax cross-entropy",title:"Softmax gradient sums to zero",concept:"Softmax logits compete: increasing one class decreases others, so gradients sum to zero.",objective:"Return the sum of the gradient entries.",difficulty:"core",starterCode:`function gradientSum(gradient) {
  let total = 0;

  for (let i = 0; i < gradient.length; i++) {
    // TODO: add the current gradient entry.
    total += 0;
  }

  return total;
}`,testCode:`const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: approxEqual(actual, expected) });
}

check('binary gradient', gradientSum([-0.3, 0.3]), 0);
check('three-class gradient', gradientSum([0.1, -0.2, 0.1]), 0);
check('general sum', gradientSum([0.25, 0.25, -0.5]), 0);

return results;`,hints:["Loop over the gradient entries.","Add each entry into total.","total += gradient[i];"],solution:`function gradientSum(gradient) {
  let total = 0;

  for (let i = 0; i < gradient.length; i++) {
    total += gradient[i];
  }

  return total;
}`,explanation:"Softmax probabilities are coupled; probability mass shifts between classes."},{id:"batch-size",stepLabel:"35.1",group:"Batch matrix shapes",title:"Batch size",concept:"A batch matrix has one row per example.",objective:"Return the number of examples in X.",difficulty:"warmup",starterCode:`function batchSize(X) {
  // TODO: return the number of rows.
  return 0;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('two examples', batchSize([[1, 2], [3, 4]]), 2);
check('three examples', batchSize([[1], [2], [3]]), 3);
check('one example', batchSize([[5, 6, 7]]), 1);

return results;`,hints:["Rows are examples.","The number of rows is X.length.","return X.length;"],solution:`function batchSize(X) {
  return X.length;
}`,explanation:"In many ML libraries, a data batch X has shape batch x features."},{id:"feature-count",stepLabel:"35.2",group:"Batch matrix shapes",title:"Feature count",concept:"A batch matrix has one column per input feature.",objective:"Return the number of columns in X.",difficulty:"warmup",starterCode:`function featureCount(X) {
  // TODO: return the number of features.
  return 0;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('two features', featureCount([[1, 2], [3, 4]]), 2);
check('one feature', featureCount([[1], [2], [3]]), 1);
check('three features', featureCount([[5, 6, 7]]), 3);

return results;`,hints:["Features are columns.","The first row length gives the number of features.","return X[0].length;"],solution:`function featureCount(X) {
  return X[0].length;
}`,explanation:"The feature count determines how many input weights each neuron needs."},{id:"dense-output-shape",stepLabel:"35.3",group:"Batch matrix shapes",title:"Dense layer output shape",concept:"If X is batch x inputDim and W is inputDim x outputDim, then XW is batch x outputDim.",objective:"Return [batchSize, outputDim].",difficulty:"core",starterCode:`function denseOutputShape(X, W) {
  const batch = X.length;
  const outputDim = W[0].length;

  // TODO: return the output shape.
  return [];
}`,testCode:`const results = [];

function sameArray(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function check(name, actual, expected) {
  results.push({
    name,
    actual: JSON.stringify(actual),
    expected: JSON.stringify(expected),
    passed: sameArray(actual, expected),
  });
}

check('2x3 times 3x4', denseOutputShape([[1,2,3],[4,5,6]], [[1,2,3,4],[5,6,7,8],[9,10,11,12]]), [2, 4]);
check('1x2 times 2x3', denseOutputShape([[1,2]], [[1,2,3],[4,5,6]]), [1, 3]);
check('3x1 times 1x2', denseOutputShape([[1],[2],[3]], [[4,5]]), [3, 2]);

return results;`,hints:["Rows come from X.","Output columns come from W.","return [batch, outputDim];"],solution:`function denseOutputShape(X, W) {
  const batch = X.length;
  const outputDim = W[0].length;
  return [batch, outputDim];
}`,explanation:"Dense layers are matrix multiplication with a batch dimension."},{id:"dense-shape-compatible",stepLabel:"35.4",group:"Batch matrix shapes",title:"Dense layer shape check",concept:"The feature count of X must match the input dimension of W.",objective:"Return whether X and W can multiply.",difficulty:"core",starterCode:`function denseShapesCompatible(X, W) {
  const inputFeatures = X[0].length;
  const weightInputDim = W.length;

  // TODO: return whether the inner dimensions match.
  return false;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('2x3 and 3x4 compatible', denseShapesCompatible([[1,2,3],[4,5,6]], [[1,2,3,4],[5,6,7,8],[9,10,11,12]]), true);
check('2x2 and 3x4 incompatible', denseShapesCompatible([[1,2],[3,4]], [[1,2,3,4],[5,6,7,8],[9,10,11,12]]), false);
check('1x2 and 2x1 compatible', denseShapesCompatible([[1,2]], [[3],[4]]), true);

return results;`,hints:["The inner dimensions must match.","Compare X[0].length with W.length.","return inputFeatures === weightInputDim;"],solution:`function denseShapesCompatible(X, W) {
  const inputFeatures = X[0].length;
  const weightInputDim = W.length;
  return inputFeatures === weightInputDim;
}`,explanation:"Many neural-network bugs are shape bugs. This check catches the most common dense-layer mismatch."},{id:"dense-add-bias-each-row",stepLabel:"35.5",group:"Batch matrix shapes",title:"Add bias to each row",concept:"Dense-layer bias is added to every example in the batch.",objective:"Add bias[col] to each output cell.",difficulty:"challenge",starterCode:`function addBias(Y, bias) {
  const result = [];

  for (let row = 0; row < Y.length; row++) {
    const values = [];

    for (let col = 0; col < Y[0].length; col++) {
      // TODO: add the bias for this output feature.
      values.push(Y[row][col]);
    }

    result.push(values);
  }

  return result;
}`,testCode:`const results = [];

function sameMatrix(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function check(name, actual, expected) {
  results.push({
    name,
    actual: JSON.stringify(actual),
    expected: JSON.stringify(expected),
    passed: sameMatrix(actual, expected),
  });
}

check('add bias to two rows', addBias([[1,2],[3,4]], [10,20]), [[11,22],[13,24]]);
check('zero bias', addBias([[1,2,3]], [0,0,0]), [[1,2,3]]);
check('negative bias', addBias([[5,5]], [-1,2]), [[4,7]]);

return results;`,hints:["Bias has one value per output column.","Use bias[col].","values.push(Y[row][col] + bias[col]);"],solution:`function addBias(Y, bias) {
  const result = [];

  for (let row = 0; row < Y.length; row++) {
    const values = [];

    for (let col = 0; col < Y[0].length; col++) {
      values.push(Y[row][col] + bias[col]);
    }

    result.push(values);
  }

  return result;
}`,explanation:"Bias broadcasts across the batch: every example gets the same output-feature offsets."}];function te(){return e.jsx(K,{exercises:ee})}const re=[[1,2],[3,1]],ne=[[2,1,3],[1,4,2]],oe=[[4,9,7],[7,7,11]],b=[{row:0,col:0,hint:"Multiply Row 1 of A with Column 1 of B: (1×2) + (2×1)",answer:4},{row:0,col:1,hint:"Multiply Row 1 of A with Column 2 of B: (1×1) + (2×4)",answer:9},{row:0,col:2,hint:"Multiply Row 1 of A with Column 3 of B: (1×3) + (2×2)",answer:7},{row:1,col:0,hint:"Multiply Row 2 of A with Column 1 of B: (3×2) + (1×1)",answer:7},{row:1,col:1,hint:"Multiply Row 2 of A with Column 2 of B: (3×1) + (1×4)",answer:7},{row:1,col:2,hint:"Multiply Row 2 of A with Column 3 of B: (3×3) + (1×2)",answer:11}];function se(){const[n,u]=y.useState(0),[c,r]=y.useState(""),[a,s]=y.useState(""),[w,f]=y.useState(!1),[k,A]=y.useState(Array(6).fill(null)),[D,M]=y.useState(!1),[B,S]=y.useState(0),[O,T]=y.useState(0),o=b[n],j=()=>{const i=parseInt(c,10);if(T(d=>d+1),i===o.answer){s("✓ Correct!"),S(g=>g+1);const d=[...k];d[n]=i,A(d),setTimeout(()=>{n<b.length-1?(u(g=>g+1),r(""),s(""),f(!1)):(M(!0),s("🎉 Excellent! You completed all steps!"))},1e3)}else s("✗ Not quite. Try again or ask for a hint.")},C=()=>{f(!0)},L=()=>{u(0),r(""),s(""),f(!1),A(Array(6).fill(null)),M(!1),S(0),T(0)},q=i=>{i.key==="Enter"&&c.trim()!==""&&j()},z=i=>n<b.length&&b[n].row===i,P=i=>n<b.length&&b[n].col===i;return e.jsxs("div",{className:"flex flex-col items-center p-3 h-full",children:[e.jsx("h2",{className:"text-xl font-bold text-gray-800 mb-2",children:"Practice Exercise"}),e.jsxs("div",{className:"bg-white rounded-lg shadow-lg p-4 w-full",children:[e.jsxs("div",{className:"flex items-center justify-center gap-2 flex-wrap",children:[e.jsxs("div",{className:"flex flex-col items-center",children:[e.jsx("span",{className:"text-lg font-bold mb-1",children:"A"}),e.jsx("div",{className:"grid grid-cols-2 gap-1",children:re.map((i,d)=>i.map((g,x)=>e.jsx("div",{className:`w-10 h-10 flex items-center justify-center font-bold text-black rounded ${z(d)?"bg-blue-300 scale-110 ring-2 ring-blue-500":"bg-blue-400"} transition-all`,children:g},`a-${d}-${x}`)))})]}),e.jsx("span",{className:"text-2xl font-bold mx-2",children:"×"}),e.jsxs("div",{className:"flex flex-col items-center",children:[e.jsx("span",{className:"text-lg font-bold mb-1",children:"B"}),e.jsx("div",{className:"grid grid-cols-3 gap-1",children:ne.map((i,d)=>i.map((g,x)=>e.jsx("div",{className:`w-10 h-10 flex items-center justify-center font-bold text-black rounded ${P(x)?"bg-green-300 scale-110 ring-2 ring-green-500":"bg-green-400"} transition-all`,children:g},`b-${d}-${x}`)))})]}),e.jsx("span",{className:"text-2xl font-bold mx-2",children:"="}),e.jsxs("div",{className:"flex flex-col items-center",children:[e.jsx("span",{className:"text-lg font-bold mb-1",children:"C"}),e.jsx("div",{className:"grid grid-cols-3 gap-1",children:oe.map((i,d)=>i.map((g,x)=>{const N=d*3+x,E=n===N,t=k[N]!==null;return e.jsx("div",{className:`w-10 h-10 flex items-center justify-center font-bold text-black rounded transition-all ${E?"bg-yellow-300 ring-2 ring-yellow-500 scale-110":t?"bg-orange-400":"bg-orange-200"}`,children:t?k[N]:"?"},`r-${d}-${x}`)}))})]})]}),e.jsxs("div",{className:"mt-4 text-center",children:[e.jsxs("p",{className:"text-gray-700 font-medium",children:["Step ",n+1," of ",b.length,": Calculate C[",o.row+1,"][",o.col+1,"]"]}),e.jsxs("p",{className:"text-sm text-gray-700 mt-1",children:["Row ",o.row+1," of A × Column ",o.col+1," of B"]})]})]}),D?e.jsx("div",{className:"mt-4 w-full max-w-sm text-center",children:e.jsxs("div",{className:"p-4 bg-green-100 rounded-lg border border-green-300",children:[e.jsx("p",{className:"text-green-700 font-bold text-lg",children:"🎉 Congratulations!"}),e.jsxs("p",{className:"text-green-600 mt-2",children:["Score: ",B," / ",b.length," correct"]}),e.jsxs("p",{className:"text-sm",children:["Total attempts: ",O]})]})}):e.jsxs("div",{className:"mt-4 w-full max-w-sm",children:[e.jsxs("div",{className:"flex gap-2",children:[e.jsx("input",{type:"number",value:c,onChange:i=>r(i.target.value),onKeyPress:q,placeholder:"Your answer...",className:"flex-1 px-4 py-2 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none text-center text-lg font-bold"}),e.jsx("button",{onClick:j,disabled:c.trim()==="",className:"px-4 py-2 bg-green-500 hover:bg-green-600 disabled:bg-gray-400 text-white font-bold rounded-lg transition-colors",children:"Submit"})]}),e.jsx("button",{onClick:C,className:"mt-2 w-full px-4 py-2 bg-yellow-500 hover:bg-yellow-600 text-white font-bold rounded-lg transition-colors",children:"💡 Show Hint"}),w&&e.jsx("div",{className:"mt-2 p-3 bg-yellow-100 rounded-lg border border-yellow-300",children:e.jsx("p",{className:"text-sm",children:o.hint})}),a&&e.jsx("div",{className:`mt-2 p-3 rounded-lg text-center font-bold ${a.includes("✓")?"bg-green-100 text-green-700":"bg-red-100 text-red-700"}`,children:a})]}),e.jsxs("div",{className:"mt-4 flex items-center gap-4",children:[e.jsxs("div",{className:"text-sm text-gray-800",children:["Progress: ",k.filter(i=>i!==null).length," / ",b.length]}),e.jsx("button",{onClick:L,className:"px-4 py-2 bg-red-500 hover:bg-red-600 text-white font-bold rounded-lg transition-colors text-sm",children:"↺ Reset"})]}),e.jsx("div",{className:"mt-8 w-full",children:e.jsx(te,{})})]})}export{se as default};
