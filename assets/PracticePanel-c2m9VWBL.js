import{a as v,j as e,k as y}from"./react-vendor-Cdu38Wyn.js";import{F as G,I as U,aZ as F,aQ as W,ar as X,$ as Q,_ as H}from"./icons-C7miCxLM.js";let K=1;function Y({userCode:a,testCode:u,timeoutMs:c=1200}){const r=K++;return new Promise(o=>{const s=new Worker(new URL("/ml-animations/assets/jsEvalWorker-BiN8n73y.js",import.meta.url),{type:"module"}),k=window.setTimeout(()=>{s.terminate(),o({ok:!1,results:[],error:"Execution timed out. Check for an infinite loop."})},c);s.onmessage=g=>{g.data.id===r&&(window.clearTimeout(k),s.terminate(),o({ok:g.data.ok,results:g.data.results||[],error:g.data.error}))},s.postMessage({id:r,userCode:a,testCode:u})})}function I(a,u){return u?"error":a!=null&&a.length?a.every(c=>c.passed)?"passed":"failed":"idle"}function z(a){return typeof a=="string"?a:JSON.stringify(a)}const $=/(\/\/.*|\/\*[\s\S]*?\*\/|(["'`])(?:\\.|(?!\2)[^\\])*\2|\b(?:const|let|var|function|return|if|else|for|while|new|throw|true|false|null|undefined)\b|\b\d+(?:\.\d+)?\b|\b[a-zA-Z_$][\w$]*(?=\s*\())/g;function _(a){const u=[];let c=0;for(const r of a.matchAll($)){r.index>c&&u.push(a.slice(c,r.index));const o=r[0];let s="plain";o.startsWith("//")||o.startsWith("/*")?s="comment":/^["'`]/.test(o)?s="string":/^\d/.test(o)?s="number":/^(const|let|var|function|return|if|else|for|while|new|throw|true|false|null|undefined)$/.test(o)?s="keyword":s="call",u.push(e.jsx("span",{className:`ua-code-token-${s}`,children:o},`${r.index}-${o}`)),c=r.index+o.length}return c<a.length&&u.push(a.slice(c)),u}function Z({exercises:a}){var P;const[u,c]=v.useState(0),r=a[u],o=v.useRef(null),[s,k]=v.useState(()=>Object.fromEntries(a.map(t=>[t.id,t.starterCode]))),[g,w]=v.useState({}),[O,N]=v.useState({}),[M,B]=v.useState(!1),[S,A]=v.useState(!1),T=s[r.id],n=O[r.id],C=I(n==null?void 0:n.results,n==null?void 0:n.error),j=g[r.id]||0,D=r.hints.slice(0,j),L=!!(n||j>0),J=v.useMemo(()=>{const t=[];return a.forEach((l,h)=>{const m=l.group||"Exercises",p=t[t.length-1];if((p==null?void 0:p.name)===m){p.items.push({exercise:l,index:h});return}t.push({name:m,items:[{exercise:l,index:h}]})}),t},[a]);async function E(){B(!0),A(!1);const t=await Y({userCode:T,testCode:r.testCode});N(l=>({...l,[r.id]:t})),B(!1)}function i(){k(t=>({...t,[r.id]:r.starterCode})),N(t=>({...t,[r.id]:null})),w(t=>({...t,[r.id]:0})),A(!1)}function d(){w(t=>({...t,[r.id]:Math.min(r.hints.length,j+1)}))}function f(){k(t=>({...t,[r.id]:r.solution})),A(!1)}function b(t){o.current&&(o.current.scrollTop=t.currentTarget.scrollTop,o.current.scrollLeft=t.currentTarget.scrollLeft)}const q=Object.values(O).filter(t=>{var l;return((l=t==null?void 0:t.results)==null?void 0:l.length)&&t.results.every(h=>h.passed)}).length;return e.jsxs("section",{className:"ua-codefix-lab",children:[e.jsxs("div",{className:"ua-codefix-head",children:[e.jsx("span",{children:"Code Completion-style lab"}),e.jsx("h2",{children:"Fix the TODOs, run the tests"}),e.jsx("p",{children:"Each exercise is almost complete. Change the smallest piece of code needed to make the tests pass."})]}),e.jsx("div",{className:"ua-codefix-progress",children:J.map(t=>{const l=t.items.filter(({exercise:h})=>{var p;const m=O[h.id];return((p=m==null?void 0:m.results)==null?void 0:p.length)&&m.results.every(R=>R.passed)}).length;return e.jsxs("div",{className:"ua-codefix-progress-group",children:[e.jsxs("div",{className:"ua-codefix-progress-label",children:[e.jsx("strong",{children:t.name}),e.jsxs("span",{children:[l,"/",t.items.length]})]}),e.jsx("div",{className:"ua-codefix-progress-steps",children:t.items.map(({exercise:h,index:m})=>{const p=O[h.id],R=I(p==null?void 0:p.results,p==null?void 0:p.error),V=R==="passed"?G:U;return e.jsxs("button",{type:"button",onClick:()=>{c(m),A(!1)},className:`ua-codefix-step ${m===u?"active":""} ${R}`,children:[e.jsx(V,{size:15,"aria-hidden":"true"}),e.jsxs("span",{children:[h.stepLabel||`${m+1}.`," ",h.title]})]},h.id)})})]},t.name)})}),e.jsxs("div",{className:"ua-codefix-grid",children:[e.jsxs("article",{className:"ua-codefix-card ua-codefix-instructions",children:[e.jsx("span",{children:r.difficulty}),e.jsx("h3",{children:r.title}),e.jsx("p",{children:r.objective}),e.jsxs("div",{className:"ua-codefix-concept",children:[e.jsx("strong",{children:"Concept"}),e.jsx("p",{children:r.concept})]}),e.jsxs("div",{className:"ua-codefix-explanation",children:[e.jsx("strong",{children:"After you pass"}),e.jsx("p",{children:r.explanation})]})]}),e.jsxs("article",{className:"ua-codefix-card ua-codefix-editor-card",children:[e.jsxs("div",{className:"ua-codefix-card-head",children:[e.jsxs("div",{children:[e.jsx("span",{children:"Editor"}),e.jsx("h3",{children:"Complete the TODO"})]}),e.jsxs("button",{type:"button",onClick:i,children:[e.jsx(F,{size:14,"aria-hidden":"true"}),"Reset"]})]}),e.jsxs("div",{className:"ua-codefix-editor-shell",children:[e.jsx("pre",{className:"ua-codefix-highlight","aria-hidden":"true",ref:o,children:_(T)}),e.jsx("textarea",{className:"ua-codefix-editor",value:T,spellCheck:!1,"aria-label":`${r.title} code editor`,onScroll:b,onChange:t=>k(l=>({...l,[r.id]:t.target.value}))})]}),e.jsxs("div",{className:"ua-codefix-actions",children:[e.jsxs("button",{type:"button",onClick:E,disabled:M,children:[e.jsx(W,{size:15,"aria-hidden":"true"}),M?"Running...":"Run tests"]}),e.jsxs("button",{type:"button",onClick:d,disabled:j>=r.hints.length,children:[e.jsx(X,{size:15,"aria-hidden":"true"}),j===0?"Show hint":"Next hint"]}),e.jsxs("button",{type:"button",onClick:()=>A(t=>!t),disabled:!L,title:L?void 0:"Run tests or use a hint before revealing the solution.",children:[S?e.jsx(Q,{size:15,"aria-hidden":"true"}):e.jsx(H,{size:15,"aria-hidden":"true"}),S?"Hide solution":L?"See solution":"Try first"]})]})]}),e.jsxs("article",{className:"ua-codefix-card ua-codefix-feedback",children:[e.jsx("span",{children:"Checks"}),e.jsxs("h3",{children:[C==="passed"&&"All tests passed",C==="failed"&&"Keep going",C==="error"&&"Code error",C==="idle"&&"Run tests to begin"]}),(n==null?void 0:n.error)&&e.jsx("pre",{className:"ua-codefix-error",children:n.error}),((P=n==null?void 0:n.results)==null?void 0:P.length)>0?e.jsx("ul",{className:"ua-codefix-checks",children:n.results.map(t=>e.jsxs("li",{className:t.passed?"passed":"failed",children:[e.jsxs("strong",{children:[t.passed?"Pass":"Fail",": ",t.name]}),!t.passed&&e.jsxs("small",{children:["Expected ",z(t.expected),", got ",z(t.actual)]})]},t.name))}):e.jsx("p",{className:"ua-codefix-empty",children:"Run the tests. If one fails, use the smallest hint that helps."}),D.length>0&&e.jsxs("div",{className:"ua-codefix-hints",children:[e.jsx("strong",{children:"Hints"}),D.map((t,l)=>t.includes(`
`)?e.jsxs("div",{className:"ua-codefix-hint",children:[e.jsxs("b",{children:["Hint ",l+1,":"]}),e.jsx("pre",{className:"ua-codefix-hint-code",children:t})]},t):e.jsxs("p",{children:[e.jsxs("b",{children:["Hint ",l+1,":"]})," ",t]},t))]}),S&&e.jsxs("div",{className:"ua-codefix-solution",children:[e.jsx("strong",{children:"Solution"}),e.jsx("pre",{children:r.solution}),e.jsx("button",{type:"button",onClick:f,children:"Apply solution to editor"})]})]})]}),e.jsxs("div",{className:"ua-codefix-footer",children:[e.jsxs("strong",{children:[q," / ",a.length]}),e.jsx("span",{children:"exercises passed"})]})]})}const ee=[{id:"dot-product-first-pair",stepLabel:"1.1",group:"Dot product",title:"First matching pair",concept:"A dot product starts by multiplying entries with the same index.",objective:"Replace one number with the first pair product.",difficulty:"warmup",starterCode:`function firstPairProduct(a, b) {
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
}`,explanation:"Bias broadcasts across the batch: every example gets the same output-feature offsets."},{id:"dense-one-output-neuron",stepLabel:"36.1",group:"Mini neural network layer",title:"One dense output",concept:"One dense-layer output is one input vector dotted with one weight vector plus bias.",objective:"Return dot(x, weights) + bias.",difficulty:"core",starterCode:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function denseOne(x, weights, bias) {
  // TODO: return dot(x, weights) + bias.
  return 0;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('simple dense output', denseOne([1, 2], [3, 4], 0), 11);
check('with bias', denseOne([1, 2], [3, 4], -1), 10);
check('negative weight', denseOne([-1, 2], [3, 5], 1), 8);

return results;`,hints:["A dense neuron is a dot product plus a bias.","Use the dot helper.","return dot(x, weights) + bias;"],solution:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function denseOne(x, weights, bias) {
  return dot(x, weights) + bias;
}`,explanation:"A dense layer is many versions of this one-neuron calculation."},{id:"dense-multiple-outputs",stepLabel:"36.2",group:"Mini neural network layer",title:"Multiple dense outputs",concept:"A dense layer has one weight vector and one bias per output feature.",objective:"Push one output for each output weight vector.",difficulty:"core",starterCode:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function denseLayer(x, weightColumns, biases) {
  const outputs = [];

  for (let j = 0; j < weightColumns.length; j++) {
    // TODO: push dot(x, weightColumns[j]) + biases[j].
    outputs.push(0);
  }

  return outputs;
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

check('two outputs', denseLayer([1, 2], [[3, 4], [5, 6]], [0, 1]), [11, 18]);
check('three outputs', denseLayer([2, 1], [[1, 0], [0, 1], [1, 1]], [0, 0, -1]), [2, 1, 2]);

return results;`,hints:["Each output j has its own weight vector and bias.","Use dot(x, weightColumns[j]) + biases[j].","outputs.push(dot(x, weightColumns[j]) + biases[j]);"],solution:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function denseLayer(x, weightColumns, biases) {
  const outputs = [];

  for (let j = 0; j < weightColumns.length; j++) {
    outputs.push(dot(x, weightColumns[j]) + biases[j]);
  }

  return outputs;
}`,explanation:"A dense layer maps one input vector to several output features by using several weight vectors."},{id:"dense-relu-vector",stepLabel:"36.3",group:"Mini neural network layer",title:"ReLU on a vector",concept:"Neural layers apply activations element by element.",objective:"Push Math.max(0, values[i]) for every coordinate.",difficulty:"warmup",starterCode:`function reluVector(values) {
  const activated = [];

  for (let i = 0; i < values.length; i++) {
    // TODO: push ReLU of values[i].
    activated.push(values[i]);
  }

  return activated;
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

check('mixed values', reluVector([-2, 0, 3]), [0, 0, 3]);
check('all positive', reluVector([1, 2, 3]), [1, 2, 3]);
check('all negative', reluVector([-1, -2]), [0, 0]);

return results;`,hints:["ReLU is max(0, value).","Use Math.max(0, values[i]).","activated.push(Math.max(0, values[i]));"],solution:`function reluVector(values) {
  const activated = [];

  for (let i = 0; i < values.length; i++) {
    activated.push(Math.max(0, values[i]));
  }

  return activated;
}`,explanation:"Activations usually apply coordinate by coordinate after a linear transformation."},{id:"two-layer-mini-network",stepLabel:"36.4",group:"Mini neural network layer",title:"Two-layer mini network",concept:"A simple network can be dense -> ReLU -> dense.",objective:"Feed hidden activations into the output layer.",difficulty:"challenge",starterCode:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function reluVector(values) {
  return values.map((value) => Math.max(0, value));
}

function denseLayer(x, weightColumns, biases) {
  return weightColumns.map((weights, j) => dot(x, weights) + biases[j]);
}

function twoLayerNetwork(x, hiddenWeights, hiddenBiases, outputWeights, outputBiases) {
  const hiddenPre = denseLayer(x, hiddenWeights, hiddenBiases);
  const hidden = reluVector(hiddenPre);

  // TODO: return the output dense layer applied to hidden.
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

check('two-layer network', twoLayerNetwork([1, 2], [[1, 0], [0, 1]], [0, 0], [[1, 1]], [0]), [3]);
check('hidden ReLU clips negative', twoLayerNetwork([-1, 2], [[1, 0], [0, 1]], [0, 0], [[1, 1]], [0]), [2]);

return results;`,hints:["The hidden activations are already computed.","Use denseLayer(hidden, outputWeights, outputBiases).","return denseLayer(hidden, outputWeights, outputBiases);"],solution:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function reluVector(values) {
  return values.map((value) => Math.max(0, value));
}

function denseLayer(x, weightColumns, biases) {
  return weightColumns.map((weights, j) => dot(x, weights) + biases[j]);
}

function twoLayerNetwork(x, hiddenWeights, hiddenBiases, outputWeights, outputBiases) {
  const hiddenPre = denseLayer(x, hiddenWeights, hiddenBiases);
  const hidden = reluVector(hiddenPre);
  return denseLayer(hidden, outputWeights, outputBiases);
}`,explanation:"Stacking layers means using one layer output as the next layer input."},{id:"training-loop-one-prediction",stepLabel:"37.1",group:"Training loop mechanics",title:"One prediction",concept:"Training begins with a prediction from current parameters.",objective:"Return weight * x + bias.",difficulty:"warmup",starterCode:`function predictLinear(x, weight, bias) {
  // TODO: return weight * x + bias.
  return 0;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('predict 2*3+1', predictLinear(3, 2, 1), 7);
check('predict -1*4+2', predictLinear(4, -1, 2), -2);
check('bias only', predictLinear(10, 0, 5), 5);

return results;`,hints:["Linear prediction is slope times input plus bias.","Use weight * x + bias.","return weight * x + bias;"],solution:`function predictLinear(x, weight, bias) {
  return weight * x + bias;
}`,explanation:"A training loop repeatedly predicts, measures error, computes gradients, and updates parameters."},{id:"training-loop-one-loss",stepLabel:"37.2",group:"Training loop mechanics",title:"One-example loss",concept:"Squared error loss measures prediction error squared.",objective:"Return (prediction - target)^2.",difficulty:"warmup",starterCode:`function squaredLoss(prediction, target) {
  const error = prediction - target;

  // TODO: return squared error.
  return 0;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('error 3', squaredLoss(10, 7), 9);
check('error -5', squaredLoss(4, 9), 25);
check('perfect', squaredLoss(5, 5), 0);

return results;`,hints:["Squared error means error times error.","The error variable is already computed.","return error * error;"],solution:`function squaredLoss(prediction, target) {
  const error = prediction - target;
  return error * error;
}`,explanation:"The loss is the number the training loop tries to reduce."},{id:"training-loop-average-loss",stepLabel:"37.3",group:"Training loop mechanics",title:"Average batch loss",concept:"Batch loss averages losses over examples.",objective:"Divide total loss by the number of examples.",difficulty:"core",starterCode:`function averageLoss(losses) {
  let total = 0;

  for (let i = 0; i < losses.length; i++) {
    total += losses[i];
  }

  // TODO: return the average loss.
  return total;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('average [1,2,3]', averageLoss([1, 2, 3]), 2);
check('average [10,20]', averageLoss([10, 20]), 15);
check('average zeros', averageLoss([0, 0, 0]), 0);

return results;`,hints:["Average means total divided by count.","The count is losses.length.","return total / losses.length;"],solution:`function averageLoss(losses) {
  let total = 0;

  for (let i = 0; i < losses.length; i++) {
    total += losses[i];
  }

  return total / losses.length;
}`,explanation:"Training reports average loss so batches of different sizes are comparable."},{id:"training-loop-step-summary",stepLabel:"37.4",group:"Training loop mechanics",title:"One training step",concept:"A training step computes prediction, error, gradients, and updated parameters.",objective:"Return updated weight after one gradient step.",difficulty:"challenge",starterCode:`function oneStepWeightUpdate(x, target, weight, bias, learningRate) {
  const prediction = weight * x + bias;
  const error = prediction - target;

  // Gradient of squared error without the factor 2 for simplicity.
  const weightGradient = error * x;

  // TODO: return updated weight.
  return weight;
}`,testCode:`const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: approxEqual(actual, expected) });
}

check('update decreases high prediction', oneStepWeightUpdate(2, 3, 2, 0, 0.1), 1.8);
check('update increases low prediction', oneStepWeightUpdate(2, 10, 2, 0, 0.1), 3.2);
check('perfect no update', oneStepWeightUpdate(2, 4, 2, 0, 0.1), 2);

return results;`,hints:["Gradient descent subtracts learningRate * gradient.","The weightGradient is already computed.","return weight - learningRate * weightGradient;"],solution:`function oneStepWeightUpdate(x, target, weight, bias, learningRate) {
  const prediction = weight * x + bias;
  const error = prediction - target;
  const weightGradient = error * x;

  return weight - learningRate * weightGradient;
}`,explanation:"One training step nudges parameters opposite the gradient."},{id:"optimizer-sgd-update",stepLabel:"38.1",group:"Optimizer updates",title:"SGD update",concept:"Stochastic gradient descent subtracts learningRate times gradient.",objective:"Return parameter - learningRate * gradient.",difficulty:"warmup",starterCode:`function sgdUpdate(parameter, gradient, learningRate) {
  // TODO: return the updated parameter.
  return parameter;
}`,testCode:`const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: approxEqual(actual, expected) });
}

check('positive gradient', sgdUpdate(1, 3, 0.1), 0.7);
check('negative gradient', sgdUpdate(1, -2, 0.5), 2);
check('zero gradient', sgdUpdate(5, 0, 0.1), 5);

return results;`,hints:["Move opposite the gradient.","Subtract learningRate * gradient.","return parameter - learningRate * gradient;"],solution:`function sgdUpdate(parameter, gradient, learningRate) {
  return parameter - learningRate * gradient;
}`,explanation:"SGD is the simplest optimizer: follow the negative gradient."},{id:"optimizer-momentum-velocity",stepLabel:"38.2",group:"Optimizer updates",title:"Momentum velocity",concept:"Momentum keeps a moving velocity of recent gradients.",objective:"Return beta * velocity + gradient.",difficulty:"core",starterCode:`function updateVelocity(velocity, gradient, beta) {
  // TODO: combine old velocity and current gradient.
  return 0;
}`,testCode:`const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: approxEqual(actual, expected) });
}

check('new velocity', updateVelocity(0, 3, 0.9), 3);
check('carry velocity', updateVelocity(10, 3, 0.9), 12);
check('negative gradient', updateVelocity(5, -2, 0.8), 2);

return results;`,hints:["Momentum mixes previous velocity with current gradient.","Use beta * velocity + gradient.","return beta * velocity + gradient;"],solution:`function updateVelocity(velocity, gradient, beta) {
  return beta * velocity + gradient;
}`,explanation:"Momentum smooths updates by remembering previous gradient direction."},{id:"optimizer-momentum-update",stepLabel:"38.3",group:"Optimizer updates",title:"Momentum update",concept:"Momentum updates parameters using velocity rather than the raw current gradient only.",objective:"Subtract learningRate times velocity.",difficulty:"core",starterCode:`function momentumParameterUpdate(parameter, velocity, learningRate) {
  // TODO: update parameter using velocity.
  return parameter;
}`,testCode:`const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: approxEqual(actual, expected) });
}

check('velocity 3', momentumParameterUpdate(1, 3, 0.1), 0.7);
check('negative velocity', momentumParameterUpdate(1, -2, 0.5), 2);
check('zero velocity', momentumParameterUpdate(5, 0, 0.1), 5);

return results;`,hints:["Velocity acts like the gradient direction to follow.","Subtract learningRate * velocity.","return parameter - learningRate * velocity;"],solution:`function momentumParameterUpdate(parameter, velocity, learningRate) {
  return parameter - learningRate * velocity;
}`,explanation:"Momentum can accelerate updates in consistent directions and damp zig-zagging."},{id:"optimizer-adam-first-moment",stepLabel:"38.4",group:"Optimizer updates",title:"Adam first moment",concept:"Adam keeps an exponential moving average of gradients.",objective:"Return beta1 * m + (1 - beta1) * gradient.",difficulty:"core",starterCode:`function adamFirstMoment(m, gradient, beta1) {
  // TODO: update the first moment estimate.
  return 0;
}`,testCode:`const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: approxEqual(actual, expected) });
}

check('first moment from zero', adamFirstMoment(0, 10, 0.9), 1);
check('carry moment', adamFirstMoment(5, 10, 0.9), 5.5);
check('negative gradient', adamFirstMoment(1, -9, 0.8), -1);

return results;`,hints:["Adam first moment is a weighted average of old m and new gradient.","Use beta1 for old m and 1 - beta1 for gradient.","return beta1 * m + (1 - beta1) * gradient;"],solution:`function adamFirstMoment(m, gradient, beta1) {
  return beta1 * m + (1 - beta1) * gradient;
}`,explanation:"Adam first moment behaves like momentum but with exponential averaging."},{id:"optimizer-adam-second-moment",stepLabel:"38.5",group:"Optimizer updates",title:"Adam second moment",concept:"Adam tracks an exponential moving average of squared gradients.",objective:"Return beta2 * v + (1 - beta2) * gradient squared.",difficulty:"core",starterCode:`function adamSecondMoment(v, gradient, beta2) {
  // TODO: update the second moment estimate.
  return 0;
}`,testCode:`const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: approxEqual(actual, expected) });
}

check('second moment from zero', adamSecondMoment(0, 10, 0.99), 1);
check('carry second moment', adamSecondMoment(5, 10, 0.9), 14.5);
check('negative gradient squares', adamSecondMoment(0, -3, 0.9), 0.9);

return results;`,hints:["Use gradient * gradient.","Mix old v with squared gradient.","return beta2 * v + (1 - beta2) * gradient * gradient;"],solution:`function adamSecondMoment(v, gradient, beta2) {
  return beta2 * v + (1 - beta2) * gradient * gradient;
}`,explanation:"Adam uses the second moment to scale updates by recent gradient magnitude."},{id:"regularization-l2-penalty",stepLabel:"39.1",group:"Regularization",title:"L2 penalty",concept:"L2 regularization penalizes large weights by adding lambda times sum of squared weights.",objective:"Accumulate weight squared.",difficulty:"core",starterCode:`function l2Penalty(weights, lambda) {
  let sumSquares = 0;

  for (let i = 0; i < weights.length; i++) {
    // TODO: add squared weight.
    sumSquares += 0;
  }

  return lambda * sumSquares;
}`,testCode:`const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: approxEqual(actual, expected) });
}

check('simple L2', l2Penalty([3, 4], 1), 25);
check('lambda half', l2Penalty([3, 4], 0.5), 12.5);
check('zero weights', l2Penalty([0, 0], 10), 0);

return results;`,hints:["L2 uses squared weights.","Add weights[i] * weights[i].","sumSquares += weights[i] * weights[i];"],solution:`function l2Penalty(weights, lambda) {
  let sumSquares = 0;

  for (let i = 0; i < weights.length; i++) {
    sumSquares += weights[i] * weights[i];
  }

  return lambda * sumSquares;
}`,explanation:"L2 discourages very large weights, often improving generalization."},{id:"regularization-l2-gradient",stepLabel:"39.2",group:"Regularization",title:"L2 gradient",concept:"The derivative of lambda times w squared with respect to w is 2 * lambda * w.",objective:"Push 2 * lambda * weight.",difficulty:"core",starterCode:`function l2Gradient(weights, lambda) {
  const gradients = [];

  for (let i = 0; i < weights.length; i++) {
    // TODO: push the L2 gradient for this weight.
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

check('lambda 1', l2Gradient([3, 4], 1), [6, 8]);
check('lambda half', l2Gradient([3, 4], 0.5), [3, 4]);
check('negative weights', l2Gradient([-1, 2], 1), [-2, 4]);

return results;`,hints:["Derivative of w squared is 2w.","Multiply by lambda.","gradients.push(2 * lambda * weights[i]);"],solution:`function l2Gradient(weights, lambda) {
  const gradients = [];

  for (let i = 0; i < weights.length; i++) {
    gradients.push(2 * lambda * weights[i]);
  }

  return gradients;
}`,explanation:"L2 gradient pulls weights toward zero."},{id:"regularization-dropout-mask",stepLabel:"39.3",group:"Regularization",title:"Apply dropout mask",concept:"Dropout removes selected activations during training.",objective:"Multiply each activation by its mask value.",difficulty:"warmup",starterCode:`function applyDropoutMask(activations, mask) {
  const dropped = [];

  for (let i = 0; i < activations.length; i++) {
    // TODO: multiply activation by mask.
    dropped.push(activations[i]);
  }

  return dropped;
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

check('drop middle', applyDropoutMask([1, 2, 3], [1, 0, 1]), [1, 0, 3]);
check('drop all', applyDropoutMask([1, 2], [0, 0]), [0, 0]);
check('keep all', applyDropoutMask([1, 2], [1, 1]), [1, 2]);

return results;`,hints:["Mask values are 0 or 1.","Multiply activations[i] by mask[i].","dropped.push(activations[i] * mask[i]);"],solution:`function applyDropoutMask(activations, mask) {
  const dropped = [];

  for (let i = 0; i < activations.length; i++) {
    dropped.push(activations[i] * mask[i]);
  }

  return dropped;
}`,explanation:"Dropout forces the network not to rely too heavily on any one activation."},{id:"regularization-inverted-dropout",stepLabel:"39.4",group:"Regularization",title:"Inverted dropout scaling",concept:"Inverted dropout divides kept activations by keep probability.",objective:"Apply mask and divide by keepProbability.",difficulty:"core",starterCode:`function invertedDropout(activations, mask, keepProbability) {
  const output = [];

  for (let i = 0; i < activations.length; i++) {
    // TODO: apply inverted dropout scaling.
    output.push(activations[i]);
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

check('keep prob 0.5', invertedDropout([1, 2, 3], [1, 0, 1], 0.5), [2, 0, 6]);
check('keep prob 1', invertedDropout([1, 2], [1, 1], 1), [1, 2]);
check('drop all', invertedDropout([1, 2], [0, 0], 0.5), [0, 0]);

return results;`,hints:["First multiply by mask[i].","Then divide by keepProbability.","output.push((activations[i] * mask[i]) / keepProbability);"],solution:`function invertedDropout(activations, mask, keepProbability) {
  const output = [];

  for (let i = 0; i < activations.length; i++) {
    output.push((activations[i] * mask[i]) / keepProbability);
  }

  return output;
}`,explanation:"Inverted dropout keeps expected activation scale roughly stable during training."},{id:"matmul-backprop-a-entry",stepLabel:"40.1",group:"Matrix multiplication backprop",title:"Gradient for A entry",concept:"If C[i][j] = sum over k of A[i][k] * B[k][j], then the derivative with respect to A[i][k] is B[k][j].",objective:"Return B[k][j].",difficulty:"core",starterCode:`function gradCellWithRespectToA(B, k, j) {
  // TODO: return the derivative of C[i][j] with respect to A[i][k].
  return 0;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

const B = [
  [2, 1, 3],
  [1, 4, 2],
];

check('k=0 j=0', gradCellWithRespectToA(B, 0, 0), 2);
check('k=0 j=2', gradCellWithRespectToA(B, 0, 2), 3);
check('k=1 j=1', gradCellWithRespectToA(B, 1, 1), 4);

return results;`,hints:["A[i][k] is multiplied by B[k][j].","The derivative with respect to A[i][k] is B[k][j].","return B[k][j];"],solution:`function gradCellWithRespectToA(B, k, j) {
  return B[k][j];
}`,explanation:"Backprop through multiplication sends the other factor backward."},{id:"matmul-backprop-b-entry",stepLabel:"40.2",group:"Matrix multiplication backprop",title:"Gradient for B entry",concept:"If C[i][j] = sum over k of A[i][k] * B[k][j], then the derivative with respect to B[k][j] is A[i][k].",objective:"Return A[i][k].",difficulty:"core",starterCode:`function gradCellWithRespectToB(A, i, k) {
  // TODO: return the derivative of C[i][j] with respect to B[k][j].
  return 0;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

const A = [
  [1, 2],
  [3, 1],
];

check('i=0 k=0', gradCellWithRespectToB(A, 0, 0), 1);
check('i=0 k=1', gradCellWithRespectToB(A, 0, 1), 2);
check('i=1 k=0', gradCellWithRespectToB(A, 1, 0), 3);

return results;`,hints:["B[k][j] is multiplied by A[i][k].","The derivative with respect to B[k][j] is A[i][k].","return A[i][k];"],solution:`function gradCellWithRespectToB(A, i, k) {
  return A[i][k];
}`,explanation:"Again, the gradient through multiplication sends the other factor backward."},{id:"matmul-backprop-dA",stepLabel:"40.3",group:"Matrix multiplication backprop",title:"dA from dC",concept:"For C = AB, the gradient with respect to A is dC times B transposed.",objective:"Return matmul(dC, transpose(B)).",difficulty:"challenge",starterCode:`function transpose(A) {
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

function gradA(dC, B) {
  // TODO: return dC times B transposed.
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

check('dA simple', gradA([[1, 0]], [[2, 3], [4, 5]]), [[2, 4]]);
check('dA two rows', gradA([[1, 1], [0, 1]], [[2, 3], [4, 5]]), [[5, 9], [3, 5]]);

return results;`,hints:["The formula is dA = dC times B transpose.","Use transpose(B).","return matmul(dC, transpose(B));"],solution:`function transpose(A) {
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

function gradA(dC, B) {
  return matmul(dC, transpose(B));
}`,explanation:"Matrix backprop uses transposes to send gradients to the correct side of the multiplication."},{id:"matmul-backprop-dB",stepLabel:"40.4",group:"Matrix multiplication backprop",title:"dB from dC",concept:"For C = AB, the gradient with respect to B is A transposed times dC.",objective:"Return matmul(transpose(A), dC).",difficulty:"challenge",starterCode:`function transpose(A) {
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

function gradB(A, dC) {
  // TODO: return A transposed times dC.
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

check('dB simple', gradB([[2, 4]], [[1, 0]]), [[2, 0], [4, 0]]);
check('dB two examples', gradB([[1, 2], [3, 4]], [[1, 0], [0, 1]]), [[1, 3], [2, 4]]);

return results;`,hints:["The formula is dB = A transpose times dC.","Use transpose(A).","return matmul(transpose(A), dC);"],solution:`function transpose(A) {
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

function gradB(A, dC) {
  return matmul(transpose(A), dC);
}`,explanation:"This is the dense-layer weight-gradient formula used in neural-network training."},{id:"transformer-token-embedding-lookup",stepLabel:"41.1",group:"Transformer mini-block shapes",title:"Token embedding lookup",concept:"A token ID selects one row from the embedding table.",objective:"Return embeddingTable[tokenId].",difficulty:"warmup",starterCode:`function lookupEmbedding(embeddingTable, tokenId) {
  // TODO: return the embedding vector for tokenId.
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

const E = [
  [1, 0],
  [0, 1],
  [2, 3],
];

check('token 0', lookupEmbedding(E, 0), [1, 0]);
check('token 1', lookupEmbedding(E, 1), [0, 1]);
check('token 2', lookupEmbedding(E, 2), [2, 3]);

return results;`,hints:["The embedding table is indexed by token ID.","Return the row at tokenId.","return embeddingTable[tokenId];"],solution:`function lookupEmbedding(embeddingTable, tokenId) {
  return embeddingTable[tokenId];
}`,explanation:"Token IDs become vectors by selecting rows from an embedding matrix."},{id:"transformer-add-position",stepLabel:"41.2",group:"Transformer mini-block shapes",title:"Add positional embedding",concept:"Token embeddings and position embeddings are added coordinate by coordinate.",objective:"Push tokenEmbedding[i] + positionEmbedding[i].",difficulty:"warmup",starterCode:`function addPosition(tokenEmbedding, positionEmbedding) {
  const result = [];

  for (let i = 0; i < tokenEmbedding.length; i++) {
    // TODO: add token and position coordinate.
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

check('add position', addPosition([1, 2], [10, 20]), [11, 22]);
check('zero position', addPosition([1, 2, 3], [0, 0, 0]), [1, 2, 3]);
check('negative position', addPosition([5, 5], [-1, 2]), [4, 7]);

return results;`,hints:["Embeddings have the same dimension.","Add coordinate by coordinate.","result.push(tokenEmbedding[i] + positionEmbedding[i]);"],solution:`function addPosition(tokenEmbedding, positionEmbedding) {
  const result = [];

  for (let i = 0; i < tokenEmbedding.length; i++) {
    result.push(tokenEmbedding[i] + positionEmbedding[i]);
  }

  return result;
}`,explanation:"Position information lets equal tokens behave differently at different sequence positions."},{id:"transformer-project-query",stepLabel:"41.3",group:"Transformer mini-block shapes",title:"Project to query vector",concept:"A query vector is a linear projection of the hidden state.",objective:"Return hidden times Wq using row dot products.",difficulty:"core",starterCode:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function project(hidden, weightColumns) {
  const output = [];

  for (let j = 0; j < weightColumns.length; j++) {
    // TODO: push dot(hidden, weightColumns[j]).
    output.push(0);
  }

  return output;
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

check('project hidden', project([1, 2], [[3, 4], [5, 6]]), [11, 17]);
check('identity projection', project([7, 8], [[1, 0], [0, 1]]), [7, 8]);

return results;`,hints:["Each output coordinate has its own weight column.","Use dot(hidden, weightColumns[j]).","output.push(dot(hidden, weightColumns[j]));"],solution:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function project(hidden, weightColumns) {
  const output = [];

  for (let j = 0; j < weightColumns.length; j++) {
    output.push(dot(hidden, weightColumns[j]));
  }

  return output;
}`,explanation:"Transformers create Q, K, and V vectors through learned linear projections."},{id:"transformer-attention-score-shape",stepLabel:"41.4",group:"Transformer mini-block shapes",title:"Attention score shape",concept:"Q times K transposed produces one score for every query token and key token pair.",objective:"Return [numQueries, numKeys].",difficulty:"core",starterCode:`function attentionScoreShape(Q, K) {
  const numQueries = Q.length;
  const numKeys = K.length;

  // TODO: return the shape of Q times K transposed.
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

check('3 queries 3 keys', attentionScoreShape([[1],[2],[3]], [[1],[2],[3]]), [3, 3]);
check('2 queries 4 keys', attentionScoreShape([[1],[2]], [[1],[2],[3],[4]]), [2, 4]);
check('1 query 5 keys', attentionScoreShape([[1]], [[1],[2],[3],[4],[5]]), [1, 5]);

return results;`,hints:["Rows come from queries.","Columns come from keys.","return [numQueries, numKeys];"],solution:`function attentionScoreShape(Q, K) {
  const numQueries = Q.length;
  const numKeys = K.length;

  return [numQueries, numKeys];
}`,explanation:"Attention score matrices grow with sequence length squared in full attention."},{id:"transformer-causal-mask-check",stepLabel:"41.5",group:"Transformer mini-block shapes",title:"Causal mask visibility",concept:"In causal attention, a query position can read only keys at the same or earlier positions.",objective:"Return true if keyPosition <= queryPosition.",difficulty:"core",starterCode:`function canAttendCausally(queryPosition, keyPosition) {
  // TODO: return whether query can see key.
  return false;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('same position visible', canAttendCausally(2, 2), true);
check('past visible', canAttendCausally(2, 0), true);
check('future hidden', canAttendCausally(2, 3), false);
check('first token cannot see second', canAttendCausally(0, 1), false);

return results;`,hints:["Causal attention blocks future keys.","A key is visible if keyPosition is less than or equal to queryPosition.","return keyPosition <= queryPosition;"],solution:`function canAttendCausally(queryPosition, keyPosition) {
  return keyPosition <= queryPosition;
}`,explanation:"Causal masking prevents next-token models from seeing future answers."},{id:"self-attention-one-query-scores",stepLabel:"42.1",group:"Mini self-attention",title:"Scores for one query",concept:"A query compares itself to every key using dot products.",objective:"Push dot(query, keys[i]) for every key.",difficulty:"core",starterCode:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function attentionScoresForQuery(query, keys) {
  const scores = [];

  for (let i = 0; i < keys.length; i++) {
    // TODO: push dot(query, keys[i]).
    scores.push(0);
  }

  return scores;
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

check('query against two keys', attentionScoresForQuery([1, 2], [[3, 4], [5, 6]]), [11, 17]);
check('orthogonal key', attentionScoresForQuery([1, 0], [[1, 0], [0, 1]]), [1, 0]);

return results;`,hints:["Each score is one dot product.","Compare the query with each key vector.","scores.push(dot(query, keys[i]));"],solution:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function attentionScoresForQuery(query, keys) {
  const scores = [];

  for (let i = 0; i < keys.length; i++) {
    scores.push(dot(query, keys[i]));
  }

  return scores;
}`,explanation:"Self-attention starts by asking how strongly this query matches each key."},{id:"self-attention-scale-scores",stepLabel:"42.2",group:"Mini self-attention",title:"Scale attention scores",concept:"Scaled dot-product attention divides scores by sqrt(d).",objective:"Divide every score by Math.sqrt(d).",difficulty:"core",starterCode:`function scaleScores(scores, d) {
  const scaled = [];

  for (let i = 0; i < scores.length; i++) {
    // TODO: push scores[i] divided by sqrt(d).
    scaled.push(scores[i]);
  }

  return scaled;
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

check('scale by sqrt 4', scaleScores([8, 4], 4), [4, 2]);
check('scale by sqrt 9', scaleScores([12, 3], 9), [4, 1]);
check('scale by sqrt 1', scaleScores([7, -2], 1), [7, -2]);

return results;`,hints:["Use Math.sqrt(d).","Each score gets divided by the same scale.","scaled.push(scores[i] / Math.sqrt(d));"],solution:`function scaleScores(scores, d) {
  const scaled = [];

  for (let i = 0; i < scores.length; i++) {
    scaled.push(scores[i] / Math.sqrt(d));
  }

  return scaled;
}`,explanation:"Scaling prevents large dot products from making softmax too sharp too early."},{id:"self-attention-causal-mask-scores",stepLabel:"42.3",group:"Mini self-attention",title:"Apply causal mask",concept:"Causal attention hides future positions by setting their scores to -Infinity.",objective:"Keep visible scores and mask future scores.",difficulty:"core",starterCode:`function applyCausalMask(scores, queryPosition) {
  const masked = [];

  for (let keyPosition = 0; keyPosition < scores.length; keyPosition++) {
    // TODO: keep scores[keyPosition] if keyPosition <= queryPosition, otherwise -Infinity.
    masked.push(scores[keyPosition]);
  }

  return masked;
}`,testCode:`const results = [];

function sameArraySpecial(a, b) {
  return a.length === b.length && a.every((value, index) => Object.is(value, b[index]));
}

function check(name, actual, expected) {
  results.push({
    name,
    actual: JSON.stringify(actual),
    expected: JSON.stringify(expected),
    passed: sameArraySpecial(actual, expected),
  });
}

check('query at position 0', applyCausalMask([1, 2, 3], 0), [1, -Infinity, -Infinity]);
check('query at position 1', applyCausalMask([1, 2, 3], 1), [1, 2, -Infinity]);
check('query at position 2', applyCausalMask([1, 2, 3], 2), [1, 2, 3]);

return results;`,hints:["A token can attend to itself and the past.","Future key positions are greater than queryPosition.","masked.push(keyPosition <= queryPosition ? scores[keyPosition] : -Infinity);"],solution:`function applyCausalMask(scores, queryPosition) {
  const masked = [];

  for (let keyPosition = 0; keyPosition < scores.length; keyPosition++) {
    masked.push(keyPosition <= queryPosition ? scores[keyPosition] : -Infinity);
  }

  return masked;
}`,explanation:"Causal masking prevents next-token models from seeing future tokens."},{id:"self-attention-stable-softmax",stepLabel:"42.4",group:"Mini self-attention",title:"Stable softmax",concept:"Stable softmax subtracts the maximum score before exponentiating.",objective:"Use Math.exp(scores[i] - maxScore).",difficulty:"challenge",starterCode:`function stableSoftmax(scores) {
  const maxScore = Math.max(...scores);
  let denominator = 0;

  for (let i = 0; i < scores.length; i++) {
    // TODO: add exp(scores[i] - maxScore).
    denominator += 0;
  }

  const weights = [];
  for (let i = 0; i < scores.length; i++) {
    weights.push(Math.exp(scores[i] - maxScore) / denominator);
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

check('two equal scores', stableSoftmax([0, 0]), [0.5, 0.5]);
check('log ratio', stableSoftmax([0, Math.log(3)]), [0.25, 0.75]);
check('large scores stay stable', stableSoftmax([1000, 1000]), [0.5, 0.5]);

return results;`,hints:["Subtracting maxScore does not change the softmax probabilities.","It prevents overflow for large scores.","denominator += Math.exp(scores[i] - maxScore);"],solution:`function stableSoftmax(scores) {
  const maxScore = Math.max(...scores);
  let denominator = 0;

  for (let i = 0; i < scores.length; i++) {
    denominator += Math.exp(scores[i] - maxScore);
  }

  const weights = [];
  for (let i = 0; i < scores.length; i++) {
    weights.push(Math.exp(scores[i] - maxScore) / denominator);
  }

  return weights;
}`,explanation:"Stable softmax is the same math, but safer numerically."},{id:"self-attention-weighted-value-sum",stepLabel:"42.5",group:"Mini self-attention",title:"Weighted value sum",concept:"Attention output is a weighted mixture of value vectors.",objective:"Add weights[token] * values[token][dim] into output[dim].",difficulty:"challenge",starterCode:`function weightedValueSum(weights, values) {
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

return results;`,hints:["Each value vector contributes according to its attention weight.","For each dimension, add weights[token] times values[token][dim].","output[dim] += weights[token] * values[token][dim];"],solution:`function weightedValueSum(weights, values) {
  const dimension = values[0].length;
  const output = Array(dimension).fill(0);

  for (let token = 0; token < values.length; token++) {
    for (let dim = 0; dim < dimension; dim++) {
      output[dim] += weights[token] * values[token][dim];
    }
  }

  return output;
}`,explanation:"Attention does not copy one token. It mixes value vectors using attention weights."},{id:"layernorm-feature-mean",stepLabel:"43.1",group:"LayerNorm and RMSNorm",title:"Feature mean",concept:"LayerNorm computes statistics across features of one token.",objective:"Return the average of the feature vector.",difficulty:"warmup",starterCode:`function featureMean(x) {
  let total = 0;

  for (let i = 0; i < x.length; i++) {
    total += x[i];
  }

  // TODO: return the average.
  return total;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('mean [1,2,3]', featureMean([1, 2, 3]), 2);
check('mean [10,20]', featureMean([10, 20]), 15);
check('mean [-1,1]', featureMean([-1, 1]), 0);

return results;`,hints:["Average is total divided by number of features.","The number of features is x.length.","return total / x.length;"],solution:`function featureMean(x) {
  let total = 0;

  for (let i = 0; i < x.length; i++) {
    total += x[i];
  }

  return total / x.length;
}`,explanation:"LayerNorm normalizes one token vector at a time, not a whole batch."},{id:"layernorm-feature-variance",stepLabel:"43.2",group:"LayerNorm and RMSNorm",title:"Feature variance",concept:"Variance measures average squared distance from the mean.",objective:"Add squared centered values.",difficulty:"core",starterCode:`function featureVariance(x) {
  const mean = x.reduce((total, value) => total + value, 0) / x.length;
  let total = 0;

  for (let i = 0; i < x.length; i++) {
    const centered = x[i] - mean;

    // TODO: add centered squared.
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

check('variance [1,2,3]', featureVariance([1, 2, 3]), 2 / 3);
check('variance [10,20]', featureVariance([10, 20]), 25);
check('variance constant', featureVariance([5, 5, 5]), 0);

return results;`,hints:["Variance uses squared centered values.","centered is already computed.","total += centered * centered;"],solution:`function featureVariance(x) {
  const mean = x.reduce((total, value) => total + value, 0) / x.length;
  let total = 0;

  for (let i = 0; i < x.length; i++) {
    const centered = x[i] - mean;
    total += centered * centered;
  }

  return total / x.length;
}`,explanation:"LayerNorm uses variance to rescale features to a stable range."},{id:"layernorm-normalize-vector",stepLabel:"43.3",group:"LayerNorm and RMSNorm",title:"Normalize one token vector",concept:"LayerNorm subtracts mean and divides by standard deviation.",objective:"Push (x[i] - mean) / sqrt(variance + eps).",difficulty:"challenge",starterCode:`function layerNormNoAffine(x, eps = 1e-5) {
  const mean = x.reduce((total, value) => total + value, 0) / x.length;
  const variance = x.reduce((total, value) => {
    const centered = value - mean;
    return total + centered * centered;
  }, 0) / x.length;

  const normalized = [];

  for (let i = 0; i < x.length; i++) {
    // TODO: push the normalized feature.
    normalized.push(0);
  }

  return normalized;
}`,testCode:`const results = [];

function approxArray(a, b, tolerance = 1e-5) {
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

check('normalize [1,2,3]', layerNormNoAffine([1, 2, 3], 0), [-1.224744871, 0, 1.224744871]);
check('normalize [10,20]', layerNormNoAffine([10, 20], 0), [-1, 1]);

return results;`,hints:["Standard deviation is Math.sqrt(variance + eps).","Subtract mean first, then divide by std.","normalized.push((x[i] - mean) / Math.sqrt(variance + eps));"],solution:`function layerNormNoAffine(x, eps = 1e-5) {
  const mean = x.reduce((total, value) => total + value, 0) / x.length;
  const variance = x.reduce((total, value) => {
    const centered = value - mean;
    return total + centered * centered;
  }, 0) / x.length;

  const normalized = [];

  for (let i = 0; i < x.length; i++) {
    normalized.push((x[i] - mean) / Math.sqrt(variance + eps));
  }

  return normalized;
}`,explanation:"LayerNorm stabilizes the scale of each token representation before the next transformation."},{id:"rmsnorm-denominator",stepLabel:"43.4",group:"LayerNorm and RMSNorm",title:"RMSNorm denominator",concept:"RMSNorm divides by root mean square without subtracting the mean.",objective:"Return sqrt(mean square + eps).",difficulty:"core",starterCode:`function rmsDenominator(x, eps = 1e-5) {
  let meanSquare = 0;

  for (let i = 0; i < x.length; i++) {
    meanSquare += x[i] * x[i];
  }

  meanSquare = meanSquare / x.length;

  // TODO: return root mean square denominator.
  return meanSquare;
}`,testCode:`const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: approxEqual(actual, expected) });
}

check('rms [3,4] eps 0', rmsDenominator([3, 4], 0), Math.sqrt(12.5));
check('rms [1,1] eps 0', rmsDenominator([1, 1], 0), 1);
check('rms [0,0] eps 1', rmsDenominator([0, 0], 1), 1);

return results;`,hints:["RMS means root mean square.","Use Math.sqrt(meanSquare + eps).","return Math.sqrt(meanSquare + eps);"],solution:`function rmsDenominator(x, eps = 1e-5) {
  let meanSquare = 0;

  for (let i = 0; i < x.length; i++) {
    meanSquare += x[i] * x[i];
  }

  meanSquare = meanSquare / x.length;

  return Math.sqrt(meanSquare + eps);
}`,explanation:"RMSNorm stabilizes scale without centering features."},{id:"residual-add-vector",stepLabel:"44.1",group:"Residual stream mechanics",title:"Add residual",concept:"A residual connection adds a block output back to the original stream.",objective:"Push x[i] + update[i].",difficulty:"warmup",starterCode:`function addResidual(x, update) {
  const result = [];

  for (let i = 0; i < x.length; i++) {
    // TODO: add original stream and update.
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

check('simple residual', addResidual([1, 2], [10, 20]), [11, 22]);
check('zero update', addResidual([1, 2, 3], [0, 0, 0]), [1, 2, 3]);
check('negative update', addResidual([5, 5], [-1, 2]), [4, 7]);

return results;`,hints:["Residual means original plus update.","Add coordinate by coordinate.","result.push(x[i] + update[i]);"],solution:`function addResidual(x, update) {
  const result = [];

  for (let i = 0; i < x.length; i++) {
    result.push(x[i] + update[i]);
  }

  return result;
}`,explanation:"Residual connections let each block write an update into the shared representation stream."},{id:"residual-scaled-update",stepLabel:"44.2",group:"Residual stream mechanics",title:"Scaled residual update",concept:"Sometimes updates are scaled before being added to the residual stream.",objective:"Push x[i] + scale * update[i].",difficulty:"core",starterCode:`function addScaledResidual(x, update, scale) {
  const result = [];

  for (let i = 0; i < x.length; i++) {
    // TODO: add scaled update to x.
    result.push(x[i]);
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

check('scale 0.5', addScaledResidual([1, 2], [10, 20], 0.5), [6, 12]);
check('scale 0', addScaledResidual([1, 2], [10, 20], 0), [1, 2]);
check('scale 1', addScaledResidual([1, 2], [10, 20], 1), [11, 22]);

return results;`,hints:["The update is multiplied by scale before adding.","Use x[i] + scale * update[i].","result.push(x[i] + scale * update[i]);"],solution:`function addScaledResidual(x, update, scale) {
  const result = [];

  for (let i = 0; i < x.length; i++) {
    result.push(x[i] + scale * update[i]);
  }

  return result;
}`,explanation:"Scaling residual updates can help control signal size in deep networks."},{id:"residual-prenorm-block",stepLabel:"44.3",group:"Residual stream mechanics",title:"Pre-norm residual block",concept:"A pre-norm block normalizes before the sublayer, then adds the sublayer output back to the stream.",objective:"Return x plus sublayer(normedX).",difficulty:"challenge",starterCode:`function addVectors(a, b) {
  return a.map((value, i) => value + b[i]);
}

function preNormBlock(x, normedX, sublayer) {
  const update = sublayer(normedX);

  // TODO: return residual stream after the update.
  return update;
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

check('identity update', preNormBlock([1, 2], [10, 20], (h) => [h[0], h[1]]), [11, 22]);
check('zero update', preNormBlock([1, 2], [10, 20], () => [0, 0]), [1, 2]);

return results;`,hints:["Residual block returns original x plus update.","update is already computed.","return addVectors(x, update);"],solution:`function addVectors(a, b) {
  return a.map((value, i) => value + b[i]);
}

function preNormBlock(x, normedX, sublayer) {
  const update = sublayer(normedX);
  return addVectors(x, update);
}`,explanation:"Pre-norm transformers normalize the stream before attention or MLP, then add the block output back."},{id:"swiglu-silu",stepLabel:"45.1",group:"MLP and SwiGLU",title:"SiLU activation",concept:"SiLU is x * sigmoid(x), used inside SwiGLU-style MLPs.",objective:"Return x * sigmoid(x).",difficulty:"core",starterCode:`function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function silu(x) {
  // TODO: return x times sigmoid(x).
  return x;
}`,testCode:`const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: approxEqual(actual, expected) });
}

check('silu 0', silu(0), 0);
check('silu log 3', silu(Math.log(3)), Math.log(3) * 0.75);
check('silu -log 3', silu(-Math.log(3)), -Math.log(3) * 0.25);

return results;`,hints:["SiLU gates x by sigmoid(x).","sigmoid(x) is already available.","return x * sigmoid(x);"],solution:`function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function silu(x) {
  return x * sigmoid(x);
}`,explanation:"SiLU is a smooth gate: positive values mostly pass, negative values are softened."},{id:"swiglu-elementwise-gate",stepLabel:"45.2",group:"MLP and SwiGLU",title:"Elementwise gate",concept:"Gated MLPs multiply one hidden stream by another gate stream element by element.",objective:"Push values[i] * gates[i].",difficulty:"warmup",starterCode:`function elementwiseGate(values, gates) {
  const output = [];

  for (let i = 0; i < values.length; i++) {
    // TODO: multiply matching entries.
    output.push(values[i]);
  }

  return output;
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

check('simple gate', elementwiseGate([1, 2, 3], [10, 0, 2]), [10, 0, 6]);
check('all keep', elementwiseGate([1, 2], [1, 1]), [1, 2]);
check('all block', elementwiseGate([1, 2], [0, 0]), [0, 0]);

return results;`,hints:["This is elementwise multiplication.","Use values[i] * gates[i].","output.push(values[i] * gates[i]);"],solution:`function elementwiseGate(values, gates) {
  const output = [];

  for (let i = 0; i < values.length; i++) {
    output.push(values[i] * gates[i]);
  }

  return output;
}`,explanation:"Gating lets one stream decide how much of another stream passes through."},{id:"swiglu-hidden",stepLabel:"45.3",group:"MLP and SwiGLU",title:"SwiGLU hidden activation",concept:"SwiGLU combines a value stream with a SiLU-activated gate stream.",objective:"Push value[i] * silu(gate[i]).",difficulty:"challenge",starterCode:`function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function silu(x) {
  return x * sigmoid(x);
}

function swigluHidden(values, gates) {
  const output = [];

  for (let i = 0; i < values.length; i++) {
    // TODO: multiply values[i] by silu(gates[i]).
    output.push(0);
  }

  return output;
}`,testCode:`const results = [];

function approxArray(a, b, tolerance = 1e-9) {
  return a.length === b.length && a.every((value, index) => Math.abs(value - b[index]) <= tolerance);
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function siluRef(x) {
  return x * sigmoid(x);
}

function check(name, actual, expected) {
  results.push({
    name,
    actual: JSON.stringify(actual),
    expected: JSON.stringify(expected),
    passed: approxArray(actual, expected),
  });
}

check('swiglu simple', swigluHidden([2, 3], [0, Math.log(3)]), [0, 3 * siluRef(Math.log(3))]);
check('zero values', swigluHidden([0, 0], [10, 10]), [0, 0]);

return results;`,hints:["Apply SiLU to the gate stream.","Then multiply by the value stream.","output.push(values[i] * silu(gates[i]));"],solution:`function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function silu(x) {
  return x * sigmoid(x);
}

function swigluHidden(values, gates) {
  const output = [];

  for (let i = 0; i < values.length; i++) {
    output.push(values[i] * silu(gates[i]));
  }

  return output;
}`,explanation:"SwiGLU is a modern gated MLP pattern used in many transformer variants."},{id:"mlp-output-projection",stepLabel:"45.4",group:"MLP and SwiGLU",title:"MLP output projection",concept:"After hidden activation, an MLP projects back to the model dimension.",objective:"Return denseLayer(hidden, outputWeights, outputBiases).",difficulty:"core",starterCode:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function denseLayer(x, weightColumns, biases) {
  return weightColumns.map((weights, j) => dot(x, weights) + biases[j]);
}

function mlpOutput(hidden, outputWeights, outputBiases) {
  // TODO: project hidden back to output dimension.
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

check('project hidden to 2 outputs', mlpOutput([1, 2], [[3, 4], [5, 6]], [0, 1]), [11, 18]);
check('identity projection', mlpOutput([7, 8], [[1, 0], [0, 1]], [0, 0]), [7, 8]);

return results;`,hints:["The helper denseLayer is already available.","Use hidden as the input vector.","return denseLayer(hidden, outputWeights, outputBiases);"],solution:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

function denseLayer(x, weightColumns, biases) {
  return weightColumns.map((weights, j) => dot(x, weights) + biases[j]);
}

function mlpOutput(hidden, outputWeights, outputBiases) {
  return denseLayer(hidden, outputWeights, outputBiases);
}`,explanation:"Transformer MLPs expand, activate or gate, then project back into the residual stream dimension."},{id:"transformer-attention-residual-update",stepLabel:"46.1",group:"Tiny transformer block",title:"Attention residual update",concept:"The attention sublayer writes an update into the residual stream.",objective:"Return x + attentionOutput.",difficulty:"warmup",starterCode:`function addVectors(a, b) {
  return a.map((value, i) => value + b[i]);
}

function attentionResidual(x, attentionOutput) {
  // TODO: return residual stream after attention.
  return attentionOutput;
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

check('attention update', attentionResidual([1, 2], [10, 20]), [11, 22]);
check('zero update', attentionResidual([1, 2], [0, 0]), [1, 2]);

return results;`,hints:["Residual means original stream plus update.","Use addVectors.","return addVectors(x, attentionOutput);"],solution:`function addVectors(a, b) {
  return a.map((value, i) => value + b[i]);
}

function attentionResidual(x, attentionOutput) {
  return addVectors(x, attentionOutput);
}`,explanation:"Attention reads from the sequence and writes an update back into each token residual stream."},{id:"transformer-mlp-residual-update",stepLabel:"46.2",group:"Tiny transformer block",title:"MLP residual update",concept:"After attention, the MLP sublayer also writes into the residual stream.",objective:"Return streamAfterAttention + mlpOutput.",difficulty:"warmup",starterCode:`function addVectors(a, b) {
  return a.map((value, i) => value + b[i]);
}

function mlpResidual(streamAfterAttention, mlpOutput) {
  // TODO: return residual stream after MLP.
  return mlpOutput;
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

check('mlp update', mlpResidual([11, 22], [3, 4]), [14, 26]);
check('zero update', mlpResidual([11, 22], [0, 0]), [11, 22]);

return results;`,hints:["The MLP update is added to the current stream.","Use addVectors.","return addVectors(streamAfterAttention, mlpOutput);"],solution:`function addVectors(a, b) {
  return a.map((value, i) => value + b[i]);
}

function mlpResidual(streamAfterAttention, mlpOutput) {
  return addVectors(streamAfterAttention, mlpOutput);
}`,explanation:"Transformer blocks usually contain two residual writes: attention, then MLP."},{id:"transformer-prenorm-block-forward",stepLabel:"46.3",group:"Tiny transformer block",title:"Pre-norm transformer block",concept:"A pre-norm transformer block normalizes before attention and before MLP.",objective:"Return x + attention(norm1(x)) + mlp(norm2(afterAttention)).",difficulty:"challenge",starterCode:`function addVectors(a, b) {
  return a.map((value, i) => value + b[i]);
}

function tinyPreNormBlock(x, norm1, attention, norm2, mlp) {
  const attentionInput = norm1(x);
  const attentionOutput = attention(attentionInput);
  const afterAttention = addVectors(x, attentionOutput);

  const mlpInput = norm2(afterAttention);
  const mlpOutput = mlp(mlpInput);

  // TODO: return afterAttention plus mlpOutput.
  return mlpOutput;
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

check('simple block', tinyPreNormBlock([1, 2], (x) => x, () => [10, 20], (x) => x, () => [3, 4]), [14, 26]);
check('zero updates', tinyPreNormBlock([1, 2], (x) => x, () => [0, 0], (x) => x, () => [0, 0]), [1, 2]);

return results;`,hints:["afterAttention is already x plus attention output.","The final step adds mlpOutput to afterAttention.","return addVectors(afterAttention, mlpOutput);"],solution:`function addVectors(a, b) {
  return a.map((value, i) => value + b[i]);
}

function tinyPreNormBlock(x, norm1, attention, norm2, mlp) {
  const attentionInput = norm1(x);
  const attentionOutput = attention(attentionInput);
  const afterAttention = addVectors(x, attentionOutput);

  const mlpInput = norm2(afterAttention);
  const mlpOutput = mlp(mlpInput);

  return addVectors(afterAttention, mlpOutput);
}`,explanation:"This is the transformer-block skeleton: normalize, attention, residual, normalize, MLP, residual."},{id:"transformer-stack-two-blocks",stepLabel:"46.4",group:"Tiny transformer block",title:"Stack two blocks",concept:"Transformer depth comes from feeding one block output into the next block.",objective:"Return block2(block1(x)).",difficulty:"core",starterCode:`function stackTwoBlocks(x, block1, block2) {
  const afterBlock1 = block1(x);

  // TODO: feed afterBlock1 into block2.
  return afterBlock1;
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

check('two additive blocks', stackTwoBlocks([1, 2], (x) => x.map((v) => v + 10), (x) => x.map((v) => v * 2)), [22, 24]);
check('identity then shift', stackTwoBlocks([1, 2], (x) => x, (x) => x.map((v) => v + 1)), [2, 3]);

return results;`,hints:["Depth means sequential composition.","block2 receives the output of block1.","return block2(afterBlock1);"],solution:`function stackTwoBlocks(x, block1, block2) {
  const afterBlock1 = block1(x);
  return block2(afterBlock1);
}`,explanation:"Deep transformers repeatedly update the residual stream through many blocks."},{id:"debug-attention-weights-sum",stepLabel:"47.1",group:"Transformer debugging checks",title:"Attention weights sum to one",concept:"Softmax attention weights should sum to 1.",objective:"Return the sum of weights.",difficulty:"warmup",starterCode:`function sumWeights(weights) {
  let total = 0;

  for (let i = 0; i < weights.length; i++) {
    // TODO: add each weight.
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

check('two weights', sumWeights([0.5, 0.5]), 1);
check('three weights', sumWeights([0.2, 0.3, 0.5]), 1);
check('one weight', sumWeights([1]), 1);

return results;`,hints:["Loop over all weights.","Add weights[i] into total.","total += weights[i];"],solution:`function sumWeights(weights) {
  let total = 0;

  for (let i = 0; i < weights.length; i++) {
    total += weights[i];
  }

  return total;
}`,explanation:"If attention weights do not sum to one, the softmax or mask logic is likely broken."},{id:"debug-causal-leak",stepLabel:"47.2",group:"Transformer debugging checks",title:"Detect future attention leak",concept:"A causal mask fails if any query attends to a future key.",objective:"Return true if keyPosition is greater than queryPosition.",difficulty:"core",starterCode:`function isFutureLeak(queryPosition, keyPosition) {
  // TODO: return true when key is in the future.
  return false;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('past is not leak', isFutureLeak(3, 1), false);
check('same position is not leak', isFutureLeak(3, 3), false);
check('future is leak', isFutureLeak(3, 4), true);
check('first query cannot see second key', isFutureLeak(0, 1), true);

return results;`,hints:["Future means keyPosition is greater than queryPosition.","Same position is allowed in causal attention.","return keyPosition > queryPosition;"],solution:`function isFutureLeak(queryPosition, keyPosition) {
  return keyPosition > queryPosition;
}`,explanation:"Future leakage lets next-token models cheat during training."},{id:"debug-residual-norm-explosion",stepLabel:"47.3",group:"Transformer debugging checks",title:"Detect residual norm explosion",concept:"Very large residual norms can indicate unstable updates.",objective:"Return true when norm exceeds threshold.",difficulty:"core",starterCode:`function norm(v) {
  let total = 0;
  for (let i = 0; i < v.length; i++) {
    total += v[i] * v[i];
  }
  return Math.sqrt(total);
}

function residualNormTooLarge(stream, threshold) {
  // TODO: return whether norm(stream) is greater than threshold.
  return false;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('small stream', residualNormTooLarge([3, 4], 10), false);
check('large stream', residualNormTooLarge([30, 40], 10), true);
check('equal threshold is not greater', residualNormTooLarge([3, 4], 5), false);

return results;`,hints:["Use the norm helper.","Compare norm(stream) with threshold.","return norm(stream) > threshold;"],solution:`function norm(v) {
  let total = 0;
  for (let i = 0; i < v.length; i++) {
    total += v[i] * v[i];
  }
  return Math.sqrt(total);
}

function residualNormTooLarge(stream, threshold) {
  return norm(stream) > threshold;
}`,explanation:"Monitoring residual stream norms can help diagnose instability in deep networks."},{id:"debug-attention-shape-mismatch",stepLabel:"47.4",group:"Transformer debugging checks",title:"Detect Q/K dimension mismatch",concept:"Queries and keys must have the same feature dimension for dot products.",objective:"Return whether queryDim equals keyDim.",difficulty:"core",starterCode:`function attentionDimsCompatible(query, key) {
  const queryDim = query.length;
  const keyDim = key.length;

  // TODO: return whether dimensions match.
  return false;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('same dimension', attentionDimsCompatible([1, 2], [3, 4]), true);
check('different dimension', attentionDimsCompatible([1, 2, 3], [4, 5]), false);
check('one-dimensional same', attentionDimsCompatible([1], [2]), true);

return results;`,hints:["Dot products require matching lengths.","Compare queryDim and keyDim.","return queryDim === keyDim;"],solution:`function attentionDimsCompatible(query, key) {
  const queryDim = query.length;
  const keyDim = key.length;

  return queryDim === keyDim;
}`,explanation:"Many transformer bugs are shape bugs: Q and K must line up for similarity scores."},{id:"lm-vocab-size",stepLabel:"48.1",group:"Mini vocabulary and logits",title:"Vocabulary size",concept:"A language model predicts one score per vocabulary token.",objective:"Return the number of tokens in the vocabulary.",difficulty:"warmup",starterCode:`function vocabSize(vocab) {
  // TODO: return the number of tokens.
  return 0;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('three-token vocab', vocabSize(['cat', 'dog', 'fish']), 3);
check('one-token vocab', vocabSize(['<eos>']), 1);
check('five-token vocab', vocabSize(['a', 'b', 'c', 'd', 'e']), 5);

return results;`,hints:["The vocabulary is an array.","Array length gives the number of tokens.","return vocab.length;"],solution:`function vocabSize(vocab) {
  return vocab.length;
}`,explanation:"A model with vocabulary size V produces V logits at each prediction position."},{id:"lm-argmax-logit",stepLabel:"48.2",group:"Mini vocabulary and logits",title:"Argmax logit",concept:"Greedy decoding chooses the token with the largest logit.",objective:"Return the index of the largest logit.",difficulty:"core",starterCode:`function argmax(logits) {
  let bestIndex = 0;
  let bestValue = logits[0];

  for (let i = 1; i < logits.length; i++) {
    // TODO: update bestIndex and bestValue when logits[i] is larger.
  }

  return bestIndex;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('largest at index 0', argmax([5, 1, 2]), 0);
check('largest at index 1', argmax([1, 5, 2]), 1);
check('largest at index 2', argmax([-3, -2, -1]), 2);

return results;`,hints:["Compare logits[i] with bestValue.","If logits[i] is larger, update both bestValue and bestIndex.",`if (logits[i] > bestValue) {
  bestValue = logits[i];
  bestIndex = i;
}`],solution:`function argmax(logits) {
  let bestIndex = 0;
  let bestValue = logits[0];

  for (let i = 1; i < logits.length; i++) {
    if (logits[i] > bestValue) {
      bestValue = logits[i];
      bestIndex = i;
    }
  }

  return bestIndex;
}`,explanation:"Argmax decoding is deterministic: it always picks the highest-scoring token."},{id:"lm-decode-argmax-token",stepLabel:"48.3",group:"Mini vocabulary and logits",title:"Decode predicted token",concept:"A predicted token ID becomes text by indexing into the vocabulary.",objective:"Return vocab[argmax(logits)].",difficulty:"core",starterCode:`function argmax(logits) {
  let bestIndex = 0;
  let bestValue = logits[0];

  for (let i = 1; i < logits.length; i++) {
    if (logits[i] > bestValue) {
      bestValue = logits[i];
      bestIndex = i;
    }
  }

  return bestIndex;
}

function greedyToken(vocab, logits) {
  // TODO: return the vocabulary token with the largest logit.
  return '';
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

const vocab = ['cat', 'dog', 'fish'];

check('predict cat', greedyToken(vocab, [5, 1, 2]), 'cat');
check('predict dog', greedyToken(vocab, [1, 5, 2]), 'dog');
check('predict fish', greedyToken(vocab, [-3, -2, -1]), 'fish');

return results;`,hints:["First get the best token index.","Then use that index to read from vocab.","return vocab[argmax(logits)];"],solution:`function argmax(logits) {
  let bestIndex = 0;
  let bestValue = logits[0];

  for (let i = 1; i < logits.length; i++) {
    if (logits[i] > bestValue) {
      bestValue = logits[i];
      bestIndex = i;
    }
  }

  return bestIndex;
}

function greedyToken(vocab, logits) {
  return vocab[argmax(logits)];
}`,explanation:"The model predicts token IDs. The tokenizer vocabulary maps those IDs back to text pieces."},{id:"lm-logits-to-probabilities",stepLabel:"48.4",group:"Mini vocabulary and logits",title:"Logits to probabilities",concept:"Softmax converts arbitrary logits into probabilities that sum to 1.",objective:"Return stable softmax probabilities.",difficulty:"challenge",starterCode:`function softmax(logits) {
  const maxLogit = Math.max(...logits);
  let denominator = 0;

  for (let i = 0; i < logits.length; i++) {
    denominator += Math.exp(logits[i] - maxLogit);
  }

  const probabilities = [];

  for (let i = 0; i < logits.length; i++) {
    // TODO: push normalized probability for logits[i].
    probabilities.push(0);
  }

  return probabilities;
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

check('equal logits', softmax([0, 0]), [0.5, 0.5]);
check('log ratio', softmax([0, Math.log(3)]), [0.25, 0.75]);
check('large equal logits', softmax([1000, 1000]), [0.5, 0.5]);

return results;`,hints:["Use the same shifted exponentials as the denominator.","Probability = exp(logit - maxLogit) / denominator.","probabilities.push(Math.exp(logits[i] - maxLogit) / denominator);"],solution:`function softmax(logits) {
  const maxLogit = Math.max(...logits);
  let denominator = 0;

  for (let i = 0; i < logits.length; i++) {
    denominator += Math.exp(logits[i] - maxLogit);
  }

  const probabilities = [];

  for (let i = 0; i < logits.length; i++) {
    probabilities.push(Math.exp(logits[i] - maxLogit) / denominator);
  }

  return probabilities;
}`,explanation:"Logits are raw scores. Softmax turns them into a probability distribution over tokens."},{id:"sequence-target-probability",stepLabel:"49.1",group:"Cross-entropy over sequence positions",title:"Target token probability",concept:"At one position, the loss uses the probability assigned to the true next token.",objective:"Return probabilities[targetTokenId].",difficulty:"warmup",starterCode:`function targetProbability(probabilities, targetTokenId) {
  // TODO: return probability of the target token.
  return 0;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('target 0', targetProbability([0.7, 0.2, 0.1], 0), 0.7);
check('target 1', targetProbability([0.7, 0.2, 0.1], 1), 0.2);
check('target 2', targetProbability([0.7, 0.2, 0.1], 2), 0.1);

return results;`,hints:["targetTokenId is an array index.","Read that probability from the probabilities array.","return probabilities[targetTokenId];"],solution:`function targetProbability(probabilities, targetTokenId) {
  return probabilities[targetTokenId];
}`,explanation:"Cross-entropy only cares how much probability the model assigned to the correct token."},{id:"sequence-nll-one-position",stepLabel:"49.2",group:"Cross-entropy over sequence positions",title:"Negative log-likelihood",concept:"Token loss is -log(probability assigned to the true token).",objective:"Return -Math.log(targetProbability).",difficulty:"core",starterCode:`function tokenNLL(targetProbability) {
  // TODO: return negative log likelihood.
  return 0;
}`,testCode:`const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: approxEqual(actual, expected) });
}

check('p=0.5', tokenNLL(0.5), -Math.log(0.5));
check('p=0.8', tokenNLL(0.8), -Math.log(0.8));
check('p=0.25', tokenNLL(0.25), -Math.log(0.25));

return results;`,hints:["Use Math.log.","The loss is negative log probability.","return -Math.log(targetProbability);"],solution:`function tokenNLL(targetProbability) {
  return -Math.log(targetProbability);
}`,explanation:"Confident correct predictions have low loss; low probability on the true token gives high loss."},{id:"sequence-average-token-loss",stepLabel:"49.3",group:"Cross-entropy over sequence positions",title:"Average token loss",concept:"Language-model loss is usually averaged across predicted positions.",objective:"Return average of token losses.",difficulty:"core",starterCode:`function averageTokenLoss(tokenLosses) {
  let total = 0;

  for (let i = 0; i < tokenLosses.length; i++) {
    total += tokenLosses[i];
  }

  // TODO: return average loss.
  return total;
}`,testCode:`const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: approxEqual(actual, expected) });
}

check('average [1,2,3]', averageTokenLoss([1, 2, 3]), 2);
check('average two losses', averageTokenLoss([0.5, 1.5]), 1);
check('zero losses', averageTokenLoss([0, 0, 0]), 0);

return results;`,hints:["Average means total divided by count.","The count is tokenLosses.length.","return total / tokenLosses.length;"],solution:`function averageTokenLoss(tokenLosses) {
  let total = 0;

  for (let i = 0; i < tokenLosses.length; i++) {
    total += tokenLosses[i];
  }

  return total / tokenLosses.length;
}`,explanation:"A sequence loss summarizes many next-token prediction losses into one training number."},{id:"sequence-perplexity",stepLabel:"49.4",group:"Cross-entropy over sequence positions",title:"Perplexity",concept:"Perplexity is exp(average cross-entropy loss).",objective:"Return Math.exp(averageLoss).",difficulty:"core",starterCode:`function perplexity(averageLoss) {
  // TODO: return exp of averageLoss.
  return averageLoss;
}`,testCode:`const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: approxEqual(actual, expected) });
}

check('loss 0', perplexity(0), 1);
check('loss log 2', perplexity(Math.log(2)), 2);
check('loss log 10', perplexity(Math.log(10)), 10);

return results;`,hints:["Use Math.exp.","Perplexity = e raised to average loss.","return Math.exp(averageLoss);"],solution:`function perplexity(averageLoss) {
  return Math.exp(averageLoss);
}`,explanation:"Perplexity loosely means how many choices the model is confused among on average."},{id:"lm-select-position-logits",stepLabel:"50.1",group:"Tiny language-model loss",title:"Select position logits",concept:"A language model produces one logit vector per sequence position.",objective:"Return logitsByPosition[position].",difficulty:"warmup",starterCode:`function positionLogits(logitsByPosition, position) {
  // TODO: return logits for this sequence position.
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

const logits = [
  [1, 2, 3],
  [4, 5, 6],
  [7, 8, 9],
];

check('position 0', positionLogits(logits, 0), [1, 2, 3]);
check('position 1', positionLogits(logits, 1), [4, 5, 6]);
check('position 2', positionLogits(logits, 2), [7, 8, 9]);

return results;`,hints:["Position is an array index.","Each row is the logits for one position.","return logitsByPosition[position];"],solution:`function positionLogits(logitsByPosition, position) {
  return logitsByPosition[position];
}`,explanation:"For a sequence of length T, the model returns T logit vectors, one for each position."},{id:"lm-one-position-loss",stepLabel:"50.2",group:"Tiny language-model loss",title:"One-position loss",concept:"One LM loss position is cross-entropy between logits and the true next token ID.",objective:"Convert logits to probabilities, then return -log target probability.",difficulty:"challenge",starterCode:`function softmax(logits) {
  const maxLogit = Math.max(...logits);
  let denominator = 0;

  for (let i = 0; i < logits.length; i++) {
    denominator += Math.exp(logits[i] - maxLogit);
  }

  return logits.map((logit) => Math.exp(logit - maxLogit) / denominator);
}

function onePositionLoss(logits, targetTokenId) {
  const probabilities = softmax(logits);

  // TODO: return negative log probability of targetTokenId.
  return 0;
}`,testCode:`const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: approxEqual(actual, expected) });
}

check('target 0 equal logits', onePositionLoss([0, 0], 0), -Math.log(0.5));
check('target 1 log ratio', onePositionLoss([0, Math.log(3)], 1), -Math.log(0.75));
check('target 0 log ratio', onePositionLoss([0, Math.log(3)], 0), -Math.log(0.25));

return results;`,hints:["The target probability is probabilities[targetTokenId].","Loss is -Math.log(target probability).","return -Math.log(probabilities[targetTokenId]);"],solution:`function softmax(logits) {
  const maxLogit = Math.max(...logits);
  let denominator = 0;

  for (let i = 0; i < logits.length; i++) {
    denominator += Math.exp(logits[i] - maxLogit);
  }

  return logits.map((logit) => Math.exp(logit - maxLogit) / denominator);
}

function onePositionLoss(logits, targetTokenId) {
  const probabilities = softmax(logits);
  return -Math.log(probabilities[targetTokenId]);
}`,explanation:"The model is trained to put high probability on the true next token."},{id:"lm-average-loss",stepLabel:"50.3",group:"Tiny language-model loss",title:"Average language-model loss",concept:"The final LM loss averages next-token losses across positions.",objective:"Accumulate onePositionLoss for each position and divide by count.",difficulty:"challenge",starterCode:`function softmax(logits) {
  const maxLogit = Math.max(...logits);
  const exps = logits.map((x) => Math.exp(x - maxLogit));
  const denom = exps.reduce((a, b) => a + b, 0);
  return exps.map((x) => x / denom);
}

function onePositionLoss(logits, targetTokenId) {
  const probabilities = softmax(logits);
  return -Math.log(probabilities[targetTokenId]);
}

function languageModelLoss(logitsByPosition, targetTokenIds) {
  let total = 0;

  for (let position = 0; position < targetTokenIds.length; position++) {
    // TODO: add loss for this position.
    total += 0;
  }

  return total / targetTokenIds.length;
}`,testCode:`const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: approxEqual(actual, expected) });
}

check('two positions equal logits', languageModelLoss([[0, 0], [0, 0]], [0, 1]), -Math.log(0.5));
check('two positions log ratios', languageModelLoss([[0, Math.log(3)], [Math.log(3), 0]], [1, 0]), -Math.log(0.75));

return results;`,hints:["Use onePositionLoss(logitsByPosition[position], targetTokenIds[position]).","Add it to total.","total += onePositionLoss(logitsByPosition[position], targetTokenIds[position]);"],solution:`function softmax(logits) {
  const maxLogit = Math.max(...logits);
  const exps = logits.map((x) => Math.exp(x - maxLogit));
  const denom = exps.reduce((a, b) => a + b, 0);
  return exps.map((x) => x / denom);
}

function onePositionLoss(logits, targetTokenId) {
  const probabilities = softmax(logits);
  return -Math.log(probabilities[targetTokenId]);
}

function languageModelLoss(logitsByPosition, targetTokenIds) {
  let total = 0;

  for (let position = 0; position < targetTokenIds.length; position++) {
    total += onePositionLoss(logitsByPosition[position], targetTokenIds[position]);
  }

  return total / targetTokenIds.length;
}`,explanation:"Language modeling is many small classification losses, one for each predicted next token."},{id:"teacher-forcing-previous-token",stepLabel:"51.1",group:"Teacher forcing",title:"True previous token",concept:"Teacher forcing feeds the true previous token during training.",objective:"Return trueTokens[position - 1].",difficulty:"warmup",starterCode:`function previousTrueToken(trueTokens, position) {
  // position is greater than 0.
  // TODO: return the true previous token.
  return null;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('previous at position 1', previousTrueToken(['A', 'B', 'C'], 1), 'A');
check('previous at position 2', previousTrueToken(['A', 'B', 'C'], 2), 'B');
check('previous at position 3', previousTrueToken(['A', 'B', 'C', 'D'], 3), 'C');

return results;`,hints:["Previous position is position - 1.","Index into trueTokens.","return trueTokens[position - 1];"],solution:`function previousTrueToken(trueTokens, position) {
  return trueTokens[position - 1];
}`,explanation:"During training, teacher forcing gives the model the correct previous context instead of its own sampled mistakes."},{id:"teacher-forcing-inputs",stepLabel:"51.2",group:"Teacher forcing",title:"Teacher-forced inputs",concept:"Training inputs are usually shifted right: start token followed by all true tokens except the last.",objective:"Build [startToken, ...tokensWithoutLast].",difficulty:"core",starterCode:`function teacherForcedInputs(tokens, startToken) {
  const inputs = [startToken];

  for (let i = 0; i < tokens.length - 1; i++) {
    // TODO: append the true token at position i.
    inputs.push(null);
  }

  return inputs;
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

check('ABC', teacherForcedInputs(['A', 'B', 'C'], '<bos>'), ['<bos>', 'A', 'B']);
check('one token', teacherForcedInputs(['A'], '<bos>'), ['<bos>']);
check('four tokens', teacherForcedInputs(['A', 'B', 'C', 'D'], '<bos>'), ['<bos>', 'A', 'B', 'C']);

return results;`,hints:["The loop already stops before the last token.","Push tokens[i].","inputs.push(tokens[i]);"],solution:`function teacherForcedInputs(tokens, startToken) {
  const inputs = [startToken];

  for (let i = 0; i < tokens.length - 1; i++) {
    inputs.push(tokens[i]);
  }

  return inputs;
}`,explanation:"Teacher forcing trains the model to predict token t using the true tokens before t."},{id:"teacher-forcing-targets",stepLabel:"51.3",group:"Teacher forcing",title:"Teacher-forced targets",concept:"For next-token training, targets are the original token sequence.",objective:"Return a copy of tokens.",difficulty:"warmup",starterCode:`function teacherForcedTargets(tokens) {
  // TODO: return the target tokens.
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

check('ABC', teacherForcedTargets(['A', 'B', 'C']), ['A', 'B', 'C']);
check('one token', teacherForcedTargets(['A']), ['A']);

return results;`,hints:["Targets are the true sequence.","Return a shallow copy so you do not mutate the input.","return tokens.slice();"],solution:`function teacherForcedTargets(tokens) {
  return tokens.slice();
}`,explanation:"Inputs are shifted right; targets are the true next tokens to predict."},{id:"causal-labels-drop-first",stepLabel:"52.1",group:"Causal label shifting",title:"Drop first token for labels",concept:"In causal LM training, each position predicts the next token.",objective:"Return tokens from index 1 onward.",difficulty:"warmup",starterCode:`function nextTokenLabels(tokens) {
  // TODO: return all tokens except the first.
  return tokens;
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

check('ABC labels', nextTokenLabels(['A', 'B', 'C']), ['B', 'C']);
check('AB labels', nextTokenLabels(['A', 'B']), ['B']);
check('one token labels', nextTokenLabels(['A']), []);

return results;`,hints:["The first token has no previous token predicting it in this simple setup.","Use slice starting at index 1.","return tokens.slice(1);"],solution:`function nextTokenLabels(tokens) {
  return tokens.slice(1);
}`,explanation:"For sequence A B C, the model can learn A -> B and B -> C."},{id:"causal-inputs-drop-last",stepLabel:"52.2",group:"Causal label shifting",title:"Drop last token for inputs",concept:"The last token has no next-token target inside the sequence.",objective:"Return all tokens except the last.",difficulty:"warmup",starterCode:`function causalInputs(tokens) {
  // TODO: return all tokens except the last.
  return tokens;
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

check('ABC inputs', causalInputs(['A', 'B', 'C']), ['A', 'B']);
check('AB inputs', causalInputs(['A', 'B']), ['A']);
check('one token inputs', causalInputs(['A']), []);

return results;`,hints:["Use slice from the start to length - 1.","The last token is a target, not an input for a next token within this sequence.","return tokens.slice(0, tokens.length - 1);"],solution:`function causalInputs(tokens) {
  return tokens.slice(0, tokens.length - 1);
}`,explanation:"Causal inputs and next-token labels are offset by one position."},{id:"causal-input-label-pairs",stepLabel:"52.3",group:"Causal label shifting",title:"Input-label pairs",concept:"Causal language modeling turns a sequence into pairs: current token -> next token.",objective:"Push [tokens[i], tokens[i + 1]].",difficulty:"core",starterCode:`function causalPairs(tokens) {
  const pairs = [];

  for (let i = 0; i < tokens.length - 1; i++) {
    // TODO: push current token and next token as a pair.
    pairs.push([]);
  }

  return pairs;
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

check('ABC pairs', causalPairs(['A', 'B', 'C']), [['A', 'B'], ['B', 'C']]);
check('AB pairs', causalPairs(['A', 'B']), [['A', 'B']]);
check('one token pairs', causalPairs(['A']), []);

return results;`,hints:["Each pair is current token and next token.","Use tokens[i] and tokens[i + 1].","pairs.push([tokens[i], tokens[i + 1]]);"],solution:`function causalPairs(tokens) {
  const pairs = [];

  for (let i = 0; i < tokens.length - 1; i++) {
    pairs.push([tokens[i], tokens[i + 1]]);
  }

  return pairs;
}`,explanation:"Next-token prediction is supervised learning over shifted token pairs."},{id:"token-training-logit-gradient",stepLabel:"53.1",group:"Mini token training step",title:"Logit gradient",concept:"For softmax + cross-entropy, gradient is probabilities minus one-hot target.",objective:"Push probabilities[i] - target.",difficulty:"core",starterCode:`function logitGradient(probabilities, targetId) {
  const gradient = [];

  for (let i = 0; i < probabilities.length; i++) {
    const target = i === targetId ? 1 : 0;

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

check('target 0', logitGradient([0.7, 0.3], 0), [-0.3, 0.3]);
check('target 1', logitGradient([0.7, 0.3], 1), [0.7, -0.7]);
check('three classes', logitGradient([0.1, 0.8, 0.1], 1), [0.1, -0.2, 0.1]);

return results;`,hints:["The formula is p - y.","target is 1 for the true class and 0 otherwise.","gradient.push(probabilities[i] - target);"],solution:`function logitGradient(probabilities, targetId) {
  const gradient = [];

  for (let i = 0; i < probabilities.length; i++) {
    const target = i === targetId ? 1 : 0;
    gradient.push(probabilities[i] - target);
  }

  return gradient;
}`,explanation:"The true token logit is pushed up, and competing token logits are pushed down."},{id:"token-training-update-logit",stepLabel:"53.2",group:"Mini token training step",title:"Update one logit",concept:"A gradient step subtracts learningRate times gradient from a parameter.",objective:"Return logit - learningRate * gradient.",difficulty:"warmup",starterCode:`function updateLogit(logit, gradient, learningRate) {
  // TODO: return updated logit.
  return logit;
}`,testCode:`const results = [];

function approxEqual(a, b, tolerance = 1e-9) {
  return Math.abs(a - b) <= tolerance;
}

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: approxEqual(actual, expected) });
}

check('negative gradient increases logit', updateLogit(1, -0.3, 0.1), 1.03);
check('positive gradient decreases logit', updateLogit(1, 0.7, 0.1), 0.93);
check('zero gradient no change', updateLogit(5, 0, 0.1), 5);

return results;`,hints:["Gradient descent subtracts the gradient step.","Use logit - learningRate * gradient.","return logit - learningRate * gradient;"],solution:`function updateLogit(logit, gradient, learningRate) {
  return logit - learningRate * gradient;
}`,explanation:"When the true class gradient is negative, subtracting it increases that logit."},{id:"token-training-update-all-logits",stepLabel:"53.3",group:"Mini token training step",title:"Update all logits",concept:"One token-prediction training step updates every vocabulary logit.",objective:"Push logits[i] - learningRate * gradients[i].",difficulty:"core",starterCode:`function updateAllLogits(logits, gradients, learningRate) {
  const updated = [];

  for (let i = 0; i < logits.length; i++) {
    // TODO: update this logit.
    updated.push(logits[i]);
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

check('binary update', updateAllLogits([1, 1], [-0.3, 0.3], 0.1), [1.03, 0.97]);
check('three-class update', updateAllLogits([0, 0, 0], [0.1, -0.2, 0.1], 0.5), [-0.05, 0.1, -0.05]);

return results;`,hints:["Use the same SGD rule for every logit.","Subtract learningRate * gradients[i].","updated.push(logits[i] - learningRate * gradients[i]);"],solution:`function updateAllLogits(logits, gradients, learningRate) {
  const updated = [];

  for (let i = 0; i < logits.length; i++) {
    updated.push(logits[i] - learningRate * gradients[i]);
  }

  return updated;
}`,explanation:"A training step increases the true token score and lowers competing scores."},{id:"sampling-cumulative-pick",stepLabel:"54.1",group:"Sampling from logits",title:"Pick from cumulative probabilities",concept:"Sampling chooses the first cumulative probability that exceeds a random number.",objective:"Return the first index where cumulative probability exceeds r.",difficulty:"core",starterCode:`function sampleFromProbabilities(probabilities, r) {
  let cumulative = 0;

  for (let i = 0; i < probabilities.length; i++) {
    cumulative += probabilities[i];

    // TODO: return i when r is less than cumulative.
  }

  return probabilities.length - 1;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('r in first bucket', sampleFromProbabilities([0.2, 0.3, 0.5], 0.1), 0);
check('r in second bucket', sampleFromProbabilities([0.2, 0.3, 0.5], 0.25), 1);
check('r in third bucket', sampleFromProbabilities([0.2, 0.3, 0.5], 0.8), 2);

return results;`,hints:["cumulative is the probability mass up to index i.","If r < cumulative, choose i.","if (r < cumulative) return i;"],solution:`function sampleFromProbabilities(probabilities, r) {
  let cumulative = 0;

  for (let i = 0; i < probabilities.length; i++) {
    cumulative += probabilities[i];

    if (r < cumulative) return i;
  }

  return probabilities.length - 1;
}`,explanation:"Sampling turns a probability distribution into one selected token ID."},{id:"sampling-token-from-vocab",stepLabel:"54.2",group:"Sampling from logits",title:"Sample token from vocabulary",concept:"After sampling a token ID, decode it through the vocabulary.",objective:"Return vocab[sampledIndex].",difficulty:"warmup",starterCode:`function sampleFromProbabilities(probabilities, r) {
  let cumulative = 0;

  for (let i = 0; i < probabilities.length; i++) {
    cumulative += probabilities[i];
    if (r < cumulative) return i;
  }

  return probabilities.length - 1;
}

function sampleToken(vocab, probabilities, r) {
  const sampledIndex = sampleFromProbabilities(probabilities, r);

  // TODO: return the token at sampledIndex.
  return '';
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

const vocab = ['cat', 'dog', 'fish'];

check('sample cat', sampleToken(vocab, [0.2, 0.3, 0.5], 0.1), 'cat');
check('sample dog', sampleToken(vocab, [0.2, 0.3, 0.5], 0.25), 'dog');
check('sample fish', sampleToken(vocab, [0.2, 0.3, 0.5], 0.8), 'fish');

return results;`,hints:["sampledIndex is already computed.","Use it to index into vocab.","return vocab[sampledIndex];"],solution:`function sampleFromProbabilities(probabilities, r) {
  let cumulative = 0;

  for (let i = 0; i < probabilities.length; i++) {
    cumulative += probabilities[i];
    if (r < cumulative) return i;
  }

  return probabilities.length - 1;
}

function sampleToken(vocab, probabilities, r) {
  const sampledIndex = sampleFromProbabilities(probabilities, r);
  return vocab[sampledIndex];
}`,explanation:"Sampling can produce different valid continuations from the same model distribution."},{id:"sampling-greedy-or-sample",stepLabel:"54.3",group:"Sampling from logits",title:"Greedy or sample",concept:"Generation can choose the highest-probability token or sample from the distribution.",objective:'Use greedy when mode is "greedy", otherwise sample.',difficulty:"core",starterCode:`function argmax(values) {
  let bestIndex = 0;
  let bestValue = values[0];

  for (let i = 1; i < values.length; i++) {
    if (values[i] > bestValue) {
      bestValue = values[i];
      bestIndex = i;
    }
  }

  return bestIndex;
}

function sampleFromProbabilities(probabilities, r) {
  let cumulative = 0;
  for (let i = 0; i < probabilities.length; i++) {
    cumulative += probabilities[i];
    if (r < cumulative) return i;
  }
  return probabilities.length - 1;
}

function chooseTokenId(probabilities, mode, r) {
  // TODO: if mode is "greedy", return argmax; otherwise sample.
  return 0;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('greedy chooses largest', chooseTokenId([0.2, 0.3, 0.5], 'greedy', 0.1), 2);
check('sample first bucket', chooseTokenId([0.2, 0.3, 0.5], 'sample', 0.1), 0);
check('sample second bucket', chooseTokenId([0.2, 0.3, 0.5], 'sample', 0.25), 1);

return results;`,hints:["Greedy ignores r and picks argmax.","Sampling uses sampleFromProbabilities.",'return mode === "greedy" ? argmax(probabilities) : sampleFromProbabilities(probabilities, r);'],solution:`function argmax(values) {
  let bestIndex = 0;
  let bestValue = values[0];

  for (let i = 1; i < values.length; i++) {
    if (values[i] > bestValue) {
      bestValue = values[i];
      bestIndex = i;
    }
  }

  return bestIndex;
}

function sampleFromProbabilities(probabilities, r) {
  let cumulative = 0;
  for (let i = 0; i < probabilities.length; i++) {
    cumulative += probabilities[i];
    if (r < cumulative) return i;
  }
  return probabilities.length - 1;
}

function chooseTokenId(probabilities, mode, r) {
  return mode === "greedy" ? argmax(probabilities) : sampleFromProbabilities(probabilities, r);
}`,explanation:"Greedy decoding is stable but can be dull; sampling is more diverse but less predictable."},{id:"temperature-scale-logits",stepLabel:"55.1",group:"Temperature and top-k / top-p",title:"Temperature-scaled logits",concept:"Temperature divides logits before softmax. Lower temperature sharpens; higher temperature flattens.",objective:"Push logits[i] / temperature.",difficulty:"core",starterCode:`function applyTemperature(logits, temperature) {
  const scaled = [];

  for (let i = 0; i < logits.length; i++) {
    // TODO: divide logit by temperature.
    scaled.push(logits[i]);
  }

  return scaled;
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

check('temperature 1', applyTemperature([2, 4], 1), [2, 4]);
check('temperature 2', applyTemperature([2, 4], 2), [1, 2]);
check('temperature 0.5', applyTemperature([2, 4], 0.5), [4, 8]);

return results;`,hints:["Temperature rescales every logit.","Divide by temperature.","scaled.push(logits[i] / temperature);"],solution:`function applyTemperature(logits, temperature) {
  const scaled = [];

  for (let i = 0; i < logits.length; i++) {
    scaled.push(logits[i] / temperature);
  }

  return scaled;
}`,explanation:"Temperature changes how sharp the final softmax distribution becomes."},{id:"top-k-indices",stepLabel:"55.2",group:"Temperature and top-k / top-p",title:"Top-k indices",concept:"Top-k sampling keeps only the k highest-scoring tokens.",objective:"Return indices of the top k logits.",difficulty:"challenge",starterCode:`function topKIndices(logits, k) {
  const indexed = logits.map((value, index) => ({ value, index }));

  indexed.sort((a, b) => b.value - a.value);

  // TODO: return the first k indices.
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

check('top 1', topKIndices([1, 5, 3], 1), [1]);
check('top 2', topKIndices([1, 5, 3], 2), [1, 2]);
check('top 3', topKIndices([-1, -5, 0], 3), [2, 0, 1]);

return results;`,hints:["indexed is already sorted from largest to smallest.","Take the first k entries and return their index fields.","return indexed.slice(0, k).map((item) => item.index);"],solution:`function topKIndices(logits, k) {
  const indexed = logits.map((value, index) => ({ value, index }));

  indexed.sort((a, b) => b.value - a.value);

  return indexed.slice(0, k).map((item) => item.index);
}`,explanation:"Top-k prevents low-ranked tokens from being sampled at all."},{id:"top-k-mask-logits",stepLabel:"55.3",group:"Temperature and top-k / top-p",title:"Mask non-top-k logits",concept:"Tokens outside top-k are masked to -Infinity before softmax.",objective:"Keep logits in allowed indices, otherwise -Infinity.",difficulty:"challenge",starterCode:`function maskToTopK(logits, allowedIndices) {
  const masked = [];

  for (let i = 0; i < logits.length; i++) {
    // TODO: keep logits[i] only if i is in allowedIndices.
    masked.push(logits[i]);
  }

  return masked;
}`,testCode:`const results = [];

function sameArraySpecial(a, b) {
  return a.length === b.length && a.every((value, index) => Object.is(value, b[index]));
}

function check(name, actual, expected) {
  results.push({
    name,
    actual: JSON.stringify(actual),
    expected: JSON.stringify(expected),
    passed: sameArraySpecial(actual, expected),
  });
}

check('keep indices 1 and 2', maskToTopK([1, 5, 3], [1, 2]), [-Infinity, 5, 3]);
check('keep index 0', maskToTopK([1, 5, 3], [0]), [1, -Infinity, -Infinity]);

return results;`,hints:["Use allowedIndices.includes(i).","Keep the logit when allowed; otherwise use -Infinity.","masked.push(allowedIndices.includes(i) ? logits[i] : -Infinity);"],solution:`function maskToTopK(logits, allowedIndices) {
  const masked = [];

  for (let i = 0; i < logits.length; i++) {
    masked.push(allowedIndices.includes(i) ? logits[i] : -Infinity);
  }

  return masked;
}`,explanation:"Masking before softmax makes excluded tokens receive zero probability."},{id:"top-p-cutoff",stepLabel:"55.4",group:"Temperature and top-k / top-p",title:"Top-p cutoff",concept:"Top-p keeps the smallest set of high-probability tokens whose cumulative mass reaches p.",objective:"Return how many sorted probabilities are needed to reach p.",difficulty:"challenge",starterCode:`function topPCount(sortedProbabilities, p) {
  let cumulative = 0;

  for (let i = 0; i < sortedProbabilities.length; i++) {
    cumulative += sortedProbabilities[i];

    // TODO: return i + 1 once cumulative reaches p.
  }

  return sortedProbabilities.length;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('one token enough', topPCount([0.8, 0.1, 0.1], 0.7), 1);
check('two tokens needed', topPCount([0.5, 0.3, 0.2], 0.8), 2);
check('all tokens needed', topPCount([0.4, 0.3, 0.2, 0.1], 0.95), 4);

return results;`,hints:["sortedProbabilities are already largest to smallest.","When cumulative >= p, return the number of tokens included.","if (cumulative >= p) return i + 1;"],solution:`function topPCount(sortedProbabilities, p) {
  let cumulative = 0;

  for (let i = 0; i < sortedProbabilities.length; i++) {
    cumulative += sortedProbabilities[i];

    if (cumulative >= p) return i + 1;
  }

  return sortedProbabilities.length;
}`,explanation:"Top-p adapts the candidate set size to the shape of the probability distribution."}];function te(){return e.jsx(Z,{exercises:ee})}const re=[[1,2],[3,1]],ae=[[2,1,3],[1,4,2]],ne=[[4,9,7],[7,7,11]],x=[{row:0,col:0,hint:"Multiply Row 1 of A with Column 1 of B: (1×2) + (2×1)",answer:4},{row:0,col:1,hint:"Multiply Row 1 of A with Column 2 of B: (1×1) + (2×4)",answer:9},{row:0,col:2,hint:"Multiply Row 1 of A with Column 3 of B: (1×3) + (2×2)",answer:7},{row:1,col:0,hint:"Multiply Row 2 of A with Column 1 of B: (3×2) + (1×1)",answer:7},{row:1,col:1,hint:"Multiply Row 2 of A with Column 2 of B: (3×1) + (1×4)",answer:7},{row:1,col:2,hint:"Multiply Row 2 of A with Column 3 of B: (3×3) + (1×2)",answer:11}];function se(){const[a,u]=y.useState(0),[c,r]=y.useState(""),[o,s]=y.useState(""),[k,g]=y.useState(!1),[w,O]=y.useState(Array(6).fill(null)),[N,M]=y.useState(!1),[B,S]=y.useState(0),[A,T]=y.useState(0),n=x[a],C=()=>{const i=parseInt(c,10);if(T(d=>d+1),i===n.answer){s("✓ Correct!"),S(f=>f+1);const d=[...w];d[a]=i,O(d),setTimeout(()=>{a<x.length-1?(u(f=>f+1),r(""),s(""),g(!1)):(M(!0),s("🎉 Excellent! You completed all steps!"))},1e3)}else s("✗ Not quite. Try again or ask for a hint.")},j=()=>{g(!0)},D=()=>{u(0),r(""),s(""),g(!1),O(Array(6).fill(null)),M(!1),S(0),T(0)},L=i=>{i.key==="Enter"&&c.trim()!==""&&C()},J=i=>a<x.length&&x[a].row===i,E=i=>a<x.length&&x[a].col===i;return e.jsxs("div",{className:"flex flex-col items-center p-3 h-full",children:[e.jsx("h2",{className:"text-xl font-bold text-gray-800 mb-2",children:"Practice Exercise"}),e.jsxs("div",{className:"bg-white rounded-lg shadow-lg p-4 w-full",children:[e.jsxs("div",{className:"flex items-center justify-center gap-2 flex-wrap",children:[e.jsxs("div",{className:"flex flex-col items-center",children:[e.jsx("span",{className:"text-lg font-bold mb-1",children:"A"}),e.jsx("div",{className:"grid grid-cols-2 gap-1",children:re.map((i,d)=>i.map((f,b)=>e.jsx("div",{className:`w-10 h-10 flex items-center justify-center font-bold text-black rounded ${J(d)?"bg-blue-300 scale-110 ring-2 ring-blue-500":"bg-blue-400"} transition-all`,children:f},`a-${d}-${b}`)))})]}),e.jsx("span",{className:"text-2xl font-bold mx-2",children:"×"}),e.jsxs("div",{className:"flex flex-col items-center",children:[e.jsx("span",{className:"text-lg font-bold mb-1",children:"B"}),e.jsx("div",{className:"grid grid-cols-3 gap-1",children:ae.map((i,d)=>i.map((f,b)=>e.jsx("div",{className:`w-10 h-10 flex items-center justify-center font-bold text-black rounded ${E(b)?"bg-green-300 scale-110 ring-2 ring-green-500":"bg-green-400"} transition-all`,children:f},`b-${d}-${b}`)))})]}),e.jsx("span",{className:"text-2xl font-bold mx-2",children:"="}),e.jsxs("div",{className:"flex flex-col items-center",children:[e.jsx("span",{className:"text-lg font-bold mb-1",children:"C"}),e.jsx("div",{className:"grid grid-cols-3 gap-1",children:ne.map((i,d)=>i.map((f,b)=>{const q=d*3+b,P=a===q,t=w[q]!==null;return e.jsx("div",{className:`w-10 h-10 flex items-center justify-center font-bold text-black rounded transition-all ${P?"bg-yellow-300 ring-2 ring-yellow-500 scale-110":t?"bg-orange-400":"bg-orange-200"}`,children:t?w[q]:"?"},`r-${d}-${b}`)}))})]})]}),e.jsxs("div",{className:"mt-4 text-center",children:[e.jsxs("p",{className:"text-gray-700 font-medium",children:["Step ",a+1," of ",x.length,": Calculate C[",n.row+1,"][",n.col+1,"]"]}),e.jsxs("p",{className:"text-sm text-gray-700 mt-1",children:["Row ",n.row+1," of A × Column ",n.col+1," of B"]})]})]}),N?e.jsx("div",{className:"mt-4 w-full max-w-sm text-center",children:e.jsxs("div",{className:"p-4 bg-green-100 rounded-lg border border-green-300",children:[e.jsx("p",{className:"text-green-700 font-bold text-lg",children:"🎉 Congratulations!"}),e.jsxs("p",{className:"text-green-600 mt-2",children:["Score: ",B," / ",x.length," correct"]}),e.jsxs("p",{className:"text-sm",children:["Total attempts: ",A]})]})}):e.jsxs("div",{className:"mt-4 w-full max-w-sm",children:[e.jsxs("div",{className:"flex gap-2",children:[e.jsx("input",{type:"number",value:c,onChange:i=>r(i.target.value),onKeyPress:L,placeholder:"Your answer...",className:"flex-1 px-4 py-2 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none text-center text-lg font-bold"}),e.jsx("button",{onClick:C,disabled:c.trim()==="",className:"px-4 py-2 bg-green-500 hover:bg-green-600 disabled:bg-gray-400 text-white font-bold rounded-lg transition-colors",children:"Submit"})]}),e.jsx("button",{onClick:j,className:"mt-2 w-full px-4 py-2 bg-yellow-500 hover:bg-yellow-600 text-white font-bold rounded-lg transition-colors",children:"💡 Show Hint"}),k&&e.jsx("div",{className:"mt-2 p-3 bg-yellow-100 rounded-lg border border-yellow-300",children:e.jsx("p",{className:"text-sm",children:n.hint})}),o&&e.jsx("div",{className:`mt-2 p-3 rounded-lg text-center font-bold ${o.includes("✓")?"bg-green-100 text-green-700":"bg-red-100 text-red-700"}`,children:o})]}),e.jsxs("div",{className:"mt-4 flex items-center gap-4",children:[e.jsxs("div",{className:"text-sm text-gray-800",children:["Progress: ",w.filter(i=>i!==null).length," / ",x.length]}),e.jsx("button",{onClick:D,className:"px-4 py-2 bg-red-500 hover:bg-red-600 text-white font-bold rounded-lg transition-colors text-sm",children:"↺ Reset"})]}),e.jsx("div",{className:"mt-8 w-full",children:e.jsx(te,{})})]})}export{se as default};
