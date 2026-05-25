import{a as y,j as e,k as w}from"./react-vendor-Cdu38Wyn.js";import{F as $,I as V,aZ as z,aQ as F,ar as _,$ as K,_ as W}from"./icons-C7miCxLM.js";let G=1;function Y({userCode:r,testCode:u,timeoutMs:c=1200}){const o=G++;return new Promise(s=>{const a=new Worker(new URL("/ml-animations/assets/jsEvalWorker-BiN8n73y.js",import.meta.url),{type:"module"}),v=window.setTimeout(()=>{a.terminate(),s({ok:!1,results:[],error:"Execution timed out. Check for an infinite loop."})},c);a.onmessage=f=>{f.data.id===o&&(window.clearTimeout(v),a.terminate(),s({ok:f.data.ok,results:f.data.results||[],error:f.data.error}))},a.postMessage({id:o,userCode:r,testCode:u})})}function P(r,u){return u?"error":r!=null&&r.length?r.every(c=>c.passed)?"passed":"failed":"idle"}function U(r){return typeof r=="string"?r:JSON.stringify(r)}const Z=/(\/\/.*|\/\*[\s\S]*?\*\/|(["'`])(?:\\.|(?!\2)[^\\])*\2|\b(?:const|let|var|function|return|if|else|for|while|new|throw|true|false|null|undefined)\b|\b\d+(?:\.\d+)?\b|\b[a-zA-Z_$][\w$]*(?=\s*\())/g;function Q(r){const u=[];let c=0;for(const o of r.matchAll(Z)){o.index>c&&u.push(r.slice(c,o.index));const s=o[0];let a="plain";s.startsWith("//")||s.startsWith("/*")?a="comment":/^["'`]/.test(s)?a="string":/^\d/.test(s)?a="number":/^(const|let|var|function|return|if|else|for|while|new|throw|true|false|null|undefined)$/.test(s)?a="keyword":a="call",u.push(e.jsx("span",{className:`ua-code-token-${a}`,children:s},`${o.index}-${s}`)),c=o.index+s.length}return c<r.length&&u.push(r.slice(c)),u}function X({exercises:r}){var R;const[u,c]=y.useState(0),o=r[u],s=y.useRef(null),[a,v]=y.useState(()=>Object.fromEntries(r.map(t=>[t.id,t.starterCode]))),[f,j]=y.useState({}),[k,E]=y.useState({}),[N,L]=y.useState(!1),[O,A]=y.useState(!1),T=a[o.id],n=k[o.id],C=P(n==null?void 0:n.results,n==null?void 0:n.error),S=f[o.id]||0,M=o.hints.slice(0,S),B=!!(n||S>0),I=y.useMemo(()=>{const t=[];return r.forEach((l,p)=>{const m=l.group||"Exercises",h=t[t.length-1];if((h==null?void 0:h.name)===m){h.items.push({exercise:l,index:p});return}t.push({name:m,items:[{exercise:l,index:p}]})}),t},[r]);async function J(){L(!0),A(!1);const t=await Y({userCode:T,testCode:o.testCode});E(l=>({...l,[o.id]:t})),L(!1)}function i(){v(t=>({...t,[o.id]:o.starterCode})),E(t=>({...t,[o.id]:null})),j(t=>({...t,[o.id]:0})),A(!1)}function d(){j(t=>({...t,[o.id]:Math.min(o.hints.length,S+1)}))}function x(){v(t=>({...t,[o.id]:o.solution})),A(!1)}function g(t){s.current&&(s.current.scrollTop=t.currentTarget.scrollTop,s.current.scrollLeft=t.currentTarget.scrollLeft)}const q=Object.values(k).filter(t=>{var l;return((l=t==null?void 0:t.results)==null?void 0:l.length)&&t.results.every(p=>p.passed)}).length;return e.jsxs("section",{className:"ua-codefix-lab",children:[e.jsxs("div",{className:"ua-codefix-head",children:[e.jsx("span",{children:"Code Completion-style lab"}),e.jsx("h2",{children:"Fix the TODOs, run the tests"}),e.jsx("p",{children:"Each exercise is almost complete. Change the smallest piece of code needed to make the tests pass."})]}),e.jsx("div",{className:"ua-codefix-progress",children:I.map(t=>{const l=t.items.filter(({exercise:p})=>{var h;const m=k[p.id];return((h=m==null?void 0:m.results)==null?void 0:h.length)&&m.results.every(D=>D.passed)}).length;return e.jsxs("div",{className:"ua-codefix-progress-group",children:[e.jsxs("div",{className:"ua-codefix-progress-label",children:[e.jsx("strong",{children:t.name}),e.jsxs("span",{children:[l,"/",t.items.length]})]}),e.jsx("div",{className:"ua-codefix-progress-steps",children:t.items.map(({exercise:p,index:m})=>{const h=k[p.id],D=P(h==null?void 0:h.results,h==null?void 0:h.error),H=D==="passed"?$:V;return e.jsxs("button",{type:"button",onClick:()=>{c(m),A(!1)},className:`ua-codefix-step ${m===u?"active":""} ${D}`,children:[e.jsx(H,{size:15,"aria-hidden":"true"}),e.jsxs("span",{children:[p.stepLabel||`${m+1}.`," ",p.title]})]},p.id)})})]},t.name)})}),e.jsxs("div",{className:"ua-codefix-grid",children:[e.jsxs("article",{className:"ua-codefix-card ua-codefix-instructions",children:[e.jsx("span",{children:o.difficulty}),e.jsx("h3",{children:o.title}),e.jsx("p",{children:o.objective}),e.jsxs("div",{className:"ua-codefix-concept",children:[e.jsx("strong",{children:"Concept"}),e.jsx("p",{children:o.concept})]}),e.jsxs("div",{className:"ua-codefix-explanation",children:[e.jsx("strong",{children:"After you pass"}),e.jsx("p",{children:o.explanation})]})]}),e.jsxs("article",{className:"ua-codefix-card ua-codefix-editor-card",children:[e.jsxs("div",{className:"ua-codefix-card-head",children:[e.jsxs("div",{children:[e.jsx("span",{children:"Editor"}),e.jsx("h3",{children:"Complete the TODO"})]}),e.jsxs("button",{type:"button",onClick:i,children:[e.jsx(z,{size:14,"aria-hidden":"true"}),"Reset"]})]}),e.jsxs("div",{className:"ua-codefix-editor-shell",children:[e.jsx("pre",{className:"ua-codefix-highlight","aria-hidden":"true",ref:s,children:Q(T)}),e.jsx("textarea",{className:"ua-codefix-editor",value:T,spellCheck:!1,"aria-label":`${o.title} code editor`,onScroll:g,onChange:t=>v(l=>({...l,[o.id]:t.target.value}))})]}),e.jsxs("div",{className:"ua-codefix-actions",children:[e.jsxs("button",{type:"button",onClick:J,disabled:N,children:[e.jsx(F,{size:15,"aria-hidden":"true"}),N?"Running...":"Run tests"]}),e.jsxs("button",{type:"button",onClick:d,disabled:S>=o.hints.length,children:[e.jsx(_,{size:15,"aria-hidden":"true"}),S===0?"Show hint":"Next hint"]}),e.jsxs("button",{type:"button",onClick:()=>A(t=>!t),disabled:!B,title:B?void 0:"Run tests or use a hint before revealing the solution.",children:[O?e.jsx(K,{size:15,"aria-hidden":"true"}):e.jsx(W,{size:15,"aria-hidden":"true"}),O?"Hide solution":B?"See solution":"Try first"]})]})]}),e.jsxs("article",{className:"ua-codefix-card ua-codefix-feedback",children:[e.jsx("span",{children:"Checks"}),e.jsxs("h3",{children:[C==="passed"&&"All tests passed",C==="failed"&&"Keep going",C==="error"&&"Code error",C==="idle"&&"Run tests to begin"]}),(n==null?void 0:n.error)&&e.jsx("pre",{className:"ua-codefix-error",children:n.error}),((R=n==null?void 0:n.results)==null?void 0:R.length)>0?e.jsx("ul",{className:"ua-codefix-checks",children:n.results.map(t=>e.jsxs("li",{className:t.passed?"passed":"failed",children:[e.jsxs("strong",{children:[t.passed?"Pass":"Fail",": ",t.name]}),!t.passed&&e.jsxs("small",{children:["Expected ",U(t.expected),", got ",U(t.actual)]})]},t.name))}):e.jsx("p",{className:"ua-codefix-empty",children:"Run the tests. If one fails, use the smallest hint that helps."}),M.length>0&&e.jsxs("div",{className:"ua-codefix-hints",children:[e.jsx("strong",{children:"Hints"}),M.map((t,l)=>t.includes(`
`)?e.jsxs("div",{className:"ua-codefix-hint",children:[e.jsxs("b",{children:["Hint ",l+1,":"]}),e.jsx("pre",{className:"ua-codefix-hint-code",children:t})]},t):e.jsxs("p",{children:[e.jsxs("b",{children:["Hint ",l+1,":"]})," ",t]},t))]}),O&&e.jsxs("div",{className:"ua-codefix-solution",children:[e.jsx("strong",{children:"Solution"}),e.jsx("pre",{children:o.solution}),e.jsx("button",{type:"button",onClick:x,children:"Apply solution to editor"})]})]})]}),e.jsxs("div",{className:"ua-codefix-footer",children:[e.jsxs("strong",{children:[q," / ",r.length]}),e.jsx("span",{children:"exercises passed"})]})]})}const ee=[{id:"dot-product-first-pair",stepLabel:"1.1",group:"Dot product",title:"First matching pair",concept:"A dot product starts by multiplying entries with the same index.",objective:"Replace one number with the first pair product.",difficulty:"warmup",starterCode:`function firstPairProduct(a, b) {
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
}`,explanation:"Least squares minimizes RSS, the squared length of the error vector b - Ax."}];function te(){return e.jsx(X,{exercises:ee})}const oe=[[1,2],[3,1]],re=[[2,1,3],[1,4,2]],ne=[[4,9,7],[7,7,11]],b=[{row:0,col:0,hint:"Multiply Row 1 of A with Column 1 of B: (1×2) + (2×1)",answer:4},{row:0,col:1,hint:"Multiply Row 1 of A with Column 2 of B: (1×1) + (2×4)",answer:9},{row:0,col:2,hint:"Multiply Row 1 of A with Column 3 of B: (1×3) + (2×2)",answer:7},{row:1,col:0,hint:"Multiply Row 2 of A with Column 1 of B: (3×2) + (1×1)",answer:7},{row:1,col:1,hint:"Multiply Row 2 of A with Column 2 of B: (3×1) + (1×4)",answer:7},{row:1,col:2,hint:"Multiply Row 2 of A with Column 3 of B: (3×3) + (1×2)",answer:11}];function ae(){const[r,u]=w.useState(0),[c,o]=w.useState(""),[s,a]=w.useState(""),[v,f]=w.useState(!1),[j,k]=w.useState(Array(6).fill(null)),[E,N]=w.useState(!1),[L,O]=w.useState(0),[A,T]=w.useState(0),n=b[r],C=()=>{const i=parseInt(c,10);if(T(d=>d+1),i===n.answer){a("✓ Correct!"),O(x=>x+1);const d=[...j];d[r]=i,k(d),setTimeout(()=>{r<b.length-1?(u(x=>x+1),o(""),a(""),f(!1)):(N(!0),a("🎉 Excellent! You completed all steps!"))},1e3)}else a("✗ Not quite. Try again or ask for a hint.")},S=()=>{f(!0)},M=()=>{u(0),o(""),a(""),f(!1),k(Array(6).fill(null)),N(!1),O(0),T(0)},B=i=>{i.key==="Enter"&&c.trim()!==""&&C()},I=i=>r<b.length&&b[r].row===i,J=i=>r<b.length&&b[r].col===i;return e.jsxs("div",{className:"flex flex-col items-center p-3 h-full",children:[e.jsx("h2",{className:"text-xl font-bold text-gray-800 mb-2",children:"Practice Exercise"}),e.jsxs("div",{className:"bg-white rounded-lg shadow-lg p-4 w-full",children:[e.jsxs("div",{className:"flex items-center justify-center gap-2 flex-wrap",children:[e.jsxs("div",{className:"flex flex-col items-center",children:[e.jsx("span",{className:"text-lg font-bold mb-1",children:"A"}),e.jsx("div",{className:"grid grid-cols-2 gap-1",children:oe.map((i,d)=>i.map((x,g)=>e.jsx("div",{className:`w-10 h-10 flex items-center justify-center font-bold text-black rounded ${I(d)?"bg-blue-300 scale-110 ring-2 ring-blue-500":"bg-blue-400"} transition-all`,children:x},`a-${d}-${g}`)))})]}),e.jsx("span",{className:"text-2xl font-bold mx-2",children:"×"}),e.jsxs("div",{className:"flex flex-col items-center",children:[e.jsx("span",{className:"text-lg font-bold mb-1",children:"B"}),e.jsx("div",{className:"grid grid-cols-3 gap-1",children:re.map((i,d)=>i.map((x,g)=>e.jsx("div",{className:`w-10 h-10 flex items-center justify-center font-bold text-black rounded ${J(g)?"bg-green-300 scale-110 ring-2 ring-green-500":"bg-green-400"} transition-all`,children:x},`b-${d}-${g}`)))})]}),e.jsx("span",{className:"text-2xl font-bold mx-2",children:"="}),e.jsxs("div",{className:"flex flex-col items-center",children:[e.jsx("span",{className:"text-lg font-bold mb-1",children:"C"}),e.jsx("div",{className:"grid grid-cols-3 gap-1",children:ne.map((i,d)=>i.map((x,g)=>{const q=d*3+g,R=r===q,t=j[q]!==null;return e.jsx("div",{className:`w-10 h-10 flex items-center justify-center font-bold text-black rounded transition-all ${R?"bg-yellow-300 ring-2 ring-yellow-500 scale-110":t?"bg-orange-400":"bg-orange-200"}`,children:t?j[q]:"?"},`r-${d}-${g}`)}))})]})]}),e.jsxs("div",{className:"mt-4 text-center",children:[e.jsxs("p",{className:"text-gray-700 font-medium",children:["Step ",r+1," of ",b.length,": Calculate C[",n.row+1,"][",n.col+1,"]"]}),e.jsxs("p",{className:"text-sm text-gray-700 mt-1",children:["Row ",n.row+1," of A × Column ",n.col+1," of B"]})]})]}),E?e.jsx("div",{className:"mt-4 w-full max-w-sm text-center",children:e.jsxs("div",{className:"p-4 bg-green-100 rounded-lg border border-green-300",children:[e.jsx("p",{className:"text-green-700 font-bold text-lg",children:"🎉 Congratulations!"}),e.jsxs("p",{className:"text-green-600 mt-2",children:["Score: ",L," / ",b.length," correct"]}),e.jsxs("p",{className:"text-sm",children:["Total attempts: ",A]})]})}):e.jsxs("div",{className:"mt-4 w-full max-w-sm",children:[e.jsxs("div",{className:"flex gap-2",children:[e.jsx("input",{type:"number",value:c,onChange:i=>o(i.target.value),onKeyPress:B,placeholder:"Your answer...",className:"flex-1 px-4 py-2 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none text-center text-lg font-bold"}),e.jsx("button",{onClick:C,disabled:c.trim()==="",className:"px-4 py-2 bg-green-500 hover:bg-green-600 disabled:bg-gray-400 text-white font-bold rounded-lg transition-colors",children:"Submit"})]}),e.jsx("button",{onClick:S,className:"mt-2 w-full px-4 py-2 bg-yellow-500 hover:bg-yellow-600 text-white font-bold rounded-lg transition-colors",children:"💡 Show Hint"}),v&&e.jsx("div",{className:"mt-2 p-3 bg-yellow-100 rounded-lg border border-yellow-300",children:e.jsx("p",{className:"text-sm",children:n.hint})}),s&&e.jsx("div",{className:`mt-2 p-3 rounded-lg text-center font-bold ${s.includes("✓")?"bg-green-100 text-green-700":"bg-red-100 text-red-700"}`,children:s})]}),e.jsxs("div",{className:"mt-4 flex items-center gap-4",children:[e.jsxs("div",{className:"text-sm text-gray-800",children:["Progress: ",j.filter(i=>i!==null).length," / ",b.length]}),e.jsx("button",{onClick:M,className:"px-4 py-2 bg-red-500 hover:bg-red-600 text-white font-bold rounded-lg transition-colors text-sm",children:"↺ Reset"})]}),e.jsx("div",{className:"mt-8 w-full",children:e.jsx(te,{})})]})}export{ae as default};
