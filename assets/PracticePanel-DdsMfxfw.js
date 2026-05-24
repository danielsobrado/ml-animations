import{a as k,j as e,k as g}from"./react-vendor-Cdu38Wyn.js";import{F as J,I as D,aZ as z,aQ as F,ar as _,$ as U,_ as K}from"./icons-C7miCxLM.js";let W=1;function Y({userCode:o,testCode:d,timeoutMs:c=1200}){const s=W++;return new Promise(r=>{const i=new Worker(new URL("/ml-animations/assets/jsEvalWorker-BiN8n73y.js",import.meta.url),{type:"module"}),w=window.setTimeout(()=>{i.terminate(),r({ok:!1,results:[],error:"Execution timed out. Check for an infinite loop."})},c);i.onmessage=h=>{h.data.id===s&&(window.clearTimeout(w),i.terminate(),r({ok:h.data.ok,results:h.data.results||[],error:h.data.error}))},i.postMessage({id:s,userCode:o,testCode:d})})}function L(o,d){return d?"error":o!=null&&o.length?o.every(c=>c.passed)?"passed":"failed":"idle"}function H(o){return typeof o=="string"?o:JSON.stringify(o)}const Z=/(\/\/.*|\/\*[\s\S]*?\*\/|(["'`])(?:\\.|(?!\2)[^\\])*\2|\b(?:const|let|var|function|return|if|else|for|while|new|throw|true|false|null|undefined)\b|\b\d+(?:\.\d+)?\b|\b[a-zA-Z_$][\w$]*(?=\s*\())/g;function q(o){const d=[];let c=0;for(const s of o.matchAll(Z)){s.index>c&&d.push(o.slice(c,s.index));const r=s[0];let i="plain";r.startsWith("//")||r.startsWith("/*")?i="comment":/^["'`]/.test(r)?i="string":/^\d/.test(r)?i="number":/^(const|let|var|function|return|if|else|for|while|new|throw|true|false|null|undefined)$/.test(r)?i="keyword":i="call",d.push(e.jsx("span",{className:`ua-code-token-${i}`,children:r},`${s.index}-${r}`)),c=s.index+r.length}return c<o.length&&d.push(o.slice(c)),d}function G({exercises:o}){var N;const[d,c]=k.useState(0),s=o[d],r=k.useRef(null),[i,w]=k.useState(()=>Object.fromEntries(o.map(t=>[t.id,t.starterCode]))),[h,b]=k.useState({}),[A,O]=k.useState({}),[S,R]=k.useState(!1),[v,j]=k.useState(!1),B=i[s.id],l=A[s.id],C=L(l==null?void 0:l.results,l==null?void 0:l.error),y=h[s.id]||0,P=s.hints.slice(0,y),T=!!(l||y>0);async function E(){R(!0),j(!1);const t=await Y({userCode:B,testCode:s.testCode});O(a=>({...a,[s.id]:t})),R(!1)}function $(){w(t=>({...t,[s.id]:s.starterCode})),O(t=>({...t,[s.id]:null})),b(t=>({...t,[s.id]:0})),j(!1)}function n(){b(t=>({...t,[s.id]:Math.min(s.hints.length,y+1)}))}function u(){w(t=>({...t,[s.id]:s.solution})),j(!1)}function p(t){r.current&&(r.current.scrollTop=t.currentTarget.scrollTop,r.current.scrollLeft=t.currentTarget.scrollLeft)}const m=Object.values(A).filter(t=>{var a;return((a=t==null?void 0:t.results)==null?void 0:a.length)&&t.results.every(x=>x.passed)}).length;return e.jsxs("section",{className:"ua-codefix-lab",children:[e.jsxs("div",{className:"ua-codefix-head",children:[e.jsx("span",{children:"Code Completion-style lab"}),e.jsx("h2",{children:"Fix the TODOs, run the tests"}),e.jsx("p",{children:"Each exercise is almost complete. Change the smallest piece of code needed to make the tests pass."})]}),e.jsx("div",{className:"ua-codefix-progress",children:o.map((t,a)=>{const x=A[t.id],I=L(x==null?void 0:x.results,x==null?void 0:x.error),M=I==="passed"?J:D;return e.jsxs("button",{type:"button",onClick:()=>{c(a),j(!1)},className:`ua-codefix-step ${a===d?"active":""} ${I}`,children:[e.jsx(M,{size:15,"aria-hidden":"true"}),e.jsxs("span",{children:[t.stepLabel||`${a+1}.`," ",t.title]})]},t.id)})}),e.jsxs("div",{className:"ua-codefix-grid",children:[e.jsxs("article",{className:"ua-codefix-card ua-codefix-instructions",children:[e.jsx("span",{children:s.difficulty}),e.jsx("h3",{children:s.title}),e.jsx("p",{children:s.objective}),e.jsxs("div",{className:"ua-codefix-concept",children:[e.jsx("strong",{children:"Concept"}),e.jsx("p",{children:s.concept})]}),e.jsxs("div",{className:"ua-codefix-explanation",children:[e.jsx("strong",{children:"After you pass"}),e.jsx("p",{children:s.explanation})]})]}),e.jsxs("article",{className:"ua-codefix-card ua-codefix-editor-card",children:[e.jsxs("div",{className:"ua-codefix-card-head",children:[e.jsxs("div",{children:[e.jsx("span",{children:"Editor"}),e.jsx("h3",{children:"Complete the TODO"})]}),e.jsxs("button",{type:"button",onClick:$,children:[e.jsx(z,{size:14,"aria-hidden":"true"}),"Reset"]})]}),e.jsxs("div",{className:"ua-codefix-editor-shell",children:[e.jsx("pre",{className:"ua-codefix-highlight","aria-hidden":"true",ref:r,children:q(B)}),e.jsx("textarea",{className:"ua-codefix-editor",value:B,spellCheck:!1,"aria-label":`${s.title} code editor`,onScroll:p,onChange:t=>w(a=>({...a,[s.id]:t.target.value}))})]}),e.jsxs("div",{className:"ua-codefix-actions",children:[e.jsxs("button",{type:"button",onClick:E,disabled:S,children:[e.jsx(F,{size:15,"aria-hidden":"true"}),S?"Running...":"Run tests"]}),e.jsxs("button",{type:"button",onClick:n,disabled:y>=s.hints.length,children:[e.jsx(_,{size:15,"aria-hidden":"true"}),y===0?"Show hint":"Next hint"]}),e.jsxs("button",{type:"button",onClick:()=>j(t=>!t),disabled:!T,title:T?void 0:"Run tests or use a hint before revealing the solution.",children:[v?e.jsx(U,{size:15,"aria-hidden":"true"}):e.jsx(K,{size:15,"aria-hidden":"true"}),v?"Hide solution":T?"See solution":"Try first"]})]})]}),e.jsxs("article",{className:"ua-codefix-card ua-codefix-feedback",children:[e.jsx("span",{children:"Checks"}),e.jsxs("h3",{children:[C==="passed"&&"All tests passed",C==="failed"&&"Keep going",C==="error"&&"Code error",C==="idle"&&"Run tests to begin"]}),(l==null?void 0:l.error)&&e.jsx("pre",{className:"ua-codefix-error",children:l.error}),((N=l==null?void 0:l.results)==null?void 0:N.length)>0?e.jsx("ul",{className:"ua-codefix-checks",children:l.results.map(t=>e.jsxs("li",{className:t.passed?"passed":"failed",children:[e.jsxs("strong",{children:[t.passed?"Pass":"Fail",": ",t.name]}),!t.passed&&e.jsxs("small",{children:["Expected ",H(t.expected),", got ",H(t.actual)]})]},t.name))}):e.jsx("p",{className:"ua-codefix-empty",children:"Run the tests. If one fails, use the smallest hint that helps."}),P.length>0&&e.jsxs("div",{className:"ua-codefix-hints",children:[e.jsx("strong",{children:"Hints"}),P.map((t,a)=>t.includes(`
`)?e.jsxs("div",{className:"ua-codefix-hint",children:[e.jsxs("b",{children:["Hint ",a+1,":"]}),e.jsx("pre",{className:"ua-codefix-hint-code",children:t})]},t):e.jsxs("p",{children:[e.jsxs("b",{children:["Hint ",a+1,":"]})," ",t]},t))]}),v&&e.jsxs("div",{className:"ua-codefix-solution",children:[e.jsx("strong",{children:"Solution"}),e.jsx("pre",{children:s.solution}),e.jsx("button",{type:"button",onClick:u,children:"Apply solution to editor"})]})]})]}),e.jsxs("div",{className:"ua-codefix-footer",children:[e.jsxs("strong",{children:[m," / ",o.length]}),e.jsx("span",{children:"exercises passed"})]})]})}const Q=[{id:"dot-product-first-pair",stepLabel:"1.1",title:"First matching pair",concept:"A dot product starts by multiplying entries with the same index.",objective:"Replace one number with the first pair product.",difficulty:"warmup",starterCode:`function firstPairProduct(a, b) {
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
}`,explanation:"The first contribution to a dot product comes from multiplying the two index-0 entries."},{id:"dot-product-two-pairs",stepLabel:"1.2",title:"Add two pair products",concept:"A two-entry dot product adds the first pair product and the second pair product.",objective:"Replace one expression with the missing second pair product.",difficulty:"warmup",starterCode:`function dot2(a, b) {
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
}`,explanation:"A two-entry dot product is a[0] * b[0] plus a[1] * b[1]."},{id:"dot-product-loop-update",stepLabel:"1.3",title:"Loop over every pair",concept:"The loop repeats the same pair-product rule for vectors of any length.",objective:"Complete the one accumulator update inside the loop.",difficulty:"core",starterCode:`function dot(a, b) {
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
}`,explanation:"The loop version is the same rule as dot2, repeated until every matching pair has contributed."},{id:"matrix-cell-one-term",stepLabel:"2.1",title:"One cell, first term",concept:"One matrix-product cell begins with A[row][0] times B[0][col].",objective:"Replace one expression with the first term of a row-column dot product.",difficulty:"core",starterCode:`function firstCellTerm(A, B, row, col) {
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
}`,explanation:"A matrix cell is a dot product; this is the first product in that dot product."},{id:"matrix-cell-loop-update",stepLabel:"2.2",title:"One cell loop",concept:"The index k moves across a row of A and down a column of B.",objective:"Complete the one accumulator update for a matrix cell.",difficulty:"core",starterCode:`function matrixCell(A, B, row, col) {
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
}`,explanation:"The complete cell is the sum of every row-column product for that row and column."},{id:"matrix-multiply-column-count",stepLabel:"3.1",title:"Output column count",concept:"The product A * B has one output column for each column in B.",objective:"Replace one number so the inner loop visits every output column.",difficulty:"challenge",starterCode:`function matrixCell(A, B, row, col) {
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
}`,explanation:"The shape of A * B is rows of A by columns of B, so the inner loop must run once per column in B."},{id:"matrix-multiply-push-cell",stepLabel:"3.2",title:"Push each computed cell",concept:"The nested loops choose each output position; matrixCell computes the value for that position.",objective:"Replace one argument so each row receives the computed cell value.",difficulty:"challenge",starterCode:`function matrixCell(A, B, row, col) {
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
}`,explanation:"The full matrix product is the matrixCell rule repeated for every row and every column."}];function V(){return e.jsx(G,{exercises:Q})}const X=[[1,2],[3,1]],ee=[[2,1,3],[1,4,2]],te=[[4,9,7],[7,7,11]],f=[{row:0,col:0,hint:"Multiply Row 1 of A with Column 1 of B: (1×2) + (2×1)",answer:4},{row:0,col:1,hint:"Multiply Row 1 of A with Column 2 of B: (1×1) + (2×4)",answer:9},{row:0,col:2,hint:"Multiply Row 1 of A with Column 3 of B: (1×3) + (2×2)",answer:7},{row:1,col:0,hint:"Multiply Row 2 of A with Column 1 of B: (3×2) + (1×1)",answer:7},{row:1,col:1,hint:"Multiply Row 2 of A with Column 2 of B: (3×1) + (1×4)",answer:7},{row:1,col:2,hint:"Multiply Row 2 of A with Column 3 of B: (3×3) + (1×2)",answer:11}];function le(){const[o,d]=g.useState(0),[c,s]=g.useState(""),[r,i]=g.useState(""),[w,h]=g.useState(!1),[b,A]=g.useState(Array(6).fill(null)),[O,S]=g.useState(!1),[R,v]=g.useState(0),[j,B]=g.useState(0),l=f[o],C=()=>{const n=parseInt(c,10);if(B(u=>u+1),n===l.answer){i("✓ Correct!"),v(p=>p+1);const u=[...b];u[o]=n,A(u),setTimeout(()=>{o<f.length-1?(d(p=>p+1),s(""),i(""),h(!1)):(S(!0),i("🎉 Excellent! You completed all steps!"))},1e3)}else i("✗ Not quite. Try again or ask for a hint.")},y=()=>{h(!0)},P=()=>{d(0),s(""),i(""),h(!1),A(Array(6).fill(null)),S(!1),v(0),B(0)},T=n=>{n.key==="Enter"&&c.trim()!==""&&C()},E=n=>o<f.length&&f[o].row===n,$=n=>o<f.length&&f[o].col===n;return e.jsxs("div",{className:"flex flex-col items-center p-3 h-full",children:[e.jsx("h2",{className:"text-xl font-bold text-gray-800 mb-2",children:"Practice Exercise"}),e.jsxs("div",{className:"bg-white rounded-lg shadow-lg p-4 w-full",children:[e.jsxs("div",{className:"flex items-center justify-center gap-2 flex-wrap",children:[e.jsxs("div",{className:"flex flex-col items-center",children:[e.jsx("span",{className:"text-lg font-bold mb-1",children:"A"}),e.jsx("div",{className:"grid grid-cols-2 gap-1",children:X.map((n,u)=>n.map((p,m)=>e.jsx("div",{className:`w-10 h-10 flex items-center justify-center font-bold text-black rounded ${E(u)?"bg-blue-300 scale-110 ring-2 ring-blue-500":"bg-blue-400"} transition-all`,children:p},`a-${u}-${m}`)))})]}),e.jsx("span",{className:"text-2xl font-bold mx-2",children:"×"}),e.jsxs("div",{className:"flex flex-col items-center",children:[e.jsx("span",{className:"text-lg font-bold mb-1",children:"B"}),e.jsx("div",{className:"grid grid-cols-3 gap-1",children:ee.map((n,u)=>n.map((p,m)=>e.jsx("div",{className:`w-10 h-10 flex items-center justify-center font-bold text-black rounded ${$(m)?"bg-green-300 scale-110 ring-2 ring-green-500":"bg-green-400"} transition-all`,children:p},`b-${u}-${m}`)))})]}),e.jsx("span",{className:"text-2xl font-bold mx-2",children:"="}),e.jsxs("div",{className:"flex flex-col items-center",children:[e.jsx("span",{className:"text-lg font-bold mb-1",children:"C"}),e.jsx("div",{className:"grid grid-cols-3 gap-1",children:te.map((n,u)=>n.map((p,m)=>{const N=u*3+m,t=o===N,a=b[N]!==null;return e.jsx("div",{className:`w-10 h-10 flex items-center justify-center font-bold text-black rounded transition-all ${t?"bg-yellow-300 ring-2 ring-yellow-500 scale-110":a?"bg-orange-400":"bg-orange-200"}`,children:a?b[N]:"?"},`r-${u}-${m}`)}))})]})]}),e.jsxs("div",{className:"mt-4 text-center",children:[e.jsxs("p",{className:"text-gray-700 font-medium",children:["Step ",o+1," of ",f.length,": Calculate C[",l.row+1,"][",l.col+1,"]"]}),e.jsxs("p",{className:"text-sm text-gray-700 mt-1",children:["Row ",l.row+1," of A × Column ",l.col+1," of B"]})]})]}),O?e.jsx("div",{className:"mt-4 w-full max-w-sm text-center",children:e.jsxs("div",{className:"p-4 bg-green-100 rounded-lg border border-green-300",children:[e.jsx("p",{className:"text-green-700 font-bold text-lg",children:"🎉 Congratulations!"}),e.jsxs("p",{className:"text-green-600 mt-2",children:["Score: ",R," / ",f.length," correct"]}),e.jsxs("p",{className:"text-sm",children:["Total attempts: ",j]})]})}):e.jsxs("div",{className:"mt-4 w-full max-w-sm",children:[e.jsxs("div",{className:"flex gap-2",children:[e.jsx("input",{type:"number",value:c,onChange:n=>s(n.target.value),onKeyPress:T,placeholder:"Your answer...",className:"flex-1 px-4 py-2 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none text-center text-lg font-bold"}),e.jsx("button",{onClick:C,disabled:c.trim()==="",className:"px-4 py-2 bg-green-500 hover:bg-green-600 disabled:bg-gray-400 text-white font-bold rounded-lg transition-colors",children:"Submit"})]}),e.jsx("button",{onClick:y,className:"mt-2 w-full px-4 py-2 bg-yellow-500 hover:bg-yellow-600 text-white font-bold rounded-lg transition-colors",children:"💡 Show Hint"}),w&&e.jsx("div",{className:"mt-2 p-3 bg-yellow-100 rounded-lg border border-yellow-300",children:e.jsx("p",{className:"text-sm",children:l.hint})}),r&&e.jsx("div",{className:`mt-2 p-3 rounded-lg text-center font-bold ${r.includes("✓")?"bg-green-100 text-green-700":"bg-red-100 text-red-700"}`,children:r})]}),e.jsxs("div",{className:"mt-4 flex items-center gap-4",children:[e.jsxs("div",{className:"text-sm text-gray-800",children:["Progress: ",b.filter(n=>n!==null).length," / ",f.length]}),e.jsx("button",{onClick:P,className:"px-4 py-2 bg-red-500 hover:bg-red-600 text-white font-bold rounded-lg transition-colors text-sm",children:"↺ Reset"})]}),e.jsx("div",{className:"mt-8 w-full",children:e.jsx(V,{})})]})}export{le as default};
