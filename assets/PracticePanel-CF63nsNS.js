import{a as C,j as e,k as b}from"./react-vendor-Cdu38Wyn.js";import{F as P,I as F,aZ as L,aQ as D,ar as z,$ as J,_ as U}from"./icons-C7miCxLM.js";let _=1;function K({userCode:l,testCode:m,timeoutMs:h=1200}){const s=_++;return new Promise(f=>{const r=new Worker(new URL("/ml-animations/assets/jsEvalWorker-BiN8n73y.js",import.meta.url),{type:"module"}),y=window.setTimeout(()=>{r.terminate(),f({ok:!1,results:[],error:"Execution timed out. Check for an infinite loop."})},h);r.onmessage=c=>{c.data.id===s&&(window.clearTimeout(y),r.terminate(),f({ok:c.data.ok,results:c.data.results||[],error:c.data.error}))},r.postMessage({id:s,userCode:l,testCode:m})})}function M(l,m){return m?"error":l!=null&&l.length?l.every(h=>h.passed)?"passed":"failed":"idle"}function $(l){return typeof l=="string"?l:JSON.stringify(l)}function Y({exercises:l}){var u;const[m,h]=C.useState(0),s=l[m],[f,r]=C.useState(()=>Object.fromEntries(l.map(t=>[t.id,t.starterCode]))),[y,c]=C.useState({}),[g,N]=C.useState({}),[S,v]=C.useState(!1),[k,j]=C.useState(!1),O=f[s.id],n=g[s.id],d=M(n==null?void 0:n.results,n==null?void 0:n.error),w=y[s.id]||0,E=s.hints.slice(0,w),A=!!(n||w>0);async function R(){v(!0),j(!1);const t=await K({userCode:O,testCode:s.testCode});N(i=>({...i,[s.id]:t})),v(!1)}function T(){r(t=>({...t,[s.id]:s.starterCode})),N(t=>({...t,[s.id]:null})),c(t=>({...t,[s.id]:0})),j(!1)}function I(){c(t=>({...t,[s.id]:Math.min(s.hints.length,w+1)}))}function o(){r(t=>({...t,[s.id]:s.solution})),j(!1)}const a=Object.values(g).filter(t=>{var i;return((i=t==null?void 0:t.results)==null?void 0:i.length)&&t.results.every(x=>x.passed)}).length;return e.jsxs("section",{className:"ua-codefix-lab",children:[e.jsxs("div",{className:"ua-codefix-head",children:[e.jsx("span",{children:"Rustlings-style lab"}),e.jsx("h2",{children:"Fix the TODOs, run the tests"}),e.jsx("p",{children:"Each exercise is almost complete. Change the smallest piece of code needed to make the tests pass."})]}),e.jsx("div",{className:"ua-codefix-progress",children:l.map((t,i)=>{const x=g[t.id],B=M(x==null?void 0:x.results,x==null?void 0:x.error),H=B==="passed"?P:F;return e.jsxs("button",{type:"button",onClick:()=>{h(i),j(!1)},className:`ua-codefix-step ${i===m?"active":""} ${B}`,children:[e.jsx(H,{size:15,"aria-hidden":"true"}),e.jsxs("span",{children:[i+1,". ",t.title]})]},t.id)})}),e.jsxs("div",{className:"ua-codefix-grid",children:[e.jsxs("article",{className:"ua-codefix-card ua-codefix-instructions",children:[e.jsx("span",{children:s.difficulty}),e.jsx("h3",{children:s.title}),e.jsx("p",{children:s.objective}),e.jsxs("div",{className:"ua-codefix-concept",children:[e.jsx("strong",{children:"Concept"}),e.jsx("p",{children:s.concept})]}),e.jsxs("div",{className:"ua-codefix-explanation",children:[e.jsx("strong",{children:"After you pass"}),e.jsx("p",{children:s.explanation})]})]}),e.jsxs("article",{className:"ua-codefix-card ua-codefix-editor-card",children:[e.jsxs("div",{className:"ua-codefix-card-head",children:[e.jsxs("div",{children:[e.jsx("span",{children:"Editor"}),e.jsx("h3",{children:"Complete the TODO"})]}),e.jsxs("button",{type:"button",onClick:T,children:[e.jsx(L,{size:14,"aria-hidden":"true"}),"Reset"]})]}),e.jsx("textarea",{className:"ua-codefix-editor",value:O,spellCheck:!1,"aria-label":`${s.title} code editor`,onChange:t=>r(i=>({...i,[s.id]:t.target.value}))}),e.jsxs("div",{className:"ua-codefix-actions",children:[e.jsxs("button",{type:"button",onClick:R,disabled:S,children:[e.jsx(D,{size:15,"aria-hidden":"true"}),S?"Running...":"Run tests"]}),e.jsxs("button",{type:"button",onClick:I,disabled:w>=s.hints.length,children:[e.jsx(z,{size:15,"aria-hidden":"true"}),w===0?"Show hint":"Next hint"]}),e.jsxs("button",{type:"button",onClick:()=>j(t=>!t),disabled:!A,title:A?void 0:"Run tests or use a hint before revealing the solution.",children:[k?e.jsx(J,{size:15,"aria-hidden":"true"}):e.jsx(U,{size:15,"aria-hidden":"true"}),k?"Hide solution":A?"See solution":"Try first"]})]})]}),e.jsxs("article",{className:"ua-codefix-card ua-codefix-feedback",children:[e.jsx("span",{children:"Checks"}),e.jsxs("h3",{children:[d==="passed"&&"All tests passed",d==="failed"&&"Keep going",d==="error"&&"Code error",d==="idle"&&"Run tests to begin"]}),(n==null?void 0:n.error)&&e.jsx("pre",{className:"ua-codefix-error",children:n.error}),((u=n==null?void 0:n.results)==null?void 0:u.length)>0?e.jsx("ul",{className:"ua-codefix-checks",children:n.results.map(t=>e.jsxs("li",{className:t.passed?"passed":"failed",children:[e.jsxs("strong",{children:[t.passed?"Pass":"Fail",": ",t.name]}),!t.passed&&e.jsxs("small",{children:["Expected ",$(t.expected),", got ",$(t.actual)]})]},t.name))}):e.jsx("p",{className:"ua-codefix-empty",children:"Run the tests. If one fails, use the smallest hint that helps."}),E.length>0&&e.jsxs("div",{className:"ua-codefix-hints",children:[e.jsx("strong",{children:"Hints"}),E.map((t,i)=>t.includes(`
`)?e.jsxs("div",{className:"ua-codefix-hint",children:[e.jsxs("b",{children:["Hint ",i+1,":"]}),e.jsx("pre",{className:"ua-codefix-hint-code",children:t})]},t):e.jsxs("p",{children:[e.jsxs("b",{children:["Hint ",i+1,":"]})," ",t]},t))]}),k&&e.jsxs("div",{className:"ua-codefix-solution",children:[e.jsx("strong",{children:"Solution"}),e.jsx("pre",{children:s.solution}),e.jsx("button",{type:"button",onClick:o,children:"Apply solution to editor"})]})]})]}),e.jsxs("div",{className:"ua-codefix-footer",children:[e.jsxs("strong",{children:[a," / ",l.length]}),e.jsx("span",{children:"exercises passed"})]})]})}const W=[{id:"dot-product-basic",title:"Dot product",concept:"Multiply matching entries, then add the products.",objective:"Complete a dot product function.",difficulty:"warmup",starterCode:`function dot(a, b) {
  // TODO: replace 0 with the dot product.
  // Example: dot([1, 2], [3, 4]) = 1*3 + 2*4 = 11
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

check('dot([1, 2], [3, 4])', dot([1, 2], [3, 4]), 11);
check('dot([0, 5], [10, 2])', dot([0, 5], [10, 2]), 10);
check('dot([-1, 2], [3, 5])', dot([-1, 2], [3, 5]), 7);
check('dot([2, 2, 2], [1, 2, 3])', dot([2, 2, 2], [1, 2, 3]), 12);

return results;`,hints:["The dot product pairs entries by index: a[0] with b[0], a[1] with b[1], and so on.","For two vectors, use a loop and keep a running total.",`let total = 0;
for (let i = 0; i < a.length; i++) {
  total += a[i] * b[i];
}
return total;`],solution:`function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}`,explanation:"A dot product is a weighted sum: each entry of one vector weights the matching entry of the other vector."},{id:"matrix-cell",title:"One matrix multiplication cell",concept:"One output cell is one row-column dot product.",objective:"Complete the function that computes C[row][col].",difficulty:"core",starterCode:`function matrixCell(A, B, row, col) {
  // TODO: compute one cell of C = A * B.
  // Use row "row" from A and column "col" from B.
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

check('C[0][0]', matrixCell(A, B, 0, 0), 4);
check('C[0][1]', matrixCell(A, B, 0, 1), 9);
check('C[0][2]', matrixCell(A, B, 0, 2), 7);
check('C[1][0]', matrixCell(A, B, 1, 0), 7);
check('C[1][1]', matrixCell(A, B, 1, 1), 7);
check('C[1][2]', matrixCell(A, B, 1, 2), 11);

return results;`,hints:["The number of terms in the dot product is the number of columns in A, which is also the number of rows in B.","Use A[row][k] and B[k][col]. The index k moves across the row of A and down the column of B.",`let total = 0;
for (let k = 0; k < B.length; k++) {
  total += A[row][k] * B[k][col];
}
return total;`],solution:`function matrixCell(A, B, row, col) {
  let total = 0;
  for (let k = 0; k < B.length; k++) {
    total += A[row][k] * B[k][col];
  }
  return total;
}`,explanation:"Matrix multiplication is repeated dot products. Each output cell C[row][col] is row row of A dotted with column col of B."},{id:"matrix-multiply-full",title:"Full matrix multiplication",concept:"Fill every output cell by reusing the row-column rule.",objective:"Complete a full matrix multiplication function.",difficulty:"challenge",starterCode:`function matrixCell(A, B, row, col) {
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
      // TODO: push the correct cell value into row.
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

return results;`,hints:["You already have matrixCell(A, B, i, j). Use it inside the nested loops.","The outer loop chooses the output row i. The inner loop chooses the output column j.","row.push(matrixCell(A, B, i, j));"],solution:`function matrixCell(A, B, row, col) {
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
}`,explanation:"The full matrix product is just the matrixCell rule repeated for every row and every column."}];function q(){return e.jsx(Y,{exercises:W})}const G=[[1,2],[3,1]],Q=[[2,1,3],[1,4,2]],V=[[4,9,7],[7,7,11]],p=[{row:0,col:0,hint:"Multiply Row 1 of A with Column 1 of B: (1×2) + (2×1)",answer:4},{row:0,col:1,hint:"Multiply Row 1 of A with Column 2 of B: (1×1) + (2×4)",answer:9},{row:0,col:2,hint:"Multiply Row 1 of A with Column 3 of B: (1×3) + (2×2)",answer:7},{row:1,col:0,hint:"Multiply Row 2 of A with Column 1 of B: (3×2) + (1×1)",answer:7},{row:1,col:1,hint:"Multiply Row 2 of A with Column 2 of B: (3×1) + (1×4)",answer:7},{row:1,col:2,hint:"Multiply Row 2 of A with Column 3 of B: (3×3) + (1×2)",answer:11}];function ee(){const[l,m]=b.useState(0),[h,s]=b.useState(""),[f,r]=b.useState(""),[y,c]=b.useState(!1),[g,N]=b.useState(Array(6).fill(null)),[S,v]=b.useState(!1),[k,j]=b.useState(0),[O,n]=b.useState(0),d=p[l],w=()=>{const o=parseInt(h,10);if(n(a=>a+1),o===d.answer){r("✓ Correct!"),j(u=>u+1);const a=[...g];a[l]=o,N(a),setTimeout(()=>{l<p.length-1?(m(u=>u+1),s(""),r(""),c(!1)):(v(!0),r("🎉 Excellent! You completed all steps!"))},1e3)}else r("✗ Not quite. Try again or ask for a hint.")},E=()=>{c(!0)},A=()=>{m(0),s(""),r(""),c(!1),N(Array(6).fill(null)),v(!1),j(0),n(0)},R=o=>{o.key==="Enter"&&h.trim()!==""&&w()},T=o=>l<p.length&&p[l].row===o,I=o=>l<p.length&&p[l].col===o;return e.jsxs("div",{className:"flex flex-col items-center p-3 h-full",children:[e.jsx("h2",{className:"text-xl font-bold text-gray-800 mb-2",children:"Practice Exercise"}),e.jsxs("div",{className:"bg-white rounded-lg shadow-lg p-4 w-full",children:[e.jsxs("div",{className:"flex items-center justify-center gap-2 flex-wrap",children:[e.jsxs("div",{className:"flex flex-col items-center",children:[e.jsx("span",{className:"text-lg font-bold mb-1",children:"A"}),e.jsx("div",{className:"grid grid-cols-2 gap-1",children:G.map((o,a)=>o.map((u,t)=>e.jsx("div",{className:`w-10 h-10 flex items-center justify-center font-bold text-black rounded ${T(a)?"bg-blue-300 scale-110 ring-2 ring-blue-500":"bg-blue-400"} transition-all`,children:u},`a-${a}-${t}`)))})]}),e.jsx("span",{className:"text-2xl font-bold mx-2",children:"×"}),e.jsxs("div",{className:"flex flex-col items-center",children:[e.jsx("span",{className:"text-lg font-bold mb-1",children:"B"}),e.jsx("div",{className:"grid grid-cols-3 gap-1",children:Q.map((o,a)=>o.map((u,t)=>e.jsx("div",{className:`w-10 h-10 flex items-center justify-center font-bold text-black rounded ${I(t)?"bg-green-300 scale-110 ring-2 ring-green-500":"bg-green-400"} transition-all`,children:u},`b-${a}-${t}`)))})]}),e.jsx("span",{className:"text-2xl font-bold mx-2",children:"="}),e.jsxs("div",{className:"flex flex-col items-center",children:[e.jsx("span",{className:"text-lg font-bold mb-1",children:"C"}),e.jsx("div",{className:"grid grid-cols-3 gap-1",children:V.map((o,a)=>o.map((u,t)=>{const i=a*3+t,x=l===i,B=g[i]!==null;return e.jsx("div",{className:`w-10 h-10 flex items-center justify-center font-bold text-black rounded transition-all ${x?"bg-yellow-300 ring-2 ring-yellow-500 scale-110":B?"bg-orange-400":"bg-orange-200"}`,children:B?g[i]:"?"},`r-${a}-${t}`)}))})]})]}),e.jsxs("div",{className:"mt-4 text-center",children:[e.jsxs("p",{className:"text-gray-700 font-medium",children:["Step ",l+1," of ",p.length,": Calculate C[",d.row+1,"][",d.col+1,"]"]}),e.jsxs("p",{className:"text-sm text-gray-700 mt-1",children:["Row ",d.row+1," of A × Column ",d.col+1," of B"]})]})]}),S?e.jsx("div",{className:"mt-4 w-full max-w-sm text-center",children:e.jsxs("div",{className:"p-4 bg-green-100 rounded-lg border border-green-300",children:[e.jsx("p",{className:"text-green-700 font-bold text-lg",children:"🎉 Congratulations!"}),e.jsxs("p",{className:"text-green-600 mt-2",children:["Score: ",k," / ",p.length," correct"]}),e.jsxs("p",{className:"text-sm",children:["Total attempts: ",O]})]})}):e.jsxs("div",{className:"mt-4 w-full max-w-sm",children:[e.jsxs("div",{className:"flex gap-2",children:[e.jsx("input",{type:"number",value:h,onChange:o=>s(o.target.value),onKeyPress:R,placeholder:"Your answer...",className:"flex-1 px-4 py-2 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none text-center text-lg font-bold"}),e.jsx("button",{onClick:w,disabled:h.trim()==="",className:"px-4 py-2 bg-green-500 hover:bg-green-600 disabled:bg-gray-400 text-white font-bold rounded-lg transition-colors",children:"Submit"})]}),e.jsx("button",{onClick:E,className:"mt-2 w-full px-4 py-2 bg-yellow-500 hover:bg-yellow-600 text-white font-bold rounded-lg transition-colors",children:"💡 Show Hint"}),y&&e.jsx("div",{className:"mt-2 p-3 bg-yellow-100 rounded-lg border border-yellow-300",children:e.jsx("p",{className:"text-sm",children:d.hint})}),f&&e.jsx("div",{className:`mt-2 p-3 rounded-lg text-center font-bold ${f.includes("✓")?"bg-green-100 text-green-700":"bg-red-100 text-red-700"}`,children:f})]}),e.jsxs("div",{className:"mt-4 flex items-center gap-4",children:[e.jsxs("div",{className:"text-sm text-gray-800",children:["Progress: ",g.filter(o=>o!==null).length," / ",p.length]}),e.jsx("button",{onClick:A,className:"px-4 py-2 bg-red-500 hover:bg-red-600 text-white font-bold rounded-lg transition-colors text-sm",children:"↺ Reset"})]}),e.jsx("div",{className:"mt-8 w-full",children:e.jsx(q,{})})]})}export{ee as default};
