import{a as N,j as e,k as g}from"./react-vendor-Cdu38Wyn.js";import{F,I as J,aZ as z,aQ as D,ar as _,$ as U,_ as K}from"./icons-C7miCxLM.js";let W=1;function Y({userCode:l,testCode:d,timeoutMs:a=1200}){const s=W++;return new Promise(n=>{const r=new Worker(new URL("/ml-animations/assets/jsEvalWorker-BiN8n73y.js",import.meta.url),{type:"module"}),j=window.setTimeout(()=>{r.terminate(),n({ok:!1,results:[],error:"Execution timed out. Check for an infinite loop."})},a);r.onmessage=h=>{h.data.id===s&&(window.clearTimeout(j),r.terminate(),n({ok:h.data.ok,results:h.data.results||[],error:h.data.error}))},r.postMessage({id:s,userCode:l,testCode:d})})}function M(l,d){return d?"error":l!=null&&l.length?l.every(a=>a.passed)?"passed":"failed":"idle"}function P(l){return typeof l=="string"?l:JSON.stringify(l)}const Z=/(\/\/.*|\/\*[\s\S]*?\*\/|(["'`])(?:\\.|(?!\2)[^\\])*\2|\b(?:const|let|var|function|return|if|else|for|while|new|throw|true|false|null|undefined)\b|\b\d+(?:\.\d+)?\b|\b[a-zA-Z_$][\w$]*(?=\s*\())/g;function q(l){const d=[];let a=0;for(const s of l.matchAll(Z)){s.index>a&&d.push(l.slice(a,s.index));const n=s[0];let r="plain";n.startsWith("//")||n.startsWith("/*")?r="comment":/^["'`]/.test(n)?r="string":/^\d/.test(n)?r="number":/^(const|let|var|function|return|if|else|for|while|new|throw|true|false|null|undefined)$/.test(n)?r="keyword":r="call",d.push(e.jsx("span",{className:`ua-code-token-${r}`,children:n},`${s.index}-${n}`)),a=s.index+n.length}return a<l.length&&d.push(l.slice(a)),d}function G({exercises:l}){var B;const[d,a]=N.useState(0),s=l[d],n=N.useRef(null),[r,j]=N.useState(()=>Object.fromEntries(l.map(t=>[t.id,t.starterCode]))),[h,w]=N.useState({}),[k,E]=N.useState({}),[S,O]=N.useState(!1),[v,b]=N.useState(!1),A=r[s.id],o=k[s.id],C=M(o==null?void 0:o.results,o==null?void 0:o.error),y=h[s.id]||0,R=s.hints.slice(0,y),T=!!(o||y>0);async function $(){O(!0),b(!1);const t=await Y({userCode:A,testCode:s.testCode});E(c=>({...c,[s.id]:t})),O(!1)}function I(){j(t=>({...t,[s.id]:s.starterCode})),E(t=>({...t,[s.id]:null})),w(t=>({...t,[s.id]:0})),b(!1)}function i(){w(t=>({...t,[s.id]:Math.min(s.hints.length,y+1)}))}function u(){j(t=>({...t,[s.id]:s.solution})),b(!1)}function x(t){n.current&&(n.current.scrollTop=t.currentTarget.scrollTop,n.current.scrollLeft=t.currentTarget.scrollLeft)}const m=Object.values(k).filter(t=>{var c;return((c=t==null?void 0:t.results)==null?void 0:c.length)&&t.results.every(p=>p.passed)}).length;return e.jsxs("section",{className:"ua-codefix-lab",children:[e.jsxs("div",{className:"ua-codefix-head",children:[e.jsx("span",{children:"Code Completion-style lab"}),e.jsx("h2",{children:"Fix the TODOs, run the tests"}),e.jsx("p",{children:"Each exercise is almost complete. Change the smallest piece of code needed to make the tests pass."})]}),e.jsx("div",{className:"ua-codefix-progress",children:l.map((t,c)=>{const p=k[t.id],H=M(p==null?void 0:p.results,p==null?void 0:p.error),L=H==="passed"?F:J;return e.jsxs("button",{type:"button",onClick:()=>{a(c),b(!1)},className:`ua-codefix-step ${c===d?"active":""} ${H}`,children:[e.jsx(L,{size:15,"aria-hidden":"true"}),e.jsxs("span",{children:[c+1,". ",t.title]})]},t.id)})}),e.jsxs("div",{className:"ua-codefix-grid",children:[e.jsxs("article",{className:"ua-codefix-card ua-codefix-instructions",children:[e.jsx("span",{children:s.difficulty}),e.jsx("h3",{children:s.title}),e.jsx("p",{children:s.objective}),e.jsxs("div",{className:"ua-codefix-concept",children:[e.jsx("strong",{children:"Concept"}),e.jsx("p",{children:s.concept})]}),e.jsxs("div",{className:"ua-codefix-explanation",children:[e.jsx("strong",{children:"After you pass"}),e.jsx("p",{children:s.explanation})]})]}),e.jsxs("article",{className:"ua-codefix-card ua-codefix-editor-card",children:[e.jsxs("div",{className:"ua-codefix-card-head",children:[e.jsxs("div",{children:[e.jsx("span",{children:"Editor"}),e.jsx("h3",{children:"Complete the TODO"})]}),e.jsxs("button",{type:"button",onClick:I,children:[e.jsx(z,{size:14,"aria-hidden":"true"}),"Reset"]})]}),e.jsxs("div",{className:"ua-codefix-editor-shell",children:[e.jsx("pre",{className:"ua-codefix-highlight","aria-hidden":"true",ref:n,children:q(A)}),e.jsx("textarea",{className:"ua-codefix-editor",value:A,spellCheck:!1,"aria-label":`${s.title} code editor`,onScroll:x,onChange:t=>j(c=>({...c,[s.id]:t.target.value}))})]}),e.jsxs("div",{className:"ua-codefix-actions",children:[e.jsxs("button",{type:"button",onClick:$,disabled:S,children:[e.jsx(D,{size:15,"aria-hidden":"true"}),S?"Running...":"Run tests"]}),e.jsxs("button",{type:"button",onClick:i,disabled:y>=s.hints.length,children:[e.jsx(_,{size:15,"aria-hidden":"true"}),y===0?"Show hint":"Next hint"]}),e.jsxs("button",{type:"button",onClick:()=>b(t=>!t),disabled:!T,title:T?void 0:"Run tests or use a hint before revealing the solution.",children:[v?e.jsx(U,{size:15,"aria-hidden":"true"}):e.jsx(K,{size:15,"aria-hidden":"true"}),v?"Hide solution":T?"See solution":"Try first"]})]})]}),e.jsxs("article",{className:"ua-codefix-card ua-codefix-feedback",children:[e.jsx("span",{children:"Checks"}),e.jsxs("h3",{children:[C==="passed"&&"All tests passed",C==="failed"&&"Keep going",C==="error"&&"Code error",C==="idle"&&"Run tests to begin"]}),(o==null?void 0:o.error)&&e.jsx("pre",{className:"ua-codefix-error",children:o.error}),((B=o==null?void 0:o.results)==null?void 0:B.length)>0?e.jsx("ul",{className:"ua-codefix-checks",children:o.results.map(t=>e.jsxs("li",{className:t.passed?"passed":"failed",children:[e.jsxs("strong",{children:[t.passed?"Pass":"Fail",": ",t.name]}),!t.passed&&e.jsxs("small",{children:["Expected ",P(t.expected),", got ",P(t.actual)]})]},t.name))}):e.jsx("p",{className:"ua-codefix-empty",children:"Run the tests. If one fails, use the smallest hint that helps."}),R.length>0&&e.jsxs("div",{className:"ua-codefix-hints",children:[e.jsx("strong",{children:"Hints"}),R.map((t,c)=>t.includes(`
`)?e.jsxs("div",{className:"ua-codefix-hint",children:[e.jsxs("b",{children:["Hint ",c+1,":"]}),e.jsx("pre",{className:"ua-codefix-hint-code",children:t})]},t):e.jsxs("p",{children:[e.jsxs("b",{children:["Hint ",c+1,":"]})," ",t]},t))]}),v&&e.jsxs("div",{className:"ua-codefix-solution",children:[e.jsx("strong",{children:"Solution"}),e.jsx("pre",{children:s.solution}),e.jsx("button",{type:"button",onClick:u,children:"Apply solution to editor"})]})]})]}),e.jsxs("div",{className:"ua-codefix-footer",children:[e.jsxs("strong",{children:[m," / ",l.length]}),e.jsx("span",{children:"exercises passed"})]})]})}const Q=[{id:"dot-product-basic",title:"Dot product",concept:"Multiply matching entries, then add the products.",objective:"Complete a dot product function.",difficulty:"warmup",starterCode:`function dot(a, b) {
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
}`,explanation:"The full matrix product is just the matrixCell rule repeated for every row and every column."}];function V(){return e.jsx(G,{exercises:Q})}const X=[[1,2],[3,1]],ee=[[2,1,3],[1,4,2]],te=[[4,9,7],[7,7,11]],f=[{row:0,col:0,hint:"Multiply Row 1 of A with Column 1 of B: (1×2) + (2×1)",answer:4},{row:0,col:1,hint:"Multiply Row 1 of A with Column 2 of B: (1×1) + (2×4)",answer:9},{row:0,col:2,hint:"Multiply Row 1 of A with Column 3 of B: (1×3) + (2×2)",answer:7},{row:1,col:0,hint:"Multiply Row 2 of A with Column 1 of B: (3×2) + (1×1)",answer:7},{row:1,col:1,hint:"Multiply Row 2 of A with Column 2 of B: (3×1) + (1×4)",answer:7},{row:1,col:2,hint:"Multiply Row 2 of A with Column 3 of B: (3×3) + (1×2)",answer:11}];function oe(){const[l,d]=g.useState(0),[a,s]=g.useState(""),[n,r]=g.useState(""),[j,h]=g.useState(!1),[w,k]=g.useState(Array(6).fill(null)),[E,S]=g.useState(!1),[O,v]=g.useState(0),[b,A]=g.useState(0),o=f[l],C=()=>{const i=parseInt(a,10);if(A(u=>u+1),i===o.answer){r("✓ Correct!"),v(x=>x+1);const u=[...w];u[l]=i,k(u),setTimeout(()=>{l<f.length-1?(d(x=>x+1),s(""),r(""),h(!1)):(S(!0),r("🎉 Excellent! You completed all steps!"))},1e3)}else r("✗ Not quite. Try again or ask for a hint.")},y=()=>{h(!0)},R=()=>{d(0),s(""),r(""),h(!1),k(Array(6).fill(null)),S(!1),v(0),A(0)},T=i=>{i.key==="Enter"&&a.trim()!==""&&C()},$=i=>l<f.length&&f[l].row===i,I=i=>l<f.length&&f[l].col===i;return e.jsxs("div",{className:"flex flex-col items-center p-3 h-full",children:[e.jsx("h2",{className:"text-xl font-bold text-gray-800 mb-2",children:"Practice Exercise"}),e.jsxs("div",{className:"bg-white rounded-lg shadow-lg p-4 w-full",children:[e.jsxs("div",{className:"flex items-center justify-center gap-2 flex-wrap",children:[e.jsxs("div",{className:"flex flex-col items-center",children:[e.jsx("span",{className:"text-lg font-bold mb-1",children:"A"}),e.jsx("div",{className:"grid grid-cols-2 gap-1",children:X.map((i,u)=>i.map((x,m)=>e.jsx("div",{className:`w-10 h-10 flex items-center justify-center font-bold text-black rounded ${$(u)?"bg-blue-300 scale-110 ring-2 ring-blue-500":"bg-blue-400"} transition-all`,children:x},`a-${u}-${m}`)))})]}),e.jsx("span",{className:"text-2xl font-bold mx-2",children:"×"}),e.jsxs("div",{className:"flex flex-col items-center",children:[e.jsx("span",{className:"text-lg font-bold mb-1",children:"B"}),e.jsx("div",{className:"grid grid-cols-3 gap-1",children:ee.map((i,u)=>i.map((x,m)=>e.jsx("div",{className:`w-10 h-10 flex items-center justify-center font-bold text-black rounded ${I(m)?"bg-green-300 scale-110 ring-2 ring-green-500":"bg-green-400"} transition-all`,children:x},`b-${u}-${m}`)))})]}),e.jsx("span",{className:"text-2xl font-bold mx-2",children:"="}),e.jsxs("div",{className:"flex flex-col items-center",children:[e.jsx("span",{className:"text-lg font-bold mb-1",children:"C"}),e.jsx("div",{className:"grid grid-cols-3 gap-1",children:te.map((i,u)=>i.map((x,m)=>{const B=u*3+m,t=l===B,c=w[B]!==null;return e.jsx("div",{className:`w-10 h-10 flex items-center justify-center font-bold text-black rounded transition-all ${t?"bg-yellow-300 ring-2 ring-yellow-500 scale-110":c?"bg-orange-400":"bg-orange-200"}`,children:c?w[B]:"?"},`r-${u}-${m}`)}))})]})]}),e.jsxs("div",{className:"mt-4 text-center",children:[e.jsxs("p",{className:"text-gray-700 font-medium",children:["Step ",l+1," of ",f.length,": Calculate C[",o.row+1,"][",o.col+1,"]"]}),e.jsxs("p",{className:"text-sm text-gray-700 mt-1",children:["Row ",o.row+1," of A × Column ",o.col+1," of B"]})]})]}),E?e.jsx("div",{className:"mt-4 w-full max-w-sm text-center",children:e.jsxs("div",{className:"p-4 bg-green-100 rounded-lg border border-green-300",children:[e.jsx("p",{className:"text-green-700 font-bold text-lg",children:"🎉 Congratulations!"}),e.jsxs("p",{className:"text-green-600 mt-2",children:["Score: ",O," / ",f.length," correct"]}),e.jsxs("p",{className:"text-sm",children:["Total attempts: ",b]})]})}):e.jsxs("div",{className:"mt-4 w-full max-w-sm",children:[e.jsxs("div",{className:"flex gap-2",children:[e.jsx("input",{type:"number",value:a,onChange:i=>s(i.target.value),onKeyPress:T,placeholder:"Your answer...",className:"flex-1 px-4 py-2 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none text-center text-lg font-bold"}),e.jsx("button",{onClick:C,disabled:a.trim()==="",className:"px-4 py-2 bg-green-500 hover:bg-green-600 disabled:bg-gray-400 text-white font-bold rounded-lg transition-colors",children:"Submit"})]}),e.jsx("button",{onClick:y,className:"mt-2 w-full px-4 py-2 bg-yellow-500 hover:bg-yellow-600 text-white font-bold rounded-lg transition-colors",children:"💡 Show Hint"}),j&&e.jsx("div",{className:"mt-2 p-3 bg-yellow-100 rounded-lg border border-yellow-300",children:e.jsx("p",{className:"text-sm",children:o.hint})}),n&&e.jsx("div",{className:`mt-2 p-3 rounded-lg text-center font-bold ${n.includes("✓")?"bg-green-100 text-green-700":"bg-red-100 text-red-700"}`,children:n})]}),e.jsxs("div",{className:"mt-4 flex items-center gap-4",children:[e.jsxs("div",{className:"text-sm text-gray-800",children:["Progress: ",w.filter(i=>i!==null).length," / ",f.length]}),e.jsx("button",{onClick:R,className:"px-4 py-2 bg-red-500 hover:bg-red-600 text-white font-bold rounded-lg transition-colors text-sm",children:"↺ Reset"})]}),e.jsx("div",{className:"mt-8 w-full",children:e.jsx(V,{})})]})}export{oe as default};
