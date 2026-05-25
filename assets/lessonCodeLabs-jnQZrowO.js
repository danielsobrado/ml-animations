import{a as g,j as t}from"./react-vendor-Cdu38Wyn.js";import{F as G,I as K,aZ as Y,aQ as Z,ar as V,$ as Q,_ as X}from"./icons-C7miCxLM.js";import{a as B}from"./assessment-data-KPydSR8a.js";let ee=1;function te({userCode:e,testCode:i,timeoutMs:o=1200}){const s=ee++;return new Promise(r=>{const a=new Worker(new URL("/ml-animations/assets/jsEvalWorker-BiN8n73y.js",import.meta.url),{type:"module"}),p=window.setTimeout(()=>{a.terminate(),r({ok:!1,results:[],error:"Execution timed out. Check for an infinite loop."})},o);a.onmessage=h=>{h.data.id===s&&(window.clearTimeout(p),a.terminate(),r({ok:h.data.ok,results:h.data.results||[],error:h.data.error}))},a.postMessage({id:s,userCode:e,testCode:i})})}function _(e,i){return i?"error":e!=null&&e.length?e.every(o=>o.passed)?"passed":"failed":"idle"}function A(e){return typeof e=="string"?e:JSON.stringify(e)}const se=/(\/\/.*|\/\*[\s\S]*?\*\/|(["'`])(?:\\.|(?!\2)[^\\])*\2|\b(?:const|let|var|function|return|if|else|for|while|new|throw|true|false|null|undefined)\b|\b\d+(?:\.\d+)?\b|\b[a-zA-Z_$][\w$]*(?=\s*\())/g;function ne(e){const i=[];let o=0;for(const s of e.matchAll(se)){s.index>o&&i.push(e.slice(o,s.index));const r=s[0];let a="plain";r.startsWith("//")||r.startsWith("/*")?a="comment":/^["'`]/.test(r)?a="string":/^\d/.test(r)?a="number":/^(const|let|var|function|return|if|else|for|while|new|throw|true|false|null|undefined)$/.test(r)?a="keyword":a="call",i.push(t.jsx("span",{className:`ua-code-token-${a}`,children:r},`${s.index}-${r}`)),o=s.index+r.length}return o<e.length&&i.push(e.slice(o)),i}function Ae({exercises:e}){var C;const[i,o]=g.useState(0),s=e[i],r=g.useRef(null),[a,p]=g.useState(()=>Object.fromEntries(e.map(n=>[n.id,n.starterCode]))),[h,S]=g.useState({}),[y,$]=g.useState({}),[E,O]=g.useState(!1),[j,x]=g.useState(!1),w=a[s.id],c=y[s.id],N=_(c==null?void 0:c.results,c==null?void 0:c.error),b=h[s.id]||0,v=s.hints.slice(0,b),L=!!(c||b>0),M=g.useMemo(()=>{const n=[];return e.forEach((l,m)=>{const f=l.group||"Exercises",d=n[n.length-1];if((d==null?void 0:d.name)===f){d.items.push({exercise:l,index:m});return}n.push({name:f,items:[{exercise:l,index:m}]})}),n},[e]);async function q(){O(!0),x(!1);const n=await te({userCode:w,testCode:s.testCode});$(l=>({...l,[s.id]:n})),O(!1)}function D(){p(n=>({...n,[s.id]:s.starterCode})),$(n=>({...n,[s.id]:null})),S(n=>({...n,[s.id]:0})),x(!1)}function F(){S(n=>({...n,[s.id]:Math.min(s.hints.length,b+1)}))}function H(){p(n=>({...n,[s.id]:s.solution})),x(!1)}function P(n){r.current&&(r.current.scrollTop=n.currentTarget.scrollTop,r.current.scrollLeft=n.currentTarget.scrollLeft)}const U=Object.values(y).filter(n=>{var l;return((l=n==null?void 0:n.results)==null?void 0:l.length)&&n.results.every(m=>m.passed)}).length;return t.jsxs("section",{className:"ua-codefix-lab",children:[t.jsxs("div",{className:"ua-codefix-head",children:[t.jsx("span",{children:"Code Completion-style lab"}),t.jsx("h2",{children:"Fix the TODOs, run the tests"}),t.jsx("p",{children:"Each exercise is almost complete. Change the smallest piece of code needed to make the tests pass."})]}),t.jsx("div",{className:"ua-codefix-progress",children:M.map(n=>{const l=n.items.filter(({exercise:m})=>{var d;const f=y[m.id];return((d=f==null?void 0:f.results)==null?void 0:d.length)&&f.results.every(k=>k.passed)}).length;return t.jsxs("div",{className:"ua-codefix-progress-group",children:[t.jsxs("div",{className:"ua-codefix-progress-label",children:[t.jsx("strong",{children:n.name}),t.jsxs("span",{children:[l,"/",n.items.length]})]}),t.jsx("div",{className:"ua-codefix-progress-steps",children:n.items.map(({exercise:m,index:f})=>{const d=y[m.id],k=_(d==null?void 0:d.results,d==null?void 0:d.error),W=k==="passed"?G:K;return t.jsxs("button",{type:"button",onClick:()=>{o(f),x(!1)},className:`ua-codefix-step ${f===i?"active":""} ${k}`,children:[t.jsx(W,{size:15,"aria-hidden":"true"}),t.jsxs("span",{children:[m.stepLabel||`${f+1}.`," ",m.title]})]},m.id)})})]},n.name)})}),t.jsxs("div",{className:"ua-codefix-grid",children:[t.jsxs("article",{className:"ua-codefix-card ua-codefix-instructions",children:[t.jsx("span",{children:s.difficulty}),t.jsx("h3",{children:s.title}),t.jsx("p",{children:s.objective}),t.jsxs("div",{className:"ua-codefix-concept",children:[t.jsx("strong",{children:"Concept"}),t.jsx("p",{children:s.concept})]}),t.jsxs("div",{className:"ua-codefix-explanation",children:[t.jsx("strong",{children:"After you pass"}),t.jsx("p",{children:s.explanation})]})]}),t.jsxs("article",{className:"ua-codefix-card ua-codefix-editor-card",children:[t.jsxs("div",{className:"ua-codefix-card-head",children:[t.jsxs("div",{children:[t.jsx("span",{children:"Editor"}),t.jsx("h3",{children:"Complete the TODO"})]}),t.jsxs("button",{type:"button",onClick:D,children:[t.jsx(Y,{size:14,"aria-hidden":"true"}),"Reset"]})]}),t.jsxs("div",{className:"ua-codefix-editor-shell",children:[t.jsx("pre",{className:"ua-codefix-highlight","aria-hidden":"true",ref:r,children:ne(w)}),t.jsx("textarea",{className:"ua-codefix-editor",value:w,spellCheck:!1,"aria-label":`${s.title} code editor`,onScroll:P,onChange:n=>p(l=>({...l,[s.id]:n.target.value}))})]}),t.jsxs("div",{className:"ua-codefix-actions",children:[t.jsxs("button",{type:"button",onClick:q,disabled:E,children:[t.jsx(Z,{size:15,"aria-hidden":"true"}),E?"Running...":"Run tests"]}),t.jsxs("button",{type:"button",onClick:F,disabled:b>=s.hints.length,children:[t.jsx(V,{size:15,"aria-hidden":"true"}),b===0?"Show hint":"Next hint"]}),t.jsxs("button",{type:"button",onClick:()=>x(n=>!n),disabled:!L,title:L?void 0:"Run tests or use a hint before revealing the solution.",children:[j?t.jsx(Q,{size:15,"aria-hidden":"true"}):t.jsx(X,{size:15,"aria-hidden":"true"}),j?"Hide solution":L?"See solution":"Try first"]})]})]}),t.jsxs("article",{className:"ua-codefix-card ua-codefix-feedback",children:[t.jsx("span",{children:"Checks"}),t.jsxs("h3",{children:[N==="passed"&&"All tests passed",N==="failed"&&"Keep going",N==="error"&&"Code error",N==="idle"&&"Run tests to begin"]}),(c==null?void 0:c.error)&&t.jsx("pre",{className:"ua-codefix-error",children:c.error}),((C=c==null?void 0:c.results)==null?void 0:C.length)>0?t.jsx("ul",{className:"ua-codefix-checks",children:c.results.map(n=>t.jsxs("li",{className:n.passed?"passed":"failed",children:[t.jsxs("strong",{children:[n.passed?"Pass":"Fail",": ",n.name]}),!n.passed&&t.jsxs("small",{children:["Expected ",A(n.expected),", got ",A(n.actual)]})]},n.name))}):t.jsx("p",{className:"ua-codefix-empty",children:"Run the tests. If one fails, use the smallest hint that helps."}),v.length>0&&t.jsxs("div",{className:"ua-codefix-hints",children:[t.jsx("strong",{children:"Hints"}),v.map((n,l)=>n.includes(`
`)?t.jsxs("div",{className:"ua-codefix-hint",children:[t.jsxs("b",{children:["Hint ",l+1,":"]}),t.jsx("pre",{className:"ua-codefix-hint-code",children:n})]},n):t.jsxs("p",{children:[t.jsxs("b",{children:["Hint ",l+1,":"]})," ",n]},n))]}),j&&t.jsxs("div",{className:"ua-codefix-solution",children:[t.jsx("strong",{children:"Solution"}),t.jsx("pre",{children:s.solution}),t.jsx("button",{type:"button",onClick:H,children:"Apply solution to editor"})]})]})]}),t.jsxs("div",{className:"ua-codefix-footer",children:[t.jsxs("strong",{children:[U," / ",e.length]}),t.jsx("span",{children:"exercises passed"})]})]})}const ie=77,ae=new Set(["and","for","the","as","of","to","in","vs","with","from","into","overview","track","comprehensive"]),re=new Map(B.map((e,i)=>[e.id,i]));function z(e){return[...new Set(e)]}function T(e){return z(String(e).toLowerCase().split(/[^a-z0-9]+/).filter(i=>i.length>1&&!ae.has(i)))}function oe(e){return String(e).split(/[^a-zA-Z0-9]+/).filter(Boolean).map(s=>`${s[0].toUpperCase()}${s.slice(1)}`).join("")||"Lesson"}function R(e,i){const o=e.toLowerCase();return i.filter(s=>o.includes(s)).length}function ce(e){const i=z([...T(e.id),...T(e.name)]);return i.length>0?i.slice(0,3):["lesson"]}function I(e){return`/animation/${e.id}`}function le({lesson:e,stepLabel:i,suffix:o,keyword:s,domain:r}){const a=`has${o}Keyword`;return{id:`${e.id}-keyword-check`,group:e.name,stepLabel:i,title:"Recognize the lesson keyword",concept:`${e.name} can be indexed by a stable keyword before deeper ${r.kind} logic runs.`,objective:"Return true when text contains the lesson keyword, case-insensitively.",difficulty:"warmup",starterCode:`function ${a}(text) {
  const keyword = ${JSON.stringify(s)};

  // TODO: return whether text contains keyword, ignoring case.
  return false;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('lesson reference matches', ${a}(${JSON.stringify(`${e.name} ${I(e)}`)}), true);
check('lesson route matches', ${a}(${JSON.stringify(I(e))}), true);
check('unrelated text misses', ${a}('zzzz yyyy xxxx'), false);

return results;`,hints:["Convert the incoming text to lowercase before checking.","Use text.toLowerCase().includes(keyword).","return text.toLowerCase().includes(keyword);"],solution:`function ${a}(text) {
  const keyword = ${JSON.stringify(s)};
  return text.toLowerCase().includes(keyword);
}`,explanation:`Stable keywords help route learners and examples to the right ${e.name} code path.`}}function de({lesson:e,stepLabel:i,suffix:o,terms:s,domain:r}){const a=`count${o}FocusTerms`,p=R(e.name,s),h=`${e.name} ${e.description}`,S=R(h,s);return{id:`${e.id}-focus-term-count`,group:e.name,stepLabel:i,title:"Count focus terms",concept:`${r.kind} systems often reduce text into small signals before ranking or checking.`,objective:"Count how many lesson focus terms appear in the text.",difficulty:"core",starterCode:`function ${a}(text) {
  const terms = ${JSON.stringify(s)};
  const lower = text.toLowerCase();
  let count = 0;

  for (let i = 0; i < terms.length; i++) {
    // TODO: increment count when lower contains terms[i].
  }

  return count;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('title terms', ${a}(${JSON.stringify(e.name)}), ${p});
check('description terms', ${a}(${JSON.stringify(h)}), ${S});
check('no matching terms', ${a}('zzzz yyyy xxxx'), 0);

return results;`,hints:["lower and terms are already prepared.","Use lower.includes(terms[i]) inside the loop.","if (lower.includes(terms[i])) count += 1;"],solution:`function ${a}(text) {
  const terms = ${JSON.stringify(s)};
  const lower = text.toLowerCase();
  let count = 0;

  for (let i = 0; i < terms.length; i++) {
    if (lower.includes(terms[i])) count += 1;
  }

  return count;
}`,explanation:`This mirrors the small feature checks behind search, routing, and lesson-specific ${r.signalName} logic.`}}function ue({lesson:e,stepLabel:i,suffix:o,domain:s}){const r=`best${o}Candidate`;return{id:`${e.id}-best-candidate`,group:e.name,stepLabel:i,title:"Select the best candidate",concept:`${s.kind} workflows often rank candidates by a score before choosing the next action.`,objective:"Return the id of the candidate with the highest score.",difficulty:"core",starterCode:`function ${r}(candidates) {
  let best = candidates[0];

  for (let i = 1; i < candidates.length; i++) {
    // TODO: update best when candidates[i] has a higher score.
  }

  return best.id;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('lesson candidate wins', ${r}([
  { id: 'baseline', score: 0.2 },
  { id: ${JSON.stringify(e.id)}, score: 0.9 },
  { id: 'distractor', score: 0.4 },
]), ${JSON.stringify(e.id)});

check('last candidate wins', ${r}([
  { id: 'first', score: 0.1 },
  { id: 'second', score: 0.3 },
  { id: 'third', score: 0.8 },
]), 'third');

return results;`,hints:["Compare candidates[i].score with best.score.","If the current score is larger, replace best.","if (candidates[i].score > best.score) best = candidates[i];"],solution:`function ${r}(candidates) {
  let best = candidates[0];

  for (let i = 1; i < candidates.length; i++) {
    if (candidates[i].score > best.score) best = candidates[i];
  }

  return best.id;
}`,explanation:`Ranking by ${s.signalName} is a reusable pattern across ${e.categoryName} lessons.`}}function he({lesson:e,stepLabel:i,suffix:o,domain:s}){const r=`has${o}PipelineStages`,a=s.stages,p=a.slice(0,-1),h=[...a].reverse();return{id:`${e.id}-pipeline-stage-check`,group:e.name,stepLabel:i,title:"Check required stages",concept:`${e.name} is easier to debug when the expected ${s.kind} stages are explicit.`,objective:"Return false when any required stage is missing.",difficulty:"challenge",starterCode:`function ${r}(stages) {
  const requiredStages = ${JSON.stringify(a)};

  for (let i = 0; i < requiredStages.length; i++) {
    // TODO: return false if stages does not include requiredStages[i].
  }

  return true;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('all stages present', ${r}(${JSON.stringify(a)}), true);
check('order does not matter', ${r}(${JSON.stringify(h)}), true);
check('missing one stage', ${r}(${JSON.stringify(p)}), false);

return results;`,hints:["Use stages.includes(requiredStages[i]).","Return false as soon as one required stage is absent.","if (!stages.includes(requiredStages[i])) return false;"],solution:`function ${r}(stages) {
  const requiredStages = ${JSON.stringify(a)};

  for (let i = 0; i < requiredStages.length; i++) {
    if (!stages.includes(requiredStages[i])) return false;
  }

  return true;
}`,explanation:`${s.stageExplanation} This check makes that dependency visible in code.`}}function me(e,i){const o=re.get(e.id),s=ie+o,r=ce(e),a=oe(e.id);return{lessonId:e.id,lessonName:e.name,categoryId:e.categoryId,categoryName:e.categoryName,groupNumber:s,exercises:[le({lesson:e,stepLabel:`${s}.1`,suffix:a,keyword:r[0],domain:i}),de({lesson:e,stepLabel:`${s}.2`,suffix:a,terms:r,domain:i}),ue({lesson:e,stepLabel:`${s}.3`,suffix:a,domain:i}),he({lesson:e,stepLabel:`${s}.4`,suffix:a,domain:i})]}}function u(e,i){return B.filter(o=>o.categoryId===e).map(o=>me(o,i))}const fe=u("nlp",{kind:"text representation",signalName:"text relevance",stages:["tokenize","vectorize","compare"],stageExplanation:"NLP code usually has to tokenize text before it can build or compare representations."}),pe=u("transformers",{kind:"sequence-modeling",signalName:"attention or routing score",stages:["project","score","mix"],stageExplanation:"Transformer internals depend on explicit projection, scoring, and mixing stages."}),ge=u("frontier-llms",{kind:"frontier-model evaluation",signalName:"capability or risk score",stages:["measure","compare","gate"],stageExplanation:"Frontier-model systems need measurement, comparison, and release gates before deployment decisions."}),xe=u("neural-networks",{kind:"neural-network computation",signalName:"activation or gradient signal",stages:["forward","loss","update"],stageExplanation:"Neural-network code is built around forward computation, loss measurement, and parameter updates."}),be=u("advanced-models",{kind:"advanced-model pipeline",signalName:"retrieval or multimodal score",stages:["encode","retrieve","ground"],stageExplanation:"Advanced model systems often encode inputs, retrieve or combine evidence, and check grounding."}),Se=u("math-fundamentals",{kind:"mathematical computation",signalName:"numeric fit or stability score",stages:["represent","compute","check"],stageExplanation:"Math code is clearer when representation, computation, and result checks are separate."}),ye=u("core-ml",{kind:"machine-learning workflow",signalName:"validation metric",stages:["split","train","evaluate"],stageExplanation:"Core ML workflows depend on clean data splits, training logic, and honest evaluation."}),Ne=u("model-reliability",{kind:"reliability check",signalName:"monitoring or risk score",stages:["observe","alert","triage"],stageExplanation:"Reliability systems observe behavior, alert on meaningful shifts, and triage failures."}),ke=u("experimentation-causal-ml",{kind:"experiment analysis",signalName:"effect or balance score",stages:["assign","measure","compare"],stageExplanation:"Experiment code must separate assignment, measurement, and comparison to support causal claims."}),je=u("probability-stats",{kind:"probability/statistics calculation",signalName:"probability or uncertainty score",stages:["count","normalize","summarize"],stageExplanation:"Probability code often counts outcomes, normalizes them, then summarizes uncertainty."}),we=u("reinforcement-learning",{kind:"reinforcement-learning loop",signalName:"return or action-value score",stages:["observe","act","learn"],stageExplanation:"RL loops observe state, choose actions, and update behavior from feedback."}),Le=u("algorithms",{kind:"algorithmic data structure",signalName:"rank or membership score",stages:["insert","query","verify"],stageExplanation:"Algorithmic structures are useful when updates, queries, and checks are kept explicit."}),$e=u("diffusion-models",{kind:"diffusion-model pipeline",signalName:"noise or denoising score",stages:["noise","condition","denoise"],stageExplanation:"Diffusion systems manage noise, conditioning, and denoising as separate implementation stages."}),J=[...fe,...pe,...ge,...xe,...be,...Se,...ye,...Ne,...ke,...je,...we,...Le,...$e],Ee=Object.fromEntries(J.map(e=>[e.lessonId,e]));J.flatMap(e=>e.exercises);function Oe(e){return Ee[e]||null}function Te(e){var i;return((i=Oe(e))==null?void 0:i.exercises)||[]}export{Ae as C,J as L,Te as g};
