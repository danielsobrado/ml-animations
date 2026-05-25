import{a as m}from"./assessment-data-BcR-seWL.js";const h=77,S=new Set(["and","for","the","as","of","to","in","vs","with","from","into","overview","track","comprehensive"]),x=new Map(m.map((e,t)=>[e.id,t]));function g(e){return[...new Set(e)]}function d(e){return g(String(e).toLowerCase().split(/[^a-z0-9]+/).filter(t=>t.length>1&&!S.has(t)))}function k(e){return String(e).split(/[^a-zA-Z0-9]+/).filter(Boolean).map(n=>`${n[0].toUpperCase()}${n.slice(1)}`).join("")||"Lesson"}function l(e,t){const a=e.toLowerCase();return t.filter(n=>a.includes(n)).length}function b(e){const t=g([...d(e.id),...d(e.name)]);return t.length>0?t.slice(0,3):["lesson"]}function u(e){return`/animation/${e.id}`}function y({lesson:e,stepLabel:t,suffix:a,keyword:n,domain:i}){const s=`has${a}Keyword`;return{id:`${e.id}-keyword-check`,group:e.name,stepLabel:t,title:"Recognize the lesson keyword",concept:`${e.name} can be indexed by a stable keyword before deeper ${i.kind} logic runs.`,objective:"Return true when text contains the lesson keyword, case-insensitively.",difficulty:"warmup",starterCode:`function ${s}(text) {
  const keyword = ${JSON.stringify(n)};

  // TODO: return whether text contains keyword, ignoring case.
  return false;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('lesson reference matches', ${s}(${JSON.stringify(`${e.name} ${u(e)}`)}), true);
check('lesson route matches', ${s}(${JSON.stringify(u(e))}), true);
check('unrelated text misses', ${s}('zzzz yyyy xxxx'), false);

return results;`,hints:["Convert the incoming text to lowercase before checking.","Use text.toLowerCase().includes(keyword).","return text.toLowerCase().includes(keyword);"],solution:`function ${s}(text) {
  const keyword = ${JSON.stringify(n)};
  return text.toLowerCase().includes(keyword);
}`,explanation:`Stable keywords help route learners and examples to the right ${e.name} code path.`}}function N({lesson:e,stepLabel:t,suffix:a,terms:n,domain:i}){const s=`count${a}FocusTerms`,c=l(e.name,n),o=`${e.name} ${e.description}`,p=l(o,n);return{id:`${e.id}-focus-term-count`,group:e.name,stepLabel:t,title:"Count focus terms",concept:`${i.kind} systems often reduce text into small signals before ranking or checking.`,objective:"Count how many lesson focus terms appear in the text.",difficulty:"core",starterCode:`function ${s}(text) {
  const terms = ${JSON.stringify(n)};
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

check('title terms', ${s}(${JSON.stringify(e.name)}), ${c});
check('description terms', ${s}(${JSON.stringify(o)}), ${p});
check('no matching terms', ${s}('zzzz yyyy xxxx'), 0);

return results;`,hints:["lower and terms are already prepared.","Use lower.includes(terms[i]) inside the loop.","if (lower.includes(terms[i])) count += 1;"],solution:`function ${s}(text) {
  const terms = ${JSON.stringify(n)};
  const lower = text.toLowerCase();
  let count = 0;

  for (let i = 0; i < terms.length; i++) {
    if (lower.includes(terms[i])) count += 1;
  }

  return count;
}`,explanation:`This mirrors the small feature checks behind search, routing, and lesson-specific ${i.signalName} logic.`}}function L({lesson:e,stepLabel:t,suffix:a,domain:n}){const i=`best${a}Candidate`;return{id:`${e.id}-best-candidate`,group:e.name,stepLabel:t,title:"Select the best candidate",concept:`${n.kind} workflows often rank candidates by a score before choosing the next action.`,objective:"Return the id of the candidate with the highest score.",difficulty:"core",starterCode:`function ${i}(candidates) {
  let best = candidates[0];

  for (let i = 1; i < candidates.length; i++) {
    // TODO: update best when candidates[i] has a higher score.
  }

  return best.id;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('lesson candidate wins', ${i}([
  { id: 'baseline', score: 0.2 },
  { id: ${JSON.stringify(e.id)}, score: 0.9 },
  { id: 'distractor', score: 0.4 },
]), ${JSON.stringify(e.id)});

check('last candidate wins', ${i}([
  { id: 'first', score: 0.1 },
  { id: 'second', score: 0.3 },
  { id: 'third', score: 0.8 },
]), 'third');

return results;`,hints:["Compare candidates[i].score with best.score.","If the current score is larger, replace best.","if (candidates[i].score > best.score) best = candidates[i];"],solution:`function ${i}(candidates) {
  let best = candidates[0];

  for (let i = 1; i < candidates.length; i++) {
    if (candidates[i].score > best.score) best = candidates[i];
  }

  return best.id;
}`,explanation:`Ranking by ${n.signalName} is a reusable pattern across ${e.categoryName} lessons.`}}function $({lesson:e,stepLabel:t,suffix:a,domain:n}){const i=`has${a}PipelineStages`,s=n.stages,c=s.slice(0,-1),o=[...s].reverse();return{id:`${e.id}-pipeline-stage-check`,group:e.name,stepLabel:t,title:"Check required stages",concept:`${e.name} is easier to debug when the expected ${n.kind} stages are explicit.`,objective:"Return false when any required stage is missing.",difficulty:"challenge",starterCode:`function ${i}(stages) {
  const requiredStages = ${JSON.stringify(s)};

  for (let i = 0; i < requiredStages.length; i++) {
    // TODO: return false if stages does not include requiredStages[i].
  }

  return true;
}`,testCode:`const results = [];

function check(name, actual, expected) {
  results.push({ name, actual, expected, passed: Object.is(actual, expected) });
}

check('all stages present', ${i}(${JSON.stringify(s)}), true);
check('order does not matter', ${i}(${JSON.stringify(o)}), true);
check('missing one stage', ${i}(${JSON.stringify(c)}), false);

return results;`,hints:["Use stages.includes(requiredStages[i]).","Return false as soon as one required stage is absent.","if (!stages.includes(requiredStages[i])) return false;"],solution:`function ${i}(stages) {
  const requiredStages = ${JSON.stringify(s)};

  for (let i = 0; i < requiredStages.length; i++) {
    if (!stages.includes(requiredStages[i])) return false;
  }

  return true;
}`,explanation:`${n.stageExplanation} This check makes that dependency visible in code.`}}function w(e,t){const a=x.get(e.id),n=h+a,i=b(e),s=k(e.id);return{lessonId:e.id,lessonName:e.name,categoryId:e.categoryId,categoryName:e.categoryName,groupNumber:n,exercises:[y({lesson:e,stepLabel:`${n}.1`,suffix:s,keyword:i[0],domain:t}),N({lesson:e,stepLabel:`${n}.2`,suffix:s,terms:i,domain:t}),L({lesson:e,stepLabel:`${n}.3`,suffix:s,domain:t}),$({lesson:e,stepLabel:`${n}.4`,suffix:s,domain:t})]}}function r(e,t){return m.filter(a=>a.categoryId===e).map(a=>w(a,t))}const O=r("nlp",{kind:"text representation",signalName:"text relevance",stages:["tokenize","vectorize","compare"],stageExplanation:"NLP code usually has to tokenize text before it can build or compare representations."}),E=r("transformers",{kind:"sequence-modeling",signalName:"attention or routing score",stages:["project","score","mix"],stageExplanation:"Transformer internals depend on explicit projection, scoring, and mixing stages."}),_=r("frontier-llms",{kind:"frontier-model evaluation",signalName:"capability or risk score",stages:["measure","compare","gate"],stageExplanation:"Frontier-model systems need measurement, comparison, and release gates before deployment decisions."}),v=r("neural-networks",{kind:"neural-network computation",signalName:"activation or gradient signal",stages:["forward","loss","update"],stageExplanation:"Neural-network code is built around forward computation, loss measurement, and parameter updates."}),A=r("advanced-models",{kind:"advanced-model pipeline",signalName:"retrieval or multimodal score",stages:["encode","retrieve","ground"],stageExplanation:"Advanced model systems often encode inputs, retrieve or combine evidence, and check grounding."}),C=r("math-fundamentals",{kind:"mathematical computation",signalName:"numeric fit or stability score",stages:["represent","compute","check"],stageExplanation:"Math code is clearer when representation, computation, and result checks are separate."}),R=r("core-ml",{kind:"machine-learning workflow",signalName:"validation metric",stages:["split","train","evaluate"],stageExplanation:"Core ML workflows depend on clean data splits, training logic, and honest evaluation."}),B=r("model-reliability",{kind:"reliability check",signalName:"monitoring or risk score",stages:["observe","alert","triage"],stageExplanation:"Reliability systems observe behavior, alert on meaningful shifts, and triage failures."}),T=r("experimentation-causal-ml",{kind:"experiment analysis",signalName:"effect or balance score",stages:["assign","measure","compare"],stageExplanation:"Experiment code must separate assignment, measurement, and comparison to support causal claims."}),I=r("probability-stats",{kind:"probability/statistics calculation",signalName:"probability or uncertainty score",stages:["count","normalize","summarize"],stageExplanation:"Probability code often counts outcomes, normalizes them, then summarizes uncertainty."}),z=r("reinforcement-learning",{kind:"reinforcement-learning loop",signalName:"return or action-value score",stages:["observe","act","learn"],stageExplanation:"RL loops observe state, choose actions, and update behavior from feedback."}),q=r("algorithms",{kind:"algorithmic data structure",signalName:"rank or membership score",stages:["insert","query","verify"],stageExplanation:"Algorithmic structures are useful when updates, queries, and checks are kept explicit."}),J=r("diffusion-models",{kind:"diffusion-model pipeline",signalName:"noise or denoising score",stages:["noise","condition","denoise"],stageExplanation:"Diffusion systems manage noise, conditioning, and denoising as separate implementation stages."}),f=[...O,...E,..._,...v,...A,...C,...R,...B,...T,...I,...z,...q,...J],D=Object.fromEntries(f.map(e=>[e.lessonId,e]));f.flatMap(e=>e.exercises);function M(e){return D[e]||null}function F(e){var t;return((t=M(e))==null?void 0:t.exercises)||[]}export{f as L,F as g};
