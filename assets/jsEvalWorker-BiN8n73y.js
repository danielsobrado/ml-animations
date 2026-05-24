(function(){"use strict";self.onmessage=t=>{const{id:e,userCode:n,testCode:r}=t.data;try{const o=new Function(`
      "use strict";
      ${n}

      return (function runTests() {
        ${r}
      })();
    `)();self.postMessage({id:e,ok:!0,results:o,error:null})}catch(s){self.postMessage({id:e,ok:!1,results:[],error:(s==null?void 0:s.message)||String(s)})}}})();
