import{K as w,z as x}from"./index-B6Xo-yhU.js";var b={exports:{}},h={},$={exports:{}},j={};/**
 * @license React
 * use-sync-external-store-shim.production.js
 *
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */var s=w;function V(e,t){return e===t&&(e!==0||1/e===1/t)||e!==e&&t!==t}var D=typeof Object.is=="function"?Object.is:V,R=s.useState,z=s.useEffect,O=s.useLayoutEffect,g=s.useDebugValue;function I(e,t){var u=t(),a=R({inst:{value:u,getSnapshot:t}}),r=a[0].inst,n=a[1];return O(function(){r.value=u,r.getSnapshot=t,m(r)&&n({inst:r})},[e,u,t]),z(function(){return m(r)&&n({inst:r}),e(function(){m(r)&&n({inst:r})})},[e]),g(u),u}function m(e){var t=e.getSnapshot;e=e.value;try{var u=t();return!D(e,u)}catch{return!0}}function M(e,t){return t()}var _=typeof window>"u"||typeof window.document>"u"||typeof window.document.createElement>"u"?M:I;j.useSyncExternalStore=s.useSyncExternalStore!==void 0?s.useSyncExternalStore:_;$.exports=j;var C=$.exports;/**
 * @license React
 * use-sync-external-store-shim/with-selector.production.js
 *
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */var d=w,G=C;function L(e,t){return e===t&&(e!==0||1/e===1/t)||e!==e&&t!==t}var k=typeof Object.is=="function"?Object.is:L,F=G.useSyncExternalStore,K=d.useRef,U=d.useEffect,W=d.useMemo,A=d.useDebugValue;h.useSyncExternalStoreWithSelector=function(e,t,u,a,r){var n=K(null);if(n.current===null){var f={hasValue:!1,value:null};n.current=f}else f=n.current;n=W(function(){function E(o){if(!S){if(S=!0,l=o,o=a(o),r!==void 0&&f.hasValue){var c=f.value;if(r(c,o))return v=c}return v=o}if(c=v,k(l,o))return c;var y=a(o);return r!==void 0&&r(c,y)?(l=o,c):(l=o,v=y)}var S=!1,l,v,p=u===void 0?null:u;return[function(){return E(t())},p===null?void 0:function(){return E(p())}]},[t,u,a,r]);var i=F(e,n[0],n[1]);return U(function(){f.hasValue=!0,f.value=i},[i]),A(i),i};b.exports=h;var B=b.exports;const J=x(B);export{J as u,B as w};
