function A0(e,t){for(var n=0;n<t.length;n++){const r=t[n];if(typeof r!="string"&&!Array.isArray(r)){for(const s in r)if(s!=="default"&&!(s in e)){const i=Object.getOwnPropertyDescriptor(r,s);i&&Object.defineProperty(e,s,i.get?i:{enumerable:!0,get:()=>r[s]})}}}return Object.freeze(Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}))}(function(){const t=document.createElement("link").relList;if(t&&t.supports&&t.supports("modulepreload"))return;for(const s of document.querySelectorAll('link[rel="modulepreload"]'))r(s);new MutationObserver(s=>{for(const i of s)if(i.type==="childList")for(const a of i.addedNodes)a.tagName==="LINK"&&a.rel==="modulepreload"&&r(a)}).observe(document,{childList:!0,subtree:!0});function n(s){const i={};return s.integrity&&(i.integrity=s.integrity),s.referrerPolicy&&(i.referrerPolicy=s.referrerPolicy),s.crossOrigin==="use-credentials"?i.credentials="include":s.crossOrigin==="anonymous"?i.credentials="omit":i.credentials="same-origin",i}function r(s){if(s.ep)return;s.ep=!0;const i=n(s);fetch(s.href,i)}})();function R0(e){return e&&e.__esModule&&Object.prototype.hasOwnProperty.call(e,"default")?e.default:e}var mp={exports:{}},na={},gp={exports:{}},K={};/**
 * @license React
 * react.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */var pi=Symbol.for("react.element"),P0=Symbol.for("react.portal"),O0=Symbol.for("react.fragment"),D0=Symbol.for("react.strict_mode"),M0=Symbol.for("react.profiler"),L0=Symbol.for("react.provider"),U0=Symbol.for("react.context"),B0=Symbol.for("react.forward_ref"),F0=Symbol.for("react.suspense"),z0=Symbol.for("react.memo"),V0=Symbol.for("react.lazy"),Yd=Symbol.iterator;function $0(e){return e===null||typeof e!="object"?null:(e=Yd&&e[Yd]||e["@@iterator"],typeof e=="function"?e:null)}var yp={isMounted:function(){return!1},enqueueForceUpdate:function(){},enqueueReplaceState:function(){},enqueueSetState:function(){}},vp=Object.assign,wp={};function qr(e,t,n){this.props=e,this.context=t,this.refs=wp,this.updater=n||yp}qr.prototype.isReactComponent={};qr.prototype.setState=function(e,t){if(typeof e!="object"&&typeof e!="function"&&e!=null)throw Error("setState(...): takes an object of state variables to update or a function which returns an object of state variables.");this.updater.enqueueSetState(this,e,t,"setState")};qr.prototype.forceUpdate=function(e){this.updater.enqueueForceUpdate(this,e,"forceUpdate")};function xp(){}xp.prototype=qr.prototype;function Hu(e,t,n){this.props=e,this.context=t,this.refs=wp,this.updater=n||yp}var Gu=Hu.prototype=new xp;Gu.constructor=Hu;vp(Gu,qr.prototype);Gu.isPureReactComponent=!0;var Qd=Array.isArray,Sp=Object.prototype.hasOwnProperty,Wu={current:null},Ep={key:!0,ref:!0,__self:!0,__source:!0};function kp(e,t,n){var r,s={},i=null,a=null;if(t!=null)for(r in t.ref!==void 0&&(a=t.ref),t.key!==void 0&&(i=""+t.key),t)Sp.call(t,r)&&!Ep.hasOwnProperty(r)&&(s[r]=t[r]);var u=arguments.length-2;if(u===1)s.children=n;else if(1<u){for(var c=Array(u),f=0;f<u;f++)c[f]=arguments[f+2];s.children=c}if(e&&e.defaultProps)for(r in u=e.defaultProps,u)s[r]===void 0&&(s[r]=u[r]);return{$$typeof:pi,type:e,key:i,ref:a,props:s,_owner:Wu.current}}function H0(e,t){return{$$typeof:pi,type:e.type,key:t,ref:e.ref,props:e.props,_owner:e._owner}}function qu(e){return typeof e=="object"&&e!==null&&e.$$typeof===pi}function G0(e){var t={"=":"=0",":":"=2"};return"$"+e.replace(/[=:]/g,function(n){return t[n]})}var Zd=/\/+/g;function Xa(e,t){return typeof e=="object"&&e!==null&&e.key!=null?G0(""+e.key):t.toString(36)}function oo(e,t,n,r,s){var i=typeof e;(i==="undefined"||i==="boolean")&&(e=null);var a=!1;if(e===null)a=!0;else switch(i){case"string":case"number":a=!0;break;case"object":switch(e.$$typeof){case pi:case P0:a=!0}}if(a)return a=e,s=s(a),e=r===""?"."+Xa(a,0):r,Qd(s)?(n="",e!=null&&(n=e.replace(Zd,"$&/")+"/"),oo(s,t,n,"",function(f){return f})):s!=null&&(qu(s)&&(s=H0(s,n+(!s.key||a&&a.key===s.key?"":(""+s.key).replace(Zd,"$&/")+"/")+e)),t.push(s)),1;if(a=0,r=r===""?".":r+":",Qd(e))for(var u=0;u<e.length;u++){i=e[u];var c=r+Xa(i,u);a+=oo(i,t,n,c,s)}else if(c=$0(e),typeof c=="function")for(e=c.call(e),u=0;!(i=e.next()).done;)i=i.value,c=r+Xa(i,u++),a+=oo(i,t,n,c,s);else if(i==="object")throw t=String(e),Error("Objects are not valid as a React child (found: "+(t==="[object Object]"?"object with keys {"+Object.keys(e).join(", ")+"}":t)+"). If you meant to render a collection of children, use an array instead.");return a}function Mi(e,t,n){if(e==null)return e;var r=[],s=0;return oo(e,r,"","",function(i){return t.call(n,i,s++)}),r}function W0(e){if(e._status===-1){var t=e._result;t=t(),t.then(function(n){(e._status===0||e._status===-1)&&(e._status=1,e._result=n)},function(n){(e._status===0||e._status===-1)&&(e._status=2,e._result=n)}),e._status===-1&&(e._status=0,e._result=t)}if(e._status===1)return e._result.default;throw e._result}var Ve={current:null},ao={transition:null},q0={ReactCurrentDispatcher:Ve,ReactCurrentBatchConfig:ao,ReactCurrentOwner:Wu};function bp(){throw Error("act(...) is not supported in production builds of React.")}K.Children={map:Mi,forEach:function(e,t,n){Mi(e,function(){t.apply(this,arguments)},n)},count:function(e){var t=0;return Mi(e,function(){t++}),t},toArray:function(e){return Mi(e,function(t){return t})||[]},only:function(e){if(!qu(e))throw Error("React.Children.only expected to receive a single React element child.");return e}};K.Component=qr;K.Fragment=O0;K.Profiler=M0;K.PureComponent=Hu;K.StrictMode=D0;K.Suspense=F0;K.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED=q0;K.act=bp;K.cloneElement=function(e,t,n){if(e==null)throw Error("React.cloneElement(...): The argument must be a React element, but you passed "+e+".");var r=vp({},e.props),s=e.key,i=e.ref,a=e._owner;if(t!=null){if(t.ref!==void 0&&(i=t.ref,a=Wu.current),t.key!==void 0&&(s=""+t.key),e.type&&e.type.defaultProps)var u=e.type.defaultProps;for(c in t)Sp.call(t,c)&&!Ep.hasOwnProperty(c)&&(r[c]=t[c]===void 0&&u!==void 0?u[c]:t[c])}var c=arguments.length-2;if(c===1)r.children=n;else if(1<c){u=Array(c);for(var f=0;f<c;f++)u[f]=arguments[f+2];r.children=u}return{$$typeof:pi,type:e.type,key:s,ref:i,props:r,_owner:a}};K.createContext=function(e){return e={$$typeof:U0,_currentValue:e,_currentValue2:e,_threadCount:0,Provider:null,Consumer:null,_defaultValue:null,_globalName:null},e.Provider={$$typeof:L0,_context:e},e.Consumer=e};K.createElement=kp;K.createFactory=function(e){var t=kp.bind(null,e);return t.type=e,t};K.createRef=function(){return{current:null}};K.forwardRef=function(e){return{$$typeof:B0,render:e}};K.isValidElement=qu;K.lazy=function(e){return{$$typeof:V0,_payload:{_status:-1,_result:e},_init:W0}};K.memo=function(e,t){return{$$typeof:z0,type:e,compare:t===void 0?null:t}};K.startTransition=function(e){var t=ao.transition;ao.transition={};try{e()}finally{ao.transition=t}};K.unstable_act=bp;K.useCallback=function(e,t){return Ve.current.useCallback(e,t)};K.useContext=function(e){return Ve.current.useContext(e)};K.useDebugValue=function(){};K.useDeferredValue=function(e){return Ve.current.useDeferredValue(e)};K.useEffect=function(e,t){return Ve.current.useEffect(e,t)};K.useId=function(){return Ve.current.useId()};K.useImperativeHandle=function(e,t,n){return Ve.current.useImperativeHandle(e,t,n)};K.useInsertionEffect=function(e,t){return Ve.current.useInsertionEffect(e,t)};K.useLayoutEffect=function(e,t){return Ve.current.useLayoutEffect(e,t)};K.useMemo=function(e,t){return Ve.current.useMemo(e,t)};K.useReducer=function(e,t,n){return Ve.current.useReducer(e,t,n)};K.useRef=function(e){return Ve.current.useRef(e)};K.useState=function(e){return Ve.current.useState(e)};K.useSyncExternalStore=function(e,t,n){return Ve.current.useSyncExternalStore(e,t,n)};K.useTransition=function(){return Ve.current.useTransition()};K.version="18.3.1";gp.exports=K;var N=gp.exports;const fn=R0(N),K0=A0({__proto__:null,default:fn},[N]);/**
 * @license React
 * react-jsx-runtime.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */var X0=N,J0=Symbol.for("react.element"),Y0=Symbol.for("react.fragment"),Q0=Object.prototype.hasOwnProperty,Z0=X0.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED.ReactCurrentOwner,ev={key:!0,ref:!0,__self:!0,__source:!0};function _p(e,t,n){var r,s={},i=null,a=null;n!==void 0&&(i=""+n),t.key!==void 0&&(i=""+t.key),t.ref!==void 0&&(a=t.ref);for(r in t)Q0.call(t,r)&&!ev.hasOwnProperty(r)&&(s[r]=t[r]);if(e&&e.defaultProps)for(r in t=e.defaultProps,t)s[r]===void 0&&(s[r]=t[r]);return{$$typeof:J0,type:e,key:i,ref:a,props:s,_owner:Z0.current}}na.Fragment=Y0;na.jsx=_p;na.jsxs=_p;mp.exports=na;var m=mp.exports,Ll={},Cp={exports:{}},rt={},Ip={exports:{}},Tp={};/**
 * @license React
 * scheduler.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */(function(e){function t(F,H){var W=F.length;F.push(H);e:for(;0<W;){var V=W-1>>>1,G=F[V];if(0<s(G,H))F[V]=H,F[W]=G,W=V;else break e}}function n(F){return F.length===0?null:F[0]}function r(F){if(F.length===0)return null;var H=F[0],W=F.pop();if(W!==H){F[0]=W;e:for(var V=0,G=F.length,ge=G>>>1;V<ge;){var _e=2*(V+1)-1,Ge=F[_e],ft=_e+1,Xt=F[ft];if(0>s(Ge,W))ft<G&&0>s(Xt,Ge)?(F[V]=Xt,F[ft]=W,V=ft):(F[V]=Ge,F[_e]=W,V=_e);else if(ft<G&&0>s(Xt,W))F[V]=Xt,F[ft]=W,V=ft;else break e}}return H}function s(F,H){var W=F.sortIndex-H.sortIndex;return W!==0?W:F.id-H.id}if(typeof performance=="object"&&typeof performance.now=="function"){var i=performance;e.unstable_now=function(){return i.now()}}else{var a=Date,u=a.now();e.unstable_now=function(){return a.now()-u}}var c=[],f=[],E=1,w=null,_=3,A=!1,R=!1,j=!1,D=typeof setTimeout=="function"?setTimeout:null,b=typeof clearTimeout=="function"?clearTimeout:null,x=typeof setImmediate<"u"?setImmediate:null;typeof navigator<"u"&&navigator.scheduling!==void 0&&navigator.scheduling.isInputPending!==void 0&&navigator.scheduling.isInputPending.bind(navigator.scheduling);function C(F){for(var H=n(f);H!==null;){if(H.callback===null)r(f);else if(H.startTime<=F)r(f),H.sortIndex=H.expirationTime,t(c,H);else break;H=n(f)}}function O(F){if(j=!1,C(F),!R)if(n(c)!==null)R=!0,te(M);else{var H=n(f);H!==null&&He(O,H.startTime-F)}}function M(F,H){R=!1,j&&(j=!1,b(p),p=-1),A=!0;var W=_;try{for(C(H),w=n(c);w!==null&&(!(w.expirationTime>H)||F&&!k());){var V=w.callback;if(typeof V=="function"){w.callback=null,_=w.priorityLevel;var G=V(w.expirationTime<=H);H=e.unstable_now(),typeof G=="function"?w.callback=G:w===n(c)&&r(c),C(H)}else r(c);w=n(c)}if(w!==null)var ge=!0;else{var _e=n(f);_e!==null&&He(O,_e.startTime-H),ge=!1}return ge}finally{w=null,_=W,A=!1}}var U=!1,g=null,p=-1,v=5,S=-1;function k(){return!(e.unstable_now()-S<v)}function I(){if(g!==null){var F=e.unstable_now();S=F;var H=!0;try{H=g(!0,F)}finally{H?y():(U=!1,g=null)}}else U=!1}var y;if(typeof x=="function")y=function(){x(I)};else if(typeof MessageChannel<"u"){var Y=new MessageChannel,$=Y.port2;Y.port1.onmessage=I,y=function(){$.postMessage(null)}}else y=function(){D(I,0)};function te(F){g=F,U||(U=!0,y())}function He(F,H){p=D(function(){F(e.unstable_now())},H)}e.unstable_IdlePriority=5,e.unstable_ImmediatePriority=1,e.unstable_LowPriority=4,e.unstable_NormalPriority=3,e.unstable_Profiling=null,e.unstable_UserBlockingPriority=2,e.unstable_cancelCallback=function(F){F.callback=null},e.unstable_continueExecution=function(){R||A||(R=!0,te(M))},e.unstable_forceFrameRate=function(F){0>F||125<F?console.error("forceFrameRate takes a positive int between 0 and 125, forcing frame rates higher than 125 fps is not supported"):v=0<F?Math.floor(1e3/F):5},e.unstable_getCurrentPriorityLevel=function(){return _},e.unstable_getFirstCallbackNode=function(){return n(c)},e.unstable_next=function(F){switch(_){case 1:case 2:case 3:var H=3;break;default:H=_}var W=_;_=H;try{return F()}finally{_=W}},e.unstable_pauseExecution=function(){},e.unstable_requestPaint=function(){},e.unstable_runWithPriority=function(F,H){switch(F){case 1:case 2:case 3:case 4:case 5:break;default:F=3}var W=_;_=F;try{return H()}finally{_=W}},e.unstable_scheduleCallback=function(F,H,W){var V=e.unstable_now();switch(typeof W=="object"&&W!==null?(W=W.delay,W=typeof W=="number"&&0<W?V+W:V):W=V,F){case 1:var G=-1;break;case 2:G=250;break;case 5:G=1073741823;break;case 4:G=1e4;break;default:G=5e3}return G=W+G,F={id:E++,callback:H,priorityLevel:F,startTime:W,expirationTime:G,sortIndex:-1},W>V?(F.sortIndex=W,t(f,F),n(c)===null&&F===n(f)&&(j?(b(p),p=-1):j=!0,He(O,W-V))):(F.sortIndex=G,t(c,F),R||A||(R=!0,te(M))),F},e.unstable_shouldYield=k,e.unstable_wrapCallback=function(F){var H=_;return function(){var W=_;_=H;try{return F.apply(this,arguments)}finally{_=W}}}})(Tp);Ip.exports=Tp;var tv=Ip.exports;/**
 * @license React
 * react-dom.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */var nv=N,nt=tv;function L(e){for(var t="https://reactjs.org/docs/error-decoder.html?invariant="+e,n=1;n<arguments.length;n++)t+="&args[]="+encodeURIComponent(arguments[n]);return"Minified React error #"+e+"; visit "+t+" for the full message or use the non-minified dev environment for full errors and additional helpful warnings."}var Np=new Set,Hs={};function sr(e,t){Ur(e,t),Ur(e+"Capture",t)}function Ur(e,t){for(Hs[e]=t,e=0;e<t.length;e++)Np.add(t[e])}var Vt=!(typeof window>"u"||typeof window.document>"u"||typeof window.document.createElement>"u"),Ul=Object.prototype.hasOwnProperty,rv=/^[:A-Z_a-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD][:A-Z_a-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD\-.0-9\u00B7\u0300-\u036F\u203F-\u2040]*$/,ef={},tf={};function sv(e){return Ul.call(tf,e)?!0:Ul.call(ef,e)?!1:rv.test(e)?tf[e]=!0:(ef[e]=!0,!1)}function iv(e,t,n,r){if(n!==null&&n.type===0)return!1;switch(typeof t){case"function":case"symbol":return!0;case"boolean":return r?!1:n!==null?!n.acceptsBooleans:(e=e.toLowerCase().slice(0,5),e!=="data-"&&e!=="aria-");default:return!1}}function ov(e,t,n,r){if(t===null||typeof t>"u"||iv(e,t,n,r))return!0;if(r)return!1;if(n!==null)switch(n.type){case 3:return!t;case 4:return t===!1;case 5:return isNaN(t);case 6:return isNaN(t)||1>t}return!1}function $e(e,t,n,r,s,i,a){this.acceptsBooleans=t===2||t===3||t===4,this.attributeName=r,this.attributeNamespace=s,this.mustUseProperty=n,this.propertyName=e,this.type=t,this.sanitizeURL=i,this.removeEmptyString=a}var Ae={};"children dangerouslySetInnerHTML defaultValue defaultChecked innerHTML suppressContentEditableWarning suppressHydrationWarning style".split(" ").forEach(function(e){Ae[e]=new $e(e,0,!1,e,null,!1,!1)});[["acceptCharset","accept-charset"],["className","class"],["htmlFor","for"],["httpEquiv","http-equiv"]].forEach(function(e){var t=e[0];Ae[t]=new $e(t,1,!1,e[1],null,!1,!1)});["contentEditable","draggable","spellCheck","value"].forEach(function(e){Ae[e]=new $e(e,2,!1,e.toLowerCase(),null,!1,!1)});["autoReverse","externalResourcesRequired","focusable","preserveAlpha"].forEach(function(e){Ae[e]=new $e(e,2,!1,e,null,!1,!1)});"allowFullScreen async autoFocus autoPlay controls default defer disabled disablePictureInPicture disableRemotePlayback formNoValidate hidden loop noModule noValidate open playsInline readOnly required reversed scoped seamless itemScope".split(" ").forEach(function(e){Ae[e]=new $e(e,3,!1,e.toLowerCase(),null,!1,!1)});["checked","multiple","muted","selected"].forEach(function(e){Ae[e]=new $e(e,3,!0,e,null,!1,!1)});["capture","download"].forEach(function(e){Ae[e]=new $e(e,4,!1,e,null,!1,!1)});["cols","rows","size","span"].forEach(function(e){Ae[e]=new $e(e,6,!1,e,null,!1,!1)});["rowSpan","start"].forEach(function(e){Ae[e]=new $e(e,5,!1,e.toLowerCase(),null,!1,!1)});var Ku=/[\-:]([a-z])/g;function Xu(e){return e[1].toUpperCase()}"accent-height alignment-baseline arabic-form baseline-shift cap-height clip-path clip-rule color-interpolation color-interpolation-filters color-profile color-rendering dominant-baseline enable-background fill-opacity fill-rule flood-color flood-opacity font-family font-size font-size-adjust font-stretch font-style font-variant font-weight glyph-name glyph-orientation-horizontal glyph-orientation-vertical horiz-adv-x horiz-origin-x image-rendering letter-spacing lighting-color marker-end marker-mid marker-start overline-position overline-thickness paint-order panose-1 pointer-events rendering-intent shape-rendering stop-color stop-opacity strikethrough-position strikethrough-thickness stroke-dasharray stroke-dashoffset stroke-linecap stroke-linejoin stroke-miterlimit stroke-opacity stroke-width text-anchor text-decoration text-rendering underline-position underline-thickness unicode-bidi unicode-range units-per-em v-alphabetic v-hanging v-ideographic v-mathematical vector-effect vert-adv-y vert-origin-x vert-origin-y word-spacing writing-mode xmlns:xlink x-height".split(" ").forEach(function(e){var t=e.replace(Ku,Xu);Ae[t]=new $e(t,1,!1,e,null,!1,!1)});"xlink:actuate xlink:arcrole xlink:role xlink:show xlink:title xlink:type".split(" ").forEach(function(e){var t=e.replace(Ku,Xu);Ae[t]=new $e(t,1,!1,e,"http://www.w3.org/1999/xlink",!1,!1)});["xml:base","xml:lang","xml:space"].forEach(function(e){var t=e.replace(Ku,Xu);Ae[t]=new $e(t,1,!1,e,"http://www.w3.org/XML/1998/namespace",!1,!1)});["tabIndex","crossOrigin"].forEach(function(e){Ae[e]=new $e(e,1,!1,e.toLowerCase(),null,!1,!1)});Ae.xlinkHref=new $e("xlinkHref",1,!1,"xlink:href","http://www.w3.org/1999/xlink",!0,!1);["src","href","action","formAction"].forEach(function(e){Ae[e]=new $e(e,1,!1,e.toLowerCase(),null,!0,!0)});function Ju(e,t,n,r){var s=Ae.hasOwnProperty(t)?Ae[t]:null;(s!==null?s.type!==0:r||!(2<t.length)||t[0]!=="o"&&t[0]!=="O"||t[1]!=="n"&&t[1]!=="N")&&(ov(t,n,s,r)&&(n=null),r||s===null?sv(t)&&(n===null?e.removeAttribute(t):e.setAttribute(t,""+n)):s.mustUseProperty?e[s.propertyName]=n===null?s.type===3?!1:"":n:(t=s.attributeName,r=s.attributeNamespace,n===null?e.removeAttribute(t):(s=s.type,n=s===3||s===4&&n===!0?"":""+n,r?e.setAttributeNS(r,t,n):e.setAttribute(t,n))))}var qt=nv.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED,Li=Symbol.for("react.element"),hr=Symbol.for("react.portal"),pr=Symbol.for("react.fragment"),Yu=Symbol.for("react.strict_mode"),Bl=Symbol.for("react.profiler"),jp=Symbol.for("react.provider"),Ap=Symbol.for("react.context"),Qu=Symbol.for("react.forward_ref"),Fl=Symbol.for("react.suspense"),zl=Symbol.for("react.suspense_list"),Zu=Symbol.for("react.memo"),sn=Symbol.for("react.lazy"),Rp=Symbol.for("react.offscreen"),nf=Symbol.iterator;function ms(e){return e===null||typeof e!="object"?null:(e=nf&&e[nf]||e["@@iterator"],typeof e=="function"?e:null)}var pe=Object.assign,Ja;function ks(e){if(Ja===void 0)try{throw Error()}catch(n){var t=n.stack.trim().match(/\n( *(at )?)/);Ja=t&&t[1]||""}return`
`+Ja+e}var Ya=!1;function Qa(e,t){if(!e||Ya)return"";Ya=!0;var n=Error.prepareStackTrace;Error.prepareStackTrace=void 0;try{if(t)if(t=function(){throw Error()},Object.defineProperty(t.prototype,"props",{set:function(){throw Error()}}),typeof Reflect=="object"&&Reflect.construct){try{Reflect.construct(t,[])}catch(f){var r=f}Reflect.construct(e,[],t)}else{try{t.call()}catch(f){r=f}e.call(t.prototype)}else{try{throw Error()}catch(f){r=f}e()}}catch(f){if(f&&r&&typeof f.stack=="string"){for(var s=f.stack.split(`
`),i=r.stack.split(`
`),a=s.length-1,u=i.length-1;1<=a&&0<=u&&s[a]!==i[u];)u--;for(;1<=a&&0<=u;a--,u--)if(s[a]!==i[u]){if(a!==1||u!==1)do if(a--,u--,0>u||s[a]!==i[u]){var c=`
`+s[a].replace(" at new "," at ");return e.displayName&&c.includes("<anonymous>")&&(c=c.replace("<anonymous>",e.displayName)),c}while(1<=a&&0<=u);break}}}finally{Ya=!1,Error.prepareStackTrace=n}return(e=e?e.displayName||e.name:"")?ks(e):""}function av(e){switch(e.tag){case 5:return ks(e.type);case 16:return ks("Lazy");case 13:return ks("Suspense");case 19:return ks("SuspenseList");case 0:case 2:case 15:return e=Qa(e.type,!1),e;case 11:return e=Qa(e.type.render,!1),e;case 1:return e=Qa(e.type,!0),e;default:return""}}function Vl(e){if(e==null)return null;if(typeof e=="function")return e.displayName||e.name||null;if(typeof e=="string")return e;switch(e){case pr:return"Fragment";case hr:return"Portal";case Bl:return"Profiler";case Yu:return"StrictMode";case Fl:return"Suspense";case zl:return"SuspenseList"}if(typeof e=="object")switch(e.$$typeof){case Ap:return(e.displayName||"Context")+".Consumer";case jp:return(e._context.displayName||"Context")+".Provider";case Qu:var t=e.render;return e=e.displayName,e||(e=t.displayName||t.name||"",e=e!==""?"ForwardRef("+e+")":"ForwardRef"),e;case Zu:return t=e.displayName||null,t!==null?t:Vl(e.type)||"Memo";case sn:t=e._payload,e=e._init;try{return Vl(e(t))}catch{}}return null}function lv(e){var t=e.type;switch(e.tag){case 24:return"Cache";case 9:return(t.displayName||"Context")+".Consumer";case 10:return(t._context.displayName||"Context")+".Provider";case 18:return"DehydratedFragment";case 11:return e=t.render,e=e.displayName||e.name||"",t.displayName||(e!==""?"ForwardRef("+e+")":"ForwardRef");case 7:return"Fragment";case 5:return t;case 4:return"Portal";case 3:return"Root";case 6:return"Text";case 16:return Vl(t);case 8:return t===Yu?"StrictMode":"Mode";case 22:return"Offscreen";case 12:return"Profiler";case 21:return"Scope";case 13:return"Suspense";case 19:return"SuspenseList";case 25:return"TracingMarker";case 1:case 0:case 17:case 2:case 14:case 15:if(typeof t=="function")return t.displayName||t.name||null;if(typeof t=="string")return t}return null}function _n(e){switch(typeof e){case"boolean":case"number":case"string":case"undefined":return e;case"object":return e;default:return""}}function Pp(e){var t=e.type;return(e=e.nodeName)&&e.toLowerCase()==="input"&&(t==="checkbox"||t==="radio")}function uv(e){var t=Pp(e)?"checked":"value",n=Object.getOwnPropertyDescriptor(e.constructor.prototype,t),r=""+e[t];if(!e.hasOwnProperty(t)&&typeof n<"u"&&typeof n.get=="function"&&typeof n.set=="function"){var s=n.get,i=n.set;return Object.defineProperty(e,t,{configurable:!0,get:function(){return s.call(this)},set:function(a){r=""+a,i.call(this,a)}}),Object.defineProperty(e,t,{enumerable:n.enumerable}),{getValue:function(){return r},setValue:function(a){r=""+a},stopTracking:function(){e._valueTracker=null,delete e[t]}}}}function Ui(e){e._valueTracker||(e._valueTracker=uv(e))}function Op(e){if(!e)return!1;var t=e._valueTracker;if(!t)return!0;var n=t.getValue(),r="";return e&&(r=Pp(e)?e.checked?"true":"false":e.value),e=r,e!==n?(t.setValue(e),!0):!1}function ko(e){if(e=e||(typeof document<"u"?document:void 0),typeof e>"u")return null;try{return e.activeElement||e.body}catch{return e.body}}function $l(e,t){var n=t.checked;return pe({},t,{defaultChecked:void 0,defaultValue:void 0,value:void 0,checked:n??e._wrapperState.initialChecked})}function rf(e,t){var n=t.defaultValue==null?"":t.defaultValue,r=t.checked!=null?t.checked:t.defaultChecked;n=_n(t.value!=null?t.value:n),e._wrapperState={initialChecked:r,initialValue:n,controlled:t.type==="checkbox"||t.type==="radio"?t.checked!=null:t.value!=null}}function Dp(e,t){t=t.checked,t!=null&&Ju(e,"checked",t,!1)}function Hl(e,t){Dp(e,t);var n=_n(t.value),r=t.type;if(n!=null)r==="number"?(n===0&&e.value===""||e.value!=n)&&(e.value=""+n):e.value!==""+n&&(e.value=""+n);else if(r==="submit"||r==="reset"){e.removeAttribute("value");return}t.hasOwnProperty("value")?Gl(e,t.type,n):t.hasOwnProperty("defaultValue")&&Gl(e,t.type,_n(t.defaultValue)),t.checked==null&&t.defaultChecked!=null&&(e.defaultChecked=!!t.defaultChecked)}function sf(e,t,n){if(t.hasOwnProperty("value")||t.hasOwnProperty("defaultValue")){var r=t.type;if(!(r!=="submit"&&r!=="reset"||t.value!==void 0&&t.value!==null))return;t=""+e._wrapperState.initialValue,n||t===e.value||(e.value=t),e.defaultValue=t}n=e.name,n!==""&&(e.name=""),e.defaultChecked=!!e._wrapperState.initialChecked,n!==""&&(e.name=n)}function Gl(e,t,n){(t!=="number"||ko(e.ownerDocument)!==e)&&(n==null?e.defaultValue=""+e._wrapperState.initialValue:e.defaultValue!==""+n&&(e.defaultValue=""+n))}var bs=Array.isArray;function Ir(e,t,n,r){if(e=e.options,t){t={};for(var s=0;s<n.length;s++)t["$"+n[s]]=!0;for(n=0;n<e.length;n++)s=t.hasOwnProperty("$"+e[n].value),e[n].selected!==s&&(e[n].selected=s),s&&r&&(e[n].defaultSelected=!0)}else{for(n=""+_n(n),t=null,s=0;s<e.length;s++){if(e[s].value===n){e[s].selected=!0,r&&(e[s].defaultSelected=!0);return}t!==null||e[s].disabled||(t=e[s])}t!==null&&(t.selected=!0)}}function Wl(e,t){if(t.dangerouslySetInnerHTML!=null)throw Error(L(91));return pe({},t,{value:void 0,defaultValue:void 0,children:""+e._wrapperState.initialValue})}function of(e,t){var n=t.value;if(n==null){if(n=t.children,t=t.defaultValue,n!=null){if(t!=null)throw Error(L(92));if(bs(n)){if(1<n.length)throw Error(L(93));n=n[0]}t=n}t==null&&(t=""),n=t}e._wrapperState={initialValue:_n(n)}}function Mp(e,t){var n=_n(t.value),r=_n(t.defaultValue);n!=null&&(n=""+n,n!==e.value&&(e.value=n),t.defaultValue==null&&e.defaultValue!==n&&(e.defaultValue=n)),r!=null&&(e.defaultValue=""+r)}function af(e){var t=e.textContent;t===e._wrapperState.initialValue&&t!==""&&t!==null&&(e.value=t)}function Lp(e){switch(e){case"svg":return"http://www.w3.org/2000/svg";case"math":return"http://www.w3.org/1998/Math/MathML";default:return"http://www.w3.org/1999/xhtml"}}function ql(e,t){return e==null||e==="http://www.w3.org/1999/xhtml"?Lp(t):e==="http://www.w3.org/2000/svg"&&t==="foreignObject"?"http://www.w3.org/1999/xhtml":e}var Bi,Up=function(e){return typeof MSApp<"u"&&MSApp.execUnsafeLocalFunction?function(t,n,r,s){MSApp.execUnsafeLocalFunction(function(){return e(t,n,r,s)})}:e}(function(e,t){if(e.namespaceURI!=="http://www.w3.org/2000/svg"||"innerHTML"in e)e.innerHTML=t;else{for(Bi=Bi||document.createElement("div"),Bi.innerHTML="<svg>"+t.valueOf().toString()+"</svg>",t=Bi.firstChild;e.firstChild;)e.removeChild(e.firstChild);for(;t.firstChild;)e.appendChild(t.firstChild)}});function Gs(e,t){if(t){var n=e.firstChild;if(n&&n===e.lastChild&&n.nodeType===3){n.nodeValue=t;return}}e.textContent=t}var Ts={animationIterationCount:!0,aspectRatio:!0,borderImageOutset:!0,borderImageSlice:!0,borderImageWidth:!0,boxFlex:!0,boxFlexGroup:!0,boxOrdinalGroup:!0,columnCount:!0,columns:!0,flex:!0,flexGrow:!0,flexPositive:!0,flexShrink:!0,flexNegative:!0,flexOrder:!0,gridArea:!0,gridRow:!0,gridRowEnd:!0,gridRowSpan:!0,gridRowStart:!0,gridColumn:!0,gridColumnEnd:!0,gridColumnSpan:!0,gridColumnStart:!0,fontWeight:!0,lineClamp:!0,lineHeight:!0,opacity:!0,order:!0,orphans:!0,tabSize:!0,widows:!0,zIndex:!0,zoom:!0,fillOpacity:!0,floodOpacity:!0,stopOpacity:!0,strokeDasharray:!0,strokeDashoffset:!0,strokeMiterlimit:!0,strokeOpacity:!0,strokeWidth:!0},cv=["Webkit","ms","Moz","O"];Object.keys(Ts).forEach(function(e){cv.forEach(function(t){t=t+e.charAt(0).toUpperCase()+e.substring(1),Ts[t]=Ts[e]})});function Bp(e,t,n){return t==null||typeof t=="boolean"||t===""?"":n||typeof t!="number"||t===0||Ts.hasOwnProperty(e)&&Ts[e]?(""+t).trim():t+"px"}function Fp(e,t){e=e.style;for(var n in t)if(t.hasOwnProperty(n)){var r=n.indexOf("--")===0,s=Bp(n,t[n],r);n==="float"&&(n="cssFloat"),r?e.setProperty(n,s):e[n]=s}}var dv=pe({menuitem:!0},{area:!0,base:!0,br:!0,col:!0,embed:!0,hr:!0,img:!0,input:!0,keygen:!0,link:!0,meta:!0,param:!0,source:!0,track:!0,wbr:!0});function Kl(e,t){if(t){if(dv[e]&&(t.children!=null||t.dangerouslySetInnerHTML!=null))throw Error(L(137,e));if(t.dangerouslySetInnerHTML!=null){if(t.children!=null)throw Error(L(60));if(typeof t.dangerouslySetInnerHTML!="object"||!("__html"in t.dangerouslySetInnerHTML))throw Error(L(61))}if(t.style!=null&&typeof t.style!="object")throw Error(L(62))}}function Xl(e,t){if(e.indexOf("-")===-1)return typeof t.is=="string";switch(e){case"annotation-xml":case"color-profile":case"font-face":case"font-face-src":case"font-face-uri":case"font-face-format":case"font-face-name":case"missing-glyph":return!1;default:return!0}}var Jl=null;function ec(e){return e=e.target||e.srcElement||window,e.correspondingUseElement&&(e=e.correspondingUseElement),e.nodeType===3?e.parentNode:e}var Yl=null,Tr=null,Nr=null;function lf(e){if(e=yi(e)){if(typeof Yl!="function")throw Error(L(280));var t=e.stateNode;t&&(t=aa(t),Yl(e.stateNode,e.type,t))}}function zp(e){Tr?Nr?Nr.push(e):Nr=[e]:Tr=e}function Vp(){if(Tr){var e=Tr,t=Nr;if(Nr=Tr=null,lf(e),t)for(e=0;e<t.length;e++)lf(t[e])}}function $p(e,t){return e(t)}function Hp(){}var Za=!1;function Gp(e,t,n){if(Za)return e(t,n);Za=!0;try{return $p(e,t,n)}finally{Za=!1,(Tr!==null||Nr!==null)&&(Hp(),Vp())}}function Ws(e,t){var n=e.stateNode;if(n===null)return null;var r=aa(n);if(r===null)return null;n=r[t];e:switch(t){case"onClick":case"onClickCapture":case"onDoubleClick":case"onDoubleClickCapture":case"onMouseDown":case"onMouseDownCapture":case"onMouseMove":case"onMouseMoveCapture":case"onMouseUp":case"onMouseUpCapture":case"onMouseEnter":(r=!r.disabled)||(e=e.type,r=!(e==="button"||e==="input"||e==="select"||e==="textarea")),e=!r;break e;default:e=!1}if(e)return null;if(n&&typeof n!="function")throw Error(L(231,t,typeof n));return n}var Ql=!1;if(Vt)try{var gs={};Object.defineProperty(gs,"passive",{get:function(){Ql=!0}}),window.addEventListener("test",gs,gs),window.removeEventListener("test",gs,gs)}catch{Ql=!1}function fv(e,t,n,r,s,i,a,u,c){var f=Array.prototype.slice.call(arguments,3);try{t.apply(n,f)}catch(E){this.onError(E)}}var Ns=!1,bo=null,_o=!1,Zl=null,hv={onError:function(e){Ns=!0,bo=e}};function pv(e,t,n,r,s,i,a,u,c){Ns=!1,bo=null,fv.apply(hv,arguments)}function mv(e,t,n,r,s,i,a,u,c){if(pv.apply(this,arguments),Ns){if(Ns){var f=bo;Ns=!1,bo=null}else throw Error(L(198));_o||(_o=!0,Zl=f)}}function ir(e){var t=e,n=e;if(e.alternate)for(;t.return;)t=t.return;else{e=t;do t=e,t.flags&4098&&(n=t.return),e=t.return;while(e)}return t.tag===3?n:null}function Wp(e){if(e.tag===13){var t=e.memoizedState;if(t===null&&(e=e.alternate,e!==null&&(t=e.memoizedState)),t!==null)return t.dehydrated}return null}function uf(e){if(ir(e)!==e)throw Error(L(188))}function gv(e){var t=e.alternate;if(!t){if(t=ir(e),t===null)throw Error(L(188));return t!==e?null:e}for(var n=e,r=t;;){var s=n.return;if(s===null)break;var i=s.alternate;if(i===null){if(r=s.return,r!==null){n=r;continue}break}if(s.child===i.child){for(i=s.child;i;){if(i===n)return uf(s),e;if(i===r)return uf(s),t;i=i.sibling}throw Error(L(188))}if(n.return!==r.return)n=s,r=i;else{for(var a=!1,u=s.child;u;){if(u===n){a=!0,n=s,r=i;break}if(u===r){a=!0,r=s,n=i;break}u=u.sibling}if(!a){for(u=i.child;u;){if(u===n){a=!0,n=i,r=s;break}if(u===r){a=!0,r=i,n=s;break}u=u.sibling}if(!a)throw Error(L(189))}}if(n.alternate!==r)throw Error(L(190))}if(n.tag!==3)throw Error(L(188));return n.stateNode.current===n?e:t}function qp(e){return e=gv(e),e!==null?Kp(e):null}function Kp(e){if(e.tag===5||e.tag===6)return e;for(e=e.child;e!==null;){var t=Kp(e);if(t!==null)return t;e=e.sibling}return null}var Xp=nt.unstable_scheduleCallback,cf=nt.unstable_cancelCallback,yv=nt.unstable_shouldYield,vv=nt.unstable_requestPaint,ve=nt.unstable_now,wv=nt.unstable_getCurrentPriorityLevel,tc=nt.unstable_ImmediatePriority,Jp=nt.unstable_UserBlockingPriority,Co=nt.unstable_NormalPriority,xv=nt.unstable_LowPriority,Yp=nt.unstable_IdlePriority,ra=null,Pt=null;function Sv(e){if(Pt&&typeof Pt.onCommitFiberRoot=="function")try{Pt.onCommitFiberRoot(ra,e,void 0,(e.current.flags&128)===128)}catch{}}var wt=Math.clz32?Math.clz32:bv,Ev=Math.log,kv=Math.LN2;function bv(e){return e>>>=0,e===0?32:31-(Ev(e)/kv|0)|0}var Fi=64,zi=4194304;function _s(e){switch(e&-e){case 1:return 1;case 2:return 2;case 4:return 4;case 8:return 8;case 16:return 16;case 32:return 32;case 64:case 128:case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:return e&4194240;case 4194304:case 8388608:case 16777216:case 33554432:case 67108864:return e&130023424;case 134217728:return 134217728;case 268435456:return 268435456;case 536870912:return 536870912;case 1073741824:return 1073741824;default:return e}}function Io(e,t){var n=e.pendingLanes;if(n===0)return 0;var r=0,s=e.suspendedLanes,i=e.pingedLanes,a=n&268435455;if(a!==0){var u=a&~s;u!==0?r=_s(u):(i&=a,i!==0&&(r=_s(i)))}else a=n&~s,a!==0?r=_s(a):i!==0&&(r=_s(i));if(r===0)return 0;if(t!==0&&t!==r&&!(t&s)&&(s=r&-r,i=t&-t,s>=i||s===16&&(i&4194240)!==0))return t;if(r&4&&(r|=n&16),t=e.entangledLanes,t!==0)for(e=e.entanglements,t&=r;0<t;)n=31-wt(t),s=1<<n,r|=e[n],t&=~s;return r}function _v(e,t){switch(e){case 1:case 2:case 4:return t+250;case 8:case 16:case 32:case 64:case 128:case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:return t+5e3;case 4194304:case 8388608:case 16777216:case 33554432:case 67108864:return-1;case 134217728:case 268435456:case 536870912:case 1073741824:return-1;default:return-1}}function Cv(e,t){for(var n=e.suspendedLanes,r=e.pingedLanes,s=e.expirationTimes,i=e.pendingLanes;0<i;){var a=31-wt(i),u=1<<a,c=s[a];c===-1?(!(u&n)||u&r)&&(s[a]=_v(u,t)):c<=t&&(e.expiredLanes|=u),i&=~u}}function eu(e){return e=e.pendingLanes&-1073741825,e!==0?e:e&1073741824?1073741824:0}function Qp(){var e=Fi;return Fi<<=1,!(Fi&4194240)&&(Fi=64),e}function el(e){for(var t=[],n=0;31>n;n++)t.push(e);return t}function mi(e,t,n){e.pendingLanes|=t,t!==536870912&&(e.suspendedLanes=0,e.pingedLanes=0),e=e.eventTimes,t=31-wt(t),e[t]=n}function Iv(e,t){var n=e.pendingLanes&~t;e.pendingLanes=t,e.suspendedLanes=0,e.pingedLanes=0,e.expiredLanes&=t,e.mutableReadLanes&=t,e.entangledLanes&=t,t=e.entanglements;var r=e.eventTimes;for(e=e.expirationTimes;0<n;){var s=31-wt(n),i=1<<s;t[s]=0,r[s]=-1,e[s]=-1,n&=~i}}function nc(e,t){var n=e.entangledLanes|=t;for(e=e.entanglements;n;){var r=31-wt(n),s=1<<r;s&t|e[r]&t&&(e[r]|=t),n&=~s}}var re=0;function Zp(e){return e&=-e,1<e?4<e?e&268435455?16:536870912:4:1}var em,rc,tm,nm,rm,tu=!1,Vi=[],hn=null,pn=null,mn=null,qs=new Map,Ks=new Map,an=[],Tv="mousedown mouseup touchcancel touchend touchstart auxclick dblclick pointercancel pointerdown pointerup dragend dragstart drop compositionend compositionstart keydown keypress keyup input textInput copy cut paste click change contextmenu reset submit".split(" ");function df(e,t){switch(e){case"focusin":case"focusout":hn=null;break;case"dragenter":case"dragleave":pn=null;break;case"mouseover":case"mouseout":mn=null;break;case"pointerover":case"pointerout":qs.delete(t.pointerId);break;case"gotpointercapture":case"lostpointercapture":Ks.delete(t.pointerId)}}function ys(e,t,n,r,s,i){return e===null||e.nativeEvent!==i?(e={blockedOn:t,domEventName:n,eventSystemFlags:r,nativeEvent:i,targetContainers:[s]},t!==null&&(t=yi(t),t!==null&&rc(t)),e):(e.eventSystemFlags|=r,t=e.targetContainers,s!==null&&t.indexOf(s)===-1&&t.push(s),e)}function Nv(e,t,n,r,s){switch(t){case"focusin":return hn=ys(hn,e,t,n,r,s),!0;case"dragenter":return pn=ys(pn,e,t,n,r,s),!0;case"mouseover":return mn=ys(mn,e,t,n,r,s),!0;case"pointerover":var i=s.pointerId;return qs.set(i,ys(qs.get(i)||null,e,t,n,r,s)),!0;case"gotpointercapture":return i=s.pointerId,Ks.set(i,ys(Ks.get(i)||null,e,t,n,r,s)),!0}return!1}function sm(e){var t=zn(e.target);if(t!==null){var n=ir(t);if(n!==null){if(t=n.tag,t===13){if(t=Wp(n),t!==null){e.blockedOn=t,rm(e.priority,function(){tm(n)});return}}else if(t===3&&n.stateNode.current.memoizedState.isDehydrated){e.blockedOn=n.tag===3?n.stateNode.containerInfo:null;return}}}e.blockedOn=null}function lo(e){if(e.blockedOn!==null)return!1;for(var t=e.targetContainers;0<t.length;){var n=nu(e.domEventName,e.eventSystemFlags,t[0],e.nativeEvent);if(n===null){n=e.nativeEvent;var r=new n.constructor(n.type,n);Jl=r,n.target.dispatchEvent(r),Jl=null}else return t=yi(n),t!==null&&rc(t),e.blockedOn=n,!1;t.shift()}return!0}function ff(e,t,n){lo(e)&&n.delete(t)}function jv(){tu=!1,hn!==null&&lo(hn)&&(hn=null),pn!==null&&lo(pn)&&(pn=null),mn!==null&&lo(mn)&&(mn=null),qs.forEach(ff),Ks.forEach(ff)}function vs(e,t){e.blockedOn===t&&(e.blockedOn=null,tu||(tu=!0,nt.unstable_scheduleCallback(nt.unstable_NormalPriority,jv)))}function Xs(e){function t(s){return vs(s,e)}if(0<Vi.length){vs(Vi[0],e);for(var n=1;n<Vi.length;n++){var r=Vi[n];r.blockedOn===e&&(r.blockedOn=null)}}for(hn!==null&&vs(hn,e),pn!==null&&vs(pn,e),mn!==null&&vs(mn,e),qs.forEach(t),Ks.forEach(t),n=0;n<an.length;n++)r=an[n],r.blockedOn===e&&(r.blockedOn=null);for(;0<an.length&&(n=an[0],n.blockedOn===null);)sm(n),n.blockedOn===null&&an.shift()}var jr=qt.ReactCurrentBatchConfig,To=!0;function Av(e,t,n,r){var s=re,i=jr.transition;jr.transition=null;try{re=1,sc(e,t,n,r)}finally{re=s,jr.transition=i}}function Rv(e,t,n,r){var s=re,i=jr.transition;jr.transition=null;try{re=4,sc(e,t,n,r)}finally{re=s,jr.transition=i}}function sc(e,t,n,r){if(To){var s=nu(e,t,n,r);if(s===null)cl(e,t,r,No,n),df(e,r);else if(Nv(s,e,t,n,r))r.stopPropagation();else if(df(e,r),t&4&&-1<Tv.indexOf(e)){for(;s!==null;){var i=yi(s);if(i!==null&&em(i),i=nu(e,t,n,r),i===null&&cl(e,t,r,No,n),i===s)break;s=i}s!==null&&r.stopPropagation()}else cl(e,t,r,null,n)}}var No=null;function nu(e,t,n,r){if(No=null,e=ec(r),e=zn(e),e!==null)if(t=ir(e),t===null)e=null;else if(n=t.tag,n===13){if(e=Wp(t),e!==null)return e;e=null}else if(n===3){if(t.stateNode.current.memoizedState.isDehydrated)return t.tag===3?t.stateNode.containerInfo:null;e=null}else t!==e&&(e=null);return No=e,null}function im(e){switch(e){case"cancel":case"click":case"close":case"contextmenu":case"copy":case"cut":case"auxclick":case"dblclick":case"dragend":case"dragstart":case"drop":case"focusin":case"focusout":case"input":case"invalid":case"keydown":case"keypress":case"keyup":case"mousedown":case"mouseup":case"paste":case"pause":case"play":case"pointercancel":case"pointerdown":case"pointerup":case"ratechange":case"reset":case"resize":case"seeked":case"submit":case"touchcancel":case"touchend":case"touchstart":case"volumechange":case"change":case"selectionchange":case"textInput":case"compositionstart":case"compositionend":case"compositionupdate":case"beforeblur":case"afterblur":case"beforeinput":case"blur":case"fullscreenchange":case"focus":case"hashchange":case"popstate":case"select":case"selectstart":return 1;case"drag":case"dragenter":case"dragexit":case"dragleave":case"dragover":case"mousemove":case"mouseout":case"mouseover":case"pointermove":case"pointerout":case"pointerover":case"scroll":case"toggle":case"touchmove":case"wheel":case"mouseenter":case"mouseleave":case"pointerenter":case"pointerleave":return 4;case"message":switch(wv()){case tc:return 1;case Jp:return 4;case Co:case xv:return 16;case Yp:return 536870912;default:return 16}default:return 16}}var un=null,ic=null,uo=null;function om(){if(uo)return uo;var e,t=ic,n=t.length,r,s="value"in un?un.value:un.textContent,i=s.length;for(e=0;e<n&&t[e]===s[e];e++);var a=n-e;for(r=1;r<=a&&t[n-r]===s[i-r];r++);return uo=s.slice(e,1<r?1-r:void 0)}function co(e){var t=e.keyCode;return"charCode"in e?(e=e.charCode,e===0&&t===13&&(e=13)):e=t,e===10&&(e=13),32<=e||e===13?e:0}function $i(){return!0}function hf(){return!1}function st(e){function t(n,r,s,i,a){this._reactName=n,this._targetInst=s,this.type=r,this.nativeEvent=i,this.target=a,this.currentTarget=null;for(var u in e)e.hasOwnProperty(u)&&(n=e[u],this[u]=n?n(i):i[u]);return this.isDefaultPrevented=(i.defaultPrevented!=null?i.defaultPrevented:i.returnValue===!1)?$i:hf,this.isPropagationStopped=hf,this}return pe(t.prototype,{preventDefault:function(){this.defaultPrevented=!0;var n=this.nativeEvent;n&&(n.preventDefault?n.preventDefault():typeof n.returnValue!="unknown"&&(n.returnValue=!1),this.isDefaultPrevented=$i)},stopPropagation:function(){var n=this.nativeEvent;n&&(n.stopPropagation?n.stopPropagation():typeof n.cancelBubble!="unknown"&&(n.cancelBubble=!0),this.isPropagationStopped=$i)},persist:function(){},isPersistent:$i}),t}var Kr={eventPhase:0,bubbles:0,cancelable:0,timeStamp:function(e){return e.timeStamp||Date.now()},defaultPrevented:0,isTrusted:0},oc=st(Kr),gi=pe({},Kr,{view:0,detail:0}),Pv=st(gi),tl,nl,ws,sa=pe({},gi,{screenX:0,screenY:0,clientX:0,clientY:0,pageX:0,pageY:0,ctrlKey:0,shiftKey:0,altKey:0,metaKey:0,getModifierState:ac,button:0,buttons:0,relatedTarget:function(e){return e.relatedTarget===void 0?e.fromElement===e.srcElement?e.toElement:e.fromElement:e.relatedTarget},movementX:function(e){return"movementX"in e?e.movementX:(e!==ws&&(ws&&e.type==="mousemove"?(tl=e.screenX-ws.screenX,nl=e.screenY-ws.screenY):nl=tl=0,ws=e),tl)},movementY:function(e){return"movementY"in e?e.movementY:nl}}),pf=st(sa),Ov=pe({},sa,{dataTransfer:0}),Dv=st(Ov),Mv=pe({},gi,{relatedTarget:0}),rl=st(Mv),Lv=pe({},Kr,{animationName:0,elapsedTime:0,pseudoElement:0}),Uv=st(Lv),Bv=pe({},Kr,{clipboardData:function(e){return"clipboardData"in e?e.clipboardData:window.clipboardData}}),Fv=st(Bv),zv=pe({},Kr,{data:0}),mf=st(zv),Vv={Esc:"Escape",Spacebar:" ",Left:"ArrowLeft",Up:"ArrowUp",Right:"ArrowRight",Down:"ArrowDown",Del:"Delete",Win:"OS",Menu:"ContextMenu",Apps:"ContextMenu",Scroll:"ScrollLock",MozPrintableKey:"Unidentified"},$v={8:"Backspace",9:"Tab",12:"Clear",13:"Enter",16:"Shift",17:"Control",18:"Alt",19:"Pause",20:"CapsLock",27:"Escape",32:" ",33:"PageUp",34:"PageDown",35:"End",36:"Home",37:"ArrowLeft",38:"ArrowUp",39:"ArrowRight",40:"ArrowDown",45:"Insert",46:"Delete",112:"F1",113:"F2",114:"F3",115:"F4",116:"F5",117:"F6",118:"F7",119:"F8",120:"F9",121:"F10",122:"F11",123:"F12",144:"NumLock",145:"ScrollLock",224:"Meta"},Hv={Alt:"altKey",Control:"ctrlKey",Meta:"metaKey",Shift:"shiftKey"};function Gv(e){var t=this.nativeEvent;return t.getModifierState?t.getModifierState(e):(e=Hv[e])?!!t[e]:!1}function ac(){return Gv}var Wv=pe({},gi,{key:function(e){if(e.key){var t=Vv[e.key]||e.key;if(t!=="Unidentified")return t}return e.type==="keypress"?(e=co(e),e===13?"Enter":String.fromCharCode(e)):e.type==="keydown"||e.type==="keyup"?$v[e.keyCode]||"Unidentified":""},code:0,location:0,ctrlKey:0,shiftKey:0,altKey:0,metaKey:0,repeat:0,locale:0,getModifierState:ac,charCode:function(e){return e.type==="keypress"?co(e):0},keyCode:function(e){return e.type==="keydown"||e.type==="keyup"?e.keyCode:0},which:function(e){return e.type==="keypress"?co(e):e.type==="keydown"||e.type==="keyup"?e.keyCode:0}}),qv=st(Wv),Kv=pe({},sa,{pointerId:0,width:0,height:0,pressure:0,tangentialPressure:0,tiltX:0,tiltY:0,twist:0,pointerType:0,isPrimary:0}),gf=st(Kv),Xv=pe({},gi,{touches:0,targetTouches:0,changedTouches:0,altKey:0,metaKey:0,ctrlKey:0,shiftKey:0,getModifierState:ac}),Jv=st(Xv),Yv=pe({},Kr,{propertyName:0,elapsedTime:0,pseudoElement:0}),Qv=st(Yv),Zv=pe({},sa,{deltaX:function(e){return"deltaX"in e?e.deltaX:"wheelDeltaX"in e?-e.wheelDeltaX:0},deltaY:function(e){return"deltaY"in e?e.deltaY:"wheelDeltaY"in e?-e.wheelDeltaY:"wheelDelta"in e?-e.wheelDelta:0},deltaZ:0,deltaMode:0}),ew=st(Zv),tw=[9,13,27,32],lc=Vt&&"CompositionEvent"in window,js=null;Vt&&"documentMode"in document&&(js=document.documentMode);var nw=Vt&&"TextEvent"in window&&!js,am=Vt&&(!lc||js&&8<js&&11>=js),yf=" ",vf=!1;function lm(e,t){switch(e){case"keyup":return tw.indexOf(t.keyCode)!==-1;case"keydown":return t.keyCode!==229;case"keypress":case"mousedown":case"focusout":return!0;default:return!1}}function um(e){return e=e.detail,typeof e=="object"&&"data"in e?e.data:null}var mr=!1;function rw(e,t){switch(e){case"compositionend":return um(t);case"keypress":return t.which!==32?null:(vf=!0,yf);case"textInput":return e=t.data,e===yf&&vf?null:e;default:return null}}function sw(e,t){if(mr)return e==="compositionend"||!lc&&lm(e,t)?(e=om(),uo=ic=un=null,mr=!1,e):null;switch(e){case"paste":return null;case"keypress":if(!(t.ctrlKey||t.altKey||t.metaKey)||t.ctrlKey&&t.altKey){if(t.char&&1<t.char.length)return t.char;if(t.which)return String.fromCharCode(t.which)}return null;case"compositionend":return am&&t.locale!=="ko"?null:t.data;default:return null}}var iw={color:!0,date:!0,datetime:!0,"datetime-local":!0,email:!0,month:!0,number:!0,password:!0,range:!0,search:!0,tel:!0,text:!0,time:!0,url:!0,week:!0};function wf(e){var t=e&&e.nodeName&&e.nodeName.toLowerCase();return t==="input"?!!iw[e.type]:t==="textarea"}function cm(e,t,n,r){zp(r),t=jo(t,"onChange"),0<t.length&&(n=new oc("onChange","change",null,n,r),e.push({event:n,listeners:t}))}var As=null,Js=null;function ow(e){Sm(e,0)}function ia(e){var t=vr(e);if(Op(t))return e}function aw(e,t){if(e==="change")return t}var dm=!1;if(Vt){var sl;if(Vt){var il="oninput"in document;if(!il){var xf=document.createElement("div");xf.setAttribute("oninput","return;"),il=typeof xf.oninput=="function"}sl=il}else sl=!1;dm=sl&&(!document.documentMode||9<document.documentMode)}function Sf(){As&&(As.detachEvent("onpropertychange",fm),Js=As=null)}function fm(e){if(e.propertyName==="value"&&ia(Js)){var t=[];cm(t,Js,e,ec(e)),Gp(ow,t)}}function lw(e,t,n){e==="focusin"?(Sf(),As=t,Js=n,As.attachEvent("onpropertychange",fm)):e==="focusout"&&Sf()}function uw(e){if(e==="selectionchange"||e==="keyup"||e==="keydown")return ia(Js)}function cw(e,t){if(e==="click")return ia(t)}function dw(e,t){if(e==="input"||e==="change")return ia(t)}function fw(e,t){return e===t&&(e!==0||1/e===1/t)||e!==e&&t!==t}var Et=typeof Object.is=="function"?Object.is:fw;function Ys(e,t){if(Et(e,t))return!0;if(typeof e!="object"||e===null||typeof t!="object"||t===null)return!1;var n=Object.keys(e),r=Object.keys(t);if(n.length!==r.length)return!1;for(r=0;r<n.length;r++){var s=n[r];if(!Ul.call(t,s)||!Et(e[s],t[s]))return!1}return!0}function Ef(e){for(;e&&e.firstChild;)e=e.firstChild;return e}function kf(e,t){var n=Ef(e);e=0;for(var r;n;){if(n.nodeType===3){if(r=e+n.textContent.length,e<=t&&r>=t)return{node:n,offset:t-e};e=r}e:{for(;n;){if(n.nextSibling){n=n.nextSibling;break e}n=n.parentNode}n=void 0}n=Ef(n)}}function hm(e,t){return e&&t?e===t?!0:e&&e.nodeType===3?!1:t&&t.nodeType===3?hm(e,t.parentNode):"contains"in e?e.contains(t):e.compareDocumentPosition?!!(e.compareDocumentPosition(t)&16):!1:!1}function pm(){for(var e=window,t=ko();t instanceof e.HTMLIFrameElement;){try{var n=typeof t.contentWindow.location.href=="string"}catch{n=!1}if(n)e=t.contentWindow;else break;t=ko(e.document)}return t}function uc(e){var t=e&&e.nodeName&&e.nodeName.toLowerCase();return t&&(t==="input"&&(e.type==="text"||e.type==="search"||e.type==="tel"||e.type==="url"||e.type==="password")||t==="textarea"||e.contentEditable==="true")}function hw(e){var t=pm(),n=e.focusedElem,r=e.selectionRange;if(t!==n&&n&&n.ownerDocument&&hm(n.ownerDocument.documentElement,n)){if(r!==null&&uc(n)){if(t=r.start,e=r.end,e===void 0&&(e=t),"selectionStart"in n)n.selectionStart=t,n.selectionEnd=Math.min(e,n.value.length);else if(e=(t=n.ownerDocument||document)&&t.defaultView||window,e.getSelection){e=e.getSelection();var s=n.textContent.length,i=Math.min(r.start,s);r=r.end===void 0?i:Math.min(r.end,s),!e.extend&&i>r&&(s=r,r=i,i=s),s=kf(n,i);var a=kf(n,r);s&&a&&(e.rangeCount!==1||e.anchorNode!==s.node||e.anchorOffset!==s.offset||e.focusNode!==a.node||e.focusOffset!==a.offset)&&(t=t.createRange(),t.setStart(s.node,s.offset),e.removeAllRanges(),i>r?(e.addRange(t),e.extend(a.node,a.offset)):(t.setEnd(a.node,a.offset),e.addRange(t)))}}for(t=[],e=n;e=e.parentNode;)e.nodeType===1&&t.push({element:e,left:e.scrollLeft,top:e.scrollTop});for(typeof n.focus=="function"&&n.focus(),n=0;n<t.length;n++)e=t[n],e.element.scrollLeft=e.left,e.element.scrollTop=e.top}}var pw=Vt&&"documentMode"in document&&11>=document.documentMode,gr=null,ru=null,Rs=null,su=!1;function bf(e,t,n){var r=n.window===n?n.document:n.nodeType===9?n:n.ownerDocument;su||gr==null||gr!==ko(r)||(r=gr,"selectionStart"in r&&uc(r)?r={start:r.selectionStart,end:r.selectionEnd}:(r=(r.ownerDocument&&r.ownerDocument.defaultView||window).getSelection(),r={anchorNode:r.anchorNode,anchorOffset:r.anchorOffset,focusNode:r.focusNode,focusOffset:r.focusOffset}),Rs&&Ys(Rs,r)||(Rs=r,r=jo(ru,"onSelect"),0<r.length&&(t=new oc("onSelect","select",null,t,n),e.push({event:t,listeners:r}),t.target=gr)))}function Hi(e,t){var n={};return n[e.toLowerCase()]=t.toLowerCase(),n["Webkit"+e]="webkit"+t,n["Moz"+e]="moz"+t,n}var yr={animationend:Hi("Animation","AnimationEnd"),animationiteration:Hi("Animation","AnimationIteration"),animationstart:Hi("Animation","AnimationStart"),transitionend:Hi("Transition","TransitionEnd")},ol={},mm={};Vt&&(mm=document.createElement("div").style,"AnimationEvent"in window||(delete yr.animationend.animation,delete yr.animationiteration.animation,delete yr.animationstart.animation),"TransitionEvent"in window||delete yr.transitionend.transition);function oa(e){if(ol[e])return ol[e];if(!yr[e])return e;var t=yr[e],n;for(n in t)if(t.hasOwnProperty(n)&&n in mm)return ol[e]=t[n];return e}var gm=oa("animationend"),ym=oa("animationiteration"),vm=oa("animationstart"),wm=oa("transitionend"),xm=new Map,_f="abort auxClick cancel canPlay canPlayThrough click close contextMenu copy cut drag dragEnd dragEnter dragExit dragLeave dragOver dragStart drop durationChange emptied encrypted ended error gotPointerCapture input invalid keyDown keyPress keyUp load loadedData loadedMetadata loadStart lostPointerCapture mouseDown mouseMove mouseOut mouseOver mouseUp paste pause play playing pointerCancel pointerDown pointerMove pointerOut pointerOver pointerUp progress rateChange reset resize seeked seeking stalled submit suspend timeUpdate touchCancel touchEnd touchStart volumeChange scroll toggle touchMove waiting wheel".split(" ");function Tn(e,t){xm.set(e,t),sr(t,[e])}for(var al=0;al<_f.length;al++){var ll=_f[al],mw=ll.toLowerCase(),gw=ll[0].toUpperCase()+ll.slice(1);Tn(mw,"on"+gw)}Tn(gm,"onAnimationEnd");Tn(ym,"onAnimationIteration");Tn(vm,"onAnimationStart");Tn("dblclick","onDoubleClick");Tn("focusin","onFocus");Tn("focusout","onBlur");Tn(wm,"onTransitionEnd");Ur("onMouseEnter",["mouseout","mouseover"]);Ur("onMouseLeave",["mouseout","mouseover"]);Ur("onPointerEnter",["pointerout","pointerover"]);Ur("onPointerLeave",["pointerout","pointerover"]);sr("onChange","change click focusin focusout input keydown keyup selectionchange".split(" "));sr("onSelect","focusout contextmenu dragend focusin keydown keyup mousedown mouseup selectionchange".split(" "));sr("onBeforeInput",["compositionend","keypress","textInput","paste"]);sr("onCompositionEnd","compositionend focusout keydown keypress keyup mousedown".split(" "));sr("onCompositionStart","compositionstart focusout keydown keypress keyup mousedown".split(" "));sr("onCompositionUpdate","compositionupdate focusout keydown keypress keyup mousedown".split(" "));var Cs="abort canplay canplaythrough durationchange emptied encrypted ended error loadeddata loadedmetadata loadstart pause play playing progress ratechange resize seeked seeking stalled suspend timeupdate volumechange waiting".split(" "),yw=new Set("cancel close invalid load scroll toggle".split(" ").concat(Cs));function Cf(e,t,n){var r=e.type||"unknown-event";e.currentTarget=n,mv(r,t,void 0,e),e.currentTarget=null}function Sm(e,t){t=(t&4)!==0;for(var n=0;n<e.length;n++){var r=e[n],s=r.event;r=r.listeners;e:{var i=void 0;if(t)for(var a=r.length-1;0<=a;a--){var u=r[a],c=u.instance,f=u.currentTarget;if(u=u.listener,c!==i&&s.isPropagationStopped())break e;Cf(s,u,f),i=c}else for(a=0;a<r.length;a++){if(u=r[a],c=u.instance,f=u.currentTarget,u=u.listener,c!==i&&s.isPropagationStopped())break e;Cf(s,u,f),i=c}}}if(_o)throw e=Zl,_o=!1,Zl=null,e}function ue(e,t){var n=t[uu];n===void 0&&(n=t[uu]=new Set);var r=e+"__bubble";n.has(r)||(Em(t,e,2,!1),n.add(r))}function ul(e,t,n){var r=0;t&&(r|=4),Em(n,e,r,t)}var Gi="_reactListening"+Math.random().toString(36).slice(2);function Qs(e){if(!e[Gi]){e[Gi]=!0,Np.forEach(function(n){n!=="selectionchange"&&(yw.has(n)||ul(n,!1,e),ul(n,!0,e))});var t=e.nodeType===9?e:e.ownerDocument;t===null||t[Gi]||(t[Gi]=!0,ul("selectionchange",!1,t))}}function Em(e,t,n,r){switch(im(t)){case 1:var s=Av;break;case 4:s=Rv;break;default:s=sc}n=s.bind(null,t,n,e),s=void 0,!Ql||t!=="touchstart"&&t!=="touchmove"&&t!=="wheel"||(s=!0),r?s!==void 0?e.addEventListener(t,n,{capture:!0,passive:s}):e.addEventListener(t,n,!0):s!==void 0?e.addEventListener(t,n,{passive:s}):e.addEventListener(t,n,!1)}function cl(e,t,n,r,s){var i=r;if(!(t&1)&&!(t&2)&&r!==null)e:for(;;){if(r===null)return;var a=r.tag;if(a===3||a===4){var u=r.stateNode.containerInfo;if(u===s||u.nodeType===8&&u.parentNode===s)break;if(a===4)for(a=r.return;a!==null;){var c=a.tag;if((c===3||c===4)&&(c=a.stateNode.containerInfo,c===s||c.nodeType===8&&c.parentNode===s))return;a=a.return}for(;u!==null;){if(a=zn(u),a===null)return;if(c=a.tag,c===5||c===6){r=i=a;continue e}u=u.parentNode}}r=r.return}Gp(function(){var f=i,E=ec(n),w=[];e:{var _=xm.get(e);if(_!==void 0){var A=oc,R=e;switch(e){case"keypress":if(co(n)===0)break e;case"keydown":case"keyup":A=qv;break;case"focusin":R="focus",A=rl;break;case"focusout":R="blur",A=rl;break;case"beforeblur":case"afterblur":A=rl;break;case"click":if(n.button===2)break e;case"auxclick":case"dblclick":case"mousedown":case"mousemove":case"mouseup":case"mouseout":case"mouseover":case"contextmenu":A=pf;break;case"drag":case"dragend":case"dragenter":case"dragexit":case"dragleave":case"dragover":case"dragstart":case"drop":A=Dv;break;case"touchcancel":case"touchend":case"touchmove":case"touchstart":A=Jv;break;case gm:case ym:case vm:A=Uv;break;case wm:A=Qv;break;case"scroll":A=Pv;break;case"wheel":A=ew;break;case"copy":case"cut":case"paste":A=Fv;break;case"gotpointercapture":case"lostpointercapture":case"pointercancel":case"pointerdown":case"pointermove":case"pointerout":case"pointerover":case"pointerup":A=gf}var j=(t&4)!==0,D=!j&&e==="scroll",b=j?_!==null?_+"Capture":null:_;j=[];for(var x=f,C;x!==null;){C=x;var O=C.stateNode;if(C.tag===5&&O!==null&&(C=O,b!==null&&(O=Ws(x,b),O!=null&&j.push(Zs(x,O,C)))),D)break;x=x.return}0<j.length&&(_=new A(_,R,null,n,E),w.push({event:_,listeners:j}))}}if(!(t&7)){e:{if(_=e==="mouseover"||e==="pointerover",A=e==="mouseout"||e==="pointerout",_&&n!==Jl&&(R=n.relatedTarget||n.fromElement)&&(zn(R)||R[$t]))break e;if((A||_)&&(_=E.window===E?E:(_=E.ownerDocument)?_.defaultView||_.parentWindow:window,A?(R=n.relatedTarget||n.toElement,A=f,R=R?zn(R):null,R!==null&&(D=ir(R),R!==D||R.tag!==5&&R.tag!==6)&&(R=null)):(A=null,R=f),A!==R)){if(j=pf,O="onMouseLeave",b="onMouseEnter",x="mouse",(e==="pointerout"||e==="pointerover")&&(j=gf,O="onPointerLeave",b="onPointerEnter",x="pointer"),D=A==null?_:vr(A),C=R==null?_:vr(R),_=new j(O,x+"leave",A,n,E),_.target=D,_.relatedTarget=C,O=null,zn(E)===f&&(j=new j(b,x+"enter",R,n,E),j.target=C,j.relatedTarget=D,O=j),D=O,A&&R)t:{for(j=A,b=R,x=0,C=j;C;C=cr(C))x++;for(C=0,O=b;O;O=cr(O))C++;for(;0<x-C;)j=cr(j),x--;for(;0<C-x;)b=cr(b),C--;for(;x--;){if(j===b||b!==null&&j===b.alternate)break t;j=cr(j),b=cr(b)}j=null}else j=null;A!==null&&If(w,_,A,j,!1),R!==null&&D!==null&&If(w,D,R,j,!0)}}e:{if(_=f?vr(f):window,A=_.nodeName&&_.nodeName.toLowerCase(),A==="select"||A==="input"&&_.type==="file")var M=aw;else if(wf(_))if(dm)M=dw;else{M=uw;var U=lw}else(A=_.nodeName)&&A.toLowerCase()==="input"&&(_.type==="checkbox"||_.type==="radio")&&(M=cw);if(M&&(M=M(e,f))){cm(w,M,n,E);break e}U&&U(e,_,f),e==="focusout"&&(U=_._wrapperState)&&U.controlled&&_.type==="number"&&Gl(_,"number",_.value)}switch(U=f?vr(f):window,e){case"focusin":(wf(U)||U.contentEditable==="true")&&(gr=U,ru=f,Rs=null);break;case"focusout":Rs=ru=gr=null;break;case"mousedown":su=!0;break;case"contextmenu":case"mouseup":case"dragend":su=!1,bf(w,n,E);break;case"selectionchange":if(pw)break;case"keydown":case"keyup":bf(w,n,E)}var g;if(lc)e:{switch(e){case"compositionstart":var p="onCompositionStart";break e;case"compositionend":p="onCompositionEnd";break e;case"compositionupdate":p="onCompositionUpdate";break e}p=void 0}else mr?lm(e,n)&&(p="onCompositionEnd"):e==="keydown"&&n.keyCode===229&&(p="onCompositionStart");p&&(am&&n.locale!=="ko"&&(mr||p!=="onCompositionStart"?p==="onCompositionEnd"&&mr&&(g=om()):(un=E,ic="value"in un?un.value:un.textContent,mr=!0)),U=jo(f,p),0<U.length&&(p=new mf(p,e,null,n,E),w.push({event:p,listeners:U}),g?p.data=g:(g=um(n),g!==null&&(p.data=g)))),(g=nw?rw(e,n):sw(e,n))&&(f=jo(f,"onBeforeInput"),0<f.length&&(E=new mf("onBeforeInput","beforeinput",null,n,E),w.push({event:E,listeners:f}),E.data=g))}Sm(w,t)})}function Zs(e,t,n){return{instance:e,listener:t,currentTarget:n}}function jo(e,t){for(var n=t+"Capture",r=[];e!==null;){var s=e,i=s.stateNode;s.tag===5&&i!==null&&(s=i,i=Ws(e,n),i!=null&&r.unshift(Zs(e,i,s)),i=Ws(e,t),i!=null&&r.push(Zs(e,i,s))),e=e.return}return r}function cr(e){if(e===null)return null;do e=e.return;while(e&&e.tag!==5);return e||null}function If(e,t,n,r,s){for(var i=t._reactName,a=[];n!==null&&n!==r;){var u=n,c=u.alternate,f=u.stateNode;if(c!==null&&c===r)break;u.tag===5&&f!==null&&(u=f,s?(c=Ws(n,i),c!=null&&a.unshift(Zs(n,c,u))):s||(c=Ws(n,i),c!=null&&a.push(Zs(n,c,u)))),n=n.return}a.length!==0&&e.push({event:t,listeners:a})}var vw=/\r\n?/g,ww=/\u0000|\uFFFD/g;function Tf(e){return(typeof e=="string"?e:""+e).replace(vw,`
`).replace(ww,"")}function Wi(e,t,n){if(t=Tf(t),Tf(e)!==t&&n)throw Error(L(425))}function Ao(){}var iu=null,ou=null;function au(e,t){return e==="textarea"||e==="noscript"||typeof t.children=="string"||typeof t.children=="number"||typeof t.dangerouslySetInnerHTML=="object"&&t.dangerouslySetInnerHTML!==null&&t.dangerouslySetInnerHTML.__html!=null}var lu=typeof setTimeout=="function"?setTimeout:void 0,xw=typeof clearTimeout=="function"?clearTimeout:void 0,Nf=typeof Promise=="function"?Promise:void 0,Sw=typeof queueMicrotask=="function"?queueMicrotask:typeof Nf<"u"?function(e){return Nf.resolve(null).then(e).catch(Ew)}:lu;function Ew(e){setTimeout(function(){throw e})}function dl(e,t){var n=t,r=0;do{var s=n.nextSibling;if(e.removeChild(n),s&&s.nodeType===8)if(n=s.data,n==="/$"){if(r===0){e.removeChild(s),Xs(t);return}r--}else n!=="$"&&n!=="$?"&&n!=="$!"||r++;n=s}while(n);Xs(t)}function gn(e){for(;e!=null;e=e.nextSibling){var t=e.nodeType;if(t===1||t===3)break;if(t===8){if(t=e.data,t==="$"||t==="$!"||t==="$?")break;if(t==="/$")return null}}return e}function jf(e){e=e.previousSibling;for(var t=0;e;){if(e.nodeType===8){var n=e.data;if(n==="$"||n==="$!"||n==="$?"){if(t===0)return e;t--}else n==="/$"&&t++}e=e.previousSibling}return null}var Xr=Math.random().toString(36).slice(2),jt="__reactFiber$"+Xr,ei="__reactProps$"+Xr,$t="__reactContainer$"+Xr,uu="__reactEvents$"+Xr,kw="__reactListeners$"+Xr,bw="__reactHandles$"+Xr;function zn(e){var t=e[jt];if(t)return t;for(var n=e.parentNode;n;){if(t=n[$t]||n[jt]){if(n=t.alternate,t.child!==null||n!==null&&n.child!==null)for(e=jf(e);e!==null;){if(n=e[jt])return n;e=jf(e)}return t}e=n,n=e.parentNode}return null}function yi(e){return e=e[jt]||e[$t],!e||e.tag!==5&&e.tag!==6&&e.tag!==13&&e.tag!==3?null:e}function vr(e){if(e.tag===5||e.tag===6)return e.stateNode;throw Error(L(33))}function aa(e){return e[ei]||null}var cu=[],wr=-1;function Nn(e){return{current:e}}function ce(e){0>wr||(e.current=cu[wr],cu[wr]=null,wr--)}function oe(e,t){wr++,cu[wr]=e.current,e.current=t}var Cn={},Me=Nn(Cn),Ke=Nn(!1),Yn=Cn;function Br(e,t){var n=e.type.contextTypes;if(!n)return Cn;var r=e.stateNode;if(r&&r.__reactInternalMemoizedUnmaskedChildContext===t)return r.__reactInternalMemoizedMaskedChildContext;var s={},i;for(i in n)s[i]=t[i];return r&&(e=e.stateNode,e.__reactInternalMemoizedUnmaskedChildContext=t,e.__reactInternalMemoizedMaskedChildContext=s),s}function Xe(e){return e=e.childContextTypes,e!=null}function Ro(){ce(Ke),ce(Me)}function Af(e,t,n){if(Me.current!==Cn)throw Error(L(168));oe(Me,t),oe(Ke,n)}function km(e,t,n){var r=e.stateNode;if(t=t.childContextTypes,typeof r.getChildContext!="function")return n;r=r.getChildContext();for(var s in r)if(!(s in t))throw Error(L(108,lv(e)||"Unknown",s));return pe({},n,r)}function Po(e){return e=(e=e.stateNode)&&e.__reactInternalMemoizedMergedChildContext||Cn,Yn=Me.current,oe(Me,e),oe(Ke,Ke.current),!0}function Rf(e,t,n){var r=e.stateNode;if(!r)throw Error(L(169));n?(e=km(e,t,Yn),r.__reactInternalMemoizedMergedChildContext=e,ce(Ke),ce(Me),oe(Me,e)):ce(Ke),oe(Ke,n)}var Lt=null,la=!1,fl=!1;function bm(e){Lt===null?Lt=[e]:Lt.push(e)}function _w(e){la=!0,bm(e)}function jn(){if(!fl&&Lt!==null){fl=!0;var e=0,t=re;try{var n=Lt;for(re=1;e<n.length;e++){var r=n[e];do r=r(!0);while(r!==null)}Lt=null,la=!1}catch(s){throw Lt!==null&&(Lt=Lt.slice(e+1)),Xp(tc,jn),s}finally{re=t,fl=!1}}return null}var xr=[],Sr=0,Oo=null,Do=0,ot=[],at=0,Qn=null,Bt=1,Ft="";function Ln(e,t){xr[Sr++]=Do,xr[Sr++]=Oo,Oo=e,Do=t}function _m(e,t,n){ot[at++]=Bt,ot[at++]=Ft,ot[at++]=Qn,Qn=e;var r=Bt;e=Ft;var s=32-wt(r)-1;r&=~(1<<s),n+=1;var i=32-wt(t)+s;if(30<i){var a=s-s%5;i=(r&(1<<a)-1).toString(32),r>>=a,s-=a,Bt=1<<32-wt(t)+s|n<<s|r,Ft=i+e}else Bt=1<<i|n<<s|r,Ft=e}function cc(e){e.return!==null&&(Ln(e,1),_m(e,1,0))}function dc(e){for(;e===Oo;)Oo=xr[--Sr],xr[Sr]=null,Do=xr[--Sr],xr[Sr]=null;for(;e===Qn;)Qn=ot[--at],ot[at]=null,Ft=ot[--at],ot[at]=null,Bt=ot[--at],ot[at]=null}var tt=null,et=null,de=!1,gt=null;function Cm(e,t){var n=lt(5,null,null,0);n.elementType="DELETED",n.stateNode=t,n.return=e,t=e.deletions,t===null?(e.deletions=[n],e.flags|=16):t.push(n)}function Pf(e,t){switch(e.tag){case 5:var n=e.type;return t=t.nodeType!==1||n.toLowerCase()!==t.nodeName.toLowerCase()?null:t,t!==null?(e.stateNode=t,tt=e,et=gn(t.firstChild),!0):!1;case 6:return t=e.pendingProps===""||t.nodeType!==3?null:t,t!==null?(e.stateNode=t,tt=e,et=null,!0):!1;case 13:return t=t.nodeType!==8?null:t,t!==null?(n=Qn!==null?{id:Bt,overflow:Ft}:null,e.memoizedState={dehydrated:t,treeContext:n,retryLane:1073741824},n=lt(18,null,null,0),n.stateNode=t,n.return=e,e.child=n,tt=e,et=null,!0):!1;default:return!1}}function du(e){return(e.mode&1)!==0&&(e.flags&128)===0}function fu(e){if(de){var t=et;if(t){var n=t;if(!Pf(e,t)){if(du(e))throw Error(L(418));t=gn(n.nextSibling);var r=tt;t&&Pf(e,t)?Cm(r,n):(e.flags=e.flags&-4097|2,de=!1,tt=e)}}else{if(du(e))throw Error(L(418));e.flags=e.flags&-4097|2,de=!1,tt=e}}}function Of(e){for(e=e.return;e!==null&&e.tag!==5&&e.tag!==3&&e.tag!==13;)e=e.return;tt=e}function qi(e){if(e!==tt)return!1;if(!de)return Of(e),de=!0,!1;var t;if((t=e.tag!==3)&&!(t=e.tag!==5)&&(t=e.type,t=t!=="head"&&t!=="body"&&!au(e.type,e.memoizedProps)),t&&(t=et)){if(du(e))throw Im(),Error(L(418));for(;t;)Cm(e,t),t=gn(t.nextSibling)}if(Of(e),e.tag===13){if(e=e.memoizedState,e=e!==null?e.dehydrated:null,!e)throw Error(L(317));e:{for(e=e.nextSibling,t=0;e;){if(e.nodeType===8){var n=e.data;if(n==="/$"){if(t===0){et=gn(e.nextSibling);break e}t--}else n!=="$"&&n!=="$!"&&n!=="$?"||t++}e=e.nextSibling}et=null}}else et=tt?gn(e.stateNode.nextSibling):null;return!0}function Im(){for(var e=et;e;)e=gn(e.nextSibling)}function Fr(){et=tt=null,de=!1}function fc(e){gt===null?gt=[e]:gt.push(e)}var Cw=qt.ReactCurrentBatchConfig;function xs(e,t,n){if(e=n.ref,e!==null&&typeof e!="function"&&typeof e!="object"){if(n._owner){if(n=n._owner,n){if(n.tag!==1)throw Error(L(309));var r=n.stateNode}if(!r)throw Error(L(147,e));var s=r,i=""+e;return t!==null&&t.ref!==null&&typeof t.ref=="function"&&t.ref._stringRef===i?t.ref:(t=function(a){var u=s.refs;a===null?delete u[i]:u[i]=a},t._stringRef=i,t)}if(typeof e!="string")throw Error(L(284));if(!n._owner)throw Error(L(290,e))}return e}function Ki(e,t){throw e=Object.prototype.toString.call(t),Error(L(31,e==="[object Object]"?"object with keys {"+Object.keys(t).join(", ")+"}":e))}function Df(e){var t=e._init;return t(e._payload)}function Tm(e){function t(b,x){if(e){var C=b.deletions;C===null?(b.deletions=[x],b.flags|=16):C.push(x)}}function n(b,x){if(!e)return null;for(;x!==null;)t(b,x),x=x.sibling;return null}function r(b,x){for(b=new Map;x!==null;)x.key!==null?b.set(x.key,x):b.set(x.index,x),x=x.sibling;return b}function s(b,x){return b=xn(b,x),b.index=0,b.sibling=null,b}function i(b,x,C){return b.index=C,e?(C=b.alternate,C!==null?(C=C.index,C<x?(b.flags|=2,x):C):(b.flags|=2,x)):(b.flags|=1048576,x)}function a(b){return e&&b.alternate===null&&(b.flags|=2),b}function u(b,x,C,O){return x===null||x.tag!==6?(x=wl(C,b.mode,O),x.return=b,x):(x=s(x,C),x.return=b,x)}function c(b,x,C,O){var M=C.type;return M===pr?E(b,x,C.props.children,O,C.key):x!==null&&(x.elementType===M||typeof M=="object"&&M!==null&&M.$$typeof===sn&&Df(M)===x.type)?(O=s(x,C.props),O.ref=xs(b,x,C),O.return=b,O):(O=vo(C.type,C.key,C.props,null,b.mode,O),O.ref=xs(b,x,C),O.return=b,O)}function f(b,x,C,O){return x===null||x.tag!==4||x.stateNode.containerInfo!==C.containerInfo||x.stateNode.implementation!==C.implementation?(x=xl(C,b.mode,O),x.return=b,x):(x=s(x,C.children||[]),x.return=b,x)}function E(b,x,C,O,M){return x===null||x.tag!==7?(x=qn(C,b.mode,O,M),x.return=b,x):(x=s(x,C),x.return=b,x)}function w(b,x,C){if(typeof x=="string"&&x!==""||typeof x=="number")return x=wl(""+x,b.mode,C),x.return=b,x;if(typeof x=="object"&&x!==null){switch(x.$$typeof){case Li:return C=vo(x.type,x.key,x.props,null,b.mode,C),C.ref=xs(b,null,x),C.return=b,C;case hr:return x=xl(x,b.mode,C),x.return=b,x;case sn:var O=x._init;return w(b,O(x._payload),C)}if(bs(x)||ms(x))return x=qn(x,b.mode,C,null),x.return=b,x;Ki(b,x)}return null}function _(b,x,C,O){var M=x!==null?x.key:null;if(typeof C=="string"&&C!==""||typeof C=="number")return M!==null?null:u(b,x,""+C,O);if(typeof C=="object"&&C!==null){switch(C.$$typeof){case Li:return C.key===M?c(b,x,C,O):null;case hr:return C.key===M?f(b,x,C,O):null;case sn:return M=C._init,_(b,x,M(C._payload),O)}if(bs(C)||ms(C))return M!==null?null:E(b,x,C,O,null);Ki(b,C)}return null}function A(b,x,C,O,M){if(typeof O=="string"&&O!==""||typeof O=="number")return b=b.get(C)||null,u(x,b,""+O,M);if(typeof O=="object"&&O!==null){switch(O.$$typeof){case Li:return b=b.get(O.key===null?C:O.key)||null,c(x,b,O,M);case hr:return b=b.get(O.key===null?C:O.key)||null,f(x,b,O,M);case sn:var U=O._init;return A(b,x,C,U(O._payload),M)}if(bs(O)||ms(O))return b=b.get(C)||null,E(x,b,O,M,null);Ki(x,O)}return null}function R(b,x,C,O){for(var M=null,U=null,g=x,p=x=0,v=null;g!==null&&p<C.length;p++){g.index>p?(v=g,g=null):v=g.sibling;var S=_(b,g,C[p],O);if(S===null){g===null&&(g=v);break}e&&g&&S.alternate===null&&t(b,g),x=i(S,x,p),U===null?M=S:U.sibling=S,U=S,g=v}if(p===C.length)return n(b,g),de&&Ln(b,p),M;if(g===null){for(;p<C.length;p++)g=w(b,C[p],O),g!==null&&(x=i(g,x,p),U===null?M=g:U.sibling=g,U=g);return de&&Ln(b,p),M}for(g=r(b,g);p<C.length;p++)v=A(g,b,p,C[p],O),v!==null&&(e&&v.alternate!==null&&g.delete(v.key===null?p:v.key),x=i(v,x,p),U===null?M=v:U.sibling=v,U=v);return e&&g.forEach(function(k){return t(b,k)}),de&&Ln(b,p),M}function j(b,x,C,O){var M=ms(C);if(typeof M!="function")throw Error(L(150));if(C=M.call(C),C==null)throw Error(L(151));for(var U=M=null,g=x,p=x=0,v=null,S=C.next();g!==null&&!S.done;p++,S=C.next()){g.index>p?(v=g,g=null):v=g.sibling;var k=_(b,g,S.value,O);if(k===null){g===null&&(g=v);break}e&&g&&k.alternate===null&&t(b,g),x=i(k,x,p),U===null?M=k:U.sibling=k,U=k,g=v}if(S.done)return n(b,g),de&&Ln(b,p),M;if(g===null){for(;!S.done;p++,S=C.next())S=w(b,S.value,O),S!==null&&(x=i(S,x,p),U===null?M=S:U.sibling=S,U=S);return de&&Ln(b,p),M}for(g=r(b,g);!S.done;p++,S=C.next())S=A(g,b,p,S.value,O),S!==null&&(e&&S.alternate!==null&&g.delete(S.key===null?p:S.key),x=i(S,x,p),U===null?M=S:U.sibling=S,U=S);return e&&g.forEach(function(I){return t(b,I)}),de&&Ln(b,p),M}function D(b,x,C,O){if(typeof C=="object"&&C!==null&&C.type===pr&&C.key===null&&(C=C.props.children),typeof C=="object"&&C!==null){switch(C.$$typeof){case Li:e:{for(var M=C.key,U=x;U!==null;){if(U.key===M){if(M=C.type,M===pr){if(U.tag===7){n(b,U.sibling),x=s(U,C.props.children),x.return=b,b=x;break e}}else if(U.elementType===M||typeof M=="object"&&M!==null&&M.$$typeof===sn&&Df(M)===U.type){n(b,U.sibling),x=s(U,C.props),x.ref=xs(b,U,C),x.return=b,b=x;break e}n(b,U);break}else t(b,U);U=U.sibling}C.type===pr?(x=qn(C.props.children,b.mode,O,C.key),x.return=b,b=x):(O=vo(C.type,C.key,C.props,null,b.mode,O),O.ref=xs(b,x,C),O.return=b,b=O)}return a(b);case hr:e:{for(U=C.key;x!==null;){if(x.key===U)if(x.tag===4&&x.stateNode.containerInfo===C.containerInfo&&x.stateNode.implementation===C.implementation){n(b,x.sibling),x=s(x,C.children||[]),x.return=b,b=x;break e}else{n(b,x);break}else t(b,x);x=x.sibling}x=xl(C,b.mode,O),x.return=b,b=x}return a(b);case sn:return U=C._init,D(b,x,U(C._payload),O)}if(bs(C))return R(b,x,C,O);if(ms(C))return j(b,x,C,O);Ki(b,C)}return typeof C=="string"&&C!==""||typeof C=="number"?(C=""+C,x!==null&&x.tag===6?(n(b,x.sibling),x=s(x,C),x.return=b,b=x):(n(b,x),x=wl(C,b.mode,O),x.return=b,b=x),a(b)):n(b,x)}return D}var zr=Tm(!0),Nm=Tm(!1),Mo=Nn(null),Lo=null,Er=null,hc=null;function pc(){hc=Er=Lo=null}function mc(e){var t=Mo.current;ce(Mo),e._currentValue=t}function hu(e,t,n){for(;e!==null;){var r=e.alternate;if((e.childLanes&t)!==t?(e.childLanes|=t,r!==null&&(r.childLanes|=t)):r!==null&&(r.childLanes&t)!==t&&(r.childLanes|=t),e===n)break;e=e.return}}function Ar(e,t){Lo=e,hc=Er=null,e=e.dependencies,e!==null&&e.firstContext!==null&&(e.lanes&t&&(qe=!0),e.firstContext=null)}function ct(e){var t=e._currentValue;if(hc!==e)if(e={context:e,memoizedValue:t,next:null},Er===null){if(Lo===null)throw Error(L(308));Er=e,Lo.dependencies={lanes:0,firstContext:e}}else Er=Er.next=e;return t}var Vn=null;function gc(e){Vn===null?Vn=[e]:Vn.push(e)}function jm(e,t,n,r){var s=t.interleaved;return s===null?(n.next=n,gc(t)):(n.next=s.next,s.next=n),t.interleaved=n,Ht(e,r)}function Ht(e,t){e.lanes|=t;var n=e.alternate;for(n!==null&&(n.lanes|=t),n=e,e=e.return;e!==null;)e.childLanes|=t,n=e.alternate,n!==null&&(n.childLanes|=t),n=e,e=e.return;return n.tag===3?n.stateNode:null}var on=!1;function yc(e){e.updateQueue={baseState:e.memoizedState,firstBaseUpdate:null,lastBaseUpdate:null,shared:{pending:null,interleaved:null,lanes:0},effects:null}}function Am(e,t){e=e.updateQueue,t.updateQueue===e&&(t.updateQueue={baseState:e.baseState,firstBaseUpdate:e.firstBaseUpdate,lastBaseUpdate:e.lastBaseUpdate,shared:e.shared,effects:e.effects})}function zt(e,t){return{eventTime:e,lane:t,tag:0,payload:null,callback:null,next:null}}function yn(e,t,n){var r=e.updateQueue;if(r===null)return null;if(r=r.shared,J&2){var s=r.pending;return s===null?t.next=t:(t.next=s.next,s.next=t),r.pending=t,Ht(e,n)}return s=r.interleaved,s===null?(t.next=t,gc(r)):(t.next=s.next,s.next=t),r.interleaved=t,Ht(e,n)}function fo(e,t,n){if(t=t.updateQueue,t!==null&&(t=t.shared,(n&4194240)!==0)){var r=t.lanes;r&=e.pendingLanes,n|=r,t.lanes=n,nc(e,n)}}function Mf(e,t){var n=e.updateQueue,r=e.alternate;if(r!==null&&(r=r.updateQueue,n===r)){var s=null,i=null;if(n=n.firstBaseUpdate,n!==null){do{var a={eventTime:n.eventTime,lane:n.lane,tag:n.tag,payload:n.payload,callback:n.callback,next:null};i===null?s=i=a:i=i.next=a,n=n.next}while(n!==null);i===null?s=i=t:i=i.next=t}else s=i=t;n={baseState:r.baseState,firstBaseUpdate:s,lastBaseUpdate:i,shared:r.shared,effects:r.effects},e.updateQueue=n;return}e=n.lastBaseUpdate,e===null?n.firstBaseUpdate=t:e.next=t,n.lastBaseUpdate=t}function Uo(e,t,n,r){var s=e.updateQueue;on=!1;var i=s.firstBaseUpdate,a=s.lastBaseUpdate,u=s.shared.pending;if(u!==null){s.shared.pending=null;var c=u,f=c.next;c.next=null,a===null?i=f:a.next=f,a=c;var E=e.alternate;E!==null&&(E=E.updateQueue,u=E.lastBaseUpdate,u!==a&&(u===null?E.firstBaseUpdate=f:u.next=f,E.lastBaseUpdate=c))}if(i!==null){var w=s.baseState;a=0,E=f=c=null,u=i;do{var _=u.lane,A=u.eventTime;if((r&_)===_){E!==null&&(E=E.next={eventTime:A,lane:0,tag:u.tag,payload:u.payload,callback:u.callback,next:null});e:{var R=e,j=u;switch(_=t,A=n,j.tag){case 1:if(R=j.payload,typeof R=="function"){w=R.call(A,w,_);break e}w=R;break e;case 3:R.flags=R.flags&-65537|128;case 0:if(R=j.payload,_=typeof R=="function"?R.call(A,w,_):R,_==null)break e;w=pe({},w,_);break e;case 2:on=!0}}u.callback!==null&&u.lane!==0&&(e.flags|=64,_=s.effects,_===null?s.effects=[u]:_.push(u))}else A={eventTime:A,lane:_,tag:u.tag,payload:u.payload,callback:u.callback,next:null},E===null?(f=E=A,c=w):E=E.next=A,a|=_;if(u=u.next,u===null){if(u=s.shared.pending,u===null)break;_=u,u=_.next,_.next=null,s.lastBaseUpdate=_,s.shared.pending=null}}while(!0);if(E===null&&(c=w),s.baseState=c,s.firstBaseUpdate=f,s.lastBaseUpdate=E,t=s.shared.interleaved,t!==null){s=t;do a|=s.lane,s=s.next;while(s!==t)}else i===null&&(s.shared.lanes=0);er|=a,e.lanes=a,e.memoizedState=w}}function Lf(e,t,n){if(e=t.effects,t.effects=null,e!==null)for(t=0;t<e.length;t++){var r=e[t],s=r.callback;if(s!==null){if(r.callback=null,r=n,typeof s!="function")throw Error(L(191,s));s.call(r)}}}var vi={},Ot=Nn(vi),ti=Nn(vi),ni=Nn(vi);function $n(e){if(e===vi)throw Error(L(174));return e}function vc(e,t){switch(oe(ni,t),oe(ti,e),oe(Ot,vi),e=t.nodeType,e){case 9:case 11:t=(t=t.documentElement)?t.namespaceURI:ql(null,"");break;default:e=e===8?t.parentNode:t,t=e.namespaceURI||null,e=e.tagName,t=ql(t,e)}ce(Ot),oe(Ot,t)}function Vr(){ce(Ot),ce(ti),ce(ni)}function Rm(e){$n(ni.current);var t=$n(Ot.current),n=ql(t,e.type);t!==n&&(oe(ti,e),oe(Ot,n))}function wc(e){ti.current===e&&(ce(Ot),ce(ti))}var fe=Nn(0);function Bo(e){for(var t=e;t!==null;){if(t.tag===13){var n=t.memoizedState;if(n!==null&&(n=n.dehydrated,n===null||n.data==="$?"||n.data==="$!"))return t}else if(t.tag===19&&t.memoizedProps.revealOrder!==void 0){if(t.flags&128)return t}else if(t.child!==null){t.child.return=t,t=t.child;continue}if(t===e)break;for(;t.sibling===null;){if(t.return===null||t.return===e)return null;t=t.return}t.sibling.return=t.return,t=t.sibling}return null}var hl=[];function xc(){for(var e=0;e<hl.length;e++)hl[e]._workInProgressVersionPrimary=null;hl.length=0}var ho=qt.ReactCurrentDispatcher,pl=qt.ReactCurrentBatchConfig,Zn=0,he=null,ke=null,Ie=null,Fo=!1,Ps=!1,ri=0,Iw=0;function Pe(){throw Error(L(321))}function Sc(e,t){if(t===null)return!1;for(var n=0;n<t.length&&n<e.length;n++)if(!Et(e[n],t[n]))return!1;return!0}function Ec(e,t,n,r,s,i){if(Zn=i,he=t,t.memoizedState=null,t.updateQueue=null,t.lanes=0,ho.current=e===null||e.memoizedState===null?Aw:Rw,e=n(r,s),Ps){i=0;do{if(Ps=!1,ri=0,25<=i)throw Error(L(301));i+=1,Ie=ke=null,t.updateQueue=null,ho.current=Pw,e=n(r,s)}while(Ps)}if(ho.current=zo,t=ke!==null&&ke.next!==null,Zn=0,Ie=ke=he=null,Fo=!1,t)throw Error(L(300));return e}function kc(){var e=ri!==0;return ri=0,e}function It(){var e={memoizedState:null,baseState:null,baseQueue:null,queue:null,next:null};return Ie===null?he.memoizedState=Ie=e:Ie=Ie.next=e,Ie}function dt(){if(ke===null){var e=he.alternate;e=e!==null?e.memoizedState:null}else e=ke.next;var t=Ie===null?he.memoizedState:Ie.next;if(t!==null)Ie=t,ke=e;else{if(e===null)throw Error(L(310));ke=e,e={memoizedState:ke.memoizedState,baseState:ke.baseState,baseQueue:ke.baseQueue,queue:ke.queue,next:null},Ie===null?he.memoizedState=Ie=e:Ie=Ie.next=e}return Ie}function si(e,t){return typeof t=="function"?t(e):t}function ml(e){var t=dt(),n=t.queue;if(n===null)throw Error(L(311));n.lastRenderedReducer=e;var r=ke,s=r.baseQueue,i=n.pending;if(i!==null){if(s!==null){var a=s.next;s.next=i.next,i.next=a}r.baseQueue=s=i,n.pending=null}if(s!==null){i=s.next,r=r.baseState;var u=a=null,c=null,f=i;do{var E=f.lane;if((Zn&E)===E)c!==null&&(c=c.next={lane:0,action:f.action,hasEagerState:f.hasEagerState,eagerState:f.eagerState,next:null}),r=f.hasEagerState?f.eagerState:e(r,f.action);else{var w={lane:E,action:f.action,hasEagerState:f.hasEagerState,eagerState:f.eagerState,next:null};c===null?(u=c=w,a=r):c=c.next=w,he.lanes|=E,er|=E}f=f.next}while(f!==null&&f!==i);c===null?a=r:c.next=u,Et(r,t.memoizedState)||(qe=!0),t.memoizedState=r,t.baseState=a,t.baseQueue=c,n.lastRenderedState=r}if(e=n.interleaved,e!==null){s=e;do i=s.lane,he.lanes|=i,er|=i,s=s.next;while(s!==e)}else s===null&&(n.lanes=0);return[t.memoizedState,n.dispatch]}function gl(e){var t=dt(),n=t.queue;if(n===null)throw Error(L(311));n.lastRenderedReducer=e;var r=n.dispatch,s=n.pending,i=t.memoizedState;if(s!==null){n.pending=null;var a=s=s.next;do i=e(i,a.action),a=a.next;while(a!==s);Et(i,t.memoizedState)||(qe=!0),t.memoizedState=i,t.baseQueue===null&&(t.baseState=i),n.lastRenderedState=i}return[i,r]}function Pm(){}function Om(e,t){var n=he,r=dt(),s=t(),i=!Et(r.memoizedState,s);if(i&&(r.memoizedState=s,qe=!0),r=r.queue,bc(Lm.bind(null,n,r,e),[e]),r.getSnapshot!==t||i||Ie!==null&&Ie.memoizedState.tag&1){if(n.flags|=2048,ii(9,Mm.bind(null,n,r,s,t),void 0,null),Te===null)throw Error(L(349));Zn&30||Dm(n,t,s)}return s}function Dm(e,t,n){e.flags|=16384,e={getSnapshot:t,value:n},t=he.updateQueue,t===null?(t={lastEffect:null,stores:null},he.updateQueue=t,t.stores=[e]):(n=t.stores,n===null?t.stores=[e]:n.push(e))}function Mm(e,t,n,r){t.value=n,t.getSnapshot=r,Um(t)&&Bm(e)}function Lm(e,t,n){return n(function(){Um(t)&&Bm(e)})}function Um(e){var t=e.getSnapshot;e=e.value;try{var n=t();return!Et(e,n)}catch{return!0}}function Bm(e){var t=Ht(e,1);t!==null&&xt(t,e,1,-1)}function Uf(e){var t=It();return typeof e=="function"&&(e=e()),t.memoizedState=t.baseState=e,e={pending:null,interleaved:null,lanes:0,dispatch:null,lastRenderedReducer:si,lastRenderedState:e},t.queue=e,e=e.dispatch=jw.bind(null,he,e),[t.memoizedState,e]}function ii(e,t,n,r){return e={tag:e,create:t,destroy:n,deps:r,next:null},t=he.updateQueue,t===null?(t={lastEffect:null,stores:null},he.updateQueue=t,t.lastEffect=e.next=e):(n=t.lastEffect,n===null?t.lastEffect=e.next=e:(r=n.next,n.next=e,e.next=r,t.lastEffect=e)),e}function Fm(){return dt().memoizedState}function po(e,t,n,r){var s=It();he.flags|=e,s.memoizedState=ii(1|t,n,void 0,r===void 0?null:r)}function ua(e,t,n,r){var s=dt();r=r===void 0?null:r;var i=void 0;if(ke!==null){var a=ke.memoizedState;if(i=a.destroy,r!==null&&Sc(r,a.deps)){s.memoizedState=ii(t,n,i,r);return}}he.flags|=e,s.memoizedState=ii(1|t,n,i,r)}function Bf(e,t){return po(8390656,8,e,t)}function bc(e,t){return ua(2048,8,e,t)}function zm(e,t){return ua(4,2,e,t)}function Vm(e,t){return ua(4,4,e,t)}function $m(e,t){if(typeof t=="function")return e=e(),t(e),function(){t(null)};if(t!=null)return e=e(),t.current=e,function(){t.current=null}}function Hm(e,t,n){return n=n!=null?n.concat([e]):null,ua(4,4,$m.bind(null,t,e),n)}function _c(){}function Gm(e,t){var n=dt();t=t===void 0?null:t;var r=n.memoizedState;return r!==null&&t!==null&&Sc(t,r[1])?r[0]:(n.memoizedState=[e,t],e)}function Wm(e,t){var n=dt();t=t===void 0?null:t;var r=n.memoizedState;return r!==null&&t!==null&&Sc(t,r[1])?r[0]:(e=e(),n.memoizedState=[e,t],e)}function qm(e,t,n){return Zn&21?(Et(n,t)||(n=Qp(),he.lanes|=n,er|=n,e.baseState=!0),t):(e.baseState&&(e.baseState=!1,qe=!0),e.memoizedState=n)}function Tw(e,t){var n=re;re=n!==0&&4>n?n:4,e(!0);var r=pl.transition;pl.transition={};try{e(!1),t()}finally{re=n,pl.transition=r}}function Km(){return dt().memoizedState}function Nw(e,t,n){var r=wn(e);if(n={lane:r,action:n,hasEagerState:!1,eagerState:null,next:null},Xm(e))Jm(t,n);else if(n=jm(e,t,n,r),n!==null){var s=ze();xt(n,e,r,s),Ym(n,t,r)}}function jw(e,t,n){var r=wn(e),s={lane:r,action:n,hasEagerState:!1,eagerState:null,next:null};if(Xm(e))Jm(t,s);else{var i=e.alternate;if(e.lanes===0&&(i===null||i.lanes===0)&&(i=t.lastRenderedReducer,i!==null))try{var a=t.lastRenderedState,u=i(a,n);if(s.hasEagerState=!0,s.eagerState=u,Et(u,a)){var c=t.interleaved;c===null?(s.next=s,gc(t)):(s.next=c.next,c.next=s),t.interleaved=s;return}}catch{}finally{}n=jm(e,t,s,r),n!==null&&(s=ze(),xt(n,e,r,s),Ym(n,t,r))}}function Xm(e){var t=e.alternate;return e===he||t!==null&&t===he}function Jm(e,t){Ps=Fo=!0;var n=e.pending;n===null?t.next=t:(t.next=n.next,n.next=t),e.pending=t}function Ym(e,t,n){if(n&4194240){var r=t.lanes;r&=e.pendingLanes,n|=r,t.lanes=n,nc(e,n)}}var zo={readContext:ct,useCallback:Pe,useContext:Pe,useEffect:Pe,useImperativeHandle:Pe,useInsertionEffect:Pe,useLayoutEffect:Pe,useMemo:Pe,useReducer:Pe,useRef:Pe,useState:Pe,useDebugValue:Pe,useDeferredValue:Pe,useTransition:Pe,useMutableSource:Pe,useSyncExternalStore:Pe,useId:Pe,unstable_isNewReconciler:!1},Aw={readContext:ct,useCallback:function(e,t){return It().memoizedState=[e,t===void 0?null:t],e},useContext:ct,useEffect:Bf,useImperativeHandle:function(e,t,n){return n=n!=null?n.concat([e]):null,po(4194308,4,$m.bind(null,t,e),n)},useLayoutEffect:function(e,t){return po(4194308,4,e,t)},useInsertionEffect:function(e,t){return po(4,2,e,t)},useMemo:function(e,t){var n=It();return t=t===void 0?null:t,e=e(),n.memoizedState=[e,t],e},useReducer:function(e,t,n){var r=It();return t=n!==void 0?n(t):t,r.memoizedState=r.baseState=t,e={pending:null,interleaved:null,lanes:0,dispatch:null,lastRenderedReducer:e,lastRenderedState:t},r.queue=e,e=e.dispatch=Nw.bind(null,he,e),[r.memoizedState,e]},useRef:function(e){var t=It();return e={current:e},t.memoizedState=e},useState:Uf,useDebugValue:_c,useDeferredValue:function(e){return It().memoizedState=e},useTransition:function(){var e=Uf(!1),t=e[0];return e=Tw.bind(null,e[1]),It().memoizedState=e,[t,e]},useMutableSource:function(){},useSyncExternalStore:function(e,t,n){var r=he,s=It();if(de){if(n===void 0)throw Error(L(407));n=n()}else{if(n=t(),Te===null)throw Error(L(349));Zn&30||Dm(r,t,n)}s.memoizedState=n;var i={value:n,getSnapshot:t};return s.queue=i,Bf(Lm.bind(null,r,i,e),[e]),r.flags|=2048,ii(9,Mm.bind(null,r,i,n,t),void 0,null),n},useId:function(){var e=It(),t=Te.identifierPrefix;if(de){var n=Ft,r=Bt;n=(r&~(1<<32-wt(r)-1)).toString(32)+n,t=":"+t+"R"+n,n=ri++,0<n&&(t+="H"+n.toString(32)),t+=":"}else n=Iw++,t=":"+t+"r"+n.toString(32)+":";return e.memoizedState=t},unstable_isNewReconciler:!1},Rw={readContext:ct,useCallback:Gm,useContext:ct,useEffect:bc,useImperativeHandle:Hm,useInsertionEffect:zm,useLayoutEffect:Vm,useMemo:Wm,useReducer:ml,useRef:Fm,useState:function(){return ml(si)},useDebugValue:_c,useDeferredValue:function(e){var t=dt();return qm(t,ke.memoizedState,e)},useTransition:function(){var e=ml(si)[0],t=dt().memoizedState;return[e,t]},useMutableSource:Pm,useSyncExternalStore:Om,useId:Km,unstable_isNewReconciler:!1},Pw={readContext:ct,useCallback:Gm,useContext:ct,useEffect:bc,useImperativeHandle:Hm,useInsertionEffect:zm,useLayoutEffect:Vm,useMemo:Wm,useReducer:gl,useRef:Fm,useState:function(){return gl(si)},useDebugValue:_c,useDeferredValue:function(e){var t=dt();return ke===null?t.memoizedState=e:qm(t,ke.memoizedState,e)},useTransition:function(){var e=gl(si)[0],t=dt().memoizedState;return[e,t]},useMutableSource:Pm,useSyncExternalStore:Om,useId:Km,unstable_isNewReconciler:!1};function pt(e,t){if(e&&e.defaultProps){t=pe({},t),e=e.defaultProps;for(var n in e)t[n]===void 0&&(t[n]=e[n]);return t}return t}function pu(e,t,n,r){t=e.memoizedState,n=n(r,t),n=n==null?t:pe({},t,n),e.memoizedState=n,e.lanes===0&&(e.updateQueue.baseState=n)}var ca={isMounted:function(e){return(e=e._reactInternals)?ir(e)===e:!1},enqueueSetState:function(e,t,n){e=e._reactInternals;var r=ze(),s=wn(e),i=zt(r,s);i.payload=t,n!=null&&(i.callback=n),t=yn(e,i,s),t!==null&&(xt(t,e,s,r),fo(t,e,s))},enqueueReplaceState:function(e,t,n){e=e._reactInternals;var r=ze(),s=wn(e),i=zt(r,s);i.tag=1,i.payload=t,n!=null&&(i.callback=n),t=yn(e,i,s),t!==null&&(xt(t,e,s,r),fo(t,e,s))},enqueueForceUpdate:function(e,t){e=e._reactInternals;var n=ze(),r=wn(e),s=zt(n,r);s.tag=2,t!=null&&(s.callback=t),t=yn(e,s,r),t!==null&&(xt(t,e,r,n),fo(t,e,r))}};function Ff(e,t,n,r,s,i,a){return e=e.stateNode,typeof e.shouldComponentUpdate=="function"?e.shouldComponentUpdate(r,i,a):t.prototype&&t.prototype.isPureReactComponent?!Ys(n,r)||!Ys(s,i):!0}function Qm(e,t,n){var r=!1,s=Cn,i=t.contextType;return typeof i=="object"&&i!==null?i=ct(i):(s=Xe(t)?Yn:Me.current,r=t.contextTypes,i=(r=r!=null)?Br(e,s):Cn),t=new t(n,i),e.memoizedState=t.state!==null&&t.state!==void 0?t.state:null,t.updater=ca,e.stateNode=t,t._reactInternals=e,r&&(e=e.stateNode,e.__reactInternalMemoizedUnmaskedChildContext=s,e.__reactInternalMemoizedMaskedChildContext=i),t}function zf(e,t,n,r){e=t.state,typeof t.componentWillReceiveProps=="function"&&t.componentWillReceiveProps(n,r),typeof t.UNSAFE_componentWillReceiveProps=="function"&&t.UNSAFE_componentWillReceiveProps(n,r),t.state!==e&&ca.enqueueReplaceState(t,t.state,null)}function mu(e,t,n,r){var s=e.stateNode;s.props=n,s.state=e.memoizedState,s.refs={},yc(e);var i=t.contextType;typeof i=="object"&&i!==null?s.context=ct(i):(i=Xe(t)?Yn:Me.current,s.context=Br(e,i)),s.state=e.memoizedState,i=t.getDerivedStateFromProps,typeof i=="function"&&(pu(e,t,i,n),s.state=e.memoizedState),typeof t.getDerivedStateFromProps=="function"||typeof s.getSnapshotBeforeUpdate=="function"||typeof s.UNSAFE_componentWillMount!="function"&&typeof s.componentWillMount!="function"||(t=s.state,typeof s.componentWillMount=="function"&&s.componentWillMount(),typeof s.UNSAFE_componentWillMount=="function"&&s.UNSAFE_componentWillMount(),t!==s.state&&ca.enqueueReplaceState(s,s.state,null),Uo(e,n,s,r),s.state=e.memoizedState),typeof s.componentDidMount=="function"&&(e.flags|=4194308)}function $r(e,t){try{var n="",r=t;do n+=av(r),r=r.return;while(r);var s=n}catch(i){s=`
Error generating stack: `+i.message+`
`+i.stack}return{value:e,source:t,stack:s,digest:null}}function yl(e,t,n){return{value:e,source:null,stack:n??null,digest:t??null}}function gu(e,t){try{console.error(t.value)}catch(n){setTimeout(function(){throw n})}}var Ow=typeof WeakMap=="function"?WeakMap:Map;function Zm(e,t,n){n=zt(-1,n),n.tag=3,n.payload={element:null};var r=t.value;return n.callback=function(){$o||($o=!0,Cu=r),gu(e,t)},n}function eg(e,t,n){n=zt(-1,n),n.tag=3;var r=e.type.getDerivedStateFromError;if(typeof r=="function"){var s=t.value;n.payload=function(){return r(s)},n.callback=function(){gu(e,t)}}var i=e.stateNode;return i!==null&&typeof i.componentDidCatch=="function"&&(n.callback=function(){gu(e,t),typeof r!="function"&&(vn===null?vn=new Set([this]):vn.add(this));var a=t.stack;this.componentDidCatch(t.value,{componentStack:a!==null?a:""})}),n}function Vf(e,t,n){var r=e.pingCache;if(r===null){r=e.pingCache=new Ow;var s=new Set;r.set(t,s)}else s=r.get(t),s===void 0&&(s=new Set,r.set(t,s));s.has(n)||(s.add(n),e=Kw.bind(null,e,t,n),t.then(e,e))}function $f(e){do{var t;if((t=e.tag===13)&&(t=e.memoizedState,t=t!==null?t.dehydrated!==null:!0),t)return e;e=e.return}while(e!==null);return null}function Hf(e,t,n,r,s){return e.mode&1?(e.flags|=65536,e.lanes=s,e):(e===t?e.flags|=65536:(e.flags|=128,n.flags|=131072,n.flags&=-52805,n.tag===1&&(n.alternate===null?n.tag=17:(t=zt(-1,1),t.tag=2,yn(n,t,1))),n.lanes|=1),e)}var Dw=qt.ReactCurrentOwner,qe=!1;function Fe(e,t,n,r){t.child=e===null?Nm(t,null,n,r):zr(t,e.child,n,r)}function Gf(e,t,n,r,s){n=n.render;var i=t.ref;return Ar(t,s),r=Ec(e,t,n,r,i,s),n=kc(),e!==null&&!qe?(t.updateQueue=e.updateQueue,t.flags&=-2053,e.lanes&=~s,Gt(e,t,s)):(de&&n&&cc(t),t.flags|=1,Fe(e,t,r,s),t.child)}function Wf(e,t,n,r,s){if(e===null){var i=n.type;return typeof i=="function"&&!Pc(i)&&i.defaultProps===void 0&&n.compare===null&&n.defaultProps===void 0?(t.tag=15,t.type=i,tg(e,t,i,r,s)):(e=vo(n.type,null,r,t,t.mode,s),e.ref=t.ref,e.return=t,t.child=e)}if(i=e.child,!(e.lanes&s)){var a=i.memoizedProps;if(n=n.compare,n=n!==null?n:Ys,n(a,r)&&e.ref===t.ref)return Gt(e,t,s)}return t.flags|=1,e=xn(i,r),e.ref=t.ref,e.return=t,t.child=e}function tg(e,t,n,r,s){if(e!==null){var i=e.memoizedProps;if(Ys(i,r)&&e.ref===t.ref)if(qe=!1,t.pendingProps=r=i,(e.lanes&s)!==0)e.flags&131072&&(qe=!0);else return t.lanes=e.lanes,Gt(e,t,s)}return yu(e,t,n,r,s)}function ng(e,t,n){var r=t.pendingProps,s=r.children,i=e!==null?e.memoizedState:null;if(r.mode==="hidden")if(!(t.mode&1))t.memoizedState={baseLanes:0,cachePool:null,transitions:null},oe(br,Ye),Ye|=n;else{if(!(n&1073741824))return e=i!==null?i.baseLanes|n:n,t.lanes=t.childLanes=1073741824,t.memoizedState={baseLanes:e,cachePool:null,transitions:null},t.updateQueue=null,oe(br,Ye),Ye|=e,null;t.memoizedState={baseLanes:0,cachePool:null,transitions:null},r=i!==null?i.baseLanes:n,oe(br,Ye),Ye|=r}else i!==null?(r=i.baseLanes|n,t.memoizedState=null):r=n,oe(br,Ye),Ye|=r;return Fe(e,t,s,n),t.child}function rg(e,t){var n=t.ref;(e===null&&n!==null||e!==null&&e.ref!==n)&&(t.flags|=512,t.flags|=2097152)}function yu(e,t,n,r,s){var i=Xe(n)?Yn:Me.current;return i=Br(t,i),Ar(t,s),n=Ec(e,t,n,r,i,s),r=kc(),e!==null&&!qe?(t.updateQueue=e.updateQueue,t.flags&=-2053,e.lanes&=~s,Gt(e,t,s)):(de&&r&&cc(t),t.flags|=1,Fe(e,t,n,s),t.child)}function qf(e,t,n,r,s){if(Xe(n)){var i=!0;Po(t)}else i=!1;if(Ar(t,s),t.stateNode===null)mo(e,t),Qm(t,n,r),mu(t,n,r,s),r=!0;else if(e===null){var a=t.stateNode,u=t.memoizedProps;a.props=u;var c=a.context,f=n.contextType;typeof f=="object"&&f!==null?f=ct(f):(f=Xe(n)?Yn:Me.current,f=Br(t,f));var E=n.getDerivedStateFromProps,w=typeof E=="function"||typeof a.getSnapshotBeforeUpdate=="function";w||typeof a.UNSAFE_componentWillReceiveProps!="function"&&typeof a.componentWillReceiveProps!="function"||(u!==r||c!==f)&&zf(t,a,r,f),on=!1;var _=t.memoizedState;a.state=_,Uo(t,r,a,s),c=t.memoizedState,u!==r||_!==c||Ke.current||on?(typeof E=="function"&&(pu(t,n,E,r),c=t.memoizedState),(u=on||Ff(t,n,u,r,_,c,f))?(w||typeof a.UNSAFE_componentWillMount!="function"&&typeof a.componentWillMount!="function"||(typeof a.componentWillMount=="function"&&a.componentWillMount(),typeof a.UNSAFE_componentWillMount=="function"&&a.UNSAFE_componentWillMount()),typeof a.componentDidMount=="function"&&(t.flags|=4194308)):(typeof a.componentDidMount=="function"&&(t.flags|=4194308),t.memoizedProps=r,t.memoizedState=c),a.props=r,a.state=c,a.context=f,r=u):(typeof a.componentDidMount=="function"&&(t.flags|=4194308),r=!1)}else{a=t.stateNode,Am(e,t),u=t.memoizedProps,f=t.type===t.elementType?u:pt(t.type,u),a.props=f,w=t.pendingProps,_=a.context,c=n.contextType,typeof c=="object"&&c!==null?c=ct(c):(c=Xe(n)?Yn:Me.current,c=Br(t,c));var A=n.getDerivedStateFromProps;(E=typeof A=="function"||typeof a.getSnapshotBeforeUpdate=="function")||typeof a.UNSAFE_componentWillReceiveProps!="function"&&typeof a.componentWillReceiveProps!="function"||(u!==w||_!==c)&&zf(t,a,r,c),on=!1,_=t.memoizedState,a.state=_,Uo(t,r,a,s);var R=t.memoizedState;u!==w||_!==R||Ke.current||on?(typeof A=="function"&&(pu(t,n,A,r),R=t.memoizedState),(f=on||Ff(t,n,f,r,_,R,c)||!1)?(E||typeof a.UNSAFE_componentWillUpdate!="function"&&typeof a.componentWillUpdate!="function"||(typeof a.componentWillUpdate=="function"&&a.componentWillUpdate(r,R,c),typeof a.UNSAFE_componentWillUpdate=="function"&&a.UNSAFE_componentWillUpdate(r,R,c)),typeof a.componentDidUpdate=="function"&&(t.flags|=4),typeof a.getSnapshotBeforeUpdate=="function"&&(t.flags|=1024)):(typeof a.componentDidUpdate!="function"||u===e.memoizedProps&&_===e.memoizedState||(t.flags|=4),typeof a.getSnapshotBeforeUpdate!="function"||u===e.memoizedProps&&_===e.memoizedState||(t.flags|=1024),t.memoizedProps=r,t.memoizedState=R),a.props=r,a.state=R,a.context=c,r=f):(typeof a.componentDidUpdate!="function"||u===e.memoizedProps&&_===e.memoizedState||(t.flags|=4),typeof a.getSnapshotBeforeUpdate!="function"||u===e.memoizedProps&&_===e.memoizedState||(t.flags|=1024),r=!1)}return vu(e,t,n,r,i,s)}function vu(e,t,n,r,s,i){rg(e,t);var a=(t.flags&128)!==0;if(!r&&!a)return s&&Rf(t,n,!1),Gt(e,t,i);r=t.stateNode,Dw.current=t;var u=a&&typeof n.getDerivedStateFromError!="function"?null:r.render();return t.flags|=1,e!==null&&a?(t.child=zr(t,e.child,null,i),t.child=zr(t,null,u,i)):Fe(e,t,u,i),t.memoizedState=r.state,s&&Rf(t,n,!0),t.child}function sg(e){var t=e.stateNode;t.pendingContext?Af(e,t.pendingContext,t.pendingContext!==t.context):t.context&&Af(e,t.context,!1),vc(e,t.containerInfo)}function Kf(e,t,n,r,s){return Fr(),fc(s),t.flags|=256,Fe(e,t,n,r),t.child}var wu={dehydrated:null,treeContext:null,retryLane:0};function xu(e){return{baseLanes:e,cachePool:null,transitions:null}}function ig(e,t,n){var r=t.pendingProps,s=fe.current,i=!1,a=(t.flags&128)!==0,u;if((u=a)||(u=e!==null&&e.memoizedState===null?!1:(s&2)!==0),u?(i=!0,t.flags&=-129):(e===null||e.memoizedState!==null)&&(s|=1),oe(fe,s&1),e===null)return fu(t),e=t.memoizedState,e!==null&&(e=e.dehydrated,e!==null)?(t.mode&1?e.data==="$!"?t.lanes=8:t.lanes=1073741824:t.lanes=1,null):(a=r.children,e=r.fallback,i?(r=t.mode,i=t.child,a={mode:"hidden",children:a},!(r&1)&&i!==null?(i.childLanes=0,i.pendingProps=a):i=ha(a,r,0,null),e=qn(e,r,n,null),i.return=t,e.return=t,i.sibling=e,t.child=i,t.child.memoizedState=xu(n),t.memoizedState=wu,e):Cc(t,a));if(s=e.memoizedState,s!==null&&(u=s.dehydrated,u!==null))return Mw(e,t,a,r,u,s,n);if(i){i=r.fallback,a=t.mode,s=e.child,u=s.sibling;var c={mode:"hidden",children:r.children};return!(a&1)&&t.child!==s?(r=t.child,r.childLanes=0,r.pendingProps=c,t.deletions=null):(r=xn(s,c),r.subtreeFlags=s.subtreeFlags&14680064),u!==null?i=xn(u,i):(i=qn(i,a,n,null),i.flags|=2),i.return=t,r.return=t,r.sibling=i,t.child=r,r=i,i=t.child,a=e.child.memoizedState,a=a===null?xu(n):{baseLanes:a.baseLanes|n,cachePool:null,transitions:a.transitions},i.memoizedState=a,i.childLanes=e.childLanes&~n,t.memoizedState=wu,r}return i=e.child,e=i.sibling,r=xn(i,{mode:"visible",children:r.children}),!(t.mode&1)&&(r.lanes=n),r.return=t,r.sibling=null,e!==null&&(n=t.deletions,n===null?(t.deletions=[e],t.flags|=16):n.push(e)),t.child=r,t.memoizedState=null,r}function Cc(e,t){return t=ha({mode:"visible",children:t},e.mode,0,null),t.return=e,e.child=t}function Xi(e,t,n,r){return r!==null&&fc(r),zr(t,e.child,null,n),e=Cc(t,t.pendingProps.children),e.flags|=2,t.memoizedState=null,e}function Mw(e,t,n,r,s,i,a){if(n)return t.flags&256?(t.flags&=-257,r=yl(Error(L(422))),Xi(e,t,a,r)):t.memoizedState!==null?(t.child=e.child,t.flags|=128,null):(i=r.fallback,s=t.mode,r=ha({mode:"visible",children:r.children},s,0,null),i=qn(i,s,a,null),i.flags|=2,r.return=t,i.return=t,r.sibling=i,t.child=r,t.mode&1&&zr(t,e.child,null,a),t.child.memoizedState=xu(a),t.memoizedState=wu,i);if(!(t.mode&1))return Xi(e,t,a,null);if(s.data==="$!"){if(r=s.nextSibling&&s.nextSibling.dataset,r)var u=r.dgst;return r=u,i=Error(L(419)),r=yl(i,r,void 0),Xi(e,t,a,r)}if(u=(a&e.childLanes)!==0,qe||u){if(r=Te,r!==null){switch(a&-a){case 4:s=2;break;case 16:s=8;break;case 64:case 128:case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:case 4194304:case 8388608:case 16777216:case 33554432:case 67108864:s=32;break;case 536870912:s=268435456;break;default:s=0}s=s&(r.suspendedLanes|a)?0:s,s!==0&&s!==i.retryLane&&(i.retryLane=s,Ht(e,s),xt(r,e,s,-1))}return Rc(),r=yl(Error(L(421))),Xi(e,t,a,r)}return s.data==="$?"?(t.flags|=128,t.child=e.child,t=Xw.bind(null,e),s._reactRetry=t,null):(e=i.treeContext,et=gn(s.nextSibling),tt=t,de=!0,gt=null,e!==null&&(ot[at++]=Bt,ot[at++]=Ft,ot[at++]=Qn,Bt=e.id,Ft=e.overflow,Qn=t),t=Cc(t,r.children),t.flags|=4096,t)}function Xf(e,t,n){e.lanes|=t;var r=e.alternate;r!==null&&(r.lanes|=t),hu(e.return,t,n)}function vl(e,t,n,r,s){var i=e.memoizedState;i===null?e.memoizedState={isBackwards:t,rendering:null,renderingStartTime:0,last:r,tail:n,tailMode:s}:(i.isBackwards=t,i.rendering=null,i.renderingStartTime=0,i.last=r,i.tail=n,i.tailMode=s)}function og(e,t,n){var r=t.pendingProps,s=r.revealOrder,i=r.tail;if(Fe(e,t,r.children,n),r=fe.current,r&2)r=r&1|2,t.flags|=128;else{if(e!==null&&e.flags&128)e:for(e=t.child;e!==null;){if(e.tag===13)e.memoizedState!==null&&Xf(e,n,t);else if(e.tag===19)Xf(e,n,t);else if(e.child!==null){e.child.return=e,e=e.child;continue}if(e===t)break e;for(;e.sibling===null;){if(e.return===null||e.return===t)break e;e=e.return}e.sibling.return=e.return,e=e.sibling}r&=1}if(oe(fe,r),!(t.mode&1))t.memoizedState=null;else switch(s){case"forwards":for(n=t.child,s=null;n!==null;)e=n.alternate,e!==null&&Bo(e)===null&&(s=n),n=n.sibling;n=s,n===null?(s=t.child,t.child=null):(s=n.sibling,n.sibling=null),vl(t,!1,s,n,i);break;case"backwards":for(n=null,s=t.child,t.child=null;s!==null;){if(e=s.alternate,e!==null&&Bo(e)===null){t.child=s;break}e=s.sibling,s.sibling=n,n=s,s=e}vl(t,!0,n,null,i);break;case"together":vl(t,!1,null,null,void 0);break;default:t.memoizedState=null}return t.child}function mo(e,t){!(t.mode&1)&&e!==null&&(e.alternate=null,t.alternate=null,t.flags|=2)}function Gt(e,t,n){if(e!==null&&(t.dependencies=e.dependencies),er|=t.lanes,!(n&t.childLanes))return null;if(e!==null&&t.child!==e.child)throw Error(L(153));if(t.child!==null){for(e=t.child,n=xn(e,e.pendingProps),t.child=n,n.return=t;e.sibling!==null;)e=e.sibling,n=n.sibling=xn(e,e.pendingProps),n.return=t;n.sibling=null}return t.child}function Lw(e,t,n){switch(t.tag){case 3:sg(t),Fr();break;case 5:Rm(t);break;case 1:Xe(t.type)&&Po(t);break;case 4:vc(t,t.stateNode.containerInfo);break;case 10:var r=t.type._context,s=t.memoizedProps.value;oe(Mo,r._currentValue),r._currentValue=s;break;case 13:if(r=t.memoizedState,r!==null)return r.dehydrated!==null?(oe(fe,fe.current&1),t.flags|=128,null):n&t.child.childLanes?ig(e,t,n):(oe(fe,fe.current&1),e=Gt(e,t,n),e!==null?e.sibling:null);oe(fe,fe.current&1);break;case 19:if(r=(n&t.childLanes)!==0,e.flags&128){if(r)return og(e,t,n);t.flags|=128}if(s=t.memoizedState,s!==null&&(s.rendering=null,s.tail=null,s.lastEffect=null),oe(fe,fe.current),r)break;return null;case 22:case 23:return t.lanes=0,ng(e,t,n)}return Gt(e,t,n)}var ag,Su,lg,ug;ag=function(e,t){for(var n=t.child;n!==null;){if(n.tag===5||n.tag===6)e.appendChild(n.stateNode);else if(n.tag!==4&&n.child!==null){n.child.return=n,n=n.child;continue}if(n===t)break;for(;n.sibling===null;){if(n.return===null||n.return===t)return;n=n.return}n.sibling.return=n.return,n=n.sibling}};Su=function(){};lg=function(e,t,n,r){var s=e.memoizedProps;if(s!==r){e=t.stateNode,$n(Ot.current);var i=null;switch(n){case"input":s=$l(e,s),r=$l(e,r),i=[];break;case"select":s=pe({},s,{value:void 0}),r=pe({},r,{value:void 0}),i=[];break;case"textarea":s=Wl(e,s),r=Wl(e,r),i=[];break;default:typeof s.onClick!="function"&&typeof r.onClick=="function"&&(e.onclick=Ao)}Kl(n,r);var a;n=null;for(f in s)if(!r.hasOwnProperty(f)&&s.hasOwnProperty(f)&&s[f]!=null)if(f==="style"){var u=s[f];for(a in u)u.hasOwnProperty(a)&&(n||(n={}),n[a]="")}else f!=="dangerouslySetInnerHTML"&&f!=="children"&&f!=="suppressContentEditableWarning"&&f!=="suppressHydrationWarning"&&f!=="autoFocus"&&(Hs.hasOwnProperty(f)?i||(i=[]):(i=i||[]).push(f,null));for(f in r){var c=r[f];if(u=s!=null?s[f]:void 0,r.hasOwnProperty(f)&&c!==u&&(c!=null||u!=null))if(f==="style")if(u){for(a in u)!u.hasOwnProperty(a)||c&&c.hasOwnProperty(a)||(n||(n={}),n[a]="");for(a in c)c.hasOwnProperty(a)&&u[a]!==c[a]&&(n||(n={}),n[a]=c[a])}else n||(i||(i=[]),i.push(f,n)),n=c;else f==="dangerouslySetInnerHTML"?(c=c?c.__html:void 0,u=u?u.__html:void 0,c!=null&&u!==c&&(i=i||[]).push(f,c)):f==="children"?typeof c!="string"&&typeof c!="number"||(i=i||[]).push(f,""+c):f!=="suppressContentEditableWarning"&&f!=="suppressHydrationWarning"&&(Hs.hasOwnProperty(f)?(c!=null&&f==="onScroll"&&ue("scroll",e),i||u===c||(i=[])):(i=i||[]).push(f,c))}n&&(i=i||[]).push("style",n);var f=i;(t.updateQueue=f)&&(t.flags|=4)}};ug=function(e,t,n,r){n!==r&&(t.flags|=4)};function Ss(e,t){if(!de)switch(e.tailMode){case"hidden":t=e.tail;for(var n=null;t!==null;)t.alternate!==null&&(n=t),t=t.sibling;n===null?e.tail=null:n.sibling=null;break;case"collapsed":n=e.tail;for(var r=null;n!==null;)n.alternate!==null&&(r=n),n=n.sibling;r===null?t||e.tail===null?e.tail=null:e.tail.sibling=null:r.sibling=null}}function Oe(e){var t=e.alternate!==null&&e.alternate.child===e.child,n=0,r=0;if(t)for(var s=e.child;s!==null;)n|=s.lanes|s.childLanes,r|=s.subtreeFlags&14680064,r|=s.flags&14680064,s.return=e,s=s.sibling;else for(s=e.child;s!==null;)n|=s.lanes|s.childLanes,r|=s.subtreeFlags,r|=s.flags,s.return=e,s=s.sibling;return e.subtreeFlags|=r,e.childLanes=n,t}function Uw(e,t,n){var r=t.pendingProps;switch(dc(t),t.tag){case 2:case 16:case 15:case 0:case 11:case 7:case 8:case 12:case 9:case 14:return Oe(t),null;case 1:return Xe(t.type)&&Ro(),Oe(t),null;case 3:return r=t.stateNode,Vr(),ce(Ke),ce(Me),xc(),r.pendingContext&&(r.context=r.pendingContext,r.pendingContext=null),(e===null||e.child===null)&&(qi(t)?t.flags|=4:e===null||e.memoizedState.isDehydrated&&!(t.flags&256)||(t.flags|=1024,gt!==null&&(Nu(gt),gt=null))),Su(e,t),Oe(t),null;case 5:wc(t);var s=$n(ni.current);if(n=t.type,e!==null&&t.stateNode!=null)lg(e,t,n,r,s),e.ref!==t.ref&&(t.flags|=512,t.flags|=2097152);else{if(!r){if(t.stateNode===null)throw Error(L(166));return Oe(t),null}if(e=$n(Ot.current),qi(t)){r=t.stateNode,n=t.type;var i=t.memoizedProps;switch(r[jt]=t,r[ei]=i,e=(t.mode&1)!==0,n){case"dialog":ue("cancel",r),ue("close",r);break;case"iframe":case"object":case"embed":ue("load",r);break;case"video":case"audio":for(s=0;s<Cs.length;s++)ue(Cs[s],r);break;case"source":ue("error",r);break;case"img":case"image":case"link":ue("error",r),ue("load",r);break;case"details":ue("toggle",r);break;case"input":rf(r,i),ue("invalid",r);break;case"select":r._wrapperState={wasMultiple:!!i.multiple},ue("invalid",r);break;case"textarea":of(r,i),ue("invalid",r)}Kl(n,i),s=null;for(var a in i)if(i.hasOwnProperty(a)){var u=i[a];a==="children"?typeof u=="string"?r.textContent!==u&&(i.suppressHydrationWarning!==!0&&Wi(r.textContent,u,e),s=["children",u]):typeof u=="number"&&r.textContent!==""+u&&(i.suppressHydrationWarning!==!0&&Wi(r.textContent,u,e),s=["children",""+u]):Hs.hasOwnProperty(a)&&u!=null&&a==="onScroll"&&ue("scroll",r)}switch(n){case"input":Ui(r),sf(r,i,!0);break;case"textarea":Ui(r),af(r);break;case"select":case"option":break;default:typeof i.onClick=="function"&&(r.onclick=Ao)}r=s,t.updateQueue=r,r!==null&&(t.flags|=4)}else{a=s.nodeType===9?s:s.ownerDocument,e==="http://www.w3.org/1999/xhtml"&&(e=Lp(n)),e==="http://www.w3.org/1999/xhtml"?n==="script"?(e=a.createElement("div"),e.innerHTML="<script><\/script>",e=e.removeChild(e.firstChild)):typeof r.is=="string"?e=a.createElement(n,{is:r.is}):(e=a.createElement(n),n==="select"&&(a=e,r.multiple?a.multiple=!0:r.size&&(a.size=r.size))):e=a.createElementNS(e,n),e[jt]=t,e[ei]=r,ag(e,t,!1,!1),t.stateNode=e;e:{switch(a=Xl(n,r),n){case"dialog":ue("cancel",e),ue("close",e),s=r;break;case"iframe":case"object":case"embed":ue("load",e),s=r;break;case"video":case"audio":for(s=0;s<Cs.length;s++)ue(Cs[s],e);s=r;break;case"source":ue("error",e),s=r;break;case"img":case"image":case"link":ue("error",e),ue("load",e),s=r;break;case"details":ue("toggle",e),s=r;break;case"input":rf(e,r),s=$l(e,r),ue("invalid",e);break;case"option":s=r;break;case"select":e._wrapperState={wasMultiple:!!r.multiple},s=pe({},r,{value:void 0}),ue("invalid",e);break;case"textarea":of(e,r),s=Wl(e,r),ue("invalid",e);break;default:s=r}Kl(n,s),u=s;for(i in u)if(u.hasOwnProperty(i)){var c=u[i];i==="style"?Fp(e,c):i==="dangerouslySetInnerHTML"?(c=c?c.__html:void 0,c!=null&&Up(e,c)):i==="children"?typeof c=="string"?(n!=="textarea"||c!=="")&&Gs(e,c):typeof c=="number"&&Gs(e,""+c):i!=="suppressContentEditableWarning"&&i!=="suppressHydrationWarning"&&i!=="autoFocus"&&(Hs.hasOwnProperty(i)?c!=null&&i==="onScroll"&&ue("scroll",e):c!=null&&Ju(e,i,c,a))}switch(n){case"input":Ui(e),sf(e,r,!1);break;case"textarea":Ui(e),af(e);break;case"option":r.value!=null&&e.setAttribute("value",""+_n(r.value));break;case"select":e.multiple=!!r.multiple,i=r.value,i!=null?Ir(e,!!r.multiple,i,!1):r.defaultValue!=null&&Ir(e,!!r.multiple,r.defaultValue,!0);break;default:typeof s.onClick=="function"&&(e.onclick=Ao)}switch(n){case"button":case"input":case"select":case"textarea":r=!!r.autoFocus;break e;case"img":r=!0;break e;default:r=!1}}r&&(t.flags|=4)}t.ref!==null&&(t.flags|=512,t.flags|=2097152)}return Oe(t),null;case 6:if(e&&t.stateNode!=null)ug(e,t,e.memoizedProps,r);else{if(typeof r!="string"&&t.stateNode===null)throw Error(L(166));if(n=$n(ni.current),$n(Ot.current),qi(t)){if(r=t.stateNode,n=t.memoizedProps,r[jt]=t,(i=r.nodeValue!==n)&&(e=tt,e!==null))switch(e.tag){case 3:Wi(r.nodeValue,n,(e.mode&1)!==0);break;case 5:e.memoizedProps.suppressHydrationWarning!==!0&&Wi(r.nodeValue,n,(e.mode&1)!==0)}i&&(t.flags|=4)}else r=(n.nodeType===9?n:n.ownerDocument).createTextNode(r),r[jt]=t,t.stateNode=r}return Oe(t),null;case 13:if(ce(fe),r=t.memoizedState,e===null||e.memoizedState!==null&&e.memoizedState.dehydrated!==null){if(de&&et!==null&&t.mode&1&&!(t.flags&128))Im(),Fr(),t.flags|=98560,i=!1;else if(i=qi(t),r!==null&&r.dehydrated!==null){if(e===null){if(!i)throw Error(L(318));if(i=t.memoizedState,i=i!==null?i.dehydrated:null,!i)throw Error(L(317));i[jt]=t}else Fr(),!(t.flags&128)&&(t.memoizedState=null),t.flags|=4;Oe(t),i=!1}else gt!==null&&(Nu(gt),gt=null),i=!0;if(!i)return t.flags&65536?t:null}return t.flags&128?(t.lanes=n,t):(r=r!==null,r!==(e!==null&&e.memoizedState!==null)&&r&&(t.child.flags|=8192,t.mode&1&&(e===null||fe.current&1?be===0&&(be=3):Rc())),t.updateQueue!==null&&(t.flags|=4),Oe(t),null);case 4:return Vr(),Su(e,t),e===null&&Qs(t.stateNode.containerInfo),Oe(t),null;case 10:return mc(t.type._context),Oe(t),null;case 17:return Xe(t.type)&&Ro(),Oe(t),null;case 19:if(ce(fe),i=t.memoizedState,i===null)return Oe(t),null;if(r=(t.flags&128)!==0,a=i.rendering,a===null)if(r)Ss(i,!1);else{if(be!==0||e!==null&&e.flags&128)for(e=t.child;e!==null;){if(a=Bo(e),a!==null){for(t.flags|=128,Ss(i,!1),r=a.updateQueue,r!==null&&(t.updateQueue=r,t.flags|=4),t.subtreeFlags=0,r=n,n=t.child;n!==null;)i=n,e=r,i.flags&=14680066,a=i.alternate,a===null?(i.childLanes=0,i.lanes=e,i.child=null,i.subtreeFlags=0,i.memoizedProps=null,i.memoizedState=null,i.updateQueue=null,i.dependencies=null,i.stateNode=null):(i.childLanes=a.childLanes,i.lanes=a.lanes,i.child=a.child,i.subtreeFlags=0,i.deletions=null,i.memoizedProps=a.memoizedProps,i.memoizedState=a.memoizedState,i.updateQueue=a.updateQueue,i.type=a.type,e=a.dependencies,i.dependencies=e===null?null:{lanes:e.lanes,firstContext:e.firstContext}),n=n.sibling;return oe(fe,fe.current&1|2),t.child}e=e.sibling}i.tail!==null&&ve()>Hr&&(t.flags|=128,r=!0,Ss(i,!1),t.lanes=4194304)}else{if(!r)if(e=Bo(a),e!==null){if(t.flags|=128,r=!0,n=e.updateQueue,n!==null&&(t.updateQueue=n,t.flags|=4),Ss(i,!0),i.tail===null&&i.tailMode==="hidden"&&!a.alternate&&!de)return Oe(t),null}else 2*ve()-i.renderingStartTime>Hr&&n!==1073741824&&(t.flags|=128,r=!0,Ss(i,!1),t.lanes=4194304);i.isBackwards?(a.sibling=t.child,t.child=a):(n=i.last,n!==null?n.sibling=a:t.child=a,i.last=a)}return i.tail!==null?(t=i.tail,i.rendering=t,i.tail=t.sibling,i.renderingStartTime=ve(),t.sibling=null,n=fe.current,oe(fe,r?n&1|2:n&1),t):(Oe(t),null);case 22:case 23:return Ac(),r=t.memoizedState!==null,e!==null&&e.memoizedState!==null!==r&&(t.flags|=8192),r&&t.mode&1?Ye&1073741824&&(Oe(t),t.subtreeFlags&6&&(t.flags|=8192)):Oe(t),null;case 24:return null;case 25:return null}throw Error(L(156,t.tag))}function Bw(e,t){switch(dc(t),t.tag){case 1:return Xe(t.type)&&Ro(),e=t.flags,e&65536?(t.flags=e&-65537|128,t):null;case 3:return Vr(),ce(Ke),ce(Me),xc(),e=t.flags,e&65536&&!(e&128)?(t.flags=e&-65537|128,t):null;case 5:return wc(t),null;case 13:if(ce(fe),e=t.memoizedState,e!==null&&e.dehydrated!==null){if(t.alternate===null)throw Error(L(340));Fr()}return e=t.flags,e&65536?(t.flags=e&-65537|128,t):null;case 19:return ce(fe),null;case 4:return Vr(),null;case 10:return mc(t.type._context),null;case 22:case 23:return Ac(),null;case 24:return null;default:return null}}var Ji=!1,De=!1,Fw=typeof WeakSet=="function"?WeakSet:Set,z=null;function kr(e,t){var n=e.ref;if(n!==null)if(typeof n=="function")try{n(null)}catch(r){me(e,t,r)}else n.current=null}function Eu(e,t,n){try{n()}catch(r){me(e,t,r)}}var Jf=!1;function zw(e,t){if(iu=To,e=pm(),uc(e)){if("selectionStart"in e)var n={start:e.selectionStart,end:e.selectionEnd};else e:{n=(n=e.ownerDocument)&&n.defaultView||window;var r=n.getSelection&&n.getSelection();if(r&&r.rangeCount!==0){n=r.anchorNode;var s=r.anchorOffset,i=r.focusNode;r=r.focusOffset;try{n.nodeType,i.nodeType}catch{n=null;break e}var a=0,u=-1,c=-1,f=0,E=0,w=e,_=null;t:for(;;){for(var A;w!==n||s!==0&&w.nodeType!==3||(u=a+s),w!==i||r!==0&&w.nodeType!==3||(c=a+r),w.nodeType===3&&(a+=w.nodeValue.length),(A=w.firstChild)!==null;)_=w,w=A;for(;;){if(w===e)break t;if(_===n&&++f===s&&(u=a),_===i&&++E===r&&(c=a),(A=w.nextSibling)!==null)break;w=_,_=w.parentNode}w=A}n=u===-1||c===-1?null:{start:u,end:c}}else n=null}n=n||{start:0,end:0}}else n=null;for(ou={focusedElem:e,selectionRange:n},To=!1,z=t;z!==null;)if(t=z,e=t.child,(t.subtreeFlags&1028)!==0&&e!==null)e.return=t,z=e;else for(;z!==null;){t=z;try{var R=t.alternate;if(t.flags&1024)switch(t.tag){case 0:case 11:case 15:break;case 1:if(R!==null){var j=R.memoizedProps,D=R.memoizedState,b=t.stateNode,x=b.getSnapshotBeforeUpdate(t.elementType===t.type?j:pt(t.type,j),D);b.__reactInternalSnapshotBeforeUpdate=x}break;case 3:var C=t.stateNode.containerInfo;C.nodeType===1?C.textContent="":C.nodeType===9&&C.documentElement&&C.removeChild(C.documentElement);break;case 5:case 6:case 4:case 17:break;default:throw Error(L(163))}}catch(O){me(t,t.return,O)}if(e=t.sibling,e!==null){e.return=t.return,z=e;break}z=t.return}return R=Jf,Jf=!1,R}function Os(e,t,n){var r=t.updateQueue;if(r=r!==null?r.lastEffect:null,r!==null){var s=r=r.next;do{if((s.tag&e)===e){var i=s.destroy;s.destroy=void 0,i!==void 0&&Eu(t,n,i)}s=s.next}while(s!==r)}}function da(e,t){if(t=t.updateQueue,t=t!==null?t.lastEffect:null,t!==null){var n=t=t.next;do{if((n.tag&e)===e){var r=n.create;n.destroy=r()}n=n.next}while(n!==t)}}function ku(e){var t=e.ref;if(t!==null){var n=e.stateNode;switch(e.tag){case 5:e=n;break;default:e=n}typeof t=="function"?t(e):t.current=e}}function cg(e){var t=e.alternate;t!==null&&(e.alternate=null,cg(t)),e.child=null,e.deletions=null,e.sibling=null,e.tag===5&&(t=e.stateNode,t!==null&&(delete t[jt],delete t[ei],delete t[uu],delete t[kw],delete t[bw])),e.stateNode=null,e.return=null,e.dependencies=null,e.memoizedProps=null,e.memoizedState=null,e.pendingProps=null,e.stateNode=null,e.updateQueue=null}function dg(e){return e.tag===5||e.tag===3||e.tag===4}function Yf(e){e:for(;;){for(;e.sibling===null;){if(e.return===null||dg(e.return))return null;e=e.return}for(e.sibling.return=e.return,e=e.sibling;e.tag!==5&&e.tag!==6&&e.tag!==18;){if(e.flags&2||e.child===null||e.tag===4)continue e;e.child.return=e,e=e.child}if(!(e.flags&2))return e.stateNode}}function bu(e,t,n){var r=e.tag;if(r===5||r===6)e=e.stateNode,t?n.nodeType===8?n.parentNode.insertBefore(e,t):n.insertBefore(e,t):(n.nodeType===8?(t=n.parentNode,t.insertBefore(e,n)):(t=n,t.appendChild(e)),n=n._reactRootContainer,n!=null||t.onclick!==null||(t.onclick=Ao));else if(r!==4&&(e=e.child,e!==null))for(bu(e,t,n),e=e.sibling;e!==null;)bu(e,t,n),e=e.sibling}function _u(e,t,n){var r=e.tag;if(r===5||r===6)e=e.stateNode,t?n.insertBefore(e,t):n.appendChild(e);else if(r!==4&&(e=e.child,e!==null))for(_u(e,t,n),e=e.sibling;e!==null;)_u(e,t,n),e=e.sibling}var Ne=null,mt=!1;function nn(e,t,n){for(n=n.child;n!==null;)fg(e,t,n),n=n.sibling}function fg(e,t,n){if(Pt&&typeof Pt.onCommitFiberUnmount=="function")try{Pt.onCommitFiberUnmount(ra,n)}catch{}switch(n.tag){case 5:De||kr(n,t);case 6:var r=Ne,s=mt;Ne=null,nn(e,t,n),Ne=r,mt=s,Ne!==null&&(mt?(e=Ne,n=n.stateNode,e.nodeType===8?e.parentNode.removeChild(n):e.removeChild(n)):Ne.removeChild(n.stateNode));break;case 18:Ne!==null&&(mt?(e=Ne,n=n.stateNode,e.nodeType===8?dl(e.parentNode,n):e.nodeType===1&&dl(e,n),Xs(e)):dl(Ne,n.stateNode));break;case 4:r=Ne,s=mt,Ne=n.stateNode.containerInfo,mt=!0,nn(e,t,n),Ne=r,mt=s;break;case 0:case 11:case 14:case 15:if(!De&&(r=n.updateQueue,r!==null&&(r=r.lastEffect,r!==null))){s=r=r.next;do{var i=s,a=i.destroy;i=i.tag,a!==void 0&&(i&2||i&4)&&Eu(n,t,a),s=s.next}while(s!==r)}nn(e,t,n);break;case 1:if(!De&&(kr(n,t),r=n.stateNode,typeof r.componentWillUnmount=="function"))try{r.props=n.memoizedProps,r.state=n.memoizedState,r.componentWillUnmount()}catch(u){me(n,t,u)}nn(e,t,n);break;case 21:nn(e,t,n);break;case 22:n.mode&1?(De=(r=De)||n.memoizedState!==null,nn(e,t,n),De=r):nn(e,t,n);break;default:nn(e,t,n)}}function Qf(e){var t=e.updateQueue;if(t!==null){e.updateQueue=null;var n=e.stateNode;n===null&&(n=e.stateNode=new Fw),t.forEach(function(r){var s=Jw.bind(null,e,r);n.has(r)||(n.add(r),r.then(s,s))})}}function ht(e,t){var n=t.deletions;if(n!==null)for(var r=0;r<n.length;r++){var s=n[r];try{var i=e,a=t,u=a;e:for(;u!==null;){switch(u.tag){case 5:Ne=u.stateNode,mt=!1;break e;case 3:Ne=u.stateNode.containerInfo,mt=!0;break e;case 4:Ne=u.stateNode.containerInfo,mt=!0;break e}u=u.return}if(Ne===null)throw Error(L(160));fg(i,a,s),Ne=null,mt=!1;var c=s.alternate;c!==null&&(c.return=null),s.return=null}catch(f){me(s,t,f)}}if(t.subtreeFlags&12854)for(t=t.child;t!==null;)hg(t,e),t=t.sibling}function hg(e,t){var n=e.alternate,r=e.flags;switch(e.tag){case 0:case 11:case 14:case 15:if(ht(t,e),Ct(e),r&4){try{Os(3,e,e.return),da(3,e)}catch(j){me(e,e.return,j)}try{Os(5,e,e.return)}catch(j){me(e,e.return,j)}}break;case 1:ht(t,e),Ct(e),r&512&&n!==null&&kr(n,n.return);break;case 5:if(ht(t,e),Ct(e),r&512&&n!==null&&kr(n,n.return),e.flags&32){var s=e.stateNode;try{Gs(s,"")}catch(j){me(e,e.return,j)}}if(r&4&&(s=e.stateNode,s!=null)){var i=e.memoizedProps,a=n!==null?n.memoizedProps:i,u=e.type,c=e.updateQueue;if(e.updateQueue=null,c!==null)try{u==="input"&&i.type==="radio"&&i.name!=null&&Dp(s,i),Xl(u,a);var f=Xl(u,i);for(a=0;a<c.length;a+=2){var E=c[a],w=c[a+1];E==="style"?Fp(s,w):E==="dangerouslySetInnerHTML"?Up(s,w):E==="children"?Gs(s,w):Ju(s,E,w,f)}switch(u){case"input":Hl(s,i);break;case"textarea":Mp(s,i);break;case"select":var _=s._wrapperState.wasMultiple;s._wrapperState.wasMultiple=!!i.multiple;var A=i.value;A!=null?Ir(s,!!i.multiple,A,!1):_!==!!i.multiple&&(i.defaultValue!=null?Ir(s,!!i.multiple,i.defaultValue,!0):Ir(s,!!i.multiple,i.multiple?[]:"",!1))}s[ei]=i}catch(j){me(e,e.return,j)}}break;case 6:if(ht(t,e),Ct(e),r&4){if(e.stateNode===null)throw Error(L(162));s=e.stateNode,i=e.memoizedProps;try{s.nodeValue=i}catch(j){me(e,e.return,j)}}break;case 3:if(ht(t,e),Ct(e),r&4&&n!==null&&n.memoizedState.isDehydrated)try{Xs(t.containerInfo)}catch(j){me(e,e.return,j)}break;case 4:ht(t,e),Ct(e);break;case 13:ht(t,e),Ct(e),s=e.child,s.flags&8192&&(i=s.memoizedState!==null,s.stateNode.isHidden=i,!i||s.alternate!==null&&s.alternate.memoizedState!==null||(Nc=ve())),r&4&&Qf(e);break;case 22:if(E=n!==null&&n.memoizedState!==null,e.mode&1?(De=(f=De)||E,ht(t,e),De=f):ht(t,e),Ct(e),r&8192){if(f=e.memoizedState!==null,(e.stateNode.isHidden=f)&&!E&&e.mode&1)for(z=e,E=e.child;E!==null;){for(w=z=E;z!==null;){switch(_=z,A=_.child,_.tag){case 0:case 11:case 14:case 15:Os(4,_,_.return);break;case 1:kr(_,_.return);var R=_.stateNode;if(typeof R.componentWillUnmount=="function"){r=_,n=_.return;try{t=r,R.props=t.memoizedProps,R.state=t.memoizedState,R.componentWillUnmount()}catch(j){me(r,n,j)}}break;case 5:kr(_,_.return);break;case 22:if(_.memoizedState!==null){eh(w);continue}}A!==null?(A.return=_,z=A):eh(w)}E=E.sibling}e:for(E=null,w=e;;){if(w.tag===5){if(E===null){E=w;try{s=w.stateNode,f?(i=s.style,typeof i.setProperty=="function"?i.setProperty("display","none","important"):i.display="none"):(u=w.stateNode,c=w.memoizedProps.style,a=c!=null&&c.hasOwnProperty("display")?c.display:null,u.style.display=Bp("display",a))}catch(j){me(e,e.return,j)}}}else if(w.tag===6){if(E===null)try{w.stateNode.nodeValue=f?"":w.memoizedProps}catch(j){me(e,e.return,j)}}else if((w.tag!==22&&w.tag!==23||w.memoizedState===null||w===e)&&w.child!==null){w.child.return=w,w=w.child;continue}if(w===e)break e;for(;w.sibling===null;){if(w.return===null||w.return===e)break e;E===w&&(E=null),w=w.return}E===w&&(E=null),w.sibling.return=w.return,w=w.sibling}}break;case 19:ht(t,e),Ct(e),r&4&&Qf(e);break;case 21:break;default:ht(t,e),Ct(e)}}function Ct(e){var t=e.flags;if(t&2){try{e:{for(var n=e.return;n!==null;){if(dg(n)){var r=n;break e}n=n.return}throw Error(L(160))}switch(r.tag){case 5:var s=r.stateNode;r.flags&32&&(Gs(s,""),r.flags&=-33);var i=Yf(e);_u(e,i,s);break;case 3:case 4:var a=r.stateNode.containerInfo,u=Yf(e);bu(e,u,a);break;default:throw Error(L(161))}}catch(c){me(e,e.return,c)}e.flags&=-3}t&4096&&(e.flags&=-4097)}function Vw(e,t,n){z=e,pg(e)}function pg(e,t,n){for(var r=(e.mode&1)!==0;z!==null;){var s=z,i=s.child;if(s.tag===22&&r){var a=s.memoizedState!==null||Ji;if(!a){var u=s.alternate,c=u!==null&&u.memoizedState!==null||De;u=Ji;var f=De;if(Ji=a,(De=c)&&!f)for(z=s;z!==null;)a=z,c=a.child,a.tag===22&&a.memoizedState!==null?th(s):c!==null?(c.return=a,z=c):th(s);for(;i!==null;)z=i,pg(i),i=i.sibling;z=s,Ji=u,De=f}Zf(e)}else s.subtreeFlags&8772&&i!==null?(i.return=s,z=i):Zf(e)}}function Zf(e){for(;z!==null;){var t=z;if(t.flags&8772){var n=t.alternate;try{if(t.flags&8772)switch(t.tag){case 0:case 11:case 15:De||da(5,t);break;case 1:var r=t.stateNode;if(t.flags&4&&!De)if(n===null)r.componentDidMount();else{var s=t.elementType===t.type?n.memoizedProps:pt(t.type,n.memoizedProps);r.componentDidUpdate(s,n.memoizedState,r.__reactInternalSnapshotBeforeUpdate)}var i=t.updateQueue;i!==null&&Lf(t,i,r);break;case 3:var a=t.updateQueue;if(a!==null){if(n=null,t.child!==null)switch(t.child.tag){case 5:n=t.child.stateNode;break;case 1:n=t.child.stateNode}Lf(t,a,n)}break;case 5:var u=t.stateNode;if(n===null&&t.flags&4){n=u;var c=t.memoizedProps;switch(t.type){case"button":case"input":case"select":case"textarea":c.autoFocus&&n.focus();break;case"img":c.src&&(n.src=c.src)}}break;case 6:break;case 4:break;case 12:break;case 13:if(t.memoizedState===null){var f=t.alternate;if(f!==null){var E=f.memoizedState;if(E!==null){var w=E.dehydrated;w!==null&&Xs(w)}}}break;case 19:case 17:case 21:case 22:case 23:case 25:break;default:throw Error(L(163))}De||t.flags&512&&ku(t)}catch(_){me(t,t.return,_)}}if(t===e){z=null;break}if(n=t.sibling,n!==null){n.return=t.return,z=n;break}z=t.return}}function eh(e){for(;z!==null;){var t=z;if(t===e){z=null;break}var n=t.sibling;if(n!==null){n.return=t.return,z=n;break}z=t.return}}function th(e){for(;z!==null;){var t=z;try{switch(t.tag){case 0:case 11:case 15:var n=t.return;try{da(4,t)}catch(c){me(t,n,c)}break;case 1:var r=t.stateNode;if(typeof r.componentDidMount=="function"){var s=t.return;try{r.componentDidMount()}catch(c){me(t,s,c)}}var i=t.return;try{ku(t)}catch(c){me(t,i,c)}break;case 5:var a=t.return;try{ku(t)}catch(c){me(t,a,c)}}}catch(c){me(t,t.return,c)}if(t===e){z=null;break}var u=t.sibling;if(u!==null){u.return=t.return,z=u;break}z=t.return}}var $w=Math.ceil,Vo=qt.ReactCurrentDispatcher,Ic=qt.ReactCurrentOwner,ut=qt.ReactCurrentBatchConfig,J=0,Te=null,xe=null,je=0,Ye=0,br=Nn(0),be=0,oi=null,er=0,fa=0,Tc=0,Ds=null,We=null,Nc=0,Hr=1/0,Mt=null,$o=!1,Cu=null,vn=null,Yi=!1,cn=null,Ho=0,Ms=0,Iu=null,go=-1,yo=0;function ze(){return J&6?ve():go!==-1?go:go=ve()}function wn(e){return e.mode&1?J&2&&je!==0?je&-je:Cw.transition!==null?(yo===0&&(yo=Qp()),yo):(e=re,e!==0||(e=window.event,e=e===void 0?16:im(e.type)),e):1}function xt(e,t,n,r){if(50<Ms)throw Ms=0,Iu=null,Error(L(185));mi(e,n,r),(!(J&2)||e!==Te)&&(e===Te&&(!(J&2)&&(fa|=n),be===4&&ln(e,je)),Je(e,r),n===1&&J===0&&!(t.mode&1)&&(Hr=ve()+500,la&&jn()))}function Je(e,t){var n=e.callbackNode;Cv(e,t);var r=Io(e,e===Te?je:0);if(r===0)n!==null&&cf(n),e.callbackNode=null,e.callbackPriority=0;else if(t=r&-r,e.callbackPriority!==t){if(n!=null&&cf(n),t===1)e.tag===0?_w(nh.bind(null,e)):bm(nh.bind(null,e)),Sw(function(){!(J&6)&&jn()}),n=null;else{switch(Zp(r)){case 1:n=tc;break;case 4:n=Jp;break;case 16:n=Co;break;case 536870912:n=Yp;break;default:n=Co}n=Eg(n,mg.bind(null,e))}e.callbackPriority=t,e.callbackNode=n}}function mg(e,t){if(go=-1,yo=0,J&6)throw Error(L(327));var n=e.callbackNode;if(Rr()&&e.callbackNode!==n)return null;var r=Io(e,e===Te?je:0);if(r===0)return null;if(r&30||r&e.expiredLanes||t)t=Go(e,r);else{t=r;var s=J;J|=2;var i=yg();(Te!==e||je!==t)&&(Mt=null,Hr=ve()+500,Wn(e,t));do try{Ww();break}catch(u){gg(e,u)}while(!0);pc(),Vo.current=i,J=s,xe!==null?t=0:(Te=null,je=0,t=be)}if(t!==0){if(t===2&&(s=eu(e),s!==0&&(r=s,t=Tu(e,s))),t===1)throw n=oi,Wn(e,0),ln(e,r),Je(e,ve()),n;if(t===6)ln(e,r);else{if(s=e.current.alternate,!(r&30)&&!Hw(s)&&(t=Go(e,r),t===2&&(i=eu(e),i!==0&&(r=i,t=Tu(e,i))),t===1))throw n=oi,Wn(e,0),ln(e,r),Je(e,ve()),n;switch(e.finishedWork=s,e.finishedLanes=r,t){case 0:case 1:throw Error(L(345));case 2:Un(e,We,Mt);break;case 3:if(ln(e,r),(r&130023424)===r&&(t=Nc+500-ve(),10<t)){if(Io(e,0)!==0)break;if(s=e.suspendedLanes,(s&r)!==r){ze(),e.pingedLanes|=e.suspendedLanes&s;break}e.timeoutHandle=lu(Un.bind(null,e,We,Mt),t);break}Un(e,We,Mt);break;case 4:if(ln(e,r),(r&4194240)===r)break;for(t=e.eventTimes,s=-1;0<r;){var a=31-wt(r);i=1<<a,a=t[a],a>s&&(s=a),r&=~i}if(r=s,r=ve()-r,r=(120>r?120:480>r?480:1080>r?1080:1920>r?1920:3e3>r?3e3:4320>r?4320:1960*$w(r/1960))-r,10<r){e.timeoutHandle=lu(Un.bind(null,e,We,Mt),r);break}Un(e,We,Mt);break;case 5:Un(e,We,Mt);break;default:throw Error(L(329))}}}return Je(e,ve()),e.callbackNode===n?mg.bind(null,e):null}function Tu(e,t){var n=Ds;return e.current.memoizedState.isDehydrated&&(Wn(e,t).flags|=256),e=Go(e,t),e!==2&&(t=We,We=n,t!==null&&Nu(t)),e}function Nu(e){We===null?We=e:We.push.apply(We,e)}function Hw(e){for(var t=e;;){if(t.flags&16384){var n=t.updateQueue;if(n!==null&&(n=n.stores,n!==null))for(var r=0;r<n.length;r++){var s=n[r],i=s.getSnapshot;s=s.value;try{if(!Et(i(),s))return!1}catch{return!1}}}if(n=t.child,t.subtreeFlags&16384&&n!==null)n.return=t,t=n;else{if(t===e)break;for(;t.sibling===null;){if(t.return===null||t.return===e)return!0;t=t.return}t.sibling.return=t.return,t=t.sibling}}return!0}function ln(e,t){for(t&=~Tc,t&=~fa,e.suspendedLanes|=t,e.pingedLanes&=~t,e=e.expirationTimes;0<t;){var n=31-wt(t),r=1<<n;e[n]=-1,t&=~r}}function nh(e){if(J&6)throw Error(L(327));Rr();var t=Io(e,0);if(!(t&1))return Je(e,ve()),null;var n=Go(e,t);if(e.tag!==0&&n===2){var r=eu(e);r!==0&&(t=r,n=Tu(e,r))}if(n===1)throw n=oi,Wn(e,0),ln(e,t),Je(e,ve()),n;if(n===6)throw Error(L(345));return e.finishedWork=e.current.alternate,e.finishedLanes=t,Un(e,We,Mt),Je(e,ve()),null}function jc(e,t){var n=J;J|=1;try{return e(t)}finally{J=n,J===0&&(Hr=ve()+500,la&&jn())}}function tr(e){cn!==null&&cn.tag===0&&!(J&6)&&Rr();var t=J;J|=1;var n=ut.transition,r=re;try{if(ut.transition=null,re=1,e)return e()}finally{re=r,ut.transition=n,J=t,!(J&6)&&jn()}}function Ac(){Ye=br.current,ce(br)}function Wn(e,t){e.finishedWork=null,e.finishedLanes=0;var n=e.timeoutHandle;if(n!==-1&&(e.timeoutHandle=-1,xw(n)),xe!==null)for(n=xe.return;n!==null;){var r=n;switch(dc(r),r.tag){case 1:r=r.type.childContextTypes,r!=null&&Ro();break;case 3:Vr(),ce(Ke),ce(Me),xc();break;case 5:wc(r);break;case 4:Vr();break;case 13:ce(fe);break;case 19:ce(fe);break;case 10:mc(r.type._context);break;case 22:case 23:Ac()}n=n.return}if(Te=e,xe=e=xn(e.current,null),je=Ye=t,be=0,oi=null,Tc=fa=er=0,We=Ds=null,Vn!==null){for(t=0;t<Vn.length;t++)if(n=Vn[t],r=n.interleaved,r!==null){n.interleaved=null;var s=r.next,i=n.pending;if(i!==null){var a=i.next;i.next=s,r.next=a}n.pending=r}Vn=null}return e}function gg(e,t){do{var n=xe;try{if(pc(),ho.current=zo,Fo){for(var r=he.memoizedState;r!==null;){var s=r.queue;s!==null&&(s.pending=null),r=r.next}Fo=!1}if(Zn=0,Ie=ke=he=null,Ps=!1,ri=0,Ic.current=null,n===null||n.return===null){be=1,oi=t,xe=null;break}e:{var i=e,a=n.return,u=n,c=t;if(t=je,u.flags|=32768,c!==null&&typeof c=="object"&&typeof c.then=="function"){var f=c,E=u,w=E.tag;if(!(E.mode&1)&&(w===0||w===11||w===15)){var _=E.alternate;_?(E.updateQueue=_.updateQueue,E.memoizedState=_.memoizedState,E.lanes=_.lanes):(E.updateQueue=null,E.memoizedState=null)}var A=$f(a);if(A!==null){A.flags&=-257,Hf(A,a,u,i,t),A.mode&1&&Vf(i,f,t),t=A,c=f;var R=t.updateQueue;if(R===null){var j=new Set;j.add(c),t.updateQueue=j}else R.add(c);break e}else{if(!(t&1)){Vf(i,f,t),Rc();break e}c=Error(L(426))}}else if(de&&u.mode&1){var D=$f(a);if(D!==null){!(D.flags&65536)&&(D.flags|=256),Hf(D,a,u,i,t),fc($r(c,u));break e}}i=c=$r(c,u),be!==4&&(be=2),Ds===null?Ds=[i]:Ds.push(i),i=a;do{switch(i.tag){case 3:i.flags|=65536,t&=-t,i.lanes|=t;var b=Zm(i,c,t);Mf(i,b);break e;case 1:u=c;var x=i.type,C=i.stateNode;if(!(i.flags&128)&&(typeof x.getDerivedStateFromError=="function"||C!==null&&typeof C.componentDidCatch=="function"&&(vn===null||!vn.has(C)))){i.flags|=65536,t&=-t,i.lanes|=t;var O=eg(i,u,t);Mf(i,O);break e}}i=i.return}while(i!==null)}wg(n)}catch(M){t=M,xe===n&&n!==null&&(xe=n=n.return);continue}break}while(!0)}function yg(){var e=Vo.current;return Vo.current=zo,e===null?zo:e}function Rc(){(be===0||be===3||be===2)&&(be=4),Te===null||!(er&268435455)&&!(fa&268435455)||ln(Te,je)}function Go(e,t){var n=J;J|=2;var r=yg();(Te!==e||je!==t)&&(Mt=null,Wn(e,t));do try{Gw();break}catch(s){gg(e,s)}while(!0);if(pc(),J=n,Vo.current=r,xe!==null)throw Error(L(261));return Te=null,je=0,be}function Gw(){for(;xe!==null;)vg(xe)}function Ww(){for(;xe!==null&&!yv();)vg(xe)}function vg(e){var t=Sg(e.alternate,e,Ye);e.memoizedProps=e.pendingProps,t===null?wg(e):xe=t,Ic.current=null}function wg(e){var t=e;do{var n=t.alternate;if(e=t.return,t.flags&32768){if(n=Bw(n,t),n!==null){n.flags&=32767,xe=n;return}if(e!==null)e.flags|=32768,e.subtreeFlags=0,e.deletions=null;else{be=6,xe=null;return}}else if(n=Uw(n,t,Ye),n!==null){xe=n;return}if(t=t.sibling,t!==null){xe=t;return}xe=t=e}while(t!==null);be===0&&(be=5)}function Un(e,t,n){var r=re,s=ut.transition;try{ut.transition=null,re=1,qw(e,t,n,r)}finally{ut.transition=s,re=r}return null}function qw(e,t,n,r){do Rr();while(cn!==null);if(J&6)throw Error(L(327));n=e.finishedWork;var s=e.finishedLanes;if(n===null)return null;if(e.finishedWork=null,e.finishedLanes=0,n===e.current)throw Error(L(177));e.callbackNode=null,e.callbackPriority=0;var i=n.lanes|n.childLanes;if(Iv(e,i),e===Te&&(xe=Te=null,je=0),!(n.subtreeFlags&2064)&&!(n.flags&2064)||Yi||(Yi=!0,Eg(Co,function(){return Rr(),null})),i=(n.flags&15990)!==0,n.subtreeFlags&15990||i){i=ut.transition,ut.transition=null;var a=re;re=1;var u=J;J|=4,Ic.current=null,zw(e,n),hg(n,e),hw(ou),To=!!iu,ou=iu=null,e.current=n,Vw(n),vv(),J=u,re=a,ut.transition=i}else e.current=n;if(Yi&&(Yi=!1,cn=e,Ho=s),i=e.pendingLanes,i===0&&(vn=null),Sv(n.stateNode),Je(e,ve()),t!==null)for(r=e.onRecoverableError,n=0;n<t.length;n++)s=t[n],r(s.value,{componentStack:s.stack,digest:s.digest});if($o)throw $o=!1,e=Cu,Cu=null,e;return Ho&1&&e.tag!==0&&Rr(),i=e.pendingLanes,i&1?e===Iu?Ms++:(Ms=0,Iu=e):Ms=0,jn(),null}function Rr(){if(cn!==null){var e=Zp(Ho),t=ut.transition,n=re;try{if(ut.transition=null,re=16>e?16:e,cn===null)var r=!1;else{if(e=cn,cn=null,Ho=0,J&6)throw Error(L(331));var s=J;for(J|=4,z=e.current;z!==null;){var i=z,a=i.child;if(z.flags&16){var u=i.deletions;if(u!==null){for(var c=0;c<u.length;c++){var f=u[c];for(z=f;z!==null;){var E=z;switch(E.tag){case 0:case 11:case 15:Os(8,E,i)}var w=E.child;if(w!==null)w.return=E,z=w;else for(;z!==null;){E=z;var _=E.sibling,A=E.return;if(cg(E),E===f){z=null;break}if(_!==null){_.return=A,z=_;break}z=A}}}var R=i.alternate;if(R!==null){var j=R.child;if(j!==null){R.child=null;do{var D=j.sibling;j.sibling=null,j=D}while(j!==null)}}z=i}}if(i.subtreeFlags&2064&&a!==null)a.return=i,z=a;else e:for(;z!==null;){if(i=z,i.flags&2048)switch(i.tag){case 0:case 11:case 15:Os(9,i,i.return)}var b=i.sibling;if(b!==null){b.return=i.return,z=b;break e}z=i.return}}var x=e.current;for(z=x;z!==null;){a=z;var C=a.child;if(a.subtreeFlags&2064&&C!==null)C.return=a,z=C;else e:for(a=x;z!==null;){if(u=z,u.flags&2048)try{switch(u.tag){case 0:case 11:case 15:da(9,u)}}catch(M){me(u,u.return,M)}if(u===a){z=null;break e}var O=u.sibling;if(O!==null){O.return=u.return,z=O;break e}z=u.return}}if(J=s,jn(),Pt&&typeof Pt.onPostCommitFiberRoot=="function")try{Pt.onPostCommitFiberRoot(ra,e)}catch{}r=!0}return r}finally{re=n,ut.transition=t}}return!1}function rh(e,t,n){t=$r(n,t),t=Zm(e,t,1),e=yn(e,t,1),t=ze(),e!==null&&(mi(e,1,t),Je(e,t))}function me(e,t,n){if(e.tag===3)rh(e,e,n);else for(;t!==null;){if(t.tag===3){rh(t,e,n);break}else if(t.tag===1){var r=t.stateNode;if(typeof t.type.getDerivedStateFromError=="function"||typeof r.componentDidCatch=="function"&&(vn===null||!vn.has(r))){e=$r(n,e),e=eg(t,e,1),t=yn(t,e,1),e=ze(),t!==null&&(mi(t,1,e),Je(t,e));break}}t=t.return}}function Kw(e,t,n){var r=e.pingCache;r!==null&&r.delete(t),t=ze(),e.pingedLanes|=e.suspendedLanes&n,Te===e&&(je&n)===n&&(be===4||be===3&&(je&130023424)===je&&500>ve()-Nc?Wn(e,0):Tc|=n),Je(e,t)}function xg(e,t){t===0&&(e.mode&1?(t=zi,zi<<=1,!(zi&130023424)&&(zi=4194304)):t=1);var n=ze();e=Ht(e,t),e!==null&&(mi(e,t,n),Je(e,n))}function Xw(e){var t=e.memoizedState,n=0;t!==null&&(n=t.retryLane),xg(e,n)}function Jw(e,t){var n=0;switch(e.tag){case 13:var r=e.stateNode,s=e.memoizedState;s!==null&&(n=s.retryLane);break;case 19:r=e.stateNode;break;default:throw Error(L(314))}r!==null&&r.delete(t),xg(e,n)}var Sg;Sg=function(e,t,n){if(e!==null)if(e.memoizedProps!==t.pendingProps||Ke.current)qe=!0;else{if(!(e.lanes&n)&&!(t.flags&128))return qe=!1,Lw(e,t,n);qe=!!(e.flags&131072)}else qe=!1,de&&t.flags&1048576&&_m(t,Do,t.index);switch(t.lanes=0,t.tag){case 2:var r=t.type;mo(e,t),e=t.pendingProps;var s=Br(t,Me.current);Ar(t,n),s=Ec(null,t,r,e,s,n);var i=kc();return t.flags|=1,typeof s=="object"&&s!==null&&typeof s.render=="function"&&s.$$typeof===void 0?(t.tag=1,t.memoizedState=null,t.updateQueue=null,Xe(r)?(i=!0,Po(t)):i=!1,t.memoizedState=s.state!==null&&s.state!==void 0?s.state:null,yc(t),s.updater=ca,t.stateNode=s,s._reactInternals=t,mu(t,r,e,n),t=vu(null,t,r,!0,i,n)):(t.tag=0,de&&i&&cc(t),Fe(null,t,s,n),t=t.child),t;case 16:r=t.elementType;e:{switch(mo(e,t),e=t.pendingProps,s=r._init,r=s(r._payload),t.type=r,s=t.tag=Qw(r),e=pt(r,e),s){case 0:t=yu(null,t,r,e,n);break e;case 1:t=qf(null,t,r,e,n);break e;case 11:t=Gf(null,t,r,e,n);break e;case 14:t=Wf(null,t,r,pt(r.type,e),n);break e}throw Error(L(306,r,""))}return t;case 0:return r=t.type,s=t.pendingProps,s=t.elementType===r?s:pt(r,s),yu(e,t,r,s,n);case 1:return r=t.type,s=t.pendingProps,s=t.elementType===r?s:pt(r,s),qf(e,t,r,s,n);case 3:e:{if(sg(t),e===null)throw Error(L(387));r=t.pendingProps,i=t.memoizedState,s=i.element,Am(e,t),Uo(t,r,null,n);var a=t.memoizedState;if(r=a.element,i.isDehydrated)if(i={element:r,isDehydrated:!1,cache:a.cache,pendingSuspenseBoundaries:a.pendingSuspenseBoundaries,transitions:a.transitions},t.updateQueue.baseState=i,t.memoizedState=i,t.flags&256){s=$r(Error(L(423)),t),t=Kf(e,t,r,n,s);break e}else if(r!==s){s=$r(Error(L(424)),t),t=Kf(e,t,r,n,s);break e}else for(et=gn(t.stateNode.containerInfo.firstChild),tt=t,de=!0,gt=null,n=Nm(t,null,r,n),t.child=n;n;)n.flags=n.flags&-3|4096,n=n.sibling;else{if(Fr(),r===s){t=Gt(e,t,n);break e}Fe(e,t,r,n)}t=t.child}return t;case 5:return Rm(t),e===null&&fu(t),r=t.type,s=t.pendingProps,i=e!==null?e.memoizedProps:null,a=s.children,au(r,s)?a=null:i!==null&&au(r,i)&&(t.flags|=32),rg(e,t),Fe(e,t,a,n),t.child;case 6:return e===null&&fu(t),null;case 13:return ig(e,t,n);case 4:return vc(t,t.stateNode.containerInfo),r=t.pendingProps,e===null?t.child=zr(t,null,r,n):Fe(e,t,r,n),t.child;case 11:return r=t.type,s=t.pendingProps,s=t.elementType===r?s:pt(r,s),Gf(e,t,r,s,n);case 7:return Fe(e,t,t.pendingProps,n),t.child;case 8:return Fe(e,t,t.pendingProps.children,n),t.child;case 12:return Fe(e,t,t.pendingProps.children,n),t.child;case 10:e:{if(r=t.type._context,s=t.pendingProps,i=t.memoizedProps,a=s.value,oe(Mo,r._currentValue),r._currentValue=a,i!==null)if(Et(i.value,a)){if(i.children===s.children&&!Ke.current){t=Gt(e,t,n);break e}}else for(i=t.child,i!==null&&(i.return=t);i!==null;){var u=i.dependencies;if(u!==null){a=i.child;for(var c=u.firstContext;c!==null;){if(c.context===r){if(i.tag===1){c=zt(-1,n&-n),c.tag=2;var f=i.updateQueue;if(f!==null){f=f.shared;var E=f.pending;E===null?c.next=c:(c.next=E.next,E.next=c),f.pending=c}}i.lanes|=n,c=i.alternate,c!==null&&(c.lanes|=n),hu(i.return,n,t),u.lanes|=n;break}c=c.next}}else if(i.tag===10)a=i.type===t.type?null:i.child;else if(i.tag===18){if(a=i.return,a===null)throw Error(L(341));a.lanes|=n,u=a.alternate,u!==null&&(u.lanes|=n),hu(a,n,t),a=i.sibling}else a=i.child;if(a!==null)a.return=i;else for(a=i;a!==null;){if(a===t){a=null;break}if(i=a.sibling,i!==null){i.return=a.return,a=i;break}a=a.return}i=a}Fe(e,t,s.children,n),t=t.child}return t;case 9:return s=t.type,r=t.pendingProps.children,Ar(t,n),s=ct(s),r=r(s),t.flags|=1,Fe(e,t,r,n),t.child;case 14:return r=t.type,s=pt(r,t.pendingProps),s=pt(r.type,s),Wf(e,t,r,s,n);case 15:return tg(e,t,t.type,t.pendingProps,n);case 17:return r=t.type,s=t.pendingProps,s=t.elementType===r?s:pt(r,s),mo(e,t),t.tag=1,Xe(r)?(e=!0,Po(t)):e=!1,Ar(t,n),Qm(t,r,s),mu(t,r,s,n),vu(null,t,r,!0,e,n);case 19:return og(e,t,n);case 22:return ng(e,t,n)}throw Error(L(156,t.tag))};function Eg(e,t){return Xp(e,t)}function Yw(e,t,n,r){this.tag=e,this.key=n,this.sibling=this.child=this.return=this.stateNode=this.type=this.elementType=null,this.index=0,this.ref=null,this.pendingProps=t,this.dependencies=this.memoizedState=this.updateQueue=this.memoizedProps=null,this.mode=r,this.subtreeFlags=this.flags=0,this.deletions=null,this.childLanes=this.lanes=0,this.alternate=null}function lt(e,t,n,r){return new Yw(e,t,n,r)}function Pc(e){return e=e.prototype,!(!e||!e.isReactComponent)}function Qw(e){if(typeof e=="function")return Pc(e)?1:0;if(e!=null){if(e=e.$$typeof,e===Qu)return 11;if(e===Zu)return 14}return 2}function xn(e,t){var n=e.alternate;return n===null?(n=lt(e.tag,t,e.key,e.mode),n.elementType=e.elementType,n.type=e.type,n.stateNode=e.stateNode,n.alternate=e,e.alternate=n):(n.pendingProps=t,n.type=e.type,n.flags=0,n.subtreeFlags=0,n.deletions=null),n.flags=e.flags&14680064,n.childLanes=e.childLanes,n.lanes=e.lanes,n.child=e.child,n.memoizedProps=e.memoizedProps,n.memoizedState=e.memoizedState,n.updateQueue=e.updateQueue,t=e.dependencies,n.dependencies=t===null?null:{lanes:t.lanes,firstContext:t.firstContext},n.sibling=e.sibling,n.index=e.index,n.ref=e.ref,n}function vo(e,t,n,r,s,i){var a=2;if(r=e,typeof e=="function")Pc(e)&&(a=1);else if(typeof e=="string")a=5;else e:switch(e){case pr:return qn(n.children,s,i,t);case Yu:a=8,s|=8;break;case Bl:return e=lt(12,n,t,s|2),e.elementType=Bl,e.lanes=i,e;case Fl:return e=lt(13,n,t,s),e.elementType=Fl,e.lanes=i,e;case zl:return e=lt(19,n,t,s),e.elementType=zl,e.lanes=i,e;case Rp:return ha(n,s,i,t);default:if(typeof e=="object"&&e!==null)switch(e.$$typeof){case jp:a=10;break e;case Ap:a=9;break e;case Qu:a=11;break e;case Zu:a=14;break e;case sn:a=16,r=null;break e}throw Error(L(130,e==null?e:typeof e,""))}return t=lt(a,n,t,s),t.elementType=e,t.type=r,t.lanes=i,t}function qn(e,t,n,r){return e=lt(7,e,r,t),e.lanes=n,e}function ha(e,t,n,r){return e=lt(22,e,r,t),e.elementType=Rp,e.lanes=n,e.stateNode={isHidden:!1},e}function wl(e,t,n){return e=lt(6,e,null,t),e.lanes=n,e}function xl(e,t,n){return t=lt(4,e.children!==null?e.children:[],e.key,t),t.lanes=n,t.stateNode={containerInfo:e.containerInfo,pendingChildren:null,implementation:e.implementation},t}function Zw(e,t,n,r,s){this.tag=t,this.containerInfo=e,this.finishedWork=this.pingCache=this.current=this.pendingChildren=null,this.timeoutHandle=-1,this.callbackNode=this.pendingContext=this.context=null,this.callbackPriority=0,this.eventTimes=el(0),this.expirationTimes=el(-1),this.entangledLanes=this.finishedLanes=this.mutableReadLanes=this.expiredLanes=this.pingedLanes=this.suspendedLanes=this.pendingLanes=0,this.entanglements=el(0),this.identifierPrefix=r,this.onRecoverableError=s,this.mutableSourceEagerHydrationData=null}function Oc(e,t,n,r,s,i,a,u,c){return e=new Zw(e,t,n,u,c),t===1?(t=1,i===!0&&(t|=8)):t=0,i=lt(3,null,null,t),e.current=i,i.stateNode=e,i.memoizedState={element:r,isDehydrated:n,cache:null,transitions:null,pendingSuspenseBoundaries:null},yc(i),e}function e1(e,t,n){var r=3<arguments.length&&arguments[3]!==void 0?arguments[3]:null;return{$$typeof:hr,key:r==null?null:""+r,children:e,containerInfo:t,implementation:n}}function kg(e){if(!e)return Cn;e=e._reactInternals;e:{if(ir(e)!==e||e.tag!==1)throw Error(L(170));var t=e;do{switch(t.tag){case 3:t=t.stateNode.context;break e;case 1:if(Xe(t.type)){t=t.stateNode.__reactInternalMemoizedMergedChildContext;break e}}t=t.return}while(t!==null);throw Error(L(171))}if(e.tag===1){var n=e.type;if(Xe(n))return km(e,n,t)}return t}function bg(e,t,n,r,s,i,a,u,c){return e=Oc(n,r,!0,e,s,i,a,u,c),e.context=kg(null),n=e.current,r=ze(),s=wn(n),i=zt(r,s),i.callback=t??null,yn(n,i,s),e.current.lanes=s,mi(e,s,r),Je(e,r),e}function pa(e,t,n,r){var s=t.current,i=ze(),a=wn(s);return n=kg(n),t.context===null?t.context=n:t.pendingContext=n,t=zt(i,a),t.payload={element:e},r=r===void 0?null:r,r!==null&&(t.callback=r),e=yn(s,t,a),e!==null&&(xt(e,s,a,i),fo(e,s,a)),a}function Wo(e){if(e=e.current,!e.child)return null;switch(e.child.tag){case 5:return e.child.stateNode;default:return e.child.stateNode}}function sh(e,t){if(e=e.memoizedState,e!==null&&e.dehydrated!==null){var n=e.retryLane;e.retryLane=n!==0&&n<t?n:t}}function Dc(e,t){sh(e,t),(e=e.alternate)&&sh(e,t)}function t1(){return null}var _g=typeof reportError=="function"?reportError:function(e){console.error(e)};function Mc(e){this._internalRoot=e}ma.prototype.render=Mc.prototype.render=function(e){var t=this._internalRoot;if(t===null)throw Error(L(409));pa(e,t,null,null)};ma.prototype.unmount=Mc.prototype.unmount=function(){var e=this._internalRoot;if(e!==null){this._internalRoot=null;var t=e.containerInfo;tr(function(){pa(null,e,null,null)}),t[$t]=null}};function ma(e){this._internalRoot=e}ma.prototype.unstable_scheduleHydration=function(e){if(e){var t=nm();e={blockedOn:null,target:e,priority:t};for(var n=0;n<an.length&&t!==0&&t<an[n].priority;n++);an.splice(n,0,e),n===0&&sm(e)}};function Lc(e){return!(!e||e.nodeType!==1&&e.nodeType!==9&&e.nodeType!==11)}function ga(e){return!(!e||e.nodeType!==1&&e.nodeType!==9&&e.nodeType!==11&&(e.nodeType!==8||e.nodeValue!==" react-mount-point-unstable "))}function ih(){}function n1(e,t,n,r,s){if(s){if(typeof r=="function"){var i=r;r=function(){var f=Wo(a);i.call(f)}}var a=bg(t,r,e,0,null,!1,!1,"",ih);return e._reactRootContainer=a,e[$t]=a.current,Qs(e.nodeType===8?e.parentNode:e),tr(),a}for(;s=e.lastChild;)e.removeChild(s);if(typeof r=="function"){var u=r;r=function(){var f=Wo(c);u.call(f)}}var c=Oc(e,0,!1,null,null,!1,!1,"",ih);return e._reactRootContainer=c,e[$t]=c.current,Qs(e.nodeType===8?e.parentNode:e),tr(function(){pa(t,c,n,r)}),c}function ya(e,t,n,r,s){var i=n._reactRootContainer;if(i){var a=i;if(typeof s=="function"){var u=s;s=function(){var c=Wo(a);u.call(c)}}pa(t,a,e,s)}else a=n1(n,t,e,s,r);return Wo(a)}em=function(e){switch(e.tag){case 3:var t=e.stateNode;if(t.current.memoizedState.isDehydrated){var n=_s(t.pendingLanes);n!==0&&(nc(t,n|1),Je(t,ve()),!(J&6)&&(Hr=ve()+500,jn()))}break;case 13:tr(function(){var r=Ht(e,1);if(r!==null){var s=ze();xt(r,e,1,s)}}),Dc(e,1)}};rc=function(e){if(e.tag===13){var t=Ht(e,134217728);if(t!==null){var n=ze();xt(t,e,134217728,n)}Dc(e,134217728)}};tm=function(e){if(e.tag===13){var t=wn(e),n=Ht(e,t);if(n!==null){var r=ze();xt(n,e,t,r)}Dc(e,t)}};nm=function(){return re};rm=function(e,t){var n=re;try{return re=e,t()}finally{re=n}};Yl=function(e,t,n){switch(t){case"input":if(Hl(e,n),t=n.name,n.type==="radio"&&t!=null){for(n=e;n.parentNode;)n=n.parentNode;for(n=n.querySelectorAll("input[name="+JSON.stringify(""+t)+'][type="radio"]'),t=0;t<n.length;t++){var r=n[t];if(r!==e&&r.form===e.form){var s=aa(r);if(!s)throw Error(L(90));Op(r),Hl(r,s)}}}break;case"textarea":Mp(e,n);break;case"select":t=n.value,t!=null&&Ir(e,!!n.multiple,t,!1)}};$p=jc;Hp=tr;var r1={usingClientEntryPoint:!1,Events:[yi,vr,aa,zp,Vp,jc]},Es={findFiberByHostInstance:zn,bundleType:0,version:"18.3.1",rendererPackageName:"react-dom"},s1={bundleType:Es.bundleType,version:Es.version,rendererPackageName:Es.rendererPackageName,rendererConfig:Es.rendererConfig,overrideHookState:null,overrideHookStateDeletePath:null,overrideHookStateRenamePath:null,overrideProps:null,overridePropsDeletePath:null,overridePropsRenamePath:null,setErrorHandler:null,setSuspenseHandler:null,scheduleUpdate:null,currentDispatcherRef:qt.ReactCurrentDispatcher,findHostInstanceByFiber:function(e){return e=qp(e),e===null?null:e.stateNode},findFiberByHostInstance:Es.findFiberByHostInstance||t1,findHostInstancesForRefresh:null,scheduleRefresh:null,scheduleRoot:null,setRefreshHandler:null,getCurrentFiber:null,reconcilerVersion:"18.3.1-next-f1338f8080-20240426"};if(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__<"u"){var Qi=__REACT_DEVTOOLS_GLOBAL_HOOK__;if(!Qi.isDisabled&&Qi.supportsFiber)try{ra=Qi.inject(s1),Pt=Qi}catch{}}rt.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED=r1;rt.createPortal=function(e,t){var n=2<arguments.length&&arguments[2]!==void 0?arguments[2]:null;if(!Lc(t))throw Error(L(200));return e1(e,t,null,n)};rt.createRoot=function(e,t){if(!Lc(e))throw Error(L(299));var n=!1,r="",s=_g;return t!=null&&(t.unstable_strictMode===!0&&(n=!0),t.identifierPrefix!==void 0&&(r=t.identifierPrefix),t.onRecoverableError!==void 0&&(s=t.onRecoverableError)),t=Oc(e,1,!1,null,null,n,!1,r,s),e[$t]=t.current,Qs(e.nodeType===8?e.parentNode:e),new Mc(t)};rt.findDOMNode=function(e){if(e==null)return null;if(e.nodeType===1)return e;var t=e._reactInternals;if(t===void 0)throw typeof e.render=="function"?Error(L(188)):(e=Object.keys(e).join(","),Error(L(268,e)));return e=qp(t),e=e===null?null:e.stateNode,e};rt.flushSync=function(e){return tr(e)};rt.hydrate=function(e,t,n){if(!ga(t))throw Error(L(200));return ya(null,e,t,!0,n)};rt.hydrateRoot=function(e,t,n){if(!Lc(e))throw Error(L(405));var r=n!=null&&n.hydratedSources||null,s=!1,i="",a=_g;if(n!=null&&(n.unstable_strictMode===!0&&(s=!0),n.identifierPrefix!==void 0&&(i=n.identifierPrefix),n.onRecoverableError!==void 0&&(a=n.onRecoverableError)),t=bg(t,null,e,1,n??null,s,!1,i,a),e[$t]=t.current,Qs(e),r)for(e=0;e<r.length;e++)n=r[e],s=n._getVersion,s=s(n._source),t.mutableSourceEagerHydrationData==null?t.mutableSourceEagerHydrationData=[n,s]:t.mutableSourceEagerHydrationData.push(n,s);return new ma(t)};rt.render=function(e,t,n){if(!ga(t))throw Error(L(200));return ya(null,e,t,!1,n)};rt.unmountComponentAtNode=function(e){if(!ga(e))throw Error(L(40));return e._reactRootContainer?(tr(function(){ya(null,null,e,!1,function(){e._reactRootContainer=null,e[$t]=null})}),!0):!1};rt.unstable_batchedUpdates=jc;rt.unstable_renderSubtreeIntoContainer=function(e,t,n,r){if(!ga(n))throw Error(L(200));if(e==null||e._reactInternals===void 0)throw Error(L(38));return ya(e,t,n,!1,r)};rt.version="18.3.1-next-f1338f8080-20240426";function Cg(){if(!(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__>"u"||typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE!="function"))try{__REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE(Cg)}catch(e){console.error(e)}}Cg(),Cp.exports=rt;var i1=Cp.exports,oh=i1;Ll.createRoot=oh.createRoot,Ll.hydrateRoot=oh.hydrateRoot;/**
 * @remix-run/router v1.23.2
 *
 * Copyright (c) Remix Software Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 *
 * @license MIT
 */function ai(){return ai=Object.assign?Object.assign.bind():function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var r in n)Object.prototype.hasOwnProperty.call(n,r)&&(e[r]=n[r])}return e},ai.apply(this,arguments)}var dn;(function(e){e.Pop="POP",e.Push="PUSH",e.Replace="REPLACE"})(dn||(dn={}));const ah="popstate";function o1(e){e===void 0&&(e={});function t(r,s){let{pathname:i,search:a,hash:u}=r.location;return ju("",{pathname:i,search:a,hash:u},s.state&&s.state.usr||null,s.state&&s.state.key||"default")}function n(r,s){return typeof s=="string"?s:qo(s)}return l1(t,n,null,e)}function we(e,t){if(e===!1||e===null||typeof e>"u")throw new Error(t)}function Uc(e,t){if(!e){typeof console<"u"&&console.warn(t);try{throw new Error(t)}catch{}}}function a1(){return Math.random().toString(36).substr(2,8)}function lh(e,t){return{usr:e.state,key:e.key,idx:t}}function ju(e,t,n,r){return n===void 0&&(n=null),ai({pathname:typeof e=="string"?e:e.pathname,search:"",hash:""},typeof t=="string"?Jr(t):t,{state:n,key:t&&t.key||r||a1()})}function qo(e){let{pathname:t="/",search:n="",hash:r=""}=e;return n&&n!=="?"&&(t+=n.charAt(0)==="?"?n:"?"+n),r&&r!=="#"&&(t+=r.charAt(0)==="#"?r:"#"+r),t}function Jr(e){let t={};if(e){let n=e.indexOf("#");n>=0&&(t.hash=e.substr(n),e=e.substr(0,n));let r=e.indexOf("?");r>=0&&(t.search=e.substr(r),e=e.substr(0,r)),e&&(t.pathname=e)}return t}function l1(e,t,n,r){r===void 0&&(r={});let{window:s=document.defaultView,v5Compat:i=!1}=r,a=s.history,u=dn.Pop,c=null,f=E();f==null&&(f=0,a.replaceState(ai({},a.state,{idx:f}),""));function E(){return(a.state||{idx:null}).idx}function w(){u=dn.Pop;let D=E(),b=D==null?null:D-f;f=D,c&&c({action:u,location:j.location,delta:b})}function _(D,b){u=dn.Push;let x=ju(j.location,D,b);f=E()+1;let C=lh(x,f),O=j.createHref(x);try{a.pushState(C,"",O)}catch(M){if(M instanceof DOMException&&M.name==="DataCloneError")throw M;s.location.assign(O)}i&&c&&c({action:u,location:j.location,delta:1})}function A(D,b){u=dn.Replace;let x=ju(j.location,D,b);f=E();let C=lh(x,f),O=j.createHref(x);a.replaceState(C,"",O),i&&c&&c({action:u,location:j.location,delta:0})}function R(D){let b=s.location.origin!=="null"?s.location.origin:s.location.href,x=typeof D=="string"?D:qo(D);return x=x.replace(/ $/,"%20"),we(b,"No window.location.(origin|href) available to create URL for href: "+x),new URL(x,b)}let j={get action(){return u},get location(){return e(s,a)},listen(D){if(c)throw new Error("A history only accepts one active listener");return s.addEventListener(ah,w),c=D,()=>{s.removeEventListener(ah,w),c=null}},createHref(D){return t(s,D)},createURL:R,encodeLocation(D){let b=R(D);return{pathname:b.pathname,search:b.search,hash:b.hash}},push:_,replace:A,go(D){return a.go(D)}};return j}var uh;(function(e){e.data="data",e.deferred="deferred",e.redirect="redirect",e.error="error"})(uh||(uh={}));function u1(e,t,n){return n===void 0&&(n="/"),c1(e,t,n)}function c1(e,t,n,r){let s=typeof t=="string"?Jr(t):t,i=Bc(s.pathname||"/",n);if(i==null)return null;let a=Ig(e);d1(a);let u=null;for(let c=0;u==null&&c<a.length;++c){let f=k1(i);u=x1(a[c],f)}return u}function Ig(e,t,n,r){t===void 0&&(t=[]),n===void 0&&(n=[]),r===void 0&&(r="");let s=(i,a,u)=>{let c={relativePath:u===void 0?i.path||"":u,caseSensitive:i.caseSensitive===!0,childrenIndex:a,route:i};c.relativePath.startsWith("/")&&(we(c.relativePath.startsWith(r),'Absolute route path "'+c.relativePath+'" nested under path '+('"'+r+'" is not valid. An absolute child route path ')+"must start with the combined path of all its parent routes."),c.relativePath=c.relativePath.slice(r.length));let f=Sn([r,c.relativePath]),E=n.concat(c);i.children&&i.children.length>0&&(we(i.index!==!0,"Index routes must not have child routes. Please remove "+('all child routes from route path "'+f+'".')),Ig(i.children,t,E,f)),!(i.path==null&&!i.index)&&t.push({path:f,score:v1(f,i.index),routesMeta:E})};return e.forEach((i,a)=>{var u;if(i.path===""||!((u=i.path)!=null&&u.includes("?")))s(i,a);else for(let c of Tg(i.path))s(i,a,c)}),t}function Tg(e){let t=e.split("/");if(t.length===0)return[];let[n,...r]=t,s=n.endsWith("?"),i=n.replace(/\?$/,"");if(r.length===0)return s?[i,""]:[i];let a=Tg(r.join("/")),u=[];return u.push(...a.map(c=>c===""?i:[i,c].join("/"))),s&&u.push(...a),u.map(c=>e.startsWith("/")&&c===""?"/":c)}function d1(e){e.sort((t,n)=>t.score!==n.score?n.score-t.score:w1(t.routesMeta.map(r=>r.childrenIndex),n.routesMeta.map(r=>r.childrenIndex)))}const f1=/^:[\w-]+$/,h1=3,p1=2,m1=1,g1=10,y1=-2,ch=e=>e==="*";function v1(e,t){let n=e.split("/"),r=n.length;return n.some(ch)&&(r+=y1),t&&(r+=p1),n.filter(s=>!ch(s)).reduce((s,i)=>s+(f1.test(i)?h1:i===""?m1:g1),r)}function w1(e,t){return e.length===t.length&&e.slice(0,-1).every((r,s)=>r===t[s])?e[e.length-1]-t[t.length-1]:0}function x1(e,t,n){let{routesMeta:r}=e,s={},i="/",a=[];for(let u=0;u<r.length;++u){let c=r[u],f=u===r.length-1,E=i==="/"?t:t.slice(i.length)||"/",w=S1({path:c.relativePath,caseSensitive:c.caseSensitive,end:f},E),_=c.route;if(!w)return null;Object.assign(s,w.params),a.push({params:s,pathname:Sn([i,w.pathname]),pathnameBase:T1(Sn([i,w.pathnameBase])),route:_}),w.pathnameBase!=="/"&&(i=Sn([i,w.pathnameBase]))}return a}function S1(e,t){typeof e=="string"&&(e={path:e,caseSensitive:!1,end:!0});let[n,r]=E1(e.path,e.caseSensitive,e.end),s=t.match(n);if(!s)return null;let i=s[0],a=i.replace(/(.)\/+$/,"$1"),u=s.slice(1);return{params:r.reduce((f,E,w)=>{let{paramName:_,isOptional:A}=E;if(_==="*"){let j=u[w]||"";a=i.slice(0,i.length-j.length).replace(/(.)\/+$/,"$1")}const R=u[w];return A&&!R?f[_]=void 0:f[_]=(R||"").replace(/%2F/g,"/"),f},{}),pathname:i,pathnameBase:a,pattern:e}}function E1(e,t,n){t===void 0&&(t=!1),n===void 0&&(n=!0),Uc(e==="*"||!e.endsWith("*")||e.endsWith("/*"),'Route path "'+e+'" will be treated as if it were '+('"'+e.replace(/\*$/,"/*")+'" because the `*` character must ')+"always follow a `/` in the pattern. To get rid of this warning, "+('please change the route path to "'+e.replace(/\*$/,"/*")+'".'));let r=[],s="^"+e.replace(/\/*\*?$/,"").replace(/^\/*/,"/").replace(/[\\.*+^${}|()[\]]/g,"\\$&").replace(/\/:([\w-]+)(\?)?/g,(a,u,c)=>(r.push({paramName:u,isOptional:c!=null}),c?"/?([^\\/]+)?":"/([^\\/]+)"));return e.endsWith("*")?(r.push({paramName:"*"}),s+=e==="*"||e==="/*"?"(.*)$":"(?:\\/(.+)|\\/*)$"):n?s+="\\/*$":e!==""&&e!=="/"&&(s+="(?:(?=\\/|$))"),[new RegExp(s,t?void 0:"i"),r]}function k1(e){try{return e.split("/").map(t=>decodeURIComponent(t).replace(/\//g,"%2F")).join("/")}catch(t){return Uc(!1,'The URL path "'+e+'" could not be decoded because it is is a malformed URL segment. This is probably due to a bad percent '+("encoding ("+t+").")),e}}function Bc(e,t){if(t==="/")return e;if(!e.toLowerCase().startsWith(t.toLowerCase()))return null;let n=t.endsWith("/")?t.length-1:t.length,r=e.charAt(n);return r&&r!=="/"?null:e.slice(n)||"/"}const b1=/^(?:[a-z][a-z0-9+.-]*:|\/\/)/i,_1=e=>b1.test(e);function C1(e,t){t===void 0&&(t="/");let{pathname:n,search:r="",hash:s=""}=typeof e=="string"?Jr(e):e,i;if(n)if(_1(n))i=n;else{if(n.includes("//")){let a=n;n=n.replace(/\/\/+/g,"/"),Uc(!1,"Pathnames cannot have embedded double slashes - normalizing "+(a+" -> "+n))}n.startsWith("/")?i=dh(n.substring(1),"/"):i=dh(n,t)}else i=t;return{pathname:i,search:N1(r),hash:j1(s)}}function dh(e,t){let n=t.replace(/\/+$/,"").split("/");return e.split("/").forEach(s=>{s===".."?n.length>1&&n.pop():s!=="."&&n.push(s)}),n.length>1?n.join("/"):"/"}function Sl(e,t,n,r){return"Cannot include a '"+e+"' character in a manually specified "+("`to."+t+"` field ["+JSON.stringify(r)+"].  Please separate it out to the ")+("`to."+n+"` field. Alternatively you may provide the full path as ")+'a string in <Link to="..."> and the router will parse it for you.'}function I1(e){return e.filter((t,n)=>n===0||t.route.path&&t.route.path.length>0)}function Fc(e,t){let n=I1(e);return t?n.map((r,s)=>s===n.length-1?r.pathname:r.pathnameBase):n.map(r=>r.pathnameBase)}function zc(e,t,n,r){r===void 0&&(r=!1);let s;typeof e=="string"?s=Jr(e):(s=ai({},e),we(!s.pathname||!s.pathname.includes("?"),Sl("?","pathname","search",s)),we(!s.pathname||!s.pathname.includes("#"),Sl("#","pathname","hash",s)),we(!s.search||!s.search.includes("#"),Sl("#","search","hash",s)));let i=e===""||s.pathname==="",a=i?"/":s.pathname,u;if(a==null)u=n;else{let w=t.length-1;if(!r&&a.startsWith("..")){let _=a.split("/");for(;_[0]==="..";)_.shift(),w-=1;s.pathname=_.join("/")}u=w>=0?t[w]:"/"}let c=C1(s,u),f=a&&a!=="/"&&a.endsWith("/"),E=(i||a===".")&&n.endsWith("/");return!c.pathname.endsWith("/")&&(f||E)&&(c.pathname+="/"),c}const Sn=e=>e.join("/").replace(/\/\/+/g,"/"),T1=e=>e.replace(/\/+$/,"").replace(/^\/*/,"/"),N1=e=>!e||e==="?"?"":e.startsWith("?")?e:"?"+e,j1=e=>!e||e==="#"?"":e.startsWith("#")?e:"#"+e;function A1(e){return e!=null&&typeof e.status=="number"&&typeof e.statusText=="string"&&typeof e.internal=="boolean"&&"data"in e}const Ng=["post","put","patch","delete"];new Set(Ng);const R1=["get",...Ng];new Set(R1);/**
 * React Router v6.30.3
 *
 * Copyright (c) Remix Software Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 *
 * @license MIT
 */function li(){return li=Object.assign?Object.assign.bind():function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var r in n)Object.prototype.hasOwnProperty.call(n,r)&&(e[r]=n[r])}return e},li.apply(this,arguments)}const Vc=N.createContext(null),P1=N.createContext(null),An=N.createContext(null),va=N.createContext(null),Kt=N.createContext({outlet:null,matches:[],isDataRoute:!1}),jg=N.createContext(null);function O1(e,t){let{relative:n}=t===void 0?{}:t;Yr()||we(!1);let{basename:r,navigator:s}=N.useContext(An),{hash:i,pathname:a,search:u}=Rg(e,{relative:n}),c=a;return r!=="/"&&(c=a==="/"?r:Sn([r,a])),s.createHref({pathname:c,search:u,hash:i})}function Yr(){return N.useContext(va)!=null}function wi(){return Yr()||we(!1),N.useContext(va).location}function Ag(e){N.useContext(An).static||N.useLayoutEffect(e)}function Qr(){let{isDataRoute:e}=N.useContext(Kt);return e?K1():D1()}function D1(){Yr()||we(!1);let e=N.useContext(Vc),{basename:t,future:n,navigator:r}=N.useContext(An),{matches:s}=N.useContext(Kt),{pathname:i}=wi(),a=JSON.stringify(Fc(s,n.v7_relativeSplatPath)),u=N.useRef(!1);return Ag(()=>{u.current=!0}),N.useCallback(function(f,E){if(E===void 0&&(E={}),!u.current)return;if(typeof f=="number"){r.go(f);return}let w=zc(f,JSON.parse(a),i,E.relative==="path");e==null&&t!=="/"&&(w.pathname=w.pathname==="/"?t:Sn([t,w.pathname])),(E.replace?r.replace:r.push)(w,E.state,E)},[t,r,a,i,e])}function M1(){let{matches:e}=N.useContext(Kt),t=e[e.length-1];return t?t.params:{}}function Rg(e,t){let{relative:n}=t===void 0?{}:t,{future:r}=N.useContext(An),{matches:s}=N.useContext(Kt),{pathname:i}=wi(),a=JSON.stringify(Fc(s,r.v7_relativeSplatPath));return N.useMemo(()=>zc(e,JSON.parse(a),i,n==="path"),[e,a,i,n])}function L1(e,t){return U1(e,t)}function U1(e,t,n,r){Yr()||we(!1);let{navigator:s}=N.useContext(An),{matches:i}=N.useContext(Kt),a=i[i.length-1],u=a?a.params:{};a&&a.pathname;let c=a?a.pathnameBase:"/";a&&a.route;let f=wi(),E;if(t){var w;let D=typeof t=="string"?Jr(t):t;c==="/"||(w=D.pathname)!=null&&w.startsWith(c)||we(!1),E=D}else E=f;let _=E.pathname||"/",A=_;if(c!=="/"){let D=c.replace(/^\//,"").split("/");A="/"+_.replace(/^\//,"").split("/").slice(D.length).join("/")}let R=u1(e,{pathname:A}),j=$1(R&&R.map(D=>Object.assign({},D,{params:Object.assign({},u,D.params),pathname:Sn([c,s.encodeLocation?s.encodeLocation(D.pathname).pathname:D.pathname]),pathnameBase:D.pathnameBase==="/"?c:Sn([c,s.encodeLocation?s.encodeLocation(D.pathnameBase).pathname:D.pathnameBase])})),i,n,r);return t&&j?N.createElement(va.Provider,{value:{location:li({pathname:"/",search:"",hash:"",state:null,key:"default"},E),navigationType:dn.Pop}},j):j}function B1(){let e=q1(),t=A1(e)?e.status+" "+e.statusText:e instanceof Error?e.message:JSON.stringify(e),n=e instanceof Error?e.stack:null,s={padding:"0.5rem",backgroundColor:"rgba(200,200,200, 0.5)"};return N.createElement(N.Fragment,null,N.createElement("h2",null,"Unexpected Application Error!"),N.createElement("h3",{style:{fontStyle:"italic"}},t),n?N.createElement("pre",{style:s},n):null,null)}const F1=N.createElement(B1,null);class z1 extends N.Component{constructor(t){super(t),this.state={location:t.location,revalidation:t.revalidation,error:t.error}}static getDerivedStateFromError(t){return{error:t}}static getDerivedStateFromProps(t,n){return n.location!==t.location||n.revalidation!=="idle"&&t.revalidation==="idle"?{error:t.error,location:t.location,revalidation:t.revalidation}:{error:t.error!==void 0?t.error:n.error,location:n.location,revalidation:t.revalidation||n.revalidation}}componentDidCatch(t,n){console.error("React Router caught the following error during render",t,n)}render(){return this.state.error!==void 0?N.createElement(Kt.Provider,{value:this.props.routeContext},N.createElement(jg.Provider,{value:this.state.error,children:this.props.component})):this.props.children}}function V1(e){let{routeContext:t,match:n,children:r}=e,s=N.useContext(Vc);return s&&s.static&&s.staticContext&&(n.route.errorElement||n.route.ErrorBoundary)&&(s.staticContext._deepestRenderedBoundaryId=n.route.id),N.createElement(Kt.Provider,{value:t},r)}function $1(e,t,n,r){var s;if(t===void 0&&(t=[]),n===void 0&&(n=null),r===void 0&&(r=null),e==null){var i;if(!n)return null;if(n.errors)e=n.matches;else if((i=r)!=null&&i.v7_partialHydration&&t.length===0&&!n.initialized&&n.matches.length>0)e=n.matches;else return null}let a=e,u=(s=n)==null?void 0:s.errors;if(u!=null){let E=a.findIndex(w=>w.route.id&&(u==null?void 0:u[w.route.id])!==void 0);E>=0||we(!1),a=a.slice(0,Math.min(a.length,E+1))}let c=!1,f=-1;if(n&&r&&r.v7_partialHydration)for(let E=0;E<a.length;E++){let w=a[E];if((w.route.HydrateFallback||w.route.hydrateFallbackElement)&&(f=E),w.route.id){let{loaderData:_,errors:A}=n,R=w.route.loader&&_[w.route.id]===void 0&&(!A||A[w.route.id]===void 0);if(w.route.lazy||R){c=!0,f>=0?a=a.slice(0,f+1):a=[a[0]];break}}}return a.reduceRight((E,w,_)=>{let A,R=!1,j=null,D=null;n&&(A=u&&w.route.id?u[w.route.id]:void 0,j=w.route.errorElement||F1,c&&(f<0&&_===0?(X1("route-fallback"),R=!0,D=null):f===_&&(R=!0,D=w.route.hydrateFallbackElement||null)));let b=t.concat(a.slice(0,_+1)),x=()=>{let C;return A?C=j:R?C=D:w.route.Component?C=N.createElement(w.route.Component,null):w.route.element?C=w.route.element:C=E,N.createElement(V1,{match:w,routeContext:{outlet:E,matches:b,isDataRoute:n!=null},children:C})};return n&&(w.route.ErrorBoundary||w.route.errorElement||_===0)?N.createElement(z1,{location:n.location,revalidation:n.revalidation,component:j,error:A,children:x(),routeContext:{outlet:null,matches:b,isDataRoute:!0}}):x()},null)}var Pg=function(e){return e.UseBlocker="useBlocker",e.UseRevalidator="useRevalidator",e.UseNavigateStable="useNavigate",e}(Pg||{}),Og=function(e){return e.UseBlocker="useBlocker",e.UseLoaderData="useLoaderData",e.UseActionData="useActionData",e.UseRouteError="useRouteError",e.UseNavigation="useNavigation",e.UseRouteLoaderData="useRouteLoaderData",e.UseMatches="useMatches",e.UseRevalidator="useRevalidator",e.UseNavigateStable="useNavigate",e.UseRouteId="useRouteId",e}(Og||{});function H1(e){let t=N.useContext(Vc);return t||we(!1),t}function G1(e){let t=N.useContext(P1);return t||we(!1),t}function W1(e){let t=N.useContext(Kt);return t||we(!1),t}function Dg(e){let t=W1(),n=t.matches[t.matches.length-1];return n.route.id||we(!1),n.route.id}function q1(){var e;let t=N.useContext(jg),n=G1(),r=Dg();return t!==void 0?t:(e=n.errors)==null?void 0:e[r]}function K1(){let{router:e}=H1(Pg.UseNavigateStable),t=Dg(Og.UseNavigateStable),n=N.useRef(!1);return Ag(()=>{n.current=!0}),N.useCallback(function(s,i){i===void 0&&(i={}),n.current&&(typeof s=="number"?e.navigate(s):e.navigate(s,li({fromRouteId:t},i)))},[e,t])}const fh={};function X1(e,t,n){fh[e]||(fh[e]=!0)}function J1(e,t){e==null||e.v7_startTransition,e==null||e.v7_relativeSplatPath}function Mg(e){let{to:t,replace:n,state:r,relative:s}=e;Yr()||we(!1);let{future:i,static:a}=N.useContext(An),{matches:u}=N.useContext(Kt),{pathname:c}=wi(),f=Qr(),E=zc(t,Fc(u,i.v7_relativeSplatPath),c,s==="path"),w=JSON.stringify(E);return N.useEffect(()=>f(JSON.parse(w),{replace:n,state:r,relative:s}),[f,w,s,n,r]),null}function dr(e){we(!1)}function Y1(e){let{basename:t="/",children:n=null,location:r,navigationType:s=dn.Pop,navigator:i,static:a=!1,future:u}=e;Yr()&&we(!1);let c=t.replace(/^\/*/,"/"),f=N.useMemo(()=>({basename:c,navigator:i,static:a,future:li({v7_relativeSplatPath:!1},u)}),[c,u,i,a]);typeof r=="string"&&(r=Jr(r));let{pathname:E="/",search:w="",hash:_="",state:A=null,key:R="default"}=r,j=N.useMemo(()=>{let D=Bc(E,c);return D==null?null:{location:{pathname:D,search:w,hash:_,state:A,key:R},navigationType:s}},[c,E,w,_,A,R,s]);return j==null?null:N.createElement(An.Provider,{value:f},N.createElement(va.Provider,{children:n,value:j}))}function Q1(e){let{children:t,location:n}=e;return L1(Au(t),n)}new Promise(()=>{});function Au(e,t){t===void 0&&(t=[]);let n=[];return N.Children.forEach(e,(r,s)=>{if(!N.isValidElement(r))return;let i=[...t,s];if(r.type===N.Fragment){n.push.apply(n,Au(r.props.children,i));return}r.type!==dr&&we(!1),!r.props.index||!r.props.children||we(!1);let a={id:r.props.id||i.join("-"),caseSensitive:r.props.caseSensitive,element:r.props.element,Component:r.props.Component,index:r.props.index,path:r.props.path,loader:r.props.loader,action:r.props.action,errorElement:r.props.errorElement,ErrorBoundary:r.props.ErrorBoundary,hasErrorBoundary:r.props.ErrorBoundary!=null||r.props.errorElement!=null,shouldRevalidate:r.props.shouldRevalidate,handle:r.props.handle,lazy:r.props.lazy};r.props.children&&(a.children=Au(r.props.children,i)),n.push(a)}),n}/**
 * React Router DOM v6.30.3
 *
 * Copyright (c) Remix Software Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 *
 * @license MIT
 */function Ru(){return Ru=Object.assign?Object.assign.bind():function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var r in n)Object.prototype.hasOwnProperty.call(n,r)&&(e[r]=n[r])}return e},Ru.apply(this,arguments)}function Z1(e,t){if(e==null)return{};var n={},r=Object.keys(e),s,i;for(i=0;i<r.length;i++)s=r[i],!(t.indexOf(s)>=0)&&(n[s]=e[s]);return n}function ex(e){return!!(e.metaKey||e.altKey||e.ctrlKey||e.shiftKey)}function tx(e,t){return e.button===0&&(!t||t==="_self")&&!ex(e)}const nx=["onClick","relative","reloadDocument","replace","state","target","to","preventScrollReset","viewTransition"],rx="6";try{window.__reactRouterVersion=rx}catch{}const sx="startTransition",hh=K0[sx];function ix(e){let{basename:t,children:n,future:r,window:s}=e,i=N.useRef();i.current==null&&(i.current=o1({window:s,v5Compat:!0}));let a=i.current,[u,c]=N.useState({action:a.action,location:a.location}),{v7_startTransition:f}=r||{},E=N.useCallback(w=>{f&&hh?hh(()=>c(w)):c(w)},[c,f]);return N.useLayoutEffect(()=>a.listen(E),[a,E]),N.useEffect(()=>J1(r),[r]),N.createElement(Y1,{basename:t,children:n,location:u.location,navigationType:u.action,navigator:a,future:r})}const ox=typeof window<"u"&&typeof window.document<"u"&&typeof window.document.createElement<"u",ax=/^(?:[a-z][a-z0-9+.-]*:|\/\/)/i,xi=N.forwardRef(function(t,n){let{onClick:r,relative:s,reloadDocument:i,replace:a,state:u,target:c,to:f,preventScrollReset:E,viewTransition:w}=t,_=Z1(t,nx),{basename:A}=N.useContext(An),R,j=!1;if(typeof f=="string"&&ax.test(f)&&(R=f,ox))try{let C=new URL(window.location.href),O=f.startsWith("//")?new URL(C.protocol+f):new URL(f),M=Bc(O.pathname,A);O.origin===C.origin&&M!=null?f=M+O.search+O.hash:j=!0}catch{}let D=O1(f,{relative:s}),b=lx(f,{replace:a,state:u,target:c,preventScrollReset:E,relative:s,viewTransition:w});function x(C){r&&r(C),C.defaultPrevented||b(C)}return N.createElement("a",Ru({},_,{href:R||D,onClick:j||i?r:x,ref:n,target:c}))});var ph;(function(e){e.UseScrollRestoration="useScrollRestoration",e.UseSubmit="useSubmit",e.UseSubmitFetcher="useSubmitFetcher",e.UseFetcher="useFetcher",e.useViewTransitionState="useViewTransitionState"})(ph||(ph={}));var mh;(function(e){e.UseFetcher="useFetcher",e.UseFetchers="useFetchers",e.UseScrollRestoration="useScrollRestoration"})(mh||(mh={}));function lx(e,t){let{target:n,replace:r,state:s,preventScrollReset:i,relative:a,viewTransition:u}=t===void 0?{}:t,c=Qr(),f=wi(),E=Rg(e,{relative:a});return N.useCallback(w=>{if(tx(w,n)){w.preventDefault();let _=r!==void 0?r:qo(f)===qo(E);c(e,{replace:_,state:s,preventScrollReset:i,relative:a,viewTransition:u})}},[f,c,E,r,s,n,e,i,a,u])}const ux=()=>{};var gh={};/**
 * @license
 * Copyright 2017 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const Lg=function(e){const t=[];let n=0;for(let r=0;r<e.length;r++){let s=e.charCodeAt(r);s<128?t[n++]=s:s<2048?(t[n++]=s>>6|192,t[n++]=s&63|128):(s&64512)===55296&&r+1<e.length&&(e.charCodeAt(r+1)&64512)===56320?(s=65536+((s&1023)<<10)+(e.charCodeAt(++r)&1023),t[n++]=s>>18|240,t[n++]=s>>12&63|128,t[n++]=s>>6&63|128,t[n++]=s&63|128):(t[n++]=s>>12|224,t[n++]=s>>6&63|128,t[n++]=s&63|128)}return t},cx=function(e){const t=[];let n=0,r=0;for(;n<e.length;){const s=e[n++];if(s<128)t[r++]=String.fromCharCode(s);else if(s>191&&s<224){const i=e[n++];t[r++]=String.fromCharCode((s&31)<<6|i&63)}else if(s>239&&s<365){const i=e[n++],a=e[n++],u=e[n++],c=((s&7)<<18|(i&63)<<12|(a&63)<<6|u&63)-65536;t[r++]=String.fromCharCode(55296+(c>>10)),t[r++]=String.fromCharCode(56320+(c&1023))}else{const i=e[n++],a=e[n++];t[r++]=String.fromCharCode((s&15)<<12|(i&63)<<6|a&63)}}return t.join("")},Ug={byteToCharMap_:null,charToByteMap_:null,byteToCharMapWebSafe_:null,charToByteMapWebSafe_:null,ENCODED_VALS_BASE:"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",get ENCODED_VALS(){return this.ENCODED_VALS_BASE+"+/="},get ENCODED_VALS_WEBSAFE(){return this.ENCODED_VALS_BASE+"-_."},HAS_NATIVE_SUPPORT:typeof atob=="function",encodeByteArray(e,t){if(!Array.isArray(e))throw Error("encodeByteArray takes an array as a parameter");this.init_();const n=t?this.byteToCharMapWebSafe_:this.byteToCharMap_,r=[];for(let s=0;s<e.length;s+=3){const i=e[s],a=s+1<e.length,u=a?e[s+1]:0,c=s+2<e.length,f=c?e[s+2]:0,E=i>>2,w=(i&3)<<4|u>>4;let _=(u&15)<<2|f>>6,A=f&63;c||(A=64,a||(_=64)),r.push(n[E],n[w],n[_],n[A])}return r.join("")},encodeString(e,t){return this.HAS_NATIVE_SUPPORT&&!t?btoa(e):this.encodeByteArray(Lg(e),t)},decodeString(e,t){return this.HAS_NATIVE_SUPPORT&&!t?atob(e):cx(this.decodeStringToByteArray(e,t))},decodeStringToByteArray(e,t){this.init_();const n=t?this.charToByteMapWebSafe_:this.charToByteMap_,r=[];for(let s=0;s<e.length;){const i=n[e.charAt(s++)],u=s<e.length?n[e.charAt(s)]:0;++s;const f=s<e.length?n[e.charAt(s)]:64;++s;const w=s<e.length?n[e.charAt(s)]:64;if(++s,i==null||u==null||f==null||w==null)throw new dx;const _=i<<2|u>>4;if(r.push(_),f!==64){const A=u<<4&240|f>>2;if(r.push(A),w!==64){const R=f<<6&192|w;r.push(R)}}}return r},init_(){if(!this.byteToCharMap_){this.byteToCharMap_={},this.charToByteMap_={},this.byteToCharMapWebSafe_={},this.charToByteMapWebSafe_={};for(let e=0;e<this.ENCODED_VALS.length;e++)this.byteToCharMap_[e]=this.ENCODED_VALS.charAt(e),this.charToByteMap_[this.byteToCharMap_[e]]=e,this.byteToCharMapWebSafe_[e]=this.ENCODED_VALS_WEBSAFE.charAt(e),this.charToByteMapWebSafe_[this.byteToCharMapWebSafe_[e]]=e,e>=this.ENCODED_VALS_BASE.length&&(this.charToByteMap_[this.ENCODED_VALS_WEBSAFE.charAt(e)]=e,this.charToByteMapWebSafe_[this.ENCODED_VALS.charAt(e)]=e)}}};class dx extends Error{constructor(){super(...arguments),this.name="DecodeBase64StringError"}}const fx=function(e){const t=Lg(e);return Ug.encodeByteArray(t,!0)},Bg=function(e){return fx(e).replace(/\./g,"")},Fg=function(e){try{return Ug.decodeString(e,!0)}catch(t){console.error("base64Decode failed: ",t)}return null};/**
 * @license
 * Copyright 2022 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */function hx(){if(typeof self<"u")return self;if(typeof window<"u")return window;if(typeof global<"u")return global;throw new Error("Unable to locate global object.")}/**
 * @license
 * Copyright 2022 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const px=()=>hx().__FIREBASE_DEFAULTS__,mx=()=>{if(typeof process>"u"||typeof gh>"u")return;const e=gh.__FIREBASE_DEFAULTS__;if(e)return JSON.parse(e)},gx=()=>{if(typeof document>"u")return;let e;try{e=document.cookie.match(/__FIREBASE_DEFAULTS__=([^;]+)/)}catch{return}const t=e&&Fg(e[1]);return t&&JSON.parse(t)},yx=()=>{try{return ux()||px()||mx()||gx()}catch(e){console.info(`Unable to get __FIREBASE_DEFAULTS__ due to: ${e}`);return}},vx=e=>{var t;return(t=yx())==null?void 0:t[`_${e}`]};/**
 * @license
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */function zg(e){try{return(e.startsWith("http://")||e.startsWith("https://")?new URL(e).hostname:e).endsWith(".cloudworkstations.dev")}catch{return!1}}/**
 * @license
 * Copyright 2017 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */function Dt(){return typeof navigator<"u"&&typeof navigator.userAgent=="string"?navigator.userAgent:""}function wx(){return typeof window<"u"&&!!(window.cordova||window.phonegap||window.PhoneGap)&&/ios|iphone|ipod|ipad|android|blackberry|iemobile/i.test(Dt())}function xx(){return typeof navigator<"u"&&navigator.userAgent==="Cloudflare-Workers"}function Sx(){const e=typeof chrome=="object"?chrome.runtime:typeof browser=="object"?browser.runtime:void 0;return typeof e=="object"&&e.id!==void 0}function Ex(){return typeof navigator=="object"&&navigator.product==="ReactNative"}function kx(){try{return typeof indexedDB=="object"}catch{return!1}}function bx(){return new Promise((e,t)=>{try{let n=!0;const r="validate-browser-context-for-indexeddb-analytics-module",s=self.indexedDB.open(r);s.onsuccess=()=>{s.result.close(),n||self.indexedDB.deleteDatabase(r),e(!0)},s.onupgradeneeded=()=>{n=!1},s.onerror=()=>{var i;t(((i=s.error)==null?void 0:i.message)||"")}}catch(n){t(n)}})}/**
 * @license
 * Copyright 2017 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const _x="FirebaseError";class Rn extends Error{constructor(t,n,r){super(n),this.code=t,this.customData=r,this.name=_x,Object.setPrototypeOf(this,Rn.prototype),Error.captureStackTrace&&Error.captureStackTrace(this,Si.prototype.create)}}class Si{constructor(t,n,r){this.service=t,this.serviceName=n,this.errors=r}create(t,...n){const r=n[0]||{},s=`${this.service}/${t}`,i=this.errors[t],a=i?Cx(i,r):"Error",u=`${this.serviceName}: ${a} (${s}).`;return new Rn(s,u,r)}}function Cx(e,t){return e.replace(Ix,(n,r)=>{const s=t[r];return s!=null?String(s):`<${r}?>`})}const Ix=/\{\$([^}]+)}/g;/**
 * @license
 * Copyright 2017 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */function Vg(e){const t=[];for(const[n,r]of Object.entries(e))Array.isArray(r)?r.forEach(s=>{t.push(encodeURIComponent(n)+"="+encodeURIComponent(s))}):t.push(encodeURIComponent(n)+"="+encodeURIComponent(r));return t.length?"&"+t.join("&"):""}function Tx(e,t){const n=new Nx(e,t);return n.subscribe.bind(n)}class Nx{constructor(t,n){this.observers=[],this.unsubscribes=[],this.observerCount=0,this.task=Promise.resolve(),this.finalized=!1,this.onNoObservers=n,this.task.then(()=>{t(this)}).catch(r=>{this.error(r)})}next(t){this.forEachObserver(n=>{n.next(t)})}error(t){this.forEachObserver(n=>{n.error(t)}),this.close(t)}complete(){this.forEachObserver(t=>{t.complete()}),this.close()}subscribe(t,n,r){let s;if(t===void 0&&n===void 0&&r===void 0)throw new Error("Missing Observer.");jx(t,["next","error","complete"])?s=t:s={next:t,error:n,complete:r},s.next===void 0&&(s.next=El),s.error===void 0&&(s.error=El),s.complete===void 0&&(s.complete=El);const i=this.unsubscribeOne.bind(this,this.observers.length);return this.finalized&&this.task.then(()=>{try{this.finalError?s.error(this.finalError):s.complete()}catch{}}),this.observers.push(s),i}unsubscribeOne(t){this.observers===void 0||this.observers[t]===void 0||(delete this.observers[t],this.observerCount-=1,this.observerCount===0&&this.onNoObservers!==void 0&&this.onNoObservers(this))}forEachObserver(t){if(!this.finalized)for(let n=0;n<this.observers.length;n++)this.sendOne(n,t)}sendOne(t,n){this.task.then(()=>{if(this.observers!==void 0&&this.observers[t]!==void 0)try{n(this.observers[t])}catch(r){typeof console<"u"&&console.error&&console.error(r)}})}close(t){this.finalized||(this.finalized=!0,t!==void 0&&(this.finalError=t),this.task.then(()=>{this.observers=void 0,this.onNoObservers=void 0}))}}function jx(e,t){if(typeof e!="object"||e===null)return!1;for(const n of t)if(n in e&&typeof e[n]=="function")return!0;return!1}function El(){}/**
 * @license
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */function Ei(e){return e&&e._delegate?e._delegate:e}class Gr{constructor(t,n,r){this.name=t,this.instanceFactory=n,this.type=r,this.multipleInstances=!1,this.serviceProps={},this.instantiationMode="LAZY",this.onInstanceCreated=null}setInstantiationMode(t){return this.instantiationMode=t,this}setMultipleInstances(t){return this.multipleInstances=t,this}setServiceProps(t){return this.serviceProps=t,this}setInstanceCreatedCallback(t){return this.onInstanceCreated=t,this}}/**
 * @license
 * Copyright 2017 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */var ne;(function(e){e[e.DEBUG=0]="DEBUG",e[e.VERBOSE=1]="VERBOSE",e[e.INFO=2]="INFO",e[e.WARN=3]="WARN",e[e.ERROR=4]="ERROR",e[e.SILENT=5]="SILENT"})(ne||(ne={}));const Ax={debug:ne.DEBUG,verbose:ne.VERBOSE,info:ne.INFO,warn:ne.WARN,error:ne.ERROR,silent:ne.SILENT},Rx=ne.INFO,Px={[ne.DEBUG]:"log",[ne.VERBOSE]:"log",[ne.INFO]:"info",[ne.WARN]:"warn",[ne.ERROR]:"error"},Ox=(e,t,...n)=>{if(t<e.logLevel)return;const r=new Date().toISOString(),s=Px[t];if(s)console[s](`[${r}]  ${e.name}:`,...n);else throw new Error(`Attempted to log a message with an invalid logType (value: ${t})`)};class $c{constructor(t){this.name=t,this._logLevel=Rx,this._logHandler=Ox,this._userLogHandler=null}get logLevel(){return this._logLevel}set logLevel(t){if(!(t in ne))throw new TypeError(`Invalid value "${t}" assigned to \`logLevel\``);this._logLevel=t}setLogLevel(t){this._logLevel=typeof t=="string"?Ax[t]:t}get logHandler(){return this._logHandler}set logHandler(t){if(typeof t!="function")throw new TypeError("Value assigned to `logHandler` must be a function");this._logHandler=t}get userLogHandler(){return this._userLogHandler}set userLogHandler(t){this._userLogHandler=t}debug(...t){this._userLogHandler&&this._userLogHandler(this,ne.DEBUG,...t),this._logHandler(this,ne.DEBUG,...t)}log(...t){this._userLogHandler&&this._userLogHandler(this,ne.VERBOSE,...t),this._logHandler(this,ne.VERBOSE,...t)}info(...t){this._userLogHandler&&this._userLogHandler(this,ne.INFO,...t),this._logHandler(this,ne.INFO,...t)}warn(...t){this._userLogHandler&&this._userLogHandler(this,ne.WARN,...t),this._logHandler(this,ne.WARN,...t)}error(...t){this._userLogHandler&&this._userLogHandler(this,ne.ERROR,...t),this._logHandler(this,ne.ERROR,...t)}}const Dx=(e,t)=>t.some(n=>e instanceof n);let yh,vh;function Mx(){return yh||(yh=[IDBDatabase,IDBObjectStore,IDBIndex,IDBCursor,IDBTransaction])}function Lx(){return vh||(vh=[IDBCursor.prototype.advance,IDBCursor.prototype.continue,IDBCursor.prototype.continuePrimaryKey])}const $g=new WeakMap,Pu=new WeakMap,Hg=new WeakMap,kl=new WeakMap,Hc=new WeakMap;function Ux(e){const t=new Promise((n,r)=>{const s=()=>{e.removeEventListener("success",i),e.removeEventListener("error",a)},i=()=>{n(En(e.result)),s()},a=()=>{r(e.error),s()};e.addEventListener("success",i),e.addEventListener("error",a)});return t.then(n=>{n instanceof IDBCursor&&$g.set(n,e)}).catch(()=>{}),Hc.set(t,e),t}function Bx(e){if(Pu.has(e))return;const t=new Promise((n,r)=>{const s=()=>{e.removeEventListener("complete",i),e.removeEventListener("error",a),e.removeEventListener("abort",a)},i=()=>{n(),s()},a=()=>{r(e.error||new DOMException("AbortError","AbortError")),s()};e.addEventListener("complete",i),e.addEventListener("error",a),e.addEventListener("abort",a)});Pu.set(e,t)}let Ou={get(e,t,n){if(e instanceof IDBTransaction){if(t==="done")return Pu.get(e);if(t==="objectStoreNames")return e.objectStoreNames||Hg.get(e);if(t==="store")return n.objectStoreNames[1]?void 0:n.objectStore(n.objectStoreNames[0])}return En(e[t])},set(e,t,n){return e[t]=n,!0},has(e,t){return e instanceof IDBTransaction&&(t==="done"||t==="store")?!0:t in e}};function Fx(e){Ou=e(Ou)}function zx(e){return e===IDBDatabase.prototype.transaction&&!("objectStoreNames"in IDBTransaction.prototype)?function(t,...n){const r=e.call(bl(this),t,...n);return Hg.set(r,t.sort?t.sort():[t]),En(r)}:Lx().includes(e)?function(...t){return e.apply(bl(this),t),En($g.get(this))}:function(...t){return En(e.apply(bl(this),t))}}function Vx(e){return typeof e=="function"?zx(e):(e instanceof IDBTransaction&&Bx(e),Dx(e,Mx())?new Proxy(e,Ou):e)}function En(e){if(e instanceof IDBRequest)return Ux(e);if(kl.has(e))return kl.get(e);const t=Vx(e);return t!==e&&(kl.set(e,t),Hc.set(t,e)),t}const bl=e=>Hc.get(e);function $x(e,t,{blocked:n,upgrade:r,blocking:s,terminated:i}={}){const a=indexedDB.open(e,t),u=En(a);return r&&a.addEventListener("upgradeneeded",c=>{r(En(a.result),c.oldVersion,c.newVersion,En(a.transaction),c)}),n&&a.addEventListener("blocked",c=>n(c.oldVersion,c.newVersion,c)),u.then(c=>{i&&c.addEventListener("close",()=>i()),s&&c.addEventListener("versionchange",f=>s(f.oldVersion,f.newVersion,f))}).catch(()=>{}),u}const Hx=["get","getKey","getAll","getAllKeys","count"],Gx=["put","add","delete","clear"],_l=new Map;function wh(e,t){if(!(e instanceof IDBDatabase&&!(t in e)&&typeof t=="string"))return;if(_l.get(t))return _l.get(t);const n=t.replace(/FromIndex$/,""),r=t!==n,s=Gx.includes(n);if(!(n in(r?IDBIndex:IDBObjectStore).prototype)||!(s||Hx.includes(n)))return;const i=async function(a,...u){const c=this.transaction(a,s?"readwrite":"readonly");let f=c.store;return r&&(f=f.index(u.shift())),(await Promise.all([f[n](...u),s&&c.done]))[0]};return _l.set(t,i),i}Fx(e=>({...e,get:(t,n,r)=>wh(t,n)||e.get(t,n,r),has:(t,n)=>!!wh(t,n)||e.has(t,n)}));/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class Wx{constructor(t){this.container=t}getPlatformInfoString(){return this.container.getProviders().map(n=>{if(qx(n)){const r=n.getImmediate();return`${r.library}/${r.version}`}else return null}).filter(n=>n).join(" ")}}function qx(e){const t=e.getComponent();return(t==null?void 0:t.type)==="VERSION"}const Du="@firebase/app",xh="0.14.9";/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const Wt=new $c("@firebase/app"),Kx="@firebase/app-compat",Xx="@firebase/analytics-compat",Jx="@firebase/analytics",Yx="@firebase/app-check-compat",Qx="@firebase/app-check",Zx="@firebase/auth",eS="@firebase/auth-compat",tS="@firebase/database",nS="@firebase/data-connect",rS="@firebase/database-compat",sS="@firebase/functions",iS="@firebase/functions-compat",oS="@firebase/installations",aS="@firebase/installations-compat",lS="@firebase/messaging",uS="@firebase/messaging-compat",cS="@firebase/performance",dS="@firebase/performance-compat",fS="@firebase/remote-config",hS="@firebase/remote-config-compat",pS="@firebase/storage",mS="@firebase/storage-compat",gS="@firebase/firestore",yS="@firebase/ai",vS="@firebase/firestore-compat",wS="firebase",xS="12.10.0",SS={[Du]:"fire-core",[Kx]:"fire-core-compat",[Jx]:"fire-analytics",[Xx]:"fire-analytics-compat",[Qx]:"fire-app-check",[Yx]:"fire-app-check-compat",[Zx]:"fire-auth",[eS]:"fire-auth-compat",[tS]:"fire-rtdb",[nS]:"fire-data-connect",[rS]:"fire-rtdb-compat",[sS]:"fire-fn",[iS]:"fire-fn-compat",[oS]:"fire-iid",[aS]:"fire-iid-compat",[lS]:"fire-fcm",[uS]:"fire-fcm-compat",[cS]:"fire-perf",[dS]:"fire-perf-compat",[fS]:"fire-rc",[hS]:"fire-rc-compat",[pS]:"fire-gcs",[mS]:"fire-gcs-compat",[gS]:"fire-fst",[vS]:"fire-fst-compat",[yS]:"fire-vertex","fire-js":"fire-js",[wS]:"fire-js-all"};/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const ES=new Map,kS=new Map,Sh=new Map;function Eh(e,t){try{e.container.addComponent(t)}catch(n){Wt.debug(`Component ${t.name} failed to register with FirebaseApp ${e.name}`,n)}}function Wr(e){const t=e.name;if(Sh.has(t))return Wt.debug(`There were multiple attempts to register component ${t}.`),!1;Sh.set(t,e);for(const n of ES.values())Eh(n,e);for(const n of kS.values())Eh(n,e);return!0}function Bn(e){return e==null?!1:e.settings!==void 0}/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const bS={"no-app":"No Firebase App '{$appName}' has been created - call initializeApp() first","bad-app-name":"Illegal App name: '{$appName}'","duplicate-app":"Firebase App named '{$appName}' already exists with different options or config","app-deleted":"Firebase App named '{$appName}' already deleted","server-app-deleted":"Firebase Server App has been deleted","no-options":"Need to provide options, when not being deployed to hosting via source.","invalid-app-argument":"firebase.{$appName}() takes either no argument or a Firebase App instance.","invalid-log-argument":"First argument to `onLog` must be null or a function.","idb-open":"Error thrown when opening IndexedDB. Original error: {$originalErrorMessage}.","idb-get":"Error thrown when reading from IndexedDB. Original error: {$originalErrorMessage}.","idb-set":"Error thrown when writing to IndexedDB. Original error: {$originalErrorMessage}.","idb-delete":"Error thrown when deleting from IndexedDB. Original error: {$originalErrorMessage}.","finalization-registry-not-supported":"FirebaseServerApp deleteOnDeref field defined but the JS runtime does not support FinalizationRegistry.","invalid-server-app-environment":"FirebaseServerApp is not for use in browser environments."},Gc=new Si("app","Firebase",bS);/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const wa=xS;function kn(e,t,n){let r=SS[e]??e;n&&(r+=`-${n}`);const s=r.match(/\s|\//),i=t.match(/\s|\//);if(s||i){const a=[`Unable to register library "${r}" with version "${t}":`];s&&a.push(`library name "${r}" contains illegal characters (whitespace or "/")`),s&&i&&a.push("and"),i&&a.push(`version name "${t}" contains illegal characters (whitespace or "/")`),Wt.warn(a.join(" "));return}Wr(new Gr(`${r}-version`,()=>({library:r,version:t}),"VERSION"))}/**
 * @license
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const _S="firebase-heartbeat-database",CS=1,ui="firebase-heartbeat-store";let Cl=null;function Gg(){return Cl||(Cl=$x(_S,CS,{upgrade:(e,t)=>{switch(t){case 0:try{e.createObjectStore(ui)}catch(n){console.warn(n)}}}}).catch(e=>{throw Gc.create("idb-open",{originalErrorMessage:e.message})})),Cl}async function IS(e){try{const n=(await Gg()).transaction(ui),r=await n.objectStore(ui).get(Wg(e));return await n.done,r}catch(t){if(t instanceof Rn)Wt.warn(t.message);else{const n=Gc.create("idb-get",{originalErrorMessage:t==null?void 0:t.message});Wt.warn(n.message)}}}async function kh(e,t){try{const r=(await Gg()).transaction(ui,"readwrite");await r.objectStore(ui).put(t,Wg(e)),await r.done}catch(n){if(n instanceof Rn)Wt.warn(n.message);else{const r=Gc.create("idb-set",{originalErrorMessage:n==null?void 0:n.message});Wt.warn(r.message)}}}function Wg(e){return`${e.name}!${e.options.appId}`}/**
 * @license
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const TS=1024,NS=30;class jS{constructor(t){this.container=t,this._heartbeatsCache=null;const n=this.container.getProvider("app").getImmediate();this._storage=new RS(n),this._heartbeatsCachePromise=this._storage.read().then(r=>(this._heartbeatsCache=r,r))}async triggerHeartbeat(){var t,n;try{const s=this.container.getProvider("platform-logger").getImmediate().getPlatformInfoString(),i=bh();if(((t=this._heartbeatsCache)==null?void 0:t.heartbeats)==null&&(this._heartbeatsCache=await this._heartbeatsCachePromise,((n=this._heartbeatsCache)==null?void 0:n.heartbeats)==null)||this._heartbeatsCache.lastSentHeartbeatDate===i||this._heartbeatsCache.heartbeats.some(a=>a.date===i))return;if(this._heartbeatsCache.heartbeats.push({date:i,agent:s}),this._heartbeatsCache.heartbeats.length>NS){const a=PS(this._heartbeatsCache.heartbeats);this._heartbeatsCache.heartbeats.splice(a,1)}return this._storage.overwrite(this._heartbeatsCache)}catch(r){Wt.warn(r)}}async getHeartbeatsHeader(){var t;try{if(this._heartbeatsCache===null&&await this._heartbeatsCachePromise,((t=this._heartbeatsCache)==null?void 0:t.heartbeats)==null||this._heartbeatsCache.heartbeats.length===0)return"";const n=bh(),{heartbeatsToSend:r,unsentEntries:s}=AS(this._heartbeatsCache.heartbeats),i=Bg(JSON.stringify({version:2,heartbeats:r}));return this._heartbeatsCache.lastSentHeartbeatDate=n,s.length>0?(this._heartbeatsCache.heartbeats=s,await this._storage.overwrite(this._heartbeatsCache)):(this._heartbeatsCache.heartbeats=[],this._storage.overwrite(this._heartbeatsCache)),i}catch(n){return Wt.warn(n),""}}}function bh(){return new Date().toISOString().substring(0,10)}function AS(e,t=TS){const n=[];let r=e.slice();for(const s of e){const i=n.find(a=>a.agent===s.agent);if(i){if(i.dates.push(s.date),_h(n)>t){i.dates.pop();break}}else if(n.push({agent:s.agent,dates:[s.date]}),_h(n)>t){n.pop();break}r=r.slice(1)}return{heartbeatsToSend:n,unsentEntries:r}}class RS{constructor(t){this.app=t,this._canUseIndexedDBPromise=this.runIndexedDBEnvironmentCheck()}async runIndexedDBEnvironmentCheck(){return kx()?bx().then(()=>!0).catch(()=>!1):!1}async read(){if(await this._canUseIndexedDBPromise){const n=await IS(this.app);return n!=null&&n.heartbeats?n:{heartbeats:[]}}else return{heartbeats:[]}}async overwrite(t){if(await this._canUseIndexedDBPromise){const r=await this.read();return kh(this.app,{lastSentHeartbeatDate:t.lastSentHeartbeatDate??r.lastSentHeartbeatDate,heartbeats:t.heartbeats})}else return}async add(t){if(await this._canUseIndexedDBPromise){const r=await this.read();return kh(this.app,{lastSentHeartbeatDate:t.lastSentHeartbeatDate??r.lastSentHeartbeatDate,heartbeats:[...r.heartbeats,...t.heartbeats]})}else return}}function _h(e){return Bg(JSON.stringify({version:2,heartbeats:e})).length}function PS(e){if(e.length===0)return-1;let t=0,n=e[0].date;for(let r=1;r<e.length;r++)e[r].date<n&&(n=e[r].date,t=r);return t}/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */function OS(e){Wr(new Gr("platform-logger",t=>new Wx(t),"PRIVATE")),Wr(new Gr("heartbeat",t=>new jS(t),"PRIVATE")),kn(Du,xh,e),kn(Du,xh,"esm2020"),kn("fire-js","")}OS("");var DS="firebase",MS="12.10.0";/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */kn(DS,MS,"app");function qg(){return{"dependent-sdk-initialized-before-auth":"Another Firebase SDK was initialized and is trying to use Auth before Auth is initialized. Please be sure to call `initializeAuth` or `getAuth` before starting any other Firebase SDK."}}const LS=qg,Kg=new Si("auth","Firebase",qg());/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const Ko=new $c("@firebase/auth");function US(e,...t){Ko.logLevel<=ne.WARN&&Ko.warn(`Auth (${wa}): ${e}`,...t)}function wo(e,...t){Ko.logLevel<=ne.ERROR&&Ko.error(`Auth (${wa}): ${e}`,...t)}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */function Ch(e,...t){throw Wc(e,...t)}function Xg(e,...t){return Wc(e,...t)}function Jg(e,t,n){const r={...LS(),[t]:n};return new Si("auth","Firebase",r).create(t,{appName:e.name})}function xo(e){return Jg(e,"operation-not-supported-in-this-environment","Operations that alter the current user are not supported in conjunction with FirebaseServerApp")}function Wc(e,...t){if(typeof e!="string"){const n=t[0],r=[...t.slice(1)];return r[0]&&(r[0].appName=e.name),e._errorFactory.create(n,...r)}return Kg.create(e,...t)}function Q(e,t,...n){if(!e)throw Wc(t,...n)}function Ls(e){const t="INTERNAL ASSERTION FAILED: "+e;throw wo(t),new Error(t)}function Xo(e,t){e||Ls(t)}function BS(){return Ih()==="http:"||Ih()==="https:"}function Ih(){var e;return typeof self<"u"&&((e=self.location)==null?void 0:e.protocol)||null}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */function FS(){return typeof navigator<"u"&&navigator&&"onLine"in navigator&&typeof navigator.onLine=="boolean"&&(BS()||Sx()||"connection"in navigator)?navigator.onLine:!0}function zS(){if(typeof navigator>"u")return null;const e=navigator;return e.languages&&e.languages[0]||e.language||null}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class ki{constructor(t,n){this.shortDelay=t,this.longDelay=n,Xo(n>t,"Short delay should be less than long delay!"),this.isMobile=wx()||Ex()}get(){return FS()?this.isMobile?this.longDelay:this.shortDelay:Math.min(5e3,this.shortDelay)}}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */function VS(e,t){Xo(e.emulator,"Emulator should always be set here");const{url:n}=e.emulator;return t?`${n}${t.startsWith("/")?t.slice(1):t}`:n}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class Yg{static initialize(t,n,r){this.fetchImpl=t,n&&(this.headersImpl=n),r&&(this.responseImpl=r)}static fetch(){if(this.fetchImpl)return this.fetchImpl;if(typeof self<"u"&&"fetch"in self)return self.fetch;if(typeof globalThis<"u"&&globalThis.fetch)return globalThis.fetch;if(typeof fetch<"u")return fetch;Ls("Could not find fetch implementation, make sure you call FetchProvider.initialize() with an appropriate polyfill")}static headers(){if(this.headersImpl)return this.headersImpl;if(typeof self<"u"&&"Headers"in self)return self.Headers;if(typeof globalThis<"u"&&globalThis.Headers)return globalThis.Headers;if(typeof Headers<"u")return Headers;Ls("Could not find Headers implementation, make sure you call FetchProvider.initialize() with an appropriate polyfill")}static response(){if(this.responseImpl)return this.responseImpl;if(typeof self<"u"&&"Response"in self)return self.Response;if(typeof globalThis<"u"&&globalThis.Response)return globalThis.Response;if(typeof Response<"u")return Response;Ls("Could not find Response implementation, make sure you call FetchProvider.initialize() with an appropriate polyfill")}}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const $S={CREDENTIAL_MISMATCH:"custom-token-mismatch",MISSING_CUSTOM_TOKEN:"internal-error",INVALID_IDENTIFIER:"invalid-email",MISSING_CONTINUE_URI:"internal-error",INVALID_PASSWORD:"wrong-password",MISSING_PASSWORD:"missing-password",INVALID_LOGIN_CREDENTIALS:"invalid-credential",EMAIL_EXISTS:"email-already-in-use",PASSWORD_LOGIN_DISABLED:"operation-not-allowed",INVALID_IDP_RESPONSE:"invalid-credential",INVALID_PENDING_TOKEN:"invalid-credential",FEDERATED_USER_ID_ALREADY_LINKED:"credential-already-in-use",MISSING_REQ_TYPE:"internal-error",EMAIL_NOT_FOUND:"user-not-found",RESET_PASSWORD_EXCEED_LIMIT:"too-many-requests",EXPIRED_OOB_CODE:"expired-action-code",INVALID_OOB_CODE:"invalid-action-code",MISSING_OOB_CODE:"internal-error",CREDENTIAL_TOO_OLD_LOGIN_AGAIN:"requires-recent-login",INVALID_ID_TOKEN:"invalid-user-token",TOKEN_EXPIRED:"user-token-expired",USER_NOT_FOUND:"user-token-expired",TOO_MANY_ATTEMPTS_TRY_LATER:"too-many-requests",PASSWORD_DOES_NOT_MEET_REQUIREMENTS:"password-does-not-meet-requirements",INVALID_CODE:"invalid-verification-code",INVALID_SESSION_INFO:"invalid-verification-id",INVALID_TEMPORARY_PROOF:"invalid-credential",MISSING_SESSION_INFO:"missing-verification-id",SESSION_EXPIRED:"code-expired",MISSING_ANDROID_PACKAGE_NAME:"missing-android-pkg-name",UNAUTHORIZED_DOMAIN:"unauthorized-continue-uri",INVALID_OAUTH_CLIENT_ID:"invalid-oauth-client-id",ADMIN_ONLY_OPERATION:"admin-restricted-operation",INVALID_MFA_PENDING_CREDENTIAL:"invalid-multi-factor-session",MFA_ENROLLMENT_NOT_FOUND:"multi-factor-info-not-found",MISSING_MFA_ENROLLMENT_ID:"missing-multi-factor-info",MISSING_MFA_PENDING_CREDENTIAL:"missing-multi-factor-session",SECOND_FACTOR_EXISTS:"second-factor-already-in-use",SECOND_FACTOR_LIMIT_EXCEEDED:"maximum-second-factor-count-exceeded",BLOCKING_FUNCTION_ERROR_RESPONSE:"internal-error",RECAPTCHA_NOT_ENABLED:"recaptcha-not-enabled",MISSING_RECAPTCHA_TOKEN:"missing-recaptcha-token",INVALID_RECAPTCHA_TOKEN:"invalid-recaptcha-token",INVALID_RECAPTCHA_ACTION:"invalid-recaptcha-action",MISSING_CLIENT_TYPE:"missing-client-type",MISSING_RECAPTCHA_VERSION:"missing-recaptcha-version",INVALID_RECAPTCHA_VERSION:"invalid-recaptcha-version",INVALID_REQ_TYPE:"invalid-req-type"};/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const HS=["/v1/accounts:signInWithCustomToken","/v1/accounts:signInWithEmailLink","/v1/accounts:signInWithIdp","/v1/accounts:signInWithPassword","/v1/accounts:signInWithPhoneNumber","/v1/token"],GS=new ki(3e4,6e4);function Qg(e,t){return e.tenantId&&!t.tenantId?{...t,tenantId:e.tenantId}:t}async function xa(e,t,n,r,s={}){return Zg(e,s,async()=>{let i={},a={};r&&(t==="GET"?a=r:i={body:JSON.stringify(r)});const u=Vg({key:e.config.apiKey,...a}).slice(1),c=await e._getAdditionalHeaders();c["Content-Type"]="application/json",e.languageCode&&(c["X-Firebase-Locale"]=e.languageCode);const f={method:t,headers:c,...i};return xx()||(f.referrerPolicy="no-referrer"),e.emulatorConfig&&zg(e.emulatorConfig.host)&&(f.credentials="include"),Yg.fetch()(await ey(e,e.config.apiHost,n,u),f)})}async function Zg(e,t,n){e._canInitEmulator=!1;const r={...$S,...t};try{const s=new WS(e),i=await Promise.race([n(),s.promise]);s.clearNetworkTimeout();const a=await i.json();if("needConfirmation"in a)throw Zi(e,"account-exists-with-different-credential",a);if(i.ok&&!("errorMessage"in a))return a;{const u=i.ok?a.errorMessage:a.error.message,[c,f]=u.split(" : ");if(c==="FEDERATED_USER_ID_ALREADY_LINKED")throw Zi(e,"credential-already-in-use",a);if(c==="EMAIL_EXISTS")throw Zi(e,"email-already-in-use",a);if(c==="USER_DISABLED")throw Zi(e,"user-disabled",a);const E=r[c]||c.toLowerCase().replace(/[_\s]+/g,"-");if(f)throw Jg(e,E,f);Ch(e,E)}}catch(s){if(s instanceof Rn)throw s;Ch(e,"network-request-failed",{message:String(s)})}}async function ey(e,t,n,r){const s=`${t}${n}?${r}`,i=e,a=i.config.emulator?VS(e.config,s):`${e.config.apiScheme}://${s}`;return HS.includes(n)&&(await i._persistenceManagerAvailable,i._getPersistenceType()==="COOKIE")?i._getPersistence()._getFinalTarget(a).toString():a}class WS{clearNetworkTimeout(){clearTimeout(this.timer)}constructor(t){this.auth=t,this.timer=null,this.promise=new Promise((n,r)=>{this.timer=setTimeout(()=>r(Xg(this.auth,"network-request-failed")),GS.get())})}}function Zi(e,t,n){const r={appName:e.name};n.email&&(r.email=n.email),n.phoneNumber&&(r.phoneNumber=n.phoneNumber);const s=Xg(e,t,r);return s.customData._tokenResponse=n,s}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */async function qS(e,t){return xa(e,"POST","/v1/accounts:delete",t)}async function Jo(e,t){return xa(e,"POST","/v1/accounts:lookup",t)}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */function Us(e){if(e)try{const t=new Date(Number(e));if(!isNaN(t.getTime()))return t.toUTCString()}catch{}}async function KS(e,t=!1){const n=Ei(e),r=await n.getIdToken(t),s=ty(r);Q(s&&s.exp&&s.auth_time&&s.iat,n.auth,"internal-error");const i=typeof s.firebase=="object"?s.firebase:void 0,a=i==null?void 0:i.sign_in_provider;return{claims:s,token:r,authTime:Us(Il(s.auth_time)),issuedAtTime:Us(Il(s.iat)),expirationTime:Us(Il(s.exp)),signInProvider:a||null,signInSecondFactor:(i==null?void 0:i.sign_in_second_factor)||null}}function Il(e){return Number(e)*1e3}function ty(e){const[t,n,r]=e.split(".");if(t===void 0||n===void 0||r===void 0)return wo("JWT malformed, contained fewer than 3 sections"),null;try{const s=Fg(n);return s?JSON.parse(s):(wo("Failed to decode base64 JWT payload"),null)}catch(s){return wo("Caught error parsing JWT payload as JSON",s==null?void 0:s.toString()),null}}function Th(e){const t=ty(e);return Q(t,"internal-error"),Q(typeof t.exp<"u","internal-error"),Q(typeof t.iat<"u","internal-error"),Number(t.exp)-Number(t.iat)}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */async function Mu(e,t,n=!1){if(n)return t;try{return await t}catch(r){throw r instanceof Rn&&XS(r)&&e.auth.currentUser===e&&await e.auth.signOut(),r}}function XS({code:e}){return e==="auth/user-disabled"||e==="auth/user-token-expired"}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class JS{constructor(t){this.user=t,this.isRunning=!1,this.timerId=null,this.errorBackoff=3e4}_start(){this.isRunning||(this.isRunning=!0,this.schedule())}_stop(){this.isRunning&&(this.isRunning=!1,this.timerId!==null&&clearTimeout(this.timerId))}getInterval(t){if(t){const n=this.errorBackoff;return this.errorBackoff=Math.min(this.errorBackoff*2,96e4),n}else{this.errorBackoff=3e4;const r=(this.user.stsTokenManager.expirationTime??0)-Date.now()-3e5;return Math.max(0,r)}}schedule(t=!1){if(!this.isRunning)return;const n=this.getInterval(t);this.timerId=setTimeout(async()=>{await this.iteration()},n)}async iteration(){try{await this.user.getIdToken(!0)}catch(t){(t==null?void 0:t.code)==="auth/network-request-failed"&&this.schedule(!0);return}this.schedule()}}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class Lu{constructor(t,n){this.createdAt=t,this.lastLoginAt=n,this._initializeTime()}_initializeTime(){this.lastSignInTime=Us(this.lastLoginAt),this.creationTime=Us(this.createdAt)}_copy(t){this.createdAt=t.createdAt,this.lastLoginAt=t.lastLoginAt,this._initializeTime()}toJSON(){return{createdAt:this.createdAt,lastLoginAt:this.lastLoginAt}}}/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */async function Yo(e){var w;const t=e.auth,n=await e.getIdToken(),r=await Mu(e,Jo(t,{idToken:n}));Q(r==null?void 0:r.users.length,t,"internal-error");const s=r.users[0];e._notifyReloadListener(s);const i=(w=s.providerUserInfo)!=null&&w.length?ny(s.providerUserInfo):[],a=QS(e.providerData,i),u=e.isAnonymous,c=!(e.email&&s.passwordHash)&&!(a!=null&&a.length),f=u?c:!1,E={uid:s.localId,displayName:s.displayName||null,photoURL:s.photoUrl||null,email:s.email||null,emailVerified:s.emailVerified||!1,phoneNumber:s.phoneNumber||null,tenantId:s.tenantId||null,providerData:a,metadata:new Lu(s.createdAt,s.lastLoginAt),isAnonymous:f};Object.assign(e,E)}async function YS(e){const t=Ei(e);await Yo(t),await t.auth._persistUserIfCurrent(t),t.auth._notifyListenersIfCurrent(t)}function QS(e,t){return[...e.filter(r=>!t.some(s=>s.providerId===r.providerId)),...t]}function ny(e){return e.map(({providerId:t,...n})=>({providerId:t,uid:n.rawId||"",displayName:n.displayName||null,email:n.email||null,phoneNumber:n.phoneNumber||null,photoURL:n.photoUrl||null}))}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */async function ZS(e,t){const n=await Zg(e,{},async()=>{const r=Vg({grant_type:"refresh_token",refresh_token:t}).slice(1),{tokenApiHost:s,apiKey:i}=e.config,a=await ey(e,s,"/v1/token",`key=${i}`),u=await e._getAdditionalHeaders();u["Content-Type"]="application/x-www-form-urlencoded";const c={method:"POST",headers:u,body:r};return e.emulatorConfig&&zg(e.emulatorConfig.host)&&(c.credentials="include"),Yg.fetch()(a,c)});return{accessToken:n.access_token,expiresIn:n.expires_in,refreshToken:n.refresh_token}}async function eE(e,t){return xa(e,"POST","/v2/accounts:revokeToken",Qg(e,t))}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class Pr{constructor(){this.refreshToken=null,this.accessToken=null,this.expirationTime=null}get isExpired(){return!this.expirationTime||Date.now()>this.expirationTime-3e4}updateFromServerResponse(t){Q(t.idToken,"internal-error"),Q(typeof t.idToken<"u","internal-error"),Q(typeof t.refreshToken<"u","internal-error");const n="expiresIn"in t&&typeof t.expiresIn<"u"?Number(t.expiresIn):Th(t.idToken);this.updateTokensAndExpiration(t.idToken,t.refreshToken,n)}updateFromIdToken(t){Q(t.length!==0,"internal-error");const n=Th(t);this.updateTokensAndExpiration(t,null,n)}async getToken(t,n=!1){return!n&&this.accessToken&&!this.isExpired?this.accessToken:(Q(this.refreshToken,t,"user-token-expired"),this.refreshToken?(await this.refresh(t,this.refreshToken),this.accessToken):null)}clearRefreshToken(){this.refreshToken=null}async refresh(t,n){const{accessToken:r,refreshToken:s,expiresIn:i}=await ZS(t,n);this.updateTokensAndExpiration(r,s,Number(i))}updateTokensAndExpiration(t,n,r){this.refreshToken=n||null,this.accessToken=t||null,this.expirationTime=Date.now()+r*1e3}static fromJSON(t,n){const{refreshToken:r,accessToken:s,expirationTime:i}=n,a=new Pr;return r&&(Q(typeof r=="string","internal-error",{appName:t}),a.refreshToken=r),s&&(Q(typeof s=="string","internal-error",{appName:t}),a.accessToken=s),i&&(Q(typeof i=="number","internal-error",{appName:t}),a.expirationTime=i),a}toJSON(){return{refreshToken:this.refreshToken,accessToken:this.accessToken,expirationTime:this.expirationTime}}_assign(t){this.accessToken=t.accessToken,this.refreshToken=t.refreshToken,this.expirationTime=t.expirationTime}_clone(){return Object.assign(new Pr,this.toJSON())}_performRefresh(){return Ls("not implemented")}}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */function rn(e,t){Q(typeof e=="string"||typeof e>"u","internal-error",{appName:t})}class At{constructor({uid:t,auth:n,stsTokenManager:r,...s}){this.providerId="firebase",this.proactiveRefresh=new JS(this),this.reloadUserInfo=null,this.reloadListener=null,this.uid=t,this.auth=n,this.stsTokenManager=r,this.accessToken=r.accessToken,this.displayName=s.displayName||null,this.email=s.email||null,this.emailVerified=s.emailVerified||!1,this.phoneNumber=s.phoneNumber||null,this.photoURL=s.photoURL||null,this.isAnonymous=s.isAnonymous||!1,this.tenantId=s.tenantId||null,this.providerData=s.providerData?[...s.providerData]:[],this.metadata=new Lu(s.createdAt||void 0,s.lastLoginAt||void 0)}async getIdToken(t){const n=await Mu(this,this.stsTokenManager.getToken(this.auth,t));return Q(n,this.auth,"internal-error"),this.accessToken!==n&&(this.accessToken=n,await this.auth._persistUserIfCurrent(this),this.auth._notifyListenersIfCurrent(this)),n}getIdTokenResult(t){return KS(this,t)}reload(){return YS(this)}_assign(t){this!==t&&(Q(this.uid===t.uid,this.auth,"internal-error"),this.displayName=t.displayName,this.photoURL=t.photoURL,this.email=t.email,this.emailVerified=t.emailVerified,this.phoneNumber=t.phoneNumber,this.isAnonymous=t.isAnonymous,this.tenantId=t.tenantId,this.providerData=t.providerData.map(n=>({...n})),this.metadata._copy(t.metadata),this.stsTokenManager._assign(t.stsTokenManager))}_clone(t){const n=new At({...this,auth:t,stsTokenManager:this.stsTokenManager._clone()});return n.metadata._copy(this.metadata),n}_onReload(t){Q(!this.reloadListener,this.auth,"internal-error"),this.reloadListener=t,this.reloadUserInfo&&(this._notifyReloadListener(this.reloadUserInfo),this.reloadUserInfo=null)}_notifyReloadListener(t){this.reloadListener?this.reloadListener(t):this.reloadUserInfo=t}_startProactiveRefresh(){this.proactiveRefresh._start()}_stopProactiveRefresh(){this.proactiveRefresh._stop()}async _updateTokensIfNecessary(t,n=!1){let r=!1;t.idToken&&t.idToken!==this.stsTokenManager.accessToken&&(this.stsTokenManager.updateFromServerResponse(t),r=!0),n&&await Yo(this),await this.auth._persistUserIfCurrent(this),r&&this.auth._notifyListenersIfCurrent(this)}async delete(){if(Bn(this.auth.app))return Promise.reject(xo(this.auth));const t=await this.getIdToken();return await Mu(this,qS(this.auth,{idToken:t})),this.stsTokenManager.clearRefreshToken(),this.auth.signOut()}toJSON(){return{uid:this.uid,email:this.email||void 0,emailVerified:this.emailVerified,displayName:this.displayName||void 0,isAnonymous:this.isAnonymous,photoURL:this.photoURL||void 0,phoneNumber:this.phoneNumber||void 0,tenantId:this.tenantId||void 0,providerData:this.providerData.map(t=>({...t})),stsTokenManager:this.stsTokenManager.toJSON(),_redirectEventId:this._redirectEventId,...this.metadata.toJSON(),apiKey:this.auth.config.apiKey,appName:this.auth.name}}get refreshToken(){return this.stsTokenManager.refreshToken||""}static _fromJSON(t,n){const r=n.displayName??void 0,s=n.email??void 0,i=n.phoneNumber??void 0,a=n.photoURL??void 0,u=n.tenantId??void 0,c=n._redirectEventId??void 0,f=n.createdAt??void 0,E=n.lastLoginAt??void 0,{uid:w,emailVerified:_,isAnonymous:A,providerData:R,stsTokenManager:j}=n;Q(w&&j,t,"internal-error");const D=Pr.fromJSON(this.name,j);Q(typeof w=="string",t,"internal-error"),rn(r,t.name),rn(s,t.name),Q(typeof _=="boolean",t,"internal-error"),Q(typeof A=="boolean",t,"internal-error"),rn(i,t.name),rn(a,t.name),rn(u,t.name),rn(c,t.name),rn(f,t.name),rn(E,t.name);const b=new At({uid:w,auth:t,email:s,emailVerified:_,displayName:r,isAnonymous:A,photoURL:a,phoneNumber:i,tenantId:u,stsTokenManager:D,createdAt:f,lastLoginAt:E});return R&&Array.isArray(R)&&(b.providerData=R.map(x=>({...x}))),c&&(b._redirectEventId=c),b}static async _fromIdTokenResponse(t,n,r=!1){const s=new Pr;s.updateFromServerResponse(n);const i=new At({uid:n.localId,auth:t,stsTokenManager:s,isAnonymous:r});return await Yo(i),i}static async _fromGetAccountInfoResponse(t,n,r){const s=n.users[0];Q(s.localId!==void 0,"internal-error");const i=s.providerUserInfo!==void 0?ny(s.providerUserInfo):[],a=!(s.email&&s.passwordHash)&&!(i!=null&&i.length),u=new Pr;u.updateFromIdToken(r);const c=new At({uid:s.localId,auth:t,stsTokenManager:u,isAnonymous:a}),f={uid:s.localId,displayName:s.displayName||null,photoURL:s.photoUrl||null,email:s.email||null,emailVerified:s.emailVerified||!1,phoneNumber:s.phoneNumber||null,tenantId:s.tenantId||null,providerData:i,metadata:new Lu(s.createdAt,s.lastLoginAt),isAnonymous:!(s.email&&s.passwordHash)&&!(i!=null&&i.length)};return Object.assign(c,f),c}}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const Nh=new Map;function Hn(e){Xo(e instanceof Function,"Expected a class definition");let t=Nh.get(e);return t?(Xo(t instanceof e,"Instance stored in cache mismatched with class"),t):(t=new e,Nh.set(e,t),t)}/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class ry{constructor(){this.type="NONE",this.storage={}}async _isAvailable(){return!0}async _set(t,n){this.storage[t]=n}async _get(t){const n=this.storage[t];return n===void 0?null:n}async _remove(t){delete this.storage[t]}_addListener(t,n){}_removeListener(t,n){}}ry.type="NONE";const jh=ry;/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */function Tl(e,t,n){return`firebase:${e}:${t}:${n}`}class Or{constructor(t,n,r){this.persistence=t,this.auth=n,this.userKey=r;const{config:s,name:i}=this.auth;this.fullUserKey=Tl(this.userKey,s.apiKey,i),this.fullPersistenceKey=Tl("persistence",s.apiKey,i),this.boundEventHandler=n._onStorageEvent.bind(n),this.persistence._addListener(this.fullUserKey,this.boundEventHandler)}setCurrentUser(t){return this.persistence._set(this.fullUserKey,t.toJSON())}async getCurrentUser(){const t=await this.persistence._get(this.fullUserKey);if(!t)return null;if(typeof t=="string"){const n=await Jo(this.auth,{idToken:t}).catch(()=>{});return n?At._fromGetAccountInfoResponse(this.auth,n,t):null}return At._fromJSON(this.auth,t)}removeCurrentUser(){return this.persistence._remove(this.fullUserKey)}savePersistenceForRedirect(){return this.persistence._set(this.fullPersistenceKey,this.persistence.type)}async setPersistence(t){if(this.persistence===t)return;const n=await this.getCurrentUser();if(await this.removeCurrentUser(),this.persistence=t,n)return this.setCurrentUser(n)}delete(){this.persistence._removeListener(this.fullUserKey,this.boundEventHandler)}static async create(t,n,r="authUser"){if(!n.length)return new Or(Hn(jh),t,r);const s=(await Promise.all(n.map(async f=>{if(await f._isAvailable())return f}))).filter(f=>f);let i=s[0]||Hn(jh);const a=Tl(r,t.config.apiKey,t.name);let u=null;for(const f of n)try{const E=await f._get(a);if(E){let w;if(typeof E=="string"){const _=await Jo(t,{idToken:E}).catch(()=>{});if(!_)break;w=await At._fromGetAccountInfoResponse(t,_,E)}else w=At._fromJSON(t,E);f!==i&&(u=w),i=f;break}}catch{}const c=s.filter(f=>f._shouldAllowMigration);return!i._shouldAllowMigration||!c.length?new Or(i,t,r):(i=c[0],u&&await i._set(a,u.toJSON()),await Promise.all(n.map(async f=>{if(f!==i)try{await f._remove(a)}catch{}})),new Or(i,t,r))}}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */function Ah(e){const t=e.toLowerCase();if(t.includes("opera/")||t.includes("opr/")||t.includes("opios/"))return"Opera";if(sE(t))return"IEMobile";if(t.includes("msie")||t.includes("trident/"))return"IE";if(t.includes("edge/"))return"Edge";if(tE(t))return"Firefox";if(t.includes("silk/"))return"Silk";if(oE(t))return"Blackberry";if(aE(t))return"Webos";if(nE(t))return"Safari";if((t.includes("chrome/")||rE(t))&&!t.includes("edge/"))return"Chrome";if(iE(t))return"Android";{const n=/([a-zA-Z\d\.]+)\/[a-zA-Z\d\.]*$/,r=e.match(n);if((r==null?void 0:r.length)===2)return r[1]}return"Other"}function tE(e=Dt()){return/firefox\//i.test(e)}function nE(e=Dt()){const t=e.toLowerCase();return t.includes("safari/")&&!t.includes("chrome/")&&!t.includes("crios/")&&!t.includes("android")}function rE(e=Dt()){return/crios\//i.test(e)}function sE(e=Dt()){return/iemobile/i.test(e)}function iE(e=Dt()){return/android/i.test(e)}function oE(e=Dt()){return/blackberry/i.test(e)}function aE(e=Dt()){return/webos/i.test(e)}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */function sy(e,t=[]){let n;switch(e){case"Browser":n=Ah(Dt());break;case"Worker":n=`${Ah(Dt())}-${e}`;break;default:n=e}const r=t.length?t.join(","):"FirebaseCore-web";return`${n}/JsCore/${wa}/${r}`}/**
 * @license
 * Copyright 2022 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class lE{constructor(t){this.auth=t,this.queue=[]}pushCallback(t,n){const r=i=>new Promise((a,u)=>{try{const c=t(i);a(c)}catch(c){u(c)}});r.onAbort=n,this.queue.push(r);const s=this.queue.length-1;return()=>{this.queue[s]=()=>Promise.resolve()}}async runMiddleware(t){if(this.auth.currentUser===t)return;const n=[];try{for(const r of this.queue)await r(t),r.onAbort&&n.push(r.onAbort)}catch(r){n.reverse();for(const s of n)try{s()}catch{}throw this.auth._errorFactory.create("login-blocked",{originalMessage:r==null?void 0:r.message})}}}/**
 * @license
 * Copyright 2023 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */async function uE(e,t={}){return xa(e,"GET","/v2/passwordPolicy",Qg(e,t))}/**
 * @license
 * Copyright 2023 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const cE=6;class dE{constructor(t){var r;const n=t.customStrengthOptions;this.customStrengthOptions={},this.customStrengthOptions.minPasswordLength=n.minPasswordLength??cE,n.maxPasswordLength&&(this.customStrengthOptions.maxPasswordLength=n.maxPasswordLength),n.containsLowercaseCharacter!==void 0&&(this.customStrengthOptions.containsLowercaseLetter=n.containsLowercaseCharacter),n.containsUppercaseCharacter!==void 0&&(this.customStrengthOptions.containsUppercaseLetter=n.containsUppercaseCharacter),n.containsNumericCharacter!==void 0&&(this.customStrengthOptions.containsNumericCharacter=n.containsNumericCharacter),n.containsNonAlphanumericCharacter!==void 0&&(this.customStrengthOptions.containsNonAlphanumericCharacter=n.containsNonAlphanumericCharacter),this.enforcementState=t.enforcementState,this.enforcementState==="ENFORCEMENT_STATE_UNSPECIFIED"&&(this.enforcementState="OFF"),this.allowedNonAlphanumericCharacters=((r=t.allowedNonAlphanumericCharacters)==null?void 0:r.join(""))??"",this.forceUpgradeOnSignin=t.forceUpgradeOnSignin??!1,this.schemaVersion=t.schemaVersion}validatePassword(t){const n={isValid:!0,passwordPolicy:this};return this.validatePasswordLengthOptions(t,n),this.validatePasswordCharacterOptions(t,n),n.isValid&&(n.isValid=n.meetsMinPasswordLength??!0),n.isValid&&(n.isValid=n.meetsMaxPasswordLength??!0),n.isValid&&(n.isValid=n.containsLowercaseLetter??!0),n.isValid&&(n.isValid=n.containsUppercaseLetter??!0),n.isValid&&(n.isValid=n.containsNumericCharacter??!0),n.isValid&&(n.isValid=n.containsNonAlphanumericCharacter??!0),n}validatePasswordLengthOptions(t,n){const r=this.customStrengthOptions.minPasswordLength,s=this.customStrengthOptions.maxPasswordLength;r&&(n.meetsMinPasswordLength=t.length>=r),s&&(n.meetsMaxPasswordLength=t.length<=s)}validatePasswordCharacterOptions(t,n){this.updatePasswordCharacterOptionsStatuses(n,!1,!1,!1,!1);let r;for(let s=0;s<t.length;s++)r=t.charAt(s),this.updatePasswordCharacterOptionsStatuses(n,r>="a"&&r<="z",r>="A"&&r<="Z",r>="0"&&r<="9",this.allowedNonAlphanumericCharacters.includes(r))}updatePasswordCharacterOptionsStatuses(t,n,r,s,i){this.customStrengthOptions.containsLowercaseLetter&&(t.containsLowercaseLetter||(t.containsLowercaseLetter=n)),this.customStrengthOptions.containsUppercaseLetter&&(t.containsUppercaseLetter||(t.containsUppercaseLetter=r)),this.customStrengthOptions.containsNumericCharacter&&(t.containsNumericCharacter||(t.containsNumericCharacter=s)),this.customStrengthOptions.containsNonAlphanumericCharacter&&(t.containsNonAlphanumericCharacter||(t.containsNonAlphanumericCharacter=i))}}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class fE{constructor(t,n,r,s){this.app=t,this.heartbeatServiceProvider=n,this.appCheckServiceProvider=r,this.config=s,this.currentUser=null,this.emulatorConfig=null,this.operations=Promise.resolve(),this.authStateSubscription=new Rh(this),this.idTokenSubscription=new Rh(this),this.beforeStateQueue=new lE(this),this.redirectUser=null,this.isProactiveRefreshEnabled=!1,this.EXPECTED_PASSWORD_POLICY_SCHEMA_VERSION=1,this._canInitEmulator=!0,this._isInitialized=!1,this._deleted=!1,this._initializationPromise=null,this._popupRedirectResolver=null,this._errorFactory=Kg,this._agentRecaptchaConfig=null,this._tenantRecaptchaConfigs={},this._projectPasswordPolicy=null,this._tenantPasswordPolicies={},this._resolvePersistenceManagerAvailable=void 0,this.lastNotifiedUid=void 0,this.languageCode=null,this.tenantId=null,this.settings={appVerificationDisabledForTesting:!1},this.frameworks=[],this.name=t.name,this.clientVersion=s.sdkClientVersion,this._persistenceManagerAvailable=new Promise(i=>this._resolvePersistenceManagerAvailable=i)}_initializeWithPersistence(t,n){return n&&(this._popupRedirectResolver=Hn(n)),this._initializationPromise=this.queue(async()=>{var r,s,i;if(!this._deleted&&(this.persistenceManager=await Or.create(this,t),(r=this._resolvePersistenceManagerAvailable)==null||r.call(this),!this._deleted)){if((s=this._popupRedirectResolver)!=null&&s._shouldInitProactively)try{await this._popupRedirectResolver._initialize(this)}catch{}await this.initializeCurrentUser(n),this.lastNotifiedUid=((i=this.currentUser)==null?void 0:i.uid)||null,!this._deleted&&(this._isInitialized=!0)}}),this._initializationPromise}async _onStorageEvent(){if(this._deleted)return;const t=await this.assertedPersistence.getCurrentUser();if(!(!this.currentUser&&!t)){if(this.currentUser&&t&&this.currentUser.uid===t.uid){this._currentUser._assign(t),await this.currentUser.getIdToken();return}await this._updateCurrentUser(t,!0)}}async initializeCurrentUserFromIdToken(t){try{const n=await Jo(this,{idToken:t}),r=await At._fromGetAccountInfoResponse(this,n,t);await this.directlySetCurrentUser(r)}catch(n){console.warn("FirebaseServerApp could not login user with provided authIdToken: ",n),await this.directlySetCurrentUser(null)}}async initializeCurrentUser(t){var i;if(Bn(this.app)){const a=this.app.settings.authIdToken;return a?new Promise(u=>{setTimeout(()=>this.initializeCurrentUserFromIdToken(a).then(u,u))}):this.directlySetCurrentUser(null)}const n=await this.assertedPersistence.getCurrentUser();let r=n,s=!1;if(t&&this.config.authDomain){await this.getOrInitRedirectPersistenceManager();const a=(i=this.redirectUser)==null?void 0:i._redirectEventId,u=r==null?void 0:r._redirectEventId,c=await this.tryRedirectSignIn(t);(!a||a===u)&&(c!=null&&c.user)&&(r=c.user,s=!0)}if(!r)return this.directlySetCurrentUser(null);if(!r._redirectEventId){if(s)try{await this.beforeStateQueue.runMiddleware(r)}catch(a){r=n,this._popupRedirectResolver._overrideRedirectResult(this,()=>Promise.reject(a))}return r?this.reloadAndSetCurrentUserOrClear(r):this.directlySetCurrentUser(null)}return Q(this._popupRedirectResolver,this,"argument-error"),await this.getOrInitRedirectPersistenceManager(),this.redirectUser&&this.redirectUser._redirectEventId===r._redirectEventId?this.directlySetCurrentUser(r):this.reloadAndSetCurrentUserOrClear(r)}async tryRedirectSignIn(t){let n=null;try{n=await this._popupRedirectResolver._completeRedirectFn(this,t,!0)}catch{await this._setRedirectUser(null)}return n}async reloadAndSetCurrentUserOrClear(t){try{await Yo(t)}catch(n){if((n==null?void 0:n.code)!=="auth/network-request-failed")return this.directlySetCurrentUser(null)}return this.directlySetCurrentUser(t)}useDeviceLanguage(){this.languageCode=zS()}async _delete(){this._deleted=!0}async updateCurrentUser(t){if(Bn(this.app))return Promise.reject(xo(this));const n=t?Ei(t):null;return n&&Q(n.auth.config.apiKey===this.config.apiKey,this,"invalid-user-token"),this._updateCurrentUser(n&&n._clone(this))}async _updateCurrentUser(t,n=!1){if(!this._deleted)return t&&Q(this.tenantId===t.tenantId,this,"tenant-id-mismatch"),n||await this.beforeStateQueue.runMiddleware(t),this.queue(async()=>{await this.directlySetCurrentUser(t),this.notifyAuthListeners()})}async signOut(){return Bn(this.app)?Promise.reject(xo(this)):(await this.beforeStateQueue.runMiddleware(null),(this.redirectPersistenceManager||this._popupRedirectResolver)&&await this._setRedirectUser(null),this._updateCurrentUser(null,!0))}setPersistence(t){return Bn(this.app)?Promise.reject(xo(this)):this.queue(async()=>{await this.assertedPersistence.setPersistence(Hn(t))})}_getRecaptchaConfig(){return this.tenantId==null?this._agentRecaptchaConfig:this._tenantRecaptchaConfigs[this.tenantId]}async validatePassword(t){this._getPasswordPolicyInternal()||await this._updatePasswordPolicy();const n=this._getPasswordPolicyInternal();return n.schemaVersion!==this.EXPECTED_PASSWORD_POLICY_SCHEMA_VERSION?Promise.reject(this._errorFactory.create("unsupported-password-policy-schema-version",{})):n.validatePassword(t)}_getPasswordPolicyInternal(){return this.tenantId===null?this._projectPasswordPolicy:this._tenantPasswordPolicies[this.tenantId]}async _updatePasswordPolicy(){const t=await uE(this),n=new dE(t);this.tenantId===null?this._projectPasswordPolicy=n:this._tenantPasswordPolicies[this.tenantId]=n}_getPersistenceType(){return this.assertedPersistence.persistence.type}_getPersistence(){return this.assertedPersistence.persistence}_updateErrorMap(t){this._errorFactory=new Si("auth","Firebase",t())}onAuthStateChanged(t,n,r){return this.registerStateListener(this.authStateSubscription,t,n,r)}beforeAuthStateChanged(t,n){return this.beforeStateQueue.pushCallback(t,n)}onIdTokenChanged(t,n,r){return this.registerStateListener(this.idTokenSubscription,t,n,r)}authStateReady(){return new Promise((t,n)=>{if(this.currentUser)t();else{const r=this.onAuthStateChanged(()=>{r(),t()},n)}})}async revokeAccessToken(t){if(this.currentUser){const n=await this.currentUser.getIdToken(),r={providerId:"apple.com",tokenType:"ACCESS_TOKEN",token:t,idToken:n};this.tenantId!=null&&(r.tenantId=this.tenantId),await eE(this,r)}}toJSON(){var t;return{apiKey:this.config.apiKey,authDomain:this.config.authDomain,appName:this.name,currentUser:(t=this._currentUser)==null?void 0:t.toJSON()}}async _setRedirectUser(t,n){const r=await this.getOrInitRedirectPersistenceManager(n);return t===null?r.removeCurrentUser():r.setCurrentUser(t)}async getOrInitRedirectPersistenceManager(t){if(!this.redirectPersistenceManager){const n=t&&Hn(t)||this._popupRedirectResolver;Q(n,this,"argument-error"),this.redirectPersistenceManager=await Or.create(this,[Hn(n._redirectPersistence)],"redirectUser"),this.redirectUser=await this.redirectPersistenceManager.getCurrentUser()}return this.redirectPersistenceManager}async _redirectUserForId(t){var n,r;return this._isInitialized&&await this.queue(async()=>{}),((n=this._currentUser)==null?void 0:n._redirectEventId)===t?this._currentUser:((r=this.redirectUser)==null?void 0:r._redirectEventId)===t?this.redirectUser:null}async _persistUserIfCurrent(t){if(t===this.currentUser)return this.queue(async()=>this.directlySetCurrentUser(t))}_notifyListenersIfCurrent(t){t===this.currentUser&&this.notifyAuthListeners()}_key(){return`${this.config.authDomain}:${this.config.apiKey}:${this.name}`}_startProactiveRefresh(){this.isProactiveRefreshEnabled=!0,this.currentUser&&this._currentUser._startProactiveRefresh()}_stopProactiveRefresh(){this.isProactiveRefreshEnabled=!1,this.currentUser&&this._currentUser._stopProactiveRefresh()}get _currentUser(){return this.currentUser}notifyAuthListeners(){var n;if(!this._isInitialized)return;this.idTokenSubscription.next(this.currentUser);const t=((n=this.currentUser)==null?void 0:n.uid)??null;this.lastNotifiedUid!==t&&(this.lastNotifiedUid=t,this.authStateSubscription.next(this.currentUser))}registerStateListener(t,n,r,s){if(this._deleted)return()=>{};const i=typeof n=="function"?n:n.next.bind(n);let a=!1;const u=this._isInitialized?Promise.resolve():this._initializationPromise;if(Q(u,this,"internal-error"),u.then(()=>{a||i(this.currentUser)}),typeof n=="function"){const c=t.addObserver(n,r,s);return()=>{a=!0,c()}}else{const c=t.addObserver(n);return()=>{a=!0,c()}}}async directlySetCurrentUser(t){this.currentUser&&this.currentUser!==t&&this._currentUser._stopProactiveRefresh(),t&&this.isProactiveRefreshEnabled&&t._startProactiveRefresh(),this.currentUser=t,t?await this.assertedPersistence.setCurrentUser(t):await this.assertedPersistence.removeCurrentUser()}queue(t){return this.operations=this.operations.then(t,t),this.operations}get assertedPersistence(){return Q(this.persistenceManager,this,"internal-error"),this.persistenceManager}_logFramework(t){!t||this.frameworks.includes(t)||(this.frameworks.push(t),this.frameworks.sort(),this.clientVersion=sy(this.config.clientPlatform,this._getFrameworks()))}_getFrameworks(){return this.frameworks}async _getAdditionalHeaders(){var s;const t={"X-Client-Version":this.clientVersion};this.app.options.appId&&(t["X-Firebase-gmpid"]=this.app.options.appId);const n=await((s=this.heartbeatServiceProvider.getImmediate({optional:!0}))==null?void 0:s.getHeartbeatsHeader());n&&(t["X-Firebase-Client"]=n);const r=await this._getAppCheckToken();return r&&(t["X-Firebase-AppCheck"]=r),t}async _getAppCheckToken(){var n;if(Bn(this.app)&&this.app.settings.appCheckToken)return this.app.settings.appCheckToken;const t=await((n=this.appCheckServiceProvider.getImmediate({optional:!0}))==null?void 0:n.getToken());return t!=null&&t.error&&US(`Error while retrieving App Check token: ${t.error}`),t==null?void 0:t.token}}function hE(e){return Ei(e)}class Rh{constructor(t){this.auth=t,this.observer=null,this.addObserver=Tx(n=>this.observer=n)}get next(){return Q(this.observer,this.auth,"internal-error"),this.observer.next.bind(this.observer)}}function pE(e,t){const n=(t==null?void 0:t.persistence)||[],r=(Array.isArray(n)?n:[n]).map(Hn);t!=null&&t.errorMap&&e._updateErrorMap(t.errorMap),e._initializeWithPersistence(r,t==null?void 0:t.popupRedirectResolver)}new ki(3e4,6e4);/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */new ki(2e3,1e4);/**
 * @license
 * Copyright 2020 Google LLC.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */new ki(3e4,6e4);/**
 * @license
 * Copyright 2020 Google LLC.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */new ki(5e3,15e3);var Ph="@firebase/auth",Oh="1.12.1";/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class mE{constructor(t){this.auth=t,this.internalListeners=new Map}getUid(){var t;return this.assertAuthConfigured(),((t=this.auth.currentUser)==null?void 0:t.uid)||null}async getToken(t){return this.assertAuthConfigured(),await this.auth._initializationPromise,this.auth.currentUser?{accessToken:await this.auth.currentUser.getIdToken(t)}:null}addAuthTokenListener(t){if(this.assertAuthConfigured(),this.internalListeners.has(t))return;const n=this.auth.onIdTokenChanged(r=>{t((r==null?void 0:r.stsTokenManager.accessToken)||null)});this.internalListeners.set(t,n),this.updateProactiveRefresh()}removeAuthTokenListener(t){this.assertAuthConfigured();const n=this.internalListeners.get(t);n&&(this.internalListeners.delete(t),n(),this.updateProactiveRefresh())}assertAuthConfigured(){Q(this.auth._initializationPromise,"dependent-sdk-initialized-before-auth")}updateProactiveRefresh(){this.internalListeners.size>0?this.auth._startProactiveRefresh():this.auth._stopProactiveRefresh()}}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */function gE(e){switch(e){case"Node":return"node";case"ReactNative":return"rn";case"Worker":return"webworker";case"Cordova":return"cordova";case"WebExtension":return"web-extension";default:return}}function yE(e){Wr(new Gr("auth",(t,{options:n})=>{const r=t.getProvider("app").getImmediate(),s=t.getProvider("heartbeat"),i=t.getProvider("app-check-internal"),{apiKey:a,authDomain:u}=r.options;Q(a&&!a.includes(":"),"invalid-api-key",{appName:r.name});const c={apiKey:a,authDomain:u,clientPlatform:e,apiHost:"identitytoolkit.googleapis.com",tokenApiHost:"securetoken.googleapis.com",apiScheme:"https",sdkClientVersion:sy(e)},f=new fE(r,s,i,c);return pE(f,n),f},"PUBLIC").setInstantiationMode("EXPLICIT").setInstanceCreatedCallback((t,n,r)=>{t.getProvider("auth-internal").initialize()})),Wr(new Gr("auth-internal",t=>{const n=hE(t.getProvider("auth").getImmediate());return(r=>new mE(r))(n)},"PRIVATE").setInstantiationMode("EXPLICIT")),kn(Ph,Oh,gE(e)),kn(Ph,Oh,"esm2020")}/**
 * @license
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const vE=5*60;vx("authIdTokenMaxAge");yE("Browser");var Dh=typeof globalThis<"u"?globalThis:typeof window<"u"?window:typeof global<"u"?global:typeof self<"u"?self:{};/** @license
Copyright The Closure Library Authors.
SPDX-License-Identifier: Apache-2.0
*/var qc;(function(){var e;/** @license

 Copyright The Closure Library Authors.
 SPDX-License-Identifier: Apache-2.0
*/function t(g,p){function v(){}v.prototype=p.prototype,g.F=p.prototype,g.prototype=new v,g.prototype.constructor=g,g.D=function(S,k,I){for(var y=Array(arguments.length-2),Y=2;Y<arguments.length;Y++)y[Y-2]=arguments[Y];return p.prototype[k].apply(S,y)}}function n(){this.blockSize=-1}function r(){this.blockSize=-1,this.blockSize=64,this.g=Array(4),this.C=Array(this.blockSize),this.o=this.h=0,this.u()}t(r,n),r.prototype.u=function(){this.g[0]=1732584193,this.g[1]=4023233417,this.g[2]=2562383102,this.g[3]=271733878,this.o=this.h=0};function s(g,p,v){v||(v=0);const S=Array(16);if(typeof p=="string")for(var k=0;k<16;++k)S[k]=p.charCodeAt(v++)|p.charCodeAt(v++)<<8|p.charCodeAt(v++)<<16|p.charCodeAt(v++)<<24;else for(k=0;k<16;++k)S[k]=p[v++]|p[v++]<<8|p[v++]<<16|p[v++]<<24;p=g.g[0],v=g.g[1],k=g.g[2];let I=g.g[3],y;y=p+(I^v&(k^I))+S[0]+3614090360&4294967295,p=v+(y<<7&4294967295|y>>>25),y=I+(k^p&(v^k))+S[1]+3905402710&4294967295,I=p+(y<<12&4294967295|y>>>20),y=k+(v^I&(p^v))+S[2]+606105819&4294967295,k=I+(y<<17&4294967295|y>>>15),y=v+(p^k&(I^p))+S[3]+3250441966&4294967295,v=k+(y<<22&4294967295|y>>>10),y=p+(I^v&(k^I))+S[4]+4118548399&4294967295,p=v+(y<<7&4294967295|y>>>25),y=I+(k^p&(v^k))+S[5]+1200080426&4294967295,I=p+(y<<12&4294967295|y>>>20),y=k+(v^I&(p^v))+S[6]+2821735955&4294967295,k=I+(y<<17&4294967295|y>>>15),y=v+(p^k&(I^p))+S[7]+4249261313&4294967295,v=k+(y<<22&4294967295|y>>>10),y=p+(I^v&(k^I))+S[8]+1770035416&4294967295,p=v+(y<<7&4294967295|y>>>25),y=I+(k^p&(v^k))+S[9]+2336552879&4294967295,I=p+(y<<12&4294967295|y>>>20),y=k+(v^I&(p^v))+S[10]+4294925233&4294967295,k=I+(y<<17&4294967295|y>>>15),y=v+(p^k&(I^p))+S[11]+2304563134&4294967295,v=k+(y<<22&4294967295|y>>>10),y=p+(I^v&(k^I))+S[12]+1804603682&4294967295,p=v+(y<<7&4294967295|y>>>25),y=I+(k^p&(v^k))+S[13]+4254626195&4294967295,I=p+(y<<12&4294967295|y>>>20),y=k+(v^I&(p^v))+S[14]+2792965006&4294967295,k=I+(y<<17&4294967295|y>>>15),y=v+(p^k&(I^p))+S[15]+1236535329&4294967295,v=k+(y<<22&4294967295|y>>>10),y=p+(k^I&(v^k))+S[1]+4129170786&4294967295,p=v+(y<<5&4294967295|y>>>27),y=I+(v^k&(p^v))+S[6]+3225465664&4294967295,I=p+(y<<9&4294967295|y>>>23),y=k+(p^v&(I^p))+S[11]+643717713&4294967295,k=I+(y<<14&4294967295|y>>>18),y=v+(I^p&(k^I))+S[0]+3921069994&4294967295,v=k+(y<<20&4294967295|y>>>12),y=p+(k^I&(v^k))+S[5]+3593408605&4294967295,p=v+(y<<5&4294967295|y>>>27),y=I+(v^k&(p^v))+S[10]+38016083&4294967295,I=p+(y<<9&4294967295|y>>>23),y=k+(p^v&(I^p))+S[15]+3634488961&4294967295,k=I+(y<<14&4294967295|y>>>18),y=v+(I^p&(k^I))+S[4]+3889429448&4294967295,v=k+(y<<20&4294967295|y>>>12),y=p+(k^I&(v^k))+S[9]+568446438&4294967295,p=v+(y<<5&4294967295|y>>>27),y=I+(v^k&(p^v))+S[14]+3275163606&4294967295,I=p+(y<<9&4294967295|y>>>23),y=k+(p^v&(I^p))+S[3]+4107603335&4294967295,k=I+(y<<14&4294967295|y>>>18),y=v+(I^p&(k^I))+S[8]+1163531501&4294967295,v=k+(y<<20&4294967295|y>>>12),y=p+(k^I&(v^k))+S[13]+2850285829&4294967295,p=v+(y<<5&4294967295|y>>>27),y=I+(v^k&(p^v))+S[2]+4243563512&4294967295,I=p+(y<<9&4294967295|y>>>23),y=k+(p^v&(I^p))+S[7]+1735328473&4294967295,k=I+(y<<14&4294967295|y>>>18),y=v+(I^p&(k^I))+S[12]+2368359562&4294967295,v=k+(y<<20&4294967295|y>>>12),y=p+(v^k^I)+S[5]+4294588738&4294967295,p=v+(y<<4&4294967295|y>>>28),y=I+(p^v^k)+S[8]+2272392833&4294967295,I=p+(y<<11&4294967295|y>>>21),y=k+(I^p^v)+S[11]+1839030562&4294967295,k=I+(y<<16&4294967295|y>>>16),y=v+(k^I^p)+S[14]+4259657740&4294967295,v=k+(y<<23&4294967295|y>>>9),y=p+(v^k^I)+S[1]+2763975236&4294967295,p=v+(y<<4&4294967295|y>>>28),y=I+(p^v^k)+S[4]+1272893353&4294967295,I=p+(y<<11&4294967295|y>>>21),y=k+(I^p^v)+S[7]+4139469664&4294967295,k=I+(y<<16&4294967295|y>>>16),y=v+(k^I^p)+S[10]+3200236656&4294967295,v=k+(y<<23&4294967295|y>>>9),y=p+(v^k^I)+S[13]+681279174&4294967295,p=v+(y<<4&4294967295|y>>>28),y=I+(p^v^k)+S[0]+3936430074&4294967295,I=p+(y<<11&4294967295|y>>>21),y=k+(I^p^v)+S[3]+3572445317&4294967295,k=I+(y<<16&4294967295|y>>>16),y=v+(k^I^p)+S[6]+76029189&4294967295,v=k+(y<<23&4294967295|y>>>9),y=p+(v^k^I)+S[9]+3654602809&4294967295,p=v+(y<<4&4294967295|y>>>28),y=I+(p^v^k)+S[12]+3873151461&4294967295,I=p+(y<<11&4294967295|y>>>21),y=k+(I^p^v)+S[15]+530742520&4294967295,k=I+(y<<16&4294967295|y>>>16),y=v+(k^I^p)+S[2]+3299628645&4294967295,v=k+(y<<23&4294967295|y>>>9),y=p+(k^(v|~I))+S[0]+4096336452&4294967295,p=v+(y<<6&4294967295|y>>>26),y=I+(v^(p|~k))+S[7]+1126891415&4294967295,I=p+(y<<10&4294967295|y>>>22),y=k+(p^(I|~v))+S[14]+2878612391&4294967295,k=I+(y<<15&4294967295|y>>>17),y=v+(I^(k|~p))+S[5]+4237533241&4294967295,v=k+(y<<21&4294967295|y>>>11),y=p+(k^(v|~I))+S[12]+1700485571&4294967295,p=v+(y<<6&4294967295|y>>>26),y=I+(v^(p|~k))+S[3]+2399980690&4294967295,I=p+(y<<10&4294967295|y>>>22),y=k+(p^(I|~v))+S[10]+4293915773&4294967295,k=I+(y<<15&4294967295|y>>>17),y=v+(I^(k|~p))+S[1]+2240044497&4294967295,v=k+(y<<21&4294967295|y>>>11),y=p+(k^(v|~I))+S[8]+1873313359&4294967295,p=v+(y<<6&4294967295|y>>>26),y=I+(v^(p|~k))+S[15]+4264355552&4294967295,I=p+(y<<10&4294967295|y>>>22),y=k+(p^(I|~v))+S[6]+2734768916&4294967295,k=I+(y<<15&4294967295|y>>>17),y=v+(I^(k|~p))+S[13]+1309151649&4294967295,v=k+(y<<21&4294967295|y>>>11),y=p+(k^(v|~I))+S[4]+4149444226&4294967295,p=v+(y<<6&4294967295|y>>>26),y=I+(v^(p|~k))+S[11]+3174756917&4294967295,I=p+(y<<10&4294967295|y>>>22),y=k+(p^(I|~v))+S[2]+718787259&4294967295,k=I+(y<<15&4294967295|y>>>17),y=v+(I^(k|~p))+S[9]+3951481745&4294967295,g.g[0]=g.g[0]+p&4294967295,g.g[1]=g.g[1]+(k+(y<<21&4294967295|y>>>11))&4294967295,g.g[2]=g.g[2]+k&4294967295,g.g[3]=g.g[3]+I&4294967295}r.prototype.v=function(g,p){p===void 0&&(p=g.length);const v=p-this.blockSize,S=this.C;let k=this.h,I=0;for(;I<p;){if(k==0)for(;I<=v;)s(this,g,I),I+=this.blockSize;if(typeof g=="string"){for(;I<p;)if(S[k++]=g.charCodeAt(I++),k==this.blockSize){s(this,S),k=0;break}}else for(;I<p;)if(S[k++]=g[I++],k==this.blockSize){s(this,S),k=0;break}}this.h=k,this.o+=p},r.prototype.A=function(){var g=Array((this.h<56?this.blockSize:this.blockSize*2)-this.h);g[0]=128;for(var p=1;p<g.length-8;++p)g[p]=0;p=this.o*8;for(var v=g.length-8;v<g.length;++v)g[v]=p&255,p/=256;for(this.v(g),g=Array(16),p=0,v=0;v<4;++v)for(let S=0;S<32;S+=8)g[p++]=this.g[v]>>>S&255;return g};function i(g,p){var v=u;return Object.prototype.hasOwnProperty.call(v,g)?v[g]:v[g]=p(g)}function a(g,p){this.h=p;const v=[];let S=!0;for(let k=g.length-1;k>=0;k--){const I=g[k]|0;S&&I==p||(v[k]=I,S=!1)}this.g=v}var u={};function c(g){return-128<=g&&g<128?i(g,function(p){return new a([p|0],p<0?-1:0)}):new a([g|0],g<0?-1:0)}function f(g){if(isNaN(g)||!isFinite(g))return w;if(g<0)return D(f(-g));const p=[];let v=1;for(let S=0;g>=v;S++)p[S]=g/v|0,v*=4294967296;return new a(p,0)}function E(g,p){if(g.length==0)throw Error("number format error: empty string");if(p=p||10,p<2||36<p)throw Error("radix out of range: "+p);if(g.charAt(0)=="-")return D(E(g.substring(1),p));if(g.indexOf("-")>=0)throw Error('number format error: interior "-" character');const v=f(Math.pow(p,8));let S=w;for(let I=0;I<g.length;I+=8){var k=Math.min(8,g.length-I);const y=parseInt(g.substring(I,I+k),p);k<8?(k=f(Math.pow(p,k)),S=S.j(k).add(f(y))):(S=S.j(v),S=S.add(f(y)))}return S}var w=c(0),_=c(1),A=c(16777216);e=a.prototype,e.m=function(){if(j(this))return-D(this).m();let g=0,p=1;for(let v=0;v<this.g.length;v++){const S=this.i(v);g+=(S>=0?S:4294967296+S)*p,p*=4294967296}return g},e.toString=function(g){if(g=g||10,g<2||36<g)throw Error("radix out of range: "+g);if(R(this))return"0";if(j(this))return"-"+D(this).toString(g);const p=f(Math.pow(g,6));var v=this;let S="";for(;;){const k=O(v,p).g;v=b(v,k.j(p));let I=((v.g.length>0?v.g[0]:v.h)>>>0).toString(g);if(v=k,R(v))return I+S;for(;I.length<6;)I="0"+I;S=I+S}},e.i=function(g){return g<0?0:g<this.g.length?this.g[g]:this.h};function R(g){if(g.h!=0)return!1;for(let p=0;p<g.g.length;p++)if(g.g[p]!=0)return!1;return!0}function j(g){return g.h==-1}e.l=function(g){return g=b(this,g),j(g)?-1:R(g)?0:1};function D(g){const p=g.g.length,v=[];for(let S=0;S<p;S++)v[S]=~g.g[S];return new a(v,~g.h).add(_)}e.abs=function(){return j(this)?D(this):this},e.add=function(g){const p=Math.max(this.g.length,g.g.length),v=[];let S=0;for(let k=0;k<=p;k++){let I=S+(this.i(k)&65535)+(g.i(k)&65535),y=(I>>>16)+(this.i(k)>>>16)+(g.i(k)>>>16);S=y>>>16,I&=65535,y&=65535,v[k]=y<<16|I}return new a(v,v[v.length-1]&-2147483648?-1:0)};function b(g,p){return g.add(D(p))}e.j=function(g){if(R(this)||R(g))return w;if(j(this))return j(g)?D(this).j(D(g)):D(D(this).j(g));if(j(g))return D(this.j(D(g)));if(this.l(A)<0&&g.l(A)<0)return f(this.m()*g.m());const p=this.g.length+g.g.length,v=[];for(var S=0;S<2*p;S++)v[S]=0;for(S=0;S<this.g.length;S++)for(let k=0;k<g.g.length;k++){const I=this.i(S)>>>16,y=this.i(S)&65535,Y=g.i(k)>>>16,$=g.i(k)&65535;v[2*S+2*k]+=y*$,x(v,2*S+2*k),v[2*S+2*k+1]+=I*$,x(v,2*S+2*k+1),v[2*S+2*k+1]+=y*Y,x(v,2*S+2*k+1),v[2*S+2*k+2]+=I*Y,x(v,2*S+2*k+2)}for(g=0;g<p;g++)v[g]=v[2*g+1]<<16|v[2*g];for(g=p;g<2*p;g++)v[g]=0;return new a(v,0)};function x(g,p){for(;(g[p]&65535)!=g[p];)g[p+1]+=g[p]>>>16,g[p]&=65535,p++}function C(g,p){this.g=g,this.h=p}function O(g,p){if(R(p))throw Error("division by zero");if(R(g))return new C(w,w);if(j(g))return p=O(D(g),p),new C(D(p.g),D(p.h));if(j(p))return p=O(g,D(p)),new C(D(p.g),p.h);if(g.g.length>30){if(j(g)||j(p))throw Error("slowDivide_ only works with positive integers.");for(var v=_,S=p;S.l(g)<=0;)v=M(v),S=M(S);var k=U(v,1),I=U(S,1);for(S=U(S,2),v=U(v,2);!R(S);){var y=I.add(S);y.l(g)<=0&&(k=k.add(v),I=y),S=U(S,1),v=U(v,1)}return p=b(g,k.j(p)),new C(k,p)}for(k=w;g.l(p)>=0;){for(v=Math.max(1,Math.floor(g.m()/p.m())),S=Math.ceil(Math.log(v)/Math.LN2),S=S<=48?1:Math.pow(2,S-48),I=f(v),y=I.j(p);j(y)||y.l(g)>0;)v-=S,I=f(v),y=I.j(p);R(I)&&(I=_),k=k.add(I),g=b(g,y)}return new C(k,g)}e.B=function(g){return O(this,g).h},e.and=function(g){const p=Math.max(this.g.length,g.g.length),v=[];for(let S=0;S<p;S++)v[S]=this.i(S)&g.i(S);return new a(v,this.h&g.h)},e.or=function(g){const p=Math.max(this.g.length,g.g.length),v=[];for(let S=0;S<p;S++)v[S]=this.i(S)|g.i(S);return new a(v,this.h|g.h)},e.xor=function(g){const p=Math.max(this.g.length,g.g.length),v=[];for(let S=0;S<p;S++)v[S]=this.i(S)^g.i(S);return new a(v,this.h^g.h)};function M(g){const p=g.g.length+1,v=[];for(let S=0;S<p;S++)v[S]=g.i(S)<<1|g.i(S-1)>>>31;return new a(v,g.h)}function U(g,p){const v=p>>5;p%=32;const S=g.g.length-v,k=[];for(let I=0;I<S;I++)k[I]=p>0?g.i(I+v)>>>p|g.i(I+v+1)<<32-p:g.i(I+v);return new a(k,g.h)}r.prototype.digest=r.prototype.A,r.prototype.reset=r.prototype.u,r.prototype.update=r.prototype.v,a.prototype.add=a.prototype.add,a.prototype.multiply=a.prototype.j,a.prototype.modulo=a.prototype.B,a.prototype.compare=a.prototype.l,a.prototype.toNumber=a.prototype.m,a.prototype.toString=a.prototype.toString,a.prototype.getBits=a.prototype.i,a.fromNumber=f,a.fromString=E,qc=a}).apply(typeof Dh<"u"?Dh:typeof self<"u"?self:typeof window<"u"?window:{});var eo=typeof globalThis<"u"?globalThis:typeof window<"u"?window:typeof global<"u"?global:typeof self<"u"?self:{};(function(){var e,t=Object.defineProperty;function n(o){o=[typeof globalThis=="object"&&globalThis,o,typeof window=="object"&&window,typeof self=="object"&&self,typeof eo=="object"&&eo];for(var l=0;l<o.length;++l){var d=o[l];if(d&&d.Math==Math)return d}throw Error("Cannot find global object")}var r=n(this);function s(o,l){if(l)e:{var d=r;o=o.split(".");for(var h=0;h<o.length-1;h++){var T=o[h];if(!(T in d))break e;d=d[T]}o=o[o.length-1],h=d[o],l=l(h),l!=h&&l!=null&&t(d,o,{configurable:!0,writable:!0,value:l})}}s("Symbol.dispose",function(o){return o||Symbol("Symbol.dispose")}),s("Array.prototype.values",function(o){return o||function(){return this[Symbol.iterator]()}}),s("Object.entries",function(o){return o||function(l){var d=[],h;for(h in l)Object.prototype.hasOwnProperty.call(l,h)&&d.push([h,l[h]]);return d}});/** @license

 Copyright The Closure Library Authors.
 SPDX-License-Identifier: Apache-2.0
*/var i=i||{},a=this||self;function u(o){var l=typeof o;return l=="object"&&o!=null||l=="function"}function c(o,l,d){return o.call.apply(o.bind,arguments)}function f(o,l,d){return f=c,f.apply(null,arguments)}function E(o,l){var d=Array.prototype.slice.call(arguments,1);return function(){var h=d.slice();return h.push.apply(h,arguments),o.apply(this,h)}}function w(o,l){function d(){}d.prototype=l.prototype,o.Z=l.prototype,o.prototype=new d,o.prototype.constructor=o,o.Ob=function(h,T,P){for(var B=Array(arguments.length-2),q=2;q<arguments.length;q++)B[q-2]=arguments[q];return l.prototype[T].apply(h,B)}}var _=typeof AsyncContext<"u"&&typeof AsyncContext.Snapshot=="function"?o=>o&&AsyncContext.Snapshot.wrap(o):o=>o;function A(o){const l=o.length;if(l>0){const d=Array(l);for(let h=0;h<l;h++)d[h]=o[h];return d}return[]}function R(o,l){for(let h=1;h<arguments.length;h++){const T=arguments[h];var d=typeof T;if(d=d!="object"?d:T?Array.isArray(T)?"array":d:"null",d=="array"||d=="object"&&typeof T.length=="number"){d=o.length||0;const P=T.length||0;o.length=d+P;for(let B=0;B<P;B++)o[d+B]=T[B]}else o.push(T)}}class j{constructor(l,d){this.i=l,this.j=d,this.h=0,this.g=null}get(){let l;return this.h>0?(this.h--,l=this.g,this.g=l.next,l.next=null):l=this.i(),l}}function D(o){a.setTimeout(()=>{throw o},0)}function b(){var o=g;let l=null;return o.g&&(l=o.g,o.g=o.g.next,o.g||(o.h=null),l.next=null),l}class x{constructor(){this.h=this.g=null}add(l,d){const h=C.get();h.set(l,d),this.h?this.h.next=h:this.g=h,this.h=h}}var C=new j(()=>new O,o=>o.reset());class O{constructor(){this.next=this.g=this.h=null}set(l,d){this.h=l,this.g=d,this.next=null}reset(){this.next=this.g=this.h=null}}let M,U=!1,g=new x,p=()=>{const o=Promise.resolve(void 0);M=()=>{o.then(v)}};function v(){for(var o;o=b();){try{o.h.call(o.g)}catch(d){D(d)}var l=C;l.j(o),l.h<100&&(l.h++,o.next=l.g,l.g=o)}U=!1}function S(){this.u=this.u,this.C=this.C}S.prototype.u=!1,S.prototype.dispose=function(){this.u||(this.u=!0,this.N())},S.prototype[Symbol.dispose]=function(){this.dispose()},S.prototype.N=function(){if(this.C)for(;this.C.length;)this.C.shift()()};function k(o,l){this.type=o,this.g=this.target=l,this.defaultPrevented=!1}k.prototype.h=function(){this.defaultPrevented=!0};var I=function(){if(!a.addEventListener||!Object.defineProperty)return!1;var o=!1,l=Object.defineProperty({},"passive",{get:function(){o=!0}});try{const d=()=>{};a.addEventListener("test",d,l),a.removeEventListener("test",d,l)}catch{}return o}();function y(o){return/^[\s\xa0]*$/.test(o)}function Y(o,l){k.call(this,o?o.type:""),this.relatedTarget=this.g=this.target=null,this.button=this.screenY=this.screenX=this.clientY=this.clientX=0,this.key="",this.metaKey=this.shiftKey=this.altKey=this.ctrlKey=!1,this.state=null,this.pointerId=0,this.pointerType="",this.i=null,o&&this.init(o,l)}w(Y,k),Y.prototype.init=function(o,l){const d=this.type=o.type,h=o.changedTouches&&o.changedTouches.length?o.changedTouches[0]:null;this.target=o.target||o.srcElement,this.g=l,l=o.relatedTarget,l||(d=="mouseover"?l=o.fromElement:d=="mouseout"&&(l=o.toElement)),this.relatedTarget=l,h?(this.clientX=h.clientX!==void 0?h.clientX:h.pageX,this.clientY=h.clientY!==void 0?h.clientY:h.pageY,this.screenX=h.screenX||0,this.screenY=h.screenY||0):(this.clientX=o.clientX!==void 0?o.clientX:o.pageX,this.clientY=o.clientY!==void 0?o.clientY:o.pageY,this.screenX=o.screenX||0,this.screenY=o.screenY||0),this.button=o.button,this.key=o.key||"",this.ctrlKey=o.ctrlKey,this.altKey=o.altKey,this.shiftKey=o.shiftKey,this.metaKey=o.metaKey,this.pointerId=o.pointerId||0,this.pointerType=o.pointerType,this.state=o.state,this.i=o,o.defaultPrevented&&Y.Z.h.call(this)},Y.prototype.h=function(){Y.Z.h.call(this);const o=this.i;o.preventDefault?o.preventDefault():o.returnValue=!1};var $="closure_listenable_"+(Math.random()*1e6|0),te=0;function He(o,l,d,h,T){this.listener=o,this.proxy=null,this.src=l,this.type=d,this.capture=!!h,this.ha=T,this.key=++te,this.da=this.fa=!1}function F(o){o.da=!0,o.listener=null,o.proxy=null,o.src=null,o.ha=null}function H(o,l,d){for(const h in o)l.call(d,o[h],h,o)}function W(o,l){for(const d in o)l.call(void 0,o[d],d,o)}function V(o){const l={};for(const d in o)l[d]=o[d];return l}const G="constructor hasOwnProperty isPrototypeOf propertyIsEnumerable toLocaleString toString valueOf".split(" ");function ge(o,l){let d,h;for(let T=1;T<arguments.length;T++){h=arguments[T];for(d in h)o[d]=h[d];for(let P=0;P<G.length;P++)d=G[P],Object.prototype.hasOwnProperty.call(h,d)&&(o[d]=h[d])}}function _e(o){this.src=o,this.g={},this.h=0}_e.prototype.add=function(o,l,d,h,T){const P=o.toString();o=this.g[P],o||(o=this.g[P]=[],this.h++);const B=ft(o,l,h,T);return B>-1?(l=o[B],d||(l.fa=!1)):(l=new He(l,this.src,P,!!h,T),l.fa=d,o.push(l)),l};function Ge(o,l){const d=l.type;if(d in o.g){var h=o.g[d],T=Array.prototype.indexOf.call(h,l,void 0),P;(P=T>=0)&&Array.prototype.splice.call(h,T,1),P&&(F(l),o.g[d].length==0&&(delete o.g[d],o.h--))}}function ft(o,l,d,h){for(let T=0;T<o.length;++T){const P=o[T];if(!P.da&&P.listener==l&&P.capture==!!d&&P.ha==h)return T}return-1}var Xt="closure_lm_"+(Math.random()*1e6|0),Ca={};function td(o,l,d,h,T){if(Array.isArray(l)){for(let P=0;P<l.length;P++)td(o,l[P],d,h,T);return null}return d=sd(d),o&&o[$]?o.J(l,d,u(h)?!!h.capture:!1,T):e0(o,l,d,!1,h,T)}function e0(o,l,d,h,T,P){if(!l)throw Error("Invalid event type");const B=u(T)?!!T.capture:!!T;let q=Ta(o);if(q||(o[Xt]=q=new _e(o)),d=q.add(l,d,h,B,P),d.proxy)return d;if(h=t0(),d.proxy=h,h.src=o,h.listener=d,o.addEventListener)I||(T=B),T===void 0&&(T=!1),o.addEventListener(l.toString(),h,T);else if(o.attachEvent)o.attachEvent(rd(l.toString()),h);else if(o.addListener&&o.removeListener)o.addListener(h);else throw Error("addEventListener and attachEvent are unavailable.");return d}function t0(){function o(d){return l.call(o.src,o.listener,d)}const l=n0;return o}function nd(o,l,d,h,T){if(Array.isArray(l))for(var P=0;P<l.length;P++)nd(o,l[P],d,h,T);else h=u(h)?!!h.capture:!!h,d=sd(d),o&&o[$]?(o=o.i,P=String(l).toString(),P in o.g&&(l=o.g[P],d=ft(l,d,h,T),d>-1&&(F(l[d]),Array.prototype.splice.call(l,d,1),l.length==0&&(delete o.g[P],o.h--)))):o&&(o=Ta(o))&&(l=o.g[l.toString()],o=-1,l&&(o=ft(l,d,h,T)),(d=o>-1?l[o]:null)&&Ia(d))}function Ia(o){if(typeof o!="number"&&o&&!o.da){var l=o.src;if(l&&l[$])Ge(l.i,o);else{var d=o.type,h=o.proxy;l.removeEventListener?l.removeEventListener(d,h,o.capture):l.detachEvent?l.detachEvent(rd(d),h):l.addListener&&l.removeListener&&l.removeListener(h),(d=Ta(l))?(Ge(d,o),d.h==0&&(d.src=null,l[Xt]=null)):F(o)}}}function rd(o){return o in Ca?Ca[o]:Ca[o]="on"+o}function n0(o,l){if(o.da)o=!0;else{l=new Y(l,this);const d=o.listener,h=o.ha||o.src;o.fa&&Ia(o),o=d.call(h,l)}return o}function Ta(o){return o=o[Xt],o instanceof _e?o:null}var Na="__closure_events_fn_"+(Math.random()*1e9>>>0);function sd(o){return typeof o=="function"?o:(o[Na]||(o[Na]=function(l){return o.handleEvent(l)}),o[Na])}function Re(){S.call(this),this.i=new _e(this),this.M=this,this.G=null}w(Re,S),Re.prototype[$]=!0,Re.prototype.removeEventListener=function(o,l,d,h){nd(this,o,l,d,h)};function Ue(o,l){var d,h=o.G;if(h)for(d=[];h;h=h.G)d.push(h);if(o=o.M,h=l.type||l,typeof l=="string")l=new k(l,o);else if(l instanceof k)l.target=l.target||o;else{var T=l;l=new k(h,o),ge(l,T)}T=!0;let P,B;if(d)for(B=d.length-1;B>=0;B--)P=l.g=d[B],T=Ii(P,h,!0,l)&&T;if(P=l.g=o,T=Ii(P,h,!0,l)&&T,T=Ii(P,h,!1,l)&&T,d)for(B=0;B<d.length;B++)P=l.g=d[B],T=Ii(P,h,!1,l)&&T}Re.prototype.N=function(){if(Re.Z.N.call(this),this.i){var o=this.i;for(const l in o.g){const d=o.g[l];for(let h=0;h<d.length;h++)F(d[h]);delete o.g[l],o.h--}}this.G=null},Re.prototype.J=function(o,l,d,h){return this.i.add(String(o),l,!1,d,h)},Re.prototype.K=function(o,l,d,h){return this.i.add(String(o),l,!0,d,h)};function Ii(o,l,d,h){if(l=o.i.g[String(l)],!l)return!0;l=l.concat();let T=!0;for(let P=0;P<l.length;++P){const B=l[P];if(B&&!B.da&&B.capture==d){const q=B.listener,Ee=B.ha||B.src;B.fa&&Ge(o.i,B),T=q.call(Ee,h)!==!1&&T}}return T&&!h.defaultPrevented}function r0(o,l){if(typeof o!="function")if(o&&typeof o.handleEvent=="function")o=f(o.handleEvent,o);else throw Error("Invalid listener argument");return Number(l)>2147483647?-1:a.setTimeout(o,l||0)}function id(o){o.g=r0(()=>{o.g=null,o.i&&(o.i=!1,id(o))},o.l);const l=o.h;o.h=null,o.m.apply(null,l)}class s0 extends S{constructor(l,d){super(),this.m=l,this.l=d,this.h=null,this.i=!1,this.g=null}j(l){this.h=arguments,this.g?this.i=!0:id(this)}N(){super.N(),this.g&&(a.clearTimeout(this.g),this.g=null,this.i=!1,this.h=null)}}function Zr(o){S.call(this),this.h=o,this.g={}}w(Zr,S);var od=[];function ad(o){H(o.g,function(l,d){this.g.hasOwnProperty(d)&&Ia(l)},o),o.g={}}Zr.prototype.N=function(){Zr.Z.N.call(this),ad(this)},Zr.prototype.handleEvent=function(){throw Error("EventHandler.handleEvent not implemented")};var ja=a.JSON.stringify,i0=a.JSON.parse,o0=class{stringify(o){return a.JSON.stringify(o,void 0)}parse(o){return a.JSON.parse(o,void 0)}};function ld(){}function a0(){}var es={OPEN:"a",hb:"b",ERROR:"c",tb:"d"};function Aa(){k.call(this,"d")}w(Aa,k);function Ra(){k.call(this,"c")}w(Ra,k);var or={},ud=null;function Pa(){return ud=ud||new Re}or.Ia="serverreachability";function cd(o){k.call(this,or.Ia,o)}w(cd,k);function ts(o){const l=Pa();Ue(l,new cd(l))}or.STAT_EVENT="statevent";function dd(o,l){k.call(this,or.STAT_EVENT,o),this.stat=l}w(dd,k);function Be(o){const l=Pa();Ue(l,new dd(l,o))}or.Ja="timingevent";function fd(o,l){k.call(this,or.Ja,o),this.size=l}w(fd,k);function ns(o,l){if(typeof o!="function")throw Error("Fn must not be null and must be a function");return a.setTimeout(function(){o()},l)}function rs(){this.g=!0}rs.prototype.ua=function(){this.g=!1};function l0(o,l,d,h,T,P){o.info(function(){if(o.g)if(P){var B="",q=P.split("&");for(let ie=0;ie<q.length;ie++){var Ee=q[ie].split("=");if(Ee.length>1){const Ce=Ee[0];Ee=Ee[1];const _t=Ce.split("_");B=_t.length>=2&&_t[1]=="type"?B+(Ce+"="+Ee+"&"):B+(Ce+"=redacted&")}}}else B=null;else B=P;return"XMLHTTP REQ ("+h+") [attempt "+T+"]: "+l+`
`+d+`
`+B})}function u0(o,l,d,h,T,P,B){o.info(function(){return"XMLHTTP RESP ("+h+") [ attempt "+T+"]: "+l+`
`+d+`
`+P+" "+B})}function ar(o,l,d,h){o.info(function(){return"XMLHTTP TEXT ("+l+"): "+d0(o,d)+(h?" "+h:"")})}function c0(o,l){o.info(function(){return"TIMEOUT: "+l})}rs.prototype.info=function(){};function d0(o,l){if(!o.g)return l;if(!l)return null;try{const P=JSON.parse(l);if(P){for(o=0;o<P.length;o++)if(Array.isArray(P[o])){var d=P[o];if(!(d.length<2)){var h=d[1];if(Array.isArray(h)&&!(h.length<1)){var T=h[0];if(T!="noop"&&T!="stop"&&T!="close")for(let B=1;B<h.length;B++)h[B]=""}}}}return ja(P)}catch{return l}}var Oa={NO_ERROR:0,TIMEOUT:8},f0={},hd;function Da(){}w(Da,ld),Da.prototype.g=function(){return new XMLHttpRequest},hd=new Da;function ss(o){return encodeURIComponent(String(o))}function h0(o){var l=1;o=o.split(":");const d=[];for(;l>0&&o.length;)d.push(o.shift()),l--;return o.length&&d.push(o.join(":")),d}function Jt(o,l,d,h){this.j=o,this.i=l,this.l=d,this.S=h||1,this.V=new Zr(this),this.H=45e3,this.J=null,this.o=!1,this.u=this.B=this.A=this.M=this.F=this.T=this.D=null,this.G=[],this.g=null,this.C=0,this.m=this.v=null,this.X=-1,this.K=!1,this.P=0,this.O=null,this.W=this.L=this.U=this.R=!1,this.h=new pd}function pd(){this.i=null,this.g="",this.h=!1}var md={},Ma={};function La(o,l,d){o.M=1,o.A=Ni(bt(l)),o.u=d,o.R=!0,gd(o,null)}function gd(o,l){o.F=Date.now(),Ti(o),o.B=bt(o.A);var d=o.B,h=o.S;Array.isArray(h)||(h=[String(h)]),Nd(d.i,"t",h),o.C=0,d=o.j.L,o.h=new pd,o.g=qd(o.j,d?l:null,!o.u),o.P>0&&(o.O=new s0(f(o.Y,o,o.g),o.P)),l=o.V,d=o.g,h=o.ba;var T="readystatechange";Array.isArray(T)||(T&&(od[0]=T.toString()),T=od);for(let P=0;P<T.length;P++){const B=td(d,T[P],h||l.handleEvent,!1,l.h||l);if(!B)break;l.g[B.key]=B}l=o.J?V(o.J):{},o.u?(o.v||(o.v="POST"),l["Content-Type"]="application/x-www-form-urlencoded",o.g.ea(o.B,o.v,o.u,l)):(o.v="GET",o.g.ea(o.B,o.v,null,l)),ts(),l0(o.i,o.v,o.B,o.l,o.S,o.u)}Jt.prototype.ba=function(o){o=o.target;const l=this.O;l&&Zt(o)==3?l.j():this.Y(o)},Jt.prototype.Y=function(o){try{if(o==this.g)e:{const q=Zt(this.g),Ee=this.g.ya(),ie=this.g.ca();if(!(q<3)&&(q!=3||this.g&&(this.h.h||this.g.la()||Md(this.g)))){this.K||q!=4||Ee==7||(Ee==8||ie<=0?ts(3):ts(2)),Ua(this);var l=this.g.ca();this.X=l;var d=p0(this);if(this.o=l==200,u0(this.i,this.v,this.B,this.l,this.S,q,l),this.o){if(this.U&&!this.L){t:{if(this.g){var h,T=this.g;if((h=T.g?T.g.getResponseHeader("X-HTTP-Initial-Response"):null)&&!y(h)){var P=h;break t}}P=null}if(o=P)ar(this.i,this.l,o,"Initial handshake response via X-HTTP-Initial-Response"),this.L=!0,Ba(this,o);else{this.o=!1,this.m=3,Be(12),Pn(this),is(this);break e}}if(this.R){o=!0;let Ce;for(;!this.K&&this.C<d.length;)if(Ce=m0(this,d),Ce==Ma){q==4&&(this.m=4,Be(14),o=!1),ar(this.i,this.l,null,"[Incomplete Response]");break}else if(Ce==md){this.m=4,Be(15),ar(this.i,this.l,d,"[Invalid Chunk]"),o=!1;break}else ar(this.i,this.l,Ce,null),Ba(this,Ce);if(yd(this)&&this.C!=0&&(this.h.g=this.h.g.slice(this.C),this.C=0),q!=4||d.length!=0||this.h.h||(this.m=1,Be(16),o=!1),this.o=this.o&&o,!o)ar(this.i,this.l,d,"[Invalid Chunked Response]"),Pn(this),is(this);else if(d.length>0&&!this.W){this.W=!0;var B=this.j;B.g==this&&B.aa&&!B.P&&(B.j.info("Great, no buffering proxy detected. Bytes received: "+d.length),qa(B),B.P=!0,Be(11))}}else ar(this.i,this.l,d,null),Ba(this,d);q==4&&Pn(this),this.o&&!this.K&&(q==4?$d(this.j,this):(this.o=!1,Ti(this)))}else N0(this.g),l==400&&d.indexOf("Unknown SID")>0?(this.m=3,Be(12)):(this.m=0,Be(13)),Pn(this),is(this)}}}catch{}finally{}};function p0(o){if(!yd(o))return o.g.la();const l=Md(o.g);if(l==="")return"";let d="";const h=l.length,T=Zt(o.g)==4;if(!o.h.i){if(typeof TextDecoder>"u")return Pn(o),is(o),"";o.h.i=new a.TextDecoder}for(let P=0;P<h;P++)o.h.h=!0,d+=o.h.i.decode(l[P],{stream:!(T&&P==h-1)});return l.length=0,o.h.g+=d,o.C=0,o.h.g}function yd(o){return o.g?o.v=="GET"&&o.M!=2&&o.j.Aa:!1}function m0(o,l){var d=o.C,h=l.indexOf(`
`,d);return h==-1?Ma:(d=Number(l.substring(d,h)),isNaN(d)?md:(h+=1,h+d>l.length?Ma:(l=l.slice(h,h+d),o.C=h+d,l)))}Jt.prototype.cancel=function(){this.K=!0,Pn(this)};function Ti(o){o.T=Date.now()+o.H,vd(o,o.H)}function vd(o,l){if(o.D!=null)throw Error("WatchDog timer not null");o.D=ns(f(o.aa,o),l)}function Ua(o){o.D&&(a.clearTimeout(o.D),o.D=null)}Jt.prototype.aa=function(){this.D=null;const o=Date.now();o-this.T>=0?(c0(this.i,this.B),this.M!=2&&(ts(),Be(17)),Pn(this),this.m=2,is(this)):vd(this,this.T-o)};function is(o){o.j.I==0||o.K||$d(o.j,o)}function Pn(o){Ua(o);var l=o.O;l&&typeof l.dispose=="function"&&l.dispose(),o.O=null,ad(o.V),o.g&&(l=o.g,o.g=null,l.abort(),l.dispose())}function Ba(o,l){try{var d=o.j;if(d.I!=0&&(d.g==o||Fa(d.h,o))){if(!o.L&&Fa(d.h,o)&&d.I==3){try{var h=d.Ba.g.parse(l)}catch{h=null}if(Array.isArray(h)&&h.length==3){var T=h;if(T[0]==0){e:if(!d.v){if(d.g)if(d.g.F+3e3<o.F)Oi(d),Ri(d);else break e;Wa(d),Be(18)}}else d.xa=T[1],0<d.xa-d.K&&T[2]<37500&&d.F&&d.A==0&&!d.C&&(d.C=ns(f(d.Va,d),6e3));Sd(d.h)<=1&&d.ta&&(d.ta=void 0)}else Dn(d,11)}else if((o.L||d.g==o)&&Oi(d),!y(l))for(T=d.Ba.g.parse(l),l=0;l<T.length;l++){let ie=T[l];const Ce=ie[0];if(!(Ce<=d.K))if(d.K=Ce,ie=ie[1],d.I==2)if(ie[0]=="c"){d.M=ie[1],d.ba=ie[2];const _t=ie[3];_t!=null&&(d.ka=_t,d.j.info("VER="+d.ka));const Mn=ie[4];Mn!=null&&(d.za=Mn,d.j.info("SVER="+d.za));const en=ie[5];en!=null&&typeof en=="number"&&en>0&&(h=1.5*en,d.O=h,d.j.info("backChannelRequestTimeoutMs_="+h)),h=d;const tn=o.g;if(tn){const Di=tn.g?tn.g.getResponseHeader("X-Client-Wire-Protocol"):null;if(Di){var P=h.h;P.g||Di.indexOf("spdy")==-1&&Di.indexOf("quic")==-1&&Di.indexOf("h2")==-1||(P.j=P.l,P.g=new Set,P.h&&(za(P,P.h),P.h=null))}if(h.G){const Ka=tn.g?tn.g.getResponseHeader("X-HTTP-Session-Id"):null;Ka&&(h.wa=Ka,le(h.J,h.G,Ka))}}d.I=3,d.l&&d.l.ra(),d.aa&&(d.T=Date.now()-o.F,d.j.info("Handshake RTT: "+d.T+"ms")),h=d;var B=o;if(h.na=Wd(h,h.L?h.ba:null,h.W),B.L){Ed(h.h,B);var q=B,Ee=h.O;Ee&&(q.H=Ee),q.D&&(Ua(q),Ti(q)),h.g=B}else zd(h);d.i.length>0&&Pi(d)}else ie[0]!="stop"&&ie[0]!="close"||Dn(d,7);else d.I==3&&(ie[0]=="stop"||ie[0]=="close"?ie[0]=="stop"?Dn(d,7):Ga(d):ie[0]!="noop"&&d.l&&d.l.qa(ie),d.A=0)}}ts(4)}catch{}}var g0=class{constructor(o,l){this.g=o,this.map=l}};function wd(o){this.l=o||10,a.PerformanceNavigationTiming?(o=a.performance.getEntriesByType("navigation"),o=o.length>0&&(o[0].nextHopProtocol=="hq"||o[0].nextHopProtocol=="h2")):o=!!(a.chrome&&a.chrome.loadTimes&&a.chrome.loadTimes()&&a.chrome.loadTimes().wasFetchedViaSpdy),this.j=o?this.l:1,this.g=null,this.j>1&&(this.g=new Set),this.h=null,this.i=[]}function xd(o){return o.h?!0:o.g?o.g.size>=o.j:!1}function Sd(o){return o.h?1:o.g?o.g.size:0}function Fa(o,l){return o.h?o.h==l:o.g?o.g.has(l):!1}function za(o,l){o.g?o.g.add(l):o.h=l}function Ed(o,l){o.h&&o.h==l?o.h=null:o.g&&o.g.has(l)&&o.g.delete(l)}wd.prototype.cancel=function(){if(this.i=kd(this),this.h)this.h.cancel(),this.h=null;else if(this.g&&this.g.size!==0){for(const o of this.g.values())o.cancel();this.g.clear()}};function kd(o){if(o.h!=null)return o.i.concat(o.h.G);if(o.g!=null&&o.g.size!==0){let l=o.i;for(const d of o.g.values())l=l.concat(d.G);return l}return A(o.i)}var bd=RegExp("^(?:([^:/?#.]+):)?(?://(?:([^\\\\/?#]*)@)?([^\\\\/?#]*?)(?::([0-9]+))?(?=[\\\\/?#]|$))?([^?#]+)?(?:\\?([^#]*))?(?:#([\\s\\S]*))?$");function y0(o,l){if(o){o=o.split("&");for(let d=0;d<o.length;d++){const h=o[d].indexOf("=");let T,P=null;h>=0?(T=o[d].substring(0,h),P=o[d].substring(h+1)):T=o[d],l(T,P?decodeURIComponent(P.replace(/\+/g," ")):"")}}}function Yt(o){this.g=this.o=this.j="",this.u=null,this.m=this.h="",this.l=!1;let l;o instanceof Yt?(this.l=o.l,os(this,o.j),this.o=o.o,this.g=o.g,as(this,o.u),this.h=o.h,Va(this,jd(o.i)),this.m=o.m):o&&(l=String(o).match(bd))?(this.l=!1,os(this,l[1]||"",!0),this.o=ls(l[2]||""),this.g=ls(l[3]||"",!0),as(this,l[4]),this.h=ls(l[5]||"",!0),Va(this,l[6]||"",!0),this.m=ls(l[7]||"")):(this.l=!1,this.i=new cs(null,this.l))}Yt.prototype.toString=function(){const o=[];var l=this.j;l&&o.push(us(l,_d,!0),":");var d=this.g;return(d||l=="file")&&(o.push("//"),(l=this.o)&&o.push(us(l,_d,!0),"@"),o.push(ss(d).replace(/%25([0-9a-fA-F]{2})/g,"%$1")),d=this.u,d!=null&&o.push(":",String(d))),(d=this.h)&&(this.g&&d.charAt(0)!="/"&&o.push("/"),o.push(us(d,d.charAt(0)=="/"?x0:w0,!0))),(d=this.i.toString())&&o.push("?",d),(d=this.m)&&o.push("#",us(d,E0)),o.join("")},Yt.prototype.resolve=function(o){const l=bt(this);let d=!!o.j;d?os(l,o.j):d=!!o.o,d?l.o=o.o:d=!!o.g,d?l.g=o.g:d=o.u!=null;var h=o.h;if(d)as(l,o.u);else if(d=!!o.h){if(h.charAt(0)!="/")if(this.g&&!this.h)h="/"+h;else{var T=l.h.lastIndexOf("/");T!=-1&&(h=l.h.slice(0,T+1)+h)}if(T=h,T==".."||T==".")h="";else if(T.indexOf("./")!=-1||T.indexOf("/.")!=-1){h=T.lastIndexOf("/",0)==0,T=T.split("/");const P=[];for(let B=0;B<T.length;){const q=T[B++];q=="."?h&&B==T.length&&P.push(""):q==".."?((P.length>1||P.length==1&&P[0]!="")&&P.pop(),h&&B==T.length&&P.push("")):(P.push(q),h=!0)}h=P.join("/")}else h=T}return d?l.h=h:d=o.i.toString()!=="",d?Va(l,jd(o.i)):d=!!o.m,d&&(l.m=o.m),l};function bt(o){return new Yt(o)}function os(o,l,d){o.j=d?ls(l,!0):l,o.j&&(o.j=o.j.replace(/:$/,""))}function as(o,l){if(l){if(l=Number(l),isNaN(l)||l<0)throw Error("Bad port number "+l);o.u=l}else o.u=null}function Va(o,l,d){l instanceof cs?(o.i=l,k0(o.i,o.l)):(d||(l=us(l,S0)),o.i=new cs(l,o.l))}function le(o,l,d){o.i.set(l,d)}function Ni(o){return le(o,"zx",Math.floor(Math.random()*2147483648).toString(36)+Math.abs(Math.floor(Math.random()*2147483648)^Date.now()).toString(36)),o}function ls(o,l){return o?l?decodeURI(o.replace(/%25/g,"%2525")):decodeURIComponent(o):""}function us(o,l,d){return typeof o=="string"?(o=encodeURI(o).replace(l,v0),d&&(o=o.replace(/%25([0-9a-fA-F]{2})/g,"%$1")),o):null}function v0(o){return o=o.charCodeAt(0),"%"+(o>>4&15).toString(16)+(o&15).toString(16)}var _d=/[#\/\?@]/g,w0=/[#\?:]/g,x0=/[#\?]/g,S0=/[#\?@]/g,E0=/#/g;function cs(o,l){this.h=this.g=null,this.i=o||null,this.j=!!l}function On(o){o.g||(o.g=new Map,o.h=0,o.i&&y0(o.i,function(l,d){o.add(decodeURIComponent(l.replace(/\+/g," ")),d)}))}e=cs.prototype,e.add=function(o,l){On(this),this.i=null,o=lr(this,o);let d=this.g.get(o);return d||this.g.set(o,d=[]),d.push(l),this.h+=1,this};function Cd(o,l){On(o),l=lr(o,l),o.g.has(l)&&(o.i=null,o.h-=o.g.get(l).length,o.g.delete(l))}function Id(o,l){return On(o),l=lr(o,l),o.g.has(l)}e.forEach=function(o,l){On(this),this.g.forEach(function(d,h){d.forEach(function(T){o.call(l,T,h,this)},this)},this)};function Td(o,l){On(o);let d=[];if(typeof l=="string")Id(o,l)&&(d=d.concat(o.g.get(lr(o,l))));else for(o=Array.from(o.g.values()),l=0;l<o.length;l++)d=d.concat(o[l]);return d}e.set=function(o,l){return On(this),this.i=null,o=lr(this,o),Id(this,o)&&(this.h-=this.g.get(o).length),this.g.set(o,[l]),this.h+=1,this},e.get=function(o,l){return o?(o=Td(this,o),o.length>0?String(o[0]):l):l};function Nd(o,l,d){Cd(o,l),d.length>0&&(o.i=null,o.g.set(lr(o,l),A(d)),o.h+=d.length)}e.toString=function(){if(this.i)return this.i;if(!this.g)return"";const o=[],l=Array.from(this.g.keys());for(let h=0;h<l.length;h++){var d=l[h];const T=ss(d);d=Td(this,d);for(let P=0;P<d.length;P++){let B=T;d[P]!==""&&(B+="="+ss(d[P])),o.push(B)}}return this.i=o.join("&")};function jd(o){const l=new cs;return l.i=o.i,o.g&&(l.g=new Map(o.g),l.h=o.h),l}function lr(o,l){return l=String(l),o.j&&(l=l.toLowerCase()),l}function k0(o,l){l&&!o.j&&(On(o),o.i=null,o.g.forEach(function(d,h){const T=h.toLowerCase();h!=T&&(Cd(this,h),Nd(this,T,d))},o)),o.j=l}function b0(o,l){const d=new rs;if(a.Image){const h=new Image;h.onload=E(Qt,d,"TestLoadImage: loaded",!0,l,h),h.onerror=E(Qt,d,"TestLoadImage: error",!1,l,h),h.onabort=E(Qt,d,"TestLoadImage: abort",!1,l,h),h.ontimeout=E(Qt,d,"TestLoadImage: timeout",!1,l,h),a.setTimeout(function(){h.ontimeout&&h.ontimeout()},1e4),h.src=o}else l(!1)}function _0(o,l){const d=new rs,h=new AbortController,T=setTimeout(()=>{h.abort(),Qt(d,"TestPingServer: timeout",!1,l)},1e4);fetch(o,{signal:h.signal}).then(P=>{clearTimeout(T),P.ok?Qt(d,"TestPingServer: ok",!0,l):Qt(d,"TestPingServer: server error",!1,l)}).catch(()=>{clearTimeout(T),Qt(d,"TestPingServer: error",!1,l)})}function Qt(o,l,d,h,T){try{T&&(T.onload=null,T.onerror=null,T.onabort=null,T.ontimeout=null),h(d)}catch{}}function C0(){this.g=new o0}function $a(o){this.i=o.Sb||null,this.h=o.ab||!1}w($a,ld),$a.prototype.g=function(){return new ji(this.i,this.h)};function ji(o,l){Re.call(this),this.H=o,this.o=l,this.m=void 0,this.status=this.readyState=0,this.responseType=this.responseText=this.response=this.statusText="",this.onreadystatechange=null,this.A=new Headers,this.h=null,this.F="GET",this.D="",this.g=!1,this.B=this.j=this.l=null,this.v=new AbortController}w(ji,Re),e=ji.prototype,e.open=function(o,l){if(this.readyState!=0)throw this.abort(),Error("Error reopening a connection");this.F=o,this.D=l,this.readyState=1,fs(this)},e.send=function(o){if(this.readyState!=1)throw this.abort(),Error("need to call open() first. ");if(this.v.signal.aborted)throw this.abort(),Error("Request was aborted.");this.g=!0;const l={headers:this.A,method:this.F,credentials:this.m,cache:void 0,signal:this.v.signal};o&&(l.body=o),(this.H||a).fetch(new Request(this.D,l)).then(this.Pa.bind(this),this.ga.bind(this))},e.abort=function(){this.response=this.responseText="",this.A=new Headers,this.status=0,this.v.abort(),this.j&&this.j.cancel("Request was aborted.").catch(()=>{}),this.readyState>=1&&this.g&&this.readyState!=4&&(this.g=!1,ds(this)),this.readyState=0},e.Pa=function(o){if(this.g&&(this.l=o,this.h||(this.status=this.l.status,this.statusText=this.l.statusText,this.h=o.headers,this.readyState=2,fs(this)),this.g&&(this.readyState=3,fs(this),this.g)))if(this.responseType==="arraybuffer")o.arrayBuffer().then(this.Na.bind(this),this.ga.bind(this));else if(typeof a.ReadableStream<"u"&&"body"in o){if(this.j=o.body.getReader(),this.o){if(this.responseType)throw Error('responseType must be empty for "streamBinaryChunks" mode responses.');this.response=[]}else this.response=this.responseText="",this.B=new TextDecoder;Ad(this)}else o.text().then(this.Oa.bind(this),this.ga.bind(this))};function Ad(o){o.j.read().then(o.Ma.bind(o)).catch(o.ga.bind(o))}e.Ma=function(o){if(this.g){if(this.o&&o.value)this.response.push(o.value);else if(!this.o){var l=o.value?o.value:new Uint8Array(0);(l=this.B.decode(l,{stream:!o.done}))&&(this.response=this.responseText+=l)}o.done?ds(this):fs(this),this.readyState==3&&Ad(this)}},e.Oa=function(o){this.g&&(this.response=this.responseText=o,ds(this))},e.Na=function(o){this.g&&(this.response=o,ds(this))},e.ga=function(){this.g&&ds(this)};function ds(o){o.readyState=4,o.l=null,o.j=null,o.B=null,fs(o)}e.setRequestHeader=function(o,l){this.A.append(o,l)},e.getResponseHeader=function(o){return this.h&&this.h.get(o.toLowerCase())||""},e.getAllResponseHeaders=function(){if(!this.h)return"";const o=[],l=this.h.entries();for(var d=l.next();!d.done;)d=d.value,o.push(d[0]+": "+d[1]),d=l.next();return o.join(`\r
`)};function fs(o){o.onreadystatechange&&o.onreadystatechange.call(o)}Object.defineProperty(ji.prototype,"withCredentials",{get:function(){return this.m==="include"},set:function(o){this.m=o?"include":"same-origin"}});function Rd(o){let l="";return H(o,function(d,h){l+=h,l+=":",l+=d,l+=`\r
`}),l}function Ha(o,l,d){e:{for(h in d){var h=!1;break e}h=!0}h||(d=Rd(d),typeof o=="string"?d!=null&&ss(d):le(o,l,d))}function ye(o){Re.call(this),this.headers=new Map,this.L=o||null,this.h=!1,this.g=null,this.D="",this.o=0,this.l="",this.j=this.B=this.v=this.A=!1,this.m=null,this.F="",this.H=!1}w(ye,Re);var I0=/^https?$/i,T0=["POST","PUT"];e=ye.prototype,e.Fa=function(o){this.H=o},e.ea=function(o,l,d,h){if(this.g)throw Error("[goog.net.XhrIo] Object is active with another request="+this.D+"; newUri="+o);l=l?l.toUpperCase():"GET",this.D=o,this.l="",this.o=0,this.A=!1,this.h=!0,this.g=this.L?this.L.g():hd.g(),this.g.onreadystatechange=_(f(this.Ca,this));try{this.B=!0,this.g.open(l,String(o),!0),this.B=!1}catch(P){Pd(this,P);return}if(o=d||"",d=new Map(this.headers),h)if(Object.getPrototypeOf(h)===Object.prototype)for(var T in h)d.set(T,h[T]);else if(typeof h.keys=="function"&&typeof h.get=="function")for(const P of h.keys())d.set(P,h.get(P));else throw Error("Unknown input type for opt_headers: "+String(h));h=Array.from(d.keys()).find(P=>P.toLowerCase()=="content-type"),T=a.FormData&&o instanceof a.FormData,!(Array.prototype.indexOf.call(T0,l,void 0)>=0)||h||T||d.set("Content-Type","application/x-www-form-urlencoded;charset=utf-8");for(const[P,B]of d)this.g.setRequestHeader(P,B);this.F&&(this.g.responseType=this.F),"withCredentials"in this.g&&this.g.withCredentials!==this.H&&(this.g.withCredentials=this.H);try{this.m&&(clearTimeout(this.m),this.m=null),this.v=!0,this.g.send(o),this.v=!1}catch(P){Pd(this,P)}};function Pd(o,l){o.h=!1,o.g&&(o.j=!0,o.g.abort(),o.j=!1),o.l=l,o.o=5,Od(o),Ai(o)}function Od(o){o.A||(o.A=!0,Ue(o,"complete"),Ue(o,"error"))}e.abort=function(o){this.g&&this.h&&(this.h=!1,this.j=!0,this.g.abort(),this.j=!1,this.o=o||7,Ue(this,"complete"),Ue(this,"abort"),Ai(this))},e.N=function(){this.g&&(this.h&&(this.h=!1,this.j=!0,this.g.abort(),this.j=!1),Ai(this,!0)),ye.Z.N.call(this)},e.Ca=function(){this.u||(this.B||this.v||this.j?Dd(this):this.Xa())},e.Xa=function(){Dd(this)};function Dd(o){if(o.h&&typeof i<"u"){if(o.v&&Zt(o)==4)setTimeout(o.Ca.bind(o),0);else if(Ue(o,"readystatechange"),Zt(o)==4){o.h=!1;try{const P=o.ca();e:switch(P){case 200:case 201:case 202:case 204:case 206:case 304:case 1223:var l=!0;break e;default:l=!1}var d;if(!(d=l)){var h;if(h=P===0){let B=String(o.D).match(bd)[1]||null;!B&&a.self&&a.self.location&&(B=a.self.location.protocol.slice(0,-1)),h=!I0.test(B?B.toLowerCase():"")}d=h}if(d)Ue(o,"complete"),Ue(o,"success");else{o.o=6;try{var T=Zt(o)>2?o.g.statusText:""}catch{T=""}o.l=T+" ["+o.ca()+"]",Od(o)}}finally{Ai(o)}}}}function Ai(o,l){if(o.g){o.m&&(clearTimeout(o.m),o.m=null);const d=o.g;o.g=null,l||Ue(o,"ready");try{d.onreadystatechange=null}catch{}}}e.isActive=function(){return!!this.g};function Zt(o){return o.g?o.g.readyState:0}e.ca=function(){try{return Zt(this)>2?this.g.status:-1}catch{return-1}},e.la=function(){try{return this.g?this.g.responseText:""}catch{return""}},e.La=function(o){if(this.g){var l=this.g.responseText;return o&&l.indexOf(o)==0&&(l=l.substring(o.length)),i0(l)}};function Md(o){try{if(!o.g)return null;if("response"in o.g)return o.g.response;switch(o.F){case"":case"text":return o.g.responseText;case"arraybuffer":if("mozResponseArrayBuffer"in o.g)return o.g.mozResponseArrayBuffer}return null}catch{return null}}function N0(o){const l={};o=(o.g&&Zt(o)>=2&&o.g.getAllResponseHeaders()||"").split(`\r
`);for(let h=0;h<o.length;h++){if(y(o[h]))continue;var d=h0(o[h]);const T=d[0];if(d=d[1],typeof d!="string")continue;d=d.trim();const P=l[T]||[];l[T]=P,P.push(d)}W(l,function(h){return h.join(", ")})}e.ya=function(){return this.o},e.Ha=function(){return typeof this.l=="string"?this.l:String(this.l)};function hs(o,l,d){return d&&d.internalChannelParams&&d.internalChannelParams[o]||l}function Ld(o){this.za=0,this.i=[],this.j=new rs,this.ba=this.na=this.J=this.W=this.g=this.wa=this.G=this.H=this.u=this.U=this.o=null,this.Ya=this.V=0,this.Sa=hs("failFast",!1,o),this.F=this.C=this.v=this.m=this.l=null,this.X=!0,this.xa=this.K=-1,this.Y=this.A=this.D=0,this.Qa=hs("baseRetryDelayMs",5e3,o),this.Za=hs("retryDelaySeedMs",1e4,o),this.Ta=hs("forwardChannelMaxRetries",2,o),this.va=hs("forwardChannelRequestTimeoutMs",2e4,o),this.ma=o&&o.xmlHttpFactory||void 0,this.Ua=o&&o.Rb||void 0,this.Aa=o&&o.useFetchStreams||!1,this.O=void 0,this.L=o&&o.supportsCrossDomainXhr||!1,this.M="",this.h=new wd(o&&o.concurrentRequestLimit),this.Ba=new C0,this.S=o&&o.fastHandshake||!1,this.R=o&&o.encodeInitMessageHeaders||!1,this.S&&this.R&&(this.R=!1),this.Ra=o&&o.Pb||!1,o&&o.ua&&this.j.ua(),o&&o.forceLongPolling&&(this.X=!1),this.aa=!this.S&&this.X&&o&&o.detectBufferingProxy||!1,this.ia=void 0,o&&o.longPollingTimeout&&o.longPollingTimeout>0&&(this.ia=o.longPollingTimeout),this.ta=void 0,this.T=0,this.P=!1,this.ja=this.B=null}e=Ld.prototype,e.ka=8,e.I=1,e.connect=function(o,l,d,h){Be(0),this.W=o,this.H=l||{},d&&h!==void 0&&(this.H.OSID=d,this.H.OAID=h),this.F=this.X,this.J=Wd(this,null,this.W),Pi(this)};function Ga(o){if(Ud(o),o.I==3){var l=o.V++,d=bt(o.J);if(le(d,"SID",o.M),le(d,"RID",l),le(d,"TYPE","terminate"),ps(o,d),l=new Jt(o,o.j,l),l.M=2,l.A=Ni(bt(d)),d=!1,a.navigator&&a.navigator.sendBeacon)try{d=a.navigator.sendBeacon(l.A.toString(),"")}catch{}!d&&a.Image&&(new Image().src=l.A,d=!0),d||(l.g=qd(l.j,null),l.g.ea(l.A)),l.F=Date.now(),Ti(l)}Gd(o)}function Ri(o){o.g&&(qa(o),o.g.cancel(),o.g=null)}function Ud(o){Ri(o),o.v&&(a.clearTimeout(o.v),o.v=null),Oi(o),o.h.cancel(),o.m&&(typeof o.m=="number"&&a.clearTimeout(o.m),o.m=null)}function Pi(o){if(!xd(o.h)&&!o.m){o.m=!0;var l=o.Ea;M||p(),U||(M(),U=!0),g.add(l,o),o.D=0}}function j0(o,l){return Sd(o.h)>=o.h.j-(o.m?1:0)?!1:o.m?(o.i=l.G.concat(o.i),!0):o.I==1||o.I==2||o.D>=(o.Sa?0:o.Ta)?!1:(o.m=ns(f(o.Ea,o,l),Hd(o,o.D)),o.D++,!0)}e.Ea=function(o){if(this.m)if(this.m=null,this.I==1){if(!o){this.V=Math.floor(Math.random()*1e5),o=this.V++;const T=new Jt(this,this.j,o);let P=this.o;if(this.U&&(P?(P=V(P),ge(P,this.U)):P=this.U),this.u!==null||this.R||(T.J=P,P=null),this.S)e:{for(var l=0,d=0;d<this.i.length;d++){t:{var h=this.i[d];if("__data__"in h.map&&(h=h.map.__data__,typeof h=="string")){h=h.length;break t}h=void 0}if(h===void 0)break;if(l+=h,l>4096){l=d;break e}if(l===4096||d===this.i.length-1){l=d+1;break e}}l=1e3}else l=1e3;l=Fd(this,T,l),d=bt(this.J),le(d,"RID",o),le(d,"CVER",22),this.G&&le(d,"X-HTTP-Session-Id",this.G),ps(this,d),P&&(this.R?l="headers="+ss(Rd(P))+"&"+l:this.u&&Ha(d,this.u,P)),za(this.h,T),this.Ra&&le(d,"TYPE","init"),this.S?(le(d,"$req",l),le(d,"SID","null"),T.U=!0,La(T,d,null)):La(T,d,l),this.I=2}}else this.I==3&&(o?Bd(this,o):this.i.length==0||xd(this.h)||Bd(this))};function Bd(o,l){var d;l?d=l.l:d=o.V++;const h=bt(o.J);le(h,"SID",o.M),le(h,"RID",d),le(h,"AID",o.K),ps(o,h),o.u&&o.o&&Ha(h,o.u,o.o),d=new Jt(o,o.j,d,o.D+1),o.u===null&&(d.J=o.o),l&&(o.i=l.G.concat(o.i)),l=Fd(o,d,1e3),d.H=Math.round(o.va*.5)+Math.round(o.va*.5*Math.random()),za(o.h,d),La(d,h,l)}function ps(o,l){o.H&&H(o.H,function(d,h){le(l,h,d)}),o.l&&H({},function(d,h){le(l,h,d)})}function Fd(o,l,d){d=Math.min(o.i.length,d);const h=o.l?f(o.l.Ka,o.l,o):null;e:{var T=o.i;let q=-1;for(;;){const Ee=["count="+d];q==-1?d>0?(q=T[0].g,Ee.push("ofs="+q)):q=0:Ee.push("ofs="+q);let ie=!0;for(let Ce=0;Ce<d;Ce++){var P=T[Ce].g;const _t=T[Ce].map;if(P-=q,P<0)q=Math.max(0,T[Ce].g-100),ie=!1;else try{P="req"+P+"_"||"";try{var B=_t instanceof Map?_t:Object.entries(_t);for(const[Mn,en]of B){let tn=en;u(en)&&(tn=ja(en)),Ee.push(P+Mn+"="+encodeURIComponent(tn))}}catch(Mn){throw Ee.push(P+"type="+encodeURIComponent("_badmap")),Mn}}catch{h&&h(_t)}}if(ie){B=Ee.join("&");break e}}B=void 0}return o=o.i.splice(0,d),l.G=o,B}function zd(o){if(!o.g&&!o.v){o.Y=1;var l=o.Da;M||p(),U||(M(),U=!0),g.add(l,o),o.A=0}}function Wa(o){return o.g||o.v||o.A>=3?!1:(o.Y++,o.v=ns(f(o.Da,o),Hd(o,o.A)),o.A++,!0)}e.Da=function(){if(this.v=null,Vd(this),this.aa&&!(this.P||this.g==null||this.T<=0)){var o=4*this.T;this.j.info("BP detection timer enabled: "+o),this.B=ns(f(this.Wa,this),o)}},e.Wa=function(){this.B&&(this.B=null,this.j.info("BP detection timeout reached."),this.j.info("Buffering proxy detected and switch to long-polling!"),this.F=!1,this.P=!0,Be(10),Ri(this),Vd(this))};function qa(o){o.B!=null&&(a.clearTimeout(o.B),o.B=null)}function Vd(o){o.g=new Jt(o,o.j,"rpc",o.Y),o.u===null&&(o.g.J=o.o),o.g.P=0;var l=bt(o.na);le(l,"RID","rpc"),le(l,"SID",o.M),le(l,"AID",o.K),le(l,"CI",o.F?"0":"1"),!o.F&&o.ia&&le(l,"TO",o.ia),le(l,"TYPE","xmlhttp"),ps(o,l),o.u&&o.o&&Ha(l,o.u,o.o),o.O&&(o.g.H=o.O);var d=o.g;o=o.ba,d.M=1,d.A=Ni(bt(l)),d.u=null,d.R=!0,gd(d,o)}e.Va=function(){this.C!=null&&(this.C=null,Ri(this),Wa(this),Be(19))};function Oi(o){o.C!=null&&(a.clearTimeout(o.C),o.C=null)}function $d(o,l){var d=null;if(o.g==l){Oi(o),qa(o),o.g=null;var h=2}else if(Fa(o.h,l))d=l.G,Ed(o.h,l),h=1;else return;if(o.I!=0){if(l.o)if(h==1){d=l.u?l.u.length:0,l=Date.now()-l.F;var T=o.D;h=Pa(),Ue(h,new fd(h,d)),Pi(o)}else zd(o);else if(T=l.m,T==3||T==0&&l.X>0||!(h==1&&j0(o,l)||h==2&&Wa(o)))switch(d&&d.length>0&&(l=o.h,l.i=l.i.concat(d)),T){case 1:Dn(o,5);break;case 4:Dn(o,10);break;case 3:Dn(o,6);break;default:Dn(o,2)}}}function Hd(o,l){let d=o.Qa+Math.floor(Math.random()*o.Za);return o.isActive()||(d*=2),d*l}function Dn(o,l){if(o.j.info("Error code "+l),l==2){var d=f(o.bb,o),h=o.Ua;const T=!h;h=new Yt(h||"//www.google.com/images/cleardot.gif"),a.location&&a.location.protocol=="http"||os(h,"https"),Ni(h),T?b0(h.toString(),d):_0(h.toString(),d)}else Be(2);o.I=0,o.l&&o.l.pa(l),Gd(o),Ud(o)}e.bb=function(o){o?(this.j.info("Successfully pinged google.com"),Be(2)):(this.j.info("Failed to ping google.com"),Be(1))};function Gd(o){if(o.I=0,o.ja=[],o.l){const l=kd(o.h);(l.length!=0||o.i.length!=0)&&(R(o.ja,l),R(o.ja,o.i),o.h.i.length=0,A(o.i),o.i.length=0),o.l.oa()}}function Wd(o,l,d){var h=d instanceof Yt?bt(d):new Yt(d);if(h.g!="")l&&(h.g=l+"."+h.g),as(h,h.u);else{var T=a.location;h=T.protocol,l=l?l+"."+T.hostname:T.hostname,T=+T.port;const P=new Yt(null);h&&os(P,h),l&&(P.g=l),T&&as(P,T),d&&(P.h=d),h=P}return d=o.G,l=o.wa,d&&l&&le(h,d,l),le(h,"VER",o.ka),ps(o,h),h}function qd(o,l,d){if(l&&!o.L)throw Error("Can't create secondary domain capable XhrIo object.");return l=o.Aa&&!o.ma?new ye(new $a({ab:d})):new ye(o.ma),l.Fa(o.L),l}e.isActive=function(){return!!this.l&&this.l.isActive(this)};function Kd(){}e=Kd.prototype,e.ra=function(){},e.qa=function(){},e.pa=function(){},e.oa=function(){},e.isActive=function(){return!0},e.Ka=function(){};function it(o,l){Re.call(this),this.g=new Ld(l),this.l=o,this.h=l&&l.messageUrlParams||null,o=l&&l.messageHeaders||null,l&&l.clientProtocolHeaderRequired&&(o?o["X-Client-Protocol"]="webchannel":o={"X-Client-Protocol":"webchannel"}),this.g.o=o,o=l&&l.initMessageHeaders||null,l&&l.messageContentType&&(o?o["X-WebChannel-Content-Type"]=l.messageContentType:o={"X-WebChannel-Content-Type":l.messageContentType}),l&&l.sa&&(o?o["X-WebChannel-Client-Profile"]=l.sa:o={"X-WebChannel-Client-Profile":l.sa}),this.g.U=o,(o=l&&l.Qb)&&!y(o)&&(this.g.u=o),this.A=l&&l.supportsCrossDomainXhr||!1,this.v=l&&l.sendRawJson||!1,(l=l&&l.httpSessionIdParam)&&!y(l)&&(this.g.G=l,o=this.h,o!==null&&l in o&&(o=this.h,l in o&&delete o[l])),this.j=new ur(this)}w(it,Re),it.prototype.m=function(){this.g.l=this.j,this.A&&(this.g.L=!0),this.g.connect(this.l,this.h||void 0)},it.prototype.close=function(){Ga(this.g)},it.prototype.o=function(o){var l=this.g;if(typeof o=="string"){var d={};d.__data__=o,o=d}else this.v&&(d={},d.__data__=ja(o),o=d);l.i.push(new g0(l.Ya++,o)),l.I==3&&Pi(l)},it.prototype.N=function(){this.g.l=null,delete this.j,Ga(this.g),delete this.g,it.Z.N.call(this)};function Xd(o){Aa.call(this),o.__headers__&&(this.headers=o.__headers__,this.statusCode=o.__status__,delete o.__headers__,delete o.__status__);var l=o.__sm__;if(l){e:{for(const d in l){o=d;break e}o=void 0}(this.i=o)&&(o=this.i,l=l!==null&&o in l?l[o]:void 0),this.data=l}else this.data=o}w(Xd,Aa);function Jd(){Ra.call(this),this.status=1}w(Jd,Ra);function ur(o){this.g=o}w(ur,Kd),ur.prototype.ra=function(){Ue(this.g,"a")},ur.prototype.qa=function(o){Ue(this.g,new Xd(o))},ur.prototype.pa=function(o){Ue(this.g,new Jd)},ur.prototype.oa=function(){Ue(this.g,"b")},it.prototype.send=it.prototype.o,it.prototype.open=it.prototype.m,it.prototype.close=it.prototype.close,Oa.NO_ERROR=0,Oa.TIMEOUT=8,Oa.HTTP_ERROR=6,f0.COMPLETE="complete",a0.EventType=es,es.OPEN="a",es.CLOSE="b",es.ERROR="c",es.MESSAGE="d",Re.prototype.listen=Re.prototype.J,ye.prototype.listenOnce=ye.prototype.K,ye.prototype.getLastError=ye.prototype.Ha,ye.prototype.getLastErrorCode=ye.prototype.ya,ye.prototype.getStatus=ye.prototype.ca,ye.prototype.getResponseJson=ye.prototype.La,ye.prototype.getResponseText=ye.prototype.la,ye.prototype.send=ye.prototype.ea,ye.prototype.setWithCredentials=ye.prototype.Fa}).apply(typeof eo<"u"?eo:typeof self<"u"?self:typeof window<"u"?window:{});/**
 * @license
 * Copyright 2017 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class Qe{constructor(t){this.uid=t}isAuthenticated(){return this.uid!=null}toKey(){return this.isAuthenticated()?"uid:"+this.uid:"anonymous-user"}isEqual(t){return t.uid===this.uid}}Qe.UNAUTHENTICATED=new Qe(null),Qe.GOOGLE_CREDENTIALS=new Qe("google-credentials-uid"),Qe.FIRST_PARTY=new Qe("first-party-uid"),Qe.MOCK_USER=new Qe("mock-user");/**
 * @license
 * Copyright 2017 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */let Sa="12.10.0";function wE(e){Sa=e}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *//**
 * @license
 * Copyright 2017 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const Qo=new $c("@firebase/firestore");function vt(e,...t){if(Qo.logLevel<=ne.DEBUG){const n=t.map(oy);Qo.debug(`Firestore (${Sa}): ${e}`,...n)}}function iy(e,...t){if(Qo.logLevel<=ne.ERROR){const n=t.map(oy);Qo.error(`Firestore (${Sa}): ${e}`,...n)}}function oy(e){if(typeof e=="string")return e;try{return function(n){return JSON.stringify(n)}(e)}catch{return e}}/**
 * @license
 * Copyright 2017 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */function Zo(e,t,n){let r="Unexpected state";typeof t=="string"?r=t:n=t,ay(e,r,n)}function ay(e,t,n){let r=`FIRESTORE (${Sa}) INTERNAL ASSERTION FAILED: ${t} (ID: ${e.toString(16)})`;if(n!==void 0)try{r+=" CONTEXT: "+JSON.stringify(n)}catch{r+=" CONTEXT: "+n}throw iy(r),new Error(r)}function Bs(e,t,n,r){let s="Unexpected state";typeof n=="string"?s=n:r=n,e||ay(t,s,r)}/**
 * @license
 * Copyright 2017 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const Z={CANCELLED:"cancelled",INVALID_ARGUMENT:"invalid-argument",FAILED_PRECONDITION:"failed-precondition"};class ee extends Rn{constructor(t,n){super(t,n),this.code=t,this.message=n,this.toString=()=>`${this.name}: [code=${this.code}]: ${this.message}`}}/**
 * @license
 * Copyright 2017 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class Fs{constructor(){this.promise=new Promise((t,n)=>{this.resolve=t,this.reject=n})}}/**
 * @license
 * Copyright 2017 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class xE{constructor(t,n){this.user=n,this.type="OAuth",this.headers=new Map,this.headers.set("Authorization",`Bearer ${t}`)}}class SE{getToken(){return Promise.resolve(null)}invalidateToken(){}start(t,n){t.enqueueRetryable(()=>n(Qe.UNAUTHENTICATED))}shutdown(){}}class EE{constructor(t){this.t=t,this.currentUser=Qe.UNAUTHENTICATED,this.i=0,this.forceRefresh=!1,this.auth=null}start(t,n){Bs(this.o===void 0,42304);let r=this.i;const s=c=>this.i!==r?(r=this.i,n(c)):Promise.resolve();let i=new Fs;this.o=()=>{this.i++,this.currentUser=this.u(),i.resolve(),i=new Fs,t.enqueueRetryable(()=>s(this.currentUser))};const a=()=>{const c=i;t.enqueueRetryable(async()=>{await c.promise,await s(this.currentUser)})},u=c=>{vt("FirebaseAuthCredentialsProvider","Auth detected"),this.auth=c,this.o&&(this.auth.addAuthTokenListener(this.o),a())};this.t.onInit(c=>u(c)),setTimeout(()=>{if(!this.auth){const c=this.t.getImmediate({optional:!0});c?u(c):(vt("FirebaseAuthCredentialsProvider","Auth not yet detected"),i.resolve(),i=new Fs)}},0),a()}getToken(){const t=this.i,n=this.forceRefresh;return this.forceRefresh=!1,this.auth?this.auth.getToken(n).then(r=>this.i!==t?(vt("FirebaseAuthCredentialsProvider","getToken aborted due to token change."),this.getToken()):r?(Bs(typeof r.accessToken=="string",31837,{l:r}),new xE(r.accessToken,this.currentUser)):null):Promise.resolve(null)}invalidateToken(){this.forceRefresh=!0}shutdown(){this.auth&&this.o&&this.auth.removeAuthTokenListener(this.o),this.o=void 0}u(){const t=this.auth&&this.auth.getUid();return Bs(t===null||typeof t=="string",2055,{h:t}),new Qe(t)}}class kE{constructor(t,n,r){this.P=t,this.T=n,this.I=r,this.type="FirstParty",this.user=Qe.FIRST_PARTY,this.R=new Map}A(){return this.I?this.I():null}get headers(){this.R.set("X-Goog-AuthUser",this.P);const t=this.A();return t&&this.R.set("Authorization",t),this.T&&this.R.set("X-Goog-Iam-Authorization-Token",this.T),this.R}}class bE{constructor(t,n,r){this.P=t,this.T=n,this.I=r}getToken(){return Promise.resolve(new kE(this.P,this.T,this.I))}start(t,n){t.enqueueRetryable(()=>n(Qe.FIRST_PARTY))}shutdown(){}invalidateToken(){}}class Mh{constructor(t){this.value=t,this.type="AppCheck",this.headers=new Map,t&&t.length>0&&this.headers.set("x-firebase-appcheck",this.value)}}class _E{constructor(t,n){this.V=n,this.forceRefresh=!1,this.appCheck=null,this.m=null,this.p=null,Bn(t)&&t.settings.appCheckToken&&(this.p=t.settings.appCheckToken)}start(t,n){Bs(this.o===void 0,3512);const r=i=>{i.error!=null&&vt("FirebaseAppCheckTokenProvider",`Error getting App Check token; using placeholder token instead. Error: ${i.error.message}`);const a=i.token!==this.m;return this.m=i.token,vt("FirebaseAppCheckTokenProvider",`Received ${a?"new":"existing"} token.`),a?n(i.token):Promise.resolve()};this.o=i=>{t.enqueueRetryable(()=>r(i))};const s=i=>{vt("FirebaseAppCheckTokenProvider","AppCheck detected"),this.appCheck=i,this.o&&this.appCheck.addTokenListener(this.o)};this.V.onInit(i=>s(i)),setTimeout(()=>{if(!this.appCheck){const i=this.V.getImmediate({optional:!0});i?s(i):vt("FirebaseAppCheckTokenProvider","AppCheck not yet detected")}},0)}getToken(){if(this.p)return Promise.resolve(new Mh(this.p));const t=this.forceRefresh;return this.forceRefresh=!1,this.appCheck?this.appCheck.getToken(t).then(n=>n?(Bs(typeof n.token=="string",44558,{tokenResult:n}),this.m=n.token,new Mh(n.token)):null):Promise.resolve(null)}invalidateToken(){this.forceRefresh=!0}shutdown(){this.appCheck&&this.o&&this.appCheck.removeTokenListener(this.o),this.o=void 0}}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */function CE(e){const t=typeof self<"u"&&(self.crypto||self.msCrypto),n=new Uint8Array(e);if(t&&typeof t.getRandomValues=="function")t.getRandomValues(n);else for(let r=0;r<e;r++)n[r]=Math.floor(256*Math.random());return n}/**
 * @license
 * Copyright 2017 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class IE{static newId(){const t="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",n=62*Math.floor(4.129032258064516);let r="";for(;r.length<20;){const s=CE(40);for(let i=0;i<s.length;++i)r.length<20&&s[i]<n&&(r+=t.charAt(s[i]%62))}return r}}function In(e,t){return e<t?-1:e>t?1:0}function TE(e,t){const n=Math.min(e.length,t.length);for(let r=0;r<n;r++){const s=e.charAt(r),i=t.charAt(r);if(s!==i)return Nl(s)===Nl(i)?In(s,i):Nl(s)?1:-1}return In(e.length,t.length)}const NE=55296,jE=57343;function Nl(e){const t=e.charCodeAt(0);return t>=NE&&t<=jE}/**
 * @license
 * Copyright 2017 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const Lh="__name__";class Tt{constructor(t,n,r){n===void 0?n=0:n>t.length&&Zo(637,{offset:n,range:t.length}),r===void 0?r=t.length-n:r>t.length-n&&Zo(1746,{length:r,range:t.length-n}),this.segments=t,this.offset=n,this.len=r}get length(){return this.len}isEqual(t){return Tt.comparator(this,t)===0}child(t){const n=this.segments.slice(this.offset,this.limit());return t instanceof Tt?t.forEach(r=>{n.push(r)}):n.push(t),this.construct(n)}limit(){return this.offset+this.length}popFirst(t){return t=t===void 0?1:t,this.construct(this.segments,this.offset+t,this.length-t)}popLast(){return this.construct(this.segments,this.offset,this.length-1)}firstSegment(){return this.segments[this.offset]}lastSegment(){return this.get(this.length-1)}get(t){return this.segments[this.offset+t]}isEmpty(){return this.length===0}isPrefixOf(t){if(t.length<this.length)return!1;for(let n=0;n<this.length;n++)if(this.get(n)!==t.get(n))return!1;return!0}isImmediateParentOf(t){if(this.length+1!==t.length)return!1;for(let n=0;n<this.length;n++)if(this.get(n)!==t.get(n))return!1;return!0}forEach(t){for(let n=this.offset,r=this.limit();n<r;n++)t(this.segments[n])}toArray(){return this.segments.slice(this.offset,this.limit())}static comparator(t,n){const r=Math.min(t.length,n.length);for(let s=0;s<r;s++){const i=Tt.compareSegments(t.get(s),n.get(s));if(i!==0)return i}return In(t.length,n.length)}static compareSegments(t,n){const r=Tt.isNumericId(t),s=Tt.isNumericId(n);return r&&!s?-1:!r&&s?1:r&&s?Tt.extractNumericId(t).compare(Tt.extractNumericId(n)):TE(t,n)}static isNumericId(t){return t.startsWith("__id")&&t.endsWith("__")}static extractNumericId(t){return qc.fromString(t.substring(4,t.length-2))}}class yt extends Tt{construct(t,n,r){return new yt(t,n,r)}canonicalString(){return this.toArray().join("/")}toString(){return this.canonicalString()}toUriEncodedString(){return this.toArray().map(encodeURIComponent).join("/")}static fromString(...t){const n=[];for(const r of t){if(r.indexOf("//")>=0)throw new ee(Z.INVALID_ARGUMENT,`Invalid segment (${r}). Paths must not contain // in them.`);n.push(...r.split("/").filter(s=>s.length>0))}return new yt(n)}static emptyPath(){return new yt([])}}const AE=/^[_a-zA-Z][_a-zA-Z0-9]*$/;class Fn extends Tt{construct(t,n,r){return new Fn(t,n,r)}static isValidIdentifier(t){return AE.test(t)}canonicalString(){return this.toArray().map(t=>(t=t.replace(/\\/g,"\\\\").replace(/`/g,"\\`"),Fn.isValidIdentifier(t)||(t="`"+t+"`"),t)).join(".")}toString(){return this.canonicalString()}isKeyField(){return this.length===1&&this.get(0)===Lh}static keyField(){return new Fn([Lh])}static fromServerFormat(t){const n=[];let r="",s=0;const i=()=>{if(r.length===0)throw new ee(Z.INVALID_ARGUMENT,`Invalid field path (${t}). Paths must not be empty, begin with '.', end with '.', or contain '..'`);n.push(r),r=""};let a=!1;for(;s<t.length;){const u=t[s];if(u==="\\"){if(s+1===t.length)throw new ee(Z.INVALID_ARGUMENT,"Path has trailing escape character: "+t);const c=t[s+1];if(c!=="\\"&&c!=="."&&c!=="`")throw new ee(Z.INVALID_ARGUMENT,"Path has invalid escape sequence: "+t);r+=c,s+=2}else u==="`"?(a=!a,s++):u!=="."||a?(r+=u,s++):(i(),s++)}if(i(),a)throw new ee(Z.INVALID_ARGUMENT,"Unterminated ` in path: "+t);return new Fn(n)}static emptyPath(){return new Fn([])}}/**
 * @license
 * Copyright 2017 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class Gn{constructor(t){this.path=t}static fromPath(t){return new Gn(yt.fromString(t))}static fromName(t){return new Gn(yt.fromString(t).popFirst(5))}static empty(){return new Gn(yt.emptyPath())}get collectionGroup(){return this.path.popLast().lastSegment()}hasCollectionId(t){return this.path.length>=2&&this.path.get(this.path.length-2)===t}getCollectionGroup(){return this.path.get(this.path.length-2)}getCollectionPath(){return this.path.popLast()}isEqual(t){return t!==null&&yt.comparator(this.path,t.path)===0}toString(){return this.path.toString()}static comparator(t,n){return yt.comparator(t.path,n.path)}static isDocumentKey(t){return t.length%2==0}static fromSegments(t){return new Gn(new yt(t.slice()))}}function RE(e,t,n,r){if(t===!0&&r===!0)throw new ee(Z.INVALID_ARGUMENT,`${e} and ${n} cannot be used together.`)}function PE(e){return typeof e=="object"&&e!==null&&(Object.getPrototypeOf(e)===Object.prototype||Object.getPrototypeOf(e)===null)}/**
 * @license
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */function Se(e,t){const n={typeString:e};return t&&(n.value=t),n}function bi(e,t){if(!PE(e))throw new ee(Z.INVALID_ARGUMENT,"JSON must be an object");let n;for(const r in t)if(t[r]){const s=t[r].typeString,i="value"in t[r]?{value:t[r].value}:void 0;if(!(r in e)){n=`JSON missing required field: '${r}'`;break}const a=e[r];if(s&&typeof a!==s){n=`JSON field '${r}' must be a ${s}.`;break}if(i!==void 0&&a!==i.value){n=`Expected '${r}' field to equal '${i.value}'`;break}}if(n)throw new ee(Z.INVALID_ARGUMENT,n);return!0}/**
 * @license
 * Copyright 2017 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const Uh=-62135596800,Bh=1e6;class Nt{static now(){return Nt.fromMillis(Date.now())}static fromDate(t){return Nt.fromMillis(t.getTime())}static fromMillis(t){const n=Math.floor(t/1e3),r=Math.floor((t-1e3*n)*Bh);return new Nt(n,r)}constructor(t,n){if(this.seconds=t,this.nanoseconds=n,n<0)throw new ee(Z.INVALID_ARGUMENT,"Timestamp nanoseconds out of range: "+n);if(n>=1e9)throw new ee(Z.INVALID_ARGUMENT,"Timestamp nanoseconds out of range: "+n);if(t<Uh)throw new ee(Z.INVALID_ARGUMENT,"Timestamp seconds out of range: "+t);if(t>=253402300800)throw new ee(Z.INVALID_ARGUMENT,"Timestamp seconds out of range: "+t)}toDate(){return new Date(this.toMillis())}toMillis(){return 1e3*this.seconds+this.nanoseconds/Bh}_compareTo(t){return this.seconds===t.seconds?In(this.nanoseconds,t.nanoseconds):In(this.seconds,t.seconds)}isEqual(t){return t.seconds===this.seconds&&t.nanoseconds===this.nanoseconds}toString(){return"Timestamp(seconds="+this.seconds+", nanoseconds="+this.nanoseconds+")"}toJSON(){return{type:Nt._jsonSchemaVersion,seconds:this.seconds,nanoseconds:this.nanoseconds}}static fromJSON(t){if(bi(t,Nt._jsonSchema))return new Nt(t.seconds,t.nanoseconds)}valueOf(){const t=this.seconds-Uh;return String(t).padStart(12,"0")+"."+String(this.nanoseconds).padStart(9,"0")}}Nt._jsonSchemaVersion="firestore/timestamp/1.0",Nt._jsonSchema={type:Se("string",Nt._jsonSchemaVersion),seconds:Se("number"),nanoseconds:Se("number")};function OE(e){return e.name==="IndexedDbTransactionError"}/**
 * @license
 * Copyright 2023 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class DE extends Error{constructor(){super(...arguments),this.name="Base64DecodeError"}}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class nr{constructor(t){this.binaryString=t}static fromBase64String(t){const n=function(s){try{return atob(s)}catch(i){throw typeof DOMException<"u"&&i instanceof DOMException?new DE("Invalid base64 string: "+i):i}}(t);return new nr(n)}static fromUint8Array(t){const n=function(s){let i="";for(let a=0;a<s.length;++a)i+=String.fromCharCode(s[a]);return i}(t);return new nr(n)}[Symbol.iterator](){let t=0;return{next:()=>t<this.binaryString.length?{value:this.binaryString.charCodeAt(t++),done:!1}:{value:void 0,done:!0}}}toBase64(){return function(n){return btoa(n)}(this.binaryString)}toUint8Array(){return function(n){const r=new Uint8Array(n.length);for(let s=0;s<n.length;s++)r[s]=n.charCodeAt(s);return r}(this.binaryString)}approximateByteSize(){return 2*this.binaryString.length}compareTo(t){return In(this.binaryString,t.binaryString)}isEqual(t){return this.binaryString===t.binaryString}}nr.EMPTY_BYTE_STRING=new nr("");const Fh="(default)";class ea{constructor(t,n){this.projectId=t,this.database=n||Fh}static empty(){return new ea("","")}get isDefaultDatabase(){return this.database===Fh}isEqual(t){return t instanceof ea&&t.projectId===this.projectId&&t.database===this.database}}function ME(e,t){if(!Object.prototype.hasOwnProperty.apply(e.options,["projectId"]))throw new ee(Z.INVALID_ARGUMENT,'"projectId" not provided in firebase.initializeApp.');return new ea(e.options.projectId,t)}/**
 * @license
 * Copyright 2017 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class LE{constructor(t,n=null,r=[],s=[],i=null,a="F",u=null,c=null){this.path=t,this.collectionGroup=n,this.explicitOrderBy=r,this.filters=s,this.limit=i,this.limitType=a,this.startAt=u,this.endAt=c,this.Ie=null,this.Ee=null,this.Re=null,this.startAt,this.endAt}}function UE(e){return new LE(e)}/**
 * @license
 * Copyright 2017 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */var zh,X;(X=zh||(zh={}))[X.OK=0]="OK",X[X.CANCELLED=1]="CANCELLED",X[X.UNKNOWN=2]="UNKNOWN",X[X.INVALID_ARGUMENT=3]="INVALID_ARGUMENT",X[X.DEADLINE_EXCEEDED=4]="DEADLINE_EXCEEDED",X[X.NOT_FOUND=5]="NOT_FOUND",X[X.ALREADY_EXISTS=6]="ALREADY_EXISTS",X[X.PERMISSION_DENIED=7]="PERMISSION_DENIED",X[X.UNAUTHENTICATED=16]="UNAUTHENTICATED",X[X.RESOURCE_EXHAUSTED=8]="RESOURCE_EXHAUSTED",X[X.FAILED_PRECONDITION=9]="FAILED_PRECONDITION",X[X.ABORTED=10]="ABORTED",X[X.OUT_OF_RANGE=11]="OUT_OF_RANGE",X[X.UNIMPLEMENTED=12]="UNIMPLEMENTED",X[X.INTERNAL=13]="INTERNAL",X[X.UNAVAILABLE=14]="UNAVAILABLE",X[X.DATA_LOSS=15]="DATA_LOSS";/**
 * @license
 * Copyright 2022 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */new qc([4294967295,4294967295],0);/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const BE=41943040;/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const FE=1048576;function jl(){return typeof document<"u"?document:null}class zE{constructor(t,n,r=1e3,s=1.5,i=6e4){this.Ci=t,this.timerId=n,this.R_=r,this.A_=s,this.V_=i,this.d_=0,this.m_=null,this.f_=Date.now(),this.reset()}reset(){this.d_=0}g_(){this.d_=this.V_}p_(t){this.cancel();const n=Math.floor(this.d_+this.y_()),r=Math.max(0,Date.now()-this.f_),s=Math.max(0,n-r);s>0&&vt("ExponentialBackoff",`Backing off for ${s} ms (base delay: ${this.d_} ms, delay with jitter: ${n} ms, last attempt: ${r} ms ago)`),this.m_=this.Ci.enqueueAfterDelay(this.timerId,s,()=>(this.f_=Date.now(),t())),this.d_*=this.A_,this.d_<this.R_&&(this.d_=this.R_),this.d_>this.V_&&(this.d_=this.V_)}w_(){this.m_!==null&&(this.m_.skipDelay(),this.m_=null)}cancel(){this.m_!==null&&(this.m_.cancel(),this.m_=null)}y_(){return(Math.random()-.5)*this.d_}}/**
 * @license
 * Copyright 2017 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class Kc{constructor(t,n,r,s,i){this.asyncQueue=t,this.timerId=n,this.targetTimeMs=r,this.op=s,this.removalCallback=i,this.deferred=new Fs,this.then=this.deferred.promise.then.bind(this.deferred.promise),this.deferred.promise.catch(a=>{})}get promise(){return this.deferred.promise}static createAndSchedule(t,n,r,s,i){const a=Date.now()+r,u=new Kc(t,n,a,s,i);return u.start(r),u}start(t){this.timerHandle=setTimeout(()=>this.handleDelayElapsed(),t)}skipDelay(){return this.handleDelayElapsed()}cancel(t){this.timerHandle!==null&&(this.clearTimeout(),this.deferred.reject(new ee(Z.CANCELLED,"Operation cancelled"+(t?": "+t:""))))}handleDelayElapsed(){this.asyncQueue.enqueueAndForget(()=>this.timerHandle!==null?(this.clearTimeout(),this.op().then(t=>this.deferred.resolve(t))):Promise.resolve())}clearTimeout(){this.timerHandle!==null&&(this.removalCallback(this),clearTimeout(this.timerHandle),this.timerHandle=null)}}var Vh,$h;($h=Vh||(Vh={})).Ma="default",$h.Cache="cache";/**
 * @license
 * Copyright 2023 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */function VE(e){const t={};return e.timeoutSeconds!==void 0&&(t.timeoutSeconds=e.timeoutSeconds),t}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const $E="ComponentProvider",Hh=new Map;/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const HE="firestore.googleapis.com",Gh=!0;class Wh{constructor(t){if(t.host===void 0){if(t.ssl!==void 0)throw new ee(Z.INVALID_ARGUMENT,"Can't provide ssl option if host option is not set");this.host=HE,this.ssl=Gh}else this.host=t.host,this.ssl=t.ssl??Gh;if(this.isUsingEmulator=t.emulatorOptions!==void 0,this.credentials=t.credentials,this.ignoreUndefinedProperties=!!t.ignoreUndefinedProperties,this.localCache=t.localCache,t.cacheSizeBytes===void 0)this.cacheSizeBytes=BE;else{if(t.cacheSizeBytes!==-1&&t.cacheSizeBytes<FE)throw new ee(Z.INVALID_ARGUMENT,"cacheSizeBytes must be at least 1048576");this.cacheSizeBytes=t.cacheSizeBytes}RE("experimentalForceLongPolling",t.experimentalForceLongPolling,"experimentalAutoDetectLongPolling",t.experimentalAutoDetectLongPolling),this.experimentalForceLongPolling=!!t.experimentalForceLongPolling,this.experimentalForceLongPolling?this.experimentalAutoDetectLongPolling=!1:t.experimentalAutoDetectLongPolling===void 0?this.experimentalAutoDetectLongPolling=!0:this.experimentalAutoDetectLongPolling=!!t.experimentalAutoDetectLongPolling,this.experimentalLongPollingOptions=VE(t.experimentalLongPollingOptions??{}),function(r){if(r.timeoutSeconds!==void 0){if(isNaN(r.timeoutSeconds))throw new ee(Z.INVALID_ARGUMENT,`invalid long polling timeout: ${r.timeoutSeconds} (must not be NaN)`);if(r.timeoutSeconds<5)throw new ee(Z.INVALID_ARGUMENT,`invalid long polling timeout: ${r.timeoutSeconds} (minimum allowed value is 5)`);if(r.timeoutSeconds>30)throw new ee(Z.INVALID_ARGUMENT,`invalid long polling timeout: ${r.timeoutSeconds} (maximum allowed value is 30)`)}}(this.experimentalLongPollingOptions),this.useFetchStreams=!!t.useFetchStreams}isEqual(t){return this.host===t.host&&this.ssl===t.ssl&&this.credentials===t.credentials&&this.cacheSizeBytes===t.cacheSizeBytes&&this.experimentalForceLongPolling===t.experimentalForceLongPolling&&this.experimentalAutoDetectLongPolling===t.experimentalAutoDetectLongPolling&&function(r,s){return r.timeoutSeconds===s.timeoutSeconds}(this.experimentalLongPollingOptions,t.experimentalLongPollingOptions)&&this.ignoreUndefinedProperties===t.ignoreUndefinedProperties&&this.useFetchStreams===t.useFetchStreams}}class GE{constructor(t,n,r,s){this._authCredentials=t,this._appCheckCredentials=n,this._databaseId=r,this._app=s,this.type="firestore-lite",this._persistenceKey="(lite)",this._settings=new Wh({}),this._settingsFrozen=!1,this._emulatorOptions={},this._terminateTask="notTerminated"}get app(){if(!this._app)throw new ee(Z.FAILED_PRECONDITION,"Firestore was not initialized using the Firebase SDK. 'app' is not available");return this._app}get _initialized(){return this._settingsFrozen}get _terminated(){return this._terminateTask!=="notTerminated"}_setSettings(t){if(this._settingsFrozen)throw new ee(Z.FAILED_PRECONDITION,"Firestore has already been started and its settings can no longer be changed. You can only modify settings before calling any other methods on a Firestore object.");this._settings=new Wh(t),this._emulatorOptions=t.emulatorOptions||{},t.credentials!==void 0&&(this._authCredentials=function(r){if(!r)return new SE;switch(r.type){case"firstParty":return new bE(r.sessionIndex||"0",r.iamToken||null,r.authTokenFactory||null);case"provider":return r.client;default:throw new ee(Z.INVALID_ARGUMENT,"makeAuthCredentialsProvider failed due to invalid credential type")}}(t.credentials))}_getSettings(){return this._settings}_getEmulatorOptions(){return this._emulatorOptions}_freezeSettings(){return this._settingsFrozen=!0,this._settings}_delete(){return this._terminateTask==="notTerminated"&&(this._terminateTask=this._terminate()),this._terminateTask}async _restart(){this._terminateTask==="notTerminated"?await this._terminate():this._terminateTask="notTerminated"}toJSON(){return{app:this._app,databaseId:this._databaseId,settings:this._settings}}_terminate(){return function(n){const r=Hh.get(n);r&&(vt($E,"Removing Datastore"),Hh.delete(n),r.terminate())}(this),Promise.resolve()}}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class Xc{constructor(t,n,r){this.converter=n,this._query=r,this.type="query",this.firestore=t}withConverter(t){return new Xc(this.firestore,t,this._query)}}class Rt{constructor(t,n,r){this.converter=n,this._key=r,this.type="document",this.firestore=t}get _path(){return this._key.path}get id(){return this._key.path.lastSegment()}get path(){return this._key.path.canonicalString()}get parent(){return new Jc(this.firestore,this.converter,this._key.path.popLast())}withConverter(t){return new Rt(this.firestore,t,this._key)}toJSON(){return{type:Rt._jsonSchemaVersion,referencePath:this._key.toString()}}static fromJSON(t,n,r){if(bi(n,Rt._jsonSchema))return new Rt(t,r||null,new Gn(yt.fromString(n.referencePath)))}}Rt._jsonSchemaVersion="firestore/documentReference/1.0",Rt._jsonSchema={type:Se("string",Rt._jsonSchemaVersion),referencePath:Se("string")};class Jc extends Xc{constructor(t,n,r){super(t,n,UE(r)),this._path=r,this.type="collection"}get id(){return this._query.path.lastSegment()}get path(){return this._query.path.canonicalString()}get parent(){const t=this._path.popLast();return t.isEmpty()?null:new Rt(this.firestore,null,new Gn(t))}withConverter(t){return new Jc(this.firestore,t,this._path)}}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const qh="AsyncQueue";class Kh{constructor(t=Promise.resolve()){this.Yu=[],this.ec=!1,this.tc=[],this.nc=null,this.rc=!1,this.sc=!1,this.oc=[],this.M_=new zE(this,"async_queue_retry"),this._c=()=>{const r=jl();r&&vt(qh,"Visibility state changed to "+r.visibilityState),this.M_.w_()},this.ac=t;const n=jl();n&&typeof n.addEventListener=="function"&&n.addEventListener("visibilitychange",this._c)}get isShuttingDown(){return this.ec}enqueueAndForget(t){this.enqueue(t)}enqueueAndForgetEvenWhileRestricted(t){this.uc(),this.cc(t)}enterRestrictedMode(t){if(!this.ec){this.ec=!0,this.sc=t||!1;const n=jl();n&&typeof n.removeEventListener=="function"&&n.removeEventListener("visibilitychange",this._c)}}enqueue(t){if(this.uc(),this.ec)return new Promise(()=>{});const n=new Fs;return this.cc(()=>this.ec&&this.sc?Promise.resolve():(t().then(n.resolve,n.reject),n.promise)).then(()=>n.promise)}enqueueRetryable(t){this.enqueueAndForget(()=>(this.Yu.push(t),this.lc()))}async lc(){if(this.Yu.length!==0){try{await this.Yu[0](),this.Yu.shift(),this.M_.reset()}catch(t){if(!OE(t))throw t;vt(qh,"Operation failed with retryable error: "+t)}this.Yu.length>0&&this.M_.p_(()=>this.lc())}}cc(t){const n=this.ac.then(()=>(this.rc=!0,t().catch(r=>{throw this.nc=r,this.rc=!1,iy("INTERNAL UNHANDLED ERROR: ",Xh(r)),r}).then(r=>(this.rc=!1,r))));return this.ac=n,n}enqueueAfterDelay(t,n,r){this.uc(),this.oc.indexOf(t)>-1&&(n=0);const s=Kc.createAndSchedule(this,t,n,r,i=>this.hc(i));return this.tc.push(s),s}uc(){this.nc&&Zo(47125,{Pc:Xh(this.nc)})}verifyOperationInProgress(){}async Tc(){let t;do t=this.ac,await t;while(t!==this.ac)}Ic(t){for(const n of this.tc)if(n.timerId===t)return!0;return!1}Ec(t){return this.Tc().then(()=>{this.tc.sort((n,r)=>n.targetTimeMs-r.targetTimeMs);for(const n of this.tc)if(n.skipDelay(),t!=="all"&&n.timerId===t)break;return this.Tc()})}Rc(t){this.oc.push(t)}hc(t){const n=this.tc.indexOf(t);this.tc.splice(n,1)}}function Xh(e){let t=e.message||"";return e.stack&&(t=e.stack.includes(e.message)?e.stack:e.message+`
`+e.stack),t}class WE extends GE{constructor(t,n,r,s){super(t,n,r,s),this.type="firestore",this._queue=new Kh,this._persistenceKey=(s==null?void 0:s.name)||"[DEFAULT]"}async _terminate(){if(this._firestoreClient){const t=this._firestoreClient.terminate();this._queue=new Kh(t),this._firestoreClient=void 0,await t}}}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class Ut{constructor(t){this._byteString=t}static fromBase64String(t){try{return new Ut(nr.fromBase64String(t))}catch(n){throw new ee(Z.INVALID_ARGUMENT,"Failed to construct data from Base64 string: "+n)}}static fromUint8Array(t){return new Ut(nr.fromUint8Array(t))}toBase64(){return this._byteString.toBase64()}toUint8Array(){return this._byteString.toUint8Array()}toString(){return"Bytes(base64: "+this.toBase64()+")"}isEqual(t){return this._byteString.isEqual(t._byteString)}toJSON(){return{type:Ut._jsonSchemaVersion,bytes:this.toBase64()}}static fromJSON(t){if(bi(t,Ut._jsonSchema))return Ut.fromBase64String(t.bytes)}}Ut._jsonSchemaVersion="firestore/bytes/1.0",Ut._jsonSchema={type:Se("string",Ut._jsonSchemaVersion),bytes:Se("string")};/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class ly{constructor(...t){for(let n=0;n<t.length;++n)if(t[n].length===0)throw new ee(Z.INVALID_ARGUMENT,"Invalid field name at argument $(i + 1). Field names must not be empty.");this._internalPath=new Fn(t)}isEqual(t){return this._internalPath.isEqual(t._internalPath)}}/**
 * @license
 * Copyright 2017 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class Kn{constructor(t,n){if(!isFinite(t)||t<-90||t>90)throw new ee(Z.INVALID_ARGUMENT,"Latitude must be a number between -90 and 90, but was: "+t);if(!isFinite(n)||n<-180||n>180)throw new ee(Z.INVALID_ARGUMENT,"Longitude must be a number between -180 and 180, but was: "+n);this._lat=t,this._long=n}get latitude(){return this._lat}get longitude(){return this._long}isEqual(t){return this._lat===t._lat&&this._long===t._long}_compareTo(t){return In(this._lat,t._lat)||In(this._long,t._long)}toJSON(){return{latitude:this._lat,longitude:this._long,type:Kn._jsonSchemaVersion}}static fromJSON(t){if(bi(t,Kn._jsonSchema))return new Kn(t.latitude,t.longitude)}}Kn._jsonSchemaVersion="firestore/geoPoint/1.0",Kn._jsonSchema={type:Se("string",Kn._jsonSchemaVersion),latitude:Se("number"),longitude:Se("number")};/**
 * @license
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class Xn{constructor(t){this._values=(t||[]).map(n=>n)}toArray(){return this._values.map(t=>t)}isEqual(t){return function(r,s){if(r.length!==s.length)return!1;for(let i=0;i<r.length;++i)if(r[i]!==s[i])return!1;return!0}(this._values,t._values)}toJSON(){return{type:Xn._jsonSchemaVersion,vectorValues:this._values}}static fromJSON(t){if(bi(t,Xn._jsonSchema)){if(Array.isArray(t.vectorValues)&&t.vectorValues.every(n=>typeof n=="number"))return new Xn(t.vectorValues);throw new ee(Z.INVALID_ARGUMENT,"Expected 'vectorValues' field to be a number array")}}}Xn._jsonSchemaVersion="firestore/vectorValue/1.0",Xn._jsonSchema={type:Se("string",Xn._jsonSchemaVersion),vectorValues:Se("object")};function uy(e,t,n){if((t=Ei(t))instanceof ly)return t._internalPath;if(typeof t=="string")return KE(e,t);throw Uu("Field path arguments must be of type string or ",e)}const qE=new RegExp("[~\\*/\\[\\]]");function KE(e,t,n){if(t.search(qE)>=0)throw Uu(`Invalid field path (${t}). Paths must not contain '~', '*', '/', '[', or ']'`,e);try{return new ly(...t.split("."))._internalPath}catch{throw Uu(`Invalid field path (${t}). Paths must not be empty, begin with '.', end with '.', or contain '..'`,e)}}function Uu(e,t,n,r,s){let i=`Function ${t}() called with invalid data`;i+=". ";let a="";return new ee(Z.INVALID_ARGUMENT,i+e+a)}const Jh="@firebase/firestore",Yh="4.12.0";/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class cy{constructor(t,n,r,s,i){this._firestore=t,this._userDataWriter=n,this._key=r,this._document=s,this._converter=i}get id(){return this._key.path.lastSegment()}get ref(){return new Rt(this._firestore,this._converter,this._key)}exists(){return this._document!==null}data(){if(this._document){if(this._converter){const t=new XE(this._firestore,this._userDataWriter,this._key,this._document,null);return this._converter.fromFirestore(t)}return this._userDataWriter.convertValue(this._document.data.value)}}_fieldsProto(){var t;return((t=this._document)==null?void 0:t.data.clone().value.mapValue.fields)??void 0}get(t){if(this._document){const n=this._document.data.field(uy("DocumentSnapshot.get",t));if(n!==null)return this._userDataWriter.convertValue(n)}}}class XE extends cy{data(){return super.data()}}class to{constructor(t,n){this.hasPendingWrites=t,this.fromCache=n}isEqual(t){return this.hasPendingWrites===t.hasPendingWrites&&this.fromCache===t.fromCache}}class Dr extends cy{constructor(t,n,r,s,i,a){super(t,n,r,s,a),this._firestore=t,this._firestoreImpl=t,this.metadata=i}exists(){return super.exists()}data(t={}){if(this._document){if(this._converter){const n=new So(this._firestore,this._userDataWriter,this._key,this._document,this.metadata,null);return this._converter.fromFirestore(n,t)}return this._userDataWriter.convertValue(this._document.data.value,t.serverTimestamps)}}get(t,n={}){if(this._document){const r=this._document.data.field(uy("DocumentSnapshot.get",t));if(r!==null)return this._userDataWriter.convertValue(r,n.serverTimestamps)}}toJSON(){if(this.metadata.hasPendingWrites)throw new ee(Z.FAILED_PRECONDITION,"DocumentSnapshot.toJSON() attempted to serialize a document with pending writes. Await waitForPendingWrites() before invoking toJSON().");const t=this._document,n={};return n.type=Dr._jsonSchemaVersion,n.bundle="",n.bundleSource="DocumentSnapshot",n.bundleName=this._key.toString(),!t||!t.isValidDocument()||!t.isFoundDocument()?n:(this._userDataWriter.convertObjectMap(t.data.value.mapValue.fields,"previous"),n.bundle=(this._firestore,this.ref.path,"NOT SUPPORTED"),n)}}Dr._jsonSchemaVersion="firestore/documentSnapshot/1.0",Dr._jsonSchema={type:Se("string",Dr._jsonSchemaVersion),bundleSource:Se("string","DocumentSnapshot"),bundleName:Se("string"),bundle:Se("string")};class So extends Dr{data(t={}){return super.data(t)}}class zs{constructor(t,n,r,s){this._firestore=t,this._userDataWriter=n,this._snapshot=s,this.metadata=new to(s.hasPendingWrites,s.fromCache),this.query=r}get docs(){const t=[];return this.forEach(n=>t.push(n)),t}get size(){return this._snapshot.docs.size}get empty(){return this.size===0}forEach(t,n){this._snapshot.docs.forEach(r=>{t.call(n,new So(this._firestore,this._userDataWriter,r.key,r,new to(this._snapshot.mutatedKeys.has(r.key),this._snapshot.fromCache),this.query.converter))})}docChanges(t={}){const n=!!t.includeMetadataChanges;if(n&&this._snapshot.excludesMetadataChanges)throw new ee(Z.INVALID_ARGUMENT,"To include metadata changes with your document changes, you must also pass { includeMetadataChanges:true } to onSnapshot().");return this._cachedChanges&&this._cachedChangesIncludeMetadataChanges===n||(this._cachedChanges=function(s,i){if(s._snapshot.oldDocs.isEmpty()){let a=0;return s._snapshot.docChanges.map(u=>{const c=new So(s._firestore,s._userDataWriter,u.doc.key,u.doc,new to(s._snapshot.mutatedKeys.has(u.doc.key),s._snapshot.fromCache),s.query.converter);return u.doc,{type:"added",doc:c,oldIndex:-1,newIndex:a++}})}{let a=s._snapshot.oldDocs;return s._snapshot.docChanges.filter(u=>i||u.type!==3).map(u=>{const c=new So(s._firestore,s._userDataWriter,u.doc.key,u.doc,new to(s._snapshot.mutatedKeys.has(u.doc.key),s._snapshot.fromCache),s.query.converter);let f=-1,E=-1;return u.type!==0&&(f=a.indexOf(u.doc.key),a=a.delete(u.doc.key)),u.type!==1&&(a=a.add(u.doc),E=a.indexOf(u.doc.key)),{type:JE(u.type),doc:c,oldIndex:f,newIndex:E}})}}(this,n),this._cachedChangesIncludeMetadataChanges=n),this._cachedChanges}toJSON(){if(this.metadata.hasPendingWrites)throw new ee(Z.FAILED_PRECONDITION,"QuerySnapshot.toJSON() attempted to serialize a document with pending writes. Await waitForPendingWrites() before invoking toJSON().");const t={};t.type=zs._jsonSchemaVersion,t.bundleSource="QuerySnapshot",t.bundleName=IE.newId(),this._firestore._databaseId.database,this._firestore._databaseId.projectId;const n=[],r=[],s=[];return this.docs.forEach(i=>{i._document!==null&&(n.push(i._document),r.push(this._userDataWriter.convertObjectMap(i._document.data.value.mapValue.fields,"previous")),s.push(i.ref.path))}),t.bundle=(this._firestore,this.query._query,t.bundleName,"NOT SUPPORTED"),t}}function JE(e){switch(e){case 0:return"added";case 2:case 3:return"modified";case 1:return"removed";default:return Zo(61501,{type:e})}}/**
 * @license
 * Copyright 2022 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */zs._jsonSchemaVersion="firestore/querySnapshot/1.0",zs._jsonSchema={type:Se("string",zs._jsonSchemaVersion),bundleSource:Se("string","QuerySnapshot"),bundleName:Se("string"),bundle:Se("string")};(function(t,n=!0){wE(wa),Wr(new Gr("firestore",(r,{instanceIdentifier:s,options:i})=>{const a=r.getProvider("app").getImmediate(),u=new WE(new EE(r.getProvider("auth-internal")),new _E(a,r.getProvider("app-check-internal")),ME(a,s),a);return i={useFetchStreams:n,...i},u._setSettings(i),u},"PUBLIC").setMultipleInstances(!0)),kn(Jh,Yh,t),kn(Jh,Yh,"esm2020")})();const YE={apiKey:void 0},QE=YE.apiKey;console.warn("Firebase configuration is missing or invalid. Authentication features will be disabled.");const dy=N.createContext({}),ci=()=>{const e=N.useContext(dy);if(!e)throw new Error("useAuth must be used within an AuthProvider");return e},ZE=({children:e})=>{const[t,n]=N.useState(null),[r,s]=N.useState(!0);N.useEffect(()=>{{s(!1);return}},[]);const c={user:t,loading:r,signUp:async(f,E,w)=>({data:null,error:{message:"Firebase is not configured. Please check your .env file."}}),signIn:async(f,E)=>({data:null,error:{message:"Firebase is not configured. Please check your .env file."}}),signOut:async()=>({error:null}),isConfigured:QE};return m.jsx(dy.Provider,{value:c,children:r?m.jsx("div",{className:"min-h-screen bg-slate-900 flex items-center justify-center",children:m.jsxs("div",{className:"flex flex-col items-center gap-4",children:[m.jsx("div",{className:"w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"}),m.jsx("p",{className:"text-slate-400 font-medium",children:"Loading CodeSandbox..."})]})}):e})},Qh=({children:e})=>{const{user:t,isConfigured:n}=ci();return n&&!t?m.jsx(Mg,{to:"/login",replace:!0}):e};/**
 * @license lucide-react v0.309.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */var e2={xmlns:"http://www.w3.org/2000/svg",width:24,height:24,viewBox:"0 0 24 24",fill:"none",stroke:"currentColor",strokeWidth:2,strokeLinecap:"round",strokeLinejoin:"round"};/**
 * @license lucide-react v0.309.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const t2=e=>e.replace(/([a-z0-9])([A-Z])/g,"$1-$2").toLowerCase().trim(),se=(e,t)=>{const n=N.forwardRef(({color:r="currentColor",size:s=24,strokeWidth:i=2,absoluteStrokeWidth:a,className:u="",children:c,...f},E)=>N.createElement("svg",{ref:E,...e2,width:s,height:s,stroke:r,strokeWidth:a?Number(i)*24/Number(s):i,className:["lucide",`lucide-${t2(e)}`,u].join(" "),...f},[...t.map(([w,_])=>N.createElement(w,_)),...Array.isArray(c)?c:[c]]));return n.displayName=`${e}`,n};/**
 * @license lucide-react v0.309.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const di=se("AlertCircle",[["circle",{cx:"12",cy:"12",r:"10",key:"1mglay"}],["line",{x1:"12",x2:"12",y1:"8",y2:"12",key:"1pkeuh"}],["line",{x1:"12",x2:"12.01",y1:"16",y2:"16",key:"4dfq90"}]]);/**
 * @license lucide-react v0.309.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const n2=se("ArrowLeft",[["path",{d:"m12 19-7-7 7-7",key:"1l729n"}],["path",{d:"M19 12H5",key:"x3x0zl"}]]);/**
 * @license lucide-react v0.309.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const fy=se("CheckCircle2",[["circle",{cx:"12",cy:"12",r:"10",key:"1mglay"}],["path",{d:"m9 12 2 2 4-4",key:"dzmm74"}]]);/**
 * @license lucide-react v0.309.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Yc=se("CheckCircle",[["path",{d:"M22 11.08V12a10 10 0 1 1-5.93-9.14",key:"g774vq"}],["path",{d:"m9 11 3 3L22 4",key:"1pflzl"}]]);/**
 * @license lucide-react v0.309.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const r2=se("ChevronDown",[["path",{d:"m6 9 6 6 6-6",key:"qrunsl"}]]);/**
 * @license lucide-react v0.309.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Qc=se("Code2",[["path",{d:"m18 16 4-4-4-4",key:"1inbqp"}],["path",{d:"m6 8-4 4 4 4",key:"15zrgr"}],["path",{d:"m14.5 4-5 16",key:"e7oirm"}]]);/**
 * @license lucide-react v0.309.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const s2=se("Container",[["path",{d:"M22 7.7c0-.6-.4-1.2-.8-1.5l-6.3-3.9a1.72 1.72 0 0 0-1.7 0l-10.3 6c-.5.2-.9.8-.9 1.4v6.6c0 .5.4 1.2.8 1.5l6.3 3.9a1.72 1.72 0 0 0 1.7 0l10.3-6c.5-.3.9-1 .9-1.5Z",key:"1t2lqe"}],["path",{d:"M10 21.9V14L2.1 9.1",key:"o7czzq"}],["path",{d:"m10 14 11.9-6.9",key:"zm5e20"}],["path",{d:"M14 19.8v-8.1",key:"159ecu"}],["path",{d:"M18 17.5V9.4",key:"11uown"}]]);/**
 * @license lucide-react v0.309.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const hy=se("EyeOff",[["path",{d:"M9.88 9.88a3 3 0 1 0 4.24 4.24",key:"1jxqfv"}],["path",{d:"M10.73 5.08A10.43 10.43 0 0 1 12 5c7 0 10 7 10 7a13.16 13.16 0 0 1-1.67 2.68",key:"9wicm4"}],["path",{d:"M6.61 6.61A13.526 13.526 0 0 0 2 12s3 7 10 7a9.74 9.74 0 0 0 5.39-1.61",key:"1jreej"}],["line",{x1:"2",x2:"22",y1:"2",y2:"22",key:"a6p6uj"}]]);/**
 * @license lucide-react v0.309.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const fi=se("Eye",[["path",{d:"M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z",key:"rwhkz3"}],["circle",{cx:"12",cy:"12",r:"3",key:"1v7zrd"}]]);/**
 * @license lucide-react v0.309.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const i2=se("FileCode",[["path",{d:"M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z",key:"1nnpy2"}],["polyline",{points:"14 2 14 8 20 8",key:"1ew0cm"}],["path",{d:"m10 13-2 2 2 2",key:"17smn8"}],["path",{d:"m14 17 2-2-2-2",key:"14mezr"}]]);/**
 * @license lucide-react v0.309.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const o2=se("FileText",[["path",{d:"M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z",key:"1nnpy2"}],["polyline",{points:"14 2 14 8 20 8",key:"1ew0cm"}],["line",{x1:"16",x2:"8",y1:"13",y2:"13",key:"14keom"}],["line",{x1:"16",x2:"8",y1:"17",y2:"17",key:"17nazh"}],["line",{x1:"10",x2:"8",y1:"9",y2:"9",key:"1a5vjj"}]]);/**
 * @license lucide-react v0.309.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const a2=se("Folder",[["path",{d:"M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z",key:"1kt360"}]]);/**
 * @license lucide-react v0.309.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const l2=se("Lightbulb",[["path",{d:"M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5",key:"1gvzjb"}],["path",{d:"M9 18h6",key:"x1upvd"}],["path",{d:"M10 22h4",key:"ceow96"}]]);/**
 * @license lucide-react v0.309.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const u2=se("Loader2",[["path",{d:"M21 12a9 9 0 1 1-6.219-8.56",key:"13zald"}]]);/**
 * @license lucide-react v0.309.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const c2=se("LogOut",[["path",{d:"M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4",key:"1uf3rs"}],["polyline",{points:"16 17 21 12 16 7",key:"1gabdz"}],["line",{x1:"21",x2:"9",y1:"12",y2:"12",key:"1uyos4"}]]);/**
 * @license lucide-react v0.309.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const d2=se("MessageCircle",[["path",{d:"M7.9 20A9 9 0 1 0 4 16.1L2 22Z",key:"vv11sd"}]]);/**
 * @license lucide-react v0.309.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const f2=se("Play",[["polygon",{points:"5 3 19 12 5 21 5 3",key:"191637"}]]);/**
 * @license lucide-react v0.309.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Zh=se("RefreshCw",[["path",{d:"M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8",key:"v9h5vc"}],["path",{d:"M21 3v5h-5",key:"1q7to0"}],["path",{d:"M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16",key:"3uifl3"}],["path",{d:"M8 16H3v5",key:"1cv678"}]]);/**
 * @license lucide-react v0.309.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const h2=se("RotateCcw",[["path",{d:"M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8",key:"1357e3"}],["path",{d:"M3 3v5h5",key:"1xhq8a"}]]);/**
 * @license lucide-react v0.309.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const p2=se("Save",[["path",{d:"M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z",key:"1owoqh"}],["polyline",{points:"17 21 17 13 7 13 7 21",key:"1md35c"}],["polyline",{points:"7 3 7 8 15 8",key:"8nz8an"}]]);/**
 * @license lucide-react v0.309.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const m2=se("Send",[["path",{d:"m22 2-7 20-4-9-9-4Z",key:"1q3vgg"}],["path",{d:"M22 2 11 13",key:"nzbqef"}]]);/**
 * @license lucide-react v0.309.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const g2=se("Sparkles",[["path",{d:"m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z",key:"17u4zn"}],["path",{d:"M5 3v4",key:"bklmnn"}],["path",{d:"M19 17v4",key:"iiml17"}],["path",{d:"M3 5h4",key:"nem4j1"}],["path",{d:"M17 19h4",key:"lbex7p"}]]);/**
 * @license lucide-react v0.309.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const y2=se("Target",[["circle",{cx:"12",cy:"12",r:"10",key:"1mglay"}],["circle",{cx:"12",cy:"12",r:"6",key:"1vlfrh"}],["circle",{cx:"12",cy:"12",r:"2",key:"1c9p78"}]]);/**
 * @license lucide-react v0.309.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const v2=se("Trophy",[["path",{d:"M6 9H4.5a2.5 2.5 0 0 1 0-5H6",key:"17hqa7"}],["path",{d:"M18 9h1.5a2.5 2.5 0 0 0 0-5H18",key:"lmptdp"}],["path",{d:"M4 22h16",key:"57wxv0"}],["path",{d:"M10 14.66V17c0 .55-.47.98-.97 1.21C7.85 18.75 7 20.24 7 22",key:"1nw9bq"}],["path",{d:"M14 14.66V17c0 .55.47.98.97 1.21C16.15 18.75 17 20.24 17 22",key:"1np0yb"}],["path",{d:"M18 2H6v7a6 6 0 0 0 12 0V2Z",key:"u46fv3"}]]);/**
 * @license lucide-react v0.309.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const w2=se("WifiOff",[["line",{x1:"2",x2:"22",y1:"2",y2:"22",key:"a6p6uj"}],["path",{d:"M8.5 16.5a5 5 0 0 1 7 0",key:"sej527"}],["path",{d:"M2 8.82a15 15 0 0 1 4.17-2.65",key:"11utq1"}],["path",{d:"M10.66 5c4.01-.36 8.14.9 11.34 3.76",key:"hxefdu"}],["path",{d:"M16.85 11.25a10 10 0 0 1 2.22 1.68",key:"q734kn"}],["path",{d:"M5 13a10 10 0 0 1 5.24-2.76",key:"piq4yl"}],["line",{x1:"12",x2:"12.01",y1:"20",y2:"20",key:"of4bc4"}]]);/**
 * @license lucide-react v0.309.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const x2=se("Wifi",[["path",{d:"M5 13a10 10 0 0 1 14 0",key:"6v8j51"}],["path",{d:"M8.5 16.5a5 5 0 0 1 7 0",key:"sej527"}],["path",{d:"M2 8.82a15 15 0 0 1 20 0",key:"dnpr2z"}],["line",{x1:"12",x2:"12.01",y1:"20",y2:"20",key:"of4bc4"}]]);/**
 * @license lucide-react v0.309.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const S2=se("XCircle",[["circle",{cx:"12",cy:"12",r:"10",key:"1mglay"}],["path",{d:"m15 9-6 6",key:"1uzhvr"}],["path",{d:"m9 9 6 6",key:"z0biqf"}]]);/**
 * @license lucide-react v0.309.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const E2=se("X",[["path",{d:"M18 6 6 18",key:"1bl5f8"}],["path",{d:"m6 6 12 12",key:"d8bk6v"}]]),k2=()=>{const e=Qr(),{signIn:t,isConfigured:n}=ci(),[r,s]=N.useState({username:"",password:""}),[i,a]=N.useState(""),[u,c]=N.useState(!1),[f,E]=N.useState(!1),w=async _=>{if(_.preventDefault(),a(""),c(!0),!r.username||!r.password){a("Please fill in all fields"),c(!1);return}try{const{data:A,error:R}=await t(r.username,r.password);if(R){a(R.message||"Invalid username or password"),c(!1);return}A&&e("/dashboard")}catch{a("An unexpected error occurred. Please try again."),c(!1)}};return m.jsxs("div",{className:"relative min-h-screen flex items-center justify-center overflow-hidden font-sans",children:[m.jsxs("div",{className:"bg-streaks",children:[m.jsx("div",{className:"streak-1"}),m.jsx("div",{className:"streak-2"})]}),m.jsxs("div",{className:"glass-dark p-10 rounded-2xl shadow-2xl w-full max-w-md border border-white/10 relative z-10 backdrop-blur-3xl",children:[m.jsxs("div",{className:"flex flex-col items-center mb-10",children:[m.jsx("div",{className:"text-white mb-3 p-3 glass rounded-2xl neon-border-magenta",children:m.jsx(Qc,{size:40,className:"text-accent-magenta"})}),m.jsx("h1",{className:"text-4xl font-bold text-white tracking-tight",children:"CodeSandbox"}),m.jsx("p",{className:"text-slate-400 mt-2 text-sm uppercase tracking-widest font-semibold",children:"Scenario IDE"})]}),!n&&m.jsxs("div",{className:"mb-6 p-4 glass rounded-xl flex flex-col gap-2 border-blue-500/30",children:[m.jsxs("div",{className:"flex items-center gap-2 text-blue-400 font-semibold",children:[m.jsx(di,{size:20}),m.jsx("span",{children:"Development Mode"})]}),m.jsx("p",{className:"text-xs text-slate-300",children:"Supabase is not configured yet. Authentication features are disabled, but you can still access the dashboard!"}),m.jsx("button",{onClick:()=>e("/dashboard"),className:"mt-2 text-xs font-bold text-white bg-blue-600/20 hover:bg-blue-600/40 p-2 rounded-lg transition-all flex items-center justify-center gap-1 border border-blue-500/30",children:"Skip to Dashboard →"})]}),i&&m.jsxs("div",{className:"mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg flex items-center gap-2 text-red-400",children:[m.jsx(di,{size:20}),m.jsx("span",{className:"text-sm",children:i})]}),m.jsxs("form",{onSubmit:w,className:"space-y-6",children:[m.jsxs("div",{children:[m.jsx("label",{className:"block text-xs font-bold text-slate-400 mb-2 uppercase tracking-wider",children:"Username"}),m.jsx("input",{type:"text",value:r.username,onChange:_=>s({...r,username:_.target.value}),className:"w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all text-white placeholder-slate-600",placeholder:"username",disabled:u})]}),m.jsxs("div",{children:[m.jsx("label",{className:"block text-xs font-bold text-slate-400 mb-2 uppercase tracking-wider",children:"Password"}),m.jsxs("div",{className:"relative",children:[m.jsx("input",{type:f?"text":"password",value:r.password,onChange:_=>s({...r,password:_.target.value}),className:"w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all text-white text-2xl tracking-widest placeholder-slate-600 pr-12",placeholder:"••••••••",disabled:u}),m.jsx("button",{type:"button",onClick:()=>E(!f),className:"absolute right-4 top-1/2 -translate-y-1/2 text-slate-400 hover:text-white transition-colors",children:f?m.jsx(hy,{size:20}):m.jsx(fi,{size:20})})]})]}),m.jsx("button",{type:"submit",disabled:u,className:"w-full bg-white text-[#030303] hover:bg-slate-200 font-bold py-4 rounded-xl transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg hover:shadow-white/20",children:u?"Entering...":"Sign In"})]}),m.jsx("div",{className:"mt-8 text-center",children:m.jsxs(xi,{to:"/signup",className:"text-slate-400 hover:text-white transition-colors text-sm font-medium",children:["Need an account? ",m.jsx("span",{className:"text-purple-400 font-bold underline underline-offset-4",children:"Create one"})]})})]})]})},b2=()=>{const e=Qr(),{signUp:t}=ci(),[n,r]=N.useState({fullName:"",username:"",email:"",phoneNumber:"",dateOfBirth:"",password:"",userType:""}),[s,i]=N.useState(""),[a,u]=N.useState(!1),[c,f]=N.useState(!1),[E,w]=N.useState(!1),_=j=>{r({...n,[j.target.name]:j.target.value})},A=()=>!n.fullName||!n.username||!n.email||!n.password||!n.userType?(i("Please fill in all required fields"),!1):/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(n.email)?n.password.length<6?(i("Password must be at least 6 characters long"),!1):n.username.length<3?(i("Username must be at least 3 characters long"),!1):!0:(i("Please enter a valid email address"),!1),R=async j=>{if(j.preventDefault(),i(""),u(!0),!A()){u(!1);return}try{const D={username:n.username,full_name:n.fullName,phone_number:n.phoneNumber,date_of_birth:n.dateOfBirth,user_type:n.userType},{data:b,error:x}=await t(n.email,n.password,D);if(x){x.message.includes("already registered")?i("This email is already registered"):x.message.includes("duplicate key")?i("This username is already taken"):i(x.message||"Failed to create account"),u(!1);return}b&&(f(!0),setTimeout(()=>{e("/login")},2e3))}catch{i("An unexpected error occurred. Please try again."),u(!1)}};return m.jsxs("div",{className:"relative min-h-screen flex items-center justify-center overflow-hidden font-sans py-12 px-4",children:[m.jsxs("div",{className:"bg-streaks",children:[m.jsx("div",{className:"streak-1"}),m.jsx("div",{className:"streak-2"})]}),m.jsxs("div",{className:"glass-dark p-10 rounded-2xl shadow-2xl w-full max-w-md border border-white/10 relative z-10 backdrop-blur-3xl",children:[m.jsxs("div",{className:"flex flex-col items-center mb-10",children:[m.jsx("div",{className:"text-white mb-3 p-3 glass rounded-2xl neon-border-magenta",children:m.jsx(Qc,{size:40,className:"text-accent-magenta"})}),m.jsx("h1",{className:"text-4xl font-bold text-white tracking-tight",children:"CodeSandbox"}),m.jsx("p",{className:"text-slate-400 mt-2 text-sm uppercase tracking-widest font-semibold",children:"Join the scenario"})]}),!ci().isConfigured&&m.jsxs("div",{className:"mb-6 p-4 glass rounded-xl flex flex-col gap-2 border-blue-500/30",children:[m.jsxs("div",{className:"flex items-center gap-2 text-blue-400 font-semibold",children:[m.jsx(di,{size:20}),m.jsx("span",{children:"Development Mode"})]}),m.jsx("p",{className:"text-xs text-slate-300",children:"Supabase is not configured yet. Authentication is disabled, but you can explore!"}),m.jsx("button",{onClick:()=>e("/dashboard"),className:"mt-2 text-xs font-bold text-white bg-blue-600/20 hover:bg-blue-600/40 p-2 rounded-lg transition-all flex items-center justify-center gap-1 border border-blue-500/30",children:"Skip to Dashboard →"})]}),s&&m.jsxs("div",{className:"mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg flex items-center gap-2 text-red-400",children:[m.jsx(di,{size:20}),m.jsx("span",{className:"text-sm",children:s})]}),c&&m.jsxs("div",{className:"mb-4 p-3 bg-green-500/10 border border-green-500/30 rounded-lg flex items-center gap-2 text-green-400",children:[m.jsx(Yc,{size:20}),m.jsx("span",{className:"text-sm",children:"Account created! Redirecting..."})]}),m.jsxs("form",{onSubmit:R,className:"space-y-4",children:[m.jsxs("div",{children:[m.jsxs("label",{className:"block text-xs font-bold text-slate-400 mb-1 uppercase tracking-wider",children:["Full Name ",m.jsx("span",{className:"text-accent-magenta",children:"*"})]}),m.jsx("input",{type:"text",name:"fullName",value:n.fullName,onChange:_,className:"w-full px-4 py-2 bg-white/5 border border-white/10 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all text-white placeholder-slate-600",placeholder:"Your name",disabled:a||c})]}),m.jsxs("div",{children:[m.jsxs("label",{className:"block text-xs font-bold text-slate-400 mb-1 uppercase tracking-wider",children:["Username ",m.jsx("span",{className:"text-magenta-500",children:"*"})]}),m.jsx("input",{type:"text",name:"username",value:n.username,onChange:_,className:"w-full px-4 py-2 bg-white/5 border border-white/10 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all text-white placeholder-slate-600",placeholder:"username",disabled:a||c})]}),m.jsxs("div",{children:[m.jsxs("label",{className:"block text-xs font-bold text-slate-400 mb-1 uppercase tracking-wider",children:["Email ",m.jsx("span",{className:"text-magenta-500",children:"*"})]}),m.jsx("input",{type:"email",name:"email",value:n.email,onChange:_,className:"w-full px-4 py-2 bg-white/5 border border-white/10 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all text-white placeholder-slate-600",placeholder:"email@example.com",disabled:a||c})]}),m.jsxs("div",{className:"grid grid-cols-2 gap-4",children:[m.jsxs("div",{children:[m.jsx("label",{className:"block text-xs font-bold text-slate-400 mb-1 uppercase tracking-wider",children:"Phone"}),m.jsx("input",{type:"tel",name:"phoneNumber",value:n.phoneNumber,onChange:_,className:"w-full px-4 py-2 bg-white/5 border border-white/10 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all text-white placeholder-slate-600",placeholder:"+123...",disabled:a||c})]}),m.jsxs("div",{children:[m.jsx("label",{className:"block text-xs font-bold text-slate-400 mb-1 uppercase tracking-wider",children:"DOB"}),m.jsx("input",{type:"date",name:"dateOfBirth",value:n.dateOfBirth,onChange:_,className:"w-full px-4 py-2 bg-white/5 border border-white/10 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all text-white placeholder-slate-600",disabled:a||c})]})]}),m.jsxs("div",{children:[m.jsxs("label",{className:"block text-xs font-bold text-slate-400 mb-1 uppercase tracking-wider",children:["Password ",m.jsx("span",{className:"text-magenta-500",children:"*"})]}),m.jsxs("div",{className:"relative",children:[m.jsx("input",{type:E?"text":"password",name:"password",value:n.password,onChange:_,className:"w-full px-4 py-2 bg-white/5 border border-white/10 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all text-white text-xl tracking-widest placeholder-slate-600 pr-12",placeholder:"••••••••",disabled:a||c}),m.jsx("button",{type:"button",onClick:()=>w(!E),className:"absolute right-4 top-1/2 -translate-y-1/2 text-slate-400 hover:text-white transition-colors",children:E?m.jsx(hy,{size:20}):m.jsx(fi,{size:20})})]})]}),m.jsxs("div",{children:[m.jsxs("label",{className:"block text-xs font-bold text-slate-400 mb-1 uppercase tracking-wider",children:["User Type ",m.jsx("span",{className:"text-magenta-500",children:"*"})]}),m.jsxs("select",{name:"userType",value:n.userType,onChange:_,className:"w-full px-4 py-2 bg-white/5 border border-white/10 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all text-white bg-[#030303]",disabled:a||c,children:[m.jsx("option",{value:"",children:"Select type"}),m.jsx("option",{value:"Developer",children:"Developer"}),m.jsx("option",{value:"Student",children:"Student"})]})]}),m.jsx("button",{type:"submit",disabled:a||c,className:"w-full bg-white text-[#030303] hover:bg-slate-200 font-bold py-4 rounded-xl transition-all duration-300 mt-4 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg hover:shadow-white/20",children:a?"Creating Account...":c?"Success!":"Join Now"})]}),m.jsx("div",{className:"mt-8 text-center",children:m.jsxs(xi,{to:"/login",className:"text-slate-400 hover:text-white transition-colors text-sm font-medium",children:["Already have an account? ",m.jsx("span",{className:"text-purple-400 font-bold underline underline-offset-4",children:"Sign In"})]})})]})]})},Jn=[{id:"ecommerce-tax",title:"E-Commerce Tax Engine",description:"Fix the tax calculation logic in a multi-file shopping cart system. The current implementation only returns the subtotal.",difficulty:"Medium",badgeColor:"bg-yellow-500/10 text-yellow-500 border border-yellow-500/20",files:{"index.js":{name:"index.js",language:"javascript",content:`import { ShoppingCart } from './cart.js';
const cart = new ShoppingCart(0.1); // 10% tax
cart.addItem({ price: 100, quantity: 1 });
const total = cart.calculateTotal();
console.log('RESULT_TOTAL:' + total);`},"cart.js":{name:"cart.js",language:"javascript",content:`export class ShoppingCart {
  constructor(taxRate = 0.08) {
    this.items = [];
    this.taxRate = taxRate;
  }

  addItem(item) {
    this.items.push(item);
  }

  // BUG: This only returns subtotal, ignoring taxRate
  calculateTotal() {
    const subtotal = this.items.reduce((sum, item) => {
      return sum + (item.price * item.quantity);
    }, 0);
    
    return subtotal;
  }
}`},"constants.js":{name:"constants.js",language:"javascript",content:`export const DEFAULT_TAX_RATE = 0.08;
export const MAX_ITEMS = 100;`},"README.md":{name:"README.md",language:"markdown",content:"# Tax Calculation Scenario\n\nUpdate `cart.js` to include the tax rate in the final total. Formula: subtotal * (1 + taxRate)."}},constraints:["Final total must include tax.","Ensure subtotal is calculated first."],testCheck:`
            const cartCode = files['cart.js']?.content || '';
            const hasFormula = cartCode.includes('taxRate') && (cartCode.includes('*') || cartCode.includes('+'));
            const runtimePassed = stdout.includes('RESULT_TOTAL:110'); // Accept 110 or 110.000...
            return hasFormula && runtimePassed;
        `},{id:"auth-session",title:"Secure Session Management",description:"Build a session validation layer using separate logic and constant modules. Prevent unauthorized access for expired tokens.",difficulty:"Hard",badgeColor:"bg-red-500/10 text-red-500 border border-red-500/20",files:{"index.js":{name:"index.js",language:"javascript",content:`import { isSessionActive } from './sessionManager.js';
import { SESSION_TIMEOUT } from './config.js';

const now = Date.now();
const validSession = { createdAt: now - 1000 };
const expiredSession = { createdAt: now - SESSION_TIMEOUT - 1000 };

console.log('TEST_VALID:' + isSessionActive(validSession));
console.log('TEST_EXPIRED:' + isSessionActive(expiredSession));`},"sessionManager.js":{name:"sessionManager.js",language:"javascript",content:`import { SESSION_TIMEOUT } from './config.js';

export function isSessionActive(session) {
  if (!session) return false;
  // TODO: Check if session.createdAt + SESSION_TIMEOUT > Date.now()
  return true;
}`},"config.js":{name:"config.js",language:"javascript",content:"export const SESSION_TIMEOUT = 3600000; // 1 hour in ms"},"README.md":{name:"README.md",language:"markdown",content:"# Session Security\n\nImplement the expiration check in `sessionManager.js` using the timeout from `config.js`."}},constraints:["Verify session existence.","Check timestamp against current time."],testCheck:`
            const sessionMgrCode = files['sessionManager.js']?.content || '';
            const hasLogic = sessionMgrCode.includes('SESSION_TIMEOUT') && sessionMgrCode.includes('Date.now()');
            const noTodo = !sessionMgrCode.includes('TODO');
            const runtimePassed = stdout.includes('TEST_VALID:true') && stdout.includes('TEST_EXPIRED:false');
            return hasLogic && noTodo && runtimePassed;
        `},{id:"inventory-alerts",title:"Intelligent Stock Alerts",description:"Refactor the notification system to use a data-driven threshold model instead of hardcoded values.",difficulty:"Easy",badgeColor:"bg-green-500/10 text-green-500 border border-green-500/20",files:{"notifier.js":{name:"notifier.js",language:"javascript",content:`import { THRESHOLDS } from './data.js';

export function shouldAlert(itemType, quantity) {
  // TODO: Get threshold from THRESHOLDS map and compare
  return false;
}`},"data.js":{name:"data.js",language:"javascript",content:`export const THRESHOLDS = {
  electronics: 5,
  clothing: 10,
  food: 20
};`},"README.md":{name:"README.md",language:"markdown",content:`# Inventory Alerts

Connect the notifier to the data module to check specific category thresholds.`}},constraints:["Use THRESHOLDS object.","Return true if quantity <= threshold."],testCheck:e=>e.includes("THRESHOLDS[")||e.includes("THRESHOLDS.")},{id:"api-error-boundary",title:"Graceful API Degrader",description:"Implement a resilience layer that falls back to localized mock data when the primary API client fails.",difficulty:"Hard",badgeColor:"bg-red-500/10 text-red-500 border border-red-500/20",files:{"apiClient.js":{name:"apiClient.js",language:"javascript",content:`import { MOCK_DATA } from './fallback.js';

export async function getSafeData(url) {
  try {
    const res = await fetch(url);
    return await res.json();
  } catch (err) {
    // TODO: Return MOCK_DATA instead of throwing
    throw err;
  }
}`},"fallback.js":{name:"fallback.js",language:"javascript",content:'export const MOCK_DATA = { status: "cached", items: [] };'}},constraints:["Use try-catch block.","Import and return fallback data."],testCheck:e=>e.includes("catch")&&e.includes("MOCK_DATA")},{id:"theme-orchestrator",title:"Multi-Context Theme Manager",description:"Coordinate system preferences and user overrides across a stylesheet and a logic controller.",difficulty:"Medium",badgeColor:"bg-yellow-500/10 text-yellow-500 border border-yellow-500/20",files:{"themeController.js":{name:"themeController.js",language:"javascript",content:`import { THEMES } from './palette.js';

export function getThemeClass(isDark) {
  // TODO: Return THEMES.dark if isDark is true, else THEMES.light
  return "";
}`},"palette.js":{name:"palette.js",language:"javascript",content:`export const THEMES = {
  dark: "bg-slate-900 text-white",
  light: "bg-white text-slate-900"
};`}},constraints:["Use external palette mapping.","Support conditional toggle."],testCheck:e=>e.includes("THEMES.dark")&&e.includes("THEMES.light")},{id:"data-transformer-pro",title:"Schema-Driven Transformer",description:"Map raw external data into a validated internal schema using shared transformation rules.",difficulty:"Medium",badgeColor:"bg-yellow-500/10 text-yellow-500 border border-yellow-500/20",files:{"transformer.js":{name:"transformer.js",language:"javascript",content:`import { FIELD_MAP } from './schema.js';

export function transform(rawData) {
  return rawData.map(item => {
    // TODO: Rename keys based on FIELD_MAP
    return item;
  });
}`},"schema.js":{name:"schema.js",language:"javascript",content:`export const FIELD_MAP = {
  external_id: "id",
  user_name: "name",
  email_address: "email"
};`}},constraints:["Iterate through mappings.","Do not mutate original objects."],testCheck:e=>e.includes("FIELD_MAP")&&e.includes(".map")},{id:"rbac-gatekeeper",title:"Distributed RBAC Guard",description:"Check nested permissions across a multi-role architecture to protect specific system actions.",difficulty:"Hard",badgeColor:"bg-red-500/10 text-red-500 border border-red-500/20",files:{"guard.js":{name:"guard.js",language:"javascript",content:`import { PERMISSIONS } from './roles.js';

export function hasAccess(role, action) {
  const allowed = PERMISSIONS[role] || [];
  // TODO: Check if action exists in allowed array
  return false;
}`},"roles.js":{name:"roles.js",language:"javascript",content:`export const PERMISSIONS = {
  admin: ["create", "edit", "delete", "view"],
  editor: ["edit", "view"],
  guest: ["view"]
};`}},constraints:["Support dynamic role lookup.","Handle undefined roles gracefully."],testCheck:e=>e.includes("PERMISSIONS[")&&e.includes(".includes(")},{id:"search-efficiency",title:"Multi-File Debouncer",description:"Implement a high-performance debouncer for search triggers that interacts with an external API module.",difficulty:"Hard",badgeColor:"bg-red-500/10 text-red-500 border border-red-500/20",files:{"debounce.js":{name:"debounce.js",language:"javascript",content:`export function createDebouncer(fn, delay) {
  let timeoutId;
  return (...args) => {
    // TODO: Clear previous timeout and set new one
  };
}`},"searchUI.js":{name:"searchUI.js",language:"javascript",content:`import { createDebouncer } from './debounce.js';

// Usage example for tests
export const debouncedSearch = createDebouncer(() => console.log("Searching..."), 300);`}},constraints:["Use closures for state.","Handle rapid event firing."],testCheck:e=>e.includes("clearTimeout")&&e.includes("setTimeout")},{id:"finance-wizard",title:"Compound Interest Studio",description:"Build a financial calculation module that uses separate interest rate and compounding factor configurations.",difficulty:"Medium",badgeColor:"bg-yellow-500/10 text-yellow-500 border border-yellow-500/20",files:{"calc.js":{name:"calc.js",language:"javascript",content:`import { FACTORS } from './settings.js';

export function calculateGrowth(p, r, t, frequency) {
  const n = FACTORS[frequency] || 1;
  // Formula: A = P(1 + r/n)^(nt)
  return p * (1 + r);
}`},"settings.js":{name:"settings.js",language:"javascript",content:`export const FACTORS = {
  monthly: 12,
  quarterly: 4,
  annually: 1
};`}},constraints:["Support multiple compounding frequencies.","Correct math implementation."],testCheck:e=>e.includes("Math.pow")||e.includes("**")},{id:"seo-slug-generator",title:"SEO Slug Workflows",description:"Cleanse and transform website titles into production-ready URLs using a regex-based cleansing module.",difficulty:"Easy",badgeColor:"bg-green-500/10 text-green-500 border border-green-500/20",files:{"slugger.js":{name:"slugger.js",language:"javascript",content:`import { CLEAN_REGEX } from './patterns.js';

export function toSlug(text) {
  // TODO: Lowercase, replace non-alphanumeric (using CLEAN_REGEX) with '-', and trim
  return text;
}`},"patterns.js":{name:"patterns.js",language:"javascript",content:"export const CLEAN_REGEX = /[^a-z0-9]+/g;"}},constraints:["Use regex for replacement.","Clean start/end dashes."],testCheck:e=>e.includes("CLEAN_REGEX")&&e.includes("replace")},{id:"inventory-processor",title:"Order Status Workflow",description:"Manage complex order state transitions across a dedicated processor and a status mapping file.",difficulty:"Medium",badgeColor:"bg-yellow-500/10 text-yellow-500 border border-yellow-500/20",files:{"orderProcessor.js":{name:"orderProcessor.js",language:"javascript",content:`import { STATES } from './statusCodes.js';

export function advanceOrder(currentStatus) {
  // TODO: Return next available state from STATES workflow
  return currentStatus;
}`},"statusCodes.js":{name:"statusCodes.js",language:"javascript",content:`export const STATES = [
  "PENDING",
  "PROCESSING",
  "SHIPPED",
  "DELIVERED"
];`}},constraints:["Find index of current state.",'Handle "DELIVERED" boundary.'],testCheck:e=>e.includes("STATES.indexOf")||e.includes("STATES[")},{id:"file-utility-kit",title:"Readable Size Formatter",description:"Coordinate byte-to-string conversions using a unit-map file to support KB, MB, and GB precision.",difficulty:"Medium",badgeColor:"bg-yellow-500/10 text-yellow-500 border border-yellow-500/20",files:{"formatter.js":{name:"formatter.js",language:"javascript",content:`import { UNITS } from './units.js';

export function formatSize(bytes) {
  // TODO: Iterate units and divide by 1024 until correct tier found
  return bytes + " " + UNITS[0];
}`},"units.js":{name:"units.js",language:"javascript",content:'export const UNITS = ["B", "KB", "MB", "GB", "TB"];'}},constraints:["Loop through units.","Limit to 2 decimal places."],testCheck:e=>e.includes("1024")&&e.includes("UNITS[")},{id:"config-merger-v2",title:"Recursive Config Union",description:"Merge deep JSON configurations from multiple sources into a single unified application state.",difficulty:"Hard",badgeColor:"bg-red-500/10 text-red-500 border border-red-500/20",files:{"merger.js":{name:"merger.js",language:"javascript",content:`export function deepMerge(target, source) {
  // TODO: Recursively merge keys if both are objects
  return { ...target, ...source }; 
}`},"defaults.js":{name:"defaults.js",language:"javascript",content:`export const APP_DEFAULTS = {
  api: { timeout: 1000, retry: true },
  ui: { theme: "dark" }
};`}},constraints:["Handle nested objects.","Avoid overwriting entire branches."],testCheck:e=>e.includes("typeof")&&e.includes("deepMerge")},{id:"color-parser-studio",title:"Hex to RGB Converter",description:"Implement a parsing workflow that converts HEX strings into RGB objects using bitwise or string-slice logic.",difficulty:"Medium",badgeColor:"bg-yellow-500/10 text-yellow-500 border border-yellow-500/20",files:{"parser.js":{name:"parser.js",language:"javascript",content:`export function hexToRgb(hex) {
  // TODO: Extract R, G, B components from #RRGGBB
  return { r: 0, g: 0, b: 0 };
}`},"README.md":{name:"README.md",language:"markdown",content:`# Color Logic

Convert a 6-digit hex string into its numeric RGB components.`}},constraints:["Handle optional # prefix.","Return numeric values."],testCheck:e=>e.includes("parseInt")&&e.includes("16")},{id:"security-analyzer",title:"Password Pulse Gauge",description:"Create a security analyzer that scores passwords based on complex patterns defined in a shared policy file.",difficulty:"Medium",badgeColor:"bg-yellow-500/10 text-yellow-500 border border-yellow-500/20",files:{"analyzer.js":{name:"analyzer.js",language:"javascript",content:`import { POLICY } from './policy.js';

export function checkSecurity(pass) {
  let score = 0;
  // TODO: Check against each POLICY regex
  return score;
}`},"policy.js":{name:"policy.js",language:"javascript",content:`export const POLICY = [
  { name: "length", regex: /.{8,}/ },
  { name: "digit", regex: /[0-9]/ },
  { name: "special", regex: /[^a-zA-Z0-9]/ }
];`}},constraints:["Iterate policy array.","Increment score for each match."],testCheck:e=>e.includes("POLICY.forEach")||e.includes("POLICY.reduce")},{id:"data-dedupe-engine",title:"Unique Entity Resolver",description:"Deduplicate complex data lists based on a unique identifier key specified in a settings module.",difficulty:"Medium",badgeColor:"bg-yellow-500/10 text-yellow-500 border border-yellow-500/20",files:{"deduper.js":{name:"deduper.js",language:"javascript",content:`export function removeDuplicates(list, uniqueId) {
  // TODO: Filter list to keep only first occurrence of uniqueId
  return list;
}`},"testData.js":{name:"testData.js",language:"javascript",content:`export const SAMPLE_USERS = [
  { id: 1, name: "A" },
  { id: 1, name: "A" },
  { id: 2, name: "B" }
];`}},constraints:["Maintain original order.","Efficient handling."],testCheck:e=>e.includes("Map")||e.includes("Set")||e.includes("filter")},{id:"time-relative-pro",title:"Social Media Timestamps",description:'Convert timestamps into "ago" strings using a multi-threshold interval mapping system.',difficulty:"Hard",badgeColor:"bg-red-500/10 text-red-500 border border-red-500/20",files:{"timeAgo.js":{name:"timeAgo.js",language:"javascript",content:`import { INTERVALS } from './thresholds.js';

export function getTimeAgo(date) {
  const seconds = Math.floor((Date.now() - date) / 1000);
  // TODO: Find correct interval and return it
  return "just now";
}`},"thresholds.js":{name:"thresholds.js",language:"javascript",content:`export const INTERVALS = [
  { label: "year", seconds: 31536000 },
  { label: "month", seconds: 2592000 },
  { label: "day", seconds: 86400 },
  { label: "hour", seconds: 3600 },
  { label: "minute", seconds: 60 }
];`}},constraints:["Iterate through intervals.","Return plural variants if count > 1."],testCheck:e=>e.includes("INTERVALS.find")||e.includes("INTERVALS.forEach")},{id:"form-validation-kit",title:"Cross-Field Form Validator",description:"Implement a validation engine that checks correlations between multiple fields (e.g., password and confirmation).",difficulty:"Medium",badgeColor:"bg-yellow-500/10 text-yellow-500 border border-yellow-500/20",files:{"formValidator.js":{name:"formValidator.js",language:"javascript",content:`export function validateFields(fields) {
  // fields = { password, confirmPassword }
  // TODO: Check if they match
  return { valid: true };
}`},"messages.js":{name:"messages.js",language:"javascript",content:`export const ERRORS = {
  mismatch: "Passwords must match",
  tooShort: "Too short"
};`}},constraints:["Use external error messages.","Check field equality."],testCheck:e=>e.includes("ERRORS.")&&e.includes("===")},{id:"url-query-builder",title:"Dynamic URL Assembler",description:"Construct complex URL strings with query parameters using a shared base configuration module.",difficulty:"Easy",badgeColor:"bg-green-500/10 text-green-500 border border-green-500/20",files:{"assembler.js":{name:"assembler.js",language:"javascript",content:`import { BASE_URL } from './config.js';

export function buildUrl(endpoint, params) {
  // TODO: Append params as query string to BASE_URL + endpoint
  return BASE_URL + endpoint;
}`},"config.js":{name:"config.js",language:"javascript",content:'export const BASE_URL = "https://api.sandbox.com/v1";'}},constraints:["Handle multiple parameters.","Proper URL encoding."],testCheck:e=>e.includes("?")&&(e.includes("URLSearchParams")||e.includes("&"))},{id:"localization-manager",title:"Multi-Language Store",description:"Build a translation engine that switches between language bundles based on a locale code.",difficulty:"Medium",badgeColor:"bg-yellow-500/10 text-yellow-500 border border-yellow-500/20",files:{"i18n.js":{name:"i18n.js",language:"javascript",content:`import { BUNDLES } from './locales.js';

export function translate(locale, key) {
  // TODO: Return nested value from BUNDLES[locale]
  return key;
}`},"locales.js":{name:"locales.js",language:"javascript",content:`export const BUNDLES = {
  en: { welcome: "Welcome" },
  es: { welcome: "Bienvenido" }
};`}},constraints:["Support deep nesting.","Fallback if key missing."],testCheck:e=>e.includes("BUNDLES[locale]")||e.includes("BUNDLES.")},{id:"payment-rounding-bug",title:"Payment Precision Bug (Floating Point)",description:"Debug a checkout system where fractional cents are lost during calculations, causing revenue leakage. The bug occurs when multiplying decimal values.",difficulty:"Medium",badgeColor:"bg-yellow-500/10 text-yellow-500 border border-yellow-500/20",files:{"checkout.js":{name:"checkout.js",language:"javascript",content:`import { applyDiscount, TAX_RATE, roundPrice } from './pricing.js';

export function calculateTotal(items) {
  let subtotal = 0;
  
  for (const item of items) {
    subtotal += item.price * item.quantity;
  }
  
  const discounted = applyDiscount(subtotal, 0.15); // 15% discount
  const tax = discounted * TAX_RATE;
  const total = discounted + tax;
  
  console.log('SUBTOTAL:' + subtotal);
  console.log('DISCOUNTED:' + discounted);
  console.log('TAX:' + tax);
  console.log('TOTAL:' + total);
  console.log('FINAL:' + roundPrice(total));
  
  return total;
}`},"pricing.js":{name:"pricing.js",language:"javascript",content:`export const TAX_RATE = 0.0875; // 8.75%

export function applyDiscount(amount, discount) {
  // BUG: Direct multiplication causes floating point errors
  return amount - (amount * discount);
}

export function roundPrice(price) {
  // BUG: Not properly rounding to 2 decimal places
  return Math.round(price * 100) / 100;
}`},"README.md":{name:"README.md",language:"markdown",content:`# Payment Precision Bug

Test: Purchase 3 items at $19.99 each with 15% discount and 8.75% tax.
Expected FINAL: 52.47

The system is losing precision. Fix the rounding logic.`}},constraints:["Use proper decimal math.","Round consistently to 2 places."],testCheck:`
            const pricingCode = files['pricing.js']?.content || '';
            const hasRounding = pricingCode.includes('Math.round') && pricingCode.includes('100');
            const runtimePassed = stdout.includes('FINAL:52.47');
            return hasRounding && runtimePassed;
        `},{id:"database-connection-leak",title:"Connection Pool Memory Leak",description:'Debug a service that fails to close database connections properly. Connections pile up and exhaust the pool, causing "ECONNREFUSED" errors.',difficulty:"Hard",badgeColor:"bg-red-500/10 text-red-500 border border-red-500/20",files:{"userService.js":{name:"userService.js",language:"javascript",content:`import { getConnection } from './database.js';

export async function getUserById(id) {
  const conn = getConnection();
  
  try {
    const user = conn.query('SELECT * FROM users WHERE id = ?', [id]);
    // BUG: Connection is never closed!
    return user;
  } catch (err) {
    console.log('ERROR:Connection failed');
    return null;
  }
}

export async function getUsersWithPosts(userId) {
  const conn = getConnection();
  
  try {
    const user = conn.query('SELECT * FROM users WHERE id = ?', [userId]);
    const posts = conn.query('SELECT * FROM posts WHERE user_id = ?', [userId]);
    // BUG: Early return skips closeConnection call below
    if (!user) return null;
    
    const result = { user, posts };
    // BUG: This line never executes if user is null
    closeConnection(conn);
    return result;
  } catch (err) {
    console.log('ERROR:Query failed');
  }
}`},"database.js":{name:"database.js",language:"javascript",content:`let activeConnections = 0;
const MAX_CONNECTIONS = 5;

export function getConnection() {
  if (activeConnections >= MAX_CONNECTIONS) {
    throw new Error('ECONNREFUSED:Connection pool exhausted');
  }
  activeConnections++;
  console.log('CONN_OPEN:' + activeConnections);
  return { query: () => ({ id: 1, name: 'Test' }) };
}

export function closeConnection(conn) {
  activeConnections--;
  console.log('CONN_CLOSED:' + activeConnections);
}`},"README.md":{name:"README.md",language:"markdown",content:`# Connection Leak Bug

Fix: Ensure all connection lifetimes are properly managed.
- Use try-finally or error handlers
- Close connections even when errors occur
- Never skip cleanup in early returns`}},constraints:["Use try-finally.","Close in all code paths."],testCheck:`
            const serviceCode = files['userService.js']?.content || '';
            const hasFinally = serviceCode.includes('finally');
            const hasClose = serviceCode.includes('closeConnection');
            const runtimePassed = stdout.includes('CONN_CLOSED:0');
            return hasFinally && hasClose && runtimePassed;
        `},{id:"race-condition-counter",title:"Race Condition in Async Counter",description:"Debug an inventory system where concurrent requests increment the counter incorrectly. Multiple async operations read-modify-write without coordination.",difficulty:"Hard",badgeColor:"bg-red-500/10 text-red-500 border border-red-500/20",files:{"inventory.js":{name:"inventory.js",language:"javascript",content:`import { INVENTORY } from './state.js';

export async function decrementStock(productId, quantity) {
  // BUG: Race condition - read then write without atomicity
  const current = INVENTORY[productId];
  
  // Simulate async operation (like DB call)
  await new Promise(resolve => setTimeout(resolve, 10));
  
  if (current < quantity) {
    console.log('FAILED:Insufficient stock');
    return false;
  }
  
  // By this time, another thread may have decremented!
  INVENTORY[productId] = current - quantity;
  console.log('SUCCESS:' + INVENTORY[productId]);
  return true;
}

export function addStock(productId, quantity) {
  // BUG: Same issue here
  const current = INVENTORY[productId];
  INVENTORY[productId] = current + quantity;
  console.log('ADDED:' + INVENTORY[productId]);
}`},"state.js":{name:"state.js",language:"javascript",content:`export const INVENTORY = {
  'PROD_001': 100,
  'PROD_002': 50
};`},"test.js":{name:"test.js",language:"javascript",content:`import { decrementStock, addStock } from './inventory.js';
import { INVENTORY } from './state.js';

(async () => {
  // Start 3 concurrent decrements (should each decrement by 10, final = 70)
  await Promise.all([
    decrementStock('PROD_001', 10),
    decrementStock('PROD_001', 10),
    decrementStock('PROD_001', 10)
  ]);
  
  console.log('FINAL:' + INVENTORY['PROD_001']);
})();`},"README.md":{name:"README.md",language:"markdown",content:`# Race Condition Bug

Fix: Use a lock or queue to ensure only one operation modifies inventory at a time.
Expected FINAL: 70 (100 - 30)
Actual: Will be 90 due to race condition

Solution: Implement a lock mechanism or serial queue.`}},constraints:["Serialize updates.","Use locking or queue pattern."],testCheck:`
            const inventoryCode = files['inventory.js']?.content || '';
            const hasQueue = inventoryCode.includes('queue') || inventoryCode.includes('pending') || inventoryCode.includes('lock');
            const runtimePassed = stdout.includes('FINAL:70');
            return hasQueue && runtimePassed;
        `},{id:"caching-invalidation-bug",title:"Stale Cache Not Invalidated",description:"Debug a user profile system where cached data never expires or updates. Users see outdated information even after changes.",difficulty:"Medium",badgeColor:"bg-yellow-500/10 text-yellow-500 border border-yellow-500/20",files:{"userCache.js":{name:"userCache.js",language:"javascript",content:`import { CACHE_CONFIG } from './config.js';

const cache = {};

export function getUser(userId) {
  if (cache[userId]) {
    console.log('CACHE_HIT:' + userId);
    return cache[userId];
  }
  
  console.log('CACHE_MISS:' + userId);
  const user = { id: userId, name: 'John', email: 'john@example.com', updatedAt: Date.now() };
  
  // BUG: Caching without TTL or invalidation
  cache[userId] = user;
  return user;
}

export function updateUser(userId, updates) {
  // BUG: Modifying user but cache isn't invalidated
  const user = getUser(userId);
  Object.assign(user, updates, { updatedAt: Date.now() });
  
  console.log('UPDATED:' + userId);
  return user;
}

export function clearCache(userId) {
  // BUG: This is never called in updateUser
  delete cache[userId];
  console.log('CLEARED:' + userId);
}`},"config.js":{name:"config.js",language:"javascript",content:`export const CACHE_CONFIG = {
  TTL: 5000 // 5 seconds
};`},"README.md":{name:"README.md",language:"markdown",content:`# Cache Invalidation Bug

Fix: Invalidate cache when data is updated.
- Call clearCache() in updateUser()
- Or implement TTL-based expiration

Expected: After update, next getUser() should show updated data`}},constraints:["Invalidate on update.","Fresh data always returned after mutation."],testCheck:`
            const cacheCode = files['userCache.js']?.content || '';
            const hasInvalidation = cacheCode.includes('clearCache');
            const runtimePassed = stdout.includes('CLEARED:');
            return hasInvalidation && runtimePassed;
        `},{id:"string-escaping-xss",title:"XSS Vulnerability - Unescaped HTML",description:"Debug a comment rendering system that injects malicious scripts. User input isn't sanitized before display.",difficulty:"Hard",badgeColor:"bg-red-500/10 text-red-500 border border-red-500/20",files:{"renderer.js":{name:"renderer.js",language:"javascript",content:`import { escapeHTML } from './sanitizer.js';

export function renderComment(comment) {
  // BUG: Directly inserting user input without escaping
  const html = '<div class="comment">' + comment.text + '</div>';
  console.log('RENDERED:' + html);
  return html;
}

export function renderCommentsSafe(comments) {
  const html = comments.map(c => {
    // TODO: Use escapeHTML here
    return '<p>' + c.text + '</p>';
  }).join('');
  
  console.log('SAFE_COUNT:' + comments.length);
  return html;
}`},"sanitizer.js":{name:"sanitizer.js",language:"javascript",content:`export function escapeHTML(text) {
  const map = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#039;'
  };
  return text.replace(/[&<>"']/g, m => map[m]);
}`},"README.md":{name:"README.md",language:"markdown",content:'# XSS Vulnerability\n\nFix: Use escapeHTML() on all user inputs before rendering.\nMalicious input: `<script>alert("xss")<\/script>`\nShould render as: `&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;`'}},constraints:["Escape all user input.","Use sanitizer module."],testCheck:`
            const rendererCode = files['renderer.js']?.content || '';
            const hasSanitization = rendererCode.includes('escapeHTML');
            const runtimePassed = stdout.includes('&lt;') || stdout.includes('&amp;');
            return hasSanitization && runtimePassed;
        `},{id:"event-listener-memory-leak",title:"Event Listeners Not Cleaned Up",description:"Debug a real-time chat widget where event listeners accumulate on every page load, consuming memory and firing multiple times.",difficulty:"Hard",badgeColor:"bg-red-500/10 text-red-500 border border-red-500/20",files:{"chatWidget.js":{name:"chatWidget.js",language:"javascript",content:`import { createEmitter } from './eventEmitter.js';

const emitter = createEmitter();

export function initChat() {
  // BUG: Listener is added every time init is called, never removed
  emitter.on('message', (msg) => {
    console.log('MESSAGE_RECEIVED:' + msg);
  });
  
  console.log('CHAT_INIT:done');
}

export function sendMessage(text) {
  emitter.emit('message', text);
}

export function cleanup() {
  // BUG: This method exists but is never called
  emitter.removeAllListeners();
  console.log('LISTENERS_CLEARED');
}

export function getListenerCount() {
  return emitter.listenerCount('message');
}`},"eventEmitter.js":{name:"eventEmitter.js",language:"javascript",content:`export function createEmitter() {
  const listeners = {};
  
  return {
    on(event, handler) {
      if (!listeners[event]) listeners[event] = [];
      listeners[event].push(handler);
    },
    
    emit(event, data) {
      if (listeners[event]) {
        listeners[event].forEach(h => h(data));
      }
    },
    
    removeAllListeners() {
      Object.keys(listeners).forEach(e => listeners[e] = []);
    },
    
    listenerCount(event) {
      return listeners[event]?.length || 0;
    }
  };
}`},"README.md":{name:"README.md",language:"markdown",content:`# Event Listener Memory Leak

Fix: Call cleanup() when widget unmounts or add check to prevent duplicate listeners.
Bug: Calling initChat() twice results in 2 listeners for same event.`}},constraints:["Prevent duplicate listeners.","Provide cleanup method."],testCheck:`
            const chatCode = files['chatWidget.js']?.content || '';
            const hasCleanupCall = chatCode.includes('cleanup()') || chatCode.includes('removeAllListeners');
            const runtimePassed = stdout.includes('LISTENERS_CLEARED') || stdout.includes('listenerCount:1');
            return (hasCleanupCall || chatCode.includes('if.*listenerCount')) && runtimePassed;
        `},{id:"null-pointer-chain",title:"Null Pointer Exception - Unsafe Property Access",description:"Debug an API response handler that crashes when optional fields are missing. Deep property access without null checks.",difficulty:"Medium",badgeColor:"bg-yellow-500/10 text-yellow-500 border border-yellow-500/20",files:{"apiHandler.js":{name:"apiHandler.js",language:"javascript",content:`import { extractUserData } from './extractor.js';

export function handleUserResponse(response) {
  try {
    // BUG: Unsafe chaining - response.user might be null
    const name = response.user.profile.name;
    const email = response.user.contact.email;
    const phone = response.user.contact.phone || 'N/A';
    
    console.log('NAME:' + name);
    console.log('EMAIL:' + email);
    console.log('PHONE:' + phone);
    
    return { name, email, phone };
  } catch (err) {
    console.log('ERROR:' + err.message);
    return null;
  }
}

export function processUsers(responses) {
  return responses.map(r => {
    // BUG: No validation of r structure
    const user = extractUserData(r);
    return {
      fullName: user.firstName + ' ' + user.lastName,
      id: user.id
    };
  });
}`},"extractor.js":{name:"extractor.js",language:"javascript",content:`export function extractUserData(response) {
  // BUG: Assumes deep nesting exists without checking
  return {
    firstName: response.profile.personal.firstName,
    lastName: response.profile.personal.lastName,
    id: response.metadata.id,
    email: response.contact.email.primary
  };
}`},"README.md":{name:"README.md",language:"markdown",content:`# Null Pointer / Optional Chaining Bug

Fix: Use optional chaining (?.) or guard clauses.
Test cases:
1. Missing user object
2. Missing nested profile
3. Undefined email`}},constraints:["Use optional chaining or guard clauses.","Handle missing fields gracefully."],testCheck:`
            const code = (files['apiHandler.js']?.content || '') + (files['extractor.js']?.content || '');
            const hasOptionalChaining = code.includes('?.');
            const hasGuards = code.includes('?.') || code.includes('||') || code.includes('&&');
            return hasOptionalChaining || (hasGuards && !stdout.includes('ERROR:Cannot read'));
        `},{id:"sql-injection-danger",title:"SQL Injection - Unsafe Query Building",description:"Debug a user search feature vulnerable to SQL injection. String concatenation is used instead of parameterized queries.",difficulty:"Hard",badgeColor:"bg-red-500/10 text-red-500 border border-red-500/20",files:{"db.js":{name:"db.js",language:"javascript",content:`export function query(sql, params) {
  console.log('QUERY:' + sql);
  // Simulated DB response
  if (sql.includes('DROP') || sql.includes('DELETE')) {
    console.log('DANGER:Malicious query detected!');
    return [];
  }
  return { id: 1, name: 'Test' };
}`},"userRepo.js":{name:"userRepo.js",language:"javascript",content:`import { query } from './db.js';

export function findByEmail(email) {
  // BUG: SQL Injection vulnerability!
  const sql = 'SELECT * FROM users WHERE email = '' + email + ''';
  return query(sql, []);
}

export function searchUsers(searchTerm) {
  // BUG: Unsafe string concatenation
  const sql = 'SELECT * FROM users WHERE name LIKE '%' + searchTerm + '%'';
  return query(sql, []);
}

export function findByEmailSafe(email) {
  // TODO: Use parameterized query
  const sql = 'SELECT * FROM users WHERE email = '' + email + ''';
  return query(sql, []);
}`},"README.md":{name:"README.md",language:"markdown",content:"# SQL Injection Bug\n\nFix: Use parameterized queries instead of string concatenation.\nMalicious payload: `' OR '1'='1` \nSecure: `query('SELECT * FROM users WHERE email = ?', [email])`"}},constraints:["Use parameterized queries.","Never concatenate user input into SQL."],testCheck:`
            const repoCode = files['userRepo.js']?.content || '';
            const hasParams = repoCode.includes('?') && repoCode.match(/query.*\\[.*\\]/);
            const noConcat = !repoCode.includes('\\'' + ' ');
            return hasParams && noConcat;
        `},{id:"off-by-one-pagination",title:"Off-by-One Error in Pagination",description:"Debug a data listing system where pagination boundaries are wrong. Page 2 shows duplicate or missing items.",difficulty:"Medium",badgeColor:"bg-yellow-500/10 text-yellow-500 border border-yellow-500/20",files:{"paginator.js":{name:"paginator.js",language:"javascript",content:`import { ITEMS } from './data.js';
import { PAGE_SIZE } from './config.js';

export function getPage(pageNumber) {
  // BUG: Off-by-one error in calculation
  const start = (pageNumber - 1) * PAGE_SIZE;
  const end = pageNumber * PAGE_SIZE;
  
  console.log('PAGE:' + pageNumber);
  console.log('START:' + start);
  console.log('END:' + end);
  
  const items = ITEMS.slice(start, end);
  console.log('COUNT:' + items.length);
  
  return items;
}

export function getTotalPages() {
  // BUG: Not rounding up correctly
  return ITEMS.length / PAGE_SIZE;
}

export function validatePageNumber(page, total) {
  // BUG: Off-by-one - should be <= total not < total
  if (page < 1 || page > total) {
    console.log('ERROR:Invalid page');
    return false;
  }
  return true;
}`},"data.js":{name:"data.js",language:"javascript",content:`export const ITEMS = Array.from({ length: 25 }, (_, i) => ({
  id: i + 1,
  name: 'Item ' + (i + 1)
}));`},"config.js":{name:"config.js",language:"javascript",content:"export const PAGE_SIZE = 10;"},"README.md":{name:"README.md",language:"markdown",content:`# Pagination Bug

25 items, 10 per page = 3 pages
Page 1: items 1-10
Page 2: items 11-20
Page 3: items 21-25

Fix off-by-one errors in validation and ceiling logic.`}},constraints:["Correct math for pagination.","Use Math.ceil for total pages."],testCheck:`
            const paginatorCode = files['paginator.js']?.content || '';
            const hasCeil = paginatorCode.includes('Math.ceil');
            const correctBounds = paginatorCode.includes('<= total') || paginatorCode.includes('Math.ceil');
            return hasCeil && correctBounds;
        `},{id:"async-timing-bug",title:"Async/Promise Timing Issue",description:"Debug a multi-step workflow where data is used before the async operation completes. Wrong promise chaining or missing await.",difficulty:"Hard",badgeColor:"bg-red-500/10 text-red-500 border border-red-500/20",files:{"workflow.js":{name:"workflow.js",language:"javascript",content:`import { fetchUserData, fetchUserPosts } from './api.js';

export async function getUserWithPosts(userId) {
  // BUG: Missing await - returns before data is fetched
  const user = fetchUserData(userId);
  const posts = fetchUserPosts(userId);
  
  console.log('USER_NAME:' + user.name); // undefined!
  console.log('POSTS_COUNT:' + posts.length); // undefined!
  
  return { user, posts };
}

export async function processUserWorkflow(userId) {
  // BUG: Not chaining promises correctly
  let user;
  fetchUserData(userId).then(u => {
    user = u;
  });
  
  console.log('STEP_1_COMPLETE');
  console.log('USER:' + user); // Still undefined
  
  // BUG: This executes too early
  const processedName = user.name.toUpperCase();
  console.log('PROCESSED:' + processedName);
}`},"api.js":{name:"api.js",language:"javascript",content:`export async function fetchUserData(userId) {
  await new Promise(r => setTimeout(r, 100));
  return { id: userId, name: 'Alice', email: 'alice@example.com' };
}

export async function fetchUserPosts(userId) {
  await new Promise(r => setTimeout(r, 100));
  return [{ id: 1, title: 'Post 1' }, { id: 2, title: 'Post 2' }];
}`},"README.md":{name:"README.md",language:"markdown",content:`# Async Timing Bug

Fix: Properly await async operations before using their results.
- Use await in async functions
- Don't forget Promise.all for parallel operations

Expected: USER_NAME:Alice, POSTS_COUNT:2`}},constraints:["Use proper await syntax.","Chain or parallelize promises correctly."],testCheck:`
            const workflowCode = files['workflow.js']?.content || '';
            const hasAwait = workflowCode.includes('await');
            const runtimePassed = stdout.includes('USER_NAME:Alice') && stdout.includes('POSTS_COUNT:2');
            return hasAwait && runtimePassed;
        `},{id:"case-sensitivity-bug",title:"Case Sensitivity Bug - String Comparison",description:`Debug a role-based access system where roles aren't compared case-sensitively. "Admin" and "admin" are treated as different.`,difficulty:"Easy",badgeColor:"bg-green-500/10 text-green-500 border border-green-500/20",files:{"auth.js":{name:"auth.js",language:"javascript",content:`import { ADMIN_ROLE, validateRole } from './roles.js';

export function checkAdminAccess(userRole) {
  // BUG: Case-sensitive comparison
  if (userRole === ADMIN_ROLE) {
    console.log('ACCESS:granted');
    return true;
  }
  
  console.log('ACCESS:denied');
  return false;
}

export function getUserPermissions(role) {
  // BUG: Role lookup is case-sensitive
  const perms = validateRole(role);
  console.log('PERMISSIONS:' + perms.length);
  return perms;
}`},"roles.js":{name:"roles.js",language:"javascript",content:`export const ADMIN_ROLE = 'admin';
export const USER_ROLE = 'user';

export function validateRole(role) {
  // BUG: Case sensitive check
  const validRoles = {
    admin: ['read', 'write', 'delete'],
    user: ['read']
  };
  
  return validRoles[role] || [];
}`},"README.md":{name:"README.md",language:"markdown",content:`# Case Sensitivity Bug

Fix: Normalize strings to lowercase before comparison.
Test: User role = "ADMIN" should get full access
Problem: Direct equality fails`}},constraints:["Normalize to lowercase.","Ensure case-insensitive comparison."],testCheck:`
            const authCode = files['auth.js']?.content || '';
            const rolesCode = files['roles.js']?.content || '';
            const hasLowerCase = authCode.includes('toLowerCase()') || rolesCode.includes('toLowerCase()');
            const runtimePassed = stdout.includes('ACCESS:granted');
            return hasLowerCase && runtimePassed;
        `},{id:"type-coercion-surprise",title:"Type Coercion Bug - Loose Equality",description:`Debug a validator that accepts invalid input due to JavaScript's loose equality. "0" == 0 == false causes logic errors.`,difficulty:"Medium",badgeColor:"bg-yellow-500/10 text-yellow-500 border border-yellow-500/20",files:{"validator.js":{name:"validator.js",language:"javascript",content:`import { isValidScore, isValidStatus } from './checks.js';

export function validateProduct(product) {
  // BUG: Loose equality allows "0" to be treated as 0
  if (product.price == 0) {
    console.log('INVALID:Price cannot be 0');
    return false;
  }
  
  // BUG: isNaN might coerce string "123" to 123
  if (!isValidScore(product.rating)) {
    console.log('INVALID:Bad rating');
    return false;
  }
  
  console.log('VALID:Product accepted');
  return true;
}`},"checks.js":{name:"checks.js",language:"javascript",content:`export function isValidScore(score) {
  // BUG: This accepts strings that coerce to numbers
  return score >= 1 && score <= 5;
}

export function isValidStatus(status) {
  // BUG: Using == instead of ===
  if (status == 'active') {
    return true;
  }
  return false;
}`},"README.md":{name:"README.md",language:"markdown",content:`# Type Coercion Bug

Fix: Use strict equality (===) instead of loose (==).
Test case: price = "0" should be invalid
"0" == false is true in JS but should be rejected`}},constraints:["Use strict equality (===).","Proper type checking."],testCheck:`
            const validatorCode = files['validator.js']?.content || '';
            const checksCode = files['checks.js']?.content || '';
            const strict = validatorCode.includes('===') && checksCode.includes('===');
            const runtimePassed = stdout.includes('INVALID:Price cannot be 0');
            return strict && runtimePassed;
        `},{id:"ecommerce-order-system",title:"E-Commerce Order Processing System",description:"Fix a multi-tier order processing pipeline with bugs in inventory, payment, and shipping coordination. The system processes orders but fails to update inventory correctly, duplicate charges occur, and order status is inconsistent.",difficulty:"Hard",badgeColor:"bg-red-500/10 text-red-500 border border-red-500/20",files:{"orderService.js":{name:"orderService.js",language:"javascript",content:`import { inventoryManager } from './inventory.js';
import { paymentProcessor } from './payment.js';
import { notificationService } from './notifications.js';
import { generateOrderId } from './utils.js';

const orders = {}; // In-memory order store

export const orderService = {
  async createOrder(userId, items, shippingAddress) {
    const orderId = generateOrderId();
    
    // BUG 1: No validation of items or inventory check before payment
    const orderTotal = items.reduce((sum, item) => sum + (item.price * item.quantity), 0);
    
    // BUG 2: Payment taken before inventory is reserved
    const paymentResult = await paymentProcessor.charge(userId, orderTotal);
    if (!paymentResult.success) {
      console.log('PAYMENT_FAILED:' + paymentResult.error);
      return null;
    }
    
    // BUG 3: Inventory update and payment not atomic - if inventory fails, payment isn't refunded
    const inventoryUpdate = await inventoryManager.reserveItems(items);
    if (!inventoryUpdate.success) {
      console.log('INVENTORY_FAILED:Items out of stock');
      // BUG: Should refund payment but doesn't
      return null;
    }
    
    const order = {
      id: orderId,
      userId,
      items,
      status: 'pending', // BUG 4: Never changes to confirmed
      total: orderTotal,
      createdAt: Date.now(),
      paymentId: paymentResult.id,
      shipping: { address: shippingAddress, status: 'pending' }
    };
    
    orders[orderId] = order;
    
    // BUG 5: Notification sent but doesn't verify delivery
    notificationService.sendOrderConfirmation(userId, orderId);
    
    console.log('ORDER_CREATED:' + orderId);
    return orderId;
  },
  
  getOrder(orderId) {
    // BUG 6: No authorization check - anyone can view any order
    return orders[orderId];
  },
  
  updateOrderStatus(orderId, status) {
    // BUG 7: Status can be set to invalid values, no state machine
    if (!orders[orderId]) return false;
    orders[orderId].status = status;
    return true;
  },
  
  async processRefund(orderId) {
    const order = orders[orderId];
    // BUG 8: Inventory is never restored when refunding
    const refund = await paymentProcessor.refund(order.paymentId, order.total);
    if (refund.success) {
      order.status = 'refunded';
      console.log('REFUND_PROCESSED:' + orderId);
    }
  }
};`},"inventory.js":{name:"inventory.js",language:"javascript",content:`const inventory = {
  'SKU-001': { name: 'Laptop', stock: 5, reserved: 0 },
  'SKU-002': { name: 'Mouse', stock: 50, reserved: 0 },
  'SKU-003': { name: 'Keyboard', stock: 20, reserved: 0 }
};

export const inventoryManager = {
  async reserveItems(items) {
    // BUG 1: No transaction - each item reserved separately, can fail mid-way
    for (const item of items) {
      const product = inventory[item.sku];
      
      // BUG 2: Using 'stock' directly without lock causes race conditions
      if (!product || (product.stock - product.reserved) < item.quantity) {
        return { success: false };
      }
      
      // BUG 3: Async operation without atomic update
      await new Promise(r => setTimeout(r, 10));
      product.reserved += item.quantity;
    }
    
    console.log('INVENTORY_RESERVED:success');
    return { success: true };
  },
  
  async confirmReservation(items) {
    // BUG 4: Never called, so stock is never actually decremented
    for (const item of items) {
      const product = inventory[item.sku];
      product.stock -= item.quantity;
      product.reserved -= item.quantity;
    }
  },
  
  async cancelReservation(items) {
    // BUG 5: Doesn't update stock, only unreserves
    for (const item of items) {
      const product = inventory[item.sku];
      product.reserved -= item.quantity;
    }
  },
  
  getInventory() {
    return inventory;
  }
};`},"payment.js":{name:"payment.js",language:"javascript",content:`const charges = {};
let chargeIdCounter = 1000;

export const paymentProcessor = {
  async charge(userId, amount) {
    // BUG 1: No idempotency - same request charged twice
    // Should use idempotency key
    
    // Simulate payment gateway call
    await new Promise(r => setTimeout(r, 50));
    
    // BUG 2: No validation of amount
    if (amount <= 0) {
      return { success: false, error: 'Invalid amount' };
    }
    
    const chargeId = 'CHARGE_' + (chargeIdCounter++);
    charges[chargeId] = {
      id: chargeId,
      userId,
      amount,
      status: 'succeeded', // BUG 3: Always succeeds, no actual validation
      timestamp: Date.now()
    };
    
    console.log('PAYMENT_CHARGED:' + chargeId);
    return { success: true, id: chargeId };
  },
  
  async refund(chargeId, amount) {
    // BUG 4: No check if charge exists or is already refunded
    // BUG 5: Partial refunds not tracked
    if (!charges[chargeId]) {
      return { success: false };
    }
    
    charges[chargeId].status = 'refunded';
    console.log('REFUND_ISSUED:' + chargeId);
    return { success: true };
  },
  
  getCharges() {
    return charges;
  }
};`},"notifications.js":{name:"notifications.js",language:"javascript",content:`const sentNotifications = [];

export const notificationService = {
  async sendOrderConfirmation(userId, orderId) {
    // BUG 1: Fire-and-forget, no error handling
    this.send(userId, 'ORDER_CONFIRMED', { orderId });
    // BUG 2: No tracking if notification was actually delivered
  },
  
  async sendShippingUpdate(userId, orderId, trackingId) {
    this.send(userId, 'SHIPPED', { orderId, trackingId });
  },
  
  send(userId, eventType, data) {
    // BUG 3: No validation of user existence
    const notification = {
      userId,
      type: eventType,
      data,
      sentAt: Date.now(),
      delivered: false // BUG 4: Never changes to true
    };
    
    sentNotifications.push(notification);
    console.log('NOTIFICATION_SENT:' + eventType);
    return notification;
  },
  
  getNotifications(userId) {
    // BUG 5: Includes undelivered and delivered without distinction
    return sentNotifications.filter(n => n.userId === userId);
  }
};`},"utils.js":{name:"utils.js",language:"javascript",content:`let orderIdCounter = 1;

export function generateOrderId() {
  // BUG: Simple counter, not unique across restarts
  // Should use timestamp or UUID
  return 'ORD-' + (orderIdCounter++);
}

export function validateEmail(email) {
  // BUG: Regex too permissive
  return /^.+@.+$/.test(email);
}

export function formatCurrency(amount) {
  // BUG: Not properly rounding
  return '$' + amount.toFixed(2);
}`},"integrationTest.js":{name:"integrationTest.js",language:"javascript",content:`import { orderService } from './orderService.js';
import { inventoryManager } from './inventory.js';
import { paymentProcessor } from './payment.js';

(async () => {
  console.log('=== Order System Test ===');
  
  const items = [
    { sku: 'SKU-001', quantity: 1, price: 1000 },
    { sku: 'SKU-002', quantity: 2, price: 50 }
  ];
  
  const orderId = await orderService.createOrder('user123', items, '123 Main St');
  
  if (orderId) {
    const order = orderService.getOrder(orderId);
    console.log('ORDER_TOTAL:' + order.total);
    console.log('ORDER_STATUS:' + order.status);
    console.log('INVENTORY:' + JSON.stringify(inventoryManager.getInventory()));
    console.log('CHARGES:' + Object.keys(paymentProcessor.getCharges()).length);
  }
  
  console.log('TEST_COMPLETE');
})();`}},constraints:["Atomic transactions required.","Proper state management.","Error handling with rollback."],testCheck:`
            const orderCode = files['orderService.js']?.content || '';
            const inventoryCode = files['inventory.js']?.content || '';
            const paymentCode = files['payment.js']?.content || '';
            
            const hasRefund = paymentCode.includes('refund') || orderCode.includes('refund');
            const hasConfirmation = inventoryCode.includes('confirmReservation') && orderCode.includes('confirmReservation');
            const runtimePassed = stdout.includes('ORDER_CREATED:') && stdout.includes('ORDER_TOTAL:1100');
            
            return hasRefund && hasConfirmation && runtimePassed;
        `},{id:"realtime-messaging-app",title:"Real-Time Messaging Application",description:"Debug a chat application with message ordering issues, missing messages, duplicate delivery, and user presence tracking bugs. Messages arrive out of order, some are lost, others duplicated.",difficulty:"Hard",badgeColor:"bg-red-500/10 text-red-500 border border-red-500/20",files:{"messageQueue.js":{name:"messageQueue.js",language:"javascript",content:`const messages = {}; // roomId -> message[]
let messageIdCounter = 1;

export const messageQueue = {
  async addMessage(roomId, userId, text, timestamp) {
    // BUG 1: messageId based on counter, can collide in distributed system
    const messageId = 'MSG_' + (messageIdCounter++);
    
    // BUG 2: No validation of roomId or userId
    if (!messages[roomId]) {
      messages[roomId] = [];
    }
    
    const message = {
      id: messageId,
      roomId,
      userId,
      text,
      // BUG 3: Using Date.now() instead of server timestamp
      // Client timestamps can be out of order
      timestamp: timestamp || Date.now(),
      delivered: false,
      read: false
    };
    
    // BUG 4: Push to array without ordering by timestamp
    messages[roomId].push(message);
    console.log('MESSAGE_ADDED:' + messageId);
    
    return message;
  },
  
  async getMessages(roomId, limit = 50) {
    // BUG 5: No pagination offset, always returns from start
    // BUG 6: Messages not sorted by timestamp
    const roomMessages = messages[roomId] || [];
    return roomMessages.slice(-limit);
  },
  
  async markAsDelivered(messageId) {
    // BUG 7: Inefficient linear search across all rooms
    for (const roomId in messages) {
      const msg = messages[roomId].find(m => m.id === messageId);
      if (msg) {
        msg.delivered = true;
        // BUG 8: Doesn't return, no confirmation
        return;
      }
    }
  },
  
  async markAsRead(messageId, userId) {
    // BUG 9: No validation that userId is in the room
    for (const roomId in messages) {
      const msg = messages[roomId].find(m => m.id === messageId);
      if (msg) {
        msg.read = true;
        return true;
      }
    }
  }
};`},"userPresence.js":{name:"userPresence.js",language:"javascript",content:`const presence = {}; // userId -> { status, rooms, lastSeen }

export const userPresence = {
  userJoinRoom(userId, roomId) {
    // BUG 1: No validation of userId or roomId format
    if (!presence[userId]) {
      presence[userId] = {
        status: 'online',
        rooms: [],
        lastSeen: Date.now()
      };
    }
    
    // BUG 2: No check for duplicate room, can add same room twice
    presence[userId].rooms.push(roomId);
    console.log('USER_JOINED:' + userId + ':' + roomId);
  },
  
  userLeaveRoom(userId, roomId) {
    // BUG 3: No error handling if user not in room
    const idx = presence[userId].rooms.indexOf(roomId);
    presence[userId].rooms.splice(idx, 1);
    
    // BUG 4: User should go offline only if no rooms left
    // Currently just leaves room but stays online
  },
  
  getRoomUsers(roomId) {
    // BUG 5: Very inefficient - O(n) for every room query
    const users = [];
    for (const userId in presence) {
      if (presence[userId].rooms.includes(roomId)) {
        users.push({ userId, status: presence[userId].status });
      }
    }
    return users;
  },
  
  userGoOffline(userId) {
    // BUG 6: Doesn't clear rooms, just marks offline
    // User still in rooms but offline
    if (presence[userId]) {
      presence[userId].status = 'offline';
      // BUG 7: No notification to other users
    }
  },
  
  getPresence(userId) {
    return presence[userId] || null;
  }
};`},"roomManager.js":{name:"roomManager.js",language:"javascript",content:`const rooms = {}; // roomId -> room data

export const roomManager = {
  createRoom(roomName, creatorId) {
    // BUG 1: No unique ID generation, uses name as ID
    // If 2 rooms created with same name, one overwrites
    const roomId = roomName.toLowerCase();
    
    if (rooms[roomId]) {
      console.log('ROOM_EXISTS:' + roomId);
      return null; // BUG 2: Creator probably should be added to existing room
    }
    
    rooms[roomId] = {
      id: roomId,
      name: roomName,
      createdAt: Date.now(),
      creatorId,
      members: [creatorId], // BUG 3: Creator not properly tracked
      maxMembers: 100
    };
    
    console.log('ROOM_CREATED:' + roomId);
    return roomId;
  },
  
  addUserToRoom(roomId, userId) {
    // BUG 4: No check if user already in room
    const room = rooms[roomId];
    if (!room) return false;
    
    if (room.members.length >= room.maxMembers) {
      console.log('ROOM_FULL:' + roomId);
      return false;
    }
    
    room.members.push(userId);
    console.log('USER_ADDED:' + userId);
    return true;
  },
  
  removeUserFromRoom(roomId, userId) {
    // BUG 5: No check if user was actually in room
    const room = rooms[roomId];
    const idx = room.members.indexOf(userId);
    room.members.splice(idx, 1);
    
    // BUG 6: Room should be deleted if empty, but isn't
  },
  
  getRoom(roomId) {
    return rooms[roomId];
  }
};`},"connectionHandler.js":{name:"connectionHandler.js",language:"javascript",content:`import { messageQueue } from './messageQueue.js';
import { userPresence } from './userPresence.js';
import { roomManager } from './roomManager.js';

export const connectionHandler = {
  async handleNewMessage(roomId, userId, text) {
    // BUG 1: Client timestamp used, not server timestamp
    const timestamp = Date.now();
    const message = await messageQueue.addMessage(roomId, userId, text, timestamp);
    
    // BUG 2: Should broadcast to all users in room, but doesn't
    // Message goes into queue but not sent to clients
    return message;
  },
  
  async handleUserJoin(roomId, userId) {
    // BUG 3: No atomic operation - multiple failures possible
    userPresence.userJoinRoom(userId, roomId);
    roomManager.addUserToRoom(roomId, userId);
    
    // BUG 4: History not sent to user
    // BUG 5: Presence not broadcast to others
    console.log('USER_CONNECTED:' + userId + ':' + roomId);
  },
  
  async handleUserLeave(roomId, userId) {
    userPresence.userLeaveRoom(userId, roomId);
    roomManager.removeUserFromRoom(roomId, userId);
    
    // BUG 6: No notification to remaining users
    console.log('USER_DISCONNECTED:' + userId + ':' + roomId);
  }
};`},"test.js":{name:"test.js",language:"javascript",content:`import { connectionHandler } from './connectionHandler.js';
import { messageQueue } from './messageQueue.js';
import { userPresence } from './userPresence.js';
import { roomManager } from './roomManager.js';

(async () => {
  console.log('=== Messaging App Test ===');
  
  // Create room
  const roomId = roomManager.createRoom('general', 'user1');
  
  // Users join
  await connectionHandler.handleUserJoin(roomId, 'user1');
  await connectionHandler.handleUserJoin(roomId, 'user2');
  
  // Send messages
  await connectionHandler.handleNewMessage(roomId, 'user1', 'Hello');
  await connectionHandler.handleNewMessage(roomId, 'user2', 'Hi there');
  await connectionHandler.handleNewMessage(roomId, 'user1', 'How are you?');
  
  // Check messages
  const msgs = await messageQueue.getMessages(roomId);
  console.log('MESSAGES_COUNT:' + msgs.length);
  
  // Check presence
  const roomUsers = userPresence.getRoomUsers(roomId);
  console.log('ROOM_USERS:' + roomUsers.length);
  
  console.log('TEST_COMPLETE');
})();`}},constraints:["Proper message ordering.","Atomic user operations.","Presence tracking."],testCheck:`
            const queueCode = files['messageQueue.js']?.content || '';
            const presenceCode = files['userPresence.js']?.content || '';
            const connCode = files['connectionHandler.js']?.content || '';
            
            const hasOrdering = queueCode.includes('sort') || queueCode.includes('timestamp');
            const hasPresence = presenceCode.includes('offlineonly') || presenceCode.includes('rooms.length === 0');
            const runtimePassed = stdout.includes('MESSAGES_COUNT:') && stdout.includes('ROOM_USERS:');
            
            return (hasOrdering || hasPresence) && runtimePassed;
        `},{id:"user-auth-system",title:"User Authentication & Authorization",description:"Debug a complete auth system with token expiration bugs, privilege escalation vulnerabilities, session hijacking issues, and password reset token reuse problems.",difficulty:"Hard",badgeColor:"bg-red-500/10 text-red-500 border border-red-500/20",files:{"authService.js":{name:"authService.js",language:"javascript",content:`import { tokenManager } from './tokenManager.js';
import { userStore } from './userStore.js';
import { passwordUtils } from './passwordUtils.js';

export const authService = {
  async register(email, password, name) {
    // BUG 1: No email validation, allows invalid formats
    if (!email || !password || password.length < 6) {
      console.log('INVALID_INPUT:Password too short');
      return null;
    }
    
    // BUG 2: No check for duplicate email
    const userId = Math.random().toString(36); // BUG 3: Not a real ID
    
    const passwordHash = passwordUtils.hashPassword(password);
    
    userStore.saveUser({
      id: userId,
      email,
      password: passwordHash,
      name,
      created: Date.now(),
      role: 'user', // BUG 4: Should never default to user
      suspended: false
    });
    
    console.log('USER_REGISTERED:' + userId);
    return userId;
  },
  
  async login(email, password) {
    const user = userStore.findByEmail(email);
    
    // BUG 5: User not found case doesn't fail, leaks user existence
    if (!user || !passwordUtils.verifyPassword(password, user.password)) {
      console.log('LOGIN_FAILED:Invalid credentials');
      return null;
    }
    
    // BUG 6: No check for suspended users
    // BUG 7: No login attempt rate limiting
    
    const token = tokenManager.createToken(user.id, user.role);
    console.log('LOGIN_SUCCESS:' + email);
    
    return { token, user: { id: user.id, email, name: user.name } };
  },
  
  async logout(token) {
    // BUG 8: Token never actually invalidated
    // Token can still be used after logout
    console.log('LOGOUT:' + token.substring(0, 10));
  }
};`},"tokenManager.js":{name:"tokenManager.js",language:"javascript",content:`const activeTokens = {};
const tokenBlacklist = []; // BUG: Never used for logout

export const tokenManager = {
  createToken(userId, role) {
    // BUG 1: Token format not secure (no signature)
    // Anyone can modify the payload
    const payload = {
      userId,
      role,
      exp: Date.now() + 3600000, // 1 hour
      iat: Date.now()
    };
    
    // BUG 2: Just base64 encoding, not encrypted
    const token = Buffer.from(JSON.stringify(payload)).toString('base64');
    activeTokens[token] = payload;
    
    console.log('TOKEN_CREATED:' + token.substring(0, 10));
    return token;
  },
  
  verifyToken(token) {
    // BUG 3: Blacklist never checked
    // BUG 4: No signature verification
    
    try {
      const payload = JSON.parse(Buffer.from(token, 'base64').toString());
      
      // BUG 5: Expiration check missing
      // if (payload.exp < Date.now()) return null;
      
      // BUG 6: Doesn't verify payload hasn't been modified
      return payload;
    } catch {
      return null;
    }
  },
  
  invalidateToken(token) {
    // BUG 7: Only adds to blacklist, but blacklist not checked in verify
    tokenBlacklist.push(token);
    console.log('TOKEN_INVALIDATED:' + token.substring(0, 10));
  }
};`},"authMiddleware.js":{name:"authMiddleware.js",language:"javascript",content:`import { tokenManager } from './tokenManager.js';

export function authMiddleware(req, res, next) {
  // BUG 1: Token taken from header without validation
  const token = req.headers.authorization?.split(' ')[1];
  
  if (!token) {
    console.log('NO_TOKEN:Unauthorized');
    return; // BUG 2: Should return error, not just log
  }
  
  const payload = tokenManager.verifyToken(token);
  
  if (!payload) {
    console.log('INVALID_TOKEN:Unauthorized');
    return; // BUG 3: Should prevent further processing
  }
  
  // BUG 4: No check if user still exists or is suspended
  req.userId = payload.userId;
  req.userRole = payload.role;
  
  next();
}

export function requireAdmin(req, res, next) {
  // BUG 5: Role can be spoofed by modifying token payload
  if (req.userRole !== 'admin') {
    console.log('FORBIDDEN:Admin required');
    return;
  }
  
  next();
}`},"passwordReset.js":{name:"passwordReset.js",language:"javascript",content:`import { userStore } from './userStore.js';
import { passwordUtils } from './passwordUtils.js';

const resetTokens = {}; // token -> { userId, expires, used }

export const passwordReset = {
  async requestReset(email) {
    const user = userStore.findByEmail(email);
    if (!user) {
      // BUG 1: Should not leak user existence
      console.log('USER_NOT_FOUND:' + email);
      return null;
    }
    
    // BUG 2: Token is predictable
    const resetToken = Math.random().toString(36).substring(7);
    
    resetTokens[resetToken] = {
      userId: user.id,
      // BUG 3: No expiration time set
      // BUG 4: Token never expires
      created: Date.now()
    };
    
    console.log('RESET_TOKEN_SENT:' + user.id);
    return resetToken;
  },
  
  async resetPassword(resetToken, newPassword) {
    // BUG 5: No check if token has already been used
    const tokenData = resetTokens[resetToken];
    
    if (!tokenData) {
      console.log('INVALID_TOKEN:Token not found');
      return false;
    }
    
    const user = userStore.findById(tokenData.userId);
    if (!user) return false;
    
    // BUG 6: No validation of new password strength
    const passwordHash = passwordUtils.hashPassword(newPassword);
    user.password = passwordHash;
    
    // BUG 7: Token not marked as used, can be reused
    // Should delete or mark as consumed
    console.log('PASSWORD_RESET:' + user.id);
    
    return true;
  }
};`},"userStore.js":{name:"userStore.js",language:"javascript",content:`const users = {};
const emailIndex = {};

export const userStore = {
  saveUser(user) {
    users[user.id] = user;
    emailIndex[user.email] = user.id;
  },
  
  findById(userId) {
    return users[userId];
  },
  
  findByEmail(email) {
    const userId = emailIndex[email.toLowerCase()]; // BUG: Case sensitivity
    return users[userId];
  },
  
  updateUser(userId, updates) {
    // BUG: No validation of what can be updated
    // User can change their own role
    Object.assign(users[userId], updates);
  },
  
  deleteUser(userId) {
    const user = users[userId];
    delete emailIndex[user.email];
    delete users[userId];
  }
};`},"passwordUtils.js":{name:"passwordUtils.js",language:"javascript",content:`export const passwordUtils = {
  hashPassword(password) {
    // BUG 1: Not actually hashing, just encoding
    // Should use bcrypt or similar
    return Buffer.from(password).toString('base64');
  },
  
  verifyPassword(password, hash) {
    // BUG 2: Vulnerable to timing attacks
    return Buffer.from(password).toString('base64') === hash;
  },
  
  isStrongPassword(password) {
    // BUG 3: Very weak validation
    return password.length >= 6; // Should require uppercase, numbers, special chars
  }
};`},"test.js":{name:"test.js",language:"javascript",content:`import { authService } from './authService.js';
import { tokenManager } from './tokenManager.js';
import { passwordReset } from './passwordReset.js';

(async () => {
  console.log('=== Auth System Test ===');
  
  // Register user
  const userId = await authService.register('user@example.com', 'password123', 'John');
  
  // Login
  const result = await authService.login('user@example.com', 'password123');
  if (result) {
    const token = result.token;
    console.log('LOGIN_SUCCESS:' + result.user.email);
    
    // Verify token
    const payload = tokenManager.verifyToken(token);
    console.log('TOKEN_VALID:' + !!payload);
    
    // Logout (should invalidate token)
    await authService.logout(token);
    
    // BUG: Token still works after logout
    const stillValid = tokenManager.verifyToken(token);
    console.log('AFTER_LOGOUT:' + !!stillValid);
  }
  
  console.log('TEST_COMPLETE');
})();`}},constraints:["Proper token validation.","Password security.","Token expiration."],testCheck:`
            const tokenCode = files['tokenManager.js']?.content || '';
            const authCode = files['authService.js']?.content || '';
            const resetCode = files['passwordReset.js']?.content || '';
            
            const hasExpiration = tokenCode.includes('exp') && (authCode.includes('exp') || resetCode.includes('expires'));
            const hasBlacklist = tokenCode.includes('blacklist') && authCode.includes('invalidateToken');
            const runtimePassed = stdout.includes('LOGIN_SUCCESS:') && stdout.includes('TOKEN_VALID:');
            
            return hasExpiration && hasBlacklist && runtimePassed;
        `},{id:"conduit-article-feed",title:"Conduit: Global Article Feed",description:"Implement a high-performance data transformation layer for the Conduit (RealWorld) article feed. Map raw database records to the standardized API response structure.",difficulty:"Hard",badgeColor:"bg-red-500/10 text-red-500 border border-red-500/20",files:{"mockDatabase.js":{name:"mockDatabase.js",language:"javascript",content:`export const RAW_ARTICLES = [
  {
    id: 1,
    slug: "how-to-train-your-dragon",
    title: "How to train your dragon",
    description: "Ever wonder how?",
    body: "It takes a lot of patience and berries.",
    tags: ["dragons", "training"],
    created_at: "2016-02-18T03:22:56.637Z",
    updated_at: "2016-02-18T03:48:35.824Z",
    author_id: 101,
    fav_count: 42
  },
  {
    id: 2,
    slug: "how-to-train-your-dragon-2",
    title: "How to train your dragon 2",
    description: "So toothless",
    body: "It is a dragon.",
    tags: ["dragons", "training"],
    created_at: "2016-02-18T03:30:56.637Z",
    updated_at: "2016-02-18T03:50:35.824Z",
    author_id: 101,
    fav_count: 10
  }
];

export const RAW_USERS = {
  101: {
    username: "jake",
    bio: "I work at statefarm",
    image: "https://i.stack.imgur.com/xHW68.jpg",
    following: false
  }
};`},"apiContract.js":{name:"apiContract.js",language:"javascript",content:`/**
 * Expected Output Format per RealWorld Spec:
 * {
 *   "articles": [{
 *     "slug": string,
 *     "title": string,
 *     "description": string,
 *     "body": string,
 *     "tagList": string[],
 *     "createdAt": string (ISO),
 *     "updatedAt": string (ISO),
 *     "favorited": boolean,
 *     "favoritesCount": number,
 *     "author": {
 *       "username": string,
 *       "bio": string,
 *       "image": string,
 *       "following": boolean
 *     }
 *   }],
 *   "articlesCount": number
 * }
 */`},"feedTransformer.js":{name:"feedTransformer.js",language:"javascript",content:`import { RAW_ARTICLES, RAW_USERS } from './mockDatabase.js';

export function transformFeed(articles, users, currentUserId) {
  // TODO: Transform the raw database articles into the API contract format
  // 1. Map 'tags' to 'tagList'
  // 2. camelCase the timestamps
  // 3. Nest the author object using author_id
  // 4. Calculate 'favorited' status (assume false for now)
  
  return {
    articles: articles.map(a => {
      return a; // BUG: Returning untransformed data
    }),
    articlesCount: articles.length
  };
}`},"index.js":{name:"index.js",language:"javascript",content:`import { transformFeed } from './feedTransformer.js';
import { RAW_ARTICLES, RAW_USERS } from './mockDatabase.js';

const result = transformFeed(RAW_ARTICLES, RAW_USERS, null);
console.log('ARTICLES_COUNT:' + result.articlesCount);
if (result.articles[0] && result.articles[0].author) {
  console.log('AUTHOR_NAME:' + result.articles[0].author.username);
  console.log('TAG_LIST:' + (result.articles[0].tagList ? 'VALID' : 'MISSING'));
  console.log('CAMEL_CASE:' + (result.articles[0].createdAt ? 'VALID' : 'MISSING'));
}`},"README.md":{name:"README.md",language:"markdown",content:"# Conduit: Global Article Feed\n\nYour task is to implement the `transformFeed` function in `feedTransformer.js` to match the **RealWorld (Conduit)** API specification.\n\n### Requirements:\n- Rename `tags` to `tagList`.\n- Convert `created_at` and `updated_at` to `createdAt` and `updatedAt`.\n- Rename `fav_count` to `favoritesCount`.\n- Populate the `author` object using the `author_id` to lookup data in the `users` map.\n- Add a `favorited` boolean (default to `false`)."}},constraints:["Strict adherence to OpenAPI schema.","Efficient map-lookup for authors.","Correct timestamp mapping."],testCheck:`
            const transformerCode = files['feedTransformer.js']?.content || '';
            const hasAuthorMapping = transformerCode.includes('RAW_USERS[') || transformerCode.includes('users[');
            const hasTagMapping = transformerCode.includes('tagList');
            const hasCamelCase = transformerCode.includes('createdAt');
            const runtimePassed = stdout.includes('ARTICLES_COUNT:2') && stdout.includes('AUTHOR_NAME:jake') && stdout.includes('TAG_LIST:VALID') && stdout.includes('CAMEL_CASE:VALID');
            
            return hasAuthorMapping && hasTagMapping && hasCamelCase && runtimePassed;
        `},{id:"css-flexbox-bug",type:"frontend",title:"Broken Flex Layout",description:"Three nav items should sit side-by-side in a horizontal row, but they are stacking vertically. Fix the CSS so the layout displays correctly.",difficulty:"Easy",badgeColor:"bg-green-500/10 text-green-500 border border-green-500/20",files:{"index.html":{name:"index.html",language:"html",content:`<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <style>
    body { margin: 0; font-family: sans-serif; background: #111; color: #fff; display: flex; align-items: center; justify-content: center; height: 100vh; }
    .navbar {
      display: flex;
      /* BUG: items are stacking because of wrong flex-direction */
      flex-direction: column;
      gap: 12px;
      background: #1e1e1e;
      padding: 16px 32px;
      border-radius: 12px;
      border: 1px solid #333;
    }
    .nav-item {
      padding: 10px 24px;
      background: #7c3aed;
      border-radius: 8px;
      font-weight: bold;
      color: #fff;
      text-align: center;
    }
  </style>
</head>
<body>
  <nav class="navbar">
    <div class="nav-item">Home</div>
    <div class="nav-item">About</div>
    <div class="nav-item">Contact</div>
  </nav>
</body>
</html>`}},constraints:["Items must use flexbox.","Change flex-direction to 'row'."],desiredOutput:`<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <style>
    body { margin: 0; font-family: sans-serif; background: #111; color: #fff; display: flex; align-items: center; justify-content: center; height: 100vh; }
    .navbar { display: flex; flex-direction: row; gap: 12px; background: #1e1e1e; padding: 16px 32px; border-radius: 12px; border: 1px solid #333; }
    .nav-item { padding: 10px 24px; background: #7c3aed; border-radius: 8px; font-weight: bold; color: #fff; text-align: center; }
  </style>
</head>
<body>
  <nav class="navbar">
    <div class="nav-item">Home</div>
    <div class="nav-item">About</div>
    <div class="nav-item">Contact</div>
  </nav>
</body>
</html>`,previewCheck:`
      try {
        const navbar = document.querySelector('.navbar');
        const style = window.getComputedStyle(navbar);
        const dir = style.flexDirection;
        return dir === 'row' || dir === 'row-reverse';
      } catch(e) { return false; }
    `},{id:"js-counter-bug",type:"frontend",title:"Click Counter Bug",description:"A counter button should increment the displayed number each time it is clicked. The counter is stuck at 0. Find and fix the two bugs.",difficulty:"Easy",badgeColor:"bg-green-500/10 text-green-500 border border-green-500/20",files:{"index.html":{name:"index.html",language:"html",content:`<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <style>
    body { margin: 0; font-family: sans-serif; background: #111; color: #fff; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; gap: 24px; }
    #count { font-size: 72px; font-weight: 900; color: #a78bfa; }
    button { padding: 14px 40px; background: #7c3aed; color: #fff; border: none; border-radius: 12px; font-size: 18px; font-weight: bold; cursor: pointer; }
    button:hover { background: #6d28d9; }
  </style>
</head>
<body>
  <div id="count">0</div>
  <!-- BUG 1: button type submit causes page reload -->
  <button type="submit" id="btn">Click Me</button>
  <script>
    let count = 0;
    const display = document.getElementById('count');
    // BUG 2: wrong element ID - 'button' does not exist, should be 'btn'
    document.getElementById('button').addEventListener('click', () => {
      count++;
      display.textContent = count;
    });
  <\/script>
</body>
</html>`}},constraints:["Fix button type to 'button'.","Fix getElementById to use 'btn'."],desiredOutput:`<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <style>
    body { margin: 0; font-family: sans-serif; background: #111; color: #fff; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; gap: 24px; }
    #count { font-size: 72px; font-weight: 900; color: #a78bfa; }
    button { padding: 14px 40px; background: #7c3aed; color: #fff; border: none; border-radius: 12px; font-size: 18px; font-weight: bold; cursor: pointer; }
  </style>
</head>
<body>
  <div id="count">1</div>
  <button type="button" id="btn">Click Me</button>
</body>
</html>`,previewCheck:`
      try {
        const btn = document.getElementById('btn');
        if (!btn) return false;
        const correctType = btn.type === 'button';
        btn.click();
        const count = parseInt(document.getElementById('count').textContent, 10);
        return correctType && count > 0;
      } catch(e) { return false; }
    `},{id:"dom-color-bug",type:"frontend",title:"Dynamic Color Picker",description:"Clicking a color swatch should change the background of the box below it. Nothing happens. Fix the JavaScript typo.",difficulty:"Medium",badgeColor:"bg-yellow-500/10 text-yellow-500 border border-yellow-500/20",files:{"index.html":{name:"index.html",language:"html",content:`<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <style>
    body { margin: 0; font-family: sans-serif; background: #111; color: #fff; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; gap: 32px; }
    .swatches { display: flex; gap: 16px; }
    .swatch { width: 60px; height: 60px; border-radius: 50%; cursor: pointer; border: 3px solid transparent; transition: transform 0.2s; }
    .swatch:hover { transform: scale(1.1); border-color: #fff; }
    #color-box { width: 200px; height: 200px; border-radius: 24px; background: #1e1e1e; border: 2px solid #333; transition: background 0.3s; }
    p { color: #666; font-size: 14px; }
  </style>
</head>
<body>
  <div class="swatches">
    <div class="swatch" data-color="#7c3aed" style="background:#7c3aed"></div>
    <div class="swatch" data-color="#ec4899" style="background:#ec4899"></div>
    <div class="swatch" data-color="#10b981" style="background:#10b981"></div>
    <div class="swatch" data-color="#f59e0b" style="background:#f59e0b"></div>
  </div>
  <div id="color-box"></div>
  <p>Click a color swatch above</p>
  <script>
    const swatches = document.querySelectorAll('.swatch');
    const box = document.getElementById('color-box');
    swatches.forEach(swatch => {
      swatch.addEventListener('click', () => {
        // BUG: dataset.colour does not exist (typo - should be dataset.color)
        box.style.backgroundColor = swatch.dataset.colour;
      });
    });
  <\/script>
</body>
</html>`}},constraints:["Fix 'dataset.colour' typo to 'dataset.color'.","Set box.style.backgroundColor."],desiredOutput:`<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <style>
    body { margin: 0; font-family: sans-serif; background: #111; color: #fff; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; gap: 32px; }
    .swatches { display: flex; gap: 16px; }
    .swatch { width: 60px; height: 60px; border-radius: 50%; cursor: pointer; border: 3px solid transparent; }
    #color-box { width: 200px; height: 200px; border-radius: 24px; background: #7c3aed; border: 2px solid #7c3aed; }
  </style>
</head>
<body>
  <div class="swatches">
    <div class="swatch" style="background:#7c3aed"></div>
    <div class="swatch" style="background:#ec4899"></div>
  </div>
  <div id="color-box" style="background:#7c3aed"></div>
</body>
</html>`,previewCheck:`
      try {
        const swatch = document.querySelector('.swatch');
        const box = document.getElementById('color-box');
        if (!swatch || !box) return false;
        swatch.click();
        const bg = box.style.backgroundColor || box.style.background;
        return bg !== '' && bg !== undefined && bg !== 'rgba(0, 0, 0, 0)';
      } catch(e) { return false; }
    `},{id:"css-card-hover-bug",type:"frontend",title:"Card Hover Effect Bug",description:"Three product cards should lift and glow on hover. The hover animation is completely broken. Fix the two CSS bugs.",difficulty:"Medium",badgeColor:"bg-yellow-500/10 text-yellow-500 border border-yellow-500/20",files:{"index.html":{name:"index.html",language:"html",content:`<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; font-family: sans-serif; background: #0a0a0a; color: #fff; display: flex; align-items: center; justify-content: center; height: 100vh; gap: 24px; }
    .card {
      width: 180px; padding: 28px 20px; background: #1a1a2e;
      border: 1px solid #333; border-radius: 16px; text-align: center; cursor: pointer;
      /* BUG 1: transition is missing */
    }
    /* BUG 2: wrong selector - '.cards:hover' should be '.card:hover' */
    .cards:hover {
      transform: translateY(-12px);
      box-shadow: 0 20px 60px rgba(124, 58, 237, 0.5);
      border-color: #7c3aed;
    }
    .card-icon { font-size: 40px; margin-bottom: 12px; }
    .card-title { font-size: 14px; font-weight: bold; color: #a78bfa; }
    .card-price { font-size: 22px; font-weight: 900; margin-top: 8px; }
  </style>
</head>
<body>
  <div class="card"><div class="card-icon">&#127911;</div><div class="card-title">Headphones</div><div class="card-price">$129</div></div>
  <div class="card"><div class="card-icon">&#9000;</div><div class="card-title">Keyboard</div><div class="card-price">$89</div></div>
  <div class="card"><div class="card-icon">&#128432;</div><div class="card-title">Mouse</div><div class="card-price">$59</div></div>
</body>
</html>`}},constraints:["Add 'transition: transform 0.3s, box-shadow 0.3s' to .card.","Fix hover selector from '.cards:hover' to '.card:hover'."],desiredOutput:`<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; font-family: sans-serif; background: #0a0a0a; color: #fff; display: flex; align-items: center; justify-content: center; height: 100vh; gap: 24px; }
    .card { width: 180px; padding: 28px 20px; background: #1a1a2e; border: 1px solid #7c3aed; border-radius: 16px; text-align: center; cursor: pointer; transition: transform 0.3s, box-shadow 0.3s, border-color 0.3s; transform: translateY(-12px); box-shadow: 0 20px 60px rgba(124,58,237,0.5); }
    .card-icon { font-size: 40px; margin-bottom: 12px; }
    .card-title { font-size: 14px; font-weight: bold; color: #a78bfa; }
    .card-price { font-size: 22px; font-weight: 900; margin-top: 8px; }
  </style>
</head>
<body>
  <div class="card"><div class="card-icon">&#127911;</div><div class="card-title">Headphones</div><div class="card-price">$129</div></div>
  <div class="card"><div class="card-icon">&#9000;</div><div class="card-title">Keyboard</div><div class="card-price">$89</div></div>
  <div class="card"><div class="card-icon">&#128432;</div><div class="card-title">Mouse</div><div class="card-price">$59</div></div>
</body>
</html>`,previewCheck:`
      try {
        const card = document.querySelector('.card');
        if (!card) return false;
        const style = window.getComputedStyle(card);
        const hasTransition = style.transitionDuration && style.transitionDuration !== '0s';
        let hasHoverRule = false;
        for (const sheet of document.styleSheets) {
          try {
            for (const rule of sheet.cssRules) {
              if (rule.selectorText && rule.selectorText.includes('.card:hover')) hasHoverRule = true;
            }
          } catch(e) {}
        }
        return hasTransition && hasHoverRule;
      } catch(e) { return false; }
    `},{id:"js-todo-list-bug",type:"frontend",title:"Todo List Builder Bug",description:'A todo app should add new tasks when clicking "Add". Items are not appearing and the input is never cleared. Fix the three JavaScript bugs.',difficulty:"Hard",badgeColor:"bg-red-500/10 text-red-500 border border-red-500/20",files:{"index.html":{name:"index.html",language:"html",content:`<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: sans-serif; background: #0d0d0d; color: #fff; display: flex; align-items: center; justify-content: center; height: 100vh; }
    .app { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 20px; padding: 32px; width: 380px; }
    h2 { font-size: 22px; font-weight: 900; margin-bottom: 24px; color: #a78bfa; }
    .input-row { display: flex; gap: 10px; margin-bottom: 24px; }
    input { flex: 1; padding: 12px 16px; background: #111; border: 1px solid #333; border-radius: 10px; color: #fff; font-size: 14px; outline: none; }
    input:focus { border-color: #7c3aed; }
    button { padding: 12px 20px; background: #7c3aed; border: none; border-radius: 10px; color: #fff; font-weight: bold; cursor: pointer; font-size: 14px; }
    button:hover { background: #6d28d9; }
    ul { list-style: none; display: flex; flex-direction: column; gap: 10px; }
    li { padding: 14px 18px; background: #111; border: 1px solid #2a2a2a; border-radius: 10px; font-size: 14px; display: flex; align-items: center; gap: 12px; }
    li::before { content: '\\2713'; color: #7c3aed; font-weight: bold; }
  </style>
</head>
<body>
  <div class="app">
    <h2>My Todos</h2>
    <div class="input-row">
      <input type="text" id="todo-input" placeholder="Add a task..." />
      <button id="add-btn">Add</button>
    </div>
    <ul id="todo-list"></ul>
  </div>
  <script>
    const input = document.getElementById('todo-input');
    const list = document.getElementById('todo-list');
    const btn = document.getElementById('add-btn');
    btn.addEventListener('click', addTodo);
    function addTodo() {
      const text = input.value;
      // BUG 1: No empty check - should guard with: if (!text.trim()) return;
      const li = document.createElement('li');
      // BUG 2: Using innerHTML with raw text - should use textContent
      li.innerHTML = text;
      list.appendChild(li);
      // BUG 3: Input is never cleared - should add: input.value = '';
    }
  <\/script>
</body>
</html>`}},constraints:["Guard against empty input with trim() check.","Use textContent not innerHTML.","Clear input.value after adding."],desiredOutput:`<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: sans-serif; background: #0d0d0d; color: #fff; display: flex; align-items: center; justify-content: center; height: 100vh; }
    .app { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 20px; padding: 32px; width: 380px; }
    h2 { font-size: 22px; font-weight: 900; margin-bottom: 24px; color: #a78bfa; }
    .input-row { display: flex; gap: 10px; margin-bottom: 24px; }
    input { flex: 1; padding: 12px 16px; background: #111; border: 1px solid #333; border-radius: 10px; color: #fff; font-size: 14px; }
    button { padding: 12px 20px; background: #7c3aed; border: none; border-radius: 10px; color: #fff; font-weight: bold; cursor: pointer; }
    ul { list-style: none; display: flex; flex-direction: column; gap: 10px; }
    li { padding: 14px 18px; background: #111; border: 1px solid #2a2a2a; border-radius: 10px; font-size: 14px; display: flex; align-items: center; gap: 12px; }
    li::before { content: '\\2713'; color: #7c3aed; font-weight: bold; }
  </style>
</head>
<body>
  <div class="app">
    <h2>My Todos</h2>
    <div class="input-row">
      <input type="text" id="todo-input" placeholder="Add a task..." />
      <button id="add-btn">Add</button>
    </div>
    <ul id="todo-list">
      <li>Buy groceries</li>
      <li>Write clean code</li>
    </ul>
  </div>
</body>
</html>`,previewCheck:`
      try {
        const input = document.getElementById('todo-input');
        const btn = document.getElementById('add-btn');
        const list = document.getElementById('todo-list');
        if (!input || !btn || !list) return false;
        input.value = 'Test Task';
        btn.click();
        const items = list.querySelectorAll('li');
        const added = items.length > 0;
        const cleared = input.value === '';
        const safeContent = items.length > 0 && items[items.length-1].textContent.trim() === 'Test Task';
        return added && cleared && safeContent;
      } catch(e) { return false; }
    `},{id:"dynamic-form-builder",title:"Dynamic Form Builder (Missing unique ID bug)",description:"The dynamic form builder is generating forms with missing unique IDs, causing issues with form submission and validation.",difficulty:"Medium",badgeColor:"bg-yellow-500/10 text-yellow-500 border border-yellow-500/20",files:{"index.js":{name:"index.js",language:"javascript",content:`const FormBuilder = function() {
  let forms = [];
  return {
    addForm: function(form) {
      forms.push(form);
      return this;
    },
    getForms: function() {
      return forms;
    }
  };
}

const formBuilder = FormBuilder();
formBuilder.addForm({ id: 1, name: 'Form 1' });
formBuilder.addForm({ id: 2, name: 'Form 2' });
console.log(formBuilder.getForms());`},"README.md":{name:"README.md",language:"markdown",content:`# Dynamic Form Builder (Missing unique ID bug)

## Task

Implement a dynamic form builder that adds a unique ID to each form.

## Constraints

* Each form must have a unique ID.
* The form builder must be reusable.`}},constraints:["Each form must have a unique ID.","The form builder must be reusable."],testCheck:"files.getForms().every(form => form.id !== undefined)"},{id:"infinite-scroll",title:"Infinite Scroll (Double-fetch race condition)",description:"The infinite scroll feature is causing a double-fetch race condition, resulting in duplicate data being fetched and displayed.",difficulty:"Hard",badgeColor:"bg-red-500/10 text-red-500 border border-red-500/20",files:{"index.js":{name:"index.js",language:"javascript",content:`const InfiniteScroll = function() {
  let data = [];
  let isLoading = false;
  return {
    fetchData: function() {
      if (isLoading) return;
      isLoading = true;
      // Simulate data fetching
      setTimeout(() => {
        data.push({ id: 1, name: 'Item 1' });
        data.push({ id: 2, name: 'Item 2' });
        isLoading = false;
        console.log(data);
      }, 1000);
    }
  };
}

const infiniteScroll = InfiniteScroll();
infiniteScroll.fetchData();
infiniteScroll.fetchData();`},"README.md":{name:"README.md",language:"markdown",content:`# Infinite Scroll (Double-fetch race condition)

## Task

Implement an infinite scroll feature that fetches data without causing a double-fetch race condition.

## Constraints

* The feature must fetch data without causing a double-fetch race condition.
* The feature must be reusable.`}},constraints:["The feature must fetch data without causing a double-fetch race condition.","The feature must be reusable."],testCheck:"files.infiniteScroll.isLoading === false"},{id:"deep-clone-utility",title:"Deep Clone Utility (Recursive reference bug)",description:"The deep clone utility is causing a recursive reference bug, resulting in an infinite loop when trying to clone complex objects.",difficulty:"Easy",badgeColor:"bg-green-500/10 text-green-500 border border-green-500/20",files:{"index.js":{name:"index.js",language:"javascript",content:`const clone = function(obj) {
  return JSON.parse(JSON.stringify(obj));
}

const original = {
  a: 1,
  b: 2,
  c: {
    d: 3,
    e: 4
  }
};

const cloned = clone(original);
console.log(cloned);`},"README.md":{name:"README.md",language:"markdown",content:`# Deep Clone Utility (Recursive reference bug)

## Task

Implement a deep clone utility that clones complex objects without causing a recursive reference bug.

## Constraints

* The utility must clone complex objects without causing a recursive reference bug.
* The utility must be reusable.`}},constraints:["The utility must clone complex objects without causing a recursive reference bug.","The utility must be reusable."],testCheck:"JSON.stringify(files.original) === JSON.stringify(files.cloned)"},{id:"css-filter-service",title:"CSS Filter Service (Color parsing bug)",description:"The CSS filter service is causing a color parsing bug, resulting in incorrect color values being returned.",difficulty:"Medium",badgeColor:"bg-orange-500/10 text-orange-500 border border-orange-500/20",files:{"index.js":{name:"index.js",language:"javascript",content:`const parseColor = function(color) {
  return color.match(/^#([0-9a-f]{3}){1,2}$/i);
}

const color = '#123456';
console.log(parseColor(color));`},"README.md":{name:"README.md",language:"markdown",content:`# CSS Filter Service (Color parsing bug)

## Task

Implement a CSS filter service that parses color values without causing a color parsing bug.

## Constraints

* The service must parse color values without causing a color parsing bug.
* The service must be reusable.`}},constraints:["The service must parse color values without causing a color parsing bug.","The service must be reusable."],testCheck:"files.parseColor('#123456') === null"},{id:"log-parser",title:"Log Parser (Complex Regex challenge)",description:"The log parser is struggling with a complex regex challenge, resulting in incorrect log parsing.",difficulty:"Hard",badgeColor:"bg-blue-500/10 text-blue-500 border border-blue-500/20",files:{"index.js":{name:"index.js",language:"javascript",content:`const parseLog = function(log) {
  const regex = /^([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}) \\[(INFO|ERROR|WARNING)\\] (.*)$/;
  return log.match(regex);
}

const log = '2022-01-01 12:00:00 [INFO] This is a log message.';
console.log(parseLog(log));`},"README.md":{name:"README.md",language:"markdown",content:`# Log Parser (Complex Regex challenge)

## Task

Implement a log parser that parses log messages with a complex regex.

## Constraints

* The parser must parse log messages with the complex regex.
* The parser must be reusable.`}},constraints:["The parser must parse log messages with the complex regex.","The parser must be reusable."],testCheck:"files.parseLog('2022-01-01 12:00:00 [INFO] This is a log message.') !== null"}],Al=({title:e,value:t,icon:n,color:r})=>m.jsxs("div",{className:"glass-dark p-6 rounded-2xl shadow-xl border border-white/5 flex items-center justify-between group hover:border-white/20 transition-all duration-500",children:[m.jsxs("div",{children:[m.jsx("p",{className:"text-slate-400 text-xs font-bold uppercase tracking-wider mb-2",children:e}),m.jsx("h3",{className:`text-4xl font-bold tracking-tight ${r} neon-text-purple`,children:t})]}),m.jsx("div",{className:"p-4 rounded-xl bg-white/5 border border-white/10 group-hover:scale-110 transition-transform duration-500",children:m.jsx(n,{className:r,size:28})})]}),_2=({id:e,title:t,description:n,difficulty:r,badgeColor:s,isCompleted:i})=>{var u;const a=(u=Jn.find(c=>c.id===e))!=null&&u.files?Object.keys(Jn.find(c=>c.id===e).files).length:0;return m.jsxs("div",{className:`glass-dark p-8 rounded-3xl shadow-2xl border flex flex-col h-full group transition-all duration-500 relative overflow-hidden ${i?"border-green-500/30 shadow-green-500/5":"border-white/5 hover:border-purple-500/30"}`,children:[m.jsx("div",{className:`absolute -top-24 -right-24 w-48 h-48 blur-3xl rounded-full transition-colors duration-500 ${i?"bg-green-600/10":"bg-purple-600/10 group-hover:bg-purple-600/20"}`}),m.jsxs("div",{className:"flex justify-between items-start mb-6 relative z-10",children:[m.jsxs("div",{className:"flex flex-col",children:[m.jsx("h3",{className:"text-2xl font-bold text-white tracking-tight leading-tight",children:t}),i&&m.jsxs("span",{className:"text-green-400 text-[10px] font-black uppercase tracking-widest flex items-center gap-1 mt-1",children:[m.jsx(Yc,{size:10})," Completed"]})]}),m.jsx("span",{className:`px-4 py-1.5 rounded-full text-[10px] font-black uppercase tracking-widest ${s} neon-border-magenta shadow-lg`,children:r})]}),m.jsx("p",{className:"text-slate-400 text-sm mb-8 flex-grow leading-relaxed relative z-10",children:n}),m.jsxs("div",{className:"flex items-center gap-6 text-xs font-bold text-slate-500 mb-8 relative z-10 uppercase tracking-widest",children:[m.jsxs("div",{className:"flex items-center gap-2",children:[m.jsx("div",{className:"w-1.5 h-1.5 rounded-full bg-purple-500"}),m.jsxs("span",{children:[a," files"]})]}),m.jsxs("div",{className:"flex items-center gap-2",children:[m.jsx("div",{className:"w-1.5 h-1.5 rounded-full bg-magenta-500"}),m.jsx("span",{children:"System Scenario"})]})]}),m.jsx(xi,{to:`/editor/${e}`,className:`w-full font-bold py-4 rounded-2xl text-center transition-all duration-300 relative z-10 shadow-xl mt-auto ${i?"bg-green-500/10 text-green-400 border border-green-500/30 hover:bg-green-500/20":"bg-white text-[#030303] hover:bg-slate-200 hover:shadow-white/10"}`,children:i?"Review Code":"Launch Editor"})]})},C2=()=>{var R;const{user:e,signOut:t,isConfigured:n}=ci(),r=Qr(),[s,i]=N.useState(null),[a,u]=N.useState(!e),[c,f]=N.useState([]);N.useEffect(()=>{const j=JSON.parse(localStorage.getItem("completedProjects")||"[]");f(j),(async()=>{{u(!1);return}})()},[e]);const E=async()=>{await t(),r("/login")},w=Jn.length,_=c.length,A=w>0?_/w*100:0;return m.jsxs("div",{className:"relative min-h-screen bg-[#030303] overflow-hidden font-sans pb-20",children:[m.jsxs("div",{className:"bg-streaks",children:[m.jsx("div",{className:"streak-1 opacity-50"}),m.jsx("div",{className:"streak-2 opacity-50"})]}),m.jsxs("header",{className:"glass-dark border-b border-white/5 px-10 py-5 flex justify-between items-center sticky top-0 z-50 backdrop-blur-3xl",children:[m.jsxs("div",{className:"flex items-center gap-3 text-white",children:[m.jsx("div",{className:"p-2 glass rounded-xl neon-border-magenta",children:m.jsx(Qc,{size:24,className:"text-magenta-500"})}),m.jsx("h1",{className:"text-xl font-black tracking-tight uppercase",children:"CodeSandbox"})]}),m.jsxs("div",{className:"flex items-center gap-6",children:[m.jsxs("div",{className:"text-right",children:[m.jsx("p",{className:"text-white font-bold text-sm tracking-tight",children:a?"...":(s==null?void 0:s.username)||((R=e==null?void 0:e.email)==null?void 0:R.split("@")[0])||"Guest User"}),m.jsxs("div",{className:"flex items-center justify-end gap-2",children:[(s==null?void 0:s.full_name)&&m.jsx("p",{className:"text-slate-500 text-[10px] uppercase font-black tracking-widest",children:s.full_name}),!e&&!a&&m.jsx("p",{className:"text-purple-500 text-[10px] uppercase font-black tracking-widest border border-purple-500/30 px-2 py-0.5 rounded-full",children:"Dev Mode"})]})]}),m.jsx("button",{onClick:E,className:"p-3 glass rounded-xl text-red-500 hover:text-red-400 hover:bg-white/5 transition-all duration-300",children:m.jsx(c2,{size:20})})]})]}),!n&&m.jsxs("div",{className:"bg-purple-600 text-white px-10 py-2.5 flex items-center justify-center gap-4 text-xs font-black uppercase tracking-widest relative z-40",children:[m.jsx(di,{size:14}),m.jsx("span",{children:"Supabase is not configured. Some features may be offline."}),m.jsx(xi,{to:"/login",className:"underline hover:text-purple-200",children:"Connect Now"})]}),m.jsxs("main",{className:"max-w-7xl mx-auto px-10 py-12 space-y-12 relative z-10",children:[m.jsxs("div",{className:"grid grid-cols-1 md:grid-cols-3 gap-8",children:[m.jsx(Al,{title:"Projects Available",value:String(w).padStart(2,"0"),icon:a2,color:"text-white"}),m.jsx(Al,{title:"Submissions",value:String(_).padStart(2,"0"),icon:v2,color:"text-magenta-400"}),m.jsx(Al,{title:"Avg Score",value:`${Math.round(A)}%`,icon:Yc,color:"text-purple-400"})]}),m.jsxs("div",{children:[m.jsxs("div",{className:"flex items-center gap-4 mb-10",children:[m.jsx("h2",{className:"text-4xl font-bold text-white tracking-tight",children:"Available Projects"}),m.jsx("div",{className:"h-px flex-grow bg-white/10"})]}),m.jsx("div",{className:"grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-10",children:Jn.map(j=>m.jsx(_2,{id:j.id,title:j.title,description:j.description,difficulty:j.difficulty,badgeColor:j.badgeColor,isCompleted:c.includes(j.id)},j.id))})]})]})]})};function ep(e,t){(t==null||t>e.length)&&(t=e.length);for(var n=0,r=Array(t);n<t;n++)r[n]=e[n];return r}function I2(e){if(Array.isArray(e))return e}function T2(e,t,n){return(t=D2(t))in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function N2(e,t){var n=e==null?null:typeof Symbol<"u"&&e[Symbol.iterator]||e["@@iterator"];if(n!=null){var r,s,i,a,u=[],c=!0,f=!1;try{if(i=(n=n.call(e)).next,t!==0)for(;!(c=(r=i.call(n)).done)&&(u.push(r.value),u.length!==t);c=!0);}catch(E){f=!0,s=E}finally{try{if(!c&&n.return!=null&&(a=n.return(),Object(a)!==a))return}finally{if(f)throw s}}return u}}function j2(){throw new TypeError(`Invalid attempt to destructure non-iterable instance.
In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`)}function tp(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter(function(s){return Object.getOwnPropertyDescriptor(e,s).enumerable})),n.push.apply(n,r)}return n}function np(e){for(var t=1;t<arguments.length;t++){var n=arguments[t]!=null?arguments[t]:{};t%2?tp(Object(n),!0).forEach(function(r){T2(e,r,n[r])}):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):tp(Object(n)).forEach(function(r){Object.defineProperty(e,r,Object.getOwnPropertyDescriptor(n,r))})}return e}function A2(e,t){if(e==null)return{};var n,r,s=R2(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(r=0;r<i.length;r++)n=i[r],t.indexOf(n)===-1&&{}.propertyIsEnumerable.call(e,n)&&(s[n]=e[n])}return s}function R2(e,t){if(e==null)return{};var n={};for(var r in e)if({}.hasOwnProperty.call(e,r)){if(t.indexOf(r)!==-1)continue;n[r]=e[r]}return n}function P2(e,t){return I2(e)||N2(e,t)||M2(e,t)||j2()}function O2(e,t){if(typeof e!="object"||!e)return e;var n=e[Symbol.toPrimitive];if(n!==void 0){var r=n.call(e,t);if(typeof r!="object")return r;throw new TypeError("@@toPrimitive must return a primitive value.")}return(t==="string"?String:Number)(e)}function D2(e){var t=O2(e,"string");return typeof t=="symbol"?t:t+""}function M2(e,t){if(e){if(typeof e=="string")return ep(e,t);var n={}.toString.call(e).slice(8,-1);return n==="Object"&&e.constructor&&(n=e.constructor.name),n==="Map"||n==="Set"?Array.from(e):n==="Arguments"||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)?ep(e,t):void 0}}function L2(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function rp(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter(function(s){return Object.getOwnPropertyDescriptor(e,s).enumerable})),n.push.apply(n,r)}return n}function sp(e){for(var t=1;t<arguments.length;t++){var n=arguments[t]!=null?arguments[t]:{};t%2?rp(Object(n),!0).forEach(function(r){L2(e,r,n[r])}):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):rp(Object(n)).forEach(function(r){Object.defineProperty(e,r,Object.getOwnPropertyDescriptor(n,r))})}return e}function U2(){for(var e=arguments.length,t=new Array(e),n=0;n<e;n++)t[n]=arguments[n];return function(r){return t.reduceRight(function(s,i){return i(s)},r)}}function Is(e){return function t(){for(var n=this,r=arguments.length,s=new Array(r),i=0;i<r;i++)s[i]=arguments[i];return s.length>=e.length?e.apply(this,s):function(){for(var a=arguments.length,u=new Array(a),c=0;c<a;c++)u[c]=arguments[c];return t.apply(n,[].concat(s,u))}}}function ta(e){return{}.toString.call(e).includes("Object")}function B2(e){return!Object.keys(e).length}function hi(e){return typeof e=="function"}function F2(e,t){return Object.prototype.hasOwnProperty.call(e,t)}function z2(e,t){return ta(t)||bn("changeType"),Object.keys(t).some(function(n){return!F2(e,n)})&&bn("changeField"),t}function V2(e){hi(e)||bn("selectorType")}function $2(e){hi(e)||ta(e)||bn("handlerType"),ta(e)&&Object.values(e).some(function(t){return!hi(t)})&&bn("handlersType")}function H2(e){e||bn("initialIsRequired"),ta(e)||bn("initialType"),B2(e)&&bn("initialContent")}function G2(e,t){throw new Error(e[t]||e.default)}var W2={initialIsRequired:"initial state is required",initialType:"initial state should be an object",initialContent:"initial state shouldn't be an empty object",handlerType:"handler should be an object or a function",handlersType:"all handlers should be a functions",selectorType:"selector should be a function",changeType:"provided value of changes should be an object",changeField:'it seams you want to change a field in the state which is not specified in the "initial" state',default:"an unknown error accured in `state-local` package"},bn=Is(G2)(W2),no={changes:z2,selector:V2,handler:$2,initial:H2};function q2(e){var t=arguments.length>1&&arguments[1]!==void 0?arguments[1]:{};no.initial(e),no.handler(t);var n={current:e},r=Is(J2)(n,t),s=Is(X2)(n),i=Is(no.changes)(e),a=Is(K2)(n);function u(){var f=arguments.length>0&&arguments[0]!==void 0?arguments[0]:function(E){return E};return no.selector(f),f(n.current)}function c(f){U2(r,s,i,a)(f)}return[u,c]}function K2(e,t){return hi(t)?t(e.current):t}function X2(e,t){return e.current=sp(sp({},e.current),t),t}function J2(e,t,n){return hi(t)?t(e.current):Object.keys(n).forEach(function(r){var s;return(s=t[r])===null||s===void 0?void 0:s.call(t,e.current[r])}),n}var Y2={create:q2},Q2={paths:{vs:"https://cdn.jsdelivr.net/npm/monaco-editor@0.55.1/min/vs"}};function Z2(e){return function t(){for(var n=this,r=arguments.length,s=new Array(r),i=0;i<r;i++)s[i]=arguments[i];return s.length>=e.length?e.apply(this,s):function(){for(var a=arguments.length,u=new Array(a),c=0;c<a;c++)u[c]=arguments[c];return t.apply(n,[].concat(s,u))}}}function ek(e){return{}.toString.call(e).includes("Object")}function tk(e){return e||ip("configIsRequired"),ek(e)||ip("configType"),e.urls?(nk(),{paths:{vs:e.urls.monacoBase}}):e}function nk(){console.warn(py.deprecation)}function rk(e,t){throw new Error(e[t]||e.default)}var py={configIsRequired:"the configuration object is required",configType:"the configuration object should be an object",default:"an unknown error accured in `@monaco-editor/loader` package",deprecation:`Deprecation warning!
    You are using deprecated way of configuration.

    Instead of using
      monaco.config({ urls: { monacoBase: '...' } })
    use
      monaco.config({ paths: { vs: '...' } })

    For more please check the link https://github.com/suren-atoyan/monaco-loader#config
  `},ip=Z2(rk)(py),sk={config:tk},ik=function(){for(var t=arguments.length,n=new Array(t),r=0;r<t;r++)n[r]=arguments[r];return function(s){return n.reduceRight(function(i,a){return a(i)},s)}};function my(e,t){return Object.keys(t).forEach(function(n){t[n]instanceof Object&&e[n]&&Object.assign(t[n],my(e[n],t[n]))}),np(np({},e),t)}var ok={type:"cancelation",msg:"operation is manually canceled"};function Rl(e){var t=!1,n=new Promise(function(r,s){e.then(function(i){return t?s(ok):r(i)}),e.catch(s)});return n.cancel=function(){return t=!0},n}var ak=["monaco"],lk=Y2.create({config:Q2,isInitialized:!1,resolve:null,reject:null,monaco:null}),gy=P2(lk,2),_i=gy[0],Ea=gy[1];function uk(e){var t=sk.config(e),n=t.monaco,r=A2(t,ak);Ea(function(s){return{config:my(s.config,r),monaco:n}})}function ck(){var e=_i(function(t){var n=t.monaco,r=t.isInitialized,s=t.resolve;return{monaco:n,isInitialized:r,resolve:s}});if(!e.isInitialized){if(Ea({isInitialized:!0}),e.monaco)return e.resolve(e.monaco),Rl(Pl);if(window.monaco&&window.monaco.editor)return yy(window.monaco),e.resolve(window.monaco),Rl(Pl);ik(dk,hk)(pk)}return Rl(Pl)}function dk(e){return document.body.appendChild(e)}function fk(e){var t=document.createElement("script");return e&&(t.src=e),t}function hk(e){var t=_i(function(r){var s=r.config,i=r.reject;return{config:s,reject:i}}),n=fk("".concat(t.config.paths.vs,"/loader.js"));return n.onload=function(){return e()},n.onerror=t.reject,n}function pk(){var e=_i(function(n){var r=n.config,s=n.resolve,i=n.reject;return{config:r,resolve:s,reject:i}}),t=window.require;t.config(e.config),t(["vs/editor/editor.main"],function(n){var r=n.m||n;yy(r),e.resolve(r)},function(n){e.reject(n)})}function yy(e){_i().monaco||Ea({monaco:e})}function mk(){return _i(function(e){var t=e.monaco;return t})}var Pl=new Promise(function(e,t){return Ea({resolve:e,reject:t})}),vy={config:uk,init:ck,__getMonacoInstance:mk},gk={wrapper:{display:"flex",position:"relative",textAlign:"initial"},fullWidth:{width:"100%"},hide:{display:"none"}},Ol=gk,yk={container:{display:"flex",height:"100%",width:"100%",justifyContent:"center",alignItems:"center"}},vk=yk;function wk({children:e}){return fn.createElement("div",{style:vk.container},e)}var xk=wk,Sk=xk;function Ek({width:e,height:t,isEditorReady:n,loading:r,_ref:s,className:i,wrapperProps:a}){return fn.createElement("section",{style:{...Ol.wrapper,width:e,height:t},...a},!n&&fn.createElement(Sk,null,r),fn.createElement("div",{ref:s,style:{...Ol.fullWidth,...!n&&Ol.hide},className:i}))}var kk=Ek,wy=N.memo(kk);function bk(e){N.useEffect(e,[])}var xy=bk;function _k(e,t,n=!0){let r=N.useRef(!0);N.useEffect(r.current||!n?()=>{r.current=!1}:e,t)}var Ze=_k;function Vs(){}function _r(e,t,n,r){return Ck(e,r)||Ik(e,t,n,r)}function Ck(e,t){return e.editor.getModel(Sy(e,t))}function Ik(e,t,n,r){return e.editor.createModel(t,n,r?Sy(e,r):void 0)}function Sy(e,t){return e.Uri.parse(t)}function Tk({original:e,modified:t,language:n,originalLanguage:r,modifiedLanguage:s,originalModelPath:i,modifiedModelPath:a,keepCurrentOriginalModel:u=!1,keepCurrentModifiedModel:c=!1,theme:f="light",loading:E="Loading...",options:w={},height:_="100%",width:A="100%",className:R,wrapperProps:j={},beforeMount:D=Vs,onMount:b=Vs}){let[x,C]=N.useState(!1),[O,M]=N.useState(!0),U=N.useRef(null),g=N.useRef(null),p=N.useRef(null),v=N.useRef(b),S=N.useRef(D),k=N.useRef(!1);xy(()=>{let $=vy.init();return $.then(te=>(g.current=te)&&M(!1)).catch(te=>(te==null?void 0:te.type)!=="cancelation"&&console.error("Monaco initialization: error:",te)),()=>U.current?Y():$.cancel()}),Ze(()=>{if(U.current&&g.current){let $=U.current.getOriginalEditor(),te=_r(g.current,e||"",r||n||"text",i||"");te!==$.getModel()&&$.setModel(te)}},[i],x),Ze(()=>{if(U.current&&g.current){let $=U.current.getModifiedEditor(),te=_r(g.current,t||"",s||n||"text",a||"");te!==$.getModel()&&$.setModel(te)}},[a],x),Ze(()=>{let $=U.current.getModifiedEditor();$.getOption(g.current.editor.EditorOption.readOnly)?$.setValue(t||""):t!==$.getValue()&&($.executeEdits("",[{range:$.getModel().getFullModelRange(),text:t||"",forceMoveMarkers:!0}]),$.pushUndoStop())},[t],x),Ze(()=>{var $,te;(te=($=U.current)==null?void 0:$.getModel())==null||te.original.setValue(e||"")},[e],x),Ze(()=>{let{original:$,modified:te}=U.current.getModel();g.current.editor.setModelLanguage($,r||n||"text"),g.current.editor.setModelLanguage(te,s||n||"text")},[n,r,s],x),Ze(()=>{var $;($=g.current)==null||$.editor.setTheme(f)},[f],x),Ze(()=>{var $;($=U.current)==null||$.updateOptions(w)},[w],x);let I=N.useCallback(()=>{var He;if(!g.current)return;S.current(g.current);let $=_r(g.current,e||"",r||n||"text",i||""),te=_r(g.current,t||"",s||n||"text",a||"");(He=U.current)==null||He.setModel({original:$,modified:te})},[n,t,s,e,r,i,a]),y=N.useCallback(()=>{var $;!k.current&&p.current&&(U.current=g.current.editor.createDiffEditor(p.current,{automaticLayout:!0,...w}),I(),($=g.current)==null||$.editor.setTheme(f),C(!0),k.current=!0)},[w,f,I]);N.useEffect(()=>{x&&v.current(U.current,g.current)},[x]),N.useEffect(()=>{!O&&!x&&y()},[O,x,y]);function Y(){var te,He,F,H;let $=(te=U.current)==null?void 0:te.getModel();u||((He=$==null?void 0:$.original)==null||He.dispose()),c||((F=$==null?void 0:$.modified)==null||F.dispose()),(H=U.current)==null||H.dispose()}return fn.createElement(wy,{width:A,height:_,isEditorReady:x,loading:E,_ref:p,className:R,wrapperProps:j})}var Nk=Tk;N.memo(Nk);function jk(e){let t=N.useRef();return N.useEffect(()=>{t.current=e},[e]),t.current}var Ak=jk,ro=new Map;function Rk({defaultValue:e,defaultLanguage:t,defaultPath:n,value:r,language:s,path:i,theme:a="light",line:u,loading:c="Loading...",options:f={},overrideServices:E={},saveViewState:w=!0,keepCurrentModel:_=!1,width:A="100%",height:R="100%",className:j,wrapperProps:D={},beforeMount:b=Vs,onMount:x=Vs,onChange:C,onValidate:O=Vs}){let[M,U]=N.useState(!1),[g,p]=N.useState(!0),v=N.useRef(null),S=N.useRef(null),k=N.useRef(null),I=N.useRef(x),y=N.useRef(b),Y=N.useRef(),$=N.useRef(r),te=Ak(i),He=N.useRef(!1),F=N.useRef(!1);xy(()=>{let V=vy.init();return V.then(G=>(v.current=G)&&p(!1)).catch(G=>(G==null?void 0:G.type)!=="cancelation"&&console.error("Monaco initialization: error:",G)),()=>S.current?W():V.cancel()}),Ze(()=>{var G,ge,_e,Ge;let V=_r(v.current,e||r||"",t||s||"",i||n||"");V!==((G=S.current)==null?void 0:G.getModel())&&(w&&ro.set(te,(ge=S.current)==null?void 0:ge.saveViewState()),(_e=S.current)==null||_e.setModel(V),w&&((Ge=S.current)==null||Ge.restoreViewState(ro.get(i))))},[i],M),Ze(()=>{var V;(V=S.current)==null||V.updateOptions(f)},[f],M),Ze(()=>{!S.current||r===void 0||(S.current.getOption(v.current.editor.EditorOption.readOnly)?S.current.setValue(r):r!==S.current.getValue()&&(F.current=!0,S.current.executeEdits("",[{range:S.current.getModel().getFullModelRange(),text:r,forceMoveMarkers:!0}]),S.current.pushUndoStop(),F.current=!1))},[r],M),Ze(()=>{var G,ge;let V=(G=S.current)==null?void 0:G.getModel();V&&s&&((ge=v.current)==null||ge.editor.setModelLanguage(V,s))},[s],M),Ze(()=>{var V;u!==void 0&&((V=S.current)==null||V.revealLine(u))},[u],M),Ze(()=>{var V;(V=v.current)==null||V.editor.setTheme(a)},[a],M);let H=N.useCallback(()=>{var V;if(!(!k.current||!v.current)&&!He.current){y.current(v.current);let G=i||n,ge=_r(v.current,r||e||"",t||s||"",G||"");S.current=(V=v.current)==null?void 0:V.editor.create(k.current,{model:ge,automaticLayout:!0,...f},E),w&&S.current.restoreViewState(ro.get(G)),v.current.editor.setTheme(a),u!==void 0&&S.current.revealLine(u),U(!0),He.current=!0}},[e,t,n,r,s,i,f,E,w,a,u]);N.useEffect(()=>{M&&I.current(S.current,v.current)},[M]),N.useEffect(()=>{!g&&!M&&H()},[g,M,H]),$.current=r,N.useEffect(()=>{var V,G;M&&C&&((V=Y.current)==null||V.dispose(),Y.current=(G=S.current)==null?void 0:G.onDidChangeModelContent(ge=>{F.current||C(S.current.getValue(),ge)}))},[M,C]),N.useEffect(()=>{if(M){let V=v.current.editor.onDidChangeMarkers(G=>{var _e;let ge=(_e=S.current.getModel())==null?void 0:_e.uri;if(ge&&G.find(Ge=>Ge.path===ge.path)){let Ge=v.current.editor.getModelMarkers({resource:ge});O==null||O(Ge)}});return()=>{V==null||V.dispose()}}return()=>{}},[M,O]);function W(){var V,G;(V=Y.current)==null||V.dispose(),_?w&&ro.set(i,S.current.saveViewState()):(G=S.current.getModel())==null||G.dispose(),S.current.dispose()}return fn.createElement(wy,{width:A,height:R,isEditorReady:M,loading:c,_ref:k,className:j,wrapperProps:D})}var Pk=Rk,Ok=N.memo(Pk),Dk=Ok;const fr="0.37.0";let op=!1,$s,Ey,ky,Bu,by,_y,Cy,Iy,Ty;function Mk(e,t={auto:!1}){if(op)throw new Error(`you must \`import 'groq-sdk/shims/${e.kind}'\` before importing anything else from groq-sdk`);if($s)throw new Error(`can't \`import 'groq-sdk/shims/${e.kind}'\` after \`import 'groq-sdk/shims/${$s}'\``);op=t.auto,$s=e.kind,Ey=e.fetch,ky=e.FormData,Bu=e.File,by=e.ReadableStream,_y=e.getMultipartRequestOptions,Cy=e.getDefaultAgent,Iy=e.fileFromPath,Ty=e.isFsReadStream}class Lk{constructor(t){this.body=t}get[Symbol.toStringTag](){return"MultipartBody"}}function Uk({manuallyImported:e}={}){const t=e?"You may need to use polyfills":"Add one of these imports before your first `import … from 'groq-sdk'`:\n- `import 'groq-sdk/shims/node'` (if you're running on Node)\n- `import 'groq-sdk/shims/web'` (otherwise)\n";let n,r,s,i;try{n=fetch,r=Request,s=Response,i=Headers}catch(a){throw new Error(`this environment is missing the following Web Fetch API type: ${a.message}. ${t}`)}return{kind:"web",fetch:n,Request:r,Response:s,Headers:i,FormData:typeof FormData<"u"?FormData:class{constructor(){throw new Error(`file uploads aren't supported in this environment yet as 'FormData' is undefined. ${t}`)}},Blob:typeof Blob<"u"?Blob:class{constructor(){throw new Error(`file uploads aren't supported in this environment yet as 'Blob' is undefined. ${t}`)}},File:typeof File<"u"?File:class{constructor(){throw new Error(`file uploads aren't supported in this environment yet as 'File' is undefined. ${t}`)}},ReadableStream:typeof ReadableStream<"u"?ReadableStream:class{constructor(){throw new Error(`streaming isn't supported in this environment yet as 'ReadableStream' is undefined. ${t}`)}},getMultipartRequestOptions:async(a,u)=>({...u,body:new Lk(a)}),getDefaultAgent:a=>{},fileFromPath:()=>{throw new Error("The `fileFromPath` function is only supported in Node. See the README for more details: https://www.github.com/groq/groq-typescript#file-uploads")},isFsReadStream:a=>!1}}const Ny=()=>{$s||Mk(Uk(),{auto:!0})};Ny();class St extends Error{}class Le extends St{constructor(t,n,r,s){super(`${Le.makeMessage(t,n,r)}`),this.status=t,this.headers=s,this.error=n}static makeMessage(t,n,r){const s=n!=null&&n.message?typeof n.message=="string"?n.message:JSON.stringify(n.message):n?JSON.stringify(n):r;return t&&s?`${t} ${s}`:t?`${t} status code (no body)`:s||"(no status code or body)"}static generate(t,n,r,s){if(!t||!s)return new ka({message:r,cause:Vu(n)});const i=n;return t===400?new Ay(t,i,r,s):t===401?new Ry(t,i,r,s):t===403?new Py(t,i,r,s):t===404?new Oy(t,i,r,s):t===409?new Dy(t,i,r,s):t===422?new My(t,i,r,s):t===429?new Ly(t,i,r,s):t>=500?new Uy(t,i,r,s):new Le(t,i,r,s)}}class Fu extends Le{constructor({message:t}={}){super(void 0,void 0,t||"Request was aborted.",void 0)}}class ka extends Le{constructor({message:t,cause:n}){super(void 0,void 0,t||"Connection error.",void 0),n&&(this.cause=n)}}class jy extends ka{constructor({message:t}={}){super({message:t??"Request timed out."})}}class Ay extends Le{}class Ry extends Le{}class Py extends Le{}class Oy extends Le{}class Dy extends Le{}class My extends Le{}class Ly extends Le{}class Uy extends Le{}class Cr{constructor(t,n){this.iterator=t,this.controller=n}static fromSSEResponse(t,n){let r=!1;const s=new Bk;async function*i(){if(!t.body)throw n.abort(),new St("Attempted to iterate over a response with no body");const u=new rr,c=ap(t.body);for await(const f of c)for(const E of u.decode(f)){const w=s.decode(E);w&&(yield w)}for(const f of u.flush()){const E=s.decode(f);E&&(yield E)}}async function*a(){if(r)throw new Error("Cannot iterate over a consumed stream, use `.tee()` to split the stream.");r=!0;let u=!1;try{for await(const c of i())if(!u){if(c.data.startsWith("[DONE]")){u=!0;continue}if(c.event===null||c.event==="error"){let f;try{f=JSON.parse(c.data)}catch(E){throw console.error("Could not parse message into JSON:",c.data),console.error("From chunk:",c.raw),E}if(f&&f.error)throw new Le(f.error.status_code,f.error,f.error.message,void 0);yield f}}u=!0}catch(c){if(c instanceof Error&&c.name==="AbortError")return;throw c}finally{u||n.abort()}}return new Cr(a,n)}static fromReadableStream(t,n){let r=!1;async function*s(){const a=new rr,u=ap(t);for await(const c of u)for(const f of a.decode(c))yield f;for(const c of a.flush())yield c}async function*i(){if(r)throw new Error("Cannot iterate over a consumed stream, use `.tee()` to split the stream.");r=!0;let a=!1;try{for await(const u of s())a||u&&(yield JSON.parse(u));a=!0}catch(u){if(u instanceof Error&&u.name==="AbortError")return;throw u}finally{a||n.abort()}}return new Cr(i,n)}[Symbol.asyncIterator](){return this.iterator()}tee(){const t=[],n=[],r=this.iterator(),s=i=>({next:()=>{if(i.length===0){const a=r.next();t.push(a),n.push(a)}return i.shift()}});return[new Cr(()=>s(t),this.controller),new Cr(()=>s(n),this.controller)]}toReadableStream(){const t=this;let n;const r=new TextEncoder;return new by({async start(){n=t[Symbol.asyncIterator]()},async pull(s){try{const{value:i,done:a}=await n.next();if(a)return s.close();const u=r.encode(JSON.stringify(i)+`
`);s.enqueue(u)}catch(i){s.error(i)}},async cancel(){var s;await((s=n.return)==null?void 0:s.call(n))}})}}class Bk{constructor(){this.event=null,this.data=[],this.chunks=[]}decode(t){if(t.endsWith("\r")&&(t=t.substring(0,t.length-1)),!t){if(!this.event&&!this.data.length)return null;const i={event:this.event,data:this.data.join(`
`),raw:this.chunks};return this.event=null,this.data=[],this.chunks=[],i}if(this.chunks.push(t),t.startsWith(":"))return null;let[n,r,s]=Fk(t,":");return s.startsWith(" ")&&(s=s.substring(1)),n==="event"?this.event=s:n==="data"&&this.data.push(s),null}}class rr{constructor(){this.buffer=[],this.trailingCR=!1}decode(t){let n=this.decodeText(t);if(this.trailingCR&&(n="\r"+n,this.trailingCR=!1),n.endsWith("\r")&&(this.trailingCR=!0,n=n.slice(0,-1)),!n)return[];const r=rr.NEWLINE_CHARS.has(n[n.length-1]||"");let s=n.split(rr.NEWLINE_REGEXP);return s.length===1&&!r?(this.buffer.push(s[0]),[]):(this.buffer.length>0&&(s=[this.buffer.join("")+s[0],...s.slice(1)],this.buffer=[]),r||(this.buffer=[s.pop()||""]),s)}decodeText(t){if(t==null)return"";if(typeof t=="string")return t;if(typeof Buffer<"u"){if(t instanceof Buffer)return t.toString();if(t instanceof Uint8Array)return Buffer.from(t).toString();throw new St(`Unexpected: received non-Uint8Array (${t.constructor.name}) stream chunk in an environment with a global "Buffer" defined, which this library assumes to be Node. Please report this error.`)}if(typeof TextDecoder<"u"){if(t instanceof Uint8Array||t instanceof ArrayBuffer)return this.textDecoder??(this.textDecoder=new TextDecoder("utf8")),this.textDecoder.decode(t);throw new St(`Unexpected: received non-Uint8Array/ArrayBuffer (${t.constructor.name}) in a web platform. Please report this error.`)}throw new St("Unexpected: neither Buffer nor TextDecoder are available as globals. Please report this error.")}flush(){if(!this.buffer.length&&!this.trailingCR)return[];const t=[this.buffer.join("")];return this.buffer=[],this.trailingCR=!1,t}}rr.NEWLINE_CHARS=new Set([`
`,"\r","\v","\f","","","","","\u2028","\u2029"]);rr.NEWLINE_REGEXP=/\r\n|[\n\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029]/g;function Fk(e,t){const n=e.indexOf(t);return n!==-1?[e.substring(0,n),t,e.substring(n+t.length)]:[e,"",""]}function ap(e){if(e[Symbol.asyncIterator])return e;const t=e.getReader();return{async next(){try{const n=await t.read();return n!=null&&n.done&&t.releaseLock(),n}catch(n){throw t.releaseLock(),n}},async return(){const n=t.cancel();return t.releaseLock(),await n,{done:!0,value:void 0}},[Symbol.asyncIterator](){return this}}}const By=e=>e!=null&&typeof e=="object"&&typeof e.url=="string"&&typeof e.blob=="function",Fy=e=>e!=null&&typeof e=="object"&&typeof e.name=="string"&&typeof e.lastModified=="number"&&ba(e),ba=e=>e!=null&&typeof e=="object"&&typeof e.size=="number"&&typeof e.type=="string"&&typeof e.text=="function"&&typeof e.slice=="function"&&typeof e.arrayBuffer=="function",zk=e=>Fy(e)||By(e)||Ty(e);async function zy(e,t,n){var s;if(e=await e,Fy(e))return e;if(By(e)){const i=await e.blob();t||(t=new URL(e.url).pathname.split(/[\\/]/).pop()??"unknown_file");const a=ba(i)?[await i.arrayBuffer()]:[i];return new Bu(a,t,n)}const r=await Vk(e);if(t||(t=Hk(e)??"unknown_file"),!(n!=null&&n.type)){const i=(s=r[0])==null?void 0:s.type;typeof i=="string"&&(n={...n,type:i})}return new Bu(r,t,n)}async function Vk(e){var n;let t=[];if(typeof e=="string"||ArrayBuffer.isView(e)||e instanceof ArrayBuffer)t.push(e);else if(ba(e))t.push(await e.arrayBuffer());else if(Gk(e))for await(const r of e)t.push(r);else throw new Error(`Unexpected data type: ${typeof e}; constructor: ${(n=e==null?void 0:e.constructor)==null?void 0:n.name}; props: ${$k(e)}`);return t}function $k(e){return`[${Object.getOwnPropertyNames(e).map(n=>`"${n}"`).join(", ")}]`}function Hk(e){var t;return Dl(e.name)||Dl(e.filename)||((t=Dl(e.path))==null?void 0:t.split(/[\\/]/).pop())}const Dl=e=>{if(typeof e=="string")return e;if(typeof Buffer<"u"&&e instanceof Buffer)return String(e)},Gk=e=>e!=null&&typeof e=="object"&&typeof e[Symbol.asyncIterator]=="function",lp=e=>e&&typeof e=="object"&&e.body&&e[Symbol.toStringTag]==="MultipartBody",Zc=async e=>{const t=await Wk(e.body);return _y(t,e)},Wk=async e=>{const t=new ky;return await Promise.all(Object.entries(e||{}).map(([n,r])=>zu(t,n,r))),t},zu=async(e,t,n)=>{if(n!==void 0){if(n==null)throw new TypeError(`Received null for "${t}"; to pass null in FormData, you must use the string 'null'`);if(typeof n=="string"||typeof n=="number"||typeof n=="boolean")e.append(t,String(n));else if(zk(n)){const r=await zy(n);e.append(t,r)}else if(Array.isArray(n))await Promise.all(n.map(r=>zu(e,t+"[]",r)));else if(typeof n=="object")await Promise.all(Object.entries(n).map(([r,s])=>zu(e,`${t}[${r}]`,s)));else throw new TypeError(`Invalid value given to form, expected a string, number, boolean, object, Array, File or Blob but got ${n} instead`)}};var Mr={},qk=function(e,t,n,r,s){if(typeof t=="function"?e!==t||!0:!t.has(e))throw new TypeError("Cannot write private member to an object whose class did not declare it");return t.set(e,n),n},Kk=function(e,t,n,r){if(typeof t=="function"?e!==t||!0:!t.has(e))throw new TypeError("Cannot read private member from an object whose class did not declare it");return t.get(e)},Eo;Ny();async function Vy(e){var a;const{response:t}=e;if(e.options.stream)return Lr("response",t.status,t.url,t.headers,t.body),e.options.__streamClass?e.options.__streamClass.fromSSEResponse(t,e.controller):Cr.fromSSEResponse(t,e.controller);if(t.status===204)return null;if(e.options.__binaryResponse)return t;const n=t.headers.get("content-type"),r=(a=n==null?void 0:n.split(";")[0])==null?void 0:a.trim();if((r==null?void 0:r.includes("application/json"))||(r==null?void 0:r.endsWith("+json"))){const u=await t.json();return Lr("response",t.status,t.url,t.headers,u),u}const i=await t.text();return Lr("response",t.status,t.url,t.headers,i),i}class _a extends Promise{constructor(t,n=Vy){super(r=>{r(null)}),this.responsePromise=t,this.parseResponse=n}_thenUnwrap(t){return new _a(this.responsePromise,async n=>t(await this.parseResponse(n),n))}asResponse(){return this.responsePromise.then(t=>t.response)}async withResponse(){const[t,n]=await Promise.all([this.parse(),this.asResponse()]);return{data:t,response:n}}parse(){return this.parsedPromise||(this.parsedPromise=this.responsePromise.then(this.parseResponse)),this.parsedPromise}then(t,n){return this.parse().then(t,n)}catch(t){return this.parse().catch(t)}finally(t){return this.parse().finally(t)}}class Xk{constructor({baseURL:t,baseURLOverridden:n,maxRetries:r=2,timeout:s=6e4,httpAgent:i,fetch:a}){Eo.set(this,void 0),this.baseURL=t,qk(this,Eo,n),this.maxRetries=Ml("maxRetries",r),this.timeout=Ml("timeout",s),this.httpAgent=i,this.fetch=a??Ey}authHeaders(t){return{}}defaultHeaders(t){return{Accept:"application/json",...["head","get"].includes(t.method)?{}:{"Content-Type":"application/json"},"User-Agent":this.getUserAgent(),...eb(),...this.authHeaders(t)}}validateHeaders(t,n){}defaultIdempotencyKey(){return`stainless-node-retry-${ab()}`}get(t,n){return this.methodRequest("get",t,n)}post(t,n){return this.methodRequest("post",t,n)}patch(t,n){return this.methodRequest("patch",t,n)}put(t,n){return this.methodRequest("put",t,n)}delete(t,n){return this.methodRequest("delete",t,n)}methodRequest(t,n,r){return this.request(Promise.resolve(r).then(async s=>{const i=s&&ba(s==null?void 0:s.body)?new DataView(await s.body.arrayBuffer()):(s==null?void 0:s.body)instanceof DataView?s.body:(s==null?void 0:s.body)instanceof ArrayBuffer?new DataView(s.body):s&&ArrayBuffer.isView(s==null?void 0:s.body)?new DataView(s.body.buffer):s==null?void 0:s.body;return{method:t,path:n,...s,body:i}}))}getAPIList(t,n,r){return this.requestAPIList(n,{method:"get",path:t,...r})}calculateContentLength(t){if(typeof t=="string"){if(typeof Buffer<"u")return Buffer.byteLength(t,"utf8").toString();if(typeof TextEncoder<"u")return new TextEncoder().encode(t).length.toString()}else if(ArrayBuffer.isView(t))return t.byteLength.toString();return null}async buildRequest(t,{retryCount:n=0}={}){var D;const r={...t},{method:s,path:i,query:a,defaultBaseURL:u,headers:c={}}=r,f=ArrayBuffer.isView(r.body)||r.__binaryRequest&&typeof r.body=="string"?r.body:lp(r.body)?r.body.body:r.body?JSON.stringify(r.body,null,2):null,E=this.calculateContentLength(f),w=this.buildURL(i,a,u);"timeout"in r&&Ml("timeout",r.timeout),r.timeout=r.timeout??this.timeout;const _=r.httpAgent??this.httpAgent??Cy(w),A=r.timeout+1e3;typeof((D=_==null?void 0:_.options)==null?void 0:D.timeout)=="number"&&A>(_.options.timeout??0)&&(_.options.timeout=A),this.idempotencyHeader&&s!=="get"&&(t.idempotencyKey||(t.idempotencyKey=this.defaultIdempotencyKey()),c[this.idempotencyHeader]=t.idempotencyKey);const R=this.buildHeaders({options:r,headers:c,contentLength:E,retryCount:n});return{req:{method:s,...f&&{body:f},headers:R,..._&&{agent:_},signal:r.signal??null},url:w,timeout:r.timeout}}buildHeaders({options:t,headers:n,contentLength:r,retryCount:s}){const i={};r&&(i["content-length"]=r);const a=this.defaultHeaders(t);return hp(i,a),hp(i,n),lp(t.body)&&$s!=="node"&&delete i["content-type"],so(a,"x-stainless-retry-count")===void 0&&so(n,"x-stainless-retry-count")===void 0&&(i["x-stainless-retry-count"]=String(s)),so(a,"x-stainless-timeout")===void 0&&so(n,"x-stainless-timeout")===void 0&&t.timeout&&(i["x-stainless-timeout"]=String(Math.trunc(t.timeout/1e3))),this.validateHeaders(i,n),i}async prepareOptions(t){}async prepareRequest(t,{url:n,options:r}){}parseHeaders(t){return t?Symbol.iterator in t?Object.fromEntries(Array.from(t).map(n=>[...n])):{...t}:{}}makeStatusError(t,n,r,s){return Le.generate(t,n,r,s)}request(t,n=null){return new _a(this.makeRequest(t,n))}async makeRequest(t,n){var w,_;const r=await t,s=r.maxRetries??this.maxRetries;n==null&&(n=s),await this.prepareOptions(r);const{req:i,url:a,timeout:u}=await this.buildRequest(r,{retryCount:s-n});if(await this.prepareRequest(i,{url:a,options:r}),Lr("request",a,r,i.headers),(w=r.signal)!=null&&w.aborted)throw new Fu;const c=new AbortController,f=await this.fetchWithTimeout(a,i,u,c).catch(Vu);if(f instanceof Error){if((_=r.signal)!=null&&_.aborted)throw new Fu;if(n)return this.retryRequest(r,n);throw f.name==="AbortError"?new jy:new ka({cause:f})}const E=Yk(f.headers);if(!f.ok){if(n&&this.shouldRetry(f)){const x=`retrying, ${n} attempts remaining`;return Lr(`response (error; ${x})`,f.status,a,E),this.retryRequest(r,n,E)}const A=await f.text().catch(x=>Vu(x).message),R=tb(A),j=R?void 0:A;throw Lr(`response (error; ${n?"(error; no more retries left)":"(error; not retryable)"})`,f.status,a,E,j),this.makeStatusError(f.status,R,j,E)}return{response:f,options:r,controller:c}}requestAPIList(t,n){const r=this.makeRequest(n,null);return new Jk(this,r,t)}buildURL(t,n,r){const s=!Kk(this,Eo)&&r||this.baseURL,i=rb(t)?new URL(t):new URL(s+(s.endsWith("/")&&t.startsWith("/")?t.slice(1):t)),a=this.defaultQuery();return ib(a)||(n={...a,...n}),typeof n=="object"&&n&&!Array.isArray(n)&&(i.search=this.stringifyQuery(n)),i.toString()}stringifyQuery(t){return Object.entries(t).filter(([n,r])=>typeof r<"u").map(([n,r])=>{if(typeof r=="string"||typeof r=="number"||typeof r=="boolean")return`${encodeURIComponent(n)}=${encodeURIComponent(r)}`;if(r===null)return`${encodeURIComponent(n)}=`;throw new St(`Cannot stringify type ${typeof r}; Expected string, number, boolean, or null. If you need to pass nested query parameters, you can manually encode them, e.g. { query: { 'foo[key1]': value1, 'foo[key2]': value2 } }, and please open a GitHub issue requesting better support for your use case.`)}).join("&")}async fetchWithTimeout(t,n,r,s){const{signal:i,...a}=n||{};i&&i.addEventListener("abort",()=>s.abort());const u=setTimeout(()=>s.abort(),r),c={signal:s.signal,...a};return c.method&&(c.method=c.method.toUpperCase()),this.fetch.call(void 0,t,c).finally(()=>{clearTimeout(u)})}shouldRetry(t){const n=t.headers.get("x-should-retry");return n==="true"?!0:n==="false"?!1:t.status===408||t.status===409||t.status===429||t.status>=500}async retryRequest(t,n,r){let s;const i=r==null?void 0:r["retry-after-ms"];if(i){const u=parseFloat(i);Number.isNaN(u)||(s=u)}const a=r==null?void 0:r["retry-after"];if(a&&!s){const u=parseFloat(a);Number.isNaN(u)?s=Date.parse(a)-Date.now():s=u*1e3}if(!(s&&0<=s&&s<60*1e3)){const u=t.maxRetries??this.maxRetries;s=this.calculateDefaultRetryTimeoutMillis(n,u)}return await sb(s),this.makeRequest(t,n-1)}calculateDefaultRetryTimeoutMillis(t,n){const i=n-t,a=Math.min(.5*Math.pow(2,i),8),u=1-Math.random()*.25;return a*u*1e3}getUserAgent(){return`${this.constructor.name}/JS ${fr}`}}Eo=new WeakMap;class Jk extends _a{constructor(t,n,r){super(n,async s=>new r(t,s.response,await Vy(s),s.options))}async*[Symbol.asyncIterator](){const t=await this;for await(const n of t)yield n}}const Yk=e=>new Proxy(Object.fromEntries(e.entries()),{get(t,n){const r=n.toString();return t[r.toLowerCase()]||t[r]}}),Qk=()=>{var t;if(typeof Deno<"u"&&Deno.build!=null)return{"X-Stainless-Lang":"js","X-Stainless-Package-Version":fr,"X-Stainless-OS":cp(Deno.build.os),"X-Stainless-Arch":up(Deno.build.arch),"X-Stainless-Runtime":"deno","X-Stainless-Runtime-Version":typeof Deno.version=="string"?Deno.version:((t=Deno.version)==null?void 0:t.deno)??"unknown"};if(typeof EdgeRuntime<"u")return{"X-Stainless-Lang":"js","X-Stainless-Package-Version":fr,"X-Stainless-OS":"Unknown","X-Stainless-Arch":`other:${EdgeRuntime}`,"X-Stainless-Runtime":"edge","X-Stainless-Runtime-Version":process.version};if(Object.prototype.toString.call(typeof process<"u"?process:0)==="[object process]")return{"X-Stainless-Lang":"js","X-Stainless-Package-Version":fr,"X-Stainless-OS":cp(process.platform),"X-Stainless-Arch":up(process.arch),"X-Stainless-Runtime":"node","X-Stainless-Runtime-Version":process.version};const e=Zk();return e?{"X-Stainless-Lang":"js","X-Stainless-Package-Version":fr,"X-Stainless-OS":"Unknown","X-Stainless-Arch":"unknown","X-Stainless-Runtime":`browser:${e.browser}`,"X-Stainless-Runtime-Version":e.version}:{"X-Stainless-Lang":"js","X-Stainless-Package-Version":fr,"X-Stainless-OS":"Unknown","X-Stainless-Arch":"unknown","X-Stainless-Runtime":"unknown","X-Stainless-Runtime-Version":"unknown"}};function Zk(){if(typeof navigator>"u"||!navigator)return null;const e=[{key:"edge",pattern:/Edge(?:\W+(\d+)\.(\d+)(?:\.(\d+))?)?/},{key:"ie",pattern:/MSIE(?:\W+(\d+)\.(\d+)(?:\.(\d+))?)?/},{key:"ie",pattern:/Trident(?:.*rv\:(\d+)\.(\d+)(?:\.(\d+))?)?/},{key:"chrome",pattern:/Chrome(?:\W+(\d+)\.(\d+)(?:\.(\d+))?)?/},{key:"firefox",pattern:/Firefox(?:\W+(\d+)\.(\d+)(?:\.(\d+))?)?/},{key:"safari",pattern:/(?:Version\W+(\d+)\.(\d+)(?:\.(\d+))?)?(?:\W+Mobile\S*)?\W+Safari/}];for(const{key:t,pattern:n}of e){const r=n.exec(navigator.userAgent);if(r){const s=r[1]||0,i=r[2]||0,a=r[3]||0;return{browser:t,version:`${s}.${i}.${a}`}}}return null}const up=e=>e==="x32"?"x32":e==="x86_64"||e==="x64"?"x64":e==="arm"?"arm":e==="aarch64"||e==="arm64"?"arm64":e?`other:${e}`:"unknown",cp=e=>(e=e.toLowerCase(),e.includes("ios")?"iOS":e==="android"?"Android":e==="darwin"?"MacOS":e==="win32"?"Windows":e==="freebsd"?"FreeBSD":e==="openbsd"?"OpenBSD":e==="linux"?"Linux":e?`Other:${e}`:"Unknown");let dp;const eb=()=>dp??(dp=Qk()),tb=e=>{try{return JSON.parse(e)}catch{return}},nb=/^[a-z][a-z0-9+.-]*:/i,rb=e=>nb.test(e),sb=e=>new Promise(t=>setTimeout(t,e)),Ml=(e,t)=>{if(typeof t!="number"||!Number.isInteger(t))throw new St(`${e} must be an integer`);if(t<0)throw new St(`${e} must be a positive integer`);return t},Vu=e=>{if(e instanceof Error)return e;if(typeof e=="object"&&e!==null)try{return new Error(JSON.stringify(e))}catch{}return new Error(e)},fp=e=>{var t,n,r,s;if(typeof process<"u")return((t=Mr==null?void 0:Mr[e])==null?void 0:t.trim())??void 0;if(typeof Deno<"u")return(s=(r=(n=Deno.env)==null?void 0:n.get)==null?void 0:r.call(n,e))==null?void 0:s.trim()};function ib(e){if(!e)return!0;for(const t in e)return!1;return!0}function ob(e,t){return Object.prototype.hasOwnProperty.call(e,t)}function hp(e,t){for(const n in t){if(!ob(t,n))continue;const r=n.toLowerCase();if(!r)continue;const s=t[n];s===null?delete e[r]:s!==void 0&&(e[r]=s)}}function Lr(e,...t){typeof process<"u"&&(Mr==null?void 0:Mr.DEBUG)==="true"&&console.log(`Groq:DEBUG:${e}`,...t)}const ab=()=>"xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g,e=>{const t=Math.random()*16|0;return(e==="x"?t:t&3|8).toString(16)}),lb=()=>typeof window<"u"&&typeof window.document<"u"&&typeof navigator<"u",ub=e=>typeof(e==null?void 0:e.get)=="function",so=(e,t)=>{var r;const n=t.toLowerCase();if(ub(e)){const s=((r=t[0])==null?void 0:r.toUpperCase())+t.substring(1).replace(/([^\w])(\w)/g,(i,a,u)=>a+u.toUpperCase());for(const i of[t,n,t.toUpperCase(),s]){const a=e.get(i);if(a)return a}}for(const[s,i]of Object.entries(e))if(s.toLowerCase()===n)return Array.isArray(i)?(i.length<=1||console.warn(`Received ${i.length} entries for the ${t} header, using the first entry.`),i[0]):i};class kt{constructor(t){this._client=t}}class $y extends kt{create(t,n){return this._client.post("/openai/v1/audio/speech",{body:t,...n,headers:{Accept:"audio/wav",...n==null?void 0:n.headers},__binaryResponse:!0})}}class Hy extends kt{create(t,n){return this._client.post("/openai/v1/audio/transcriptions",Zc({body:t,...n}))}}class Gy extends kt{create(t,n){return this._client.post("/openai/v1/audio/translations",Zc({body:t,...n}))}}class Ci extends kt{constructor(){super(...arguments),this.speech=new $y(this._client),this.transcriptions=new Hy(this._client),this.translations=new Gy(this._client)}}Ci.Speech=$y;Ci.Transcriptions=Hy;Ci.Translations=Gy;class Wy extends kt{create(t,n){return this._client.post("/openai/v1/batches",{body:t,...n})}retrieve(t,n){return this._client.get(`/openai/v1/batches/${t}`,n)}list(t){return this._client.get("/openai/v1/batches",t)}cancel(t,n){return this._client.post(`/openai/v1/batches/${t}/cancel`,n)}}let qy=class extends kt{create(t,n){return this._client.post("/openai/v1/chat/completions",{body:t,...n,stream:t.stream??!1})}};class ed extends kt{constructor(){super(...arguments),this.completions=new qy(this._client)}}ed.Completions=qy;class Ky extends kt{}class Xy extends kt{create(t,n){return this._client.post("/openai/v1/embeddings",{body:t,...n})}}class Jy extends kt{create(t,n){return this._client.post("/openai/v1/files",Zc({body:t,...n}))}list(t){return this._client.get("/openai/v1/files",t)}delete(t,n){return this._client.delete(`/openai/v1/files/${t}`,n)}content(t,n){return this._client.get(`/openai/v1/files/${t}/content`,{...n,headers:{Accept:"application/octet-stream",...n==null?void 0:n.headers},__binaryResponse:!0})}info(t,n){return this._client.get(`/openai/v1/files/${t}`,n)}}class Yy extends kt{retrieve(t,n){return this._client.get(`/openai/v1/models/${t}`,n)}list(t){return this._client.get("/openai/v1/models",t)}delete(t,n){return this._client.delete(`/openai/v1/models/${t}`,n)}}var Qy,Zy;class ae extends Xk{constructor({baseURL:t=fp("GROQ_BASE_URL"),apiKey:n=fp("GROQ_API_KEY"),...r}={}){if(n===void 0)throw new St("The GROQ_API_KEY environment variable is missing or empty; either provide it, or instantiate the Groq client with an apiKey option, like new Groq({ apiKey: 'My API Key' }).");const s={apiKey:n,...r,baseURL:t||"https://api.groq.com"};if(!s.dangerouslyAllowBrowser&&lb())throw new St(`It looks like you're running in a browser-like environment.

This is disabled by default, as it risks exposing your secret API credentials to attackers.
If you understand the risks and have appropriate mitigations in place,
you can set the \`dangerouslyAllowBrowser\` option to \`true\`, e.g.,

new Groq({ apiKey, dangerouslyAllowBrowser: true })`);super({baseURL:s.baseURL,baseURLOverridden:t?t!=="https://api.groq.com":!1,timeout:s.timeout??6e4,httpAgent:s.httpAgent,maxRetries:s.maxRetries,fetch:s.fetch}),Qy.add(this),this.completions=new Ky(this),this.chat=new ed(this),this.embeddings=new Xy(this),this.audio=new Ci(this),this.models=new Yy(this),this.batches=new Wy(this),this.files=new Jy(this),this._options=s,this.apiKey=n}defaultQuery(){return this._options.defaultQuery}defaultHeaders(t){return{...super.defaultHeaders(t),...this._options.defaultHeaders}}authHeaders(t){return{Authorization:`Bearer ${this.apiKey}`}}}Zy=ae,Qy=new WeakSet;ae.Groq=Zy;ae.DEFAULT_TIMEOUT=6e4;ae.GroqError=St;ae.APIError=Le;ae.APIConnectionError=ka;ae.APIConnectionTimeoutError=jy;ae.APIUserAbortError=Fu;ae.NotFoundError=Oy;ae.ConflictError=Dy;ae.RateLimitError=Ly;ae.BadRequestError=Ay;ae.AuthenticationError=Ry;ae.InternalServerError=Uy;ae.PermissionDeniedError=Py;ae.UnprocessableEntityError=My;ae.toFile=zy;ae.fileFromPath=Iy;ae.Completions=Ky;ae.Chat=ed;ae.Embeddings=Xy;ae.Audio=Ci;ae.Models=Yy;ae.Batches=Wy;ae.Files=Jy;const cb="";new ae({apiKey:cb,dangerouslyAllowBrowser:!0});const io=[{id:"openai/gpt-oss-20b",name:"GPT OSS 20B",description:"Blazing fast inference via Groq LPU",provider:"groq"}];async function db(e,t=null,n=null){throw new Error("No Groq API key found. Add VITE_GROQ_API_KEY to your .env file.")}async function fb(e,t=null){throw new Error("No Groq API key found. Add VITE_GROQ_API_KEY to your .env file.")}const hb=({isOpen:e,onClose:t,codeContext:n})=>{const[r,s]=N.useState([{role:"assistant",content:"👋 Hi! I'm your AI coding assistant. I can help you understand the code, debug issues, or answer questions about programming. How can I help you today?"}]),[i,a]=N.useState(""),[u,c]=N.useState(!1),[f,E]=N.useState(!1),[w,_]=N.useState(io[0].id),[A,R]=N.useState(!1),j=N.useRef(null),D=N.useRef(null),b=N.useRef(null),x=io.find(g=>g.id===w)||io[0],C=()=>{var g;(g=j.current)==null||g.scrollIntoView({behavior:"smooth"})};N.useEffect(()=>{C()},[r]),N.useEffect(()=>{const g=p=>{b.current&&!b.current.contains(p.target)&&R(!1)};return document.addEventListener("mousedown",g),()=>document.removeEventListener("mousedown",g)},[]),N.useEffect(()=>{var g;e&&((g=D.current)==null||g.focus())},[e]);const O=async()=>{if(!i.trim()||u)return;const g=i.trim();a(""),c(!0);const p=[...r,{role:"user",content:g}];s(p);try{const v=p.filter(k=>k.role!=="system"),S=await db(v,n,w);s([...p,{role:"assistant",content:S}])}catch(v){s([...p,{role:"assistant",content:`⚠️ Error: ${v.message}

Please make sure you've added your Groq API key to the .env file.`}])}finally{c(!1)}},M=async()=>{if(f||u)return;E(!0);const g=[...r,{role:"user",content:"💡 Give me a hint"}];s(g);try{const p=await fb(n,w);s([...g,{role:"assistant",content:`💡 **Hint:**

${p}`}])}catch(p){s([...g,{role:"assistant",content:`⚠️ Error: ${p.message}`}])}finally{E(!1)}},U=g=>{g.key==="Enter"&&!g.shiftKey&&(g.preventDefault(),O())};return e?m.jsxs("div",{className:"fixed right-0 top-0 h-screen w-[450px] glass-dark border-l border-white/10 flex flex-col z-50 backdrop-blur-3xl animate-slide-in",children:[m.jsxs("div",{className:"h-20 border-b border-white/5 flex items-center justify-between px-6 flex-shrink-0",children:[m.jsxs("div",{className:"flex items-center gap-3",children:[m.jsx("div",{className:"w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center shadow-lg",children:m.jsx(g2,{size:20,className:"text-white"})}),m.jsxs("div",{className:"relative",ref:b,children:[m.jsxs("button",{onClick:()=>R(!A),className:"flex flex-col text-left group transition-all",children:[m.jsxs("div",{className:"flex items-center gap-1",children:[m.jsx("h2",{className:"text-sm font-black uppercase tracking-tight group-hover:text-purple-400 transition-colors",children:"AI Assistant"}),m.jsx(r2,{size:12,className:`text-slate-500 transition-transform ${A?"rotate-180":""}`})]}),m.jsx("p",{className:"text-[10px] text-slate-500 font-bold uppercase tracking-widest",children:x.name})]}),A&&m.jsx("div",{className:"absolute top-full left-0 mt-2 w-64 glass-dark border border-white/10 rounded-2xl p-2 shadow-2xl z-[60] animate-in fade-in zoom-in duration-200",children:io.map(g=>m.jsxs("button",{onClick:()=>{_(g.id),R(!1)},className:`w-full text-left p-3 rounded-xl transition-all mb-1 last:mb-0 ${w===g.id?"bg-gradient-to-br from-purple-500/20 to-pink-500/20 border border-purple-500/30 text-white":"hover:bg-white/5 text-slate-400 hover:text-white border border-transparent"}`,children:[m.jsxs("div",{className:"flex items-center justify-between mb-0.5",children:[m.jsx("div",{className:"text-xs font-black uppercase tracking-tight",children:g.name}),m.jsx("span",{className:`text-[8px] px-1.5 py-0.5 rounded-md font-black uppercase tracking-widest border ${g.provider==="ollama"?"bg-blue-500/10 border-blue-500/30 text-blue-400":"bg-purple-500/10 border-purple-500/30 text-purple-400"}`,children:g.provider==="ollama"?"Local":"Cloud"})]}),m.jsx("div",{className:"text-[10px] text-slate-500 font-medium leading-tight",children:g.description})]},g.id))})]})]}),m.jsx("button",{onClick:t,className:"p-2 glass rounded-xl text-slate-400 hover:text-white transition-all",children:m.jsx(E2,{size:20})})]}),m.jsxs("div",{className:"flex-1 overflow-y-auto p-6 space-y-4 scrollbar-hide",children:[r.map((g,p)=>m.jsx("div",{className:`flex ${g.role==="user"?"justify-end":"justify-start"}`,children:m.jsx("div",{className:`max-w-[85%] rounded-2xl px-4 py-3 ${g.role==="user"?"bg-gradient-to-br from-purple-500 to-pink-500 text-white shadow-lg":"glass border border-white/10 text-slate-200"}`,children:m.jsx("p",{className:"text-sm leading-relaxed whitespace-pre-wrap font-medium",children:g.content})})},p)),u&&m.jsx("div",{className:"flex justify-start",children:m.jsxs("div",{className:"glass border border-white/10 rounded-2xl px-4 py-3 flex items-center gap-2",children:[m.jsx(u2,{size:16,className:"animate-spin text-purple-400"}),m.jsx("span",{className:"text-sm text-slate-400 font-medium",children:"Thinking..."})]})}),m.jsx("div",{ref:j})]}),m.jsx("div",{className:"px-6 pb-3 flex gap-2",children:m.jsxs("button",{onClick:M,disabled:f||u,className:"flex items-center gap-2 px-3 py-2 glass rounded-xl text-slate-400 hover:text-white hover:border-yellow-500/30 border border-white/5 transition-all text-xs font-black uppercase tracking-widest disabled:opacity-50 disabled:cursor-not-allowed",children:[m.jsx(l2,{size:14}),"Get Hint"]})}),m.jsxs("div",{className:"border-t border-white/5 p-4 flex-shrink-0",children:[m.jsxs("div",{className:"flex gap-2",children:[m.jsx("textarea",{ref:D,value:i,onChange:g=>a(g.target.value),onKeyPress:U,placeholder:"Ask me anything about coding...",className:"flex-1 bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-purple-500/50 resize-none font-medium",rows:3,disabled:u}),m.jsx("button",{onClick:O,disabled:!i.trim()||u,className:"w-12 h-12 self-end bg-gradient-to-br from-purple-500 to-pink-500 rounded-xl flex items-center justify-center text-white hover:shadow-lg hover:shadow-purple-500/20 transition-all disabled:opacity-50 disabled:cursor-not-allowed",children:m.jsx(m2,{size:18})})]}),m.jsx("p",{className:"text-[9px] text-slate-600 mt-2 font-bold uppercase tracking-widest",children:"Press Enter to send • Shift+Enter for new line"})]})]}):null},pb=({htmlContent:e,desiredOutput:t,previewCheck:n,onResult:r})=>{const s=N.useRef(null),i=N.useRef(null),[a,u]=N.useState("idle"),c=N.useRef(null),f=N.useCallback((A,R)=>{if(!R)return A;const j=`
<script>
(function() {
  function runCheck() {
    try {
      const __result = (function() { ${R} })();
      window.parent.postMessage({ type: 'PREVIEW_CHECK_RESULT', passed: !!__result }, '*');
    } catch(e) {
      window.parent.postMessage({ type: 'PREVIEW_CHECK_RESULT', passed: false, error: e.message }, '*');
    }
  }
  // Run after DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', runCheck);
  } else {
    runCheck();
  }
})();
<\/script>`;return A.includes("</body>")?A.replace("</body>",j+"</body>"):A+j},[]);N.useEffect(()=>(c.current&&clearTimeout(c.current),u("idle"),c.current=setTimeout(()=>{s.current&&(u("checking"),s.current.srcdoc=f(e,n))},600),()=>clearTimeout(c.current)),[e,n,f]),N.useEffect(()=>{i.current&&t&&(i.current.srcdoc=t)},[t]),N.useEffect(()=>{const A=R=>{var j;if(((j=R.data)==null?void 0:j.type)==="PREVIEW_CHECK_RESULT"){const D=R.data.passed;u(D?"passed":"failed"),r==null||r({passed:D})}};return window.addEventListener("message",A),()=>window.removeEventListener("message",A)},[r]);const E={idle:"text-slate-500 border-white/10",checking:"text-yellow-400 border-yellow-500/30 bg-yellow-500/5",passed:"text-green-400 border-green-500/30 bg-green-500/10",failed:"text-red-400 border-red-500/30 bg-red-500/5"},w={idle:"Preview Ready",checking:"Validating…",passed:"✅ Output Matches!",failed:"❌ Not Matching Yet"},_={idle:Zh,checking:Zh,passed:fy,failed:S2}[a];return m.jsxs("div",{className:"flex flex-col h-full bg-[#030303]",children:[m.jsxs("div",{className:"flex items-center justify-between px-6 py-3 border-b border-white/5 glass-dark flex-shrink-0",children:[m.jsx("span",{className:"text-[10px] font-black text-slate-500 uppercase tracking-widest",children:"Live IDE Preview"}),m.jsxs("div",{className:`flex items-center gap-1.5 px-3 py-1 rounded-xl text-[10px] font-black uppercase tracking-widest border transition-all duration-300 ${E[a]} ${a==="passed"?"neon-border-magenta shadow-[0_0_10px_rgba(217,70,239,0.1)]":""}`,children:[m.jsx(_,{size:11,className:a==="checking"?"animate-spin":""}),w[a]]})]}),m.jsxs("div",{className:"flex flex-1 overflow-hidden divide-x divide-white/5",children:[m.jsxs("div",{className:"flex flex-col flex-1 min-w-0 relative",children:[m.jsxs("div",{className:"flex items-center gap-2 px-4 py-2 bg-black/40 border-b border-white/5 flex-shrink-0 backdrop-blur-md",children:[m.jsx(fi,{size:12,className:"text-purple-400"}),m.jsx("span",{className:"text-[10px] font-bold text-slate-400 uppercase tracking-widest",children:"Your Environment"}),m.jsx("div",{className:`ml-auto w-2 h-2 rounded-full transition-all duration-500 ${a==="passed"?"bg-green-400 shadow-[0_0_10px_rgba(34,197,94,0.8)] scale-110":a==="failed"?"bg-red-400 shadow-[0_0_10px_rgba(239,68,68,0.6)]":"bg-slate-600"}`})]}),m.jsx("div",{className:"flex-1 bg-white relative",children:m.jsx("iframe",{ref:s,title:"Live Preview",sandbox:"allow-scripts",className:"absolute inset-0 w-full h-full border-none"})})]}),m.jsxs("div",{className:"flex flex-col flex-1 min-w-0 relative",children:[m.jsxs("div",{className:"flex items-center gap-2 px-4 py-2 bg-black/40 border-b border-white/5 flex-shrink-0 backdrop-blur-md",children:[m.jsx(y2,{size:12,className:"text-pink-400"}),m.jsx("span",{className:"text-[10px] font-bold text-slate-400 uppercase tracking-widest",children:"Target Output"}),m.jsx("span",{className:"ml-auto text-[9px] text-slate-600 font-extrabold uppercase tracking-widest px-2 py-0.5 border border-white/5 rounded-md bg-white/5",children:"Reference"})]}),m.jsx("div",{className:"flex-1 bg-white relative",children:m.jsx("iframe",{ref:i,title:"Desired Output",sandbox:"allow-scripts",className:"absolute inset-0 w-full h-full border-none opacity-90"})})]})]})]})},$u="http://localhost:3001",mb=async(e,t="index.js",n=null)=>{let r="";if(typeof n=="function"){const i=n.toString().match(/=>\s*([\s\S]+)$/);if(i){const a=i[1].trim();r=a.startsWith("{")?a.slice(1,-1).trim():`return (${a});`}}else typeof n=="string"&&(r=n);try{const s=await fetch(`${$u}/execute`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({files:e,entryPoint:t,testCheck:r}),signal:AbortSignal.timeout(3e4)});if(!s.ok){const i=await s.json().catch(()=>({}));throw new Error(i.error||`API error: ${s.status}`)}return await s.json()}catch(s){return s.name==="TimeoutError"||s.name==="AbortError"?{stdout:"",stderr:"Request timed out. The sandbox API may be unavailable.",exitCode:1,timedOut:!0,passed:!1}:s.message.includes("Failed to fetch")||s.message.includes("fetch")?{stdout:"",stderr:`⚠️  Cannot connect to Sandbox API at ${$u}.

Please start the backend:
  cd sandbox-api
  npm install
  node server.js

And make sure the Docker "sandbox-runner" image is built:
  docker build -f runner.Dockerfile -t sandbox-runner .`,exitCode:1,timedOut:!1,passed:!1}:{stdout:"",stderr:`Execution error: ${s.message}`,exitCode:1,timedOut:!1,passed:!1}}},pp=async()=>{try{return(await fetch(`${$u}/health`,{signal:AbortSignal.timeout(3e3)})).ok}catch{return!1}},gb=()=>{var p,v,S,k;const{scenarioId:e}=M1(),t=Qr(),n=Jn.find(I=>I.id===e)||Jn[0],r=n.type==="frontend",[s,i]=N.useState(Object.keys(n.files)[0]),[a,u]=N.useState(n.files),[c,f]=N.useState('No output yet. Click "Run Scenario" to execute your code.'),[E,w]=N.useState(!1),[_,A]=N.useState(!1),[R,j]=N.useState(!1),[D,b]=N.useState(null);N.useEffect(()=>{r||pp().then(I=>b(I))},[r]),N.useEffect(()=>{const I=Jn.find(y=>y.id===e);I&&(u(I.files),i(Object.keys(I.files)[0]),A(!1),f("New scenario loaded. Ready to test."))},[e]);const x=(I,y)=>{y.editor.defineTheme("custom-dark",{base:"vs-dark",inherit:!0,rules:[],colors:{"editor.background":"#060606","editor.lineHighlightBackground":"#ffffff08","editorLineNumber.foreground":"#4b5563","editorLineNumber.activeForeground":"#7c3aed"}}),y.editor.setTheme("custom-dark")},C=N.useCallback(({passed:I})=>{A(I),I&&f(`✅ Your preview matches the desired output!

All visual checks passed. Click "Submit" to complete the scenario. ✨`)},[]),O=async()=>{if(r){f(`🔍 Checking your preview against the desired output…
(The live preview updates automatically as you type.)`);return}w(!0),A(!1),f("⏳ Sending code to Docker sandbox...");const I=Object.keys(a).find($=>$.endsWith(".js"))||Object.keys(a)[0],y=await mb(a,I,n.testCheck);pp().then($=>b($));let Y="";y.timedOut?Y=`⏱️  Execution Timed Out (10s limit exceeded)

Your code ran for too long. Check for infinite loops.`:y.stderr&&!y.stdout?Y=`❌ Runtime Error

${y.stderr}`:(y.stdout&&(Y+=`📤 Output:
${y.stdout}`),y.stderr&&(Y+=`
⚠️  Warnings / Errors:
${y.stderr}`),!y.stdout&&!y.stderr&&(Y="(No output produced)")),y.passed?(Y+=`

✅ Scenario Tests Passed!
✔ Logic: Correct
✔ Expected Patterns: Found
✔ Stability: Verified

All tests passed successfully! ✨`,A(!0)):!y.timedOut&&y.exitCode===0&&(Y+=`

❌ Scenario Tests Failed
  Expected solution pattern not found.

💡 Hint: ${n.constraints[0]}`),f(Y),w(!1)},M=()=>{const I=JSON.parse(localStorage.getItem("completedProjects")||"[]");I.includes(n.id)||(I.push(n.id),localStorage.setItem("completedProjects",JSON.stringify(I))),alert("🎉 Scenario Submitted! Great job."),t("/dashboard")},U=I=>{u(y=>({...y,[s]:{...y[s],content:I}}))},g=r&&((p=a[s])==null?void 0:p.content)||"";return m.jsxs("div",{className:"relative h-screen flex flex-col bg-[#030303] text-white overflow-hidden font-sans",children:[m.jsxs("div",{className:"bg-streaks",children:[m.jsx("div",{className:"streak-1 opacity-20"}),m.jsx("div",{className:"streak-2 opacity-20"})]}),m.jsxs("header",{className:"h-20 glass-dark border-b border-white/5 flex items-center justify-between px-8 flex-shrink-0 z-50 backdrop-blur-3xl",children:[m.jsxs("div",{className:"flex items-center gap-6",children:[m.jsx(xi,{to:"/dashboard",className:"p-2 glass rounded-xl text-slate-400 hover:text-white transition-all flex items-center justify-center",children:m.jsx(n2,{size:20})}),m.jsx("div",{className:"h-8 w-px bg-white/10"}),m.jsxs("div",{className:"flex flex-col",children:[m.jsxs("div",{className:"flex items-center gap-3",children:[m.jsx("h1",{className:"font-black text-lg tracking-tight uppercase",children:n.title}),m.jsx("span",{className:`${n.badgeColor.split(" ").slice(0,2).join(" ")} px-3 py-0.5 rounded-full text-[10px] font-black uppercase tracking-widest border ${n.badgeColor.split(" ").slice(4).join(" ")} shadow-lg neon-border-magenta`,children:n.difficulty}),r&&m.jsxs("span",{className:"flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[10px] font-black uppercase tracking-widest border text-blue-400 border-blue-500/30 bg-blue-500/5",children:[m.jsx(fi,{size:10})," Frontend"]})]}),m.jsx("p",{className:"text-[10px] text-slate-500 font-bold uppercase tracking-widest",children:"Scenario IDE v2.0"})]})]}),m.jsxs("div",{className:"flex items-center gap-4",children:[!r&&m.jsx("div",{className:`flex items-center gap-2 px-3 py-1.5 rounded-xl text-[10px] font-black uppercase tracking-widest border ${D===null?"text-slate-500 border-white/10":D?"text-green-400 border-green-500/30 bg-green-500/5":"text-red-400 border-red-500/30 bg-red-500/5"}`,children:D===null?m.jsxs(m.Fragment,{children:[m.jsx(s2,{size:12})," Checking"]}):D?m.jsxs(m.Fragment,{children:[m.jsx(x2,{size:12})," Sandbox Live"]}):m.jsxs(m.Fragment,{children:[m.jsx(w2,{size:12})," API Offline"]})}),r&&m.jsxs("div",{className:"flex items-center gap-2 px-3 py-1.5 rounded-xl text-[10px] font-black uppercase tracking-widest border text-blue-400 border-blue-500/30 bg-blue-500/5",children:[m.jsx(fi,{size:12})," Live Preview"]}),m.jsxs("button",{className:"flex items-center gap-2 px-4 py-2 text-slate-400 hover:text-white glass rounded-xl transition-all text-xs font-black uppercase tracking-widest",onClick:()=>u(n.files),children:[m.jsx(h2,{size:16})," Reset"]}),m.jsxs("button",{className:"flex items-center gap-2 px-6 py-2 glass-dark border border-white/10 hover:border-white/30 text-white rounded-xl transition-all text-xs font-black uppercase tracking-widest",children:[m.jsx(p2,{size:16})," Save"]}),m.jsxs("button",{onClick:()=>j(!R),className:`flex items-center gap-2 px-6 py-2 rounded-xl transition-all text-xs font-black uppercase tracking-widest ${R?"bg-gradient-to-br from-purple-500 to-pink-500 text-white shadow-lg shadow-purple-500/20":"glass-dark border border-white/10 hover:border-purple-500/30 text-white"}`,children:[m.jsx(d2,{size:16})," AI Assistant"]}),m.jsxs("button",{onClick:O,disabled:E,className:"flex items-center gap-2 px-8 py-2.5 bg-white text-[#030303] hover:bg-slate-200 rounded-xl transition-all text-xs font-black uppercase tracking-widest disabled:opacity-50 disabled:cursor-not-allowed shadow-xl hover:shadow-white/20",children:[m.jsx(f2,{size:16,fill:"currentColor"})," ",E?"Running...":r?"Check Preview":"Run Scenario"]}),_&&m.jsxs("button",{onClick:M,className:"flex items-center gap-2 px-8 py-2.5 bg-magenta-500 text-white hover:bg-magenta-600 rounded-xl transition-all text-xs font-black uppercase tracking-widest shadow-xl shadow-magenta-500/20 neon-border-magenta animate-pulse",children:[m.jsx(fy,{size:16})," Submit Scenario"]})]})]}),m.jsxs("div",{className:"flex-1 flex overflow-hidden relative z-10",children:[m.jsxs("aside",{className:"w-72 glass-dark border-r border-white/5 flex flex-col flex-shrink-0 backdrop-blur-3xl",children:[m.jsx("div",{className:"px-6 py-4 text-[10px] font-black text-slate-500 uppercase tracking-widest border-b border-white/5",children:"Project Files"}),m.jsx("div",{className:"flex-1 overflow-y-auto py-4",children:Object.keys(a).map(I=>m.jsxs("button",{onClick:()=>i(I),className:`w-[90%] mx-auto mb-1 rounded-xl text-left px-4 py-3 flex items-center gap-3 text-sm transition-all ${s===I?"bg-white/10 text-white border border-white/10 shadow-lg":"text-slate-500 hover:text-slate-300 hover:bg-white/5 border border-transparent"}`,children:[I.endsWith(".js")||I.endsWith(".html")||I.endsWith(".css")?m.jsx(i2,{size:18,className:"text-purple-400"}):m.jsx(o2,{size:18,className:"text-slate-400"}),m.jsx("span",{className:"font-bold tracking-tight",children:I})]},I))}),m.jsxs("div",{className:"border-t border-white/5",children:[m.jsx("div",{className:"px-6 py-4 text-[10px] font-black text-slate-500 uppercase tracking-widest border-b border-white/5",children:"Scenario Constraints"}),m.jsx("div",{className:"px-4 py-4 space-y-3",children:n.constraints.map((I,y)=>m.jsxs("div",{className:"p-4 glass rounded-2xl border-white/5 group hover:border-purple-500/30 transition-all",children:[m.jsxs("h4",{className:"text-[11px] font-black text-white uppercase tracking-wider mb-1",children:["Constraint ",y+1]}),m.jsx("p",{className:"text-[10px] text-slate-500 font-bold leading-relaxed",children:I})]},y))})]})]}),m.jsxs("div",{className:"flex-1 flex flex-col min-w-0 bg-[#060606]",children:[m.jsx("div",{className:r?"h-1/2 relative border-b border-white/5":"flex-1 relative",children:m.jsx(Dk,{height:"100%",defaultLanguage:"javascript",language:((v=a[s])==null?void 0:v.language)||"javascript",value:((S=a[s])==null?void 0:S.content)||"",onChange:U,theme:"custom-dark",options:{minimap:{enabled:!1},padding:{top:30},fontSize:15,fontFamily:"'JetBrains Mono', monospace",scrollBeyondLastLine:!1,automaticLayout:!0,lineNumbers:"on",glyphMargin:!1,folding:!0,lineDecorationsWidth:10,lineNumbersMinChars:3},onMount:x})}),r?m.jsx("div",{className:"h-1/2 flex flex-col overflow-hidden",children:m.jsx(pb,{htmlContent:g,desiredOutput:n.desiredOutput,previewCheck:n.previewCheck,onResult:C})}):m.jsxs("div",{className:"h-64 glass-dark border-t border-white/5 flex flex-col flex-shrink-0 backdrop-blur-3xl",children:[m.jsxs("div",{className:"px-6 py-3 border-b border-white/5 flex items-center justify-between",children:[m.jsx("span",{className:"text-[10px] font-black text-slate-500 uppercase tracking-widest",children:"Compiler Output"}),m.jsxs("div",{className:"flex items-center gap-2",children:[m.jsx("div",{className:"w-1.5 h-1.5 rounded-full bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.5)]"}),m.jsx("span",{className:"text-[10px] font-black text-slate-400 uppercase tracking-widest",children:"System Ready"})]})]}),m.jsxs("div",{className:"flex-1 p-6 font-mono text-sm overflow-auto text-slate-300 leading-relaxed scrollbar-hide",children:[m.jsx("div",{className:"text-slate-500 mb-2",children:"$ scenario-shell --run-tests"}),c]})]})]})]}),m.jsx(hb,{isOpen:R,onClose:()=>j(!1),codeContext:{activeFile:s,code:((k=a[s])==null?void 0:k.content)||"",scenario:n.title}})]})};function yb(){return m.jsx(ZE,{children:m.jsx(ix,{children:m.jsxs(Q1,{children:[m.jsx(dr,{path:"/",element:m.jsx(Mg,{to:"/login",replace:!0})}),m.jsx(dr,{path:"/login",element:m.jsx(k2,{})}),m.jsx(dr,{path:"/signup",element:m.jsx(b2,{})}),m.jsx(dr,{path:"/dashboard",element:m.jsx(Qh,{children:m.jsx(C2,{})})}),m.jsx(dr,{path:"/editor/:scenarioId",element:m.jsx(Qh,{children:m.jsx(gb,{})})})]})})})}console.log("CodeSandbox: Starting React mount...");try{const e=document.getElementById("root");if(!e)throw new Error("Root element not found!");Ll.createRoot(e).render(m.jsx(fn.StrictMode,{children:m.jsx(yb,{})})),console.log("CodeSandbox: React mount called successfully.")}catch(e){console.error("CodeSandbox: Critical mount error:",e),document.body.innerHTML=`<div style="color: red; padding: 20px; background: #1a1a1a; min-height: 100vh;">
        <h1>Frontend Crash</h1>
        <pre>${e.stack||e.message}</pre>
    </div>`}
