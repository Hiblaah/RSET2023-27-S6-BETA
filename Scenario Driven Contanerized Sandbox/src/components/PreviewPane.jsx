import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Eye, Target, CheckCircle2, XCircle, RefreshCw } from 'lucide-react';

/**
 * PreviewPane
 * -----------
 * Renders the user's HTML code live in a sandboxed iframe (left panel)
 * and shows the desired output in a reference iframe (right panel).
 * Validates the live preview by injecting a previewCheck script into the iframe.
 *
 * Props:
 *  - htmlContent   {string}  The user's current HTML code
 *  - desiredOutput {string}  The reference / target HTML
 *  - previewCheck  {string}  A JS expression (returns boolean) that is run inside the live iframe
 *  - onResult      {fn}      Called with { passed: boolean } when validation runs
 */
const PreviewPane = ({ htmlContent, desiredOutput, previewCheck, onResult }) => {
    const liveRef = useRef(null);
    const desiredRef = useRef(null);
    const [validationState, setValidationState] = useState('idle'); // 'idle' | 'checking' | 'passed' | 'failed'
    const debounceRef = useRef(null);

    // Build the full HTML to inject into the live iframe,
    // appending the previewCheck script (if provided) at the bottom as a data marker.
    const buildSrcdoc = useCallback((html, check) => {
        if (!check) return html;
        // We inject a small helper that runs the check and posts the result to the parent
        const injected = `
<script>
(function() {
  function runCheck() {
    try {
      const __result = (function() { ${check} })();
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
</script>`;
        // Insert before closing body tag, or just append
        if (html.includes('</body>')) {
            return html.replace('</body>', injected + '</body>');
        }
        return html + injected;
    }, []);

    // Write to live iframe with debounce
    useEffect(() => {
        if (debounceRef.current) clearTimeout(debounceRef.current);
        setValidationState('idle');

        debounceRef.current = setTimeout(() => {
            if (liveRef.current) {
                setValidationState('checking');
                liveRef.current.srcdoc = buildSrcdoc(htmlContent, previewCheck);
            }
        }, 600);

        return () => clearTimeout(debounceRef.current);
    }, [htmlContent, previewCheck, buildSrcdoc]);

    // Write desired output to reference iframe once
    useEffect(() => {
        if (desiredRef.current && desiredOutput) {
            desiredRef.current.srcdoc = desiredOutput;
        }
    }, [desiredOutput]);

    // Listen for postMessage from the live iframe
    useEffect(() => {
        const handler = (event) => {
            if (event.data?.type === 'PREVIEW_CHECK_RESULT') {
                const passed = event.data.passed;
                setValidationState(passed ? 'passed' : 'failed');
                onResult?.({ passed });
            }
        };
        window.addEventListener('message', handler);
        return () => window.removeEventListener('message', handler);
    }, [onResult]);

    const statusColors = {
        idle: 'text-slate-500 border-white/10',
        checking: 'text-yellow-400 border-yellow-500/30 bg-yellow-500/5',
        passed: 'text-green-400 border-green-500/30 bg-green-500/10',
        failed: 'text-red-400 border-red-500/30 bg-red-500/5',
    };
    const statusLabel = {
        idle: 'Preview Ready',
        checking: 'Validating…',
        passed: '✅ Output Matches!',
        failed: '❌ Not Matching Yet',
    };

    const StatusIcon = {
        idle: RefreshCw,
        checking: RefreshCw,
        passed: CheckCircle2,
        failed: XCircle,
    }[validationState];

    return (
        <div className="flex flex-col h-full bg-[#030303]">
            {/* Status bar */}
            <div className="flex items-center justify-between px-6 py-3 border-b border-white/5 glass-dark flex-shrink-0">
                <span className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Live IDE Preview</span>
                <div className={`flex items-center gap-1.5 px-3 py-1 rounded-xl text-[10px] font-black uppercase tracking-widest border transition-all duration-300 ${statusColors[validationState]} ${validationState === 'passed' ? 'neon-border-magenta shadow-[0_0_10px_rgba(217,70,239,0.1)]' : ''}`}>
                    <StatusIcon size={11} className={validationState === 'checking' ? 'animate-spin' : ''} />
                    {statusLabel[validationState]}
                </div>
            </div>

            {/* Split panes */}
            <div className="flex flex-1 overflow-hidden divide-x divide-white/5">
                {/* Live Preview pane */}
                <div className="flex flex-col flex-1 min-w-0 relative">
                    <div className="flex items-center gap-2 px-4 py-2 bg-black/40 border-b border-white/5 flex-shrink-0 backdrop-blur-md">
                        <Eye size={12} className="text-purple-400" />
                        <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Your Environment</span>
                        <div className={`ml-auto w-2 h-2 rounded-full transition-all duration-500 ${validationState === 'passed' ? 'bg-green-400 shadow-[0_0_10px_rgba(34,197,94,0.8)] scale-110' :
                                validationState === 'failed' ? 'bg-red-400 shadow-[0_0_10px_rgba(239,68,68,0.6)]' :
                                    'bg-slate-600'
                            }`} />
                    </div>
                    <div className="flex-1 bg-white relative">
                        <iframe
                            ref={liveRef}
                            title="Live Preview"
                            sandbox="allow-scripts"
                            className="absolute inset-0 w-full h-full border-none"
                        />
                    </div>
                </div>

                {/* Desired Output pane */}
                <div className="flex flex-col flex-1 min-w-0 relative">
                    <div className="flex items-center gap-2 px-4 py-2 bg-black/40 border-b border-white/5 flex-shrink-0 backdrop-blur-md">
                        <Target size={12} className="text-pink-400" />
                        <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Target Output</span>
                        <span className="ml-auto text-[9px] text-slate-600 font-extrabold uppercase tracking-widest px-2 py-0.5 border border-white/5 rounded-md bg-white/5">Reference</span>
                    </div>
                    <div className="flex-1 bg-white relative">
                        <iframe
                            ref={desiredRef}
                            title="Desired Output"
                            sandbox="allow-scripts"
                            className="absolute inset-0 w-full h-full border-none opacity-90"
                        />
                    </div>
                </div>
            </div>
        </div>
    );
};

export default PreviewPane;
