
import React, { useState, useEffect, useCallback } from 'react';
import { Link, useParams, useNavigate } from 'react-router-dom';
import Editor, { useMonaco } from '@monaco-editor/react';
import {
    ArrowLeft,
    RotateCcw,
    Save,
    Play,
    FileText,
    FileCode,
    CheckCircle2,
    AlertCircle,
    MessageCircle,
    Container,
    Wifi,
    WifiOff,
    Eye
} from 'lucide-react';
import AIChatAssistant from '../components/AIChatAssistant';
import PreviewPane from '../components/PreviewPane';
import { scenarios } from '../data/scenarios';
import { executeCode, isSandboxAvailable } from '../services/executionService';

const Ide = () => {
    const { scenarioId } = useParams();
    const navigate = useNavigate();
    const scenario = scenarios.find(s => s.id === scenarioId) || scenarios[0];
    const isFrontend = scenario.type === 'frontend';

    const [activeFile, setActiveFile] = useState(Object.keys(scenario.files)[0]);
    const [files, setFiles] = useState(scenario.files);
    const [output, setOutput] = useState('No output yet. Click "Run Scenario" to execute your code.');
    const [isRunning, setIsRunning] = useState(false);
    const [testsPassed, setTestsPassed] = useState(false);
    const [isChatOpen, setIsChatOpen] = useState(false);
    const [sandboxOnline, setSandboxOnline] = useState(null);

    // Check sandbox API availability on mount (only needed for backend scenarios)
    useEffect(() => {
        if (!isFrontend) {
            isSandboxAvailable().then(online => setSandboxOnline(online));
        }
    }, [isFrontend]);

    // Sync state when scenarioId changes
    useEffect(() => {
        const currentScenario = scenarios.find(s => s.id === scenarioId);
        if (currentScenario) {
            setFiles(currentScenario.files);
            setActiveFile(Object.keys(currentScenario.files)[0]);
            setTestsPassed(false);
            setOutput('New scenario loaded. Ready to test.');
        }
    }, [scenarioId]);

    // Initial theme setup on mount
    const handleEditorDidMount = (editor, monaco) => {
        monaco.editor.defineTheme('custom-dark', {
            base: 'vs-dark',
            inherit: true,
            rules: [],
            colors: {
                'editor.background': '#060606',
                'editor.lineHighlightBackground': '#ffffff08',
                'editorLineNumber.foreground': '#4b5563',
                'editorLineNumber.activeForeground': '#7c3aed',
            },
        });
        monaco.editor.setTheme('custom-dark');
    };

    // Called by PreviewPane when validation result arrives
    const handlePreviewResult = useCallback(({ passed }) => {
        setTestsPassed(passed);
        if (passed) {
            setOutput('✅ Your preview matches the desired output!\n\nAll visual checks passed. Click "Submit" to complete the scenario. ✨');
        }
    }, []);

    // "Run Scenario" for frontend: just trigger a re-validation via key bump
    // The PreviewPane already reacts to content changes in real-time.
    // For backend: execute via Docker sandbox.
    const handleRunTests = async () => {
        if (isFrontend) {
            // Frontend validation is live — just update output
            setOutput('🔍 Checking your preview against the desired output…\n(The live preview updates automatically as you type.)');
            return;
        }

        setIsRunning(true);
        setTestsPassed(false);
        setOutput('⏳ Sending code to Docker sandbox...');

        const mainFileKey = Object.keys(files).find(f => f.endsWith('.js')) || Object.keys(files)[0];
        const result = await executeCode(files, mainFileKey, scenario.testCheck);

        isSandboxAvailable().then(online => setSandboxOnline(online));

        let outputText = '';

        if (result.timedOut) {
            outputText = `⏱️  Execution Timed Out (10s limit exceeded)\n\nYour code ran for too long. Check for infinite loops.`;
        } else if (result.stderr && !result.stdout) {
            outputText = `❌ Runtime Error\n\n${result.stderr}`;
        } else {
            if (result.stdout) {
                outputText += `📤 Output:\n${result.stdout}`;
            }
            if (result.stderr) {
                outputText += `\n⚠️  Warnings / Errors:\n${result.stderr}`;
            }
            if (!result.stdout && !result.stderr) {
                outputText = '(No output produced)';
            }
        }

        if (result.passed) {
            outputText += `\n\n✅ Scenario Tests Passed!\n✔ Logic: Correct\n✔ Expected Patterns: Found\n✔ Stability: Verified\n\nAll tests passed successfully! ✨`;
            setTestsPassed(true);
        } else if (!result.timedOut && result.exitCode === 0) {
            outputText += `\n\n❌ Scenario Tests Failed\n  Expected solution pattern not found.\n\n💡 Hint: ${scenario.constraints[0]}`;
        }

        setOutput(outputText);
        setIsRunning(false);
    };

    const handleSubmit = () => {
        const completedProjects = JSON.parse(localStorage.getItem('completedProjects') || '[]');
        if (!completedProjects.includes(scenario.id)) {
            completedProjects.push(scenario.id);
            localStorage.setItem('completedProjects', JSON.stringify(completedProjects));
        }
        alert('🎉 Scenario Submitted! Great job.');
        navigate('/dashboard');
    };

    const handleEditorChange = (value) => {
        setFiles(prev => ({
            ...prev,
            [activeFile]: {
                ...prev[activeFile],
                content: value
            }
        }));
    };

    // Get current file content for the live preview (for frontend scenarios)
    const liveHtmlContent = isFrontend ? (files[activeFile]?.content || '') : '';

    return (
        <div className="relative h-screen flex flex-col bg-[#030303] text-white overflow-hidden font-sans">
            {/* Background Streaks */}
            <div className="bg-streaks">
                <div className="streak-1 opacity-20"></div>
                <div className="streak-2 opacity-20"></div>
            </div>

            {/* Header */}
            <header className="h-20 glass-dark border-b border-white/5 flex items-center justify-between px-8 flex-shrink-0 z-50 backdrop-blur-3xl">
                <div className="flex items-center gap-6">
                    <Link to="/dashboard" className="p-2 glass rounded-xl text-slate-400 hover:text-white transition-all flex items-center justify-center">
                        <ArrowLeft size={20} />
                    </Link>
                    <div className="h-8 w-px bg-white/10"></div>
                    <div className="flex flex-col">
                        <div className="flex items-center gap-3">
                            <h1 className="font-black text-lg tracking-tight uppercase">{scenario.title}</h1>
                            <span className={`${scenario.badgeColor.split(' ').slice(0, 2).join(' ')} px-3 py-0.5 rounded-full text-[10px] font-black uppercase tracking-widest border ${scenario.badgeColor.split(' ').slice(4).join(' ')} shadow-lg neon-border-magenta`}>
                                {scenario.difficulty}
                            </span>
                            {isFrontend && (
                                <span className="flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[10px] font-black uppercase tracking-widest border text-blue-400 border-blue-500/30 bg-blue-500/5">
                                    <Eye size={10} /> Frontend
                                </span>
                            )}
                        </div>
                        <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">Scenario IDE v2.0</p>
                    </div>
                </div>

                <div className="flex items-center gap-4">
                    {/* Sandbox Status Indicator – only for backend scenarios */}
                    {!isFrontend && (
                        <div className={`flex items-center gap-2 px-3 py-1.5 rounded-xl text-[10px] font-black uppercase tracking-widest border ${sandboxOnline === null
                            ? 'text-slate-500 border-white/10'
                            : sandboxOnline
                                ? 'text-green-400 border-green-500/30 bg-green-500/5'
                                : 'text-red-400 border-red-500/30 bg-red-500/5'
                            }`}>
                            {sandboxOnline === null
                                ? <><Container size={12} /> Checking</>
                                : sandboxOnline
                                    ? <><Wifi size={12} /> Sandbox Live</>
                                    : <><WifiOff size={12} /> API Offline</>}
                        </div>
                    )}
                    {isFrontend && (
                        <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl text-[10px] font-black uppercase tracking-widest border text-blue-400 border-blue-500/30 bg-blue-500/5">
                            <Eye size={12} /> Live Preview
                        </div>
                    )}
                    <button
                        className="flex items-center gap-2 px-4 py-2 text-slate-400 hover:text-white glass rounded-xl transition-all text-xs font-black uppercase tracking-widest"
                        onClick={() => setFiles(scenario.files)}
                    >
                        <RotateCcw size={16} /> Reset
                    </button>
                    <button className="flex items-center gap-2 px-6 py-2 glass-dark border border-white/10 hover:border-white/30 text-white rounded-xl transition-all text-xs font-black uppercase tracking-widest">
                        <Save size={16} /> Save
                    </button>
                    <button
                        onClick={() => setIsChatOpen(!isChatOpen)}
                        className={`flex items-center gap-2 px-6 py-2 rounded-xl transition-all text-xs font-black uppercase tracking-widest ${isChatOpen
                            ? 'bg-gradient-to-br from-purple-500 to-pink-500 text-white shadow-lg shadow-purple-500/20'
                            : 'glass-dark border border-white/10 hover:border-purple-500/30 text-white'
                            }`}
                    >
                        <MessageCircle size={16} /> AI Assistant
                    </button>
                    <button
                        onClick={handleRunTests}
                        disabled={isRunning}
                        className="flex items-center gap-2 px-8 py-2.5 bg-white text-[#030303] hover:bg-slate-200 rounded-xl transition-all text-xs font-black uppercase tracking-widest disabled:opacity-50 disabled:cursor-not-allowed shadow-xl hover:shadow-white/20"
                    >
                        <Play size={16} fill="currentColor" /> {isRunning ? 'Running...' : isFrontend ? 'Check Preview' : 'Run Scenario'}
                    </button>

                    {testsPassed && (
                        <button
                            onClick={handleSubmit}
                            className="flex items-center gap-2 px-8 py-2.5 bg-magenta-500 text-white hover:bg-magenta-600 rounded-xl transition-all text-xs font-black uppercase tracking-widest shadow-xl shadow-magenta-500/20 neon-border-magenta animate-pulse"
                        >
                            <CheckCircle2 size={16} /> Submit Scenario
                        </button>
                    )}
                </div>
            </header>

            {/* Main Content */}
            <div className="flex-1 flex overflow-hidden relative z-10">
                {/* Sidebar */}
                <aside className="w-72 glass-dark border-r border-white/5 flex flex-col flex-shrink-0 backdrop-blur-3xl">
                    <div className="px-6 py-4 text-[10px] font-black text-slate-500 uppercase tracking-widest border-b border-white/5">Project Files</div>
                    <div className="flex-1 overflow-y-auto py-4">
                        {Object.keys(files).map((filename) => (
                            <button
                                key={filename}
                                onClick={() => setActiveFile(filename)}
                                className={`w-[90%] mx-auto mb-1 rounded-xl text-left px-4 py-3 flex items-center gap-3 text-sm transition-all ${activeFile === filename
                                    ? 'bg-white/10 text-white border border-white/10 shadow-lg'
                                    : 'text-slate-500 hover:text-slate-300 hover:bg-white/5 border border-transparent'
                                    }`}
                            >
                                {filename.endsWith('.js') || filename.endsWith('.html') || filename.endsWith('.css')
                                    ? <FileCode size={18} className="text-purple-400" />
                                    : <FileText size={18} className="text-slate-400" />}
                                <span className="font-bold tracking-tight">{filename}</span>
                            </button>
                        ))}
                    </div>

                    <div className="border-t border-white/5">
                        <div className="px-6 py-4 text-[10px] font-black text-slate-500 uppercase tracking-widest border-b border-white/5">Scenario Constraints</div>
                        <div className="px-4 py-4 space-y-3">
                            {scenario.constraints.map((constraint, idx) => (
                                <div key={idx} className="p-4 glass rounded-2xl border-white/5 group hover:border-purple-500/30 transition-all">
                                    <h4 className="text-[11px] font-black text-white uppercase tracking-wider mb-1">Constraint {idx + 1}</h4>
                                    <p className="text-[10px] text-slate-500 font-bold leading-relaxed">{constraint}</p>
                                </div>
                            ))}
                        </div>
                    </div>
                </aside>

                {/* Editor Area */}
                <div className="flex-1 flex flex-col min-w-0 bg-[#060606]">
                    {/* Monaco Editor */}
                    <div className={isFrontend ? 'h-1/2 relative border-b border-white/5' : 'flex-1 relative'}>
                        <Editor
                            height="100%"
                            defaultLanguage="javascript"
                            language={files[activeFile]?.language || 'javascript'}
                            value={files[activeFile]?.content || ''}
                            onChange={handleEditorChange}
                            theme="custom-dark"
                            options={{
                                minimap: { enabled: false },
                                padding: { top: 30 },
                                fontSize: 15,
                                fontFamily: "'JetBrains Mono', monospace",
                                scrollBeyondLastLine: false,
                                automaticLayout: true,
                                lineNumbers: 'on',
                                glyphMargin: false,
                                folding: true,
                                lineDecorationsWidth: 10,
                                lineNumbersMinChars: 3,
                            }}
                            onMount={handleEditorDidMount}
                        />
                    </div>

                    {/* Bottom Panel: Preview for frontend, Console for backend */}
                    {isFrontend ? (
                        <div className="h-1/2 flex flex-col overflow-hidden">
                            <PreviewPane
                                htmlContent={liveHtmlContent}
                                desiredOutput={scenario.desiredOutput}
                                previewCheck={scenario.previewCheck}
                                onResult={handlePreviewResult}
                            />
                        </div>
                    ) : (
                        <div className="h-64 glass-dark border-t border-white/5 flex flex-col flex-shrink-0 backdrop-blur-3xl">
                            <div className="px-6 py-3 border-b border-white/5 flex items-center justify-between">
                                <span className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Compiler Output</span>
                                <div className="flex items-center gap-2">
                                    <div className="w-1.5 h-1.5 rounded-full bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.5)]"></div>
                                    <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest">System Ready</span>
                                </div>
                            </div>
                            <div className="flex-1 p-6 font-mono text-sm overflow-auto text-slate-300 leading-relaxed scrollbar-hide">
                                <div className="text-slate-500 mb-2">$ scenario-shell --run-tests</div>
                                {output}
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* AI Chat Assistant */}
            <AIChatAssistant
                isOpen={isChatOpen}
                onClose={() => setIsChatOpen(false)}
                codeContext={{
                    activeFile,
                    code: files[activeFile]?.content || '',
                    scenario: scenario.title
                }}
            />
        </div>
    );
};


export default Ide;
