/**
 * server.js — Sandbox Execution API
 * Receives JavaScript code from the React frontend and executes it
 * safely inside an isolated Docker container.
 *
 * POST /execute   { code: string }  → { stdout, stderr, exitCode, timedOut, passed }
 * GET  /health                      → { status: "ok" }
 */

const express = require('express');
const cors = require('cors');
const os = require('os');
const { executeCode } = require('./executor');

const app = express();
const PORT = 3001;

// Allow requests from the Vite dev server and any localhost origin
app.use(cors({
    origin: ['http://localhost:5173', 'http://localhost:3000', 'http://127.0.0.1:5173'],
    methods: ['GET', 'POST'],
}));
app.use(express.json({ limit: '256kb' }));  // cap payload to prevent abuse

// ── Health check ──────────────────────────────────────────────────────────────
app.get('/health', (_req, res) => {
    res.json({ status: 'ok', service: 'sandbox-api', timestamp: new Date().toISOString() });
});

// ── Execute endpoint ────────────────────────────────────────────────────────
/**
 * Body:
 *   code        {string}  - JavaScript source to execute
 *   testCheck   {string}  - optional: serialized test-check function body,
 *                           e.g. "return code.includes('taxRate') && code.includes('*')"
 *                           Evaluated server-side against the submitted code string.
 */
app.post('/execute', async (req, res) => {
    const { code, files, entryPoint, testCheck } = req.body;

    // Support both single "code" string (legacy) and "files" object
    let executionFiles = files;
    let executionEntryPoint = entryPoint || 'index.js';

    if (!executionFiles && typeof code === 'string') {
        executionFiles = { 'index.js': { content: code } };
        executionEntryPoint = 'index.js';
    }

    if (!executionFiles || typeof executionFiles !== 'object' || Object.keys(executionFiles).length === 0) {
        return res.status(400).json({ error: 'No code or files provided.' });
    }

    try {
        const fs = require('fs');
        console.log('--- DEBUG EXECUTION ---');
        console.log('Files:', Object.keys(executionFiles));
        console.log('EntryPoint:', executionEntryPoint);
        try { console.log('Tmp Dir Contents:', fs.readdirSync(process.platform === 'win32' ? os.tmpdir() : '/tmp')); } catch (e) { console.log('Tmp Dir: (not accessible)'); }

        const result = await executeCode(executionFiles, executionEntryPoint);

        // Determine if the scenario's testCheck passes
        // We evaluate against the entry point file's code AND the execution results
        const mainCode = executionFiles[executionEntryPoint]?.content || '';
        let passed = false;
        if (typeof testCheck === 'string' && testCheck.trim()) {
            try {
                // eslint-disable-next-line no-new-func
                const checkFn = new Function('code', 'stdout', 'stderr', 'files', testCheck);
                passed = !!checkFn(mainCode, result.stdout, result.stderr, executionFiles);
            } catch (e) {
                console.warn('testCheck evaluation error:', e.message);
            }
        }

        return res.json({
            stdout: result.stdout,
            stderr: result.stderr,
            exitCode: result.exitCode,
            timedOut: result.timedOut,
            passed,
        });

    } catch (err) {
        console.error('/execute error:', err);
        return res.status(500).json({
            error: 'Internal server error during execution.',
            details: err.message,
        });
    }
});

// ── Start ─────────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
    console.log(`\n✅ Sandbox API running at http://localhost:${PORT}`);
    console.log('   POST /execute  — run user code in Docker');
    console.log('   GET  /health   — readiness check\n');
});
