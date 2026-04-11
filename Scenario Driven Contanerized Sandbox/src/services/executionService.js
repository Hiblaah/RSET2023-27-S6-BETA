/**
 * executionService.js
 * Thin client layer for communicating with the sandbox-api Docker execution backend.
 */

const SANDBOX_API_URL = import.meta.env.VITE_SANDBOX_API_URL || 'http://localhost:3001';

/**
 * Execute JavaScript code in a sandboxed Docker container.
 *
 * @param {Object} files        - Object of files { "name.js": { content: "..." } }
 * @param {string} entryPoint   - The main file to run
 * @param {Function} testCheck  - The scenario's testCheck function (to evaluate pass/fail)
 * @returns {Promise<{stdout: string, stderr: string, exitCode: number, timedOut: boolean, passed: boolean}>}
 */
export const executeCode = async (files, entryPoint = 'index.js', testCheck = null) => {
    // Serialize the testCheck function body so the API can evaluate it server-side
    let testCheckBody = '';
    if (typeof testCheck === 'function') {
        const fnStr = testCheck.toString();
        const bodyMatch = fnStr.match(/=>\s*([\s\S]+)$/);
        if (bodyMatch) {
            const body = bodyMatch[1].trim();
            testCheckBody = body.startsWith('{')
                ? body.slice(1, -1).trim()
                : `return (${body});`;
        }
    } else if (typeof testCheck === 'string') {
        testCheckBody = testCheck;
    }

    try {
        const response = await fetch(`${SANDBOX_API_URL}/execute`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                files,
                entryPoint,
                testCheck: testCheckBody
            }),
            signal: AbortSignal.timeout(30_000), // 30-second fetch timeout
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.error || `API error: ${response.status}`);
        }

        return await response.json();

    } catch (error) {
        if (error.name === 'TimeoutError' || error.name === 'AbortError') {
            return {
                stdout: '',
                stderr: 'Request timed out. The sandbox API may be unavailable.',
                exitCode: 1,
                timedOut: true,
                passed: false,
            };
        }

        if (error.message.includes('Failed to fetch') || error.message.includes('fetch')) {
            return {
                stdout: '',
                stderr: `⚠️  Cannot connect to Sandbox API at ${SANDBOX_API_URL}.\n\nPlease start the backend:\n  cd sandbox-api\n  npm install\n  node server.js\n\nAnd make sure the Docker "sandbox-runner" image is built:\n  docker build -f runner.Dockerfile -t sandbox-runner .`,
                exitCode: 1,
                timedOut: false,
                passed: false,
            };
        }

        return {
            stdout: '',
            stderr: `Execution error: ${error.message}`,
            exitCode: 1,
            timedOut: false,
            passed: false,
        };
    }
};

/**
 * Check if the sandbox API is reachable.
 * @returns {Promise<boolean>}
 */
export const isSandboxAvailable = async () => {
    try {
        const response = await fetch(`${SANDBOX_API_URL}/health`, {
            signal: AbortSignal.timeout(3_000),
        });
        return response.ok;
    } catch {
        return false;
    }
};
