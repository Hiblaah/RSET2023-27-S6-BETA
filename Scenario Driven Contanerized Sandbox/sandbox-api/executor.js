/**
 * executor.js
 * Runs user-supplied JavaScript code inside an isolated Docker container.
 * Security properties:
 *  - --network none       : no outbound internet access
 *  - --memory 64m         : hard memory cap
 *  - --cpus 0.5           : half a CPU core at most
 *  - --read-only          : container filesystem is read-only
 *  - --rm                 : container auto-deleted after run
 *  - runs as non-root (runner user defined in runner.Dockerfile)
 *  - 10-second wall-clock timeout enforced by Node
 */

const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');
const crypto = require('crypto');

const RUNNER_IMAGE = 'sandbox-runner';
const TIMEOUT_MS = 10_000; // 10 seconds

/**
 * executeCode
 * @param {Object} files - Object where keys are filenames and values have 'content'
 * @param {string} entryPoint - The main file to execute (e.g., 'main.js')
 * @returns {Promise<{stdout: string, stderr: string, exitCode: number, timedOut: boolean}>}
 */
async function executeCode(files, entryPoint = 'index.js') {
    // 1. Create a unique temp directory for this execution run
    // Use /app/tmp only when running inside Docker (Docker Compose)
    // Use system tmp otherwise (local development)
    let baseTmpDir;
    try {
        // Try to use /app/tmp if it exists (Docker container)
        if (fs.existsSync('/app/tmp') || fs.existsSync('/app')) {
            baseTmpDir = '/app/tmp';
        } else {
            baseTmpDir = require('os').tmpdir();
        }
    } catch {
        baseTmpDir = require('os').tmpdir();
    }

    if (!fs.existsSync(baseTmpDir)) {
        fs.mkdirSync(baseTmpDir, { recursive: true });
    }
    const runDir = fs.mkdtempSync(path.join(baseTmpDir, 'sandbox-run-'));
    // Set permissions to 755 so that the non-root runner user can access files
    fs.chmodSync(runDir, 0o755);

    // 2. Write all provided files to the runDir
    try {
        for (const [filename, fileObj] of Object.entries(files)) {
            const filePath = path.join(runDir, filename);
            // Ensure subdirectories exist if needed
            const dir = path.dirname(filePath);
            if (!fs.existsSync(dir)) {
                fs.mkdirSync(dir, { recursive: true });
            }
            fs.writeFileSync(filePath, fileObj.content || '', 'utf-8');
        }
        console.log(`[Executor] Files written to ${runDir}:`, Object.keys(files));
    } catch (err) {
        console.error('[Executor] Error writing files:', err);
        return {
            stdout: '',
            stderr: `System error preparing sandbox files: ${err.message}`,
            exitCode: 1,
            timedOut: false
        };
    }

    return new Promise((resolve) => {
        let stdout = '';
        let stderr = '';
        let timedOut = false;

        // Convert path format for Docker mount
        let mountPath;
        let isDockerVolume = false;

        if (process.env.SANDBOX_VOLUME_NAME) {
            // Running inside Docker, mount the named volume
            mountPath = process.env.SANDBOX_VOLUME_NAME;
            isDockerVolume = true;
            console.log(`[Executor] Using Docker volume: ${mountPath}`);
        } else if (process.platform === 'win32') {
            // On Windows host, Docker Desktop requires paths like /c/Users/... (POSIX style)
            // Convert C:\Users\foo\bar → /c/Users/foo/bar
            const absPath = path.resolve(runDir);
            mountPath = '/' + absPath[0].toLowerCase() + absPath.slice(2).replace(/\\/g, '/');
            console.log(`[Executor] Windows path conversion: ${runDir} → ${mountPath}`);
        } else {
            // On Linux/Mac host, use the path as-is
            mountPath = runDir;
            console.log(`[Executor] Unix path: ${mountPath}`);
        }

        // If using Docker volume, we need to know the path INSIDE the volume
        // But since we mount the volume to /sandbox, and the volume itself contains the temp dirs,
        // we need to know where the temp dir is relative to the volume root.
        // In docker-compose, we mount 'sandbox_tmp' to '/app/tmp'.
        // runDir is something like '/app/tmp/sandbox-run-XXXXXX'.
        // So the path relative to the volume root ('/app/tmp') is 'sandbox-run-XXXXXX'.
        let subDir;
        if (isDockerVolume) {
            subDir = path.basename(runDir);
            console.log(`[Executor] Subdirectory in volume: ${subDir}`);
        }

        const runnerEntryPath = `/sandbox/${isDockerVolume ? subDir + '/' : ''}${entryPoint}`;

        console.log(`[Executor] Mount for Docker: ${isDockerVolume ? mountPath + ' (volume)' : mountPath + ' (host path)'}`);
        console.log(`[Executor] Entry point: ${runnerEntryPath}`);

        const dockerArgs = [
            'run',
            '--rm',
            '--network', 'none',
            '--memory', '64m',
            '--memory-swap', '64m',
            '--cpus', '0.5',
            '--read-only',
            '--tmpfs', '/run:size=1m',
            '-v', `${mountPath}:/sandbox:ro`,
            RUNNER_IMAGE,
            'node', runnerEntryPath
        ];

        console.log('Docker command:', dockerArgs.join(' '));
        const proc = spawn('docker', dockerArgs);

        proc.stdout.on('data', (data) => {
            stdout += data.toString();
        });

        proc.stderr.on('data', (data) => {
            stderr += data.toString();
        });

        // Kill after timeout
        const timer = setTimeout(() => {
            timedOut = true;
            proc.kill('SIGKILL');
        }, TIMEOUT_MS);

        proc.on('close', (code) => {
            clearTimeout(timer);
            console.log(`[EXECUTOR] Docker process closed with code: ${code}`);
            console.log(`[EXECUTOR] STDOUT (last 200 chars): ${stdout.slice(-200)}`);
            console.log(`[EXECUTOR] STDERR (last 500 chars): ${stderr.slice(-500)}`);
            console.log(`[EXECUTOR] Mounted path: ${mountPath}`);
            console.log(`[EXECUTOR] Entry path: ${runnerEntryPath}`);
            console.log(`[EXECUTOR] Run directory: ${runDir}`);

            // 4. Cleanup the entire temp directory
            const cleanup = () => {
                try {
                    if (fs.existsSync(runDir)) {
                        fs.rmSync(runDir, { recursive: true, force: true });
                    }
                } catch (err) {
                    setTimeout(cleanup, 100); // retry once if it fails
                }
            };
            cleanup();

            resolve({
                stdout: stdout.slice(0, 8192),
                stderr: stderr.slice(0, 4096),
                exitCode: code ?? 1,
                timedOut,
            });
        });

        proc.on('error', (err) => {
            clearTimeout(timer);
            console.error('Docker spawn error:', err);
            try { fs.rmSync(runDir, { recursive: true, force: true }); } catch (_) { }
            resolve({
                stdout: '',
                stderr: `Failed to spawn Docker: ${err.message}\nMake sure Docker is running and the "sandbox-runner" image is built.`,
                exitCode: 1,
                timedOut: false,
            });
        });
    });
}

module.exports = { executeCode };
