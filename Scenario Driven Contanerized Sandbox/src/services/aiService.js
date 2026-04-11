/**
 * AI Service — Groq Integration via groq-sdk
 */

import Groq from "groq-sdk";

const GROQ_API_KEY = import.meta.env.VITE_GROQ_API_KEY || '';
const MODEL_ID = 'openai/gpt-oss-20b';

const groq = new Groq({
    apiKey: GROQ_API_KEY,
    dangerouslyAllowBrowser: true // Required for client-side usage in Vite
});

export const AVAILABLE_MODELS = [
    {
        id: 'openai/gpt-oss-20b',
        name: 'GPT OSS 20B',
        description: 'Blazing fast inference via Groq LPU',
        provider: 'groq'
    }
];

/**
 * Serialize codeContext (object or string) to a readable string for the prompt.
 */
function serializeContext(codeContext) {
    if (!codeContext) return '';
    if (typeof codeContext === 'string') return codeContext;
    return `Active File: ${codeContext.activeFile || 'unknown'}
Scenario: ${codeContext.scenario || 'General coding'}
Current Code:
\`\`\`javascript
${codeContext.code || ''}
\`\`\``;
}

/**
 * Send a message to Groq and get a response.
 * @param {Array} messages - [{role: 'user'|'assistant', content: string}]
 * @param {Object|string|null} codeContext - Current code context
 * @param {string} [modelOverride] - Optional
 * @returns {Promise<string>}
 */
export async function sendMessage(messages, codeContext = null, modelOverride = null) {
    if (!GROQ_API_KEY) {
        throw new Error('No Groq API key found. Add VITE_GROQ_API_KEY to your .env file.');
    }

    const contextStr = serializeContext(codeContext);

    const systemPrompt = `You are a helpful AI coding assistant inside a coding sandbox platform.
Help users learn by explaining concepts and debugging code.
Be concise, educational, and friendly. DO NOT give the full solution directly.

${contextStr ? `Code Context:\n${contextStr}\n` : ''}`;

    const completion = await groq.chat.completions.create({
        model: modelOverride || MODEL_ID,
        messages: [
            { role: 'system', content: systemPrompt },
            ...messages
        ],
    });

    return completion.choices[0]?.message?.content || "No response from AI.";
}

/**
 * Get a quick debugging hint for the current scenario.
 * @param {Object|string|null} codeContext
 * @param {string} [modelOverride] - Optional
 * @returns {Promise<string>}
 */
export async function getHint(codeContext, modelOverride = null) {
    if (!GROQ_API_KEY) {
        throw new Error('No Groq API key found. Add VITE_GROQ_API_KEY to your .env file.');
    }

    const contextStr = serializeContext(codeContext);

    const systemPrompt = `You are a coding assistant. Give a short, helpful debugging hint for this code.
Do NOT give the full solution — only guide the user toward the fix.

${contextStr || 'No code context provided.'}`;

    const completion = await groq.chat.completions.create({
        model: modelOverride || MODEL_ID,
        messages: [
            { role: 'system', content: systemPrompt },
            { role: 'user', content: "Give me a hint" }
        ],
    });

    return completion.choices[0]?.message?.content || "No hint available.";
}

/**
 * Explain a piece of code.
 * @param {string} code
 * @returns {Promise<string>}
 */
export async function explainCode(code) {
    return sendMessage([{
        role: 'user',
        content: `Please explain this code:\n\`\`\`javascript\n${code}\n\`\`\``
    }]);
}
