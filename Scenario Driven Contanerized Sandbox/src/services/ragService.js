import { scenarios } from '../data/scenarios';

const OLLAMA_BASE_URL = 'http://localhost:11434/api';
const EMBEDDING_MODEL = 'nomic-embed-text';
const FALLBACK_EMBEDDING_MODEL = 'gemma3:1b';
const WIKIPEDIA_API_URL = 'https://en.wikipedia.org/api/rest_v1/page/summary/';

/**
 * Cache for embeddings and project metadata
 */
let scenarioEmbeddings = [];
let projectMetadata = null;

const cosineSimilarity = (vecA, vecB) => {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
        normA += vecA[i] * vecA[i];
        normB += vecB[i] * vecB[i];
    }
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
};

export const generateEmbedding = async (text, model = EMBEDDING_MODEL) => {
    try {
        const response = await fetch(`${OLLAMA_BASE_URL}/embeddings`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: model, prompt: text })
        });

        if (!response.ok) {
            if (model === EMBEDDING_MODEL) return generateEmbedding(text, FALLBACK_EMBEDDING_MODEL);
            throw new Error(`Embedding failed: ${response.status}`);
        }

        const data = await response.json();
        return data.embedding;
    } catch (error) {
        console.error('Error generating embedding:', error);
        throw error;
    }
};

/**
 * Fetch general knowledge summary from Wikipedia
 */
const fetchWikipediaSummary = async (query) => {
    try {
        // Simple heuristic to extract a likely entity name from query
        const cleanQuery = query.replace(/who is|what is|tell me about|where is/gi, '').trim();
        const encodedQuery = encodeURIComponent(cleanQuery.replace(/\s+/g, '_'));

        const response = await fetch(`${WIKIPEDIA_API_URL}${encodedQuery}`);
        if (!response.ok) return null;

        const data = await response.json();
        if (data.type === 'standard' || data.type === 'disambiguation') {
            return `Wikipedia Summary for "${data.title}":\n${data.extract}\nSource: ${data.content_urls.desktop.page}`;
        }
        return null;
    } catch (error) {
        console.error('Wikipedia Fetch Error:', error);
        return null;
    }
};

/**
 * Static project metadata to help AI understand the app structure
 */
const getProjectContext = () => {
    return `
Project Overview: Sandbox_prj
Tech Stack: React, Vite, Tailwind CSS, Lucide Icons, Monaco Editor.
Key Services:
- aiService.js: Handles LLM requests (Groq/LPU).
- ragService.js: Enhanced RAG with local vector search and Wikipedia fallback.
- scenarios.js: Defines interactive coding challenges.
Architecture: Frontend-first coding sandbox with local AI capabilities.
    `.trim();
};

export const indexScenarios = async () => {
    if (scenarioEmbeddings.length > 0) return;

    const indexPromises = scenarios.map(async (scenario) => {
        const scenarioText = `Title: ${scenario.title}\nDescription: ${scenario.description}`.trim();
        try {
            const embedding = await generateEmbedding(scenarioText);
            return { id: scenario.id, title: scenario.title, content: scenarioText, embedding: embedding, originalScenario: scenario };
        } catch (error) { return null; }
    });

    const results = await Promise.all(indexPromises);
    scenarioEmbeddings = results.filter(r => r !== null);
};

export const retrieveContext = async (query, topK = 1) => {
    try {
        let finalContext = '';

        // 1. Try Scenario Search (Local Vector)
        await indexScenarios();
        if (scenarioEmbeddings.length > 0) {
            const queryEmbedding = await generateEmbedding(query);
            const results = scenarioEmbeddings.map(item => ({
                ...item,
                similarity: cosineSimilarity(queryEmbedding, item.embedding)
            }));

            const bestMatch = results.sort((a, b) => b.similarity - a.similarity)[0];
            if (bestMatch && bestMatch.similarity > 0.6) {
                finalContext += `\n\n[Scenario Knowledge]\n${bestMatch.content}\nFiles: ${Object.keys(bestMatch.originalScenario.files).join(', ')}`;
            }
        }

        // 2. Wikipedia Fallback (General Knowledge)
        // Trigger if query sounds like a factual question and no scenario match was strong
        const factualTriggers = /who|what|where|when|why|how many|is it/i;
        if (factualTriggers.test(query)) {
            const wikiSummary = await fetchWikipediaSummary(query);
            if (wikiSummary) {
                finalContext += `\n\n[General Knowledge (Wikipedia)]\n${wikiSummary}`;
            }
        }

        // 3. Always provide a tiny bit of Project Context if query is about the app
        if (/this app|the sandbox|project|how does this work/i.test(query)) {
            finalContext += `\n\n[Project Architecture]\n${getProjectContext()}`;
        }

        return finalContext;
    } catch (error) {
        console.error('RAG Retrieval failed:', error);
        return '';
    }
};
