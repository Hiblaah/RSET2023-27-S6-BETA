import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css'

console.log('CodeSandbox: Starting React mount...');

try {
    const rootElement = document.getElementById('root');
    if (!rootElement) {
        throw new Error('Root element not found!');
    }

    ReactDOM.createRoot(rootElement).render(
        <React.StrictMode>
            <App />
        </React.StrictMode>,
    );
    console.log('CodeSandbox: React mount called successfully.');
} catch (error) {
    console.error('CodeSandbox: Critical mount error:', error);
    document.body.innerHTML = `<div style="color: red; padding: 20px; background: #1a1a1a; min-height: 100vh;">
        <h1>Frontend Crash</h1>
        <pre>${error.stack || error.message}</pre>
    </div>`;
}
