import React, { useState, useRef, useEffect } from 'react';
import { X, Send, MessageCircle, Sparkles, Loader2, Lightbulb, ChevronDown } from 'lucide-react';
import { sendMessage, getHint, AVAILABLE_MODELS } from '../services/aiService';

const AIChatAssistant = ({ isOpen, onClose, codeContext }) => {
    const [messages, setMessages] = useState([
        {
            role: 'assistant',
            content: '👋 Hi! I\'m your AI coding assistant. I can help you understand the code, debug issues, or answer questions about programming. How can I help you today?'
        }
    ]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [isGettingHint, setIsGettingHint] = useState(false);
    const [selectedModel, setSelectedModel] = useState(AVAILABLE_MODELS[0].id);
    const [isModelMenuOpen, setIsModelMenuOpen] = useState(false);
    const messagesEndRef = useRef(null);
    const inputRef = useRef(null);
    const modelMenuRef = useRef(null);

    const currentModelDetails = AVAILABLE_MODELS.find(m => m.id === selectedModel) || AVAILABLE_MODELS[0];

    // Auto-scroll to bottom when new messages arrive
    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    // Close model menu when clicking outside
    useEffect(() => {
        const handleClickOutside = (event) => {
            if (modelMenuRef.current && !modelMenuRef.current.contains(event.target)) {
                setIsModelMenuOpen(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    // Focus input when chat opens
    useEffect(() => {
        if (isOpen) {
            inputRef.current?.focus();
        }
    }, [isOpen]);

    const handleSendMessage = async () => {
        if (!input.trim() || isLoading) return;

        const userMessage = input.trim();
        setInput('');
        setIsLoading(true);

        // Add user message
        const newMessages = [...messages, { role: 'user', content: userMessage }];
        setMessages(newMessages);

        try {
            // Get AI response
            const aiMessages = newMessages.filter(m => m.role !== 'system');
            const response = await sendMessage(aiMessages, codeContext, selectedModel);

            // Add AI response
            setMessages([...newMessages, { role: 'assistant', content: response }]);
        } catch (error) {
            setMessages([
                ...newMessages,
                {
                    role: 'assistant',
                    content: `⚠️ Error: ${error.message}\n\nPlease make sure you've added your Groq API key to the .env file.`
                }
            ]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleGetHint = async () => {
        if (isGettingHint || isLoading) return;

        setIsGettingHint(true);
        const newMessages = [...messages, { role: 'user', content: '💡 Give me a hint' }];
        setMessages(newMessages);

        try {
            const hint = await getHint(codeContext, selectedModel);
            setMessages([...newMessages, { role: 'assistant', content: `💡 **Hint:**\n\n${hint}` }]);
        } catch (error) {
            setMessages([
                ...newMessages,
                {
                    role: 'assistant',
                    content: `⚠️ Error: ${error.message}`
                }
            ]);
        } finally {
            setIsGettingHint(false);
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    };

    if (!isOpen) return null;

    return (
        <div className="fixed right-0 top-0 h-screen w-[450px] glass-dark border-l border-white/10 flex flex-col z-50 backdrop-blur-3xl animate-slide-in">
            {/* Header */}
            <div className="h-20 border-b border-white/5 flex items-center justify-between px-6 flex-shrink-0">
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center shadow-lg">
                        <Sparkles size={20} className="text-white" />
                    </div>
                    <div className="relative" ref={modelMenuRef}>
                        <button
                            onClick={() => setIsModelMenuOpen(!isModelMenuOpen)}
                            className="flex flex-col text-left group transition-all"
                        >
                            <div className="flex items-center gap-1">
                                <h2 className="text-sm font-black uppercase tracking-tight group-hover:text-purple-400 transition-colors">AI Assistant</h2>
                                <ChevronDown size={12} className={`text-slate-500 transition-transform ${isModelMenuOpen ? 'rotate-180' : ''}`} />
                            </div>
                            <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">{currentModelDetails.name}</p>
                        </button>

                        {/* Model Selection Dropdown */}
                        {isModelMenuOpen && (
                            <div className="absolute top-full left-0 mt-2 w-64 glass-dark border border-white/10 rounded-2xl p-2 shadow-2xl z-[60] animate-in fade-in zoom-in duration-200">
                                {AVAILABLE_MODELS.map((model) => (
                                    <button
                                        key={model.id}
                                        onClick={() => {
                                            setSelectedModel(model.id);
                                            setIsModelMenuOpen(false);
                                        }}
                                        className={`w-full text-left p-3 rounded-xl transition-all mb-1 last:mb-0 ${selectedModel === model.id
                                            ? 'bg-gradient-to-br from-purple-500/20 to-pink-500/20 border border-purple-500/30 text-white'
                                            : 'hover:bg-white/5 text-slate-400 hover:text-white border border-transparent'
                                            }`}
                                    >
                                        <div className="flex items-center justify-between mb-0.5">
                                            <div className="text-xs font-black uppercase tracking-tight">{model.name}</div>
                                            <span className={`text-[8px] px-1.5 py-0.5 rounded-md font-black uppercase tracking-widest border ${model.provider === 'ollama'
                                                ? 'bg-blue-500/10 border-blue-500/30 text-blue-400'
                                                : 'bg-purple-500/10 border-purple-500/30 text-purple-400'
                                                }`}>
                                                {model.provider === 'ollama' ? 'Local' : 'Cloud'}
                                            </span>
                                        </div>
                                        <div className="text-[10px] text-slate-500 font-medium leading-tight">{model.description}</div>
                                    </button>
                                ))}
                            </div>
                        )}
                    </div>
                </div>
                <button
                    onClick={onClose}
                    className="p-2 glass rounded-xl text-slate-400 hover:text-white transition-all"
                >
                    <X size={20} />
                </button>
            </div>

            {/* Messages Area */}
            <div className="flex-1 overflow-y-auto p-6 space-y-4 scrollbar-hide">
                {messages.map((message, index) => (
                    <div
                        key={index}
                        className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                    >
                        <div
                            className={`max-w-[85%] rounded-2xl px-4 py-3 ${message.role === 'user'
                                ? 'bg-gradient-to-br from-purple-500 to-pink-500 text-white shadow-lg'
                                : 'glass border border-white/10 text-slate-200'
                                }`}
                        >
                            <p className="text-sm leading-relaxed whitespace-pre-wrap font-medium">
                                {message.content}
                            </p>
                        </div>
                    </div>
                ))}

                {isLoading && (
                    <div className="flex justify-start">
                        <div className="glass border border-white/10 rounded-2xl px-4 py-3 flex items-center gap-2">
                            <Loader2 size={16} className="animate-spin text-purple-400" />
                            <span className="text-sm text-slate-400 font-medium">Thinking...</span>
                        </div>
                    </div>
                )}

                <div ref={messagesEndRef} />
            </div>

            {/* Quick Actions */}
            <div className="px-6 pb-3 flex gap-2">
                <button
                    onClick={handleGetHint}
                    disabled={isGettingHint || isLoading}
                    className="flex items-center gap-2 px-3 py-2 glass rounded-xl text-slate-400 hover:text-white hover:border-yellow-500/30 border border-white/5 transition-all text-xs font-black uppercase tracking-widest disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    <Lightbulb size={14} />
                    Get Hint
                </button>
            </div>

            {/* Input Area */}
            <div className="border-t border-white/5 p-4 flex-shrink-0">
                <div className="flex gap-2">
                    <textarea
                        ref={inputRef}
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyPress={handleKeyPress}
                        placeholder="Ask me anything about coding..."
                        className="flex-1 bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-purple-500/50 resize-none font-medium"
                        rows={3}
                        disabled={isLoading}
                    />
                    <button
                        onClick={handleSendMessage}
                        disabled={!input.trim() || isLoading}
                        className="w-12 h-12 self-end bg-gradient-to-br from-purple-500 to-pink-500 rounded-xl flex items-center justify-center text-white hover:shadow-lg hover:shadow-purple-500/20 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        <Send size={18} />
                    </button>
                </div>
                <p className="text-[9px] text-slate-600 mt-2 font-bold uppercase tracking-widest">
                    Press Enter to send • Shift+Enter for new line
                </p>
            </div>
        </div>
    );
};

export default AIChatAssistant;
