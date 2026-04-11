
import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Code2, AlertCircle, Eye, EyeOff } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

const Login = () => {
    const navigate = useNavigate();
    const { signIn, isConfigured } = useAuth();
    const [formData, setFormData] = useState({
        username: '',
        password: ''
    });
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);
    const [showPassword, setShowPassword] = useState(false);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setLoading(true);

        // Validation
        if (!formData.username || !formData.password) {
            setError('Please fill in all fields');
            setLoading(false);
            return;
        }

        try {
            const { data, error } = await signIn(formData.username, formData.password);

            if (error) {
                setError(error.message || 'Invalid username or password');
                setLoading(false);
                return;
            }

            if (data) {
                localStorage.clear();
                navigate('/dashboard');
            }
        } catch (err) {
            setError('An unexpected error occurred. Please try again.');
            setLoading(false);
        }
    };

    return (
        <div className="relative min-h-screen flex items-center justify-center overflow-hidden font-sans">
            {/* Background Streaks */}
            <div className="bg-streaks">
                <div className="streak-1"></div>
                <div className="streak-2"></div>
            </div>

            <div className="glass-dark p-10 rounded-2xl shadow-2xl w-full max-w-md border border-white/10 relative z-10 backdrop-blur-3xl">
                <div className="flex flex-col items-center mb-10">
                    <div className="text-white mb-3 p-3 glass rounded-2xl neon-border-magenta">
                        <Code2 size={40} className="text-accent-magenta" />
                    </div>
                    <h1 className="text-4xl font-bold text-white tracking-tight">CodeSandbox</h1>
                    <p className="text-slate-400 mt-2 text-sm uppercase tracking-widest font-semibold">Scenario IDE</p>
                </div>

                {!isConfigured && (
                    <div className="mb-6 p-4 glass rounded-xl flex flex-col gap-2 border-blue-500/30">
                        <div className="flex items-center gap-2 text-blue-400 font-semibold">
                            <AlertCircle size={20} />
                            <span>Development Mode</span>
                        </div>
                        <p className="text-xs text-slate-300">
                            Supabase is not configured yet. Authentication features are disabled, but you can still access the dashboard!
                        </p>
                        <button
                            onClick={() => navigate('/dashboard')}
                            className="mt-2 text-xs font-bold text-white bg-blue-600/20 hover:bg-blue-600/40 p-2 rounded-lg transition-all flex items-center justify-center gap-1 border border-blue-500/30"
                        >
                            Skip to Dashboard →
                        </button>
                    </div>
                )}

                {error && (
                    <div className="mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg flex items-center gap-2 text-red-400">
                        <AlertCircle size={20} />
                        <span className="text-sm">{error}</span>
                    </div>
                )}

                <form onSubmit={handleSubmit} className="space-y-6">
                    <div>
                        <label className="block text-xs font-bold text-slate-400 mb-2 uppercase tracking-wider">
                            Username
                        </label>
                        <input
                            type="text"
                            value={formData.username}
                            onChange={(e) => setFormData({ ...formData, username: e.target.value })}
                            className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all text-white placeholder-slate-600"
                            placeholder="username"
                            disabled={loading}
                        />
                    </div>

                    <div>
                        <label className="block text-xs font-bold text-slate-400 mb-2 uppercase tracking-wider">
                            Password
                        </label>
                        <div className="relative">
                            <input
                                type={showPassword ? "text" : "password"}
                                value={formData.password}
                                onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                                className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all text-white text-2xl tracking-widest placeholder-slate-600 pr-12"
                                placeholder="••••••••"
                                disabled={loading}
                            />
                            <button
                                type="button"
                                onClick={() => setShowPassword(!showPassword)}
                                className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-400 hover:text-white transition-colors"
                            >
                                {showPassword ? <EyeOff size={20} /> : <Eye size={20} />}
                            </button>
                        </div>
                    </div>

                    <button
                        type="submit"
                        disabled={loading}
                        className="w-full bg-white text-[#030303] hover:bg-slate-200 font-bold py-4 rounded-xl transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg hover:shadow-white/20"
                    >
                        {loading ? 'Entering...' : 'Sign In'}
                    </button>
                </form>

                <div className="mt-8 text-center">
                    <Link to="/signup" className="text-slate-400 hover:text-white transition-colors text-sm font-medium">
                        Need an account? <span className="text-purple-400 font-bold underline underline-offset-4">Create one</span>
                    </Link>
                </div>
            </div>
        </div>
    );
};

export default Login;
