import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { LogIn, Mail, Lock, AlertCircle } from "lucide-react";
import { supabase } from "../supabaseClient";

const SignIn = () => {
    const navigate = useNavigate();
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [formData, setFormData] = useState({
        email: "",
        password: "",
    });

    const handleChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleSignIn = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null);

        try {
            const { error } = await supabase.auth.signInWithPassword({
                email: formData.email,
                password: formData.password,
            });

            if (error) throw error;
            navigate("/");
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="flex items-center justify-between" style={{ minHeight: "100vh", padding: "2rem" }}>
            <div className="container" style={{ maxWidth: "400px" }}>
                <div className="card text-center">
                    <h2 className="text-2xl font-bold" style={{ marginBottom: "2rem" }}>Welcome Back</h2>

                    {error && (
                        <div style={{ color: "var(--danger)", marginBottom: "1rem", fontSize: "0.9rem", display: "flex", alignItems: "center", gap: "0.5rem", justifyContent: "center" }}>
                            <AlertCircle size={16} /> {error}
                        </div>
                    )}

                    <form onSubmit={handleSignIn} style={{ textAlign: "left" }}>
                        <div style={{ position: "relative" }}>
                            <Mail size={18} style={{ position: "absolute", top: "14px", left: "12px", color: "var(--text-secondary)" }} />
                            <input
                                type="email"
                                name="email"
                                placeholder="Email Address"
                                className="input"
                                style={{ paddingLeft: "2.5rem" }}
                                value={formData.email}
                                onChange={handleChange}
                                required
                            />
                        </div>

                        <div style={{ position: "relative" }}>
                            <Lock size={18} style={{ position: "absolute", top: "14px", left: "12px", color: "var(--text-secondary)" }} />
                            <input
                                type="password"
                                name="password"
                                placeholder="Password"
                                className="input"
                                style={{ paddingLeft: "2.5rem" }}
                                value={formData.password}
                                onChange={handleChange}
                                required
                            />
                        </div>

                        <button disabled={loading} className="btn btn-primary" style={{ width: "100%", marginTop: "1rem", opacity: loading ? 0.7 : 1 }}>
                            <LogIn size={18} /> {loading ? "Signing In..." : "Sign In"}
                        </button>
                    </form>

                    <p className="text-secondary" style={{ marginTop: "1.5rem", fontSize: "0.9rem" }}>
                        Don't have an account? <Link to="/signup" className="text-accent">Sign Up</Link>
                    </p>
                </div>
            </div>
        </div>
    );
};

export default SignIn;
