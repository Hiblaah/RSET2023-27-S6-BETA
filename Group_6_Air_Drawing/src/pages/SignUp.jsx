import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { UserPlus, Mail, Lock, User, AlertCircle } from "lucide-react";
import { supabase } from "../supabaseClient";

const SignUp = () => {
    const navigate = useNavigate();
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [formData, setFormData] = useState({
        fullName: "",
        email: "",
        password: "",
    });

    const handleChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleSignUp = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null);

        try {
            const { data, error: authError } = await supabase.auth.signUp({
                email: formData.email,
                password: formData.password,
            });

            if (authError) throw authError;

            if (data.user) {
                // Create user profile
                const { error: profileError } = await supabase
                    .from("profiles")
                    .insert([
                        {
                            id: data.user.id,
                            full_name: formData.fullName,
                            username: formData.email.split("@")[0], // Simple username generation
                        },
                    ]);

                if (profileError) {
                    console.error("Profile creation error:", profileError);
                    // Verify if profile was actually created by trigger or if this manual insert failed. 
                    // For now we continue as auth account is created.
                }

                navigate("/signin");
            }
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
                    <h2 className="text-2xl font-bold" style={{ marginBottom: "2rem" }}>Create Account</h2>

                    {error && (
                        <div style={{ color: "var(--danger)", marginBottom: "1rem", fontSize: "0.9rem", display: "flex", alignItems: "center", gap: "0.5rem", justifyContent: "center" }}>
                            <AlertCircle size={16} /> {error}
                        </div>
                    )}

                    <form onSubmit={handleSignUp} style={{ textAlign: "left" }}>
                        <div style={{ position: "relative" }}>
                            <User size={18} style={{ position: "absolute", top: "14px", left: "12px", color: "var(--text-secondary)" }} />
                            <input
                                type="text"
                                name="fullName"
                                placeholder="Full Name"
                                className="input"
                                style={{ paddingLeft: "2.5rem" }}
                                value={formData.fullName}
                                onChange={handleChange}
                                required
                            />
                        </div>

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
                                minLength={6}
                            />
                        </div>

                        <button disabled={loading} className="btn btn-primary" style={{ width: "100%", marginTop: "1rem", opacity: loading ? 0.7 : 1 }}>
                            <UserPlus size={18} /> {loading ? "Creating Account..." : "Sign Up"}
                        </button>
                    </form>

                    <p className="text-secondary" style={{ marginTop: "1.5rem", fontSize: "0.9rem" }}>
                        Already have an account? <Link to="/signin" className="text-accent">Sign In</Link>
                    </p>
                </div>
            </div>
        </div>
    );
};

export default SignUp;
