import { Link, useLocation, useNavigate } from "react-router-dom";
import { PenTool, User, LogIn, LogOut, Image as ImageIcon } from "lucide-react";
import { useEffect, useState } from "react";
import { supabase } from "../supabaseClient";

const Navbar = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const isAuthPage = ["/signin", "/signup"].includes(location.pathname);
  const [user, setUser] = useState(null);

  useEffect(() => {
    // Get initial session
    supabase.auth.getUser().then(({ data: { user } }) => {
      setUser(user);
    });

    // Listen for changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user ?? null);
    });

    return () => subscription.unsubscribe();
  }, []);

  const handleSignOut = async () => {
    await supabase.auth.signOut();
    navigate("/");
  };

  if (isAuthPage) return null;

  return (
    <nav style={{
      padding: "1rem 2rem",
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
      background: "rgba(15, 23, 42, 0.8)",
      backdropFilter: "blur(10px)",
      position: "sticky",
      top: 0,
      zIndex: 100,
      borderBottom: "1px solid rgba(255,255,255,0.1)"
    }}>
      <Link to="/" style={{ display: "flex", alignItems: "center", gap: "0.5rem", fontSize: "1.5rem", fontWeight: "bold", color: "white" }}>
        <PenTool className="text-accent" />
        <span>Air<span className="text-accent">Draw</span></span>
      </Link>

      <div style={{ display: "flex", gap: "1rem", alignItems: "center" }}>
        {user ? (
          <>
            <span className="text-secondary" style={{ marginRight: "0.5rem", fontSize: "0.9rem" }}>
              {user.email}
            </span>
            <Link to="/gallery" className="btn btn-outline" style={{ fontSize: "0.9rem", padding: "0.5rem 1rem" }}>
              <ImageIcon size={18} />
              Gallery
            </Link>
            <button onClick={handleSignOut} className="btn btn-outline" style={{ fontSize: "0.9rem", padding: "0.5rem 1rem" }}>
              <LogOut size={18} />
              Sign Out
            </button>
          </>
        ) : (
          <>
            <Link to="/signin" className="btn btn-outline" style={{ fontSize: "0.9rem", padding: "0.5rem 1rem" }}>
              <LogIn size={18} />
              Sign In
            </Link>
            <Link to="/signup" className="btn btn-primary" style={{ fontSize: "0.9rem", padding: "0.5rem 1rem" }}>
              <User size={18} />
              Sign Up
            </Link>
          </>
        )}
      </div>
    </nav>
  );
};

export default Navbar;
