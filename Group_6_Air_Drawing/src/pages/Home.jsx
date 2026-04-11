import { Link } from "react-router-dom";
import { Move, Layers, Share2, ArrowRight } from "lucide-react";

const Home = () => {
    return (
        <div>
            {/* Hero Section */}
            <section className="hero">
                <div className="container">
                    <h1 className="text-4xl font-bold" style={{ marginBottom: "1rem", fontSize: "3.5rem" }}>
                        Draw in the <span className="text-accent">Air</span>
                    </h1>
                    <p className="text-secondary" style={{ fontSize: "1.25rem", maxWidth: "600px", margin: "0 auto 2rem" }}>
                        Unleash your creativity with our futuristic hand-tracking technology.
                        No mouse, no tablet—just your hands and the air.
                    </p>
                    <div style={{ display: "flex", gap: "1rem", justifyContent: "center" }}>
                        <Link to="/draw" className="btn btn-primary" style={{ fontSize: "1.1rem", padding: "1rem 2rem" }}>
                            Start Drawing <ArrowRight size={20} />
                        </Link>
                        <Link to="/learn" className="btn btn-outline" style={{ fontSize: "1.1rem", padding: "1rem 2rem" }}>
                            Learn to Write
                        </Link>
                    </div>
                </div>
            </section>

            {/* Features Section */}
            <section className="container" style={{ padding: "4rem 1rem" }}>
                <h2 className="text-2xl font-bold text-center" style={{ marginBottom: "3rem" }}>
                    Experience the Future of Art
                </h2>

                <div className="feature-grid">
                    <div className="card">
                        <div className="text-accent" style={{ marginBottom: "1rem" }}>
                            <Move size={40} />
                        </div>
                        <h3 className="text-xl font-bold" style={{ marginBottom: "0.5rem" }}>Hand Tracking</h3>
                        <p className="text-secondary">
                            Precise finger tracking lets you draw naturally in mid-air using just your webcam.
                        </p>
                    </div>

                    <div className="card">
                        <div className="text-accent" style={{ marginBottom: "1rem" }}>
                            <Layers size={40} />
                        </div>
                        <h3 className="text-xl font-bold" style={{ marginBottom: "0.5rem" }}>Virtual Canvas</h3>
                        <p className="text-secondary">
                            Your screen becomes a digital canvas. Draw, erase, and create without limits.
                        </p>
                    </div>

                    <div className="card">
                        <div className="text-accent" style={{ marginBottom: "1rem" }}>
                            <Share2 size={40} />
                        </div>
                        <h3 className="text-xl font-bold" style={{ marginBottom: "0.5rem" }}>Save & Share</h3>
                        <p className="text-secondary">
                            Save your masterpieces and share them with the world instantly.
                        </p>
                    </div>
                </div>
            </section>
        </div>
    );
};

export default Home;
