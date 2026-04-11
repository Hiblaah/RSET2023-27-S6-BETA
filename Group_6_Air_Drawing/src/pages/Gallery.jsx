import { useEffect, useState } from "react";
import { supabase } from "../supabaseClient";
import { Trash2 } from "lucide-react";

const Gallery = () => {
    const [drawings, setDrawings] = useState([]);
    const [loading, setLoading] = useState(true);
    const [user, setUser] = useState(null);

    useEffect(() => {
        const fetchDrawings = async () => {
            const { data: { user } } = await supabase.auth.getUser();
            setUser(user);

            if (user) {
                const { data, error } = await supabase
                    .from("drawings")
                    .select("*")
                    .order("created_at", { ascending: false });

                if (!error) setDrawings(data);
            }
            setLoading(false);
        };

        fetchDrawings();
    }, []);

    const deleteDrawing = async (id) => {
        const { error } = await supabase.from("drawings").delete().eq("id", id);
        if (!error) {
            setDrawings(drawings.filter((d) => d.id !== id));
        }
    };

    if (loading) return <div className="text-center" style={{ padding: "4rem" }}>Loading...</div>;

    if (!user) return (
        <div className="text-center" style={{ padding: "4rem" }}>
            <h2 className="text-2xl font-bold">Please Sign In</h2>
            <p className="text-secondary">You need to be logged in to view your gallery.</p>
        </div>
    );

    return (
        <div className="container" style={{ padding: "4rem 1rem" }}>
            <h2 className="text-2xl font-bold text-center" style={{ marginBottom: "3rem" }}>
                Your Gallery
            </h2>

            {drawings.length === 0 ? (
                <p className="text-center text-secondary">No drawings yet. Go create some!</p>
            ) : (
                <div className="feature-grid">
                    {drawings.map((drawing) => (
                        <div key={drawing.id} className="card" style={{ padding: "1rem" }}>
                            <img
                                src={drawing.image_url}
                                alt="Drawing"
                                style={{ width: "100%", borderRadius: "0.5rem", marginBottom: "1rem", backgroundColor: "white" }}
                            />
                            <div className="flex justify-between items-center">
                                <span className="text-secondary" style={{ fontSize: "0.8rem" }}>
                                    {new Date(drawing.created_at).toLocaleDateString()}
                                </span>
                                <button
                                    onClick={() => deleteDrawing(drawing.id)}
                                    className="btn btn-outline"
                                    style={{ padding: "0.25rem 0.5rem", fontSize: "0.8rem", color: "var(--danger)", borderColor: "var(--danger)" }}
                                >
                                    <Trash2 size={14} />
                                </button>
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

export default Gallery;
