import { useEffect, useRef, useState } from "react";
import { Trash2, Save } from "lucide-react";
import { supabase } from "../supabaseClient";

const PRESET_COLORS = [
    { label: "Blue",    value: "#3b82f6" },
    { label: "White",   value: "#ffffff" },
    { label: "Red",     value: "#ef4444" },
    { label: "Green",   value: "#22c55e" },
    { label: "Yellow",  value: "#facc15" },
    { label: "Orange",  value: "#f97316" },
    { label: "Purple",  value: "#a855f7" },
    { label: "Pink",    value: "#ec4899" },
    { label: "Cyan",    value: "#06b6d4" },
    { label: "Lime",    value: "#84cc16" },
];

function AirDrawing() {
    const videoRef = useRef(null);
    const videoCanvasRef = useRef(null);
    const drawCanvasRef = useRef(null);

    // Active color stored in a ref so the MediaPipe callback always reads latest value
    const activeColorRef = useRef("#3b82f6");
    const [selectedColor, setSelectedColor] = useState("#3b82f6");

    const setColor = (color) => {
        activeColorRef.current = color;
        setSelectedColor(color);
    };

    let lastPoint = null;
    let isProcessing = false;

    useEffect(() => {
        const video = videoRef.current;
        const videoCanvas = videoCanvasRef.current;
        const drawCanvas = drawCanvasRef.current;

        if (!video || !videoCanvas || !drawCanvas) return;

        const vCtx = videoCanvas.getContext("2d");
        const dCtx = drawCanvas.getContext("2d");

        const hands = new window.Hands({
            locateFile: (file) =>
                `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
        });

        hands.setOptions({
            maxNumHands: 1,
            modelComplexity: 1,
            minDetectionConfidence: 0.7,
            minTrackingConfidence: 0.7,
        });

        hands.onResults((results) => {
            // Always draw live video
            vCtx.save();
            vCtx.scale(-1, 1);
            vCtx.drawImage(
                video,
                -videoCanvas.width,
                0,
                videoCanvas.width,
                videoCanvas.height
            );
            vCtx.restore();

            if (results.multiHandLandmarks?.length) {
                const lm = results.multiHandLandmarks[0];

                const indexTip = lm[8];
                const indexPIP = lm[6];
                const middleTip = lm[12];
                const ringTip = lm[16];
                const pinkyTip = lm[20];

                const indexUp = indexTip.y < indexPIP.y;
                const middleUp = middleTip.y < lm[10].y;
                const othersDown =
                    ringTip.y > lm[14].y &&
                    pinkyTip.y > lm[18].y;

                const isDrawing = indexUp && !middleUp && othersDown;
                const isErasing = indexUp && middleUp && othersDown;

                if (isDrawing || isErasing) {
                    const x = (1 - indexTip.x) * drawCanvas.width;
                    const y = indexTip.y * drawCanvas.height;

                    if (isErasing) {
                        dCtx.globalCompositeOperation = "destination-out";
                        dCtx.lineWidth = 20;
                        dCtx.lineCap = "round";
                    } else {
                        dCtx.globalCompositeOperation = "source-over";
                        dCtx.strokeStyle = activeColorRef.current;
                        dCtx.lineWidth = 6;
                        dCtx.lineCap = "round";
                    }

                    if (lastPoint) {
                        dCtx.beginPath();
                        dCtx.moveTo(lastPoint.x, lastPoint.y);
                        dCtx.lineTo(x, y);
                        dCtx.stroke();
                    }

                    lastPoint = { x, y };
                } else {
                    lastPoint = null;
                }
            } else {
                lastPoint = null;
            }
        });

        navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
            video.srcObject = stream;
            video.play();

            const processFrame = async () => {
                if (!isProcessing && video.readyState === 4) {
                    isProcessing = true;
                    await hands.send({ image: video });
                    isProcessing = false;
                }
                requestAnimationFrame(processFrame);
            };

            processFrame();
        });
    }, []);

    const clearCanvas = () => {
        const canvas = drawCanvasRef.current;
        canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
    };

    const saveDrawing = async () => {
        const canvas = drawCanvasRef.current;

        const { data: { user } } = await supabase.auth.getUser();
        if (!user) {
            alert("Please sign in to save drawings!");
            return;
        }

        canvas.toBlob(async (blob) => {
            const fileName = `${user.id}/${Date.now()}.png`;
            const { error: uploadError } = await supabase.storage
                .from('drawings')
                .upload(fileName, blob);

            if (uploadError) {
                console.error('Error uploading:', uploadError);
                alert('Failed to upload drawing.');
                return;
            }

            const { data: { publicUrl } } = supabase.storage
                .from('drawings')
                .getPublicUrl(fileName);

            const { error: dbError } = await supabase
                .from('drawings')
                .insert([
                    { user_id: user.id, image_url: publicUrl, title: `Drawing ${new Date().toLocaleTimeString()}` }
                ]);

            if (dbError) {
                console.error('Error saving to db:', dbError);
            } else {
                alert('Drawing saved!');
            }
        });
    };

    return (
        <div style={{ textAlign: "center", padding: "2rem" }}>
            <h2 className="text-2xl font-bold" style={{ marginBottom: "1rem" }}>
                Air Drawing Mode
            </h2>
            <p className="text-secondary" style={{ marginBottom: "0.75rem" }}>
                Point your index finger to draw. Raise your middle finger too to erase.
            </p>
            <p style={{
                marginBottom: "1.5rem",
                display: "inline-flex",
                alignItems: "center",
                gap: "0.5rem",
                background: "rgba(59,130,246,0.12)",
                border: "1px solid rgba(59,130,246,0.35)",
                borderRadius: "8px",
                padding: "0.5rem 1rem",
                fontSize: "0.875rem",
                color: "#93c5fd",
            }}>
                ✏️ <strong>Tip:</strong> Drawing a word? Draw each <strong>letter separately</strong> — clear the canvas between letters if needed!
            </p>

            {/* ── Color Palette ── */}
            <div style={{
                display: "flex",
                justifyContent: "center",
                alignItems: "center",
                flexWrap: "wrap",
                gap: "0.5rem",
                marginBottom: "1.5rem",
            }}>
                {PRESET_COLORS.map(({ label, value }) => (
                    <button
                        key={value}
                        title={label}
                        onClick={() => setColor(value)}
                        style={{
                            width: "32px",
                            height: "32px",
                            borderRadius: "50%",
                            background: value,
                            border: selectedColor === value
                                ? "3px solid #fff"
                                : "2px solid rgba(255,255,255,0.2)",
                            boxShadow: selectedColor === value
                                ? `0 0 0 3px ${value}88`
                                : "none",
                            cursor: "pointer",
                            transition: "transform 0.15s, box-shadow 0.15s",
                            transform: selectedColor === value ? "scale(1.25)" : "scale(1)",
                            padding: 0,
                        }}
                    />
                ))}

                {/* Custom color picker */}
                <label
                    title="Custom color"
                    style={{
                        width: "32px",
                        height: "32px",
                        borderRadius: "50%",
                        border: "2px dashed rgba(255,255,255,0.4)",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        cursor: "pointer",
                        fontSize: "18px",
                        lineHeight: 1,
                        overflow: "hidden",
                        position: "relative",
                        background: "rgba(255,255,255,0.05)",
                    }}
                >
                    🎨
                    <input
                        type="color"
                        value={selectedColor}
                        onChange={(e) => setColor(e.target.value)}
                        style={{
                            opacity: 0,
                            position: "absolute",
                            inset: 0,
                            width: "100%",
                            height: "100%",
                            cursor: "pointer",
                        }}
                    />
                </label>

                {/* Active color preview swatch */}
                <div style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "0.4rem",
                    marginLeft: "0.5rem",
                    fontSize: "0.8rem",
                    color: "rgba(255,255,255,0.6)",
                }}>
                    <span style={{
                        width: "18px",
                        height: "18px",
                        borderRadius: "4px",
                        background: selectedColor,
                        border: "1px solid rgba(255,255,255,0.25)",
                        display: "inline-block",
                    }} />
                    {selectedColor}
                </div>
            </div>

            {/* ── Action Buttons ── */}
            <div style={{ marginBottom: "2rem", display: "flex", justifyContent: "center", gap: "1rem" }}>
                <button onClick={clearCanvas} className="btn btn-outline">
                    <Trash2 size={18} /> Clear
                </button>
                <button onClick={saveDrawing} className="btn btn-primary">
                    <Save size={18} /> Save & Share
                </button>
            </div>

            <div className="drawing-container">
                <video
                    ref={videoRef}
                    autoPlay
                    muted
                    playsInline
                    style={{ display: "none" }}
                />

                {/* Video layer */}
                <canvas
                    ref={videoCanvasRef}
                    width="800"
                    height="600"
                />

                {/* Drawing layer */}
                <canvas
                    ref={drawCanvasRef}
                    width="800"
                    height="600"
                    style={{ pointerEvents: "none" }}
                />
            </div>
        </div>
    );
}

export default AirDrawing;
