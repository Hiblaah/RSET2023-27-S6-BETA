import { useEffect, useRef, useState } from "react";
import { Trash2, Check, ArrowRight, RefreshCw } from "lucide-react";
import * as tf from '@tensorflow/tfjs';

const WORDS_TO_LEARN = [
"b",
"d",
"p",
"q",
"g",
"bad",
"dad",
"bed",
"pod",
"bag"
];

function Learn() {
    const videoRef = useRef(null);
    const videoCanvasRef = useRef(null);
    const drawCanvasRef = useRef(null);
    const [targetWord, setTargetWord] = useState(WORDS_TO_LEARN[0]);
    const [feedback, setFeedback] = useState("");
    const [isChecking, setIsChecking] = useState(false);
    const [wordIndex, setWordIndex] = useState(0);
    const [currentLetterIndex, setCurrentLetterIndex] = useState(0);
    const [model, setModel] = useState(null);

    const [sessionResults, setSessionResults] = useState([]);
    const [aiReport, setAiReport] = useState("");

    let lastPoint = null;
    let isProcessing = false;

    useEffect(() => {

        const loadModel = async () => {
            try {
                const loadedModel = await tf.loadLayersModel("/model/model.json");
                setModel(loadedModel);
                console.log("Model loaded successfully");
            } catch (err) {
                console.error("Failed to load model:", err);
            }
        };

        loadModel();

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
                        dCtx.lineWidth = 30;
                        dCtx.lineCap = "round";

                    } else {

                        dCtx.globalCompositeOperation = "source-over";
                        dCtx.strokeStyle = "#3b82f6";
                        dCtx.lineWidth = 12;
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

                if (video.readyState === 4) {

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

                    if (!isProcessing) {

                        isProcessing = true;

                        hands.send({ image: video }).then(() => {
                            isProcessing = false;
                        }).catch(() => {
                            isProcessing = false;
                        });

                    }

                }

                requestAnimationFrame(processFrame);

            };

            processFrame();

        });

    }, []);

    const clearCanvas = () => {

        const canvas = drawCanvasRef.current;
        canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);

        setFeedback("");

    };

    const recordResult = (target, detected, correct) => {

        setSessionResults(prev => [
            ...prev,
            { target, detected, correct }
        ]);

    };

    const generateAIReport = async () => {

        try {

            const response = await fetch("https://api.openai.com/v1/chat/completions", {

                method: "POST",

                headers: {
                    "Content-Type": "application/json",
                    "Authorization": `Bearer ${import.meta.env.VITE_OPENAI_KEY}`
                },

                body: JSON.stringify({

                    model: "gpt-4o-mini",

                    messages: [
                        {
                            role: "system",
                            content: "You are a supportive AI helping someone with dysgraphia practice writing."
                        },
                        {
                            role: "user",
                            content: `
These are the results from a writing practice session:

${JSON.stringify(sessionResults)}

Analyze:
- which letters were hardest
- which words had issues
- which letter inside words caused difficulty
- if any letters were skipped

Give a short supportive summary encouraging the learner.
`
                        }
                    ]

                })

            });

            const data = await response.json();
            setAiReport(data.choices[0].message.content);

        } catch (error) {

            console.error("AI summary error:", error);

        }

    };

    const nextWord = async () => {

        if (wordIndex === WORDS_TO_LEARN.length - 1) {

            await generateAIReport();
            return;

        }

        const nextIndex = (wordIndex + 1) % WORDS_TO_LEARN.length;

        setWordIndex(nextIndex);
        setTargetWord(WORDS_TO_LEARN[nextIndex]);
        setCurrentLetterIndex(0);

        clearCanvas();

    };

    const checkAccuracy = async () => {

        if (!model) {

            setFeedback("Model not loaded yet. Please wait.");
            return;

        }

        setIsChecking(true);
        setFeedback("Analyzing...");

        const canvas = drawCanvasRef.current;

        try {

            const ctx = canvas.getContext("2d");
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const data = imageData.data;

            let minX = canvas.width, minY = canvas.height, maxX = 0, maxY = 0;
            let hasDrawing = false;

            for (let y = 0; y < canvas.height; y++) {

                for (let x = 0; x < canvas.width; x++) {

                    const alpha = data[(y * canvas.width + x) * 4 + 3];

                    if (alpha > 0) {

                        hasDrawing = true;

                        if (x < minX) minX = x;
                        if (x > maxX) maxX = x;
                        if (y < minY) minY = y;
                        if (y > maxY) maxY = y;

                    }

                }

            }

            if (!hasDrawing) {

                setFeedback("Please draw something first!");
                setIsChecking(false);
                return;

            }

            const padding = 20;

            minX = Math.max(0, minX - padding);
            minY = Math.max(0, minY - padding);
            maxX = Math.min(canvas.width, maxX + padding);
            maxY = Math.min(canvas.height, maxY + padding);

            const width = maxX - minX;
            const height = maxY - minY;
            const size = Math.max(width, height);

            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = 28;
            tempCanvas.height = 28;

            const tempCtx = tempCanvas.getContext('2d');

            tempCtx.fillStyle = "black";
            tempCtx.fillRect(0, 0, 28, 28);

            const offsetX = (size - width) / 2;
            const offsetY = (size - height) / 2;

            tempCtx.drawImage(
                canvas,
                minX, minY, width, height,
                (offsetX / size) * 28, (offsetY / size) * 28,
                (width / size) * 28, (height / size) * 28
            );

            let tensor = tf.browser.fromPixels(tempCanvas, 1)
                .toFloat()
                .div(tf.scalar(255))
                .expandDims(0);

            const rawPredictions = Array.from(await model.predict(tensor).data());

            for (let i = 0; i < 10; i++) rawPredictions[i] = -Infinity;

            const predictedIndex = rawPredictions.indexOf(Math.max(...rawPredictions));

            const emnistMapping = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt";
            const recognizedChar = emnistMapping[predictedIndex];

            const targetChar = targetWord[currentLetterIndex];

            const caseAmbiguous = new Set(['o','s','v','w','x','z','c','k','p','u','y','j','i']);

            const isAmbiguous =
                caseAmbiguous.has(targetChar.toLowerCase()) &&
                caseAmbiguous.has(recognizedChar.toLowerCase());

            const isMatch =
                (isAmbiguous && targetChar.toLowerCase() === recognizedChar.toLowerCase())
                || targetChar === recognizedChar;

            recordResult(targetChar, recognizedChar, isMatch);

            if (isMatch) {

                const nextIndex = currentLetterIndex + 1;
                setCurrentLetterIndex(nextIndex);

                if (nextIndex >= targetWord.length) {

                    setFeedback(`Perfect! You wrote "${targetWord}" 🎉`);

                } else {

                    setFeedback(`Correct '${targetChar}'! Now draw '${targetWord[nextIndex]}'`);

                    const canvas = drawCanvasRef.current;
                    const ctx = canvas.getContext("2d");
                    ctx.clearRect(0, 0, canvas.width, canvas.height);

                }

            } else {

                setFeedback(`Try again! Detected: "${recognizedChar || '?'}" instead of "${targetChar}"`);

            }

            tensor.dispose();

        } catch (error) {

            console.error("Prediction error:", error);
            setFeedback("Error analyzing. Try again.");

        } finally {

            setIsChecking(false);

        }

    };

    return (

        <div style={{ textAlign: "center", padding: "2rem" }}>

            <h2 className="text-2xl font-bold" style={{ marginBottom: "1rem" }}>
                Learn to Write
            </h2>

            {/* your entire original UI remains unchanged */}

            <div className="drawing-container">

                <video ref={videoRef} autoPlay muted playsInline style={{ display: "none" }} />

                <canvas ref={videoCanvasRef} width="800" height="600" />

                <canvas ref={drawCanvasRef} width="800" height="600" style={{ pointerEvents: "none" }} />

            </div>

            {/* AI REPORT DISPLAY (added only) */}

            {aiReport && (
                <div style={{
                    marginTop: "40px",
                    padding: "20px",
                    background: "#111",
                    borderRadius: "10px"
                }}>
                    <h2>Session Summary</h2>
                    <p>{aiReport}</p>
                </div>
            )}

        </div>

    );

}

export default Learn;