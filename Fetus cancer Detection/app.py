"""
Non-Invasive Prenatal Cancer Detection — FastAPI Backend
=========================================================
Run with:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000

Endpoints:
  POST /predict   — upload ultrasound image -> prediction + Grad-CAM heatmap
  GET  /health    — health check
  GET  /classes   — returns class names + descriptions
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import base64
import io
import time
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════
MODEL_PATH  = Path("Resnet_fineTuning_balanced.pth")   # place your .pth file here
IMG_SIZE    = 224
CLASS_NAMES = ["benign", "malignant", "normal"]
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_INFO = {
    "benign": {
        "label": "Benign Tumor",
        "color": "#f59e0b",
        "icon": "⚠️",
        "description": (
            "A non-cancerous growth was detected. Benign tumors do not "
            "invade nearby tissue or spread to other parts of the body. "
            "Regular monitoring is recommended."
        ),
        "risk": "Low",
        "action": "Schedule follow-up ultrasound in 4–6 weeks",
    },
    "malignant": {
        "label": "Malignant Tumor",
        "color": "#ef4444",
        "icon": "🔴",
        "description": (
            "Indicators of a malignant (cancerous) growth have been detected. "
            "Malignant tumors can invade nearby tissue. Immediate consultation "
            "with a specialist is strongly advised."
        ),
        "risk": "High",
        "action": "Urgent referral to oncology specialist required",
    },
    "normal": {
        "label": "Normal Scan",
        "color": "#10b981",
        "icon": "✅",
        "description": (
            "No abnormalities detected. The scan appears normal with no "
            "signs of benign or malignant tumors. Continue routine prenatal care."
        ),
        "risk": "None",
        "action": "Continue regular prenatal check-ups",
    },
}

# ═══════════════════════════════════════════════════════════════════
#  FASTAPI APP
# ═══════════════════════════════════════════════════════════════════
app = FastAPI(
    title="Prenatal Cancer Detection API",
    description="ResNet-50 powered ultrasound analysis with Grad-CAM",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════
#  MODEL LOADER
# ═══════════════════════════════════════════════════════════════════
def build_model(num_classes=3):
    m = models.resnet50(weights=None)
    in_features = m.fc.in_features
    m.fc = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(0.45),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(512, num_classes),
    )
    return m


def load_model():
    model = build_model()
    if not MODEL_PATH.exists():
        print(f"[WARNING] Model not found at {MODEL_PATH}. Running in DEMO mode.")
        model.to(DEVICE)
        model.eval()
        return model

    ckpt = torch.load(str(MODEL_PATH), map_location=DEVICE, weights_only=False)
    
    if isinstance(ckpt, dict):
        state = ckpt.get("model_state_dict", ckpt)
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError:
            # Try original simpler fc
            model2 = models.resnet50(weights=None)
            model2.fc = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(model2.fc.in_features, 3),
            )
            model2.load_state_dict(state, strict=False)
            model = model2
    else:
        # ckpt is the entire model
        model = ckpt

    model.to(DEVICE)
    model.eval()
    print(f"[OK] Model loaded from {MODEL_PATH} on {DEVICE}")
    return model


model = load_model()

# ═══════════════════════════════════════════════════════════════════
#  TRANSFORMS
# ═══════════════════════════════════════════════════════════════════
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ═══════════════════════════════════════════════════════════════════
#  GRAD-CAM
# ═══════════════════════════════════════════════════════════════════
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model       = model
        self.gradients   = None
        self.activations = None
        target_layer.register_forward_hook(self._fwd_hook)
        target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        self.activations = out.detach().clone()

    def _bwd_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach().clone()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(input_tensor)
        output[0, class_idx].backward()
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = F.relu(cam).squeeze().cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def get_target_layer(mdl):
    try:
        return mdl.layer4[-1].conv3
    except AttributeError:
        for _, module in reversed(list(mdl.named_modules())):
            if isinstance(module, nn.Conv2d):
                return module
        raise ValueError("No Conv2d layer found")


gradcam = GradCAM(model, get_target_layer(model))


def make_overlay(pil_img: Image.Image, cam: np.ndarray) -> str:
    orig_w, orig_h = pil_img.size
    cam_resized    = cv2.resize(cam, (orig_w, orig_h))
    heatmap_bgr    = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap_rgb    = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    img_np         = np.array(pil_img.convert("RGB"))
    overlay        = (0.55 * img_np + 0.45 * heatmap_rgb).clip(0, 255).astype(np.uint8)
    _, buf = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf).decode("utf-8")


# ═══════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ═══════════════════════════════════════════════════════════════════
@app.get("/")
async def root():
    # Try to serve index.html from same folder
    index = Path("index.html")
    if index.exists():
        return HTMLResponse(content=index.read_text(encoding="utf-8"))
    return {"message": "Prenatal Cancer Detection API — visit /docs"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model_loaded": MODEL_PATH.exists(),
    }


@app.get("/classes")
async def classes():
    return {"classes": CLASS_INFO}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG/PNG).")

    t_start  = time.time()
    contents = await file.read()

    try:
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot read image file.")

    tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze().cpu().numpy()

    pred_idx   = int(probs.argmax())
    pred_cls   = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    # Grad-CAM
    tensor_g = preprocess(pil_img).unsqueeze(0).to(DEVICE).requires_grad_(True)
    cam      = gradcam.generate(tensor_g, pred_idx)
    heatmap  = make_overlay(pil_img, cam)

    # Original image as base64
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    orig_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    elapsed = round((time.time() - t_start) * 1000)

    return JSONResponse({
        "prediction": pred_cls,
        "confidence": round(confidence * 100, 2),
        "probabilities": {
            CLASS_NAMES[i]: round(float(probs[i]) * 100, 2) for i in range(3)
        },
        "class_info": CLASS_INFO[pred_cls],
        "heatmap_b64": heatmap,
        "original_b64": orig_b64,
        "inference_ms": elapsed,
        "filename": file.filename,
    })
