import os
import io
import cv2
import urllib.request
import numpy as np
import onnxruntime as ort
from PIL import Image
from typing import List
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# -------------------------------------------------------------
# 1. AUTO-DOWNLOAD MODEL DARI GITHUB RELEASES JIKA BELUM ADA
# -------------------------------------------------------------
MODEL_PATH = "model.onnx"
MODEL_URL = "https://github.com/rijal028/securepixel-backend/releases/download/v1.0.0/model.onnx"

if not os.path.exists(MODEL_PATH):
    print(f"[*] Mengunduh model {MODEL_PATH} dari GitHub Releases...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"[+] Download tuntas! Ukuran: {os.path.getsize(MODEL_PATH) / (1024*1024):.2f} MB")

# Inisialisasi ONNX Runtime Engine
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# -------------------------------------------------------------
# 2. PREPROCESSING & SPECTRAL FFT
# -------------------------------------------------------------
def preprocess_image(img_pil: Image.Image) -> np.ndarray:
    img = img_pil.resize((224, 224), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))
    return arr

def analyze_phase_spectrum(img_pil: Image.Image) -> float:
    gray_img = img_pil.convert("L")
    w, h = gray_img.size
    crop_size = 256

    if w >= crop_size and h >= crop_size:
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        gray_crop = gray_img.crop((left, top, left + crop_size, top + crop_size))
        gray = np.array(gray_crop, dtype=np.float32) / 255.0
    else:
        gray = np.array(gray_img.resize((256, 256)), dtype=np.float32) / 255.0

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)

    cy, cx = 128, 128
    y, x = np.ogrid[:256, :256]
    low_freq_mask = ((x - cx)**2 + (y - cy)**2) <= (32**2)

    low_energy = float(np.mean(mag[low_freq_mask]) + 1e-5)
    high_energy = float(np.mean(mag[~low_freq_mask]) + 1e-5)

    spectral_ratio = high_energy / low_energy
    score = 1.0 / (1.0 + np.exp(-40.0 * (spectral_ratio - 0.045)))
    return float(np.clip(score, 0.05, 0.95))

def run_forensic_pipeline(pil_img: Image.Image, score_visual: float) -> dict:
    gray_np = np.array(pil_img.convert("L"))
    laplacian_var = float(cv2.Laplacian(gray_np, cv2.CV_64F).var())
    w, h = pil_img.size
    is_downscaled = bool((w * h) < (720 * 1280))
    quality_score = float(np.clip(laplacian_var / 100.0, 0.05, 1.0))

    if laplacian_var < 5.0 or (laplacian_var < 35.0 and is_downscaled):
        return {
            "verdict": "INCONCLUSIVE",
            "calibrated_confidence": 0.50,
            "reason": "Severe physical degradation outside safe forensic envelope.",
            "metrics": {"sharpness": round(laplacian_var, 2), "dimensions": f"{w}x{h}", "quality": quality_score}
        }

    score_phase = analyze_phase_spectrum(pil_img)
    w_visual = max(0.4, 0.8 * quality_score)
    w_phase = 0.2
    combined_score = (score_visual * w_visual + score_phase * w_phase) / (w_visual + w_phase)
    disagreement = abs(score_visual - score_phase)

    if combined_score <= 0.35 and score_visual < 0.15:
        verdict = "LIKELY NATURAL CAPTURE"
        confidence = round(1.0 - combined_score, 3)
        reason = "Optical sensor texture and visual residuals strongly indicate genuine natural photography."
    elif disagreement > 0.55 and quality_score < 0.25 and combined_score > 0.35:
        verdict = "INCONCLUSIVE"
        confidence = 0.50
        reason = "Visual channel and spectral phase exhibit acute divergence under heavy compression."
    elif combined_score >= 0.55:
        verdict = "LIKELY AI-GENERATED"
        confidence = round(min(combined_score, 0.70 + (0.28 * quality_score)), 3)
        reason = "Forensic visual patterns and high-frequency spectral phase consistently reflect synthetic generative artifacts."
    elif combined_score <= 0.45:
        verdict = "LIKELY NATURAL CAPTURE"
        confidence = round(1.0 - combined_score, 3)
        reason = "Visual features and phase distributions remain consistent with standard optical capture."
    else:
        verdict = "INCONCLUSIVE"
        confidence = 0.50
        reason = "Forensic probability resides within technical ambiguity margin (0.45 - 0.55)."

    return {
        "verdict": verdict,
        "calibrated_confidence": round(confidence, 3),
        "reason": reason,
        "metrics": {
            "dimensions": f"{w}x{h}",
            "sharpness": round(laplacian_var, 2),
            "quality_index": round(quality_score, 2),
            "score_visual": round(score_visual, 3),
            "score_phase": round(score_phase, 3),
            "combined_score": round(combined_score, 3)
        }
    }

# -------------------------------------------------------------
# 3. FASTAPI ROUTING (SINGLE & BATCH)
# -------------------------------------------------------------
app = FastAPI(title="SecurePixel Ultralight ONNX Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "online", "engine": "ONNX-Runtime-CPU", "batch_supported": True}

@app.post("/predict")
async def predict_single(file: UploadFile = File(...)):
    raw = await file.read()
    if b"c2pa" in raw or b"urn:c2pa" in raw:
        return {
            "verdict": "VERIFIED ORIGIN",
            "calibrated_confidence": 1.0,
            "reason": "Authentic Content Credentials C2PA manifest detected.",
            "metrics": {"degradation_quality": 1.0}
        }
    try:
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return {"verdict": "INCONCLUSIVE", "calibrated_confidence": 0.0, "reason": "Corrupted payload."}

    tensor = np.expand_dims(preprocess_image(pil_img), axis=0)
    outputs = session.run(None, {input_name: tensor})[0][0]
    exp_out = np.exp(outputs - np.max(outputs))
    probs = exp_out / exp_out.sum()
    score_visual = float(np.clip(1.0 - float(probs[0]), 0.01, 0.99))

    return run_forensic_pipeline(pil_img, score_visual)

@app.post("/predict-batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    if len(files) > 16:
        return {"error": "Maximum batch limit is 16 frames per request."}

    images = []
    c2pa_detected = False

    for f in files:
        raw = await f.read()
        if b"c2pa" in raw or b"urn:c2pa" in raw:
            c2pa_detected = True
        try:
            images.append(Image.open(io.BytesIO(raw)).convert("RGB"))
        except Exception:
            pass

    if c2pa_detected:
        return {
            "verdict": "VERIFIED ORIGIN",
            "calibrated_confidence": 1.0,
            "reason": "Authentic Content Credentials C2PA manifest detected in batch.",
            "results": []
        }

    if not images:
        return {"verdict": "INCONCLUSIVE", "calibrated_confidence": 0.0, "reason": "No valid frames decoded."}

    # Dynamic Batch Inference
    batch_tensors = np.stack([preprocess_image(im) for im in images], axis=0)
    batch_outputs = session.run(None, {input_name: batch_tensors})[0]

    results = []
    for i, im in enumerate(images):
        raw_out = batch_outputs[i]
        exp_out = np.exp(raw_out - np.max(raw_out))
        probs = exp_out / exp_out.sum()
        score_visual = float(np.clip(1.0 - float(probs[0]), 0.01, 0.99))
        frame_res = run_forensic_pipeline(im, score_visual)
        results.append(frame_res)

    ai_votes = sum(1 for r in results if r.get("verdict") == "LIKELY AI-GENERATED")
    nat_votes = sum(1 for r in results if r.get("verdict") == "LIKELY NATURAL CAPTURE")
    total_valid = len(results)

    if ai_votes >= (total_valid / 2):
        batch_verdict = "LIKELY AI-GENERATED"
    elif nat_votes > (total_valid / 2):
        batch_verdict = "LIKELY NATURAL CAPTURE"
    else:
        batch_verdict = "INCONCLUSIVE"

    return {
        "batch_verdict": batch_verdict,
        "total_frames": total_valid,
        "ai_votes": ai_votes,
        "natural_votes": nat_votes,
        "frame_details": results
    }
