"""
app/tools/mri_classifier.py

EfficientNet-B0 MRI Classifier – inference utility.

Usage:
    from app.tools.mri_classifier import predict_mri
    result = predict_mri("path/to/scan.jpg")
    # → {"class": "glioma", "class_index": 0, "confidence": 0.97, "probabilities": {...}}
"""

import os
from functools import lru_cache
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

# ---------------------------------------------------------------------------
# Constants  (must match your training configuration)
# ---------------------------------------------------------------------------

IMG_SIZE: tuple[int, int] = (224, 224)

# Class labels in LabelEncoder alphabetical order (matches training labels 0-3)
CLASS_NAMES: list[str] = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES: int = len(CLASS_NAMES)

# Default checkpoint path – override via environment variable or argument
DEFAULT_CHECKPOINT: str = os.getenv("MRI_EFFICIENTNET_CHECKPOINT_PATH", "efficientnet_best.pt")


# ---------------------------------------------------------------------------
# Model definition  (mirrors the EfficientNet-B0 head used in training)
# ---------------------------------------------------------------------------

def _build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    Recreate the EfficientNet-B0 + custom classification head.

    weights=None because we load our own checkpoint immediately after.
    """
    model = models.efficientnet_b0(weights=None)
    in_features: int = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.20),
        nn.Linear(512, num_classes),
    )
    return model


# ---------------------------------------------------------------------------
# Checkpoint loading  (mirrors torch.load + load_state_dict)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_model(checkpoint_path: str) -> tuple[nn.Module, torch.device]:
    """
    Load the EfficientNet-B0 checkpoint once; cache it for subsequent calls.

    The checkpoint dict is expected to contain the key 'model_state' as saved
    during training:
        torch.save({"epoch": ..., "model_state": model.state_dict(), ...}, path)

    Returns:
        (model, device)  – model is in eval mode and moved to device.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model()

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: '{checkpoint_path}'.\n"
            "Set the MRI_EFFICIENTNET_CHECKPOINT_PATH environment variable or pass the "
            "checkpoint_path argument explicitly."
        )

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, device


# ---------------------------------------------------------------------------
# Preprocessing  (same as the ResNet implementation)
# ---------------------------------------------------------------------------

def _crop_brain_region(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return img

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    pad = 5
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(img.shape[1] - x, w + 2 * pad)
    h = min(img.shape[0] - y, h + 2 * pad)

    return img[y : y + h, x : x + w]


def _preprocess_image(image_path: str) -> Optional[torch.Tensor]:
    img = cv2.imread(image_path)
    if img is None:
        return None

    img = _crop_brain_region(img)
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0

    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return tensor


# ---------------------------------------------------------------------------
# Public inference API
# ---------------------------------------------------------------------------

def predict_mri(
    image_path: str,
    checkpoint_path: str = DEFAULT_CHECKPOINT,
) -> dict:
    model, device = _load_model(checkpoint_path)

    tensor = _preprocess_image(image_path)
    if tensor is None:
        raise ValueError(
            f"Could not load image: '{image_path}'. "
            "Check that the file exists and is a supported format."
        )

    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        probs_np = probs.cpu().numpy()

    class_idx: int = int(np.argmax(probs_np))
    confidence: float = float(probs_np[class_idx])
    class_label: str = CLASS_NAMES[class_idx]

    return {
        "class": class_label,
        "class_index": class_idx,
        "confidence": round(confidence, 4),
        "probabilities": {
            CLASS_NAMES[i]: round(float(probs_np[i]), 4)
            for i in range(NUM_CLASSES)
        },
    }
