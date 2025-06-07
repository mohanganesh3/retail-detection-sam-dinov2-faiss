import os
import torch
import requests
from tqdm import tqdm
import cv2
import numpy as np
from config import SAM_MODEL_PATH, DINO_MODEL_NAME

def get_device():
    """Returns the best available device (CPU for Mac M1 stability)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("MPS available but using CPU for SAM compatibility.")
        return torch.device("cpu")
    return torch.device("cpu")

def download_model(url, path):
    """Downloads a file from a URL with a progress bar."""
    if os.path.exists(path):
        print(f"Model exists at {path}")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(path, 'wb') as f, tqdm(total=total_size, unit='iB', unit_scale=True) as bar:
        for data in response.iter_content(1024):
            bar.update(len(data))
            f.write(data)

def setup_models():
    """Downloads SAM model (DINOv2 handled by transformers)."""
    print("Setting up models...")
    download_model("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth", SAM_MODEL_PATH)
    print("Model setup complete.")

def draw_detections(image, boxes, gaps, product_count, class_name):
    """Draws bounding boxes, gaps, and info on the image."""
    output_image = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for gap in gaps:
        x1, y1, x2, y2 = map(int, gap)
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(output_image, "GAP", (x1 + 5, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    info_text = f"Class: {class_name} | Count: {product_count} | Gaps: {len(gaps)}"
    cv2.rectangle(output_image, (0, 0), (output_image.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(output_image, info_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return output_image