import os
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm
import faiss
from config import EMBEDDINGS_DIR, FAISS_INDEX_PATH, LABELS_PATH, REFERENCE_IMAGES_DIR, DINO_MODEL_NAME
from .utils import get_device

class DinoV2Embedder:
    """Handles DINOv2 embedding extraction."""
    
    def __init__(self):
        self.device = get_device()
        self.processor = AutoImageProcessor.from_pretrained(DINO_MODEL_NAME, use_fast=True)
        self.model = AutoModel.from_pretrained(DINO_MODEL_NAME).to(self.device)
    
    def get_embedding(self, image: Image.Image):
        """Generates a normalized DINOv2 embedding for a PIL image."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
        return embedding / np.linalg.norm(embedding)  # Normalize

def build_faiss_index():
    """Builds a FAISS index from reference images."""
    print("Building FAISS index...")
    embedder = DinoV2Embedder()
    embeddings = []
    labels = []
    
    class_folders = sorted([d for d in os.listdir(REFERENCE_IMAGES_DIR) if os.path.isdir(os.path.join(REFERENCE_IMAGES_DIR, d))])
    if not class_folders:
        raise ValueError("No class folders found in reference_images.")
    
    for class_name in tqdm(class_folders, desc="Processing classes"):
        class_path = os.path.join(REFERENCE_IMAGES_DIR, class_name)
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for image_file in image_files:
            try:
                image_path = os.path.join(class_path, image_file)
                image = Image.open(image_path)
                embedding = embedder.get_embedding(image)
                embeddings.append(embedding)
                labels.append(class_name)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
    
    if not embeddings:
        raise ValueError("No valid images found to build the index.")
    
    embeddings_np = np.array(embeddings).astype('float32')
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    index.add(embeddings_np)
    
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)
    np.save(LABELS_PATH, np.array(labels))
    
    print(f"FAISS index built with {index.ntotal} vectors for classes: {list(np.unique(labels))}")

if __name__ == "__main__":
    build_faiss_index()