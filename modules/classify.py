import os
import numpy as np
import faiss
from PIL import Image
from config import FAISS_INDEX_PATH, LABELS_PATH, SIMILARITY_THRESHOLD
from .embed import DinoV2Embedder

class FaissClassifier:
    """Classifies image crops using FAISS and DINOv2 embeddings."""
    
    def __init__(self):
        if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(LABELS_PATH):
            raise FileNotFoundError("FAISS index or labels not found. Run embed.py first.")
        self.index = faiss.read_index(FAISS_INDEX_PATH)
        self.labels = np.load(LABELS_PATH)
        self.embedder = DinoV2Embedder()
        print(f"Classifier loaded with {self.index.ntotal} reference items.")
    
    def classify_one(self, embedding, target_class, similarity_threshold=SIMILARITY_THRESHOLD):
        """Classifies a single embedding against the target class."""
        embedding_np = embedding.astype('float32').reshape(1, -1)
        D, I = self.index.search(embedding_np, k=1)
        return D[0][0] >= similarity_threshold and self.labels[I[0][0]] == target_class
    
    def classify_crops(self, crop_paths, target_class, similarity_threshold=SIMILARITY_THRESHOLD):
        """Classifies multiple crop images against the target class."""
        if self.index.ntotal == 0:
            print("Warning: FAISS index is empty.")
            return []
        
        embeddings = []
        for crop_path in crop_paths:
            image = Image.open(crop_path)
            embedding = self.embedder.get_embedding(image)
            embeddings.append(embedding)
        
        embeddings_np = np.array(embeddings).astype('float32')
        D, I = self.index.search(embeddings_np, k=1)
        
        matched_indices = []
        for i, (sim, idx) in enumerate(zip(D.flatten(), I.flatten())):
            if sim >= similarity_threshold and self.labels[idx] == target_class:
                matched_indices.append(i)
        
        print(f"Found {len(matched_indices)} matches for '{target_class}'.")
        return matched_indices