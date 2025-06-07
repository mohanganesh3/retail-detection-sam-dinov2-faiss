import os

# Project root directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directories
DATA_DIR = os.path.join(ROOT_DIR, "data")
SHELVES_DIR = os.path.join(DATA_DIR, "shelves")
REFERENCE_IMAGES_DIR = os.path.join(DATA_DIR, "reference_images")

# Model and embeddings directories
MODELS_DIR = os.path.join(ROOT_DIR, "models")
EMBEDDINGS_DIR = os.path.join(ROOT_DIR, "embeddings")
FAISS_INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "faiss_index.index")
LABELS_PATH = os.path.join(EMBEDDINGS_DIR, "labels.npy")

# Output directories
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
SEGMENTS_DIR = os.path.join(OUTPUTS_DIR, "segments")
VIS_DIR = os.path.join(OUTPUTS_DIR, "vis")

# Model configurations
SAM_MODEL_PATH = os.path.join(MODELS_DIR, "sam_vit_b_01ec64.pth")
DINO_MODEL_NAME = "facebook/dinov2-base"
SIMILARITY_THRESHOLD = 0.93  # Cosine similarity threshold for classification
GAP_THRESHOLD_FACTOR = 0.75  # Multiplier for expected product width to detect gaps

# Ensure directories exist
for dir_path in [DATA_DIR, SHELVES_DIR, REFERENCE_IMAGES_DIR, MODELS_DIR, EMBEDDINGS_DIR, OUTPUTS_DIR, SEGMENTS_DIR, VIS_DIR]:
    os.makedirs(dir_path, exist_ok=True)