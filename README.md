USER SELECTS PRODUCT CLASS (e.g., "VI-JHON_red")
            ↓
USER UPLOADS SHELF IMAGE (e.g., "shelf1.jpg")
            ↓
[1] SEGMENTATION STEP
    → Load SAM (Segment Anything Model)
    → Input: shelf1.jpg
    → Output: cropped object segments from the shelf
            ↓
[2] EMBEDDING STEP
    → Load DINOv2 (ViT-based vision transformer)
    → For each crop:
        → Extract 768-dim embedding using DINOv2
    → For selected product class:
        → Load precomputed FAISS index from reference images
            ↓
[3] CLASSIFICATION STEP
    → Compare crop embeddings to selected class in FAISS
    → Keep only crops with similarity > threshold (e.g., cosine > 0.9)
    → Output: bounding boxes of matched products
            ↓
[4] COUNTING STEP
    → Count the number of matching crops
    → Visualize bounding boxes on original image
            ↓
[5] GAP DETECTION STEP
    → Sort matched product boxes (left to right, top to bottom)
    → Measure gaps between them using bounding box distances
    → Flag any gaps larger than defined threshold
    → Annotate image with “GAP” warnings
            ↓
[6] FINAL OUTPUT
    → Save visual result to: `outputs/vis/shelf1_detected.jpg`
    → Return:
        - Total count of matched products
        - Detected gap positions
        - Confidence scores (optional)
