import os
import cv2
import numpy as np
import torch
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from config import SAM_MODEL_PATH
from .utils import get_device

class Segmenter:
    """Handles image segmentation using the Segment Anything Model (SAM)."""
    
    def __init__(self, model_type="vit_b"):
        """Initializes SAM with ViT-B model."""
        device = get_device()
        if str(device) == "mps":
            print("MPS detected. Using CPU for SAM due to float64 incompatibility.")
            device = torch.device("cpu")
        
        print(f"Initializing SAM on {device}...")
        sam = sam_model_registry[model_type](checkpoint=SAM_MODEL_PATH)
        sam.to(device)
        self.mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=64,
        pred_iou_thresh=0.82,
        stability_score_thresh=0.88,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
        )
        self.device = device
    
    def segment_image(self, image_path: str, output_dir: str):
        """
        Segments an image into object crops and saves them.
        
        Args:
            image_path (str): Path to the input image.
            output_dir (str): Directory to save cropped segments.
        
        Returns:
            tuple: (bounding_boxes, crop_paths)
                - bounding_boxes: List of [x1, y1, x2, y2]
                - crop_paths: List of paths to saved crop images
        """
        print(f"Segmenting image: {image_path}")
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        masks = self.mask_generator.generate(image_rgb)
        
        bounding_boxes = []
        crop_paths = []
        
        if not masks:
            print("Warning: No objects segmented.")
            return bounding_boxes, crop_paths
        
        os.makedirs(output_dir, exist_ok=True)
        for i, mask in enumerate(sorted(masks, key=lambda x: x['area'], reverse=True)):
            x, y, w, h = mask['bbox']
            crop = image_rgb[y:y+h, x:x+w]
            crop_path = os.path.join(output_dir, f"segment_{i}.jpg")
            cv2.imwrite(crop_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            bounding_boxes.append([x, y, x+w, y+h])
            crop_paths.append(crop_path)
        
        print(f"Generated {len(masks)} segments.")
        return bounding_boxes, crop_paths