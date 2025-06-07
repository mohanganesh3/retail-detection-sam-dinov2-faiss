import argparse
import os
import cv2
from modules.segment import Segmenter
from modules.classify import FaissClassifier
from modules.gap_detector import find_gaps
from modules.utils import draw_detections, setup_models
from config import SHELVES_DIR, SEGMENTS_DIR, REFERENCE_IMAGES_DIR, VIS_DIR

def main(product_class, image_path):
    """Main pipeline for product detection and gap analysis."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image {image_path} not found")
    if product_class not in os.listdir(REFERENCE_IMAGES_DIR):
        raise ValueError(f"Class {product_class} not found in reference images")
    
    setup_models()
    segmenter = Segmenter()
    classifier = FaissClassifier()
    
    bboxes, crop_paths = segmenter.segment_image(image_path, SEGMENTS_DIR)
    matched_indices = classifier.classify_crops(crop_paths, product_class)
    matched_bboxes = [bboxes[i] for i in matched_indices]
    
    count = len(matched_bboxes)
    print(f"Detected {count} instances of {product_class}")
    
    gaps = find_gaps(matched_bboxes)
    print(f"Found {len(gaps)} gaps")
    
    image = cv2.imread(image_path)
    output_image = draw_detections(image, matched_bboxes, gaps, count, product_class)
    output_path = os.path.join(VIS_DIR, f"result_{os.path.basename(image_path)}")
    cv2.imwrite(output_path, output_image)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retail Product Detector")
    parser.add_argument("--product_class", required=True, help="Product class (e.g., vijohn_foam_red)")
    parser.add_argument("--image", required=True, help="Path to shelf image")
    args = parser.parse_args()
    main(args.product_class, args.image)