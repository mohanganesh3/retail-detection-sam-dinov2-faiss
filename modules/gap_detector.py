import numpy as np
from config import GAP_THRESHOLD_FACTOR

def find_gaps(detected_boxes):
    """Detects horizontal gaps between detected products."""
    if len(detected_boxes) < 2:
        return []
    
    boxes = np.array(detected_boxes)
    sorted_indices = np.argsort(boxes[:, 0])
    sorted_boxes = boxes[sorted_indices]
    
    avg_width = np.mean(sorted_boxes[:, 2] - sorted_boxes[:, 0])
    min_gap_width = avg_width * GAP_THRESHOLD_FACTOR
    
    gaps = []
    for i in range(len(sorted_boxes) - 1):
        x1_right = sorted_boxes[i][2]
        x2_left = sorted_boxes[i+1][0]
        gap_size = x2_left - x1_right
        if gap_size > min_gap_width:
            y1 = min(sorted_boxes[i][1], sorted_boxes[i+1][1])
            y2 = max(sorted_boxes[i][3], sorted_boxes[i+1][3])
            gaps.append([x1_right, y1, x2_left, y2])
    
    print(f"Detected {len(gaps)} gaps.")
    return gaps