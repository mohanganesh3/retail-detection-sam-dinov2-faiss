import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
from modules.segment import Segmenter
from modules.embed import build_faiss_index
from modules.classify import FaissClassifier
from modules.gap_detector import find_gaps
from modules.utils import setup_models, draw_detections
from config import SHELVES_DIR, SEGMENTS_DIR, REFERENCE_IMAGES_DIR, VIS_DIR

st.set_page_config(page_title="Retail Product Detector", layout="wide")

@st.cache_resource
def load_models():
    """Loads and caches models."""
    setup_models()
    return Segmenter(), FaissClassifier()

def get_product_classes():
    """Gets product classes from reference images."""
    if not os.path.exists(REFERENCE_IMAGES_DIR):
        return []
    return sorted([d for d in os.listdir(REFERENCE_IMAGES_DIR) if os.path.isdir(os.path.join(REFERENCE_IMAGES_DIR, d))])

st.title("📦 Offline Retail Product & Gap Detector")

# Check FAISS index
if not os.path.exists(FAISS_INDEX_PATH) or not get_product_classes():
    st.warning("No reference images or FAISS index found. Add images to `data/reference_images/` and build the index.")
    if st.button("Build FAISS Index"):
        with st.spinner("Building index..."):
            build_faiss_index()
        st.success("Index built! Refresh the page.")
    st.stop()

# Load models
with st.spinner("Loading models..."):
    segmenter, classifier = load_models()

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    classes = get_product_classes()
    selected_class = st.selectbox("Select Product Class", options=classes)
    similarity_threshold = st.slider("Detection Confidence", 0.80, 1.0, 0.93, 0.01)
    uploaded_file = st.file_uploader("Upload Shelf Image", type=["jpg", "jpeg", "png"])

# Main content
if uploaded_file and selected_class:
    image_path = os.path.join(SHELVES_DIR, uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    if st.button("🚀 Detect Products and Gaps"):
        with col2:
            with st.spinner("Segmenting image..."):
                bboxes, crop_paths = segmenter.segment_image(image_path, SEGMENTS_DIR)
            
            if not bboxes:
                st.warning("No objects detected.")
                st.stop()
            
            st.info(f"Found {len(bboxes)} potential objects. Classifying...")
            matched_indices = classifier.classify_crops(crop_paths, selected_class, similarity_threshold)
            matched_bboxes = [bboxes[i] for i in matched_indices]
            
            count = len(matched_bboxes)
            gaps = find_gaps(matched_bboxes)
            
            st.success("Detection Complete!")
            st.metric(f"Count of '{selected_class}'", count)
            st.metric("Number of Gaps", len(gaps))
            
            image = cv2.imread(image_path)
            output_image = draw_detections(image, matched_bboxes, gaps, count, selected_class)
            output_path = os.path.join(VIS_DIR, f"result_{os.path.basename(image_path)}")
            cv2.imwrite(output_path, output_image)
            
            st.image(output_path, caption="Detection Result", use_column_width=True)
            with open(output_path, "rb") as f:
                st.download_button("Download Result", f, file_name=os.path.basename(output_path))
else:
    st.info("Select a product class and upload an image to begin.")