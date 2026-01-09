import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import time

# Internal imports
from utils.image_utils import ImageUtils
from utils.visualization import VisualizationUtils
from services.orb_service import ORBService
from services.sift_service import SIFTService
from services.matcher import MatcherService
from services.metrics import MetricsService

# Page configuration
st.set_page_config(
    page_title="VisionMatch - Feature Comparison",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3e4149;
    }
    .stAlert {
        border-radius: 10px;
    }
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    st.title("🔍 VisionMatch Pro")
    st.subheader("Mathematically Robust Feature Comparison")
    
    # Sidebar
    st.sidebar.title("Controls")
    algo_choice = st.sidebar.selectbox("Choose Algorithm", ["ORB", "SIFT"])
    
    st.sidebar.divider()
    
    # Algorithmic Params
    with st.sidebar.expander("Parameters", expanded=True):
        n_features = st.slider("Max Keypoints", 500, 5000, 2000) if algo_choice == "ORB" else None
        ratio_test = st.checkbox("Use Lowe's Ratio Test", value=True)
        ratio_threshold = st.slider("Ratio Threshold", 0.1, 1.0, 0.75) if (ratio_test or algo_choice == "SIFT") else 0.75
    
    # Visualization Settings
    with st.sidebar.expander("Visualization", expanded=True):
        vis_mode = st.radio("Display Mode", ["Inliers Only", "Top-K Matches", "Raw Matches"], index=0)
        top_k = st.slider("Top K matches", 10, 500, 50) if vis_mode == "Top-K Matches" else None

    # File uploads
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Image A")
        file_a = st.file_uploader("Upload Image A", type=["jpg", "jpeg", "png"], key="img_a")
    
    with col2:
        st.write("### Image B")
        file_b = st.file_uploader("Upload Image B", type=["jpg", "jpeg", "png"], key="img_b")

    if file_a and file_b:
        # Load images
        img_a = ImageUtils.load_image(file_a)
        img_b = ImageUtils.load_image(file_b)
        
        if st.sidebar.button("Analyze & Verify", type="primary"):
            with st.spinner(f"Running {algo_choice} + RANSAC Verification..."):
                # Initialize services
                detector = ORBService(n_features=n_features) if algo_choice == "ORB" else SIFTService()
                matcher = MatcherService(method=algo_choice)
                
                # 1. Feature Extraction
                res_a = detector.detect_and_compute(img_a)
                res_b = detector.detect_and_compute(img_b)
                
                # 2. Matching & Geometric Verification
                match_results = matcher.match(
                    res_a["keypoints"], res_a["descriptors"],
                    res_b["keypoints"], res_b["descriptors"],
                    ratio_test=ratio_test, 
                    ratio_threshold=ratio_threshold
                )
                
                # 3. Metrics Generation
                stats = MetricsService.calculate_similarity_stats(
                    match_results["raw_matches"], 
                    match_results["inlier_matches"],
                    res_a["count"], res_b["count"],
                    method=algo_choice
                )
                
                total_time = res_a["extraction_time"] + res_b["extraction_time"] + match_results["total_match_time"]

                # --- UI DISPLAY ---
                
                # Metrics Row
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Inlier Matches", stats["num_inliers"], f"{stats['inlier_ratio']:.1f}% ratio")
                m2.metric("Match Ratio", f"{stats['match_ratio']:.2f}%")
                m3.metric("Confidence", f"{stats['confidence_score']:.1f}/100")
                m4.metric("Total Time", f"{total_time:.3f}s")
                
                # Verdict
                if stats["confidence_score"] >= 80:
                    st.success(f"**Verdict:** {stats['verdict']} 🌟")
                elif stats["confidence_score"] >= 50:
                    st.warning(f"**Verdict:** {stats['verdict']} ⚠️")
                else:
                    st.error(f"**Verdict:** {stats['verdict']} ❌")
                
                # Result Tabs
                tab1, tab2, tab3, tab4 = st.tabs(["Visualization", "Keypoints", "Analytics", "Performance"])
                
                with tab1:
                    st.write(f"### Match Visualization ({vis_mode})")
                    
                    # Determine which matches to draw
                    if vis_mode == "Inliers Only":
                        matches_to_draw = match_results["inlier_matches"]
                    elif vis_mode == "Top-K Matches":
                        matches_to_draw = match_results["raw_matches"][:top_k]
                    else:
                        matches_to_draw = match_results["raw_matches"]

                    img_matches = VisualizationUtils.draw_matches(
                        img_a, res_a["keypoints"], 
                        img_b, res_b["keypoints"], 
                        matches_to_draw,
                        n_matches=None # already sliced
                    )
                    st.image(ImageUtils.to_rgb(img_matches), use_container_width=True)
                    
                    # Download
                    ret, buf = cv2.imencode(".png", img_matches)
                    st.download_button(
                        label="Download Visualization",
                        data=buf.tobytes(),
                        file_name=f"visionmatch_{algo_choice}_{vis_mode}.png",
                        mime="image/png"
                    )

                with tab2:
                    k1, k2 = st.columns(2)
                    with k1:
                        st.write("#### Image A Keypoints")
                        st.info(f"Detected: {res_a['count']}")
                        img_kp_a = VisualizationUtils.draw_keypoints(img_a, res_a["keypoints"])
                        st.image(ImageUtils.to_rgb(img_kp_a), use_container_width=True)
                    with k2:
                        st.write("#### Image B Keypoints")
                        st.info(f"Detected: {res_b['count']}")
                        img_kp_b = VisualizationUtils.draw_keypoints(img_b, res_b["keypoints"])
                        st.image(ImageUtils.to_rgb(img_kp_b), use_container_width=True)

                with tab3:
                    st.write("### Distance Distribution")
                    hist = VisualizationUtils.plot_distance_histogram(match_results["raw_matches"], stats, algo_choice)
                    st.plotly_chart(hist, use_container_width=True)
                    
                    st.write("#### Statistical Summary")
                    col_s1, col_s2 = st.columns(2)
                    with col_s1:
                        st.write("**Descriptor Stats**")
                        st.write(f"- Type: `{res_a['descriptor_type']}`")
                        st.write(f"- Size: `{res_a['descriptor_size']}`")
                        st.write(f"- Mean Distance: `{stats['avg_distance']:.2f}`")
                        st.write(f"- Median Distance: `{stats['median_distance']:.2f}`")
                    with col_s2:
                        st.write("**Matching Stats**")
                        st.write(f"- Raw Matches: `{stats['num_matches']}`")
                        st.write(f"- Verified Inliers: `{stats['num_inliers']}`")
                        st.write(f"- Std Deviation: `{stats['std_distance']:.2f}`")
                        st.write(f"- Min/Max: `{stats['min_distance']:.1f}` / `{stats['max_distance']:.1f}`")

                with tab4:
                    st.write("### Execution Breakdown")
                    perf_data = {
                        "Phase": ["Extraction (Img A)", "Extraction (Img B)", "Initial Matching", "RANSAC Verification", "Total"],
                        "Time (Seconds)": [
                            f"{res_a['extraction_time']:.4f}",
                            f"{res_b['extraction_time']:.4f}",
                            f"{match_results['matching_time']:.4f}",
                            f"{match_results['ransac_time']:.4f}",
                            f"{total_time:.4f}"
                        ]
                    }
                    st.table(perf_data)
                    st.info("Performance measured on the current server environment.")

    else:
        st.info("👋 Welcome to VisionMatch Pro! Please upload two images to begin your comparison.")

if __name__ == "__main__":
    main()
