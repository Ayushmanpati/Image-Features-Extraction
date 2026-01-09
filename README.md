# 🔍 VisionMatch Pro: Mathematically Robust Image Comparison

VisionMatch Pro is an interview-grade computer vision application for advanced image feature comparison. It moves beyond simple point matching by incorporating geometric verification and rigorous statistical analysis.

## 🚀 Pro Features

- **Geometric Verification (RANSAC)**: Effectively filters outlier matches by finding the best homography matrix between image pairs.
- **Dual Engine Architecture**: 
  - **ORB**: Optimized with Hamming distance-aware thresholds.
  - **SIFT**: Precise matching with Lowe's Ratio Test implementation.
- **Mathematically Sound Metrics**:
  - Clamped Match Ratio: `good_matches / min(kp1, kp2)`.
  - Multi-factor Confidence Score: Combines inlier ratio, match ratio, and normalized distance.
- **Interactive Visualization**:
  - Toggle between **Inliers Only**, **Top-K**, and **Raw Results**.
  - Advanced distance distribution plots with Mean/Median/Std Dev analysis.
- **Performance Profiling**: Granular breakdown of Extraction, Matching, and RANSAC verification times.

## 🛠️ Architecture

The project is structured following clean engineering principles:

```text
/app
  ├── app.py              # Main Streamlit UI & Entry Point
  ├── services/           # Business Logic
  │     ├── orb_service.py
  │     ├── sift_service.py
  │     ├── matcher.py
  │     └── metrics.py
  ├── utils/              # Common Utilities
  │     ├── image_utils.py
  │     └── visualization.py
  └── static/             # Assets and images
```

## 💻 Getting Started

### Prerequisites
- Python 3.8+
- OpenCV-contrib (`opencv-contrib-python`)

### Installation

1. Clone the repository:
```bash
git clone <repo-url>
cd Image-Features-Extraction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run app/app.py
```

## 📖 Use Cases

- **Duplicate Detection**: Identify similar images in a dataset.
- **Object Recognition**: Match query objects against a target scene.
- **CV Education**: Understand how different feature detectors behave under various conditions.

## 📜 License
MIT License.
