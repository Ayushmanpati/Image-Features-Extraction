import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

class VisualizationUtils:
    @staticmethod
    def draw_keypoints(img: np.ndarray, kp: list) -> np.ndarray:
        """Draw rich keypoints on the image."""
        img_with_kp = cv2.drawKeypoints(
            img, kp, None, 
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        return img_with_kp

    @staticmethod
    def draw_matches(img1: np.ndarray, kp1: list, img2: np.ndarray, kp2: list, matches: list, n_matches: int = 50) -> np.ndarray:
        """Draw top N matches between two images."""
        display_matches = matches[:n_matches] if n_matches else matches
        img_matches = cv2.drawMatches(
            img1, kp1, img2, kp2, display_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        return img_matches

    @staticmethod
    def plot_distance_histogram(matches: list, stats: dict, algo_name: str):
        """Generate an enriched histogram of match distances using Plotly."""
        distances = [m.distance for m in matches]
        fig = px.histogram(
            x=distances, 
            nbins=30, 
            title=f"Distance Distribution - {algo_name}",
            labels={'x': 'Distance', 'y': 'Count'},
            color_discrete_sequence=['#636EFA']
        )
        
        # Add vertical lines for mean and median
        fig.add_vline(x=stats['avg_distance'], line_dash="dash", line_color="orange", 
                     annotation_text=f"Mean: {stats['avg_distance']:.1f}")
        fig.add_vline(x=stats['median_distance'], line_dash="dot", line_color="red", 
                     annotation_text=f"Median: {stats['median_distance']:.1f}")
        
        fig.update_layout(
            template='plotly_dark',
            bargap=0.1,
            showlegend=False
        )
        return fig
