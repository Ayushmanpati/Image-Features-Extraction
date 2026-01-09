import numpy as np

class MetricsService:
    @staticmethod
    def calculate_similarity_stats(matches: list, inlier_matches: list, total_kp1: int, total_kp2: int, method: str):
        """Calculate rigorous similarity metrics based on inliers and raw matches."""
        num_matches = len(matches)
        num_inliers = len(inlier_matches)
        
        if num_matches == 0:
            return {
                "num_matches": 0,
                "num_inliers": 0,
                "match_ratio": 0.0,
                "inlier_ratio": 0.0,
                "avg_distance": 0.0,
                "median_distance": 0.0,
                "std_distance": 0.0,
                "min_distance": 0.0,
                "max_distance": 0.0,
                "confidence_score": 0.0,
                "verdict": "Low similarity"
            }

        distances = [m.distance for m in matches]
        avg_dist = np.mean(distances)
        median_dist = np.median(distances)
        std_dist = np.std(distances)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # 1. Correct Match Ratio (Clamped to 1.0)
        # Using minimum of both keypoint sets as baseline
        baseline_kp = min(total_kp1, total_kp2)
        match_ratio = min(1.0, num_matches / baseline_kp) if baseline_kp > 0 else 0
        
        # 2. Inlier Ratio
        inlier_ratio = num_inliers / num_matches if num_matches > 0 else 0
        
        # 3. Normalized Distance Score (0 to 1)
        # For ORB (Hamming), 60 is threshold. For SIFT (L2), distances are larger.
        if method.upper() == "ORB":
            # Higher score for lower distance. 0 at 60 distance, 1 at 0 distance.
            dist_score = max(0, (60 - avg_dist) / 60)
        else:
            # SIFT distances vary; use a heuristic normalization for display
            # Typically SIFT matches are < 200-300 for good matches
            dist_score = max(0, (300 - avg_dist) / 300)

        # 4. Real Confidence Score (0-100)
        # weights: inlier_ratio (40%), match_ratio (30%), dist_score (30%)
        confidence = (inlier_ratio * 0.4 + min(0.3, match_ratio) * 1.0 + dist_score * 0.3) * 100
        confidence = min(100, max(0, confidence))
        
        # 5. Verdict based on confidence
        if confidence >= 80:
            verdict = "High similarity"
        elif confidence >= 50:
            verdict = "Medium similarity"
        else:
            verdict = "Low similarity"
            
        return {
            "num_matches": num_matches,
            "num_inliers": num_inliers,
            "match_ratio": match_ratio * 100, # as percentage for UI
            "inlier_ratio": inlier_ratio * 100, # as percentage for UI
            "avg_distance": avg_dist,
            "median_distance": median_dist,
            "std_distance": std_dist,
            "min_distance": min_dist,
            "max_distance": max_dist,
            "confidence_score": confidence,
            "verdict": verdict
        }
