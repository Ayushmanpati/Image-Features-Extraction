import cv2
import numpy as np
import time

class MatcherService:
    def __init__(self, method="ORB"):
        self.method = method.upper()
        if self.method == "ORB":
            # For ORB use Hamming distance
            self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            self.distance_threshold = 60 # Interview-grade threshold for ORB
        else:
            # For SIFT use L2 distance
            self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            self.distance_threshold = None # SIFT uses Ratio Test instead

    def match(self, kp1, des1, kp2, des2, ratio_test=True, ratio_threshold=0.75):
        """Match descriptors and apply geometric verification."""
        if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
            return {
                "raw_matches": [],
                "inlier_matches": [],
                "inlier_ratio": 0.0,
                "matching_time": 0.0,
                "ransac_time": 0.0,
                "homography": None
            }

        start_matching = time.time()
        
        # Step 1: Initial Matching
        raw_matches = []
        if self.method == "SIFT" or ratio_test:
            # SIFT or explicit ratio test
            matches = self.bf.knnMatch(des1, des2, k=2)
            for m, n in matches:
                if m.distance < ratio_threshold * n.distance:
                    raw_matches.append(m)
        else:
            # ORB without ratio test (use distance threshold)
            matches = self.bf.match(des1, des2)
            raw_matches = [m for m in matches if m.distance < self.distance_threshold]

        # Sort by distance
        raw_matches = sorted(raw_matches, key=lambda x: x.distance)
        matching_time = time.time() - start_matching

        # Step 2: Geometric Verification (RANSAC)
        start_ransac = time.time()
        inlier_matches = []
        homography = None
        inlier_ratio = 0.0

        if len(raw_matches) >= 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in raw_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in raw_matches]).reshape(-1, 1, 2)

            homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if mask is not None:
                matches_mask = mask.ravel().tolist()
                inlier_matches = [m for i, m in enumerate(raw_matches) if matches_mask[i] == 1]
                inlier_ratio = len(inlier_matches) / len(raw_matches) if len(raw_matches) > 0 else 0

        ransac_time = time.time() - start_ransac

        return {
            "raw_matches": raw_matches,
            "inlier_matches": inlier_matches,
            "inlier_ratio": inlier_ratio,
            "matching_time": matching_time,
            "ransac_time": ransac_time,
            "total_match_time": matching_time + ransac_time,
            "homography": homography
        }
