import cv2
import time
import numpy as np

class SIFTService:
    def __init__(self):
        self.detector = cv2.SIFT_create()

    def detect_and_compute(self, img: np.ndarray):
        """Detect keypoints and compute descriptors using SIFT."""
        start_time = time.time()
        # Ensure image is grayscale for detection
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        kp, des = self.detector.detectAndCompute(gray, None)
        end_time = time.time()
        
        return {
            "keypoints": kp,
            "descriptors": des,
            "extraction_time": end_time - start_time,
            "count": len(kp),
            "descriptor_type": str(des.dtype) if des is not None else None,
            "descriptor_size": des.shape[1] if des is not None else None
        }
