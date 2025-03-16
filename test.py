import numpy as np
import cv2
import matplotlib.pyplot as plt

def image_detect_and_compute(detector, img_path):
    """Load image, convert to grayscale, and compute keypoints and descriptors"""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = detector.detectAndCompute(gray, None)
    return gray, kp, des

def draw_image_matches(detector, img1_path, img2_path, nmatches=10):
    """Draw matches between two images using specified detector"""
    # Detect features in both images
    img1, kp1, des1 = image_detect_and_compute(detector, img1_path)
    img2, kp2, des2 = image_detect_and_compute(detector, img2_path)
    
    # Match descriptors
    if isinstance(detector, cv2.ORB):
        # For ORB use Hamming distance
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
    else:
        # For SIFT use L2 distance
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des1, des2)
    
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Draw matches
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:nmatches], None, flags=2)
    
    # Display
    plt.figure(figsize=(16, 8))
    plt.title(f'{type(detector).__name__} - Top {nmatches} Matches')
    plt.imshow(img_matches, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return len(matches)

def show_keypoints(img_path, detector, title=None):
    """Show keypoints detected in the image"""
    # Load image
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect keypoints
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    
    # Draw keypoints
    img_keypoints = cv2.drawKeypoints(
        img_rgb, keypoints, None, 
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    
    # Display
    plt.figure(figsize=(12, 8))
    if title:
        plt.title(title)
    else:
        plt.title(f'{type(detector).__name__} - {len(keypoints)} Keypoints')
    plt.imshow(img_keypoints)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return len(keypoints)

# Main execution
def main(img1_path, img2_path):
    """Run feature detection and matching on the provided images"""
    print(f"Processing images: {img1_path} and {img2_path}")
    
    # Create ORB detector
    orb = cv2.ORB_create(nfeatures=2000)
    
    # Show keypoints for first image
    kp_count = show_keypoints(img1_path, orb, 'ORB Interest Points')
    print(f"ORB detected {kp_count} keypoints in first image")
    
    # Match images with ORB
    match_count = draw_image_matches(orb, img1_path, img2_path, nmatches=50)
    print(f"ORB found {match_count} matches between images")
    
    # Try SIFT if available
    if hasattr(cv2, 'SIFT_create'):
        sift = cv2.SIFT_create()
        
        # Show keypoints for first image
        kp_count = show_keypoints(img1_path, sift, 'SIFT Interest Points')
        print(f"SIFT detected {kp_count} keypoints in first image")
        
        # Match images with SIFT
        match_count = draw_image_matches(sift, img1_path, img2_path, nmatches=50)
        print(f"SIFT found {match_count} matches between images")
    else:
        print("SIFT is not available. Make sure you have OpenCV-contrib installed.")

# Run the code
main('un3.jpeg', 'un4.jpeg')  # Use your actual image filenames here