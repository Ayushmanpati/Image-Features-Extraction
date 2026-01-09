import cv2
import numpy as np
from PIL import Image
import io

class ImageUtils:
    @staticmethod
    def load_image(file) -> np.ndarray:
        """Load image from a file-like object or path."""
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return img

    @staticmethod
    def to_grayscale(img: np.ndarray) -> np.ndarray:
        """Convert BGR image to grayscale."""
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def to_rgb(img: np.ndarray) -> np.ndarray:
        """Convert BGR image to RGB."""
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def resize_image(img: np.ndarray, width: int = None, height: int = None) -> np.ndarray:
        """Resize image while maintaining aspect ratio or to fixed dimensions."""
        if width is None and height is None:
            return img
        
        h, w = img.shape[:2]
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))
        
        return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    @staticmethod
    def opencv_to_pil(img: np.ndarray) -> Image.Image:
        """Convert OpenCV BGR image to PIL Image."""
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
