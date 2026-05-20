import cv2
import numpy as np

class DSPEngine:
    def __init__(self):
        # A library of standard 3x3 DSP kernels
        self.kernels = {
            "Identity": np.array([[0, 0, 0], 
                                  [0, 1, 0], 
                                  [0, 0, 0]]),
            
            "Sharpen": np.array([[ 0, -1,  0], 
                                 [-1,  5, -1], 
                                 [ 0, -1,  0]]),
            
            "Edge Detect": np.array([[-1, -1, -1], 
                                     [-1,  8, -1], 
                                     [-1, -1, -1]]),
            
            "Emboss": np.array([[-2, -1, 0], 
                                [-1,  1, 1], 
                                [ 0,  1, 2]])
        }

    def get_blur_kernel(self, size=15):
        # Generates an N x N low-pass box blur matrix
        # We use a larger matrix for blur so it's easily visible
        matrix = np.ones((size, size), np.float32)
        return matrix / (size * size) # Normalize it so the image doesn't turn pure white

    def apply_convolution(self, frame, kernel):
        # cv2.filter2D is highly optimized C code that slides our matrix across every pixel
        return cv2.filter2D(frame, -1, kernel)

    def composite_layers(self, fg_frame, bg_frame, mask_3d):
        # Stitch the two processed streams together using the AI mask
        # If mask > 0 (Subject), use fg_frame pixel. Else, use bg_frame pixel.
        return np.where(mask_3d > 0, fg_frame, bg_frame).astype(np.uint8)