import cv2
import numpy as np
import mediapipe as mp

class BackgroundSegmenter:
    def __init__(self):
        print("Initializing MediaPipe AI...")
        # Load the Selfie Segmentation model
        self.mp_selfie = mp.solutions.selfie_segmentation
        # model_selection=0 is for general use. (1 is optimized for landscape/speed).
        self.segmentor = self.mp_selfie.SelfieSegmentation(model_selection=0)

    def get_mask(self, frame):
        # 1. Convert color space (OpenCV uses BGR, MediaPipe needs RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 2. Run the AI inference
        results = self.segmentor.process(rgb_frame)
        
        # 3. Process the results into a binary mask
        # The AI returns a matrix of probabilities (0.0 to 1.0). 
        # We say anything over 50% confidence (0.5) is the person.
        condition = results.segmentation_mask > 0.5
        
        # Convert that true/false condition into a black/white image (0 or 255)
        mask = np.where(condition, 255, 0).astype(np.uint8)
        
        return mask