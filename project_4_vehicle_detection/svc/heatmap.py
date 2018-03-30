import matplotlib.pyplot as plt
import numpy as np
import cv2
import collections
import os
from feature_extraction import extract_features_from_file_list
from config import feature_extraction_params

def compute_heatmap_from_detections(frame, hot_windows, threshold=5, verbose=False, save=False):
    h, w, c = frame.shape
    
    heatmap = np.zeros(shape=(h, w), dtype=np.uint8)
    
    for bbox in hot_windows:
        x_min, y_min = bbox[0]
        x_max, y_max = bbox[1]
        heatmap[y_min:y_max, x_min:x_max] += 1
    
    _, heatmap_thresh = cv2.threshold(heatmap, threshold, 255, type=cv2.THRESH_BINARY)
    heatmap_thresh = cv2.morphologyEx(heatmap_thresh, op=cv2.MORPH_CLOSE, 
                                      kernel=cv2.getStructuringElement(
                                          cv2.MORPH_ELLIPSE, (13, 13)), iterations=1)
    
    if verbose:
        f, ax = plt.subplots(1, 3)
        ax[0].imshow(frame)
        ax[1].imshow(heatmap, cmap='hot')
        ax[2].imshow(heatmap_thresh, cmap='hot')
        if save:
            plt.savefig(os.path.join('output_image', 'heatmap.jpg'))
        plt.show()
        
    return heatmap, heatmap_thresh
