import cv2
import os
import numpy as np
from backend.core.utils import load_config, bbox_bottom_center
from backend.core.bev import load_points, compute_perspective_transform
from backend.core.detector import YOLODetector

def export_minimap():
    config = load_config("backend/config.yaml")
    video_path = config.get("video_path", "data/raw_videos/input_video.mp4")
    
    cap = cv2.VideoCapture(video_path)
    
    # Fast forward to a good frame, e.g., frame 1072
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1072)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Failed to read video")
        return
        
    detector = YOLODetector(config)
    
    detections = detector.detect(frame)
    mc_detections = [d for d in detections if d["class_name"] == "motorcycle"]
    
    bev_w = int(config.get("bev_output_width", 1000))
    bev_h = int(config.get("bev_output_height", 700))
    
    bev_pts = load_points("data/calibration/bev_points.json")
    bev_matrix = compute_perspective_transform(bev_pts, bev_w, bev_h)
    
    # Create the warped BEV frame
    bev_view = cv2.warpPerspective(frame, bev_matrix, (bev_w, bev_h))
    
    # Calculate bottom-center points
    mc_pts_orig = [bbox_bottom_center(d["bbox"]) for d in mc_detections]
    pts_array = np.array(mc_pts_orig, dtype=np.float32).reshape(-1, 1, 2)
    
    # Transform points to BEV
    pts_bev = cv2.perspectiveTransform(pts_array, bev_matrix).reshape(-1, 2)
    
    # Draw keypoints on the BEV map
    COLOR_MOTORCYCLE = (0, 200, 0)
    for x, y in pts_bev:
        cv2.circle(bev_view, (int(x), int(y)), 15, COLOR_MOTORCYCLE, -1)
        # Add a white border to make it pop
        cv2.circle(bev_view, (int(x), int(y)), 15, (255, 255, 255), 2)
        
    out_path = "data/outputs/minimap_highres.jpg"
    cv2.imwrite(out_path, bev_view)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    export_minimap()
