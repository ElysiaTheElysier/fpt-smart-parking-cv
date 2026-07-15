import cv2
import os
import yaml
from backend.core.bev import load_points, compute_perspective_transform

def export_images():
    with open("backend/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    video_path = config.get("video_path", "data/raw_videos/input_video.mp4")
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video")
        return
        
    cv2.imwrite("data/outputs/before_bev.jpg", frame)
    print("Saved data/outputs/before_bev.jpg")
    
    bev_w = int(config.get("bev_output_width", 1000))
    bev_h = int(config.get("bev_output_height", 700))
    
    bev_pts = load_points("data/calibration/bev_points.json")
    if bev_pts is not None:
        bev_matrix = compute_perspective_transform(bev_pts, bev_w, bev_h)
        bev_frame = cv2.warpPerspective(frame, bev_matrix, (bev_w, bev_h))
        cv2.imwrite("data/outputs/after_bev.jpg", bev_frame)
        print("Saved data/outputs/after_bev.jpg")
        
        # also draw polygon on before_bev to show the mapping
        import numpy as np
        pts = np.array(bev_pts, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, (0, 255, 255), 3)
        cv2.imwrite("data/outputs/before_bev_with_polygon.jpg", frame)
        print("Saved data/outputs/before_bev_with_polygon.jpg")
    
    cap.release()

if __name__ == "__main__":
    export_images()
