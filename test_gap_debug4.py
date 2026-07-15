
import sys
sys.path.append('c:/Ki_5/fpt-smart-parking-api')
from backend.core.utils import load_config
from backend.core.video_processor import VideoProcessor
import cv2
import numpy as np

cfg = load_config('c:/Ki_5/fpt-smart-parking-api/backend/config.yaml')
cfg['max_frames'] = 1170
cfg['show_cv2'] = False

class MyVP(VideoProcessor):
    def process(self):
        print('Starting process...')
        frame_idx = 0
        while self.cap.more():
            ret, frame = self.cap.read()
            if not ret: break
            if frame_idx < 1160:
                frame_idx += 1
                continue
                
            self.frame_count = frame_idx
            
            # Just run the raw inference
            raw_detections = self._detect_and_track(frame)
            detections = self._filter_by_roi(raw_detections)
            self._last_detections = detections
            
            print(f'--- Frame {frame_idx} ---')
            from backend.core.gap_measurement import transform_points, bbox_bottom_center
            
            mc_bboxes = [d['bbox'] for d in self._last_detections if d['class_name'] == 'motorcycle']
            pts_array = np.array([bbox_bottom_center(b) for b in mc_bboxes])
            
            if self.roi_polygon is not None:
                from backend.core.gap_measurement import point_in_polygon
                keep_mask = [point_in_polygon(tuple(pt), self.roi_polygon) for pt in pts_array]
                pts_array = pts_array[keep_mask]
                mc_bboxes = [b for b, k in zip(mc_bboxes, keep_mask) if k]
                
            order = np.argsort(pts_array[:, 0])
            pts_array = pts_array[order]
            pts_bev = transform_points(pts_array, self.bev_matrix)
            
            diffs = np.diff(pts_bev, axis=0)
            distances_bev = np.linalg.norm(diffs, axis=1)
            
            for i, dist_bev in enumerate(distances_bev):
                dist_meters = dist_bev / self.gap_analyzer.bev_pixels_per_meter
                print(f'Gap {i}: dist_m={dist_meters:.2f}')
                if dist_meters < self.gap_analyzer.gap_threshold_meters:
                    print(f'  -> Failed min threshold ({self.gap_analyzer.gap_threshold_meters})')
                elif dist_meters > 3.5:
                    print(f'  -> Failed max threshold (3.5)')
                elif abs(pts_bev[i][1] - pts_bev[i + 1][1]) > 1.5 * self.gap_analyzer.bev_pixels_per_meter:
                    print(f'  -> Failed vertical blocker')
                else:
                    mid_orig = (pts_array[i] + pts_array[i + 1]) / 2.0
                    if self.gap_analyzer._midpoint_in_exclusion_zone(tuple(mid_orig.tolist())):
                        print('  -> Failed exclusion zone')
                    elif self.gap_analyzer._gap_overlaps_any_bbox(pts_array[i], pts_array[i + 1], mc_bboxes, (i, i + 1)):
                        print('  -> Failed bbox overlap check')
                    else:
                        print('  -> PASSED!')
                        
            frame_idx += 1
            if frame_idx > 1162:
                break
                
vp = MyVP(cfg)
vp.process()

