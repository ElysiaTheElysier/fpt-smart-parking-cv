
import sys
sys.path.append('c:/Ki_5/fpt-smart-parking-cv')
from backend.core.utils import load_config
from backend.core.video_processor import VideoProcessor
import cv2

cfg = load_config('c:/Ki_5/fpt-smart-parking-cv/backend/config.yaml')
cfg['model_path'] = 'c:/Ki_5/fpt-smart-parking-cv/models/yolov8m.onnx' # Use the onnx model
cfg['show_cv2'] = False
cfg['max_frames'] = 1170
cfg['video_path'] = 'c:/Ki_5/fpt-smart-parking-api/data/raw_videos/input_video.mp4'

class MyVP(VideoProcessor):
    def process_frame(self, frame):
        out = super().process_frame(frame)
        if self.frame_count % 30 == 0:
            print(f'Frame {self.frame_count}: Gaps={len(self._cached_gaps)}')
        return out

vp = MyVP(cfg)
vp.process()

