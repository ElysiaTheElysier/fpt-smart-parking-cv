
import sys
sys.path.append('c:/Ki_5/fpt-smart-parking-api')
from backend.core.utils import load_config
from backend.core.video_processor import VideoProcessor
import cv2
cfg = load_config('c:/Ki_5/fpt-smart-parking-api/backend/config.yaml')
cfg['max_frames'] = 1110
cfg['show_cv2'] = False

class MyVP(VideoProcessor):
    def process(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 1050)
        self.frame_count = 1050
        super().process()
        
    def process_frame(self, frame):
        out = super().process_frame(frame)
        if self.frame_count > 1090:
            print(f'[DEBUG Frame {self.frame_count}] Gaps = {len(self.gap_manager.get_available_gaps())}')
            # also print raw BEV distances!
            # we need to intercept gap_measurement
        return out

vp = MyVP(cfg)
vp.process()

