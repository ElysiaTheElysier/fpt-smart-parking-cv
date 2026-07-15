
import sys
sys.path.append('c:/Ki_5/fpt-smart-parking-api')
from backend.core.utils import load_config
from backend.core.video_processor import VideoProcessor
import cv2
cfg = load_config('c:/Ki_5/fpt-smart-parking-api/backend/config.yaml')
cfg['max_frames'] = 1110
cfg['show_cv2'] = False

class MyVP(VideoProcessor):
    def process_frame(self, frame):
        if self.frame_count == 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 1100)
            self.frame_count = 1100
        out = super().process_frame(frame)
        if self.frame_count >= 1104:
            print(f'[DEBUG Frame {self.frame_count}] Gaps = {len(self.gap_manager.get_available_gaps())}')
            # Access gap_manager internals
            history = self.gap_manager._gap_history
            print('History:', history)
        return out

vp = MyVP(cfg)
vp.process()

