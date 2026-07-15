
import sys
sys.path.append('c:/Ki_5/fpt-smart-parking-api')
from backend.core.utils import load_config
from backend.core.video_processor import VideoProcessor
import cv2
cfg = load_config('c:/Ki_5/fpt-smart-parking-api/backend/config.yaml')
cfg['max_frames'] = 1170
cfg['show_cv2'] = False

class MyVP(VideoProcessor):
    def process_frame(self, frame):
        if self.frame_count == 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 1150)
            self.frame_count = 1150
        out = super().process_frame(frame)
        if self.frame_count == 1162:
            print('--- Frame 1162 ---')
            for gap in self._cached_gaps:
                print('Gap:', gap)
        return out

vp = MyVP(cfg)
vp.process()

