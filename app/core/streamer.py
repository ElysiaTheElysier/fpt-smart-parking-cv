import threading
import queue
import cv2
import time
import os
from typing import Dict, Any, Optional


from backend.core.video_processor import VideoProcessor
from backend.core.utils import load_config

class VideoStreamer:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.frame_queue = queue.Queue(maxsize=10)
        self.stats = {
            "motorcycles": 0,
            "persons": 0,
            "available_gaps": 0,
            "fps": 0.0,
            "is_running": False
        }
        self.thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self):
        if self.thread is not None and self.thread.is_alive():
            return
        
        self._stop_event.clear()
        self.stats["is_running"] = True
        
        config = load_config(self.config_path)
        config["show_cv2"] = False # Must be false for API
        
        self.thread = threading.Thread(target=self._run_processor, args=(config,), daemon=True)
        self.thread.start()

    def stop(self):
        self._stop_event.set()
        self.stats["is_running"] = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None
            
        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

    def update_stats(self, stats: dict):
        for k, v in stats.items():
            self.stats[k] = v

    def _run_processor(self, config: Dict[str, Any]):
        try:
            vp = VideoProcessor(config, frame_queue=self.frame_queue, stats_callback=self.update_stats)
            vp.process()
        except Exception as e:
            print(f"[ERROR] Streamer thread failed: {e}")
        finally:
            self.stats["is_running"] = False
            self.frame_queue.put(None)

    def generate_frames(self):
        """Generator for MJPEG stream"""
        while self.stats["is_running"]:
            try:
                frame = self.frame_queue.get(timeout=1.0)
                if frame is None:
                    break
                
                # Encode frame to JPEG
                ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if not ret:
                    continue
                    
                frame_bytes = buffer.tobytes()
                
                # Yield in MJPEG format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                       
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ERROR] Generator error: {e}")
                break

streamer = VideoStreamer(config_path="backend/config.yaml")
