"""
async_video.py — Asynchronous Video I/O for 30 FPS Performance.
=============================================================
Uses multithreading to read frames from a file/camera into a Queue,
preventing I/O blocks from slowing down the main Inference thread.
"""
import time
import cv2
import threading
from queue import Queue

class AsyncVideoCapture:
    def __init__(self, src, queue_size=128):
        # Initialize video stream and read the first frame
        self.stream = cv2.VideoCapture(src)
        self.stopped = False
        
        # Queue size limits memory usage while keeping a healthy buffer
        self.Q = Queue(maxsize=queue_size)
        
        # Extract metadata
        self.fps = self.stream.get(cv2.CAP_PROP_FPS)
        self.width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        
    def start(self):
        # Start a thread to read frames from the video stream
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # Keep looping infinitely until the thread is stopped
        while not self.stopped:
            # if the queue is not full, read the next frame
            if not self.Q.full():
                (grabbed, frame) = self.stream.read()
                
                # if we have reached the end of the video, stop the thread
                if not grabbed:
                    self.stop()
                    return
                
                self.Q.put(frame)
            else:
                # Give CPU a break if queue is full
                time.sleep(0.01)

    def read(self):
        # Return next frame in the queue (blocks if empty until a frame is ready)
        import queue
        try:
            return True, self.Q.get(timeout=1.0)
        except queue.Empty:
            return False, None

    def more(self):
        # Return True if there are still frames in the queue or the stream is still running
        return self.Q.qsize() > 0 or not self.stopped

    def stop(self):
        # Indicate that the thread should be stopped
        self.stopped = True
        self.stream.release()

class AsyncVideoWriter:
    def __init__(self, path, fourcc, fps, size, queue_size=128):
        self.writer = cv2.VideoWriter(path, fourcc, fps, size)
        self.Q = Queue(maxsize=queue_size)
        self.stopped = False
        
    def start(self):
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self
        
    def update(self):
        while not self.stopped or not self.Q.empty():
            import queue
            try:
                frame = self.Q.get(timeout=0.1)
                self.writer.write(frame)
            except queue.Empty:
                time.sleep(0.01)
                
    def write(self, frame):
        self.Q.put(frame)
        
    def release(self):
        self.stopped = True
        # Wait for queue to empty
        while not self.Q.empty():
            time.sleep(0.1)
        self.writer.release()
