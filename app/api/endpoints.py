from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.core.streamer import streamer

router = APIRouter()

@router.get("/stream")
def video_stream():
    """Returns a MJPEG stream of the processed video"""
    if not streamer.stats["is_running"]:
        streamer.start()
    return StreamingResponse(streamer.generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@router.get("/stats")
def get_stats():
    """Returns real-time processing statistics"""
    return streamer.stats

@router.post("/control/start")
def start_stream():
    streamer.start()
    return {"status": "started"}

@router.post("/control/stop")
def stop_stream():
    streamer.stop()
    return {"status": "stopped"}
