"""
FastAPI Server and WebSocket backend for Real-Time Vehicle Detection and Unique Track Counting.
Serves static frontend files and manages dual input video/webcam streaming with configurable imgsz.
"""

import os
import cv2
import time
import asyncio
import logging
import shutil
from pathlib import Path
from typing import Optional, Dict

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from vision.video_processor import VideoStreamProcessor
from vision.config import (
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IMGSZ,
    DEFAULT_MAX_DET,
    DEFAULT_CONFIRMATION_FRAMES
)

# Configure Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("vision.backend")

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
TEMP_DIR = BASE_DIR / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Initialize FastAPI App
app = FastAPI(
    title="Real-Time Dense Traffic Vehicle Detection & Track Confirmation API",
    description="CCTV Dense Traffic Vehicle Detection, ByteTrack Tracking, and Persistent Unique ID Counting Web Backend",
    version="2.0.0"
)

# Enable CORS for local web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Stream State
class StreamState:
    def __init__(self):
        self.processor: Optional[VideoStreamProcessor] = None
        self.source_type: str = "none"  # "none", "video", "webcam"
        self.video_path: Optional[str] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_streaming: bool = False
        self.fps: float = 25.0

    def init_processor(self):
        if self.processor is None:
            logger.info("Initializing VideoStreamProcessor and YOLO model...")
            self.processor = VideoStreamProcessor(
                conf_thresh=DEFAULT_CONF_THRESHOLD,
                imgsz=DEFAULT_IMGSZ,
                max_det=DEFAULT_MAX_DET,
                confirmation_frames=DEFAULT_CONFIRMATION_FRAMES
            )
            logger.info("VideoStreamProcessor initialized successfully.")

    def close_capture(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                logger.error(f"Error releasing VideoCapture: {e}")
            self.cap = None

state = StreamState()


@app.on_event("startup")
async def startup_event():
    """Pre-load YOLO model once on server startup."""
    logger.info("Starting Vision Backend Server...")
    state.init_processor()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on server shutdown."""
    state.close_capture()
    logger.info("Vision Backend Server stopped.")


# REST API Endpoints

@app.get("/api/status")
async def get_status():
    """Get current backend streaming status and configuration."""
    state.init_processor()
    unique_counts = state.processor.counter.unique_summary if hasattr(state.processor.counter, 'unique_summary') else {}
    return {
        "status": "online",
        "source_type": state.source_type,
        "is_streaming": state.is_streaming,
        "conf_threshold": state.processor.detector.conf_threshold if state.processor else DEFAULT_CONF_THRESHOLD,
        "imgsz": state.processor.detector.imgsz if state.processor else DEFAULT_IMGSZ,
        "confirmation_frames": state.processor.counter.confirmation_frames if state.processor else DEFAULT_CONFIRMATION_FRAMES,
        "unique_counters": unique_counts
    }


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload MP4 traffic video file for frame-by-frame processing."""
    state.init_processor()
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(status_code=400, detail="Unsupported video format. Please upload an MP4, AVI, MOV, or MKV file.")

    state.close_capture()
    file_path = TEMP_DIR / f"uploaded_{int(time.time())}.mp4"

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        state.video_path = str(file_path)
        state.source_type = "video"
        state.processor.reset()

        cap = cv2.VideoCapture(state.video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Failed to open uploaded video file.")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        state.fps = fps if fps > 0 else 25.0
        cap.release()

        logger.info(f"Uploaded video saved to {file_path}, FPS: {state.fps}")
        return {
            "status": "success",
            "message": f"Video '{file.filename}' uploaded successfully.",
            "source": "video",
            "fps": round(state.fps, 1)
        }
    except Exception as e:
        logger.error(f"Error handling video upload: {e}")
        raise HTTPException(status_code=500, detail=f"Video upload failed: {str(e)}")


@app.post("/api/camera/start")
async def start_camera():
    """Initialize and start laptop webcam feed."""
    state.init_processor()
    state.close_capture()

    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            state.source_type = "none"
            raise HTTPException(status_code=500, detail="Webcam (Device 0) could not be opened or accessed.")

        ret, test_frame = cap.read()
        if not ret or test_frame is None:
            cap.release()
            state.source_type = "none"
            raise HTTPException(status_code=500, detail="Webcam device opened but failed to capture frame.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        state.fps = fps if fps > 0 else 30.0
        cap.release()
        state.source_type = "webcam"
        state.processor.reset()
        logger.info("Webcam started successfully.")
        return {"status": "success", "message": "Laptop webcam connected.", "source": "webcam", "fps": state.fps}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start camera: {e}")
        raise HTTPException(status_code=500, detail=f"Webcam access error: {str(e)}")


@app.post("/api/camera/stop")
async def stop_camera():
    """Stop active video/webcam stream."""
    state.close_capture()
    state.is_streaming = False
    state.source_type = "none"
    logger.info("Stream stopped.")
    return {"status": "success", "message": "Stream stopped successfully."}


@app.post("/api/config")
async def update_config(
    conf_threshold: Optional[float] = Form(None),
    imgsz: Optional[int] = Form(None),
    confirmation_frames: Optional[int] = Form(None)
):
    """Update detection confidence, inference resolution (imgsz), or confirmation frames dynamically."""
    state.init_processor()
    state.processor.update_config(
        conf_thresh=conf_threshold,
        imgsz=imgsz,
        confirmation_frames=confirmation_frames
    )

    return {
        "status": "success",
        "conf_threshold": state.processor.detector.conf_threshold,
        "imgsz": state.processor.detector.imgsz,
        "confirmation_frames": state.processor.counter.confirmation_frames
    }


@app.post("/api/reset")
async def reset_counters():
    """Reset all active and unique vehicle counters."""
    state.init_processor()
    state.processor.reset()
    return {"status": "success", "message": "Counters reset to zero."}


# Real-Time WebSocket Streaming Endpoint

@app.websocket("/ws/traffic")
async def websocket_traffic_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time video frame and telemetry streaming.
    Pushes processed JPEG images (base64) and active/unique telemetry JSON.
    """
    await websocket.accept()
    logger.info("WebSocket client connected.")
    state.init_processor()
    state.is_streaming = True

    try:
        if state.source_type == "video" and state.video_path:
            state.cap = cv2.VideoCapture(state.video_path)
        elif state.source_type == "webcam":
            state.cap = cv2.VideoCapture(0)
        else:
            state.cap = None

        while state.is_streaming:
            if state.cap is None or not state.cap.isOpened():
                if state.source_type == "video" and state.video_path:
                    state.cap = cv2.VideoCapture(state.video_path)
                elif state.source_type == "webcam":
                    state.cap = cv2.VideoCapture(0)
                else:
                    await asyncio.sleep(0.2)
                    continue

            ret, frame = state.cap.read()

            # Handle EOF for video files (loop back to start frame)
            if not ret or frame is None:
                if state.source_type == "video" and state.cap is not None:
                    logger.info("End of video reached. Looping video.")
                    state.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    await asyncio.sleep(0.05)
                    continue
                else:
                    logger.warning("Empty frame from stream.")
                    await asyncio.sleep(0.1)
                    continue

            current_time = time.time()

            # Process frame through detection, tracking, confirmation pipeline
            annotated_frame, telemetry = state.processor.process_frame(
                frame=frame,
                current_time=current_time,
                video_fps=state.fps
            )

            # Compress annotated frame to JPEG base64
            img_b64 = state.processor.encode_frame_to_base64(annotated_frame, quality=75)

            payload = {
                "frame": f"data:image/jpeg;base64,{img_b64}",
                "telemetry": telemetry,
                "timestamp": current_time
            }

            await websocket.send_json(payload)

            # Continuous execution yielding event loop control
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        logger.error(f"Error in WebSocket streaming loop: {e}")
    finally:
        state.is_streaming = False
        state.close_capture()


# Mount Static Frontend Files
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def serve_index():
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return HTMLResponse("<h2>Frontend index.html not found</h2>", status_code=404)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("vision.backend.main:app", host="127.0.0.1", port=8000, reload=True)
