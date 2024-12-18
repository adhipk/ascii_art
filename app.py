from fastapi import FastAPI, Request, Form
from fastapi.responses import StreamingResponse, HTMLResponse, Response
from fastapi.templating import Jinja2Templates
import cv2
from ascii import generate_ascii_image_sprite
from threading import Thread, Event
import time
from contextlib import asynccontextmanager
import logging
import uvicorn
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebcamVideoStream:
    def __init__(self, src=0, width=800, height=450, fps=30):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_FPS, fps)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.frame = None
        self.stopped = Event()  # Using Event for thread-safe stopping
        
        # Verify camera opened successfully
        if not self.stream.isOpened():
            raise RuntimeError("Failed to open webcam stream")
        
    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()  # Make thread daemon
        return self
    
    def update(self):
        try:
            while not self.stopped.is_set():
                success, frame = self.stream.read()
                if not success:
                    logger.warning("Failed to read frame from webcam")
                    continue
                self.frame = frame
        except Exception as e:
            logger.error(f"Error in webcam thread: {str(e)}")
            self.stopped.set()
            
    def read(self):
        return self.frame
    
    def stop(self):
        logger.info("Stopping webcam stream...")
        self.stopped.set()
        # Give the thread a moment to stop
        time.sleep(0.1)
        if self.stream.isOpened():
            self.stream.release()
        logger.info("Webcam stream stopped")

# Global variables
vs = None
current_filter = "none"
filter_settings = {
    "ascii": {
        "use_edge": True,
        "edge_threshold": 50,
        "use_color": False,
        "font_size": 8,
        "sharpness": 5.0,
        "white_point": 200
    },
    "dog": {
        "sharpness": 5.0,
        "white_point": 200
    }
}
current_resolution = {"width": 800, "height": 450}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vs
    try:
        logger.info("Starting video stream...")
        vs = None  # Ensure vs is initialized as None
        yield
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise
    finally:
        logger.info("Cleaning up resources...")
        if vs:
            vs.stop()
        logger.info("Cleanup complete")

app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

def apply_filter(frame, filter_name, filter_settings):
    if frame is None:
        return None
    try:
        return generate_ascii_image_sprite(frame, filter_settings['ascii'])
    except Exception as e:
        logger.error(f"Error applying filter {filter_name}: {str(e)}")
        return frame

async def generate_frames():
    try:
        while True:
            if vs is None or vs.stopped.is_set():
                logger.warning("Video stream is not available")
                break
                
            frame = vs.read()
            if frame is None:
                await asyncio.sleep(0.01)
                continue
                
            filtered_img = apply_filter(frame, current_filter, filter_settings)
            if filtered_img is None:
                continue
                
            ret, buffer = cv2.imencode('.jpg', filtered_img)
            if not ret:
                logger.warning("Failed to encode frame")
                continue
                
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            await asyncio.sleep(0.01)  # Use asyncio.sleep for better async behavior
    except Exception as e:
        logger.error(f"Error in frame generation: {str(e)}")
    finally:
        logger.info("Frame generation stopped")

@app.get("/")
async def home():
    return Response("Hello world")

@app.get("/video_feed")
async def video_feed():
    global vs, current_resolution
    if vs is None or vs.stopped.is_set():
        try:
            # Use default resolution if not set
            width = current_resolution.get('width', 800)
            height = current_resolution.get('height', 450)
            
            vs = WebcamVideoStream(
                src=0, 
                width=width, 
                height=height
            ).start()
        except Exception as e:
            logger.error(f"Failed to start video stream: {e}")
            return Response("Could not start video stream", status_code=503)
    
        
    response = StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="dashboard.html")

@app.post("/start_stream")
async def start_stream(
    width: int = Form(800), 
    height: int = Form(450),
    fps: int = Form(30)
):
    global vs, current_resolution
    try:
        # Stop existing stream if it exists
        if vs:
            vs.stop()
        
        # Update current resolution
        current_resolution = {"width": width, "height": height}
        
        # Start new stream with specified parameters
        vs = WebcamVideoStream(src=0, width=width, height=height, fps=fps).start()
        return {"message": "Stream started", "resolution": current_resolution}
    except Exception as e:
        logger.error(f"Error starting stream: {str(e)}")
        return Response("Failed to start stream", status_code=500)

@app.post("/set_filter")
async def set_filter(
    ascii_edge_data: bool = Form(False),
    ascii_edge_threshold: int = Form(50),
    ascii_font_size: int = Form(8),
    ascii_use_color: bool = Form(False),
    ascii_sharpness: float = Form(5.0),
    ascii_white_point: int = Form(200),
    dog_sharpness: float = Form(5.0),
    dog_white_point: int = Form(200)
):
    global current_filter, filter_settings
    try:
        current_filter = "ascii"
        filter_settings = {
            "ascii": {
                "use_edge": ascii_edge_data,
                "edge_threshold": ascii_edge_threshold,
                "use_color": ascii_use_color,
                "font_size": ascii_font_size,
                "sharpness": ascii_sharpness,
                "white_point": ascii_white_point
            },
            "dog": {
                "sharpness": dog_sharpness,
                "white_point": dog_white_point
            }
        }
        return {"message": "Filter and settings updated"}
    except Exception as e:
        logger.error(f"Error updating filter settings: {str(e)}")
        return Response("Failed to update filter settings", status_code=500)

@app.post('/close_stream')
async def close_stream():
    try:
        if vs:
            vs.stop()
        return Response("Stream Closed")
    except Exception as e:
        logger.error(f"error closing stream {str(e)}")
        return Response("error closing stream")
    
if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
    finally:
        # Ensure cleanup happens even if the server crashes
        if vs:
            vs.stop()