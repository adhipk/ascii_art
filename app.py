from fastapi import FastAPI, Request, Form
from fastapi.responses import StreamingResponse, HTMLResponse, Response
from fastapi.templating import Jinja2Templates
import cv2
from utils import difference_of_gaussian, generate_ascii_image_sprite
from threading import Thread
import time
from contextlib import asynccontextmanager


class WebcamVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.frame = None
        self.stopped = False
        
    def start(self):
        Thread(target=self.update, args=()).start()
        return self
    
    def update(self):
        while not self.stopped:
            _, self.frame = self.stream.read()
            
    def read(self):
        return self.frame
    
    def stop(self):
        self.stopped = True
        self.stream.release()

# Global variables
# Global variables
vs = None
current_filter = "ascii"
filter_settings = {
    "ascii": {
        "use_edge": False,
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize the video stream
    global vs
    print("Starting video stream...")
    vs = WebcamVideoStream().start()
    yield
    # Shutdown: Clean up resources
    print("Shutting down video stream...")
    if vs:
        vs.stop()

app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

def apply_filter(frame, filter_name,filter_settings):
    if frame is None:
        return None
    if filter_name == "ascii":
        return generate_ascii_image_sprite(frame,8,filter_settings['ascii'])
    elif filter_name == "dog":
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        settings = filter_settings['dog']
        print(settings)
        return difference_of_gaussian(frame,white_point=settings["white_point"], sharpness=settings["sharpness"])
    else:
        return frame

def generate_frames():
    while True:
        if vs is None:
            continue
            
        frame = vs.read()
        if frame is None:
            continue
        print("filter_settings", filter_settings)
        filtered_img = apply_filter(frame, current_filter,filter_settings)
        if filtered_img is None:
            continue
            
        ret, buffer = cv2.imencode('.jpg', filtered_img)
        if not ret:
            continue
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.01)  # Small delay to prevent overwhelming the CPU

@app.get("/")
async def home():
    return Response("Hello world")

@app.get("/video_feed")
async def video_feed():
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

@app.post("/set_filter")
async def set_filter(
    filter: str = Form(...),
    # ASCII filter settings
    ascii_edge_data: bool = Form(False),
    ascii_edge_threshold: int = Form(50),
    ascii_font_size:int = Form(8),
    ascii_use_color: bool = Form(False),
    ascii_sharpness: float = Form(5.0),
    ascii_white_point: int = Form(200),
    # DoG filter settings
    dog_sharpness: float = Form(5.0),
    dog_white_point: int = Form(200)
):
    global current_filter, filter_settings
    current_filter = filter
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
    return {"message": f"Filter and settings updated"}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)