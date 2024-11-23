from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import cv2

from utils import difference_of_gaussian, generate_ascii_image_sprite
app = FastAPI()

# Capture video feed
video_source = 0  # 0 for webcam, or provide a video file path
cap = cv2.VideoCapture(video_source)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        filtered_img = generate_ascii_image_sprite(frame)
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', filtered_img)
        if not ret:
            continue
        
        # Yield as MJPEG stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.get("/")
async def video_feed():
    # Use StreamingResponse for streaming MJPEG frames
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
