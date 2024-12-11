#Ascii filter

A simple ASCII filter for image and video streams.

##Usage
To run the webcam server install requirements and run the server.
```bash
    # Install dependencies
    pip install -r requirements.txt

    # Run the server
    uvicorn app:app --reload --port 8000
    
```
Video Stream available at http://127.0.0.1:8000/video_stream

To Make ASCII art from an image run the ascii.py script.
```bash
    # Run the script
    python ascii.py --image_path <path_to_image>
```
