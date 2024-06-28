# backend/main.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import cv2
import pickle
import cvzone
import numpy as np
import io
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
# Allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use specific domains in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
cap = cv2.VideoCapture("carPark.mp4")

with open('CarParkPos', 'rb') as f:
    posList = pickle.load(f)

width, height = 107, 48

def checkParkingSpace(imgPro, img):
    spaceCounter = 0
    for pos in posList:
        x, y = pos
        imgCrop = imgPro[y:y + height, x:x + width]
        count = cv2.countNonZero(imgCrop)
        if count < 900:
            color = (0, 255, 0)
            thickness = 5
            spaceCounter += 1
        else:
            color = (0, 0, 255)
            thickness = 2
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)
        cvzone.putTextRect(img, str(count), (x, y + height - 3), scale=1, thickness=2, offset=0, colorR=color)
    cvzone.putTextRect(img, f'Free: {spaceCounter}/{len(posList)}', (100, 50), scale=3, thickness=5, offset=20, colorR=(0, 200, 0))
    return img

def generate_frames():
    while True:
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        success, img = cap.read()
        if not success:
            break
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
        imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, 25, 16)
        imgMedian = cv2.medianBlur(imgThreshold, 5)
        kernel = np.ones((3, 3), np.uint8)
        imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

        imgResult = checkParkingSpace(imgDilate, img)

        _, buffer = cv2.imencode('.jpg', imgResult)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.get("/video")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")
@app.get("/counter")
def get_counter():
    # Read a single frame to analyze
    cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES))
    _, img = cap.read()

    # Preprocessing
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

    # Count free spaces
    free = 0
    for pos in posList:
        x, y = pos
        imgCrop = imgDilate[y:y + height, x:x + width]
        count = cv2.countNonZero(imgCrop)
        if count < 900:
            free += 1

    return {"free": free, "total": len(posList)}


import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Use Render's dynamic PORT
    uvicorn.run("backend_main:app", host="0.0.0.0", port=port)
 