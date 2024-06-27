import os
import io
import cv2
import json
import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import List, Tuple
from ultralytics import YOLO
from starlette.responses import StreamingResponse

app = FastAPI()

# Load the YOLOv8 model with pretrained weights
YOLO_MODEL_PATH = os.getenv('YOLO_MODEL_PATH', 'yolov8x.pt')
model = YOLO(YOLO_MODEL_PATH)

# Define vehicle widths in meters
vehicle_widths = {
    'hatchback': 1.7,
    'sedan': 1.8,
    'suv': 2.0,
    'truck': 2.5,
    'bus': 2.5
}

# In-memory storage for parking spaces
camera_parking_spaces = []

# Function to detect vehicles in an image
def detect_vehicles(image):
    results = model(image)
    vehicle_results = {'car': [], 'bus': [], 'truck': []}
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = box.cls
            if cls == 2:  # Car
                vehicle_results['car'].append(box)
            elif cls == 5:  # Bus
                vehicle_results['bus'].append(box)
            elif cls == 7:  # Truck
                vehicle_results['truck'].append(box)
    return vehicle_results

# Function to find spaces between vehicles
def find_parking_spaces(vehicle_detections, image_width, vehicle_width_threshold):
    all_detections = []
    for vehicle_list in vehicle_detections.values():
        all_detections.extend(vehicle_list)

    if len(all_detections) == 0:
        return []

    # Extract bounding boxes and sort by x-coordinate
    bboxes = [box.xyxy[0] for box in all_detections]
    bboxes.sort(key=lambda x: x[0])

    parking_spaces = []
    for i in range(len(bboxes) - 1):
        x1_end = int(bboxes[i][2])
        x2_start = int(bboxes[i + 1][0])
        y1_start = int(min(bboxes[i][1], bboxes[i + 1][1]))
        y1_end = int(max(bboxes[i][3], bboxes[i + 1][3]))

        # Check if the space between two vehicles is wide enough
        if x2_start - x1_end > vehicle_width_threshold:
            parking_spaces.append((x1_end, y1_start, x2_start, y1_end))

    return parking_spaces

# Function to check if a vehicle can fit into a parking space
def can_vehicle_fit(space, vehicle_width):
    space_width = space[2] - space[0]
    return space_width >= vehicle_width

# Function to draw bounding boxes around detected vehicles and parking spaces, and assign IDs
def draw_bounding_boxes(image, vehicle_detections, parking_spaces, vehicle_width=None):
    vehicle_id = 1
    space_id = 1

    for vehicle_type, detections in vehicle_detections.items():
        for detection in detections:
            x1, y1, x2, y2 = detection.xyxy[0]
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, f"{vehicle_type.capitalize()} {vehicle_id} - Occupied", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            vehicle_id += 1

    for space in parking_spaces:
        x1, y1, x2, y2 = space
        if vehicle_width and can_vehicle_fit(space, vehicle_width):
            color = (255, 0, 0)
            label = f"Free Space {space_id} - Fits"
        elif vehicle_width and not can_vehicle_fit(space, vehicle_width):
            color = (0, 0, 255)
            label = f"Free Space {space_id} - Doesn't Fit"
        else:
            color = (255, 0, 0)
            label = f"Free Space {space_id}"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        space_id += 1

@app.post("/detect_vehicles/")
async def detect_vehicles_endpoint():
    # Capture a frame from the camera at index 1
    CAMERA_INDEX = int(os.getenv('CAMERA_INDEX', 1))
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Failed to access camera")

    success, image = cap.read()
    cap.release()
    if not success:
        raise HTTPException(status_code=500, detail="Failed to capture image from camera")

    vehicle_detections = detect_vehicles(image)
    image_width = image.shape[1]
    vehicle_width_threshold = 2.0  # Example threshold

    parking_spaces = find_parking_spaces(vehicle_detections, image_width, vehicle_width_threshold)
    draw_bounding_boxes(image, vehicle_detections, parking_spaces)

    # Store the detected parking spaces in the camera_parking_spaces list
    global camera_parking_spaces
    camera_parking_spaces = parking_spaces

    _, img_encoded = cv2.imencode('.png', image)
    return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/png")

@app.get("/closest_parking_space/")
async def get_closest_parking_space():
    if not camera_parking_spaces:
        raise HTTPException(status_code=404, detail="No parking space data found")

    closest_space = camera_parking_spaces[0]  # Simplified for demonstration; usually, you would calculate the closest space
    result = {"status": "success", "closest_parking_space": closest_space}

    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
