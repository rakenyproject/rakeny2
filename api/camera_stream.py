import cv2

def capture_frame(camera_index=1):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise Exception("Failed to access camera")
    
    success, frame = cap.read()
    cap.release()
    
    if not success:
        raise Exception("Failed to capture image from camera")
    
    return frame
