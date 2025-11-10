from ultralytics import YOLO
import cv2
import numpy as np
import time

# Distance estimation configuration (calibrate these values)
KNOWN_WIDTH = 10.0      # Real width of your object in cm
FOCAL_LENGTH = 908.47     # Set after calibration (e.g., 800.0)

def calculate_distance(pixel_width, known_width, focal_length):
    """Calculate distance: Distance = (Real_Width × Focal_Length) / Pixel_Width"""
    if focal_length is None or focal_length <= 0:
        return None
    return (known_width * focal_length) / pixel_width

# used to record the time when we processed last frame
prev_frame_time = 0
# used to record the time at which we processed current frame
new_frame_time = 0

# Load AI Model
m_Model_01 = YOLO('my_model/result/train2/weights/best.pt')

# Initialize webcam (0 is default camera; change to 1 or 2 if needed)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set webcam resolution (optimized for performance)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

while(1):
    # Read frame from webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break
    
    # Calculating the fps
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = "%02d" % (fps,) + ' fps'
    # putting the FPS count on the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, fps, (5, 30), font, 1, (100, 255, 0), 2, cv2.LINE_AA)

    # Run inference with model
    m_Result_01 = m_Model_01(frame, conf=0.9)  # m_Model_01, confidence threshold 0.8

    # Create a copy of the frame for drawing
    annotated_frame = frame.copy()

    # ===================================================
    # Process Model 01 detections (green boxes)
    # ===================================================
    boxes_01 = m_Result_01[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2] in pixels
    confidences_01 = m_Result_01[0].boxes.conf.cpu().numpy()  # Confidence scores [0-1]
    class_ids_01 = m_Result_01[0].boxes.cls.cpu().numpy()  # Class IDs
    names_01 = m_Model_01.names  # Class name mapping

    for box, conf, cls_id in zip(boxes_01, confidences_01, class_ids_01):
        x1, y1, x2, y2 = map(int, box)
        class_name = names_01[int(cls_id)]
        
        # Calculate distance
        pixel_width = x2 - x1
        distance = calculate_distance(pixel_width, KNOWN_WIDTH, FOCAL_LENGTH)
        
        # Create label (with distance if available)
        if distance:
            label = f"[01]: {class_name}: {conf:.2f} | {distance:.1f}cm"
        else:
            label = f"[01]: {class_name}: {conf:.2f}"

        # Draw green rectangle (Model 01)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Draw label background
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
        # Draw text
        cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # ===================================================
    # Display the frame with detections from models
    # ===================================================
    cv2.imshow("Object Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
