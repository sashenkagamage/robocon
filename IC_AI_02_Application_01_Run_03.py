from ultralytics import YOLO
import cv2
import numpy as np
import time

# used to record the time when we processed last frame
prev_frame_time = 0
# used to record the time at which we processed current frame
new_frame_time = 0

# Load AI Model
m_Model_01 = YOLO('IC_AI_Model_01/result/train1/weights/best.pt')
m_Model_02 = YOLO('IC_AI_Model_02/result/train1/weights/best.pt')

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
    m_Result_02 = m_Model_02(frame, conf=0.9)  # m_Model_01, confidence threshold 0.8

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
        label = f"[01]: {class_name}: {conf:.2f}"

        # Draw green rectangle (Model 01)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Draw label background
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
        # Draw text
        cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # ===================================================
    # Process Model 02 detections (blue boxes)
    # ===================================================
    boxes_02 = m_Result_02[0].boxes.xyxy.cpu().numpy()
    confidences_02 = m_Result_02[0].boxes.conf.cpu().numpy()
    class_ids_02 = m_Result_02[0].boxes.cls.cpu().numpy()
    names_02 = m_Model_02.names

    for box, conf, cls_id in zip(boxes_02, confidences_02, class_ids_02):
        x1, y1, x2, y2 = map(int, box)
        class_name = names_02[int(cls_id)]
        label = f"[02]: {class_name}: {conf:.2f}"

        # Draw blue rectangle (Model 02)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # Draw label background
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(annotated_frame, (x1, y2 + label_size[1] + 10), (x1 + label_size[0], y2), (255, 0, 0), -1)
        # Draw text (below box to avoid overlap)
        cv2.putText(annotated_frame, label, (x1, y2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

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