"""
Simple calibration script for distance estimation
Place object at known distance and this will calculate your FOCAL_LENGTH
"""

from ultralytics import YOLO
import cv2

# ===================================================
# CONFIGURATION - Set these values
# ===================================================
MODEL_PATH = 'my_model/result/train2/weights/best.pt'
KNOWN_WIDTH = 10.0      # Real width of object in cm (CHANGE THIS!)
KNOWN_DISTANCE = 50.0   # Distance from camera in cm (CHANGE THIS!)

print("\n" + "="*60)
print("CALIBRATION TOOL")
print("="*60)
print(f"Object width: {KNOWN_WIDTH} cm")
print(f"Object distance: {KNOWN_DISTANCE} cm")
print("\nPlace object at exactly {:.1f}cm from camera".format(KNOWN_DISTANCE))
print("Press SPACE to calibrate, Q to quit")
print("="*60 + "\n")

# Load model
model = YOLO(MODEL_PATH)

# Open camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run detection
    results = model(frame, conf=0.5, verbose=False)
    
    # Draw detections
    display = frame.copy()
    
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            pixel_width = x2 - x1
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            
            # Draw box and info
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, f"{class_name}: {conf:.2f}", (x1, y1-25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display, f"Width: {pixel_width}px", (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.putText(display, "Object detected! Press SPACE to calibrate", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(display, "No object detected", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow('Calibration', display)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' ') and len(results[0].boxes) > 0:
        # Calibrate using first detection
        box = results[0].boxes[0]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        pixel_width = x2 - x1
        
        # Calculate focal length: F = (P × D) / W
        focal_length = (pixel_width * KNOWN_DISTANCE) / KNOWN_WIDTH
        
        print("\n" + "="*60)
        print("CALIBRATION COMPLETE!")
        print("="*60)
        print(f"Pixel width: {pixel_width:.2f} px")
        print(f"Known distance: {KNOWN_DISTANCE} cm")
        print(f"Known width: {KNOWN_WIDTH} cm")
        print(f"\n✓ FOCAL_LENGTH = {focal_length:.2f}")
        print("\n" + "="*60)
        print("TO USE:")
        print(f"1. Open IC_AI_02_Application_01_Run_01.py")
        print(f"2. Set: KNOWN_WIDTH = {KNOWN_WIDTH}")
        print(f"3. Set: FOCAL_LENGTH = {focal_length:.2f}")
        print("="*60 + "\n")
        break

cap.release()
cv2.destroyAllWindows()

