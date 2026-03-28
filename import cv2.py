import cv2
import os
from ultralytics import YOLO  # Make sure to: pip install ultralytics

# Load the YOLOv8 model (it will download automatically on first run)
model = YOLO('yolov8n.pt') 

# Your likely camera details
ip = "10.10.1.210"
user = "admin"
pw = "kingston123"

# Common paths to try with the Kingston password
paths = [
    "live/ch0", 
    "Streaming/Channels/101", 
    "cam/realmonitor?channel=1&subtype=0",
    "video1",
    "onvif1"
]

found = False
for path in paths:
    url = f"rtsp://{user}:{pw}@{ip}:554/{path}"
    print(f"🚀 Testing KEC Lab Path: {url}")
    
    cap = cv2.VideoCapture(url)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✅ SUCCESS! Connected to {ip}")
            found = True
            while True:
                ret, frame = cap.read()
                if not ret: break

                # --- CROWD DETECTION LOGIC ---
                results = model(frame, verbose=False) # Run YOLO
                
                # Count only 'person' class (class 0 in COCO dataset)
                person_count = 0
                for r in results:
                    for box in r.boxes:
                        if int(box.cls) == 0:
                            person_count += 1
                            # Draw box
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Display count on screen
                cv2.putText(frame, f"Crowd Count: {person_count}", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                cv2.imshow('KEC Crowd Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            cap.release()
            break
    cap.release()

if not found:
    print("\n❌ Password 'kingston123' didn't match these paths. Try checking 'http://10.10.1.210' in a browser to see the brand logo.")

cv2.destroyAllWindows()